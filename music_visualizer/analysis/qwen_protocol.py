"""Length-prefixed TCP framing for the Qwen2-Audio sidecar.

Request frame::

    [4-byte big-endian header_len][JSON header][raw float32 PCM bytes]

Response frame::

    [4-byte big-endian payload_len][JSON payload]

The header JSON carries metadata (sample_rate, instruction, max_new_tokens);
the raw bytes carry the mono float32 audio window.  Stdlib only so both the
main process and the sidecar can import this without extra deps.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

_HEADER_FMT = "!I"  # unsigned 32-bit big-endian length prefix


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` or raise ``ConnectionError``."""
    chunks: list[bytes] = []
    got = 0
    while got < n:
        part = sock.recv(n - got)
        if not part:
            raise ConnectionError("sidecar connection closed unexpectedly")
        chunks.append(part)
        got += len(part)
    return b"".join(chunks)


def send_request(
    sock: socket.socket,
    audio: np.ndarray,
    sample_rate: int,
    instruction: str,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
) -> None:
    """Send one caption request over an established socket."""
    audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    header = {
        "sample_rate": int(sample_rate),
        "instruction": instruction,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n_samples": int(audio.shape[0]),
    }
    if system_prompt:
        header["system_prompt"] = system_prompt
    header_bytes = json.dumps(header).encode("utf-8")
    payload = audio.tobytes()
    sock.sendall(struct.pack(_HEADER_FMT, len(header_bytes)))
    sock.sendall(header_bytes)
    sock.sendall(payload)


def recv_request(sock: socket.socket) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Receive one caption request; return ``(audio, header_dict)``."""
    (header_len,) = struct.unpack(_HEADER_FMT, _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    n_samples = int(header["n_samples"])
    raw = _recv_exact(sock, n_samples * 4)
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    return audio, header


def send_response(sock: socket.socket, payload: Dict[str, Any]) -> None:
    """Send a JSON response (``{"caption": "..."}`` or ``{"error": "..."}``)."""
    body = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack(_HEADER_FMT, len(body)))
    sock.sendall(body)


def recv_response(sock: socket.socket, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Receive one JSON response; honour ``timeout`` if set."""
    if timeout is not None:
        sock.settimeout(timeout)
    try:
        (payload_len,) = struct.unpack(_HEADER_FMT, _recv_exact(sock, 4))
        body = _recv_exact(sock, payload_len)
        return json.loads(body.decode("utf-8"))
    finally:
        if timeout is not None:
            sock.settimeout(None)
