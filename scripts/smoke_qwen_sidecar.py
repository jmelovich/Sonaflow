"""Smoke-test the Qwen2-Audio sidecar (load + one caption request)."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from music_visualizer.analysis.qwen_protocol import recv_response, send_request  # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    if not PYTHON.is_file():
        print("Main venv missing; run .\\setup.ps1 -Profile Full first.", file=sys.stderr)
        return 1

    proc = subprocess.Popen(
        [str(PYTHON), "-m", "music_visualizer.qwen_sidecar.server", "--4bit"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    port = None
    deadline = time.time() + 600
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline().strip()
        if line.startswith("QWEN_PORT="):
            port = int(line.split("=", 1)[1])
            break
        if not line and proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            print(f"Sidecar exited early: {err}", file=sys.stderr)
            return 2
    if port is None:
        proc.kill()
        print("Timed out waiting for QWEN_PORT", file=sys.stderr)
        return 3

    print(f"Sidecar on port {port}, sending two distinct signals...")
    sr = 16000
    t = np.linspace(0, 6.0, int(sr * 6), endpoint=False, dtype=np.float32)
    # A: calm low sine. B: bright noisy high-energy burst. Captions should differ.
    audio_a = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    rng = np.random.default_rng(0)
    audio_b = (0.6 * np.sin(2 * np.pi * 2000 * t)
               + 0.4 * rng.standard_normal(t.shape).astype(np.float32))
    audio_b = np.clip(audio_b, -1, 1).astype(np.float32)

    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    captions = []
    try:
        for label, audio in (("A calm-low", audio_a), ("B bright-noisy", audio_b)):
            send_request(
                sock, audio, sr,
                "In one short vivid phrase, describe this as a visual scene.",
                32,
            )
            resp = recv_response(sock, timeout=120)
            if "error" in resp:
                print(f"Sidecar error: {resp['error']}", file=sys.stderr)
                return 4
            cap = resp.get("caption", "")
            captions.append(cap)
            print(f"[{label}] {cap}")
    finally:
        sock.close()
        proc.terminate()

    if not all(captions):
        return 5
    if captions[0] == captions[1]:
        print("WARNING: identical captions for very different audio.", file=sys.stderr)
        return 6
    print("OK: captions differ by audio content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
