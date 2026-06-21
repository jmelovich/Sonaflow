"""Qwen2-Audio sidecar TCP server.

Run as a subprocess from the main app (same ``.venv``)::

    python -m music_visualizer.qwen_sidecar.server [--model MODEL] [--4bit]

Prints ``QWEN_PORT=<n>`` as the first stdout line; all model logs go to stderr.
"""

from __future__ import annotations

import argparse
import socket
import sys
import traceback
from pathlib import Path
from typing import Optional

# Make ``import music_visualizer.*`` work when launched as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from music_visualizer.analysis.qwen_protocol import (  # noqa: E402
    recv_request,
    send_response,
)
from music_visualizer.utils.text import english_only_text  # noqa: E402


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_model(model_id: str, load_4bit: bool):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    _log(f"Loading Qwen2-Audio: {model_id} (4bit={load_4bit})")
    processor = AutoProcessor.from_pretrained(model_id)

    if load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map={"": 0},
            quantization_config=quant,
        )
    else:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )

    model.eval()
    _log("Qwen2-Audio ready.")
    return model, processor


def _resample(audio, src_sr: int, dst_sr: int):
    if src_sr == dst_sr or audio.size == 0:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr)
    except ImportError:
        import numpy as np

        n_out = int(round(audio.shape[0] * dst_sr / float(src_sr)))
        if n_out <= 1:
            return audio.astype("float32")
        x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype("float32")


def _caption(
    model,
    processor,
    audio,
    src_sr: int,
    instruction: str,
    max_new_tokens: int,
    system_prompt: Optional[str] = None,
    do_sample: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.9,
) -> str:
    import torch

    target_sr = int(processor.feature_extractor.sampling_rate)
    audio = _resample(audio, src_sr, target_sr)

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder.wav"},
                {"type": "text", "text": instruction},
            ],
        }
    )
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    # transformers 5.x renamed the processor audio kwarg to ``audio`` (singular);
    # older releases used ``audios``. Passing the wrong name is silently ignored,
    # so the model would get no audio features and emit a generic caption
    # regardless of input. Try the new name first, fall back to the old one.
    try:
        inputs = processor(
            text=text, audio=[audio], sampling_rate=target_sr,
            return_tensors="pt", padding=True,
        )
    except (TypeError, ValueError):
        inputs = processor(
            text=text, audios=[audio], sampling_rate=target_sr,
            return_tensors="pt", padding=True,
        )
    if "input_features" not in inputs:
        _log("WARNING: processor produced no audio features; caption will be generic.")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        # Sampling makes each caption vary instead of collapsing to one generic
        # greedy attractor, so the prompt actually tracks the music over time.
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, **gen_kwargs)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    caption = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return english_only_text(caption.strip())


def _handle_client(conn: socket.socket, model, processor) -> None:
    audio, header = recv_request(conn)
    caption = _caption(
        model,
        processor,
        audio,
        int(header["sample_rate"]),
        str(header["instruction"]),
        int(header.get("max_new_tokens", 64)),
        system_prompt=header.get("system_prompt") or None,
        do_sample=bool(header.get("do_sample", True)),
        temperature=float(header.get("temperature", 0.9)),
        top_p=float(header.get("top_p", 0.9)),
    )
    send_response(conn, {"caption": caption})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Qwen2-Audio sidecar server")
    p.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    p.add_argument("--4bit", dest="load_4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    args = p.parse_args(argv)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    _host, port = server.getsockname()
    # Handshake before model load so the parent process can connect promptly.
    print(f"QWEN_PORT={port}", flush=True)

    model, processor = _load_model(args.model, args.load_4bit)

    try:
        while True:
            conn, _addr = server.accept()
            try:
                with conn:
                    while True:
                        try:
                            _handle_client(conn, model, processor)
                        except ConnectionError:
                            break
                        except Exception:
                            _log(traceback.format_exc())
                            try:
                                send_response(conn, {"error": "inference failed"})
                            except Exception:
                                break
            except Exception:
                _log(traceback.format_exc())
    except KeyboardInterrupt:
        _log("Sidecar shutting down.")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
