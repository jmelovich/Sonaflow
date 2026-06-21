"""Smoke-test unified main venv after transformers bump.

Validates imports, compel weighted prompts, and CLAP (when models are cached).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def test_imports() -> int:
    import diffusers
    import transformers
    from transformers import ClapModel, Qwen2AudioForConditionalGeneration

    import compel  # noqa: F401

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (4, 45):
        return _fail(
            f"transformers {transformers.__version__} < 4.45 (need Qwen2Audio)"
        )
    _ok(
        f"imports (transformers {transformers.__version__}, "
        f"diffusers {diffusers.__version__})"
    )
    _ok("Qwen2AudioForConditionalGeneration + ClapModel available")
    return 0


def test_compel() -> int:
    from huggingface_hub import try_to_load_from_cache

    model_id = "stabilityai/sd-turbo"
    if not try_to_load_from_cache(model_id, "config.json"):
        print("SKIP: compel (SD-Turbo not cached)")
        return 0

    import torch
    from compel import Compel
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    compel_proc = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=False,
    )
    embeds = compel_proc.build_conditioning_tensor("(neon)1.2 city")
    if embeds.ndim != 3 or embeds.shape[0] != 1:
        return _fail(f"unexpected compel embed shape: {tuple(embeds.shape)}")
    _ok(f"compel weighted prompt -> {tuple(embeds.shape)}")
    return 0


def test_clap() -> int:
    from huggingface_hub import try_to_load_from_cache

    model_id = "laion/larger_clap_music"
    if not try_to_load_from_cache(model_id, "config.json"):
        print("SKIP: CLAP (model not cached)")
        return 0

    import numpy as np

    from music_visualizer.analysis.clap_analyzer import CLAPAnalyzer

    analyzer = CLAPAnalyzer(
        vocabulary=["neon city", "dark forest", "calm ocean"],
        device="cpu",
        calibrate=False,
    )
    audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000, endpoint=False))).astype(
        np.float32
    )
    scores = analyzer.score(audio, 48000)
    if not scores:
        return _fail("CLAP returned no scores")
    _ok(f"CLAP scored {len(scores)} phrases")
    return 0


def main() -> int:
    for fn in (test_imports, test_compel, test_clap):
        rc = fn()
        if rc != 0:
            return rc
    print("All unified-env checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
