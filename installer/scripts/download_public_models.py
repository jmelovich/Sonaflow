"""Download publicly available HuggingFace models into HF_HOME.

Used by the offline installer after the app runtime is extracted. TensorRT
engines are NOT downloaded here (they are built on the target GPU at first launch).

Model IDs match setup.lock.json.
"""

from __future__ import annotations

import argparse
import os
import sys

# Public HuggingFace repos (no license gate beyond HF terms).
CORE_MODELS = (
    "stabilityai/sd-turbo",
    "madebyollin/taesd",
    "laion/larger_clap_music",
)
QWEN_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download Sonaflow public HF models.")
    p.add_argument(
        "--skip-core",
        action="store_true",
        help="Skip SD-Turbo, taesd, and CLAP (download Qwen only with --qwen).",
    )
    p.add_argument(
        "--qwen",
        action="store_true",
        help=f"Also download {QWEN_MODEL} (~16 GB).",
    )
    args = p.parse_args(argv)

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("HF_HOME is not set.", file=sys.stderr)
        return 1
    os.makedirs(hf_home, exist_ok=True)

    from huggingface_hub import snapshot_download

    # Import shared prefetch helper (repo root on sys.path when run from installer).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.setup.hf_prefetch import prepare_hf_cache

    models: list[str] = []
    if not args.skip_core:
        models.extend(CORE_MODELS)
    if args.qwen:
        models.append(QWEN_MODEL)
    if not models:
        print("Nothing to download (use default core set and/or --qwen).", file=sys.stderr)
        return 1

    prepare_hf_cache(models)

    for model_id in models:
        print(f"Downloading {model_id} ...", flush=True)
        snapshot_download(model_id)
        print(f"  OK: {model_id}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
