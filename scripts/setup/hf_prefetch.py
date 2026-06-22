"""Prefetch HuggingFace models into the hub cache.

On Windows without Developer Mode, huggingface_hub's parallel snapshot_download
workers can race while probing symlink support and hit WinError 1314. We probe
each cache directory serially before starting parallel downloads.
"""

from __future__ import annotations

import os
import sys

if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import are_symlinks_supported


def model_cache_dir(model_id: str) -> str:
    return os.path.join(HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))


def prepare_hf_cache(model_ids: list[str]) -> None:
    """Mark symlink support before huggingface_hub starts parallel workers."""
    are_symlinks_supported(HF_HUB_CACHE)
    for model_id in model_ids:
        os.makedirs(model_cache_dir(model_id), exist_ok=True)
        are_symlinks_supported(model_cache_dir(model_id))


def prefetch(model_ids: list[str]) -> None:
    prepare_hf_cache(model_ids)
    for model_id in model_ids:
        print(f"Downloading {model_id}...", flush=True)
        snapshot_download(model_id)
    print("Done.", flush=True)


def main(argv: list[str] | None = None) -> int:
    model_ids = list(argv if argv is not None else sys.argv[1:])
    if not model_ids:
        print("Usage: hf_prefetch.py <model_id> [...]", file=sys.stderr)
        return 1
    prefetch(model_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
