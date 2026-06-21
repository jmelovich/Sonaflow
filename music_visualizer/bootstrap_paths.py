"""Ensure vendored ``streamdiffusion`` (and optional dev override) is importable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Sonaflow repository root (parent of ``music_visualizer/``)."""
    return Path(__file__).resolve().parents[1]


def bootstrap_import_paths() -> None:
    """Put the vendored package on ``sys.path``; optional external checkout overrides it."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    override = os.environ.get("STREAMDIFFUSION_PATH", "").strip()
    if not override:
        return

    ext = Path(override).resolve()
    if not ext.is_dir():
        return

    # External StreamDiffusion checkouts expose the package under src/streamdiffusion.
    candidates = [ext, ext / "src"]
    for candidate in reversed(candidates):
        pkg = candidate / "streamdiffusion"
        if pkg.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
