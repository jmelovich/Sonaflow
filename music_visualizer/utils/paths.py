"""Filesystem locations, with packaged-install overrides via environment.

In a normal dev checkout, mutable data (settings, model cache, engines, logs)
lives inside the repo. In a packaged install the read-only program files must not
be written to, so the launcher points these at a writable per-user directory.

Environment overrides (set by the packaged launcher; unset in dev):

* ``SONAFLOW_DATA_DIR``      - root for all mutable data (settings, logs, ...).
* ``SONAFLOW_SETTINGS_PATH`` - explicit path to ``visualizer_settings.json``.

When unset, everything falls back to the repository root so dev workflows are
unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Sonaflow repository root (parent of ``music_visualizer/``)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Writable root for mutable data (packaged) or the repo root (dev)."""
    env = os.environ.get("SONAFLOW_DATA_DIR")
    if env:
        return Path(env)
    return repo_root()


def settings_path() -> Path:
    """Location of ``visualizer_settings.json``."""
    env = os.environ.get("SONAFLOW_SETTINGS_PATH")
    if env:
        return Path(env)
    return data_dir() / "visualizer_settings.json"
