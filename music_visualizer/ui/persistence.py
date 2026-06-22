"""Load/save visualizer settings as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

from music_visualizer.config import Config
from music_visualizer.utils.paths import settings_path as _settings_path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Honors SONAFLOW_SETTINGS_PATH / SONAFLOW_DATA_DIR in packaged installs; falls
# back to the repo root in a dev checkout.
DEFAULTS_PATH = _settings_path()

# Fields not persisted in the main settings file. ``react_routes`` lives in the
# external routes file (react_routes_path); persisting it here too would create
# a second, conflicting source of truth.
_SKIP_KEYS = frozenset({"benchmark", "react_routes"})


def _config_to_dict(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(cfg):
        if f.name in _SKIP_KEYS:
            continue
        val = getattr(cfg, f.name)
        if val is None:
            out[f.name] = None
        elif isinstance(val, list):
            out[f.name] = list(val)
        else:
            out[f.name] = val
    return out


def _apply_dict(cfg: Config, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(cfg)}
    for key, val in data.items():
        if key not in valid or key in _SKIP_KEYS:
            continue
        setattr(cfg, key, val)


def load_defaults_dict(path: Optional[Path] = None) -> dict[str, Any]:
    path = path or DEFAULTS_PATH
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_defaults(cfg: Config, path: Optional[Path] = None) -> Path:
    path = path or DEFAULTS_PATH
    data = _config_to_dict(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def export_settings(cfg: Config, path: Path) -> None:
    data = _config_to_dict(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def import_settings(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Settings file must contain a JSON object.")
    return data


def merge_defaults(base: Config, defaults: dict[str, Any], overrides: dict[str, Any]) -> Config:
    """Build config: defaults < overrides (CLI wins).

    Returns a new Config with ``__post_init__`` applied once.
    """
    merged = dict(defaults)
    merged.update(overrides)
    return Config(**merged)
