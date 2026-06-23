"""Persist settings-panel UI state (section collapse) between sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

from music_visualizer.ui.settings_schema import ParamSpec
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.paths import data_dir

logger = get_logger(__name__)

DEFAULT_COLLAPSED = True


def panel_state_path() -> Path:
    return data_dir() / "ui_panel_state.json"


def unique_groups(specs: Sequence[ParamSpec]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for spec in specs:
        if spec.group not in seen:
            seen.add(spec.group)
            ordered.append(spec.group)
    return tuple(ordered)


def load_group_collapsed(specs: Sequence[ParamSpec]) -> Dict[str, bool]:
    """Return ``{group: collapsed}`` for every group in ``specs``.

    Groups missing from the saved file default to collapsed.
    """
    saved: dict = {}
    path = panel_state_path()
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            raw = data.get("group_collapsed", {})
            if isinstance(raw, dict):
                saved = raw
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read UI panel state from %s (%s).", path, exc)
    return {group: bool(saved.get(group, DEFAULT_COLLAPSED)) for group in unique_groups(specs)}


def save_group_collapsed(states: Dict[str, bool]) -> None:
    path = panel_state_path()
    payload = {
        "_comment": "Settings panel section collapse state (true = collapsed).",
        "group_collapsed": dict(sorted(states.items())),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        logger.warning("Could not save UI panel state to %s (%s).", path, exc)


def merge_group_collapsed(
    existing: Dict[str, bool],
    specs: Sequence[ParamSpec],
) -> Dict[str, bool]:
    """Keep saved states and default any new groups to collapsed."""
    merged = dict(existing)
    for group in unique_groups(specs):
        merged.setdefault(group, DEFAULT_COLLAPSED)
    return {group: bool(merged[group]) for group in unique_groups(specs)}
