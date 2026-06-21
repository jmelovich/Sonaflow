"""Write installer-chosen generation settings into visualizer_settings.json.

Merges width/height/acceleration/qwen into any existing file so a reinstall
updates resolution without wiping other saved preferences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "app", here.parent.parent):
        if (candidate / "music_visualizer").is_dir():
            sys.path.insert(0, str(candidate))
            return
    raise ImportError("Could not locate music_visualizer package next to this script.")


_bootstrap_import_path()
from music_visualizer.utils.resolution import generation_size_for_display


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("display_width", type=int)
    p.add_argument("display_height", type=int)
    p.add_argument("settings_path", type=Path)
    p.add_argument("--qwen", action="store_true")
    args = p.parse_args(argv)

    width, height = generation_size_for_display(args.display_width, args.display_height)
    path = args.settings_path
    data: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            pass

    data["width"] = width
    data["height"] = height
    data["acceleration"] = "tensorrt"
    data["qwen"] = bool(args.qwen)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {width}x{height} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
