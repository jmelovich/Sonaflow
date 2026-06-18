"""In-window settings UI for the music visualizer."""

from music_visualizer.ui.controller import UIController
from music_visualizer.ui.persistence import DEFAULTS_PATH, load_defaults_dict, merge_defaults

__all__ = [
    "UIController",
    "DEFAULTS_PATH",
    "load_defaults_dict",
    "merge_defaults",
]
