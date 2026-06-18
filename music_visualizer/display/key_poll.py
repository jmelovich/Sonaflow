"""Lightweight keyboard edge detection (main-thread helpers)."""

from __future__ import annotations

import sys


class F1Edge:
    """True once per F1 key press (GetAsyncKeyState on Windows, main thread)."""

    def __init__(self) -> None:
        self._was_down = False

    def pressed(self) -> bool:
        if sys.platform != "win32":
            return False
        import ctypes

        down = bool(ctypes.windll.user32.GetAsyncKeyState(0x70) & 0x8000)
        edge = down and not self._was_down
        self._was_down = down
        return edge
