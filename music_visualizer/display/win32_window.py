"""Win32 helpers for OpenCV HighGUI windows (fullscreen, hwnd lookup)."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Dict, Optional

if sys.platform != "win32":
    def find_window(_title: str) -> int:
        return 0

    def toggle_borderless_fullscreen(_hwnd: int) -> bool:
        return False

else:
    user32 = ctypes.windll.user32

    GWL_STYLE = -16
    GWL_EXSTYLE = -20
    WS_POPUP = 0x80000000
    WS_VISIBLE = 0x10000000
    HWND_TOP = 0
    SWP_FRAMECHANGED = 0x0020
    SWP_SHOWWINDOW = 0x0040
    MONITOR_DEFAULTTONEAREST = 2

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    class _FullscreenState:
        __slots__ = ("active", "restored_rect", "restored_style", "restored_ex_style")

        def __init__(self) -> None:
            self.active = False
            self.restored_rect = RECT()
            self.restored_style = 0
            self.restored_ex_style = 0

    _states: Dict[int, _FullscreenState] = {}

    def find_window(title: str) -> int:
        return int(user32.FindWindowW(None, title))

    def toggle_borderless_fullscreen(hwnd: int) -> bool:
        """Toggle borderless fullscreen on a Win32 window. Returns new state."""
        if not hwnd:
            return False

        st = _states.get(hwnd)
        if st is None:
            st = _FullscreenState()
            _states[hwnd] = st

        if not st.active:
            user32.GetWindowRect(hwnd, ctypes.byref(st.restored_rect))
            st.restored_style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            st.restored_ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            mon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
            user32.GetMonitorInfoW(mon, ctypes.byref(mi))
            r = mi.rcMonitor

            user32.SetWindowLongW(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE)
            user32.SetWindowPos(
                hwnd,
                HWND_TOP,
                r.left,
                r.top,
                r.right - r.left,
                r.bottom - r.top,
                SWP_FRAMECHANGED | SWP_SHOWWINDOW,
            )
            st.active = True
        else:
            user32.SetWindowLongW(hwnd, GWL_STYLE, st.restored_style)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, st.restored_ex_style)
            r = st.restored_rect
            user32.SetWindowPos(
                hwnd,
                HWND_TOP,
                r.left,
                r.top,
                r.right - r.left,
                r.bottom - r.top,
                SWP_FRAMECHANGED | SWP_SHOWWINDOW,
            )
            st.active = False

        return st.active

    def is_borderless_fullscreen(hwnd: int) -> bool:
        st = _states.get(hwnd)
        return bool(st and st.active)
