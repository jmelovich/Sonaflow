"""Lock an OpenCV HighGUI window to a fixed aspect ratio (Windows only).

OpenCV has no built-in proportional resize, so we subclass the Win32 window
procedure and handle ``WM_SIZING`` to keep width:height fixed while the user
drags any edge or corner.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Optional

if sys.platform != "win32":
    AspectRatioHook = None  # type: ignore[misc, assignment]
else:
    user32 = ctypes.windll.user32

    GWLP_WNDPROC = -4
    WM_SIZING = 0x0214

    WMSZ_LEFT = 1
    WMSZ_RIGHT = 2
    WMSZ_TOP = 3
    WMSZ_TOPLEFT = 4
    WMSZ_TOPRIGHT = 5
    WMSZ_BOTTOM = 6
    WMSZ_BOTTOMLEFT = 7
    WMSZ_BOTTOMRIGHT = 8

    # Win64 LPARAM/WPARAM are 64-bit; wintypes.LPARAM is often only 32-bit signed
    # and overflows when forwarded to CallWindowProcW.
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        LRESULT = ctypes.c_longlong
        WPARAM_T = ctypes.c_uint64
        LPARAM_T = ctypes.c_int64
    else:
        LRESULT = ctypes.c_long
        WPARAM_T = wintypes.WPARAM
        LPARAM_T = wintypes.LPARAM

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    WNDPROC = ctypes.WINFUNCTYPE(
        LRESULT, wintypes.HWND, wintypes.UINT, WPARAM_T, LPARAM_T
    )

    CallWindowProcW = user32.CallWindowProcW
    CallWindowProcW.argtypes = [
        ctypes.c_void_p,
        wintypes.HWND,
        wintypes.UINT,
        WPARAM_T,
        LPARAM_T,
    ]
    CallWindowProcW.restype = LRESULT

    SetWindowLongPtrW = user32.SetWindowLongPtrW
    SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    SetWindowLongPtrW.restype = ctypes.c_void_p

    class AspectRatioHook:
        """Install on an OpenCV window title after ``cv2.namedWindow``."""

        def __init__(self, window_title: str, aspect_width: int, aspect_height: int):
            if aspect_width <= 0 or aspect_height <= 0:
                raise ValueError("aspect dimensions must be positive")
            self._title = window_title
            self._ratio = float(aspect_width) / float(aspect_height)
            self._hwnd: Optional[int] = None
            self._old_wndproc: Optional[ctypes.c_void_p] = None
            # Keep a strong ref so the callback is not collected.
            self._proc = WNDPROC(self._wnd_proc)

        def install(self) -> bool:
            hwnd = user32.FindWindowW(None, self._title)
            if not hwnd:
                return False
            self._hwnd = hwnd
            prev = SetWindowLongPtrW(
                hwnd, GWLP_WNDPROC, ctypes.cast(self._proc, ctypes.c_void_p)
            )
            if not prev:
                return False
            self._old_wndproc = prev
            return True

        def uninstall(self) -> None:
            if self._hwnd and self._old_wndproc:
                SetWindowLongPtrW(self._hwnd, GWLP_WNDPROC, self._old_wndproc)
            self._hwnd = None
            self._old_wndproc = None

        def _enforce(self, wparam: int, rect: RECT) -> None:
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w <= 0 or h <= 0:
                return

            if wparam == WMSZ_LEFT:
                new_h = max(1, int(round(w / self._ratio)))
                rect.top = rect.bottom - new_h
                return
            if wparam == WMSZ_RIGHT:
                new_h = max(1, int(round(w / self._ratio)))
                rect.bottom = rect.top + new_h
                return
            if wparam == WMSZ_TOP:
                new_w = max(1, int(round(h * self._ratio)))
                rect.left = rect.right - new_w
                return
            if wparam == WMSZ_BOTTOM:
                new_w = max(1, int(round(h * self._ratio)))
                rect.right = rect.left + new_w
                return
            if wparam == WMSZ_TOPLEFT:
                new_h = max(1, int(round(w / self._ratio)))
                rect.top = rect.bottom - new_h
                rect.left = rect.right - w
                return
            if wparam == WMSZ_TOPRIGHT:
                new_h = max(1, int(round(w / self._ratio)))
                rect.top = rect.bottom - new_h
                rect.right = rect.left + w
                return
            if wparam == WMSZ_BOTTOMLEFT:
                new_h = max(1, int(round(w / self._ratio)))
                rect.bottom = rect.top + new_h
                rect.left = rect.right - w
                return
            if wparam == WMSZ_BOTTOMRIGHT:
                new_h = max(1, int(round(w / self._ratio)))
                rect.bottom = rect.top + new_h
                rect.right = rect.left + w

        def _wnd_proc(self, hwnd, msg, wparam, lparam):
            if msg == WM_SIZING and self._old_wndproc:
                rect = ctypes.cast(lparam, ctypes.POINTER(RECT)).contents
                self._enforce(int(wparam), rect)
                return 1
            return CallWindowProcW(
                self._old_wndproc, hwnd, msg, wparam, lparam
            )
