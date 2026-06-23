"""ctypes bindings for the native settings UI in glpresent_bridge.dll."""

from __future__ import annotations

import ctypes
from enum import IntEnum
from typing import List, Optional, Sequence

from music_visualizer.ui.settings_schema import ParamKind, ParamSpec


class UiParamType(IntEnum):
    BOOL = 0
    INT = 1
    FLOAT = 2
    ENUM = 3
    TEXT = 4


class UiAction(IntEnum):
    SAVE_DEFAULTS = 1
    EXPORT = 2
    IMPORT = 4
    RESTART = 8
    POP_OUT = 16


class UiParamDef(ctypes.Structure):
    _fields_ = [
        ("label", ctypes.c_wchar_p),
        ("group", ctypes.c_wchar_p),
        ("tooltip", ctypes.c_wchar_p),
        ("type", ctypes.c_int),
        ("min_val", ctypes.c_double),
        ("max_val", ctypes.c_double),
        ("step", ctypes.c_double),
        ("enum_csv", ctypes.c_wchar_p),
        ("requires_restart", ctypes.c_int),
    ]


def _kind_to_native(kind: ParamKind) -> int:
    return {
        ParamKind.BOOL: UiParamType.BOOL,
        ParamKind.INT: UiParamType.INT,
        ParamKind.FLOAT: UiParamType.FLOAT,
        ParamKind.ENUM: UiParamType.ENUM,
        ParamKind.TEXT: UiParamType.TEXT,
    }[kind]


class NativeUIBridge:
    """Thin wrapper around the DLL's settings UI exports."""

    _UI_PROBE = "glpresent_ui_set_params"

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib
        self._specs: List[ParamSpec] = []
        self._available = hasattr(lib, self._UI_PROBE)
        if self._available:
            self._bind()

    @classmethod
    def available(cls, lib: ctypes.CDLL) -> bool:
        return hasattr(lib, cls._UI_PROBE)

    def _bind(self) -> None:
        lib = self._lib
        lib.glpresent_ui_set_params.argtypes = [
            ctypes.POINTER(UiParamDef), ctypes.c_int,
        ]
        lib.glpresent_ui_set_params.restype = ctypes.c_int
        lib.glpresent_ui_set_value.argtypes = [ctypes.c_int, ctypes.c_double]
        lib.glpresent_ui_set_value.restype = None
        lib.glpresent_ui_set_text.argtypes = [ctypes.c_int, ctypes.c_wchar_p]
        lib.glpresent_ui_set_text.restype = None
        lib.glpresent_ui_set_default_value.argtypes = [ctypes.c_int, ctypes.c_double]
        lib.glpresent_ui_set_default_value.restype = None
        lib.glpresent_ui_set_default_text.argtypes = [ctypes.c_int, ctypes.c_wchar_p]
        lib.glpresent_ui_set_default_text.restype = None
        lib.glpresent_ui_take_changes.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        lib.glpresent_ui_take_changes.restype = ctypes.c_int
        lib.glpresent_ui_param_changed_text.argtypes = [ctypes.c_int]
        lib.glpresent_ui_param_changed_text.restype = ctypes.c_int
        lib.glpresent_ui_get_text.argtypes = [ctypes.c_int]
        lib.glpresent_ui_get_text.restype = ctypes.c_wchar_p
        lib.glpresent_ui_take_actions.argtypes = []
        lib.glpresent_ui_take_actions.restype = ctypes.c_int
        lib.glpresent_ui_set_restart_pending.argtypes = [ctypes.c_int]
        lib.glpresent_ui_set_restart_pending.restype = None
        lib.glpresent_ui_open_file_dialog.argtypes = [
            ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int,
        ]
        lib.glpresent_ui_open_file_dialog.restype = ctypes.c_int
        lib.glpresent_take_ui_toggle.argtypes = []
        lib.glpresent_take_ui_toggle.restype = ctypes.c_int
        lib.glpresent_ui_is_visible.argtypes = []
        lib.glpresent_ui_is_visible.restype = ctypes.c_int
        lib.glpresent_ui_set_visible.argtypes = [ctypes.c_int]
        lib.glpresent_ui_set_visible.restype = None
        if hasattr(lib, "glpresent_ui_set_group_collapsed"):
            lib.glpresent_ui_set_group_collapsed.argtypes = [
                ctypes.c_wchar_p, ctypes.c_int,
            ]
            lib.glpresent_ui_set_group_collapsed.restype = None
            lib.glpresent_ui_take_group_collapsed_changes.argtypes = [
                ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ]
            lib.glpresent_ui_take_group_collapsed_changes.restype = ctypes.c_int

    @property
    def group_collapse_available(self) -> bool:
        return self._available and hasattr(self._lib, "glpresent_ui_set_group_collapsed")

    def set_schema(self, specs: Sequence[ParamSpec]) -> None:
        if not self._available:
            return
        self._specs = list(specs)
        arr = (UiParamDef * len(specs))()
        # Hold explicit buffers for every wide string. Assigning Python str
        # temporaries (e.g. "|".join(...)) directly to c_wchar_p fields of
        # struct array elements does NOT reliably keep them alive in ctypes,
        # so the C side would read freed memory (garbage enum options, etc.).
        keepalive: list = []

        def _wbuf(text: Optional[str]):
            if text is None:
                return None
            buf = ctypes.create_unicode_buffer(text)
            keepalive.append(buf)
            return ctypes.cast(buf, ctypes.c_wchar_p)

        for i, spec in enumerate(specs):
            arr[i].label = _wbuf(spec.label)
            arr[i].group = _wbuf(spec.group)
            arr[i].tooltip = _wbuf(spec.tooltip)
            arr[i].type = _kind_to_native(spec.kind)
            arr[i].min_val = spec.min_val
            arr[i].max_val = spec.max_val
            arr[i].step = spec.step
            arr[i].enum_csv = _wbuf("|".join(spec.options) if spec.options else None)
            arr[i].requires_restart = 1 if spec.requires_restart else 0
        self._schema_keepalive = keepalive
        n = self._lib.glpresent_ui_set_params(arr, len(specs))
        if n != len(specs):
            raise RuntimeError(f"glpresent_ui_set_params returned {n}, expected {len(specs)}")

    def push_value(self, idx: int, value) -> None:
        if not self._available:
            return
        spec = self._specs[idx]
        if spec.kind in (ParamKind.TEXT, ParamKind.ENUM):
            self._lib.glpresent_ui_set_text(idx, str(value or ""))
        elif spec.kind == ParamKind.BOOL:
            self._lib.glpresent_ui_set_value(idx, 1.0 if value else 0.0)
        elif spec.kind == ParamKind.INT:
            self._lib.glpresent_ui_set_value(idx, float(int(value)))
        else:
            self._lib.glpresent_ui_set_value(idx, float(value))

    def push_default(self, idx: int, value) -> None:
        if not self._available:
            return
        spec = self._specs[idx]
        if spec.kind in (ParamKind.TEXT, ParamKind.ENUM):
            self._lib.glpresent_ui_set_default_text(idx, str(value or ""))
        elif spec.kind == ParamKind.BOOL:
            self._lib.glpresent_ui_set_default_value(idx, 1.0 if value else 0.0)
        elif spec.kind == ParamKind.INT:
            self._lib.glpresent_ui_set_default_value(idx, float(int(value)))
        else:
            self._lib.glpresent_ui_set_default_value(idx, float(value))

    def take_numeric_changes(self, max_n: int = 64) -> list[tuple[int, float]]:
        idx_buf = (ctypes.c_int * max_n)()
        val_buf = (ctypes.c_double * max_n)()
        n = self._lib.glpresent_ui_take_changes(idx_buf, val_buf, max_n)
        return [(idx_buf[i], val_buf[i]) for i in range(n)]

    def take_text_changes(self) -> list[int]:
        changed: list[int] = []
        for i, spec in enumerate(self._specs):
            if spec.kind not in (ParamKind.TEXT, ParamKind.ENUM):
                continue
            if self._lib.glpresent_ui_param_changed_text(i):
                changed.append(i)
        return changed

    def get_text(self, idx: int) -> str:
        ptr = self._lib.glpresent_ui_get_text(idx)
        return ptr if ptr else ""

    def take_actions(self) -> int:
        return int(self._lib.glpresent_ui_take_actions())

    def set_restart_pending(self, pending: bool) -> None:
        self._lib.glpresent_ui_set_restart_pending(1 if pending else 0)

    def open_file_dialog(self, save: bool) -> Optional[str]:
        buf = ctypes.create_unicode_buffer(1024)
        ok = self._lib.glpresent_ui_open_file_dialog(1 if save else 0, buf, len(buf))
        return buf.value if ok else None

    def take_ui_toggle(self) -> bool:
        return bool(self._lib.glpresent_take_ui_toggle())

    def set_group_collapsed(self, group: str, collapsed: bool) -> None:
        if not self.group_collapse_available:
            return
        self._lib.glpresent_ui_set_group_collapsed(group, 1 if collapsed else 0)

    def sync_group_collapsed(self, states: dict[str, bool]) -> None:
        if not self.group_collapse_available:
            return
        for group, collapsed in states.items():
            self.set_group_collapsed(group, collapsed)

    def take_group_collapsed_changes(self, max_n: int = 32) -> list[tuple[str, bool]]:
        if not self.group_collapse_available:
            return []
        out: list[tuple[str, bool]] = []
        buf = ctypes.create_unicode_buffer(256)
        collapsed = ctypes.c_int()
        for _ in range(max_n):
            if not self._lib.glpresent_ui_take_group_collapsed_changes(
                buf, len(buf), ctypes.byref(collapsed),
            ):
                break
            out.append((buf.value, bool(collapsed.value)))
        return out
