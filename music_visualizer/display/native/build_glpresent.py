"""Build the GPU presenter bridge DLL (``glpresent_bridge.dll``).

Compiles ``glpresent_bridge.cpp`` + Dear ImGui (docking branch) into a small
DLL that the Python presenter loads via ctypes. Built on first use by
:mod:`music_visualizer.display.gl_presenter`, but can also be run by hand:

    python -m music_visualizer.display.native.build_glpresent
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from music_visualizer.utils.cuda_toolkit import require_cuda_root
from music_visualizer.utils.msvc_toolchain import require_msvc_env

HERE = Path(__file__).resolve().parent
SRC = HERE / "glpresent_bridge.cpp"
DLL = HERE / "glpresent_bridge.dll"
IMGUI = HERE / "imgui"

IMGUI_SOURCES = [
    IMGUI / "imgui.cpp",
    IMGUI / "imgui_draw.cpp",
    IMGUI / "imgui_widgets.cpp",
    IMGUI / "imgui_tables.cpp",
    IMGUI / "backends" / "imgui_impl_win32.cpp",
    IMGUI / "backends" / "imgui_impl_opengl2.cpp",
]

# Exported by the settings UI; used to detect pre-ImGui bridge builds.
_UI_PROBE_SYMBOL = "glpresent_ui_set_params"


def _cuda_root() -> Path:
    return require_cuda_root(header="cudaGL.h")


def _vcvars() -> str:
    return require_msvc_env(gl_presenter=True).call_line


def _sources_newer_than_dll() -> bool:
    if not DLL.is_file():
        return True
    dll_mtime = DLL.stat().st_mtime
    sources = [SRC, *IMGUI_SOURCES]
    return any(s.stat().st_mtime > dll_mtime for s in sources if s.is_file())


def _dll_missing_ui_exports(path: Optional[Path] = None) -> bool:
    """True when the built DLL predates the settings UI exports."""
    dll = path or DLL
    if not dll.is_file():
        return True
    try:
        lib = ctypes.CDLL(str(dll))
    except OSError:
        return True
    return not hasattr(lib, _UI_PROBE_SYMBOL)


def _needs_rebuild() -> bool:
    return _sources_newer_than_dll() or _dll_missing_ui_exports()


def ensure_dll(*, log=None) -> Path:
    """Build ``glpresent_bridge.dll`` when missing, stale, or lacking UI exports.

    Returns the path to load. If a rebuild is needed but the installed DLL is
    locked (visualizer already running), the existing DLL is kept and a warning
    is logged.
    """
    if not _needs_rebuild():
        return DLL

    if log is not None:
        if not DLL.is_file():
            log.info("glpresent_bridge.dll missing; building...")
        elif _sources_newer_than_dll():
            log.info("glpresent_bridge.dll out of date; rebuilding...")
        else:
            log.info("glpresent_bridge.dll missing settings UI; rebuilding...")

    try:
        return build(force=DLL.is_file())
    except RuntimeError as exc:
        if DLL.is_file():
            if log is not None:
                log.warning(
                    "Could not update glpresent_bridge.dll (%s). "
                    "Close any running visualizer and restart to pick up changes.",
                    exc,
                )
            return DLL
        raise


def _ensure_imgui(*, log=None) -> None:
    """Clone Dear ImGui (docking branch) when missing."""
    if all(s.is_file() for s in IMGUI_SOURCES):
        return
    if IMGUI.is_dir() and any(IMGUI.iterdir()):
        missing = [s.name for s in IMGUI_SOURCES if not s.is_file()]
        raise RuntimeError(
            f"Incomplete Dear ImGui checkout at {IMGUI} (missing {missing}). "
            "Delete the folder and rebuild."
        )
    if log is not None:
        log.info("Dear ImGui not found; cloning docking branch into %s", IMGUI)
    else:
        print(f"[glpresent] cloning Dear ImGui into {IMGUI}")
    IMGUI.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "docking",
            "https://github.com/ocornut/imgui.git",
            str(IMGUI),
        ]
    )


def build(force: bool = False) -> Path:
    if not force and not _needs_rebuild():
        return DLL

    missing = [s for s in IMGUI_SOURCES if not s.is_file()]
    if missing:
        _ensure_imgui()

    missing = [s for s in IMGUI_SOURCES if not s.is_file()]
    if missing:
        raise RuntimeError(
            "Dear ImGui sources not found after clone. Missing: "
            + ", ".join(s.name for s in missing)
        )

    cuda = _cuda_root()
    vcvars = _vcvars()

    inc_cuda = cuda / "include"
    lib_cuda = cuda / "lib" / "x64"
    inc_imgui = IMGUI
    inc_backends = IMGUI / "backends"

    src_list = " ".join(f'"{s}"' for s in [SRC, *IMGUI_SOURCES])
    tmp_dll = HERE / "glpresent_bridge.dll.tmp"
    cl_cmd = (
        "cl /nologo /O2 /EHsc /LD /MD "
        f"{src_list} "
        f'/I "{inc_cuda}" '
        f'/I "{inc_imgui}" '
        f'/I "{inc_backends}" '
        f'/Fe:"{tmp_dll}" '
        "/link "
        f'/LIBPATH:"{lib_cuda}" '
        "cuda.lib opengl32.lib gdi32.lib user32.lib comdlg32.lib ole32.lib"
    )

    with tempfile.TemporaryDirectory() as tmp:
        bat = Path(tmp) / "build.bat"
        bat.write_text(
            "@echo off\r\n"
            f"{vcvars}\r\n"
            f"cd /d \"{tmp}\"\r\n"
            f"{cl_cmd}\r\n"
            "exit /b %ERRORLEVEL%\r\n",
            encoding="utf-8",
        )
        print(f"[glpresent] compiling bridge + ImGui -> {DLL.name}")
        proc = subprocess.run(
            ["cmd", "/c", str(bat)],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not tmp_dll.is_file():
            raise RuntimeError(
                "GL presenter bridge build failed.\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )

    try:
        tmp_dll.replace(DLL)
    except PermissionError:
        if DLL.is_file():
            raise RuntimeError(
                f"Could not replace {DLL} (is the visualizer running?). "
                "Close the app and restart to load the new build."
            ) from None
        # No installed DLL yet — load the freshly built temp artifact.
        return tmp_dll

    print(f"[glpresent] built {DLL}")
    return DLL


if __name__ == "__main__":
    build(force="--force" in sys.argv)
