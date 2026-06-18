"""Build the RTX VSR bridge DLL (``rtxvsr_bridge.dll``).

Compiles ``rtxvsr_bridge.cpp`` against the NVIDIA RTX Video SDK (NGX) and the
CUDA driver API into a small DLL that the Python upscaler loads via ctypes.

This is invoked automatically by :mod:`music_visualizer.generation.upscaler` when
the DLL is missing, but can also be run by hand:

    python -m music_visualizer.generation.native.build_rtxvsr

Requirements (already present on the dev box):
* ``NV_RTX_VIDEO_SDK`` env var pointing at the SDK root (bin/include/lib/...).
* A CUDA toolkit (for ``cuda.h`` + ``cuda.lib``) -- found via ``CUDA_PATH`` or the
  default install dir.
* MSVC (Visual Studio 2019+ Build Tools) -- located via ``vswhere``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "rtxvsr_bridge.cpp"
DLL = HERE / "rtxvsr_bridge.dll"


def _sdk_root() -> Path:
    root = os.environ.get("NV_RTX_VIDEO_SDK")
    if not root:
        raise RuntimeError(
            "NV_RTX_VIDEO_SDK is not set. Point it at the RTX Video SDK root "
            "(the directory containing bin/, include/, lib/, samples/)."
        )
    p = Path(root)
    if not (p / "include" / "nvsdk_ngx.h").is_file():
        raise RuntimeError(f"NV_RTX_VIDEO_SDK={p} does not look like the SDK root.")
    return p


def _cuda_root() -> Path:
    candidates = []
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]))
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.is_dir():
        candidates.extend(sorted(base.glob("v*"), reverse=True))
    for c in candidates:
        if (c / "include" / "cuda.h").is_file() and (c / "lib" / "x64" / "cuda.lib").is_file():
            return c
    raise RuntimeError(
        "Could not find a CUDA toolkit with cuda.h + lib/x64/cuda.lib. "
        "Install the CUDA toolkit or set CUDA_PATH."
    )


def _vcvars() -> Path:
    vswhere = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    ) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError(f"vswhere.exe not found at {vswhere}. Install Visual Studio.")
    out = subprocess.check_output(
        [str(vswhere), "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        text=True,
    ).strip()
    if not out:
        raise RuntimeError("vswhere found no VS install with the C++ toolset.")
    vcvars = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        raise RuntimeError(f"vcvars64.bat not found at {vcvars}.")
    return vcvars


def build(force: bool = False) -> Path:
    if DLL.is_file() and not force and DLL.stat().st_mtime >= SRC.stat().st_mtime:
        return DLL

    sdk = _sdk_root()
    cuda = _cuda_root()
    vcvars = _vcvars()

    inc_sdk = sdk / "include"
    lib_sdk = sdk / "lib" / "Windows" / "x64"
    inc_cuda = cuda / "include"
    lib_cuda = cuda / "lib" / "x64"

    # /MT: the static NGX lib (nvsdk_ngx_s.lib) is built against the static CRT,
    # so we must match it. cudart_static.lib satisfies the CUDA *runtime* symbols
    # the NGX lib pulls in (statically, so there's no external cudart DLL dep).
    cl_cmd = (
        'cl /nologo /O2 /EHsc /LD /MT '
        f'"{SRC}" '
        f'/I "{inc_sdk}" /I "{inc_cuda}" '
        f'/Fe:"{DLL}" '
        '/link '
        f'/LIBPATH:"{lib_sdk}" /LIBPATH:"{lib_cuda}" '
        'nvsdk_ngx_s.lib cuda.lib cudart_static.lib '
        'user32.lib advapi32.lib shell32.lib'
    )

    with tempfile.TemporaryDirectory() as tmp:
        bat = Path(tmp) / "build.bat"
        bat.write_text(
            "@echo off\r\n"
            f'call "{vcvars}" >nul\r\n'
            f"cd /d \"{tmp}\"\r\n"
            f"{cl_cmd}\r\n"
            "exit /b %ERRORLEVEL%\r\n",
            encoding="utf-8",
        )
        print(f"[rtxvsr] compiling {SRC.name} -> {DLL.name}")
        proc = subprocess.run(
            ["cmd", "/c", str(bat)],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not DLL.is_file():
            raise RuntimeError(
                "RTX VSR bridge build failed.\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )

    print(f"[rtxvsr] built {DLL}")
    return DLL


if __name__ == "__main__":
    build(force="--force" in sys.argv)
