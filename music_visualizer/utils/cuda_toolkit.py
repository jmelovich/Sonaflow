"""CUDA toolkit discovery for native bridge DLL builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_CUDA_BASE = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")


def _cuda_from_registry() -> list[Path]:
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    results: list[Path] = []
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA",
        ) as key:
            index = 0
            while True:
                try:
                    subname = winreg.EnumKey(key, index)
                    with winreg.OpenKey(key, subname) as subkey:
                        install_dir, _ = winreg.QueryValueEx(subkey, "InstallDir")
                    results.append(Path(str(install_dir)))
                    index += 1
                except OSError:
                    break
    except OSError:
        pass
    return results


def cuda_candidates() -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(path)

    if os.environ.get("CUDA_PATH"):
        add(Path(os.environ["CUDA_PATH"]))
    for path in _cuda_from_registry():
        add(path)
    if _CUDA_BASE.is_dir():
        for path in sorted(_CUDA_BASE.glob("v*"), key=lambda p: p.name, reverse=True):
            add(path)
    return ordered


def find_cuda_root(*, header: str = "cuda.h") -> Optional[Path]:
    for candidate in cuda_candidates():
        if (candidate / "include" / header).is_file() and (
            candidate / "lib" / "x64" / "cuda.lib"
        ).is_file():
            return candidate
    return None


def torch_cuda_version() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    version = getattr(torch.version, "cuda", None)
    return str(version) if version else None


def cuda_toolkit_download_url(cuda_version: Optional[str] = None) -> str:
    version = cuda_version or torch_cuda_version() or "12.8"
    slug = version.replace('.', '-')
    return f"https://developer.nvidia.com/cuda-{slug}-0-download-archive"


def cuda_toolkit_install_help(*, header: str = "cuda.h", cuda_version: Optional[str] = None) -> str:
    version = cuda_version or torch_cuda_version() or "12.8"
    url = cuda_toolkit_download_url(version)
    return (
        f"Could not find a CUDA toolkit with {header} + lib/x64/cuda.lib.\n"
        f"Install CUDA Toolkit {version} (matches your PyTorch CUDA build) or set CUDA_PATH:\n"
        f"  {url}\n"
        "On Windows, pick the 'exe (network)' installer for your OS version.\n"
        "After install, open a new terminal so CUDA_PATH is visible.\n"
        "Pass --no-gl-present to use the OpenCV window instead."
    )


def require_cuda_root(*, header: str = "cuda.h") -> Path:
    root = find_cuda_root(header=header)
    if root is not None:
        return root
    raise RuntimeError(cuda_toolkit_install_help(header=header))
