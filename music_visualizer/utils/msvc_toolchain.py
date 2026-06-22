"""MSVC toolchain discovery for native bridge DLL builds."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Small bootstrapper (~3 MB). Opens Visual Studio Installer for component pick.
VS_BUILD_TOOLS_BOOTSTRAPPER = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
VS_BUILD_TOOLS_BOOTSTRAPPER_2026 = "https://aka.ms/vs/18/release/vs_BuildTools.exe"
VS_BUILD_TOOLS_PAGE = "https://visualstudio.microsoft.com/visual-cpp-build-tools/"

# Minimal components for cl.exe + Win32 headers (~2 GB, not the 6 GB C++ workload).
MSVC_COMPILER_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
WINDOWS_SDK_COMPONENT = "Microsoft.VisualStudio.Component.Windows11SDK.22621"

_WINDOWS_KITS = Path(r"C:\Program Files (x86)\Windows Kits\10")


@dataclass(frozen=True)
class MsvcEnv:
    """Commands that initialize the x64 MSVC environment before ``cl``."""

    call_line: str


def _vswhere_path() -> Path:
    return (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )


def _vswhere_install_paths(*, requires: str | None = None) -> list[Path]:
    vswhere = _vswhere_path()
    if not vswhere.is_file():
        return []
    cmd = [str(vswhere)]
    if requires:
        cmd.extend(
            [
                "-all",
                "-products",
                "*",
                "-requires",
                requires,
                "-property",
                "installationPath",
            ]
        )
    else:
        cmd.extend(["-latest", "-products", "*", "-property", "installationPath"])
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError:
        return []
    if not out:
        return []
    return [Path(line.strip()) for line in out.splitlines() if line.strip()]


def _latest_msvc_toolset(install: Path) -> Path | None:
    msvc_root = install / "VC" / "Tools" / "MSVC"
    if not msvc_root.is_dir():
        return None

    preferred: list[str] = []
    for name in (
        "Microsoft.VCToolsVersion.v143.default.txt",
        "Microsoft.VCToolsVersion.default.txt",
    ):
        version_file = install / "VC" / "Auxiliary" / "Build" / name
        if version_file.is_file():
            preferred.append(version_file.read_text(encoding="utf-8-sig").strip())

    seen: set[str] = set()
    for version in preferred:
        if not version or version in seen:
            continue
        seen.add(version)
        toolset = msvc_root / version
        if (toolset / "bin" / "HostX64" / "x64" / "cl.exe").is_file():
            return toolset

    for toolset in sorted(msvc_root.iterdir(), key=lambda p: p.name, reverse=True):
        if (toolset / "bin" / "HostX64" / "x64" / "cl.exe").is_file():
            return toolset
    return None


def _latest_windows_sdk_version() -> str | None:
    include_root = _WINDOWS_KITS / "Include"
    if not include_root.is_dir():
        return None
    versions = sorted(
        (p.name for p in include_root.iterdir() if p.is_dir() and p.name[:1].isdigit()),
        reverse=True,
    )
    return versions[0] if versions else None


def _quote(value: str) -> str:
    return value.replace('"', '""')


def _manual_env_call_line(install: Path, toolset: Path, sdk_version: str) -> str:
    cl_dir = toolset / "bin" / "HostX64" / "x64"
    includes = [
        toolset / "include",
        install / "VC" / "Auxiliary" / "VS" / "include",
        _WINDOWS_KITS / "Include" / sdk_version / "ucrt",
        _WINDOWS_KITS / "Include" / sdk_version / "um",
        _WINDOWS_KITS / "Include" / sdk_version / "shared",
    ]
    libs = [
        toolset / "lib" / "x64",
        _WINDOWS_KITS / "Lib" / sdk_version / "ucrt" / "x64",
        _WINDOWS_KITS / "Lib" / sdk_version / "um" / "x64",
    ]
    include_value = ";".join(str(p) for p in includes if p.is_dir())
    lib_value = ";".join(str(p) for p in libs if p.is_dir())
    if not include_value or not lib_value:
        raise RuntimeError(
            "Incomplete Windows SDK install for native bridge builds. "
            "Add a Windows 11 SDK component in Visual Studio Installer."
        )
    return "\r\n".join(
        [
            f'set "PATH={_quote(str(cl_dir))};%PATH%"',
            f'set "INCLUDE={_quote(include_value)}"',
            f'set "LIB={_quote(lib_value)}"',
        ]
    )


def _msvc_env_for_install(install: Path) -> MsvcEnv | None:
    toolset = _latest_msvc_toolset(install)
    if toolset is None:
        return None

    classic = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if classic.is_file():
        return MsvcEnv(f'call "{classic}" >nul')

    sdk_version = _latest_windows_sdk_version()
    if sdk_version is None:
        return None
    return MsvcEnv(_manual_env_call_line(install, toolset, sdk_version))


def find_msvc_env() -> MsvcEnv | None:
    seen: set[str] = set()
    candidates: list[Path] = []

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    for install in _vswhere_install_paths(requires=MSVC_COMPILER_COMPONENT):
        add(install)
    for install in _vswhere_install_paths():
        add(install)

    for install in candidates:
        env = _msvc_env_for_install(install)
        if env is not None:
            return env
    return None


def find_vcvars64() -> Path | None:
    """Backward-compatible alias for callers that only need a batch path."""
    for install in _vswhere_install_paths():
        classic = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if classic.is_file():
            return classic
    return None


def msvc_minimal_install_command() -> str:
    return (
        f"vs_BuildTools.exe --add {MSVC_COMPILER_COMPONENT} "
        f"--add {WINDOWS_SDK_COMPONENT}"
    )


def msvc_install_help(*, gl_presenter: bool = False, upscale: bool = False) -> str:
    lines = [
        "Microsoft Visual C++ build tools not found.",
        "Native bridge DLLs are compiled with MSVC on first use.",
        f"  Bootstrapper (~3 MB): {VS_BUILD_TOOLS_BOOTSTRAPPER}",
        f"  VS 2026 bootstrapper: {VS_BUILD_TOOLS_BOOTSTRAPPER_2026}",
        f"  Docs / alternate link: {VS_BUILD_TOOLS_PAGE}",
        "  Minimal install (~2 GB) — pick only these two components:",
        "    - MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)",
        "      (VS 2026: MSVC Build Tools for x64/x86 (Latest))",
        "    - Windows 11 SDK (10.0.22621.0 or newer)",
        "  Or run the bootstrapper from a shell:",
        f"    {msvc_minimal_install_command()}",
        "  After install, open a new terminal and re-run.",
    ]
    if gl_presenter:
        lines.append(
            "  To skip the GPU presenter for now, pass --no-gl-present."
        )
    if upscale:
        lines.append(
            "  To skip RTX upscaling for now, pass --no-upscale."
        )
    return "\n".join(lines)


def require_msvc_env(*, gl_presenter: bool = False, upscale: bool = False) -> MsvcEnv:
    env = find_msvc_env()
    if env is not None:
        return env
    raise RuntimeError(msvc_install_help(gl_presenter=gl_presenter, upscale=upscale))


def require_vcvars64(*, gl_presenter: bool = False, upscale: bool = False) -> Path:
    classic = find_vcvars64()
    if classic is not None:
        return classic
    raise RuntimeError(msvc_install_help(gl_presenter=gl_presenter, upscale=upscale))
