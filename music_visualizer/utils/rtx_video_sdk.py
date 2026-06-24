"""RTX Video SDK discovery for NVIDIA RTX VSR upscaling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from music_visualizer.utils.paths import repo_root

RTX_VIDEO_SDK_URL = "https://developer.nvidia.com/rtx-video-sdk"

_VSR_REL = Path("bin") / "Windows" / "x64" / "rel" / "nvngx_vsr.dll"
_VSR_DEV = Path("bin") / "Windows" / "x64" / "dev" / "nvngx_vsr.dll"


def _has_vsr_feature_dll(root: Path) -> bool:
    return (root / _VSR_REL).is_file() or (root / _VSR_DEV).is_file()


def bundled_rtx_sdk_root() -> Optional[Path]:
    """Return the bundled RTX Video SDK root in a packaged install, if present."""
    if not os.environ.get("SONAFLOW_DATA_DIR"):
        return None
    candidate = repo_root().parent / "thirdparty" / "rtx-video-sdk"
    if is_valid_rtx_sdk_root(str(candidate)):
        return candidate
    return None


def is_valid_rtx_sdk_root(path: Optional[str]) -> bool:
    """True for a full dev SDK tree or a bundled runtime-only layout."""
    if not path:
        return False
    root = Path(path)
    if (root / "include" / "nvsdk_ngx.h").is_file():
        return True
    return _has_vsr_feature_dll(root)


def resolve_rtx_sdk_path(config_path: Optional[str] = None) -> Optional[str]:
    """Resolve the RTX Video SDK root: config override, bundled, then env."""
    if config_path:
        return config_path
    bundled = bundled_rtx_sdk_root()
    if bundled is not None:
        return str(bundled)
    return os.environ.get("NV_RTX_VIDEO_SDK")


def resolved_sdk_path(config_path: Optional[str] = None) -> Optional[str]:
    return resolve_rtx_sdk_path(config_path)


def rtx_video_sdk_install_help(*, packaged: bool = False) -> str:
    if packaged:
        return (
            "RTX VSR upscaling is unavailable — bundled SDK files are missing.\n"
            "  Reinstall Sonaflow, or pass --no-upscale to run at native generation resolution."
        )
    return (
        "RTX Video SDK not found — upscaling requires a manual install.\n"
        f"  Download: {RTX_VIDEO_SDK_URL}\n"
        "  Set NV_RTX_VIDEO_SDK to the SDK root (directory with bin/, include/, lib/).\n"
        "  Or pass --no-upscale to run at native generation resolution."
    )


def log_rtx_sdk_missing_help(
    sdk_path: Optional[str],
    logger,
    *,
    prefix: str = "",
) -> None:
    if is_valid_rtx_sdk_root(sdk_path):
        return
    packaged = os.environ.get("SONAFLOW_DATA_DIR") is not None
    message = rtx_video_sdk_install_help(packaged=packaged)
    if prefix:
        message = f"{prefix}\n{message}"
    logger.warning(message)
