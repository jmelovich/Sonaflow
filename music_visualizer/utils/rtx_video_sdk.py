"""RTX Video SDK discovery for NVIDIA RTX VSR upscaling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

RTX_VIDEO_SDK_URL = "https://developer.nvidia.com/rtx-video-sdk"


def resolved_sdk_path(config_path: Optional[str] = None) -> Optional[str]:
    return config_path or os.environ.get("NV_RTX_VIDEO_SDK")


def is_valid_rtx_sdk_root(path: Optional[str]) -> bool:
    if not path:
        return False
    root = Path(path)
    return (root / "include" / "nvsdk_ngx.h").is_file()


def rtx_video_sdk_install_help() -> str:
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
    message = rtx_video_sdk_install_help()
    if prefix:
        message = f"{prefix}\n{message}"
    logger.warning(message)
