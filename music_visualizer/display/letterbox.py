"""Letterbox a frame into a target viewport (preserve aspect, black bars)."""

from __future__ import annotations

import numpy as np


def letterbox_frame(
    frame: np.ndarray,
    viewport_w: int,
    viewport_h: int,
) -> np.ndarray:
    """Fit ``frame`` into ``viewport_w`` x ``viewport_h`` without stretching."""
    if viewport_w <= 0 or viewport_h <= 0:
        return frame

    fh, fw = frame.shape[:2]
    if fw <= 0 or fh <= 0:
        return frame

    sx = viewport_w / fw
    sy = viewport_h / fh
    scale = min(sx, sy)
    nw = max(1, int(round(fw * scale)))
    nh = max(1, int(round(fh * scale)))

    if nw == fw and nh == fh and nw == viewport_w and nh == viewport_h:
        return frame

    import cv2

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(frame, (nw, nh), interpolation=interp)
    canvas = np.zeros((viewport_h, viewport_w, 3), dtype=np.uint8)
    ox = (viewport_w - nw) // 2
    oy = (viewport_h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = scaled
    return canvas


def visible_image_scale(frame_w: int, frame_h: int, viewport_w: int, viewport_h: int) -> float:
    """Scale factor from frame pixels to visible viewport pixels (>=1 when shrunk)."""
    if viewport_w <= 0 or viewport_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return 1.0
    sx = viewport_w / frame_w
    sy = viewport_h / frame_h
    s = min(sx, sy)
    vw = frame_w * s
    vh = frame_h * s
    return max(1.0, frame_w / vw, frame_h / vh)
