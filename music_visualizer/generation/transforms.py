"""Geometric feedback transforms (Deforum-style motion).

A pure img2img self-feedback loop converges to a near-fixed point: each frame is
almost identical to the last, so the visuals look static. Applying a small
geometric transform (zoom / pan / rotate) to the feedback frame *before* it is
fed back gives the model something slightly new to re-imagine every cycle, which
produces continuous motion -- the same trick Deforum uses for animated diffusion.

Transforms run on the GPU via ``grid_sample`` so they stay on-device (the feedback
frame is already an NCHW tensor) and add negligible latency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class FeedbackTransform:
    """Per-frame affine transform applied to the feedback image.

    All amounts are *per frame*. Identity (no motion) is ``zoom=1.0`` with zero
    pan/rotate. Values compose into a single affine warp.

    Parameters
    ----------
    zoom : float
        Per-frame scale. ``>1`` zooms in (dolly forward), ``<1`` zooms out.
        ``1.02`` is a gentle, hypnotic forward push.
    pan_x, pan_y : float
        Per-frame translation as a fraction of width/height. ``pan_x=0.01``
        drifts right by 1% each frame; ``pan_y`` positive drifts down.
    rotate : float
        Per-frame rotation in degrees (positive = clockwise).
    padding_mode : {"reflection", "border", "zeros"}
        How to fill pixels pulled in from outside the frame. ``reflection``
        avoids hard black edges accumulating when zooming/panning.
    """

    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    rotate: float = 0.0
    padding_mode: Literal["reflection", "border", "zeros"] = "reflection"

    def is_identity(self) -> bool:
        return (
            abs(self.zoom - 1.0) < 1e-6
            and abs(self.pan_x) < 1e-6
            and abs(self.pan_y) < 1e-6
            and abs(self.rotate) < 1e-6
        )

    def _theta(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Build the 2x3 affine matrix for ``affine_grid``.

        ``affine_grid`` maps *output* coordinates to *input* sampling coordinates,
        so to zoom the image IN we must sample a SMALLER region -> scale by 1/zoom.
        """
        s = 1.0 / max(self.zoom, 1e-6)
        a = math.radians(self.rotate)
        cos_a, sin_a = math.cos(a), math.sin(a)

        # Translation in normalized [-1, 1] coords (full width spans 2.0).
        tx = -2.0 * self.pan_x
        ty = -2.0 * self.pan_y

        theta = torch.tensor(
            [
                [s * cos_a, -s * sin_a, tx],
                [s * sin_a, s * cos_a, ty],
            ],
            dtype=dtype,
            device=device,
        )
        return theta.unsqueeze(0)  # (1, 2, 3)

    def apply(self, frame: torch.Tensor) -> torch.Tensor:
        """Warp an NCHW (or CHW) feedback tensor. Returns the same layout."""
        if self.is_identity():
            return frame

        squeeze = False
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
            squeeze = True

        n, _, h, w = frame.shape
        theta = self._theta(frame.dtype, frame.device).expand(n, -1, -1)
        grid = F.affine_grid(theta, frame.shape, align_corners=False)
        out = F.grid_sample(
            frame,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        return out.squeeze(0) if squeeze else out

    @classmethod
    def from_config(cls, config) -> "FeedbackTransform":
        return cls(
            zoom=config.feedback_zoom,
            pan_x=config.feedback_pan_x,
            pan_y=config.feedback_pan_y,
            rotate=config.feedback_rotate,
            padding_mode=config.feedback_padding,
        )
