"""Recenters lit feedback content when zoom drifts into large flat-color regions.

Aggressive per-frame zoom can pull the frame past the image boundary so the
center fills with padding (black, white, or any solid color) while whatever
content remains clings to the edges. This module inspects the feedback tensor
each cycle, detects that pattern (center mostly uniform fill, content at edges),
and adds a smoothed corrective pan so content drifts back toward the middle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from music_visualizer.config import Config

if TYPE_CHECKING:
    from music_visualizer.audio.features import AudioFeatures


def _ema_alpha(dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return 1.0
    return 1.0 - math.exp(-max(0.0, dt) / tau)


def _analysis_hw(frame_h: int, frame_w: int, max_side: int) -> tuple[int, int]:
    """Downsample size preserving aspect ratio (height, width)."""
    if frame_w <= 0 or frame_h <= 0:
        return max_side, max_side
    if frame_w >= frame_h:
        aw = max_side
        ah = max(8, int(round(max_side * frame_h / frame_w)))
    else:
        ah = max_side
        aw = max(8, int(round(max_side * frame_w / frame_h)))
    return ah, aw


def _weighted_centroid_norm(
    weights: torch.Tensor,
) -> tuple[float, float, float]:
    """Return normalized (cx, cy) in [0, 1] and total mass."""
    total = weights.sum()
    if total <= 0:
        return 0.5, 0.5, 0.0
    h, w = weights.shape
    gy = torch.arange(h, device=weights.device, dtype=weights.dtype) + 0.5
    gx = torch.arange(w, device=weights.device, dtype=weights.dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    cx = (weights * grid_x).sum() / total / w
    cy = (weights * grid_y).sum() / total / h
    return cx.item(), cy.item(), total.item()


def _axis_offset_from_mass(
    weights: torch.Tensor,
    axis: int,
    aspect: float,
) -> float:
    """Offset along one axis in [-0.5, 0.5]; positive = content past center."""
    total = weights.sum()
    if total <= 0:
        return 0.0

    cx, cy, _ = _weighted_centroid_norm(weights)
    centroid = cx if axis == 0 else cy
    offset = centroid - 0.5

    # On wide/tall frames, equal edge blobs cancel in the centroid — use band mass.
    is_wide = axis == 0 and aspect >= 1.25
    is_tall = axis == 1 and aspect <= 0.80
    if not is_wide and not is_tall:
        return offset

    size = weights.shape[1] if axis == 0 else weights.shape[0]
    band = max(1, int(size * 0.15))
    if axis == 0:
        low = weights[:, :band].sum()
        high = weights[:, -band:].sum()
    else:
        low = weights[:band, :].sum()
        high = weights[-band:, :].sum()

    side_total = low + high
    if side_total <= 0:
        return offset

    imb = ((high - low) / side_total).item()  # +1 => heavy on far side
    mass_offset = 0.5 * imb  # maps to [-0.5, 0.5]

    # Centroid near middle but both edges lit — trust band imbalance.
    if abs(offset) < 0.12 and side_total >= 0.08 * total:
        return mass_offset
    if side_total >= 0.05 * total:
        return 0.55 * offset + 0.45 * mass_offset
    return offset


class ContentCentering:
    """Stateful corrective pan derived from feedback-frame flat-fill analysis."""

    def __init__(self, config: Config):
        self.enabled = config.content_centering
        # Min per-channel deviation from the center fill color to count as content.
        self.content_threshold = float(config.center_black_threshold)
        self.center_region_frac = float(config.center_region_frac)
        self.center_fill_trigger = float(config.center_black_trigger)
        self.pan_gain = float(config.center_pan_gain)
        self.pan_max = float(config.center_pan_max)
        self.smooth_tau = float(config.center_smooth_tau)
        self.min_lit_frac = float(config.center_min_lit_frac)
        self.analysis_size = int(config.center_analysis_size)
        self._audio_w_bass = float(config.center_audio_bass_w)
        self._audio_w_mid = float(config.center_audio_mid_w)
        self._audio_w_treb = float(config.center_audio_treb_w)
        self._audio_min_scale = float(config.center_audio_min_scale)

        self._corr_x = 0.0
        self._corr_y = 0.0
        self.active = False
        self.audio_strength = 1.0

    def sync_from_config(self, config: Config) -> None:
        """Apply live UI edits to centering parameters."""
        self.enabled = config.content_centering
        self.content_threshold = float(config.center_black_threshold)
        self.center_region_frac = float(config.center_region_frac)
        self.center_fill_trigger = float(config.center_black_trigger)
        self.pan_gain = float(config.center_pan_gain)
        self.pan_max = float(config.center_pan_max)
        self.smooth_tau = float(config.center_smooth_tau)
        self.min_lit_frac = float(config.center_min_lit_frac)
        self.analysis_size = int(config.center_analysis_size)
        self._audio_w_bass = float(config.center_audio_bass_w)
        self._audio_w_mid = float(config.center_audio_mid_w)
        self._audio_w_treb = float(config.center_audio_treb_w)
        self._audio_min_scale = float(config.center_audio_min_scale)

    def update(
        self,
        frame: torch.Tensor,
        dt: float,
        features: Optional["AudioFeatures"] = None,
    ) -> tuple[float, float]:
        """Return additive (pan_x, pan_y) correction for this frame."""
        if not self.enabled:
            self.active = False
            self._corr_x = self._corr_y = 0.0
            self.audio_strength = 1.0
            return 0.0, 0.0

        self.audio_strength = self._audio_strength(features)
        target_x, target_y, active = self._measure(frame, self.audio_strength)
        self.active = active
        alpha = _ema_alpha(dt, self.smooth_tau)
        if active:
            self._corr_x += alpha * (target_x - self._corr_x)
            self._corr_y += alpha * (target_y - self._corr_y)
        else:
            # Ease correction back to zero when the frame looks healthy again.
            self._corr_x *= 1.0 - alpha
            self._corr_y *= 1.0 - alpha
            if abs(self._corr_x) < 1e-5:
                self._corr_x = 0.0
            if abs(self._corr_y) < 1e-5:
                self._corr_y = 0.0
        return self._corr_x, self._corr_y

    def _audio_strength(self, features: Optional["AudioFeatures"]) -> float:
        """Scale corrective pan by band energy: louder track -> stronger centering."""
        if features is None:
            return 1.0
        energy = (
            features.bass * self._audio_w_bass
            + features.mid * self._audio_w_mid
            + features.treble * self._audio_w_treb
        )
        energy = max(0.0, min(1.0, energy))
        return self._audio_min_scale + energy * (1.0 - self._audio_min_scale)

    @torch.no_grad()
    def _measure(
        self, frame: torch.Tensor, strength: float
    ) -> tuple[float, float, bool]:
        """Analyze downsampled RGB; return target pan correction + active flag."""
        x = frame.unsqueeze(0) if frame.ndim == 3 else frame
        x = x[0]  # CHW on device
        _, frame_h, frame_w = x.shape

        ah, aw = _analysis_hw(frame_h, frame_w, self.analysis_size)
        small = F.interpolate(
            x.unsqueeze(0),
            size=(ah, aw),
            mode="area",
        ).squeeze(0)

        h, w = small.shape[1], small.shape[2]
        ch = max(1, int(h * self.center_region_frac))
        cw = max(1, int(w * self.center_region_frac))
        y0, x0 = (h - ch) // 2, (w - cw) // 2

        # Estimate the flat fill color from the center region (black, white, etc.).
        center_rgb = small[:, y0 : y0 + ch, x0 : x0 + cw]
        fill_color = center_rgb.mean(dim=(1, 2))

        # Per-pixel max channel deviation from the fill color.
        dev = (small - fill_color[:, None, None]).abs().max(dim=0).values
        content = dev > self.content_threshold
        content_frac = content.float().mean().item()
        if content_frac < self.min_lit_frac:
            return 0.0, 0.0, False

        center = content[y0 : y0 + ch, x0 : x0 + cw]
        center_fill_frac = 1.0 - center.float().mean().item()

        # Also require content pixels outside the center (mass pushed to edges).
        margin = content.clone()
        margin[y0 : y0 + ch, x0 : x0 + cw] = False
        edge_content = margin.any().item()

        active = center_fill_frac >= self.center_fill_trigger and edge_content
        if not active:
            return 0.0, 0.0, False

        weights = (dev - self.content_threshold).clamp(min=0.0)
        if weights.sum() <= 0:
            return 0.0, 0.0, False

        aspect = w / max(h, 1)
        off_x = _axis_offset_from_mass(weights, axis=0, aspect=aspect)
        off_y = _axis_offset_from_mass(weights, axis=1, aspect=aspect)

        gain = self.pan_gain * strength
        pan_max = self.pan_max * strength
        # Pull content mass toward frame center via opposite pan. Scale up when far off-center.
        scale_x = 1.0 + min(2.0, abs(off_x) * 4.0)
        scale_y = 1.0 + min(2.0, abs(off_y) * 4.0)
        target_x = max(-pan_max, min(pan_max, -gain * off_x * scale_x))
        target_y = max(-pan_max, min(pan_max, -gain * off_y * scale_y))
        return target_x, target_y, True
