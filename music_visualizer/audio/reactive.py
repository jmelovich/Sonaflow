"""Maps :class:`AudioFeatures` onto Deforum feedback-motion parameters.

Different audio aspects drive different parameters so the motion reads as
"musical" rather than one blob of energy:

* beats         -> a zoom *punch* scaled by bass energy at the hit (see
                   ``react_beat_bass_scale``) on top of a bass-driven base
* loud beats    -> extra rotation in the current spin direction when mid+treble are hot
* onset/attacks -> momentary ``noise_drift`` (more re-imagining on transients)
* bpm           -> final scale on total noise in [0.8, 1.2]: slow dips, fast lifts
* treble        -> rotation speed (bright = spin faster)
* mid energy    -> a gentle circular pan/drift

Raw features are already normalized to ``[0, 1]`` by the analyzer; here we add
*temporal* shaping: time-constant EMA smoothing (so per-hop jitter doesn't make
the image vibrate) and a beat-triggered attack/decay envelope (so beats produce
a visible pulse instead of a single-frame jump). All shaping is ``dt``-based, so
it behaves the same regardless of the generation frame rate.
"""

from __future__ import annotations

import math
from typing import Optional

from music_visualizer.audio.features import AudioFeatures
from music_visualizer.config import Config


def _ema_alpha(dt: float, tau: float) -> float:
    """Time-constant EMA coefficient: fraction to move toward target over ``dt``."""
    if tau <= 1e-6:
        return 1.0
    return 1.0 - math.exp(-max(0.0, dt) / tau)


def bpm_noise_multiplier(
    bpm: float,
    pivot: float = 80.0,
    mult_min: float = 0.8,
    mult_max: float = 1.2,
    high_bpm: float = 200.0,
) -> float:
    """Map tempo to a direct noise scale in ``[mult_min, mult_max]``.

    Applied last as ``noise * scale``. Below ``pivot`` BPM scales down toward
    ``mult_min``; at ``pivot`` the scale is ``1.0``; at ``high_bpm`` and above
    it reaches ``mult_max``. Returns ``1.0`` when BPM is unknown.
    """
    if bpm <= 0:
        return 1.0
    if bpm < pivot:
        t = bpm / pivot
        return mult_min + t * (1.0 - mult_min)
    if bpm >= high_bpm:
        return mult_max
    t = (bpm - pivot) / (high_bpm - pivot)
    return 1.0 + t * (mult_max - 1.0)


class ReactiveMapper:
    """Stateful mapper. Call :meth:`update` once per generated frame."""

    def __init__(self, config: Config):
        self.config = config
        # Resting values the reactive terms modulate around (post-init resolved).
        self._base_zoom = float(config.feedback_zoom or 1.0)
        self._base_noise = float(config.feedback_noise_drift or 0.0)
        self._base_pan_x = float(config.feedback_pan_x)
        self._base_pan_y = float(config.feedback_pan_y)
        self._base_rotate = float(config.feedback_rotate)

        self._smooth_tau = float(config.react_smoothing_tau)
        self._beat_tau = float(config.react_beat_decay_tau)
        self._orbit_rate = float(config.react_pan_orbit_rate)

        # Smoothed feature state.
        self._bass = 0.0
        self._mid = 0.0
        self._treble = 0.0
        self._onset = 0.0
        # Beat impulse envelope [0, 1] and pan orbit phase.
        self._beat_env = 0.0
        self._beat_bass = 1.0  # bass level captured at the last beat hit
        self._beat_rotate_env = 0.0
        self._beat_rotate_dir = 1.0  # +1 clockwise when spin direction is ambiguous
        self._phase = 0.0
        self._last_beat_ts = -1.0
        self._bpm_noise_mult = 1.0

    def set_base(self, **kwargs) -> None:
        """Update baseline motion values (reactive mapping modulates around these)."""
        if "zoom" in kwargs and kwargs["zoom"] is not None:
            self._base_zoom = float(kwargs["zoom"])
        if "pan_x" in kwargs and kwargs["pan_x"] is not None:
            self._base_pan_x = float(kwargs["pan_x"])
        if "pan_y" in kwargs and kwargs["pan_y"] is not None:
            self._base_pan_y = float(kwargs["pan_y"])
        if "rotate" in kwargs and kwargs["rotate"] is not None:
            self._base_rotate = float(kwargs["rotate"])
        if "noise_drift" in kwargs and kwargs["noise_drift"] is not None:
            self._base_noise = float(kwargs["noise_drift"])

    def sync_from_config(self) -> None:
        """Refresh cached time constants from config after a live UI edit."""
        self._smooth_tau = float(self.config.react_smoothing_tau)
        self._beat_tau = float(self.config.react_beat_decay_tau)
        self._orbit_rate = float(self.config.react_pan_orbit_rate)

    def update(self, features: Optional[AudioFeatures], dt: float) -> dict:
        """Return motion params for :meth:`StreamPipeline.set_transform`."""
        cfg = self.config
        a = _ema_alpha(dt, self._smooth_tau)

        if features is not None:
            self._bass += a * (features.bass - self._bass)
            self._mid += a * (features.mid - self._mid)
            self._treble += a * (features.treble - self._treble)
            self._onset += a * (features.onset_strength - self._onset)
            target_bpm_mult = bpm_noise_multiplier(
                features.bpm,
                pivot=cfg.react_bpm_noise_pivot,
                mult_min=cfg.react_bpm_noise_mult_min,
                mult_max=cfg.react_bpm_noise_mult_max,
                high_bpm=cfg.react_bpm_noise_high_bpm,
            )
            self._bpm_noise_mult += a * (target_bpm_mult - self._bpm_noise_mult)
            # Trigger the beat envelope once per distinct beat timestamp.
            if features.is_beat and features.timestamp != self._last_beat_ts:
                self._beat_env = 1.0
                self._beat_bass = max(0.0, min(1.0, float(features.bass)))
                self._last_beat_ts = features.timestamp
                loud = (
                    features.rms >= cfg.react_beat_loud_rms
                    and features.mid >= cfg.react_beat_loud_mid
                    and features.treble >= cfg.react_beat_loud_treb
                )
                if loud:
                    trend = self._base_rotate + self._treble * cfg.react_rotate_gain
                    self._beat_rotate_dir = (
                        math.copysign(1.0, trend) if abs(trend) > 1e-6 else 1.0
                    )
                    self._beat_rotate_env = 1.0

        # Decay beat impulses toward zero with their own time constant.
        decay = math.exp(-max(0.0, dt) / self._beat_tau) if self._beat_tau > 1e-6 else 0.0
        self._beat_env *= decay
        self._beat_rotate_env *= decay

        # Advance the pan orbit; radius scales with mid energy.
        self._phase += dt * self._orbit_rate

        s = float(cfg.react_beat_bass_scale)
        beat_strength = (1.0 - s) + s * self._beat_bass
        zoom = (
            self._base_zoom
            + self._bass * cfg.react_zoom_gain
            + self._beat_env * cfg.react_beat_zoom * beat_strength
        )
        noise = self._base_noise + self._onset * cfg.react_noise_gain
        noise *= self._bpm_noise_mult
        rotate = self._base_rotate + self._treble * cfg.react_rotate_gain
        rotate += (
            self._beat_rotate_dir * self._beat_rotate_env * cfg.react_beat_rotate
        )
        pan_r = self._mid * cfg.react_pan_gain
        pan_x = self._base_pan_x + pan_r * math.cos(self._phase)
        pan_y = self._base_pan_y + pan_r * math.sin(self._phase)

        zoom_min = self._base_zoom
        zoom_max = float(cfg.react_zoom_max)
        rot_max = float(cfg.react_rotate_max)

        return {
            "zoom": max(zoom_min, min(zoom_max, zoom)),
            "pan_x": pan_x,
            "pan_y": pan_y,
            "rotate": max(-rot_max, min(rot_max, rotate)),
            "noise_drift": max(0.0, min(1.0, noise)),
        }
