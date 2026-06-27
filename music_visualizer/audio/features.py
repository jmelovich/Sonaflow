"""Audio feature container + a small adaptive normalizer.

The analyzer produces one :class:`AudioFeatures` per audio hop. Raw energies
(RMS, FFT band sums) have no fixed scale -- they depend on the track's mastering
level and the band's bin count -- so they are normalized to a stable ``[0, 1]``
by :class:`RollingNormalizer`, which tracks a decaying running maximum. That way
a quiet track still produces full-range motion and a loud one doesn't peg every
parameter at its cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AudioFeatures:
    """One snapshot of the audio, produced every hop (~12ms at 512/44100)."""

    timestamp: float = 0.0       # seconds of audio processed so far
    rms: float = 0.0             # overall energy/volume, normalized [0, 1]
    onset_strength: float = 0.0  # transient/attack strength, normalized [0, 1]
    is_beat: bool = False        # aubio tempo beat marker for this hop
    bpm: float = 0.0             # smoothed tempo estimate
    bass: float = 0.0            # ~20-250 Hz energy, normalized [0, 1]
    mid: float = 0.0             # ~250-4000 Hz energy, normalized [0, 1]
    treble: float = 0.0          # ~4000+ Hz energy, normalized [0, 1]
    spectral_centroid: float = 0.0  # brightness proxy, normalized [0, 1]
    spectral_rolloff: float = 0.0   # roll-off bin / Nyquist, normalized [0, 1]
    spectral_flux: float = 0.0      # spectral flux, normalized [0, 1]
    spectral_spread: float = 0.0    # spectral spread, normalized [0, 1]
    zcr: float = 0.0                # zero-crossing rate, normalized [0, 1]
    pitch: float = 0.0              # pitch mapped ~80-2000 Hz -> [0, 1]
    pitch_confidence: float = 0.0   # aubio pitch confidence [0, 1]

    @classmethod
    def silent(cls) -> "AudioFeatures":
        """A zeroed snapshot, used before any audio has arrived."""
        return cls()


class RollingNormalizer:
    """Maps an unbounded positive signal to ``[0, 1]`` via a decaying peak.

    Keeps a running maximum that decays toward the current value each update, so
    the effective full-scale slowly follows the loudest recent content. ``decay``
    near 1.0 means the peak fades slowly (stable, less reactive); lower values
    let the normalizer re-scale faster to quieter passages.
    """

    def __init__(self, decay: float = 0.999, floor: float = 1e-6):
        self.decay = float(decay)
        self.floor = float(floor)
        self._peak = floor

    def __call__(self, value: float) -> float:
        value = abs(float(value))
        # Track the peak: jump up instantly on a new max, decay down slowly.
        if value > self._peak:
            self._peak = value
        else:
            self._peak = max(self.floor, self._peak * self.decay)
        if self._peak <= self.floor:
            return 0.0
        return max(0.0, min(1.0, value / self._peak))


class FeatureNormalizer:
    """Maps an unbounded positive signal to ``[0, 1]`` in one of two modes.

    The legacy :class:`RollingNormalizer` (``adaptive=False``) tracks a decaying
    running peak. That has two failure modes that make settings track-dependent:
    one loud transient pins the peak high for seconds (later beats read weak),
    and during quiet passages the peak decays to the floor so *any* tiny signal
    normalizes to ~1.0 (the visualizer goes wild on near-silence).

    The default ``adaptive`` mode instead works in dB: it tracks a ceiling
    (instant attack, slow release) and maps a fixed ``range_db`` window below it
    to ``[0, 1]``. A ``gate_db`` floor on the ceiling forces ``0`` during true
    silence. This generalizes across loud and quiet tracks without re-tuning,
    and the dB mapping matches how loudness is perceived.
    """

    def __init__(
        self,
        decay: float = 0.999,
        floor: float = 1e-6,
        *,
        adaptive: bool = True,
        range_db: float = 50.0,
        gate_db: float = -65.0,
    ):
        self.floor = float(floor)
        self._peak = self.floor
        self._ceil_db = -120.0
        self.configure(
            adaptive=adaptive, range_db=range_db, gate_db=gate_db, decay=decay,
        )

    def configure(
        self,
        *,
        adaptive: bool,
        range_db: float,
        gate_db: float,
        decay: float,
    ) -> None:
        """Apply (possibly live-edited) mode/parameters without losing state."""
        self.adaptive = bool(adaptive)
        # A floor on the window width avoids a divide-by-near-zero blow-up.
        self.range_db = max(6.0, float(range_db))
        self.gate_db = float(gate_db)
        self.decay = float(decay)

    def __call__(self, value: float) -> float:
        value = abs(float(value))
        if not self.adaptive:
            if value > self._peak:
                self._peak = value
            else:
                self._peak = max(self.floor, self._peak * self.decay)
            if self._peak <= self.floor:
                return 0.0
            return max(0.0, min(1.0, value / self._peak))

        x_db = 20.0 * math.log10(value) if value > self.floor else -120.0
        if x_db > self._ceil_db:
            self._ceil_db = x_db  # instant attack on a louder peak
        else:
            # Slow release: ease the ceiling down toward the current level so
            # quiet sections eventually re-scale (decay near 1.0 = slow).
            self._ceil_db = self.decay * self._ceil_db + (1.0 - self.decay) * x_db
        if self._ceil_db <= self.gate_db:
            return 0.0  # everything below the gate is treated as silence
        floor_db = self._ceil_db - self.range_db
        if x_db <= floor_db:
            return 0.0
        return max(0.0, min(1.0, (x_db - floor_db) / self.range_db))
