"""Audio feature container + a small adaptive normalizer.

The analyzer produces one :class:`AudioFeatures` per audio hop. Raw energies
(RMS, FFT band sums) have no fixed scale -- they depend on the track's mastering
level and the band's bin count -- so they are normalized to a stable ``[0, 1]``
by :class:`RollingNormalizer`, which tracks a decaying running maximum. That way
a quiet track still produces full-range motion and a loud one doesn't peg every
parameter at its cap.
"""

from __future__ import annotations

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
