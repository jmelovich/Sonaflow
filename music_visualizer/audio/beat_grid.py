"""Phase-locked beat grid that fills gaps in aubio tempo output.

Experimental opt-in (--beat-grid). Synthetic grid ticks only fire when bass or
onset energy supports a real transient at the predicted beat time, so the grid
does not flash on silence between kicks.
"""

from __future__ import annotations

from collections import deque
from typing import Deque


class BeatGridLock:
    """Steady beat grid synced to aubio detections."""

    def __init__(
        self,
        hop_sec: float,
        *,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
        sync_tol_frac: float = 0.18,
        min_aubio_beats: int = 4,
        energy_gate: float = 0.28,
    ) -> None:
        self._hop_sec = max(hop_sec, 1e-6)
        self._min_bpm = float(min_bpm)
        self._max_bpm = float(max_bpm)
        self._sync_tol_frac = float(sync_tol_frac)
        self._min_aubio_beats = int(min_aubio_beats)
        self._energy_gate = float(energy_gate)

        self._bpm = 0.0
        self._next_beat = -1.0
        self._last_aubio_beat = -1.0
        self._aubio_beat_count = 0
        self._intervals: Deque[float] = deque(maxlen=8)

    @property
    def bpm(self) -> float:
        return self._bpm

    def _period(self) -> float:
        if self._bpm <= 0:
            return 0.0
        return 60.0 / self._bpm

    def _update_bpm(self, bpm: float, smoothing: float) -> None:
        if bpm <= 0:
            return
        bpm = max(self._min_bpm, min(self._max_bpm, bpm))
        if self._bpm <= 0:
            self._bpm = bpm
        else:
            s = max(0.0, min(1.0, smoothing))
            self._bpm = (1.0 - s) * self._bpm + s * bpm

    def _apply_interval_bpm(self, interval: float) -> None:
        if interval <= 0:
            return
        bpm = 60.0 / interval
        if self._min_bpm <= bpm <= self._max_bpm:
            self._update_bpm(bpm, smoothing=0.25)

    def on_aubio_beat(self, timestamp: float, bpm: float) -> None:
        """Resync the grid when aubio reports a beat."""
        if bpm > 0:
            self._update_bpm(bpm, smoothing=0.2)

        if self._last_aubio_beat >= 0:
            self._apply_interval_bpm(timestamp - self._last_aubio_beat)

        self._last_aubio_beat = timestamp
        self._aubio_beat_count += 1

        period = self._period()
        if period <= 0:
            return

        if self._next_beat < 0:
            self._next_beat = timestamp + period
            return

        expected = self._next_beat - period
        error = timestamp - expected
        if abs(error) <= period * self._sync_tol_frac:
            self._next_beat = timestamp + period
        else:
            self._next_beat = timestamp + period

    def grid_beat(self, timestamp: float, bass: float, onset: float) -> bool:
        """Return True when a grid beat should fire and energy supports it."""
        if self._aubio_beat_count < self._min_aubio_beats:
            return False
        period = self._period()
        if period <= 0 or self._next_beat < 0:
            return False

        tol = self._hop_sec * 0.5
        if timestamp + tol < self._next_beat:
            return False

        if max(bass, onset) < self._energy_gate:
            return False

        if timestamp >= self._next_beat + period:
            skipped = int((timestamp - self._next_beat) / period)
            self._next_beat += skipped * period

        self._next_beat += period
        return True
