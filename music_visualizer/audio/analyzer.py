"""aubio-based realtime feature extraction.

Wraps aubio's ``tempo`` (beat + BPM), ``onset`` (transient strength) and ``pvoc``
(phase vocoder, for a magnitude spectrum) objects. Each call to :meth:`process`
consumes exactly one hop of mono ``float32`` samples and returns one
:class:`AudioFeatures`.

Band energies (bass / mid / treble) and the spectral centroid are derived from
the pvoc magnitude spectrum and normalized to ``[0, 1]`` by per-feature
:class:`RollingNormalizer` instances, so the mapping downstream sees a stable
range regardless of the track's loudness.

Optional ``beat_grid_lock`` can fill missed beats on a BPM grid; it is off by
default because a drifting BPM estimate causes off-beat flashes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import aubio

from music_visualizer.audio.beat_grid import BeatGridLock
from music_visualizer.audio.features import AudioFeatures, RollingNormalizer
from music_visualizer.utils.logging import get_logger

logger = get_logger(__name__)

# Band edges in Hz. Treble extends up to Nyquist (computed per sample rate).
_BASS_HZ = (20.0, 250.0)
_MID_HZ = (250.0, 4000.0)
_TREBLE_LO_HZ = 4000.0
_PITCH_LO_HZ = 80.0
_PITCH_HI_HZ = 2000.0


class AubioAnalyzer:
    """Stateful per-hop analyzer. Not thread-safe; use from one thread only."""

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_size: int = 512,
        fft_size: int = 1024,
        bpm_smoothing: float = 0.1,
        norm_decay: float = 0.999,
        tempo_threshold: float = 0.3,
        tempo_silence_db: float = -90.0,
        beat_grid_lock: bool = False,
        beat_bpm_min: float = 60.0,
        beat_bpm_max: float = 200.0,
    ):
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        self.fft_size = int(fft_size)
        self._hop_sec = self.hop_size / float(self.sample_rate)

        self._tempo = aubio.tempo("default", self.fft_size, self.hop_size, self.sample_rate)
        self._tempo.set_threshold(float(tempo_threshold))
        self._tempo.set_silence(float(tempo_silence_db))
        self._onset = aubio.onset("default", self.fft_size, self.hop_size, self.sample_rate)
        self._pvoc = aubio.pvoc(self.fft_size, self.hop_size)
        spec_buf = self.fft_size
        self._rolloff = aubio.specdesc("rolloff", spec_buf)
        self._specflux = aubio.specdesc("specflux", spec_buf)
        self._spread = aubio.specdesc("spread", spec_buf)
        self._pitch = aubio.pitch("yinfft", self.fft_size, self.hop_size, self.sample_rate)

        # Precompute the bin -> band masks (bin i center freq = i * sr / fft_size).
        n_bins = self.fft_size // 2 + 1
        freqs = np.arange(n_bins, dtype=np.float32) * self.sample_rate / self.fft_size
        nyquist = self.sample_rate / 2.0
        self._bass_mask = (freqs >= _BASS_HZ[0]) & (freqs < _BASS_HZ[1])
        self._mid_mask = (freqs >= _MID_HZ[0]) & (freqs < _MID_HZ[1])
        self._treble_mask = (freqs >= _TREBLE_LO_HZ) & (freqs <= nyquist)
        self._freqs = freqs

        self._rms_norm = RollingNormalizer(decay=norm_decay)
        self._onset_norm = RollingNormalizer(decay=norm_decay)
        self._bass_norm = RollingNormalizer(decay=norm_decay)
        self._mid_norm = RollingNormalizer(decay=norm_decay)
        self._treble_norm = RollingNormalizer(decay=norm_decay)
        self._flux_norm = RollingNormalizer(decay=norm_decay)
        self._spread_norm = RollingNormalizer(decay=norm_decay)
        self._zcr_norm = RollingNormalizer(decay=norm_decay)

        self._bpm_smoothing = float(bpm_smoothing)
        self._n_bins = self.fft_size // 2 + 1
        self._bpm = 0.0
        self._frames = 0
        self._grid: Optional[BeatGridLock] = None
        if beat_grid_lock:
            self._grid = BeatGridLock(
                self._hop_sec,
                min_bpm=beat_bpm_min,
                max_bpm=beat_bpm_max,
            )

    def set_tempo_threshold(self, threshold: float) -> None:
        self._tempo.set_threshold(float(threshold))

    def set_silence_db(self, silence_db: float) -> None:
        self._tempo.set_silence(float(silence_db))

    def set_bpm_smoothing(self, rate: float) -> None:
        self._bpm_smoothing = float(rate)

    def set_beat_grid(
        self,
        enabled: bool,
        bpm_min: float = 60.0,
        bpm_max: float = 200.0,
    ) -> None:
        if enabled:
            if self._grid is None:
                self._grid = BeatGridLock(
                    self._hop_sec, min_bpm=bpm_min, max_bpm=bpm_max,
                )
        else:
            self._grid = None

    def sync_from_config(self, config) -> None:
        """Apply live UI edits to aubio tempo/grid settings."""
        self.set_tempo_threshold(config.beat_tempo_threshold)
        self.set_silence_db(config.beat_silence_db)
        self.set_bpm_smoothing(config.beat_bpm_smoothing)
        self.set_beat_grid(
            config.beat_grid_lock,
            config.beat_bpm_min,
            config.beat_bpm_max,
        )

    def process(self, block: np.ndarray) -> AudioFeatures:
        """Analyze one hop of mono float32 samples (length should be hop_size)."""
        samples = np.ascontiguousarray(block, dtype=np.float32)
        if samples.shape[0] != self.hop_size:
            samples = self._fit_hop(samples)

        is_aubio_beat = bool(self._tempo(samples)[0])
        if is_aubio_beat:
            bpm = float(self._tempo.get_bpm())
            if bpm > 0:
                self._bpm = (
                    bpm if self._bpm <= 0
                    else (1.0 - self._bpm_smoothing) * self._bpm + self._bpm_smoothing * bpm
                )

        self._onset(samples)
        onset_raw = float(self._onset.get_descriptor())

        fftgrain = self._pvoc(samples)
        mag = np.asarray(fftgrain.norm, dtype=np.float32)
        bass_raw = float(mag[self._bass_mask].sum())
        mid_raw = float(mag[self._mid_mask].sum())
        treble_raw = float(mag[self._treble_mask].sum())

        total = float(mag.sum())
        centroid_hz = float((self._freqs * mag).sum() / total) if total > 1e-9 else 0.0
        nyquist = self.sample_rate / 2.0
        centroid = max(0.0, min(1.0, centroid_hz / nyquist))

        rolloff_bin = float(self._rolloff(fftgrain)[0])
        rolloff = max(0.0, min(1.0, rolloff_bin / max(1.0, self._n_bins - 1)))
        flux_raw = float(self._specflux(fftgrain)[0])
        spread_raw = float(self._spread(fftgrain)[0])
        zcr_raw = float(aubio.zero_crossing_rate(samples))

        pitch_hz = float(self._pitch(samples)[0])
        pitch_conf = float(self._pitch.get_confidence())
        if pitch_hz > 0:
            pitch = max(
                0.0,
                min(
                    1.0,
                    (pitch_hz - _PITCH_LO_HZ) / max(1.0, _PITCH_HI_HZ - _PITCH_LO_HZ),
                ),
            )
        else:
            pitch = 0.0
        pitch_conf = max(0.0, min(1.0, pitch_conf))

        rms_raw = float(np.sqrt(np.mean(samples * samples)))

        bass = self._bass_norm(bass_raw)
        onset_strength = self._onset_norm(onset_raw)

        timestamp = (self._frames + self.hop_size) / float(self.sample_rate)
        if is_aubio_beat and self._grid is not None:
            self._grid.on_aubio_beat(timestamp, self._bpm)

        is_beat = is_aubio_beat
        if not is_beat and self._grid is not None:
            is_beat = self._grid.grid_beat(timestamp, bass, onset_strength)

        self._frames += self.hop_size
        return AudioFeatures(
            timestamp=self._frames / float(self.sample_rate),
            rms=self._rms_norm(rms_raw),
            onset_strength=onset_strength,
            is_beat=is_beat,
            bpm=self._bpm,
            bass=bass,
            mid=self._mid_norm(mid_raw),
            treble=self._treble_norm(treble_raw),
            spectral_centroid=centroid,
            spectral_rolloff=rolloff,
            spectral_flux=self._flux_norm(flux_raw),
            spectral_spread=self._spread_norm(spread_raw),
            zcr=self._zcr_norm(zcr_raw),
            pitch=pitch,
            pitch_confidence=pitch_conf,
        )

    def _fit_hop(self, samples: np.ndarray) -> np.ndarray:
        """Pad/truncate a short or long block to exactly ``hop_size`` samples."""
        if samples.shape[0] > self.hop_size:
            return samples[: self.hop_size]
        out = np.zeros(self.hop_size, dtype=np.float32)
        out[: samples.shape[0]] = samples
        return out
