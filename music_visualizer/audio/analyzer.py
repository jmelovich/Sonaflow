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

from collections import deque
from typing import Optional

import numpy as np

import aubio

from music_visualizer.audio.beat_grid import BeatGridLock
from music_visualizer.audio.features import AudioFeatures, FeatureNormalizer
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
        spectral_fft: int = 0,
        adaptive_norm: bool = True,
        norm_range_db: float = 50.0,
        norm_gate_db: float = -65.0,
        beat_adaptive_gate: bool = True,
        beat_gate_k: float = 0.6,
        beat_octave_correct: bool = True,
        beat_octave_lo: float = 90.0,
    ):
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        self.fft_size = int(fft_size)
        self._hop_sec = self.hop_size / float(self.sample_rate)

        # Spectral analysis FFT, decoupled from the tempo/onset window. A larger
        # window gives finer low-frequency resolution (the tempo tracker keeps
        # the small, timing-friendly fft_size). 0 / too-small => use fft_size.
        spec = int(spectral_fft)
        self.spec_fft = spec if spec >= self.hop_size else self.fft_size

        self._tempo = aubio.tempo("default", self.fft_size, self.hop_size, self.sample_rate)
        self._tempo.set_threshold(float(tempo_threshold))
        self._tempo.set_silence(float(tempo_silence_db))
        self._onset = aubio.onset("default", self.fft_size, self.hop_size, self.sample_rate)
        self._pvoc = aubio.pvoc(self.spec_fft, self.hop_size)
        spec_buf = self.spec_fft
        self._rolloff = aubio.specdesc("rolloff", spec_buf)
        self._specflux = aubio.specdesc("specflux", spec_buf)
        self._spread = aubio.specdesc("spread", spec_buf)
        self._pitch = aubio.pitch("yinfft", self.fft_size, self.hop_size, self.sample_rate)

        # Precompute the bin -> band masks (bin i center freq = i * sr / spec_fft).
        n_bins = self.spec_fft // 2 + 1
        freqs = np.arange(n_bins, dtype=np.float32) * self.sample_rate / self.spec_fft
        nyquist = self.sample_rate / 2.0
        self._bass_mask = (freqs >= _BASS_HZ[0]) & (freqs < _BASS_HZ[1])
        self._mid_mask = (freqs >= _MID_HZ[0]) & (freqs < _MID_HZ[1])
        self._treble_mask = (freqs >= _TREBLE_LO_HZ) & (freqs <= nyquist)
        self._freqs = freqs

        def _mk_norm() -> FeatureNormalizer:
            return FeatureNormalizer(
                decay=norm_decay,
                adaptive=adaptive_norm,
                range_db=norm_range_db,
                gate_db=norm_gate_db,
            )

        self._rms_norm = _mk_norm()
        self._onset_norm = _mk_norm()
        self._bass_norm = _mk_norm()
        self._mid_norm = _mk_norm()
        self._treble_norm = _mk_norm()
        self._flux_norm = _mk_norm()
        self._spread_norm = _mk_norm()
        self._zcr_norm = _mk_norm()
        self._norms = (
            self._rms_norm, self._onset_norm, self._bass_norm, self._mid_norm,
            self._treble_norm, self._flux_norm, self._spread_norm, self._zcr_norm,
        )

        self._bpm_smoothing = float(bpm_smoothing)
        self._n_bins = self.spec_fft // 2 + 1
        self._bpm = 0.0
        self._frames = 0

        # Adaptive beat gating: confirm aubio beats against a rolling onset
        # threshold so "beats" on weak/steady content are rejected.
        self._beat_adaptive_gate = bool(beat_adaptive_gate)
        self._beat_gate_k = float(beat_gate_k)
        self._onset_hist: deque[float] = deque(maxlen=256)  # ~3s @ 512/44100

        # Tempo octave correction: fold BPM into one preferred octave.
        self._beat_octave_correct = bool(beat_octave_correct)
        self._octave_lo = float(beat_octave_lo)
        self._beat_bpm_min = float(beat_bpm_min)
        self._beat_bpm_max = float(beat_bpm_max)

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
        self._beat_bpm_min = float(bpm_min)
        self._beat_bpm_max = float(bpm_max)
        if enabled:
            if self._grid is None:
                self._grid = BeatGridLock(
                    self._hop_sec, min_bpm=bpm_min, max_bpm=bpm_max,
                )
        else:
            self._grid = None

    def set_normalize(
        self, adaptive: bool, range_db: float, gate_db: float, decay: float,
    ) -> None:
        """Live-reconfigure every feature normalizer (no state reset)."""
        for norm in self._norms:
            norm.configure(
                adaptive=adaptive, range_db=range_db, gate_db=gate_db, decay=decay,
            )

    def set_beat_gate(self, enabled: bool, k: float) -> None:
        self._beat_adaptive_gate = bool(enabled)
        self._beat_gate_k = float(k)

    def set_octave_correct(self, enabled: bool, octave_lo: float) -> None:
        self._beat_octave_correct = bool(enabled)
        self._octave_lo = float(octave_lo)

    def sync_from_config(self, config) -> None:
        """Apply live UI edits to aubio tempo/grid/normalize settings.

        Note: ``spectral_fft`` rebuilds aubio objects and is restart-only.
        """
        self.set_tempo_threshold(config.beat_tempo_threshold)
        self.set_silence_db(config.beat_silence_db)
        self.set_bpm_smoothing(config.beat_bpm_smoothing)
        self.set_beat_grid(
            config.beat_grid_lock,
            config.beat_bpm_min,
            config.beat_bpm_max,
        )
        self.set_normalize(
            bool(config.react_adaptive_norm),
            float(config.react_norm_range_db),
            float(config.react_norm_gate_db),
            float(config.react_norm_decay),
        )
        self.set_beat_gate(
            bool(config.beat_adaptive_gate), float(config.beat_gate_k),
        )
        self.set_octave_correct(
            bool(config.beat_octave_correct), float(config.beat_octave_lo),
        )

    def _fold_octave(self, bpm: float) -> float:
        """Fold ``bpm`` into ``[octave_lo, 2*octave_lo)`` (half/double tempo fix)."""
        if bpm <= 0 or self._octave_lo <= 0:
            return bpm
        lo = self._octave_lo
        hi = 2.0 * lo
        # Guard against pathological inputs that could loop forever.
        for _ in range(8):
            if bpm < lo:
                bpm *= 2.0
            elif bpm >= hi:
                bpm *= 0.5
            else:
                break
        return bpm

    def _onset_gate_ok(self, onset_raw: float) -> bool:
        """True if ``onset_raw`` clears the adaptive median+k*MAD threshold."""
        if len(self._onset_hist) < 16:
            return True
        arr = np.fromiter(self._onset_hist, dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-9
        # 1.4826 scales MAD to a std-equivalent for ~normal noise.
        thresh = med + self._beat_gate_k * 1.4826 * mad
        return onset_raw >= thresh

    def process(self, block: np.ndarray) -> AudioFeatures:
        """Analyze one hop of mono float32 samples (length should be hop_size)."""
        samples = np.ascontiguousarray(block, dtype=np.float32)
        if samples.shape[0] != self.hop_size:
            samples = self._fit_hop(samples)

        raw_beat = bool(self._tempo(samples)[0])

        self._onset(samples)
        onset_raw = float(self._onset.get_descriptor())
        self._onset_hist.append(onset_raw)

        is_aubio_beat = raw_beat
        if raw_beat:
            # Tempo BPM tracks aubio's own (ungated) beats so the estimate stays
            # stable even when the flash gate rejects a hit.
            bpm = float(self._tempo.get_bpm())
            if self._beat_octave_correct:
                bpm = self._fold_octave(bpm)
            if bpm > 0:
                self._bpm = (
                    bpm if self._bpm <= 0
                    else (1.0 - self._bpm_smoothing) * self._bpm + self._bpm_smoothing * bpm
                )
            # Adaptive gate: drop beats that land on weak/steady content.
            if self._beat_adaptive_gate and not self._onset_gate_ok(onset_raw):
                is_aubio_beat = False

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
        # Resync the grid on aubio's true beats (ungated) so its phase stays
        # locked even when the flash gate suppresses a weak hit.
        if raw_beat and self._grid is not None:
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
