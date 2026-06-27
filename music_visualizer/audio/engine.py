"""Audio sources + the analyzer thread.

Two source kinds feed a common queue of mono ``float32`` hop blocks:

* :class:`FilePlaybackEngine` -- decode a file, play it through the default
  output device, and tee a mono copy of each played block to the analyzer.
* :class:`LiveInputEngine` -- capture from a microphone/line-in (``mic``) or
  from whatever is currently playing on the system via WASAPI loopback
  (``loopback``, Windows only).

:class:`AnalyzerThread` drains the queue, runs :class:`AubioAnalyzer`, and
publishes the latest :class:`AudioFeatures` into an
:class:`~music_visualizer.utils.shared_state.AudioState`. Keeping aubio off the
realtime audio callback (which must never block or allocate heavily) avoids
glitches/underruns in playback and capture.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from music_visualizer.audio.analyzer import AubioAnalyzer
from music_visualizer.audio.features import AudioFeatures
from music_visualizer.config import Config
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.shared_state import AudioRing, AudioState

logger = get_logger(__name__)


def _to_mono(block: np.ndarray) -> np.ndarray:
    """Downmix an (n, channels) or (n,) block to a contiguous mono float32 array."""
    if block.ndim == 2 and block.shape[1] > 1:
        block = block.mean(axis=1)
    return np.ascontiguousarray(block.reshape(-1), dtype=np.float32)


def _enqueue_drop_oldest(q: "queue.Queue", item: np.ndarray) -> None:
    """Non-blocking put that drops the oldest block when the queue is full.

    Audio callbacks must never block, and the analyzer only cares about keeping
    up with the present -- so an overfull queue means we're behind and should
    discard stale audio rather than stall the device.
    """
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class FilePlaybackEngine:
    """Plays an audio file and tees each block to the analysis queue."""

    def __init__(self, config: Config, block_queue: "queue.Queue"):
        self.config = config
        self._queue = block_queue
        self.finished = threading.Event()

        data, file_sr = sf.read(config.audio_file, dtype="float32", always_2d=True)
        self._data = data
        self.sample_rate = int(file_sr)
        self.channels = data.shape[1]
        self.hop_size = config.hop_size
        # Decoupled from hop_size: a large device buffer survives GPU/GIL stalls
        # that a 512-sample buffer would underrun into gibberish.
        self.blocksize = max(config.hop_size, int(config.audio_blocksize))
        self._pos = 0
        self._stream: Optional[sd.OutputStream] = None

        logger.info(
            "Audio file: %s (%.1fs, %d ch, %d Hz)",
            config.audio_file,
            data.shape[0] / float(self.sample_rate),
            self.channels,
            self.sample_rate,
        )

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: D401
        if status:
            logger.debug("Playback status: %s", status)
        start = self._pos
        end = start + frames
        chunk = self._data[start:end]
        n = chunk.shape[0]
        outdata[:n] = chunk
        if n < frames:
            outdata[n:] = 0.0
        self._pos = end
        _enqueue_drop_oldest(self._queue, _to_mono(chunk))
        if n < frames:
            self.finished.set()
            raise sd.CallbackStop

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            device=self.config.audio_device,
            callback=self._callback,
            finished_callback=self.finished.set,
            latency="high",
        )
        self._stream.start()
        logger.info(
            "File playback started (device buffer %d samples, %.0f ms).",
            self.blocksize, 1000.0 * self.blocksize / self.sample_rate,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class LiveInputEngine:
    """Captures live audio (microphone/line-in or WASAPI loopback)."""

    def __init__(self, config: Config, block_queue: "queue.Queue"):
        self.config = config
        self._queue = block_queue
        self.finished = threading.Event()
        self.hop_size = config.hop_size
        self.blocksize = max(config.hop_size, int(config.audio_blocksize))
        self.sample_rate = config.sample_rate
        self._stream: Optional[sd.InputStream] = None

        self._device, self.channels, extra = self._resolve_device()
        self._extra_settings = extra

    def _resolve_device(self):
        """Pick the capture device + WASAPI loopback settings for the source mode."""
        if self.config.audio_source == "loopback":
            return self._resolve_loopback()

        # mic / line-in
        device = self.config.audio_device
        if device is None:
            device = sd.default.device[0]
        info = sd.query_devices(device)
        channels = max(1, min(2, int(info["max_input_channels"])))
        logger.info("Input capture: '%s' (%d ch)", info["name"], channels)
        return device, channels, None

    def _resolve_loopback(self):
        """Resolve a system-audio capture path.

        PortAudio's WASAPI loopback is only exposed by some sounddevice builds via
        ``WasapiSettings(loopback=True)``. When available we capture the default
        output device directly. Otherwise we fall back to a "Stereo Mix"-style
        input device (a standard Windows system-audio capture that the user can
        enable in Sound settings). If neither exists we raise an actionable error.
        """
        if self.config.audio_device is not None:
            info = sd.query_devices(self.config.audio_device)
            channels = max(1, min(2, int(info["max_input_channels"]) or 2))
            logger.info("Loopback (explicit device): '%s'", info["name"])
            return self.config.audio_device, channels, None

        extra = self._wasapi_loopback_settings()
        if extra is not None:
            device = self._default_wasapi_output()
            info = sd.query_devices(device)
            channels = max(1, int(info["max_output_channels"]))
            logger.info("Loopback (WASAPI): '%s' (%d ch)", info["name"], channels)
            return device, channels, extra

        stereo_mix = self._find_stereo_mix()
        if stereo_mix is not None:
            idx, info = stereo_mix
            channels = max(1, min(2, int(info["max_input_channels"])))
            logger.info("Loopback (Stereo Mix): '%s' (%d ch)", info["name"], channels)
            return idx, channels, None

        raise RuntimeError(
            "System-audio loopback is unavailable: this sounddevice build has no "
            "WASAPI loopback and no 'Stereo Mix' input device was found. Enable "
            "Stereo Mix in Windows Sound settings, pass --audio-device <index>, or "
            "use --audio-source mic / file."
        )

    @staticmethod
    def _wasapi_loopback_settings():
        """Return loopback ``WasapiSettings`` if this build supports it, else None."""
        try:
            return sd.WasapiSettings(loopback=True)
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _find_stereo_mix():
        """Locate a system-audio capture input device by common names."""
        needles = ("stereo mix", "loopback", "what u hear", "wave out", "wave-out")
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] <= 0:
                continue
            name = dev["name"].lower()
            if any(n in name for n in needles):
                return idx, dev
        return None

    @staticmethod
    def _default_wasapi_output() -> int:
        for host in sd.query_hostapis():
            if "WASAPI" in host["name"]:
                dev = host.get("default_output_device", -1)
                if dev is not None and dev >= 0:
                    return dev
        raise RuntimeError(
            "No WASAPI output device found for loopback capture. "
            "Pass --audio-device or use --audio-source mic."
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("Capture status: %s", status)
        _enqueue_drop_oldest(self._queue, _to_mono(indata))

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            device=self._device,
            callback=self._callback,
            extra_settings=self._extra_settings,
            latency="high",
        )
        self._stream.start()
        # The device may have negotiated a different rate; reflect it.
        self.sample_rate = int(self._stream.samplerate)
        logger.info("Live capture started (%d Hz).", self.sample_rate)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class PyAudioLoopbackEngine:
    """Captures system audio via WASAPI loopback using PyAudioWPatch.

    Stock ``sounddevice`` wheels don't expose WASAPI loopback, so for "capture
    whatever is playing" we use PyAudioWPatch, which adds first-class loopback
    devices. We pick the loopback device that mirrors the default speakers and
    tee mono blocks to the analysis queue, just like the other engines.
    """

    def __init__(self, config: Config, block_queue: "queue.Queue"):
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio
        self.config = config
        self._queue = block_queue
        self.finished = threading.Event()
        self.hop_size = config.hop_size
        self.blocksize = max(config.hop_size, int(config.audio_blocksize))

        self._p = pyaudio.PyAudio()
        dev = self._resolve_device()
        self._device_index = int(dev["index"])
        self.channels = max(1, int(dev["maxInputChannels"]))
        self.sample_rate = int(dev["defaultSampleRate"])
        self._stream = None
        logger.info(
            "Loopback (PyAudioWPatch): '%s' (%d ch, %d Hz)",
            dev["name"], self.channels, self.sample_rate,
        )

    def _resolve_device(self) -> dict:
        p = self._p
        # An explicit --audio-device is interpreted as a PyAudio device index.
        if self.config.audio_device is not None:
            return p.get_device_info_by_index(int(self.config.audio_device))

        wasapi = p.get_host_api_info_by_type(self._pa.paWASAPI)
        speakers = p.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
        if speakers.get("isLoopbackDevice"):
            return speakers
        # Find the loopback device that mirrors the default speakers.
        for lb in p.get_loopback_device_info_generator():
            if speakers["name"] in lb["name"]:
                return lb
        # Otherwise take the first available loopback device.
        for lb in p.get_loopback_device_info_generator():
            return lb
        raise RuntimeError(
            "No WASAPI loopback device found. Pass --audio-device <index> "
            "or use --audio-source mic / file."
        )

    def _callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.float32)
        if self.channels > 1:
            samples = samples.reshape(-1, self.channels)
        _enqueue_drop_oldest(self._queue, _to_mono(samples))
        return (None, self._pa.paContinue)

    def start(self) -> None:
        self._stream = self._p.open(
            format=self._pa.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.blocksize,
            input=True,
            input_device_index=self._device_index,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        logger.info("Loopback capture started (%d Hz).", self.sample_rate)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._p.terminate()


def _has_pyaudiowpatch() -> bool:
    try:
        import pyaudiowpatch  # noqa: F401
        return True
    except ImportError:
        return False


def make_engine(config: Config, block_queue: "queue.Queue"):
    """Build the audio source engine for ``config.audio_source``."""
    if config.audio_source == "file":
        if not config.audio_file:
            raise ValueError("audio_source='file' requires an audio file path.")
        return FilePlaybackEngine(config, block_queue)
    if config.audio_source == "mic":
        return LiveInputEngine(config, block_queue)
    if config.audio_source == "loopback":
        # PyAudioWPatch is the reliable WASAPI loopback path; fall back to the
        # sounddevice resolver (WASAPI flag / Stereo Mix) only if it's missing.
        if _has_pyaudiowpatch():
            return PyAudioLoopbackEngine(config, block_queue)
        logger.warning(
            "PyAudioWPatch not installed; trying sounddevice loopback fallback. "
            "For reliable system-audio capture: pip install PyAudioWPatch"
        )
        return LiveInputEngine(config, block_queue)
    raise ValueError(f"Unknown audio_source: {config.audio_source!r}")


class AnalyzerThread:
    """Drains the block queue, runs aubio, and publishes the latest features."""

    def __init__(
        self,
        config: Config,
        block_queue: "queue.Queue",
        audio_state: AudioState,
        sample_rate: int,
        shutdown: threading.Event,
        audio_ring: Optional[AudioRing] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self._queue = block_queue
        self._state = audio_state
        self._shutdown = shutdown
        self._stop = stop_event or threading.Event()
        self._ring = audio_ring
        self._analyzer = AubioAnalyzer(
            sample_rate=sample_rate,
            hop_size=config.hop_size,
            fft_size=config.fft_size,
            norm_decay=config.react_norm_decay,
            bpm_smoothing=config.beat_bpm_smoothing,
            tempo_threshold=config.beat_tempo_threshold,
            tempo_silence_db=config.beat_silence_db,
            beat_grid_lock=config.beat_grid_lock,
            beat_bpm_min=config.beat_bpm_min,
            beat_bpm_max=config.beat_bpm_max,
            spectral_fft=config.react_spectral_fft,
            adaptive_norm=config.react_adaptive_norm,
            norm_range_db=config.react_norm_range_db,
            norm_gate_db=config.react_norm_gate_db,
            beat_adaptive_gate=config.beat_adaptive_gate,
            beat_gate_k=config.beat_gate_k,
            beat_octave_correct=config.beat_octave_correct,
            beat_octave_lo=config.beat_octave_lo,
        )
        self._thread = threading.Thread(target=self._run, name="audio-analyzer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    @property
    def analyzer(self) -> AubioAnalyzer:
        return self._analyzer

    def request_stop(self) -> None:
        self._stop.set()

    def sync_from_config(self, config: Config) -> None:
        self._analyzer.sync_from_config(config)

    def _run(self) -> None:
        logger.info("Audio analyzer thread started.")
        hop = self.config.hop_size
        # Carry leftover samples between blocks so aubio always receives whole,
        # gap-free hops. Devices (esp. WASAPI loopback) deliver blocks whose
        # length isn't a multiple of hop_size; zero-padding each partial hop
        # would inject periodic silence into aubio's continuous tempo tracker
        # and scramble the beat phase. Buffering the remainder keeps the hop
        # stream contiguous, which is what the tempo/onset detectors require.
        residual = np.empty(0, dtype=np.float32)
        while not self._shutdown.is_set() and not self._stop.is_set():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._ring is not None:
                # Tee the full block to the raw-audio ring for CLAP.
                self._ring.append(block)
            if residual.size:
                block = np.concatenate((residual, block))
            n_hops = block.shape[0] // hop
            for i in range(n_hops):
                chunk = block[i * hop : (i + 1) * hop]
                features: AudioFeatures = self._analyzer.process(chunk)
                self._state.set(features)
            residual = np.ascontiguousarray(block[n_hops * hop :], dtype=np.float32)
        logger.info("Audio analyzer thread stopped.")
