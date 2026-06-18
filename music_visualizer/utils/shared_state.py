"""Thread-safe primitives shared across the visualizer threads.

For milestone 1 this is intentionally small: a shutdown event plus a thread-safe
prompt holder. As the audio pipeline lands, the richer ``PromptState`` from
ProjectOutline.md (mood tags, denoising strength, seed drift, ...) will live here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from music_visualizer.audio.features import AudioFeatures


@dataclass
class SharedPrompt:
    """A thread-safe container for the currently active prompt.

    The generation loop polls :meth:`take_if_changed` each frame and only
    re-encodes the prompt embeddings when the text actually changed.
    """

    prompt: str
    negative_prompt: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _version: int = 0
    _seen_version: int = field(default=-1, repr=False)

    def set(self, prompt: str, *, allow_empty: bool = False) -> None:
        prompt = prompt.strip()
        if not prompt and not allow_empty:
            return
        with self._lock:
            if self.prompt == prompt:
                return
            self.prompt = prompt
            self._version += 1

    def take_if_changed(self) -> str | None:
        """Return the prompt if it changed since the last call, else ``None``."""
        with self._lock:
            if self._version == self._seen_version:
                return None
            self._seen_version = self._version
            return self.prompt

    def current(self) -> str:
        with self._lock:
            return self.prompt


class PromptEmbeds:
    """Thread-safe handoff for precomputed prompt-embedding tensors.

    The CLAP orchestrator encodes the composed (weighted) prompt on its own
    thread and publishes the ready embedding tensor here; the generation loop
    polls :meth:`take_if_changed` and only swaps the tensor in. This keeps the
    expensive compel parse + CLIP text-encode off the render hot path, where it
    otherwise caused a stutter once per CLAP cycle.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embeds = None
        self._version = 0
        self._seen_version = -1

    def set(self, embeds) -> None:
        with self._lock:
            self._embeds = embeds
            self._version += 1

    def take_if_changed(self):
        with self._lock:
            if self._version == self._seen_version:
                return None
            self._seen_version = self._version
            return self._embeds


class PromptBlendState:
    """Crossfade between two precomputed CLIP/compel embedding tensors.

    The Qwen orchestrator encodes the previous and next caption off the render
    thread, then the generation loop reads :meth:`current_embeds` each frame and
    linearly interpolates over ``duration_sec``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._from = None
        self._to = None
        self._start = 0.0
        self._duration = 0.0
        self._hold = None  # embeds after blend completes (or initial snap)

    def start_blend(self, from_embeds, to_embeds, duration_sec: float) -> None:
        if from_embeds is None or to_embeds is None:
            return
        with self._lock:
            self._from = from_embeds
            self._to = to_embeds
            self._start = time.perf_counter()
            self._duration = max(1e-3, float(duration_sec))

    def snap(self, embeds) -> None:
        """Jump immediately (first caption or blend disabled)."""
        if embeds is None:
            return
        with self._lock:
            self._from = None
            self._to = None
            self._hold = embeds

    def current_embeds(self):
        """Return blended, held, or ``None`` if nothing has been published yet."""
        with self._lock:
            if self._from is None or self._to is None:
                return self._hold
            elapsed = time.perf_counter() - self._start
            t = min(1.0, elapsed / self._duration)
            if t >= 1.0:
                self._hold = self._to
                self._from = None
                self._to = None
                return self._hold
            return self._from * (1.0 - t) + self._to * t

    def progress(self) -> float:
        """Blend progress in [0, 1]; 1.0 when idle/holding."""
        with self._lock:
            if self._from is None or self._to is None:
                return 1.0
            elapsed = time.perf_counter() - self._start
            return min(1.0, elapsed / self._duration)

    def is_blending(self) -> bool:
        with self._lock:
            return self._from is not None and self._to is not None


class AudioState:
    """Thread-safe holder for the most recent :class:`AudioFeatures`.

    The analyzer thread writes the latest snapshot (~80-94x/sec); the generation
    loop reads whatever is current each frame. Continuous signals (rms, bands)
    only need their freshest value, so this is a single-slot latch.

    Beats, however, are *transient events*: ``is_beat`` is true on exactly one
    hop per beat (~1-3 of every ~90 snapshots). The generation loop polls far
    slower than the analyzer writes (especially under heavy upscaling), so a
    plain latch would let the next non-beat snapshot overwrite a beat before the
    loop ever reads it -- most beats would be lost and the few caught would look
    random. To avoid that we latch beats *stickily*: any beat written since the
    last :meth:`get` is OR-ed onto the next snapshot returned, then cleared, so
    every beat is delivered exactly once regardless of the frame-rate mismatch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional["AudioFeatures"] = None
        self._beat_pending = False

    def set(self, features: "AudioFeatures") -> None:
        with self._lock:
            if features.is_beat:
                self._beat_pending = True
            self._latest = features

    def get(self, consume_beat: bool = False) -> Optional["AudioFeatures"]:
        """Return the latest snapshot, surfacing any beat seen since last read.

        ``consume_beat`` should be ``True`` for the single authoritative reader
        that drives beat-triggered motion (the generation loop), so the sticky
        beat is cleared once delivered. Display-only readers (e.g. the overlay)
        leave it ``False`` to peek at the beat without stealing it from that
        reader.
        """
        with self._lock:
            features = self._latest
            if features is None:
                return None
            if self._beat_pending and not features.is_beat:
                features = replace(features, is_beat=True)
            if consume_beat:
                self._beat_pending = False
            return features


class AudioRing:
    """Thread-safe rolling buffer of recent mono ``float32`` PCM samples.

    The analyzer thread appends each mono hop as it is processed (single
    writer); slower consumers (e.g. the CLAP orchestrator) read the most recent
    N seconds on demand. A preallocated circular buffer avoids per-append
    allocations on the audio path. Only the freshest ``capacity_seconds`` of
    audio is retained.
    """

    def __init__(self, sample_rate: int, capacity_seconds: float = 8.0):
        self.sample_rate = int(sample_rate)
        self._capacity = max(1, int(self.sample_rate * float(capacity_seconds)))
        self._buf = np.zeros(self._capacity, dtype=np.float32)
        self._write = 0          # next write index (mod capacity)
        self._filled = 0         # total valid samples currently stored (<= capacity)
        self._lock = threading.Lock()

    def append(self, block: np.ndarray) -> None:
        """Append a mono float32 block (1-D); long blocks keep only the tail."""
        samples = np.ascontiguousarray(block, dtype=np.float32).reshape(-1)
        n = samples.shape[0]
        if n == 0:
            return
        if n >= self._capacity:
            samples = samples[-self._capacity :]
            n = samples.shape[0]
        with self._lock:
            end = self._write + n
            if end <= self._capacity:
                self._buf[self._write : end] = samples
            else:
                first = self._capacity - self._write
                self._buf[self._write :] = samples[:first]
                self._buf[: end - self._capacity] = samples[first:]
            self._write = end % self._capacity
            self._filled = min(self._capacity, self._filled + n)

    def read_latest_seconds(self, seconds: float) -> np.ndarray:
        """Return the most recent ``seconds`` of audio as a contiguous 1-D array.

        Returns fewer samples if less audio has been buffered so far (possibly
        an empty array before any audio arrives).
        """
        want = int(self.sample_rate * max(0.0, float(seconds)))
        with self._lock:
            n = min(want, self._filled)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write - n) % self._capacity
            end = start + n
            if end <= self._capacity:
                return self._buf[start:end].copy()
            first = self._capacity - start
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buf[start:]
            out[first:] = self._buf[: end - self._capacity]
            return out
