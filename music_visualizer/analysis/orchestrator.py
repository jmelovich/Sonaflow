"""The CLAP prompt-injection loop, on its own thread.

Periodically (every ``clap_interval_sec``) it reads the most recent
``clap_window_sec`` of audio from the :class:`AudioRing`, scores it with
:class:`CLAPAnalyzer`, folds the scores through :class:`PromptBuilder` (along
with the latest editable base prompt), and publishes any changed result to the
active :class:`SharedPrompt` that the generation loop consumes.

Running off the generation/audio threads keeps CLAP inference (~10-30 ms) from
stalling either the realtime audio callback or the frame loop. A smaller window
reduces lag (the tag tracks the music more tightly) at the cost of CLAP
confidence; both window and interval are configurable.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

from music_visualizer.analysis.clap_analyzer import CLAPAnalyzer
from music_visualizer.analysis.prompt_builder import PromptBuilder, TagDetail
from music_visualizer.config import Config
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.shared_state import AudioRing, SharedPrompt

logger = get_logger(__name__)


class PromptOrchestrator:
    """Drives CLAP -> PromptBuilder -> active prompt on a background thread."""

    def __init__(
        self,
        config: Config,
        analyzer: CLAPAnalyzer,
        builder: PromptBuilder,
        audio_ring: AudioRing,
        base_prompt: SharedPrompt,
        active_prompt: SharedPrompt,
        shutdown: threading.Event,
        embeds_slot=None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self._analyzer = analyzer
        self._builder = builder
        self._ring = audio_ring
        self._base = base_prompt
        self._active = active_prompt
        self._shutdown = shutdown
        self._stop = stop_event or threading.Event()
        # Optional off-thread prompt encoding: when a pipeline + embeds slot are
        # wired in, this thread encodes the composed prompt and hands the ready
        # tensor to the generation loop, keeping the CLIP encode off the render
        # hot path (the source of the per-cycle stutter).
        self._embeds_slot = embeds_slot
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._qwen: Optional[object] = None
        self._current_composed: Optional[str] = None
        self._force_embeds = False
        self._lock = threading.Lock()
        self._tags: List[str] = []
        self._tag_details: List[TagDetail] = []
        self._raw_top: List[Tuple[str, float]] = []
        self._audio_sec: float = 0.0
        self._status: str = "starting"
        self._last_score_ms: float = 0.0
        self._score_ms_ema: float = 0.0
        self._thread = threading.Thread(
            target=self._run, name="prompt-orchestrator", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def set_pipeline(self, pipeline) -> None:
        """Attach the pipeline so embeddings are encoded on this thread.

        Called after the pipeline is built (it doesn't exist when the
        orchestrator starts). Forces a one-shot re-encode of the current
        composed prompt so the initial tags apply even if they've gone stable.
        """
        with self._pipeline_lock:
            self._pipeline = pipeline
            self._force_embeds = True

    def set_qwen_orchestrator(self, qwen_orchestrator) -> None:
        """Merge Qwen captions onto the manual base before CLAP composes tags."""
        self._qwen = qwen_orchestrator

    def sync_blacklist_from_config(self, cfg: Config) -> None:
        """Apply live blacklist edits to the CLAP prompt builder."""
        self.config = cfg
        b = self._builder
        b.prompt_blacklist = cfg.prompt_blacklist
        b.prompt_blacklist_mode = cfg.prompt_blacklist_mode
        b.prompt_blacklist_whole_words = cfg.prompt_blacklist_whole_words

    def sync_from_config(self, cfg: Config) -> None:
        """Apply live UI edits to CLAP builder/analyzer tuning."""
        self.config = cfg
        b = self._builder
        b.enter_threshold = cfg.clap_enter_threshold
        b.exit_threshold = cfg.clap_exit_threshold
        b.score_tau = cfg.clap_score_tau
        b.min_hold_sec = cfg.clap_min_hold_sec
        b.weight_min = cfg.clap_weight_min
        b.weight_max = cfg.clap_weight_max
        b.weight_step = cfg.clap_weight_step
        b.max_tags = cfg.clap_max_tags
        self.sync_blacklist_from_config(cfg)
        a = self._analyzer
        a.softmax_temp = cfg.clap_softmax_temp
        a.calibrate = cfg.clap_calibrate
        a.calib_alpha = cfg.clap_calib_alpha
        a.calib_warmup = cfg.clap_calib_warmup
        a.calib_spike_guard = cfg.clap_calib_spike_guard

    def _effective_base_prompt(self) -> str:
        manual = self._base.current().strip()
        if self._qwen is not None:
            caption = self._qwen.last_caption().strip()
            if caption:
                return f"{manual}, {caption}" if manual else caption
        return manual

    def _publish_composed(self, composed: str) -> None:
        """Publish a composed prompt to the active prompt channel."""
        self._current_composed = composed
        self._active.set(composed)

    def republish_from_base(self) -> None:
        """Re-compose and push the active prompt after a manual base edit."""
        self._builder.set_base_prompt(self._effective_base_prompt())
        composed = self._builder.compose_now()
        self._publish_composed(composed)
        with self._pipeline_lock:
            pipeline = self._pipeline
        if self._embeds_slot is not None and pipeline is not None:
            embeds = pipeline.compute_embeds(composed)
            if embeds is not None:
                self._embeds_slot.set(embeds)

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def request_stop(self) -> None:
        self._stop.set()

    def active_tags(self) -> List[str]:
        with self._lock:
            return list(self._tags)

    def overlay_lines(self) -> List[str]:
        """Debug overlay lines: raw CLAP top-k, injected tags, and status."""
        with self._lock:
            raw = list(self._raw_top)
            details = list(self._tag_details)
            audio_sec = self._audio_sec
            status = self._status
            score_ms = self._score_ms_ema
        lines: List[str] = []
        if audio_sec < 0.25:
            lines.append(f"CLAP: waiting for audio ({audio_sec:.2f}s buffered)")
            return lines
        if raw:
            parts = [f"{phrase} {score:.2f}" for phrase, score in raw]
            lines.append(f"CLAP top ({score_ms:.0f}ms): " + " | ".join(parts))
        else:
            lines.append(f"CLAP: no scores ({status})")
        if details:
            for phrase, weight, score in details:
                lines.append(f"  + ({phrase}) w={weight:.2f} score={score:.3f}")
        else:
            enter = self.config.clap_enter_threshold
            lines.append(f"CLAP tags: none active (need score>={enter:.2f})")
        return lines

    def _run(self) -> None:
        cfg = self.config
        interval = max(0.05, float(cfg.clap_interval_sec))
        window_sec = float(cfg.clap_window_sec)
        top_k = int(cfg.clap_top_k)
        logger.info(
            "Prompt orchestrator started (window=%.2fs interval=%.2fs top_k=%d).",
            window_sec, interval, top_k,
        )
        last = time.time()
        while not self._shutdown.is_set() and not self._stop.is_set():
            # Pick up manual base-prompt edits and any Qwen caption layer.
            self._builder.set_base_prompt(self._effective_base_prompt())

            window = self._ring.read_latest_seconds(window_sec)
            audio_sec = window.shape[0] / float(self._ring.sample_rate)
            now = time.time()
            dt = now - last
            last = now
            prev_tags = self._builder.active_tags()
            score_ms = 0.0
            if audio_sec < 0.25:
                status = "buffering audio"
                results: List[Tuple[str, float]] = []
            else:
                try:
                    t0 = time.perf_counter()
                    results = self._analyzer.score(
                        window, self._ring.sample_rate, top_k=top_k
                    )
                    score_ms = (time.perf_counter() - t0) * 1000.0
                    self._score_ms_ema = (
                        score_ms if self._score_ms_ema <= 0.0
                        else 0.8 * self._score_ms_ema + 0.2 * score_ms
                    )
                    status = "ok" if results else "empty score"
                except Exception:
                    logger.exception("CLAP scoring failed; keeping previous prompt.")
                    results = []
                    status = "score error"

            composed = self._builder.update(results, now, dt)
            tags = self._builder.active_tags()
            details = self._builder.active_tag_details()
            with self._lock:
                self._tags = tags
                self._tag_details = details
                self._raw_top = results
                self._audio_sec = audio_sec
                self._status = status
                self._last_score_ms = score_ms

            added = [t for t in tags if t not in prev_tags]
            removed = [t for t in prev_tags if t not in tags]
            logger.debug(
                "CLAP cycle audio=%.2fs score=%.1fms top=%s injected=%s",
                audio_sec, score_ms,
                [(p, round(s, 3)) for p, s in results],
                [(p, round(w, 2), round(s, 3)) for p, w, s in details],
            )
            if added:
                logger.debug("CLAP +tags: %s", ", ".join(added))
            if removed:
                logger.debug("CLAP -tags: %s", ", ".join(removed))
            if composed is not None:
                self._publish_composed(composed)
                logger.debug("CLAP prompt updated")

            # Encode the composed prompt off the render thread and hand the
            # generation loop a ready tensor. Re-encode when the prompt changed,
            # or once right after the pipeline first becomes available.
            if self._embeds_slot is not None:
                with self._pipeline_lock:
                    pipeline = self._pipeline
                    force = self._force_embeds
                    self._force_embeds = False
                if pipeline is not None and self._current_composed is not None:
                    if composed is not None or force:
                        embeds = pipeline.compute_embeds(self._current_composed)
                        if embeds is not None:
                            self._embeds_slot.set(embeds)

            # Sleep the remainder of the interval, waking early on shutdown.
            self._shutdown.wait(timeout=interval)
        logger.info("Prompt orchestrator stopped.")
