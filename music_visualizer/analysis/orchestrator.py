"""The CLAP prompt-injection loop, on its own thread.

Each vocabulary list scores on its own ``interval_sec`` schedule. The loop wakes
often enough to honor the tightest enabled interval, embeds audio once per wake
when any list is due, scores only those lists, and merges stabilized tags through
:class:`MultiPromptBuilder`.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from music_visualizer.analysis.clap_analyzer import CLAPAnalyzer
from music_visualizer.analysis.prompt_builder import MultiPromptBuilder, TagDetail
from music_visualizer.analysis.vocab_lists import VocabList, normalize_vocab_lists
from music_visualizer.config import Config
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.shared_state import AudioRing, SharedPrompt
from music_visualizer.utils.text import filter_prompt_addition

logger = get_logger(__name__)

_MIN_TICK_SEC = 0.05


class PromptOrchestrator:
    """Drives CLAP -> MultiPromptBuilder -> active prompt on a background thread."""

    def __init__(
        self,
        config: Config,
        analyzer: CLAPAnalyzer,
        builder: MultiPromptBuilder,
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
        self._embeds_slot = embeds_slot
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._qwen: Optional[object] = None
        self._current_composed: Optional[str] = None
        self._force_embeds = False
        self._lock = threading.Lock()
        self._tags: List[str] = []
        self._tag_details: List[TagDetail] = []
        self._raw_top: Dict[str, List[Tuple[str, float]]] = {}
        self._audio_sec: float = 0.0
        self._status: str = "starting"
        self._last_score_ms: float = 0.0
        self._score_ms_ema: float = 0.0
        self._last_score_time: Dict[str, float] = {}
        self._last_builder_update: Dict[str, float] = {}
        self._thread = threading.Thread(
            target=self._run, name="prompt-orchestrator", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def set_pipeline(self, pipeline) -> None:
        with self._pipeline_lock:
            self._pipeline = pipeline
            self._force_embeds = True

    def set_qwen_orchestrator(self, qwen_orchestrator) -> None:
        self._qwen = qwen_orchestrator

    def sync_blacklist_from_config(self, cfg: Config) -> None:
        self.config = cfg
        self._builder.apply_blacklist(
            cfg.prompt_blacklist_strip,
            cfg.prompt_blacklist_discard,
            cfg.prompt_blacklist_order,
            whole_words=cfg.prompt_blacklist_whole_words,
        )

    def sync_from_config(self, cfg: Config) -> None:
        self.config = cfg
        lists = normalize_vocab_lists(cfg.clap_vocab_lists)
        self._builder.sync_from_lists(lists)
        self._analyzer.sync_channels(lists)
        self.sync_blacklist_from_config(cfg)

    def _filtered_qwen_caption(self) -> str:
        if self._qwen is None:
            return ""
        caption = self._qwen.last_caption().strip()
        if not caption:
            return ""
        filtered = filter_prompt_addition(
            caption,
            strip_blacklist=self.config.prompt_blacklist_strip,
            discard_blacklist=self.config.prompt_blacklist_discard,
            order=self.config.prompt_blacklist_order,
            whole_words=self.config.prompt_blacklist_whole_words,
        )
        return (filtered or "").strip()

    def _effective_base_prompt(self) -> str:
        manual = self._base.current().strip()
        caption = self._filtered_qwen_caption()
        if caption:
            return f"{manual}, {caption}" if manual else caption
        return manual

    def _vocab_lists(self) -> List[VocabList]:
        return [VocabList.from_dict(item) for item in normalize_vocab_lists(self.config.clap_vocab_lists)]

    def _tick_interval(self) -> float:
        intervals = [
            max(_MIN_TICK_SEC, float(vl.interval_sec))
            for vl in self._vocab_lists()
            if vl.enable
        ]
        if not intervals:
            return 1.0
        return max(_MIN_TICK_SEC, min(intervals) / 4.0)

    def _due_vocab_ids(self, now: float) -> List[str]:
        due: List[str] = []
        for vl in self._vocab_lists():
            if not vl.enable:
                continue
            last = self._last_score_time.get(vl.id, 0.0)
            if last <= 0.0 or now - last >= max(_MIN_TICK_SEC, float(vl.interval_sec)):
                due.append(vl.id)
        return due

    def _publish_composed(self, composed: str) -> None:
        self._current_composed = composed
        self._active.set(composed)

    def republish_from_base(self) -> None:
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
        with self._lock:
            raw = dict(self._raw_top)
            details = list(self._tag_details)
            audio_sec = self._audio_sec
            status = self._status
            score_ms = self._score_ms_ema
        lines: List[str] = []
        if audio_sec < 0.25:
            lines.append(f"CLAP: waiting for audio ({audio_sec:.2f}s buffered)")
            return lines
        lists = self._vocab_lists()
        if raw:
            for vl in lists:
                hits = raw.get(vl.id, [])
                if hits:
                    parts = [f"{phrase} {score:.2f}" for phrase, score in hits]
                    lines.append(f"CLAP {vl.name} ({score_ms:.0f}ms): " + " | ".join(parts))
        else:
            lines.append(f"CLAP: no scores ({status})")
        if details:
            for phrase, weight, score in details:
                lines.append(f"  + ({phrase}) w={weight:.2f} score={score:.3f}")
        elif lists:
            lines.append(
                f"CLAP tags: none active (need score>={lists[0].enter_threshold:.2f})"
            )
        else:
            lines.append("CLAP tags: none active")
        return lines

    def _run(self) -> None:
        cfg = self.config
        window_sec = float(cfg.clap_window_sec)
        lists = self._vocab_lists()
        logger.info(
            "Prompt orchestrator started (window=%.2fs vocabs=%d).",
            window_sec, len(lists),
        )
        start = time.time()
        self._last_builder_update = {vl.id: start for vl in lists}
        while not self._shutdown.is_set() and not self._stop.is_set():
            self._builder.set_base_prompt(self._effective_base_prompt())

            now = time.time()
            due_ids = self._due_vocab_ids(now)
            window = self._ring.read_latest_seconds(window_sec)
            audio_sec = window.shape[0] / float(self._ring.sample_rate)
            prev_tags = self._builder.active_tags()
            score_ms = 0.0
            results: Dict[str, List[Tuple[str, float]]] = {}

            if not due_ids:
                status = "waiting"
            elif audio_sec < 0.25:
                status = "buffering audio"
            else:
                try:
                    t0 = time.perf_counter()
                    results = self._analyzer.score_all(
                        window, self._ring.sample_rate, only_ids=set(due_ids),
                    )
                    score_ms = (time.perf_counter() - t0) * 1000.0
                    self._score_ms_ema = (
                        score_ms if self._score_ms_ema <= 0.0
                        else 0.8 * self._score_ms_ema + 0.2 * score_ms
                    )
                    status = "ok" if results else "empty score"
                except Exception:
                    logger.exception("CLAP scoring failed; keeping previous prompt.")
                    status = "score error"

            channel_dt: Dict[str, float] = {}
            for vocab_id in due_ids:
                prev = self._last_builder_update.get(vocab_id, now)
                channel_dt[vocab_id] = max(0.0, now - prev)
                if vocab_id in results:
                    self._last_score_time[vocab_id] = now
                    self._last_builder_update[vocab_id] = now

            composed = None
            if due_ids and results:
                composed = self._builder.update(
                    results, now, channel_dt, scored_ids=due_ids,
                )
            if composed is None and self._builder._last_composed is None:
                composed = self._builder.compose_now()

            tags = self._builder.active_tags()
            details = self._builder.active_tag_details()
            with self._lock:
                self._tags = tags
                self._tag_details = details
                for vocab_id, hits in results.items():
                    self._raw_top[vocab_id] = hits
                self._audio_sec = audio_sec
                self._status = status
                self._last_score_ms = score_ms

            added = [t for t in tags if t not in prev_tags]
            removed = [t for t in prev_tags if t not in tags]
            if due_ids:
                logger.debug(
                    "CLAP cycle due=%s audio=%.2fs score=%.1fms raw=%s injected=%s",
                    due_ids, audio_sec, score_ms,
                    {k: [(p, round(s, 3)) for p, s in v] for k, v in results.items()},
                    [(p, round(w, 2), round(s, 3)) for p, w, s in details],
                )
            if added:
                logger.debug("CLAP +tags: %s", ", ".join(added))
            if removed:
                logger.debug("CLAP -tags: %s", ", ".join(removed))
            if composed is not None:
                self._publish_composed(composed)
                logger.debug("CLAP prompt updated")

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

            self._shutdown.wait(timeout=self._tick_interval())
        logger.info("Prompt orchestrator stopped.")
