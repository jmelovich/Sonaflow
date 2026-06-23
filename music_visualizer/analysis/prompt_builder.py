"""Turns raw CLAP scores into a stable, compel-weighted prompt string.

CLAP scores are noisy frame-to-frame, so feeding them straight to the diffusion
prompt would make it flicker. :class:`PromptBuilder` adds three stabilizers:

* **EMA smoothing** of each phrase's probability over time (``score_tau``), so a
  brief spike doesn't immediately swap the prompt.
* **Hysteresis** on the active tag set: a phrase only *enters* once its smoothed
  score crosses ``enter_threshold`` and only *leaves* once it drops below the
  lower ``exit_threshold`` (and after a ``min_hold_sec`` minimum hold). This
  Schmitt-trigger behavior stops tags chattering around a single threshold.
* **Confidence-weighted emphasis**: each active phrase's smoothed score maps to
  a compel weight in ``[weight_min, weight_max]`` so stronger matches dominate.

The composed string looks like::

    "<base prompt>, (dark pulsing energy)1.20, (neon city in the rain)0.95"

which the pipeline encodes with compel (StreamDiffusion's plain CLIP tokenizer
would otherwise treat ``(...)1.2`` as literal text).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from music_visualizer.utils.text import apply_prompt_blacklist, parse_prompt_blacklist

TagDetail = Tuple[str, float, float]  # (phrase, compel_weight, smoothed_score)

PromptBlacklistMode = Literal["discard", "strip"]


def _ema_alpha(dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return 1.0
    return 1.0 - math.exp(-max(0.0, dt) / tau)


@dataclass
class PromptBuilder:
    base_prompt: str
    enter_threshold: float = 0.12
    exit_threshold: float = 0.06
    score_tau: float = 1.5          # EMA time constant (s) for per-phrase scores
    min_hold_sec: float = 1.5       # minimum time a tag stays active once entered
    weight_min: float = 0.7
    weight_max: float = 1.4
    weight_step: float = 0.1        # quantize weights so tiny EMA drift doesn't
                                    # re-trigger a (costly) prompt re-encode each cycle
    max_tags: int = 3
    prompt_blacklist: str = ""
    prompt_blacklist_mode: PromptBlacklistMode = "strip"
    prompt_blacklist_whole_words: bool = True

    _scores: Dict[str, float] = field(default_factory=dict, init=False)
    _active: Dict[str, float] = field(default_factory=dict, init=False)  # phrase -> enter time
    _last_composed: Optional[str] = field(default=None, init=False)

    def set_base_prompt(self, prompt: str) -> None:
        prompt = (prompt or "").strip()
        if self.base_prompt == prompt:
            return
        self.base_prompt = prompt
        # Force the next update()/republish to see the new foundation text.
        self._last_composed = None

    def apply_blacklist(
        self,
        blacklist: str,
        mode: PromptBlacklistMode,
        *,
        whole_words: bool = True,
    ) -> bool:
        """Update blacklist settings; return True when compose output may change."""
        changed = (
            self.prompt_blacklist != blacklist
            or self.prompt_blacklist_mode != mode
            or self.prompt_blacklist_whole_words != whole_words
        )
        self.prompt_blacklist = blacklist
        self.prompt_blacklist_mode = mode
        self.prompt_blacklist_whole_words = whole_words
        if changed:
            self._last_composed = None
        return changed

    def compose_now(self) -> str:
        """Build the current composed prompt and mark it as published."""
        composed = self._compose()
        self._last_composed = composed
        return composed

    def update(
        self, clap_results: List[Tuple[str, float]], now: float, dt: float
    ) -> Optional[str]:
        """Fold in one round of CLAP scores; return a new prompt or ``None``.

        Returns ``None`` when the composed prompt is unchanged so the caller can
        skip re-encoding embeddings.
        """
        a = _ema_alpha(dt, self.score_tau)
        latest = {phrase: score for phrase, score in clap_results}

        # Decay every tracked phrase toward its latest score (0 if absent now).
        phrases = set(self._scores) | set(latest)
        for phrase in phrases:
            target = latest.get(phrase, 0.0)
            prev = self._scores.get(phrase, 0.0)
            self._scores[phrase] = prev + a * (target - prev)

        # Hysteresis: add phrases above the enter threshold, drop ones that have
        # fallen below the exit threshold and served their minimum hold.
        for phrase, score in self._scores.items():
            if phrase not in self._active and score >= self.enter_threshold:
                self._active[phrase] = now
        for phrase in list(self._active):
            score = self._scores.get(phrase, 0.0)
            held = now - self._active[phrase]
            if score < self.exit_threshold and held >= self.min_hold_sec:
                del self._active[phrase]

        composed = self._compose()
        if composed == self._last_composed:
            return None
        self._last_composed = composed
        return composed

    def _compose(self) -> str:
        # Keep the strongest active tags, capped at ``max_tags``.
        ranked = sorted(
            self._active.keys(), key=lambda p: self._scores.get(p, 0.0), reverse=True
        )[: self.max_tags]
        if not ranked:
            return self.base_prompt
        terms = parse_prompt_blacklist(self.prompt_blacklist)
        parts = [self.base_prompt]
        for phrase in ranked:
            filtered = apply_prompt_blacklist(
                phrase,
                terms,
                self.prompt_blacklist_mode,
                whole_words=self.prompt_blacklist_whole_words,
            )
            if not filtered:
                continue
            parts.append(f"({filtered}){self._weight_for(phrase):.2f}")
        if len(parts) == 1:
            return self.base_prompt
        return ", ".join(parts)

    def _weight_for(self, phrase: str) -> float:
        """Map a smoothed score in [enter, ~max] to a quantized compel weight.

        The result is snapped to ``weight_step`` increments. Without this the
        EMA-smoothed score drifts a little every cycle, the formatted weight
        string changes, and the generation loop re-encodes the prompt (a full
        CLIP text-encoder forward) on nearly every CLAP cycle -- a ~1s stutter.
        Quantizing means a re-encode only happens when the weight crosses a step.
        """
        score = self._scores.get(phrase, 0.0)
        lo, hi = self.enter_threshold, max(self.enter_threshold * 4.0, 0.5)
        t = (score - lo) / (hi - lo) if hi > lo else 0.0
        t = max(0.0, min(1.0, t))
        weight = self.weight_min + t * (self.weight_max - self.weight_min)
        step = self.weight_step
        if step > 1e-6:
            weight = round(weight / step) * step
        return weight

    def active_tags(self) -> List[str]:
        """Current active phrases, strongest first (for overlays/logging)."""
        return [phrase for phrase, _, _ in self.active_tag_details()]

    def active_tag_details(self) -> List[TagDetail]:
        """Injected tags with compel weight and smoothed score, strongest first."""
        ranked = sorted(
            self._active.keys(), key=lambda p: self._scores.get(p, 0.0), reverse=True
        )[: self.max_tags]
        return [
            (phrase, self._weight_for(phrase), self._scores.get(phrase, 0.0))
            for phrase in ranked
        ]

    def weighted_tag_parts(self) -> List[str]:
        """Compel-weighted tag strings without the base prompt."""
        terms = parse_prompt_blacklist(self.prompt_blacklist)
        parts: List[str] = []
        for phrase in self.active_tags():
            filtered = apply_prompt_blacklist(
                phrase,
                terms,
                self.prompt_blacklist_mode,
                whole_words=self.prompt_blacklist_whole_words,
            )
            if filtered:
                parts.append(f"({filtered}){self._weight_for(phrase):.2f}")
        return parts


@dataclass
class MultiPromptBuilder:
    """Merges stabilized tags from multiple CLAP vocabulary channels."""

    base_prompt: str
    channel_order: List[str] = field(default_factory=list)
    builders: Dict[str, PromptBuilder] = field(default_factory=dict, init=False)
    prompt_blacklist: str = ""
    prompt_blacklist_mode: PromptBlacklistMode = "strip"
    prompt_blacklist_whole_words: bool = True
    _last_composed: Optional[str] = field(default=None, init=False)

    def set_base_prompt(self, prompt: str) -> None:
        prompt = (prompt or "").strip()
        if self.base_prompt == prompt:
            return
        self.base_prompt = prompt
        self._last_composed = None

    def apply_blacklist(
        self,
        blacklist: str,
        mode: PromptBlacklistMode,
        *,
        whole_words: bool = True,
    ) -> bool:
        changed = (
            self.prompt_blacklist != blacklist
            or self.prompt_blacklist_mode != mode
            or self.prompt_blacklist_whole_words != whole_words
        )
        self.prompt_blacklist = blacklist
        self.prompt_blacklist_mode = mode
        self.prompt_blacklist_whole_words = whole_words
        for builder in self.builders.values():
            builder.apply_blacklist(blacklist, mode, whole_words=whole_words)
        if changed:
            self._last_composed = None
        return changed

    def sync_from_lists(self, lists: List[dict]) -> None:
        """Rebuild channel builders when vocab lists change (e.g. live UI edits)."""
        from music_visualizer.analysis.vocab_lists import VocabList, normalize_vocab_lists

        normalized = normalize_vocab_lists(lists)
        self.channel_order = [str(item["id"]) for item in normalized]
        new_builders: Dict[str, PromptBuilder] = {}
        for item in normalized:
            vl = VocabList.from_dict(item)
            existing = self.builders.get(vl.id)
            if existing is not None:
                existing.enter_threshold = vl.enter_threshold
                existing.exit_threshold = vl.exit_threshold
                existing.score_tau = vl.score_tau
                existing.min_hold_sec = vl.min_hold_sec
                existing.weight_min = vl.weight_min
                existing.weight_max = vl.weight_max
                existing.weight_step = vl.weight_step
                existing.max_tags = vl.max_tags
                new_builders[vl.id] = existing
            else:
                new_builders[vl.id] = PromptBuilder(
                    base_prompt="",
                    enter_threshold=vl.enter_threshold,
                    exit_threshold=vl.exit_threshold,
                    score_tau=vl.score_tau,
                    min_hold_sec=vl.min_hold_sec,
                    weight_min=vl.weight_min,
                    weight_max=vl.weight_max,
                    weight_step=vl.weight_step,
                    max_tags=vl.max_tags,
                    prompt_blacklist=self.prompt_blacklist,
                    prompt_blacklist_mode=self.prompt_blacklist_mode,
                    prompt_blacklist_whole_words=self.prompt_blacklist_whole_words,
                )
        self.builders = new_builders

    def compose_now(self) -> str:
        composed = self._compose()
        self._last_composed = composed
        return composed

    def update(
        self,
        channel_results: Dict[str, List[Tuple[str, float]]],
        now: float,
        channel_dt: Dict[str, float],
        *,
        scored_ids: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Fold in scores for ``scored_ids`` only; other channels are left unchanged."""
        ids = list(scored_ids if scored_ids is not None else channel_results.keys())
        for vocab_id in ids:
            builder = self.builders.get(vocab_id)
            if builder is None:
                continue
            builder.update(
                channel_results.get(vocab_id, []),
                now,
                channel_dt.get(vocab_id, 0.0),
            )
        composed = self._compose()
        if composed == self._last_composed:
            return None
        self._last_composed = composed
        return composed

    def _compose(self) -> str:
        parts: List[str] = []
        if self.base_prompt:
            parts.append(self.base_prompt)
        for vocab_id in self.channel_order:
            builder = self.builders.get(vocab_id)
            if builder is None:
                continue
            parts.extend(builder.weighted_tag_parts())
        if not parts:
            return self.base_prompt
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts)

    def active_tags(self) -> List[str]:
        tags: List[str] = []
        for vocab_id in self.channel_order:
            builder = self.builders.get(vocab_id)
            if builder is not None:
                tags.extend(builder.active_tags())
        return tags

    def active_tag_details(self) -> List[TagDetail]:
        details: List[TagDetail] = []
        for vocab_id in self.channel_order:
            builder = self.builders.get(vocab_id)
            if builder is not None:
                details.extend(builder.active_tag_details())
        return details

    def channel_active_tags(self, vocab_id: str) -> List[str]:
        builder = self.builders.get(vocab_id)
        return builder.active_tags() if builder is not None else []

    def channel_active_details(self, vocab_id: str) -> List[TagDetail]:
        builder = self.builders.get(vocab_id)
        return builder.active_tag_details() if builder is not None else []
