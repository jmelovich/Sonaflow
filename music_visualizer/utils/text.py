"""Text helpers for prompt strings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

# SD prompts: ASCII letters, digits, whitespace, common punctuation only.
_NON_ENGLISH = re.compile(r"[^a-zA-Z0-9\s,.'\"\-;:!?()&]+")
_WHITESPACE = re.compile(r"\s+")

PromptBlacklistOrder = Literal["strip_first", "discard_first"]
PromptFilterOutcome = Literal["ok", "discard", "empty"]


def normalize_blacklist_order(
    order: PromptBlacklistOrder | str | None,
) -> PromptBlacklistOrder:
    """Coerce UI/settings values to a known blacklist application order."""
    value = str(order or "strip_first").strip().lower()
    if value == "discard_first":
        return "discard_first"
    return "strip_first"


def english_only_text(text: str) -> str:
    """Remove CJK and other non-English script from model output."""
    cleaned = _NON_ENGLISH.sub(" ", text)
    return _WHITESPACE.sub(" ", cleaned).strip()


def parse_prompt_blacklist(text: str) -> list[str]:
    """Split a comma-separated blacklist string into trimmed terms."""
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _blacklist_terms_ordered(terms: Sequence[str]) -> list[str]:
    """Longest terms first so e.g. ``electric guitar`` wins over ``guitar``."""
    return sorted((t for t in terms if t), key=len, reverse=True)


def _blacklist_term_pattern(term: str, *, whole_words: bool = True) -> re.Pattern[str]:
    """Build a case-insensitive blacklist matcher for one entry."""
    words = term.split()
    if not words:
        return re.compile(r"(?!x)x")
    if len(words) == 1:
        escaped = re.escape(words[0])
        if whole_words:
            return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        return re.compile(escaped, re.IGNORECASE)
    # Multi-word: allow flexible whitespace between words.
    inner = r"\s+".join(re.escape(word) for word in words)
    if whole_words:
        return re.compile(rf"(?<!\w){inner}(?!\w)", re.IGNORECASE)
    return re.compile(inner, re.IGNORECASE)


def prompt_matches_blacklist(
    prompt: str,
    terms: Sequence[str],
    *,
    whole_words: bool = True,
) -> bool:
    """Return True when any blacklist entry matches ``prompt``."""
    if not prompt or not terms:
        return False
    return any(
        _blacklist_term_pattern(term, whole_words=whole_words).search(prompt)
        for term in _blacklist_terms_ordered(terms)
    )


def strip_blacklist_terms(
    prompt: str,
    terms: Sequence[str],
    *,
    whole_words: bool = True,
) -> str:
    """Remove blacklist matches from ``prompt`` (case-insensitive)."""
    result = prompt
    for term in _blacklist_terms_ordered(terms):
        result = _blacklist_term_pattern(term, whole_words=whole_words).sub("", result)
    result = _WHITESPACE.sub(" ", result)
    result = re.sub(r",\s*,+", ", ", result)
    return result.strip(" ,")


@dataclass(frozen=True)
class PromptFilterResult:
    """Outcome of filtering an auto-generated prompt addition."""

    text: Optional[str]
    outcome: PromptFilterOutcome


def apply_prompt_blacklists(
    prompt: str,
    *,
    strip_terms: Sequence[str],
    discard_terms: Sequence[str],
    order: PromptBlacklistOrder | str = "strip_first",
    whole_words: bool = True,
) -> PromptFilterResult:
    """Filter auto-generated prompt text (Qwen captions, CLAP tags).

    Returns a :class:`PromptFilterResult`. ``outcome`` is ``discard`` when the
    discard list matched, ``empty`` when stripping removed all text, and ``ok``
    otherwise. ``text`` is the filtered prompt or ``None`` when rejected/empty.
    """
    order = normalize_blacklist_order(order)
    prompt = (prompt or "").strip()
    if not prompt:
        return PromptFilterResult(prompt, "ok")
    if not strip_terms and not discard_terms:
        return PromptFilterResult(prompt, "ok")

    def discard_check(text: str) -> PromptFilterResult:
        if discard_terms and prompt_matches_blacklist(
            text, discard_terms, whole_words=whole_words
        ):
            return PromptFilterResult(None, "discard")
        return PromptFilterResult(text, "ok")

    def strip_apply(text: str) -> PromptFilterResult:
        if not strip_terms:
            return PromptFilterResult(text, "ok")
        cleaned = strip_blacklist_terms(
            text, strip_terms, whole_words=whole_words
        ).strip()
        if not cleaned:
            return PromptFilterResult(None, "empty")
        return PromptFilterResult(cleaned, "ok")

    if order == "strip_first":
        stripped = strip_apply(prompt)
        if stripped.outcome != "ok":
            return stripped
        return discard_check(stripped.text or "")
    discarded = discard_check(prompt)
    if discarded.outcome != "ok":
        return discarded
    return strip_apply(discarded.text or "")


def filter_prompt_addition(
    text: str,
    *,
    strip_blacklist: str = "",
    discard_blacklist: str = "",
    order: PromptBlacklistOrder | str = "strip_first",
    whole_words: bool = True,
) -> Optional[str]:
    """Apply blacklist settings to an auto-generated addition (not the base prompt)."""
    return apply_prompt_blacklists(
        text,
        strip_terms=parse_prompt_blacklist(strip_blacklist),
        discard_terms=parse_prompt_blacklist(discard_blacklist),
        order=order,
        whole_words=whole_words,
    ).text


def filter_prompt_addition_detailed(
    text: str,
    *,
    strip_blacklist: str = "",
    discard_blacklist: str = "",
    order: PromptBlacklistOrder | str = "strip_first",
    whole_words: bool = True,
) -> PromptFilterResult:
    """Like :func:`filter_prompt_addition` but includes the rejection reason."""
    return apply_prompt_blacklists(
        text,
        strip_terms=parse_prompt_blacklist(strip_blacklist),
        discard_terms=parse_prompt_blacklist(discard_blacklist),
        order=order,
        whole_words=whole_words,
    )


def filter_composed_addition(
    base: str,
    addition: str,
    *,
    strip_blacklist: str = "",
    discard_blacklist: str = "",
    order: PromptBlacklistOrder | str = "strip_first",
    whole_words: bool = True,
) -> Optional[str]:
    """Filter ``addition`` and append to exempt ``base`` prompt text."""
    base = (base or "").strip()
    addition = (addition or "").strip()
    if not addition:
        return base
    result = filter_prompt_addition_detailed(
        addition,
        strip_blacklist=strip_blacklist,
        discard_blacklist=discard_blacklist,
        order=order,
        whole_words=whole_words,
    )
    if result.text is None:
        if result.outcome == "discard":
            # Reject the whole addition; caller keeps the current composed prompt.
            return None
        # Strip list removed everything — fall back to the exempt base prompt.
        return base if base else None
    if not base:
        return result.text
    return f"{base}, {result.text}"


def filter_prompt_for_sd(
    prompt: str,
    *,
    strip_blacklist: str = "",
    discard_blacklist: str = "",
    order: PromptBlacklistOrder | str = "strip_first",
    whole_words: bool = True,
) -> Optional[str]:
    """Alias for :func:`filter_prompt_addition`."""
    return filter_prompt_addition(
        prompt,
        strip_blacklist=strip_blacklist,
        discard_blacklist=discard_blacklist,
        order=order,
        whole_words=whole_words,
    )
