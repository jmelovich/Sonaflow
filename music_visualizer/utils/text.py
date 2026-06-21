"""Text helpers for prompt strings."""

from __future__ import annotations

import re
from typing import Literal, Optional, Sequence

# SD prompts: ASCII letters, digits, whitespace, common punctuation only.
_NON_ENGLISH = re.compile(r"[^a-zA-Z0-9\s,.'\"\-;:!?()&]+")
_WHITESPACE = re.compile(r"\s+")

PromptBlacklistMode = Literal["discard", "strip"]


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


def apply_prompt_blacklist(
    prompt: str,
    terms: Sequence[str],
    mode: PromptBlacklistMode,
    *,
    whole_words: bool = True,
) -> Optional[str]:
    """Filter auto-generated prompt text (Qwen captions, CLAP tags).

    Returns ``None`` when the addition should be rejected and the current prompt
    kept. In ``strip`` mode, returns the cleaned text (possibly unchanged).
    """
    prompt = (prompt or "").strip()
    if not prompt or not terms:
        return prompt
    if mode == "discard":
        if prompt_matches_blacklist(prompt, terms, whole_words=whole_words):
            return None
        return prompt
    cleaned = strip_blacklist_terms(
        prompt, terms, whole_words=whole_words
    ).strip()
    if not cleaned:
        return None
    return cleaned


def filter_prompt_addition(
    text: str,
    blacklist: str,
    mode: PromptBlacklistMode,
    *,
    whole_words: bool = True,
) -> Optional[str]:
    """Apply blacklist settings to an auto-generated addition (not the base prompt)."""
    return apply_prompt_blacklist(
        text,
        parse_prompt_blacklist(blacklist),
        mode,
        whole_words=whole_words,
    )


def filter_composed_addition(
    base: str,
    addition: str,
    blacklist: str,
    mode: PromptBlacklistMode,
    *,
    whole_words: bool = True,
) -> Optional[str]:
    """Filter ``addition`` and append to exempt ``base`` prompt text."""
    base = (base or "").strip()
    addition = (addition or "").strip()
    if not addition:
        return base
    filtered = filter_prompt_addition(
        addition, blacklist, mode, whole_words=whole_words
    )
    if filtered is None:
        # Addition rejected or fully stripped — fall back to the exempt base
        # prompt instead of leaving a stale composed prompt in place.
        return base if base else None
    if not base:
        return filtered
    return f"{base}, {filtered}"


def filter_prompt_for_sd(
    prompt: str,
    blacklist: str,
    mode: PromptBlacklistMode,
    *,
    whole_words: bool = True,
) -> Optional[str]:
    """Alias for :func:`filter_prompt_addition`."""
    return filter_prompt_addition(
        prompt, blacklist, mode, whole_words=whole_words
    )
