"""Moving-window type-token ratio (vocabulary richness).

Linguistic reasoning
--------------------
Type-token ratio (unique words / total words) is a crude lexical-
diversity measure. LLM samples often reuse a comfortable mid-register
vocabulary ("impact," "journey," "community") and show a flatter TTR
curve. Humans repeat too — especially in personal essays — so we use a
*moving window* around each sentence rather than one essay-level number.
A local dip in TTR next to otherwise varied prose is more informative
than a global average.
"""

from __future__ import annotations

import re

from features.base import SentenceSpan
from features.plain_english import ttr_text

_WORD = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def window_ttr(
    sentences: list[SentenceSpan],
    window_words: int = 80,
) -> tuple[list[float], list[str], float]:
    """TTR in a word window centered on each sentence; also essay TTR."""
    sent_tokens = [tokenize(s.text) for s in sentences]
    all_tokens = [t for st in sent_tokens for t in st]
    essay_ttr = type_token_ratio(all_tokens)

    # Prefix lengths so we can slice the flattened token list.
    prefix = [0]
    for st in sent_tokens:
        prefix.append(prefix[-1] + len(st))

    values: list[float] = []
    expl: list[str] = []
    half = max(window_words // 2, 10)
    n = len(all_tokens)
    for i, st in enumerate(sent_tokens):
        center = (prefix[i] + prefix[i + 1]) // 2 if st else prefix[i]
        lo = max(0, center - half)
        hi = min(n, center + half)
        window = all_tokens[lo:hi] or st
        val = type_token_ratio(window)
        values.append(val)
        expl.append(ttr_text(val))
    return values, expl, essay_ttr
