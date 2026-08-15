"""Sentence-length mean, variance, and per-sentence z-scores.

Linguistic reasoning
--------------------
Human drafts mix fragments, long winding memories, and short punches.
LLM-generated prose — especially "polish this" output — often snaps
toward a preferred band of 18–28 words. Unusually *low variance* of
sentence length is the signal, not any particular mean. As with
burstiness, we compare each sentence to the essay it came from.
"""

from __future__ import annotations

import re

from features.base import SentenceSpan, safe_mean, safe_var, zscores
from features.plain_english import length_var_text, length_z_text

_WORD = re.compile(r"[A-Za-z0-9']+")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def length_stats(
    sentences: list[SentenceSpan],
) -> tuple[list[int], float, float, list[float], str, list[str]]:
    counts = [word_count(s.text) for s in sentences]
    mean = safe_mean([float(c) for c in counts])
    var = safe_var([float(c) for c in counts])
    zs = zscores([float(c) for c in counts])
    essay_expl = length_var_text(var, mean)
    sent_expl = [length_z_text(z) for z in zs]
    return counts, mean, var, zs, essay_expl, sent_expl
