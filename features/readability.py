"""Sentence-to-sentence readability (Flesch–Kincaid) variance.

Linguistic reasoning
--------------------
A single Flesch–Kincaid grade for the whole essay is almost useless
for detection: both humans and models can sit at grade 8–12. What
tends to differ is *stability*. Human drafts drop into monosyllables
for a punch and then climb into a denser sentence. LLM polish often
keeps every sentence in a narrow grade band. We score each sentence
and take the variance, plus a z-score for the span the user clicked.
"""

from __future__ import annotations

import re

from features.base import SentenceSpan, safe_var, zscores
from features.plain_english import fk_var_text, fk_z_text

_WORD = re.compile(r"[A-Za-z0-9']+")
_VOWELS = set("aeiouy")


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    n = 0
    prev_v = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_v:
            n += 1
        prev_v = is_v
    if w.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def flesch_kincaid_sentence(text: str) -> float:
    words = _WORD.findall(text)
    if not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    # One sentence by construction.
    words_per_sent = len(words)
    syl_per_word = syllables / len(words)
    return 0.39 * words_per_sent + 11.8 * syl_per_word - 15.59


def readability_series(
    sentences: list[SentenceSpan],
) -> tuple[list[float], list[float], float, list[str], str]:
    grades = [flesch_kincaid_sentence(s.text) for s in sentences]
    zs = zscores(grades)
    var = safe_var(grades)
    expl = [fk_z_text(z) for z in zs]
    return grades, zs, var, expl, fk_var_text(var)
