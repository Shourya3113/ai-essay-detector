"""Transition-word / formulaic-connective density.

Linguistic reasoning
--------------------
Models over-produce explicit discourse markers because they are reliable
next-token choices in argumentative prose: "Moreover," "Furthermore,"
"In conclusion," "It is important to note." Human personal essays more
often pivot with scene, time, or a dry "Anyway,". We count a closed
lexicon of high-precision LLM connectives plus a few well-known
formulaic frames. Density is per word, at both sentence and essay
level. This feature is easy to game and easy to over-trust — it is one
signal, not a tell.
"""

from __future__ import annotations

import re

from features.base import SentenceSpan
from features.plain_english import transition_text
from features.sentence_length import word_count

MARKERS: tuple[str, ...] = (
    "moreover",
    "furthermore",
    "in conclusion",
    "in summary",
    "to summarize",
    "in addition",
    "additionally",
    "consequently",
    "nevertheless",
    "nonetheless",
    "that being said",
    "it is important to note",
    "it is worth noting",
    "it is crucial",
    "a testament to",
    "plays a pivotal role",
    "pivotal role",
    "underscores",
    "underscore",
    "delve",
    "delves",
    "delving",
    "tapestry",
    "landscape of",
    "in today's world",
    "in today's society",
    "not only",
    "but also",
    "on the other hand",
    "as a result",
    "this experience taught me",
    "i learned the importance",
    "shaped me into",
    "from a young age",
    "ever since i was",
    "i have always been",
)


def _count_markers(text: str) -> int:
    lowered = text.lower()
    n = 0
    for m in MARKERS:
        start = 0
        while True:
            i = lowered.find(m, start)
            if i < 0:
                break
            n += 1
            start = i + max(len(m), 1)
    return n


def transition_features(
    sentences: list[SentenceSpan],
) -> tuple[list[int], list[float], list[str], float, str]:
    counts = [_count_markers(s.text) for s in sentences]
    densities: list[float] = []
    expl: list[str] = []
    for sent, c in zip(sentences, counts):
        words = max(word_count(sent.text), 1)
        dens = c / words
        densities.append(dens)
        expl.append(transition_text(dens, float(c)))
    total_c = sum(counts)
    total_w = max(sum(word_count(s.text) for s in sentences), 1)
    essay_dens = total_c / total_w
    essay_expl = transition_text(essay_dens, float(total_c))
    return counts, densities, expl, essay_dens, essay_expl
