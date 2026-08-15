"""Function-word frequency profile vs a general-English reference.

Linguistic reasoning
--------------------
Function words (the, of, to, in, that, would…) are chosen with little
conscious control and have a long history in authorship attribution
(Mosteller & Wallace on the Federalist Papers). LLM samples often
drift toward a generic "average English" mix, or toward a slightly
formal mix (more "therefore"/"however," fewer contractions and
discourse particles like "just"/"really"). We compare the essay's
closed-class distribution to a reference via cosine *distance*.
"""

from __future__ import annotations

import math
import re

from features.plain_english import fw_cosine_text

FUNCTION_WORDS: tuple[str, ...] = (
    "the",
    "of",
    "and",
    "to",
    "a",
    "in",
    "that",
    "is",
    "was",
    "he",
    "for",
    "it",
    "with",
    "as",
    "his",
    "on",
    "be",
    "at",
    "by",
    "i",
    "this",
    "had",
    "not",
    "are",
    "but",
    "from",
    "or",
    "have",
    "an",
    "they",
    "which",
    "you",
    "were",
    "her",
    "their",
    "we",
    "there",
    "been",
    "if",
    "more",
    "when",
    "will",
    "would",
    "who",
    "so",
    "no",
    "out",
    "up",
    "what",
    "about",
    "into",
    "than",
    "them",
    "can",
    "only",
    "other",
    "new",
    "some",
    "could",
    "time",
    "these",
    "two",
    "may",
    "then",
    "do",
    "first",
    "any",
    "my",
    "now",
    "such",
    "like",
    "our",
    "over",
    "man",
    "me",
    "even",
    "most",
    "made",
    "after",
    "also",
    "did",
    "many",
    "before",
    "must",
    "through",
    "back",
    "years",
    "where",
    "much",
    "your",
    "way",
    "well",
    "down",
    "should",
    "because",
    "each",
    "just",
    "those",
    "people",
    "mr",
    "how",
    "too",
    "little",
    "state",
    "good",
    "very",
    "make",
    "world",
    "still",
    "own",
    "see",
    "men",
    "work",
    "long",
    "get",
    "here",
    "between",
    "both",
    "life",
    "being",
    "under",
    "never",
    "day",
    "same",
    "another",
    "know",
    "while",
    "last",
    "might",
    "us",
    "off",
    "go",
    "without",
    "again",
    "come",
    "used",
    "take",
    "since",
    "against",
    "himself",
    "few",
    "during",
    "always",
    "rather",
    "whether",
    "really",
    "something",
    "yet",
    "upon",
    "though",
    "however",
    "therefore",
    "perhaps",
    "almost",
    "enough",
    "among",
    "across",
    "within",
    "along",
    "toward",
    "towards",
)

# Approximate unigram mass of these closed-class items in general
# English, renormalized over this list. Order-of-magnitude rates follow
# published English function-word tables (e.g. Norvig / Google 1T scale
# for the top items; the tail is flattened). Not a substitute for the
# full Brown corpus — see DATASET.md / README for the limitation.
_RAW_REF = {
    "the": 7.0,
    "of": 3.5,
    "and": 3.0,
    "to": 2.9,
    "a": 2.6,
    "in": 2.1,
    "that": 1.3,
    "is": 1.1,
    "was": 1.0,
    "he": 0.9,
    "for": 0.9,
    "it": 0.9,
    "with": 0.8,
    "as": 0.8,
    "his": 0.7,
    "on": 0.7,
    "be": 0.7,
    "at": 0.6,
    "by": 0.6,
    "i": 0.6,
    "this": 0.55,
    "had": 0.5,
    "not": 0.5,
    "are": 0.5,
    "but": 0.48,
    "from": 0.47,
    "or": 0.42,
    "have": 0.42,
    "an": 0.4,
    "they": 0.4,
    "which": 0.38,
    "you": 0.38,
    "were": 0.36,
    "her": 0.35,
    "their": 0.34,
    "we": 0.33,
    "there": 0.3,
    "been": 0.28,
    "if": 0.27,
    "would": 0.25,
    "when": 0.24,
    "will": 0.24,
    "who": 0.22,
    "so": 0.22,
    "my": 0.2,
    "your": 0.18,
    "because": 0.12,
    "however": 0.08,
    "therefore": 0.04,
    "just": 0.14,
    "really": 0.08,
    "though": 0.1,
    "perhaps": 0.06,
}

_WORD = re.compile(r"[A-Za-z0-9']+")


def _ref_vector() -> list[float]:
    vec = [_RAW_REF.get(w, 0.05) for w in FUNCTION_WORDS]
    s = sum(vec) or 1.0
    return [x / s for x in vec]


_REF = _ref_vector()


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    sim = max(-1.0, min(1.0, dot / (na * nb)))
    return 1.0 - sim


def function_word_distance(text: str) -> tuple[float, str]:
    tokens = [t.lower() for t in _WORD.findall(text)]
    counts = {w: 0.0 for w in FUNCTION_WORDS}
    for t in tokens:
        if t in counts:
            counts[t] += 1.0
    total = sum(counts.values())
    if total < 8:
        dist = 0.0
        return dist, "Too few function words to compare against the reference."
    vec = [counts[w] / total for w in FUNCTION_WORDS]
    dist = _cosine_distance(vec, _REF)
    return dist, fw_cosine_text(dist)
