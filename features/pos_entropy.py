"""POS n-gram entropy — how repetitive are syntactic patterns?

Linguistic reasoning
--------------------
Human writers mix clause shapes: fragments, questions, stacked
subordinates, sudden short verbs. LLM prose often falls into a small
set of templates (PP-heavy openers, "It was X that Y", coordinated
adjectives). We tag tokens, build POS bigrams, and compute Shannon
entropy. Lower entropy = more repetitive syntax. spaCy tags are
preferred; a coarse closed-class fallback keeps the function testable
without a model download.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from features.base import SentenceSpan
from features.plain_english import pos_entropy_text

_WORD = re.compile(r"[A-Za-z0-9']+")

_DET = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "his", "her", "their"}
_ADP = {"in", "on", "at", "by", "for", "with", "from", "to", "of", "into", "about", "over", "after", "before", "between", "through", "during", "without", "among"}
_PRON = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "myself", "who", "whom", "what"}
_CONJ = {"and", "or", "but", "nor", "so", "yet"}
_AUX = {"is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "should", "may", "might", "must"}
_ADV = {"not", "never", "always", "often", "really", "just", "even", "too", "very", "also"}


def coarse_pos(word: str) -> str:
    w = word.lower()
    if w in _DET:
        return "DET"
    if w in _PRON:
        return "PRON"
    if w in _ADP:
        return "ADP"
    if w in _CONJ:
        return "CCONJ"
    if w in _AUX:
        return "AUX"
    if w in _ADV or w.endswith("ly"):
        return "ADV"
    if w.endswith("ing") or w.endswith("ed"):
        return "VERB"
    if w.endswith("s") and len(w) > 3:
        return "NOUN"
    return "NOUN"


def _tags_for_sentence(text: str, nlp=None) -> list[str]:
    if nlp is not None and hasattr(nlp, "pipe_names") and "tagger" in nlp.pipe_names:
        doc = nlp(text)
        return [t.pos_ for t in doc if not t.is_space]
    return [coarse_pos(w) for w in _WORD.findall(text)]


def _bigram_entropy(tags: list[str]) -> float:
    if len(tags) < 2:
        return 0.0
    grams = [f"{a}_{b}" for a, b in zip(tags, tags[1:])]
    counts = Counter(grams)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p, 2)
    return ent


def pos_entropy(
    sentences: list[SentenceSpan],
    nlp=None,
) -> tuple[float, list[float], str, list[str]]:
    all_tags: list[str] = []
    local: list[float] = []
    local_expl: list[str] = []
    for sent in sentences:
        tags = _tags_for_sentence(sent.text, nlp)
        all_tags.extend(tags)
        val = _bigram_entropy(tags)
        local.append(val)
        local_expl.append(pos_entropy_text(val))
    essay_val = _bigram_entropy(all_tags)
    return essay_val, local, pos_entropy_text(essay_val), local_expl
