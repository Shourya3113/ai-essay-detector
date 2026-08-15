"""Consecutive-sentence embedding coherence.

Linguistic reasoning
--------------------
Real drafts change register and topic inside a paragraph: a smell, then
a rule, then a joke. LLM paragraphs — and especially "polish this"
rewrites — keep a tight semantic heading, so cosine similarity between
neighboring sentence embeddings stays high and stable. We encode
sentences with a small MiniLM model (or a bag-of-words fallback) and
measure neighbor cosine similarity. High mean similarity across the
essay is the global signal; a locally very-similar pair is the span
signal.

This is measurement of geometry, not an LLM opinion about authorship.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from features.base import SentenceSpan, safe_mean
from features.plain_english import embed_mean_text, embed_sim_text

_WORD = re.compile(r"[A-Za-z0-9']+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _bow_vectors(sentences: list[SentenceSpan]) -> list[list[float]]:
    tokenized = [_WORD.findall(s.text.lower()) for s in sentences]
    vocab = sorted({t for sent in tokenized for t in sent})
    index = {t: i for i, t in enumerate(vocab)}
    vecs: list[list[float]] = []
    for sent in tokenized:
        v = [0.0] * len(vocab)
        for t, c in Counter(sent).items():
            v[index[t]] = float(c)
        vecs.append(v)
    return vecs


def _minilm_vectors(texts: list[str], embedder) -> list[list[float]]:
    arr = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [row.tolist() for row in arr]


def embedding_coherence(
    sentences: list[SentenceSpan],
    embedder=None,
) -> tuple[list[float], list[float], float, list[str], str, str]:
    """Return prev-sim, next-sim, mean sim, per-sentence expl, essay expl, backend name."""
    n = len(sentences)
    if n == 0:
        return [], [], 0.0, [], embed_mean_text(0.0), "none"
    if n == 1:
        return [0.0], [0.0], 0.0, [embed_sim_text(0.0)], embed_mean_text(0.0), "none"

    backend = "bow"
    if embedder is not None:
        try:
            vecs = _minilm_vectors([s.text for s in sentences], embedder)
            backend = "minilm"
        except Exception:
            vecs = _bow_vectors(sentences)
    else:
        vecs = _bow_vectors(sentences)

    pair_sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(n - 1)]
    prev = [0.0] + pair_sims
    nxt = pair_sims + [0.0]
    mean_sim = safe_mean(pair_sims)
    expl = []
    for i in range(n):
        local = pair_sims[i - 1] if i > 0 else (pair_sims[0] if pair_sims else 0.0)
        expl.append(embed_sim_text(local))
    return prev, nxt, mean_sim, expl, embed_mean_text(mean_sim), backend
