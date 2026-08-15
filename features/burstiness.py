"""Burstiness: variance of per-sentence surprisal, plus within-essay z-scores.

Linguistic reasoning
--------------------
Human writing is bursty: a specific anecdote (high surprisal) sits next
to a plain transition (low surprisal). Many LLM samples are flatter —
each sentence is about as predictable as the last. We measure that as
the variance of per-sentence NLL.

Flagging is *relative to this essay*, not a global cutoff. A sentence
with a strongly negative NLL z-score is smoother than the rest of *this*
draft. That is the quantity the UI can honestly explain.
"""

from __future__ import annotations

from features.base import safe_var, zscores
from features.plain_english import burstiness_text, nll_z_text


def burstiness(nlls: list[float]) -> tuple[float, str]:
    var = safe_var(nlls)
    return var, burstiness_text(var)


def nll_zscores(nlls: list[float]) -> tuple[list[float], list[str]]:
    zs = zscores(nlls)
    return zs, [nll_z_text(z) for z in zs]
