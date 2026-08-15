"""Instance-level plain-English readings of measured features.

The UI never shows raw feature names as the primary explanation.
These strings are the product: they say *what was unusual* relative
to this essay, not what the internal column is called.
"""

from __future__ import annotations


def nll_z_text(z: float) -> str:
    if z <= -1.15:
        return "Unusually smooth predictability compared to the rest of this essay."
    if z >= 1.15:
        return "More surprising, uneven wording than the rest of this essay."
    return "Predictability is close to this essay's own baseline."


def burstiness_text(var: float, typical: float = 0.35) -> str:
    if var < typical * 0.45:
        return "Predictability is unusually even across sentences — human drafts usually vary more."
    if var > typical * 1.8:
        return "Predictability swings a lot from sentence to sentence, more like bursty human drafting."
    return "Sentence-to-sentence predictability varies within a typical range."


def length_var_text(var: float, mean: float) -> str:
    if mean <= 0:
        return "Not enough sentence-length information to judge."
    cv = (var**0.5) / mean if mean else 0.0
    if cv < 0.22:
        return "Sentence lengths are far more uniform than typical human drafting."
    if cv > 0.55:
        return "Sentence lengths vary widely, which is common in human drafts."
    return "Sentence-length variety is moderate."


def length_z_text(z: float) -> str:
    if abs(z) < 0.35:
        return "This sentence's length sits very close to the essay's average (uniform pacing)."
    if z > 1.2:
        return "This sentence is much longer than the essay's typical sentence."
    if z < -1.2:
        return "This sentence is much shorter than the essay's typical sentence."
    return "Sentence length is not an outlier relative to this essay."


def ttr_text(ttr: float) -> str:
    if ttr < 0.42:
        return "Vocabulary in this window repeats more than usual (lower lexical variety)."
    if ttr > 0.72:
        return "This window uses a relatively wide range of word types."
    return "Lexical variety in this window is unremarkable."


def fw_cosine_text(dist: float) -> str:
    if dist > 0.28:
        return "Function-word mix diverges from general English more than usual."
    if dist < 0.08:
        return "Function-word mix is very close to generic English (a smoothness some models produce)."
    return "Function-word mix is reasonably close to general English."


def pos_entropy_text(entropy: float) -> str:
    if entropy < 2.15:
        return "Syntactic patterns (part-of-speech sequences) look repetitive."
    if entropy > 3.1:
        return "Syntactic patterns are relatively varied."
    return "Syntactic variety is in a middle range."


def transition_text(density: float, count: float) -> str:
    if count >= 1 and density >= 0.04:
        return 'This sentence uses discourse glue such as "moreover," "furthermore," or "in conclusion."'
    if density >= 0.035:
        return "Transition-word density is high relative to typical human drafting."
    if density < 0.008:
        return "Few formulaic transition markers in this span."
    return "Transition-word use is moderate."


def embed_sim_text(sim: float) -> str:
    if sim >= 0.78:
        return "This sentence sits unusually close, in meaning-space, to its neighbor (very smooth topic flow)."
    if sim <= 0.28:
        return "This sentence jumps away from its neighbor more than a tightly polished paragraph would."
    return "Topic flow to the neighboring sentence is moderate."


def embed_mean_text(mean_sim: float) -> str:
    if mean_sim >= 0.72:
        return "Consecutive sentences stay unusually close in meaning across the whole essay."
    if mean_sim <= 0.38:
        return "Topic flow jumps around more, which is common in human drafts."
    return "Overall sentence-to-sentence meaning flow is moderate."


def fk_z_text(z: float) -> str:
    if abs(z) < 0.35:
        return "Reading level of this sentence matches the essay's average almost exactly."
    if z > 1.1:
        return "This sentence is denser / more complex than the rest of the essay."
    if z < -1.1:
        return "This sentence is simpler than the rest of the essay."
    return "Readability is not an outlier relative to this essay."


def fk_var_text(var: float) -> str:
    if var < 1.2:
        return "Reading level barely changes from sentence to sentence (unusually even)."
    if var > 8.0:
        return "Reading level swings between sentences, more like mixed human drafting."
    return "Readability varies moderately across the essay."
