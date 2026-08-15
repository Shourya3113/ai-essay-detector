"""
burstiness.py — Sentence length burstiness signal

HOW IT WORKS (plain English):
  We measure how much the sentence lengths vary across the whole essay.
  We use the Coefficient of Variation (CV) = standard deviation / mean length.
  A high CV means lengths jump around a lot — that's called "bursty."
  A low CV means all sentences are roughly the same length — "uniform."

WHY THIS DETECTS AI:
  When humans write, they naturally mix punchy 5-word sentences with sprawling
  35-word ones. It's not deliberate — it's how thought flows onto the page.
  AI models are optimised to produce readable, polished prose, which means they
  unconsciously regulate sentence length. The result is suspiciously uniform.

  This is a DOCUMENT-LEVEL signal — every sentence in the essay gets the same
  burstiness score because it reflects the essay as a whole, not one sentence.

WHAT IT CANNOT DO:
  A human who edits obsessively (or follows a rigid style guide) may score
  AI-like here. A short essay (< 5 sentences) has too little data.
"""

import statistics


def compute_burstiness(sentences: list[str]) -> dict:
    """
    Compute the burstiness (sentence length variance) of the full essay.

    Returns:
        score    — AI probability in [0, 1]. Higher = more uniform = more AI-like.
        cv       — Coefficient of Variation (raw value, for display).
        std_dev  — Standard deviation of sentence lengths in words.
        mean_len — Mean sentence length in words.
    """
    if len(sentences) < 4:
        # Not enough sentences to measure variance reliably
        return {
            "score": 0.5,
            "cv": None,
            "std_dev": None,
            "mean_len": None,
            "note": "Too few sentences to measure burstiness.",
        }

    lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(lengths)

    if mean_len < 1:
        return {
            "score": 0.5,
            "cv": None,
            "std_dev": None,
            "mean_len": 0,
            "note": "Sentences appear empty.",
        }

    std_dev = statistics.stdev(lengths)

    # Coefficient of Variation — normalises std_dev by mean
    # so a 3-word variation in a short essay is treated proportionally
    # the same as a 6-word variation in a long essay.
    cv = std_dev / mean_len

    # --- Normalisation ---
    # CV < 0.25 → very uniform → very AI-like → score near 1.0
    # CV = 0.50 → moderate variance → score near 0.50
    # CV > 0.75 → very bursty → very human-like → score near 0.0
    #
    # Linear mapping: score = clamp((0.75 - cv) / 0.75, 0, 1)
    score = float(max(0.0, min(1.0, (0.75 - cv) / 0.75)))

    if cv < 0.25:
        note = f"CV {cv:.2f} — highly uniform sentence lengths (AI-like)"
    elif cv < 0.50:
        note = f"CV {cv:.2f} — moderately varied sentence lengths"
    else:
        note = f"CV {cv:.2f} — varied sentence lengths (human-like)"

    return {
        "score": round(score, 4),
        "cv": round(cv, 3),
        "std_dev": round(std_dev, 2),
        "mean_len": round(mean_len, 2),
        "note": note,
    }
