"""
punctuation.py — Punctuation & stylistic irregularity signal

HOW IT WORKS (plain English):
  We count the presence of "human" punctuation and stylistic elements, such as:
  - Em-dashes (—)
  - Semicolons (;)
  - Parenthetical asides: (like this)
  - Ellipses (...)
  - Exclamation marks (!), rhetorical contractions (can't, wouldn't)

WHY THIS DETECTS AI:
  AI language models tend to generate highly standardized punctuation.
  They rely heavily on commas and full stops, and avoid informal or expressive
  punctuation unless explicitly instructed. Humans write with stylistic idiosyncrasies:
  parentheticals, em-dashes for dramatic pauses, and casual contractions.

LIMITATIONS:
  Strictly formal academic prose edited by a human might also use formal punctuation
  and avoid contractions or em-dashes. Therefore, this signal contributes a lower weight
  to the overall scoring system.
"""

import re

# Human punctuation markers that AI rarely produces naturally in standard essays
HUMAN_PUNCTUATION_PATTERNS = [
    (r"—|--", "Em-dash / conversational dash"),
    (r";", "Semicolon for compound thought"),
    (r"\([^\)]+\)", "Parenthetical aside"),
    (r"\.\.\.", "Ellipsis"),
    (r"!", "Exclamation mark"),
]


def compute_punctuation_score(sentence: str) -> dict:
    """
    Evaluates punctuation style for a sentence.

    Returns:
        score   — AI probability in [0, 1]. Lower = more human-like.
        markers — List of identified human-style punctuation markers.
    """
    matched_markers = []
    for pattern, description in HUMAN_PUNCTUATION_PATTERNS:
        if re.search(pattern, sentence):
            matched_markers.append(description)

    # Base score of 0.5 (neutral)
    # Each human punctuation marker reduces the AI score by 0.2
    score = 0.5 - (len(matched_markers) * 0.20)
    score = float(max(0.0, min(1.0, score)))

    if matched_markers:
        note = f"Human stylistic markers found: {', '.join(matched_markers)}"
    else:
        note = "Standard formal punctuation (neutral/AI-leaning)."

    return {
        "score": round(score, 4),
        "markers": matched_markers,
        "count": len(matched_markers),
        "note": note,
    }
