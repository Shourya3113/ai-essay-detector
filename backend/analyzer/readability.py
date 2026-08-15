"""
readability.py — Flesch-Kincaid Grade Level and Readability Variance Signal

HOW IT WORKS (plain English):
  We calculate the Flesch-Kincaid Grade Level and Flesch Reading Ease for each individual sentence
  and across the entire essay. We measure the standard deviation / variance of readability scores
  across all sentences in the essay.

WHY THIS DETECTS AI:
  When humans write personal admissions essays, their cognitive flow produces dramatic readability
  shifts: simple, punchy narrative hooks ("I was terrified.") sit alongside complex, compound
  philosophical reflections ("Notwithstanding our initial setback, the experience catalyzed...").
  This creates high sentence-by-sentence readability variance (Std Dev > 3.5).
  AI models, by contrast, output uniformly regulated text at a consistent ~10th–13th grade level
  across almost every sentence, resulting in monotone, flat readability variance (Std Dev < 1.8).

LIMITATIONS:
  Single-sentence inputs cannot compute cross-sentence variance.
"""

import math
import re
import statistics


def count_syllables(word: str) -> int:
    """
    Estimate the number of syllables in an English word using rule-based morphology.
    """
    w = word.lower().strip()
    if not w:
        return 1

    # Remove non-alphabetic characters
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 1

    if len(w) <= 3:
        return 1

    # Special handling for trailing 'e', 'es', 'ed'
    w_clean = w
    if w_clean.endswith("e") and not w_clean.endswith(("le", "ee", "oe")):
        w_clean = w_clean[:-1]
    elif w_clean.endswith("ed") and not w_clean.endswith(("ted", "ded")):
        w_clean = w_clean[:-2]
    elif w_clean.endswith("es") and not w_clean.endswith(("ses", "zes", "ches", "shes")):
        w_clean = w_clean[:-2]

    # Count consecutive vowel groups
    vowel_groups = re.findall(r"[aeiouy]+", w_clean)
    count = len(vowel_groups)

    # Diphthong / exception adjustments
    if re.search(r"[^aeiouy]le$", w) and not re.search(r"[aeiouy]le$", w):
        count += 1

    return max(1, count)


def compute_sentence_readability(sentence: str) -> dict:
    """Compute Flesch-Kincaid Grade Level and Reading Ease for a single sentence."""
    if not sentence or not sentence.strip():
        return {
            "fk_grade": 8.0,
            "reading_ease": 60.0,
            "words": 0,
            "syllables": 0,
        }

    words = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", sentence.lower())
    n_words = len(words)
    if n_words == 0:
        return {
            "fk_grade": 8.0,
            "reading_ease": 60.0,
            "words": 0,
            "syllables": 0,
        }

    n_syllables = sum(count_syllables(w) for w in words)
    syllables_per_word = n_syllables / n_words

    # Single-sentence Flesch-Kincaid formula (sentence count = 1)
    fk_grade = (0.39 * n_words) + (11.8 * syllables_per_word) - 15.59
    fk_grade = max(1.0, min(22.0, fk_grade))

    # Flesch Reading Ease
    reading_ease = 206.835 - (1.015 * n_words) - (84.6 * syllables_per_word)
    reading_ease = max(0.0, min(100.0, reading_ease))

    return {
        "fk_grade": round(fk_grade, 2),
        "reading_ease": round(reading_ease, 2),
        "words": n_words,
        "syllables": n_syllables,
    }


def compute_readability(text_or_sentences: str | list[str]) -> dict:
    """
    Compute document-level Flesch-Kincaid Grade Level and cross-sentence readability variance.

    Returns:
        score           — AI probability in [0, 1]. Higher = more monotone = more AI-like.
        fk_grade        — Overall document Flesch-Kincaid grade level.
        reading_ease    — Overall document Flesch Reading Ease.
        variance        — Variance of sentence grade levels.
        std_dev         — Standard deviation of sentence grade levels.
        sentence_grades — List of individual sentence grade levels.
        note            — Human-readable diagnostic.
    """
    if isinstance(text_or_sentences, list):
        sentences = [s.strip() for s in text_or_sentences if len(s.strip()) > 2]
    else:
        normalized = re.sub(r"\n+", " ", str(text_or_sentences).strip())
        normalized = re.sub(r"\s{2,}", " ", normalized)
        raw_sents = re.split(r"(?<=[.!?])\s+", normalized)
        sentences = [s.strip() for s in raw_sents if len(s.strip()) > 2]

    if not sentences:
        return {
            "score": 0.5,
            "fk_grade": None,
            "reading_ease": None,
            "variance": None,
            "std_dev": None,
            "sentence_grades": [],
            "note": "No valid text for readability analysis.",
        }

    # Evaluate each sentence
    sentence_grades = []
    total_words = 0
    total_syllables = 0

    for s in sentences:
        res = compute_sentence_readability(s)
        sentence_grades.append(res["fk_grade"])
        total_words += res["words"]
        total_syllables += res["syllables"]

    if total_words == 0:
        return {
            "score": 0.5,
            "fk_grade": None,
            "reading_ease": None,
            "variance": None,
            "std_dev": None,
            "sentence_grades": [],
            "note": "Text contains no recognizable words.",
        }

    n_sentences = len(sentences)
    avg_words_per_sent = total_words / n_sentences
    avg_syllables_per_word = total_syllables / total_words

    doc_fk_grade = (0.39 * avg_words_per_sent) + (11.8 * avg_syllables_per_word) - 15.59
    doc_fk_grade = float(max(1.0, min(22.0, doc_fk_grade)))

    doc_reading_ease = 206.835 - (1.015 * avg_words_per_sent) - (84.6 * avg_syllables_per_word)
    doc_reading_ease = float(max(0.0, min(100.0, doc_reading_ease)))

    # Compute sentence-level variance if >= 3 sentences
    if len(sentence_grades) >= 3:
        variance = float(statistics.variance(sentence_grades))
        std_dev = float(statistics.stdev(sentence_grades))

        # --- Normalisation ---
        # AI essays: monotone grade distribution (std_dev ~ 1.0 – 2.0)
        # Human essays: dynamic pacing / variance (std_dev ~ 3.5 – 6.5+)
        #
        # Linear mapping: std_dev < 1.8 -> AI score near 1.0
        #                 std_dev > 4.2 -> human score near 0.0
        score = (4.2 - std_dev) / (4.2 - 1.6)
        score = float(max(0.0, min(1.0, score)))

        if std_dev < 1.8:
            note = f"FK Grade {doc_fk_grade:.1f} with flat sentence variance (Std Dev {std_dev:.2f}) — monotone pacing (AI-like)"
        elif std_dev < 3.2:
            note = f"FK Grade {doc_fk_grade:.1f} with moderate variance (Std Dev {std_dev:.2f})"
        else:
            note = f"FK Grade {doc_fk_grade:.1f} with high sentence variance (Std Dev {std_dev:.2f}) — dynamic storytelling (human-like)"
    else:
        variance = 0.0
        std_dev = 0.0
        score = 0.5
        note = f"FK Grade {doc_fk_grade:.1f} — too few sentences to measure cross-sentence variance."

    return {
        "score": round(score, 4),
        "fk_grade": round(doc_fk_grade, 2),
        "reading_ease": round(doc_reading_ease, 2),
        "variance": round(variance, 3),
        "std_dev": round(std_dev, 3),
        "sentence_grades": [round(g, 2) for g in sentence_grades],
        "note": note,
    }
