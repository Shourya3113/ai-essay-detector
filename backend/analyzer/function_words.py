"""
function_words.py — Function word ratio and entropy distribution signal

HOW IT WORKS (plain English):
  We analyze the proportion and variety of "function words" (grammatical glue words
  like articles, prepositions, conjunctions, pronouns, and auxiliary verbs).
  We compute two core metrics:
    1. Function Word Ratio = (total function words) / (total words)
    2. Function Word Entropy = Shannon entropy of the function word frequency distribution.

WHY THIS DETECTS AI:
  AI language models rely on highly standard syntactic scaffolds, leading to a consistently
  elevated and uniform function word ratio (~48%-56%) and very predictable distribution entropy.
  Human admissions essays, by contrast, swing between vivid, noun-and-verb heavy storytelling
  (low function word ratio) and conversational or reflective introspection (variable entropy).

LIMITATIONS:
  Very short sentences (< 5 words) lack sufficient token count for stable entropy calculation.
"""

import math
import re
from collections import Counter

# Standard English grammatical function words (articles, prepositions, conjunctions, pronouns, auxiliaries)
FUNCTION_WORDS: set[str] = {
    # Articles
    "a", "an", "the",
    # Prepositions
    "about", "above", "across", "after", "against", "along", "among", "around", "at",
    "before", "behind", "below", "beneath", "beside", "between", "beyond", "by",
    "down", "during", "except", "for", "from", "in", "inside", "into", "near", "of",
    "off", "on", "onto", "out", "outside", "over", "past", "since", "through",
    "throughout", "to", "toward", "towards", "under", "underneath", "until", "up",
    "upon", "with", "within", "without",
    # Conjunctions
    "and", "but", "or", "nor", "for", "yet", "so", "although", "because", "since",
    "unless", "while", "whereas", "if", "whether", "as", "though", "even", "provided",
    # Pronouns
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "whose", "anybody", "anyone", "anything", "each", "everyone", "everybody",
    "everything", "nobody", "none", "nothing", "somebody", "someone", "something",
    # Auxiliary & Modal Verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    # Adverbs / Particles acting as function markers
    "not", "no", "also", "there", "here", "too", "very", "just", "then", "now",
}


def compute_function_words(text: str) -> dict:
    """
    Compute the function word ratio and distribution entropy of the input text.

    Returns:
        score               — AI probability in [0, 1]. Higher = more AI-like.
        ratio               — Ratio of function words to total words [0, 1].
        entropy             — Shannon entropy of function word frequencies.
        function_word_count — Total number of function words matched.
        total_words         — Total valid alphabetic words.
        note                — Diagnostic human-readable interpretation.
    """
    if not text or not text.strip():
        return {
            "score": 0.5,
            "ratio": 0.0,
            "entropy": 0.0,
            "function_word_count": 0,
            "total_words": 0,
            "note": "Empty text provided.",
        }

    words = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text.lower())
    total_words = len(words)

    if total_words < 5:
        return {
            "score": 0.5,
            "ratio": 0.0,
            "entropy": 0.0,
            "function_word_count": 0,
            "total_words": total_words,
            "note": "Too few words to evaluate function word distribution.",
        }

    fw_tokens = [w for w in words if w in FUNCTION_WORDS]
    fw_count = len(fw_tokens)
    ratio = fw_count / total_words

    # Calculate Shannon Entropy over the distribution of function words
    if fw_count > 0:
        fw_counts = Counter(fw_tokens)
        entropy = 0.0
        for count in fw_counts.values():
            p = count / fw_count
            entropy -= p * math.log2(p)
    else:
        entropy = 0.0

    # --- Normalisation ---
    # Empirical observations in college admissions essays:
    #   AI-generated text:
    #     - ratio typically in [0.47, 0.58]
    #     - entropy in standard essays: [3.2, 4.8] with uniform spread
    #   Human text:
    #     - ratio can be lower (< 0.40) in descriptive/anecdotal prose or higher in dialogue/rambling
    #     - high variance between narrative chunks

    if 0.46 <= ratio <= 0.58:
        ratio_score = 0.75 + (0.25 * (1.0 - abs(ratio - 0.52) / 0.06))
    elif ratio > 0.58:
        ratio_score = 0.60
    elif ratio >= 0.38:
        ratio_score = 0.35 + (0.35 * (ratio - 0.38) / 0.08)
    else:
        ratio_score = max(0.05, (ratio / 0.38) * 0.30)

    # Entropy component
    if fw_count >= 10:
        unique_fw = len(Counter(fw_tokens))
        max_entropy = math.log2(unique_fw) if unique_fw > 1 else 1.0
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.5
        entropy_score = float(max(0.0, min(1.0, (entropy_ratio - 0.65) / 0.30)))
    else:
        entropy_score = 0.5

    # Blended score
    score = (0.65 * ratio_score) + (0.35 * entropy_score)
    score = float(max(0.0, min(1.0, score)))

    if ratio >= 0.48 and entropy >= 3.4:
        note = f"Function word ratio {ratio:.1%} (entropy {entropy:.2f}) — elevated & uniform (AI-like)"
    elif ratio < 0.40:
        note = f"Function word ratio {ratio:.1%} (entropy {entropy:.2f}) — content-dense storytelling (human-like)"
    else:
        note = f"Function word ratio {ratio:.1%} (entropy {entropy:.2f}) — natural distribution"

    return {
        "score": round(score, 4),
        "ratio": round(ratio, 4),
        "entropy": round(entropy, 4),
        "function_word_count": fw_count,
        "total_words": total_words,
        "note": note,
    }
