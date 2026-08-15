"""
transitions.py — Discourse marker & academic transition density signal

HOW IT WORKS (plain English):
  We scan for formal academic transition markers, connective boilerplate, and discourse
  signposts that AI systems disproportionately insert to enforce rhetorical coherence.
  We calculate:
    1. Transition Density = (total matched transitions / total words) * 100
    2. Transition Clustering = whether transitions occur sequentially across adjacent sentences
       or multiple times within single paragraphs.

WHY THIS DETECTS AI:
  LLMs are heavily aligned toward "structured, easy-to-follow" argumentation. When generating
  or polishing essays, they persistently inject overt connectives ("furthermore", "moreover",
  "consequently", "it is important to note", "delve into", "a testament to", "in conclusion").
  Human applicants usually rely on narrative flow, chronology, or subtle thematic links rather
  than rigid textbook signposts.

LIMITATIONS:
  ESL writers or students adhering to rigid five-paragraph essay formulas may use standard
  transitions more frequently.
"""

import re

# Comprehensive catalog of AI-favored discourse markers & transitional phrases
TRANSITION_PATTERNS: list[tuple[str, str, str]] = [
    # Format: (pattern, canonical_name, category)
    (r"\bfurthermore\b", "furthermore", "additive"),
    (r"\bmoreover\b", "moreover", "additive"),
    (r"\bin addition\b", "in addition", "additive"),
    (r"\badditionally\b", "additionally", "additive"),
    (r"\bconsequently\b", "consequently", "causal"),
    (r"\bsubsequently\b", "subsequently", "sequential"),
    (r"\bin conclusion\b", "in conclusion", "structural"),
    (r"\bin summary\b", "in summary", "structural"),
    (r"\bto summarize\b", "to summarize", "structural"),
    (r"\bto conclude\b", "to conclude", "structural"),
    (r"\bin essence\b", "in essence", "reductive"),
    (r"\bthus\b", "thus", "causal"),
    (r"\bhence\b", "hence", "causal"),
    (r"\bnotably\b", "notably", "emphatic"),
    (r"\bsignificantly\b", "significantly", "emphatic"),
    (r"\bimportantly\b", "importantly", "emphatic"),
    (r"\bultimately\b", "ultimately", "conclusive"),
    (r"\bnonetheless\b", "nonetheless", "adversative"),
    (r"\bnevertheless\b", "nevertheless", "adversative"),
    (r"\bit is important to note\b", "it is important to note", "hedging"),
    (r"\bit is worth noting\b", "it is worth noting", "hedging"),
    (r"\bit is crucial to\b", "it is crucial to", "hedging"),
    (r"\bit is essential to\b", "it is essential to", "hedging"),
    (r"\bit is evident that\b", "it is evident that", "hedging"),
    (r"\bit is clear that\b", "it is clear that", "hedging"),
    (r"\bdelve into\b|\bdelves into\b|\bdelving into\b", "delve into", "ai_cliche"),
    (r"\bdive deep into\b|\bdiving deep into\b", "dive deep into", "ai_cliche"),
    (r"\ba testament to\b|\btestament to\b", "testament to", "ai_cliche"),
    (r"\btapestry of\b", "tapestry of", "ai_cliche"),
    (r"\bmultifaceted\b", "multifaceted", "ai_cliche"),
    (r"\bnuanced\b", "nuanced", "ai_cliche"),
    (r"\bnavigating the complexities\b|\bnavigate the complexities\b", "navigate complexities", "ai_cliche"),
    (r"\bplays a (?:pivotal|crucial|vital|key) role\b", "plays a pivotal role", "ai_cliche"),
    (r"\bserve as a beacon\b|\bbeacon of\b", "beacon of", "ai_cliche"),
    (r"\bcornerstone of\b", "cornerstone of", "ai_cliche"),
    (r"\bfoster a (?:deeper|greater) understanding\b", "foster deeper understanding", "ai_cliche"),
    (r"\bembark on a journey\b|\bembarked on a journey\b", "embark on a journey", "ai_cliche"),
    (r"\bin the realm of\b", "in the realm of", "ai_cliche"),
]


def compute_transition_score(text: str) -> dict:
    """
    Evaluate the density and clustering of AI-favored transition phrases.

    Returns:
        score                  — AI probability in [0, 1].
        count                  — Total transition matches found.
        density_per_100_words  — Normalized transition density.
        matched                — Detailed list of matched patterns.
        is_clustered           — Whether transitions appear in high density / clustering.
        note                   — Human-readable summary.
    """
    if not text or not text.strip():
        return {
            "score": 0.5,
            "count": 0,
            "density_per_100_words": 0.0,
            "matched": [],
            "is_clustered": False,
            "note": "Empty text provided.",
        }

    words = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text.lower())
    total_words = len(words)

    if total_words < 5:
        return {
            "score": 0.5,
            "count": 0,
            "density_per_100_words": 0.0,
            "matched": [],
            "is_clustered": False,
            "note": "Too few words to evaluate transition density.",
        }

    lower_text = text.lower()
    matched_items = []

    for pattern, name, category in TRANSITION_PATTERNS:
        matches = list(re.finditer(pattern, lower_text))
        for m in matches:
            matched_items.append({
                "phrase": name,
                "category": category,
                "position": m.start(),
            })

    total_matches = len(matched_items)
    density = (total_matches / total_words) * 100.0

    # Check for clustering: multiple transitions occurring within a 50-word window
    is_clustered = False
    if total_matches >= 2:
        sorted_pos = sorted(m["position"] for m in matched_items)
        for i in range(len(sorted_pos) - 1):
            if sorted_pos[i+1] - sorted_pos[i] < 200:  # within ~35-40 chars
                is_clustered = True
                break

    # --- Normalisation ---
    # Density ranges in 300-500 word essays:
    #   Natural human essay: 0 to 1.2 transitions per 100 words (score < 0.3)
    #   Moderate / ESL:      1.2 to 2.2 transitions per 100 words (score ~ 0.45 - 0.65)
    #   AI-generated text:   2.5+ transitions per 100 words + AI cliches (score > 0.80)
    base_score = min(1.0, density / 2.8)
    if is_clustered:
        base_score = min(1.0, base_score + 0.15)

    score = float(max(0.0, min(1.0, base_score)))

    if total_matches == 0:
        note = "No overt AI transition markers detected (human-like narrative flow)."
    elif density > 2.0 or is_clustered:
        unique_phrases = sorted(list({m['phrase'] for m in matched_items}))[:4]
        note = f"{total_matches} transition markers ({density:.1f}/100w, e.g., {', '.join(unique_phrases)}) — heavy signposting (AI-like)"
    else:
        unique_phrases = sorted(list({m['phrase'] for m in matched_items}))[:3]
        note = f"{total_matches} transition marker(s) ({density:.1f}/100w, e.g., {', '.join(unique_phrases)}) — standard usage"

    return {
        "score": round(score, 4),
        "count": total_matches,
        "density_per_100_words": round(density, 2),
        "matched": matched_items,
        "is_clustered": is_clustered,
        "note": note,
    }
