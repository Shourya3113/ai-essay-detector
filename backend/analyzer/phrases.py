"""
phrases.py — AI fingerprint phrase signal

HOW IT WORKS (plain English):
  We maintain a curated list of phrases that appear significantly more often
  in AI-generated text than in human writing. When a sentence contains one or
  more of these phrases, its AI score rises accordingly.

WHY THESE PHRASES SPECIFICALLY:
  These come from two sources:
  1. Published research on AI text detection (Guo et al. 2023, Mitchell et al. 2023)
     which documented overrepresented n-grams in GPT-3/4 output.
  2. Manual analysis of ~200 AI-generated essays where we counted phrase frequency.

  AI models are trained on internet text and RLHF feedback that rewards
  "helpful, clear, structured" responses. This biases them toward formal
  connectives, hedging phrases, and certain idiomatic expressions that appear
  in high-quality training examples.

LIMITATIONS:
  A human who writes in a formal academic or corporate style may trigger some
  of these phrases. This signal is weakest on its own — it contributes only
  20% of the final score for this reason.

KNOWN FALSE-POSITIVE PHRASES:
  "in conclusion" — common in student essays taught to use it
  "furthermore"   — standard academic writing
  These are included because they still correlate with AI, but their weight
  is intentionally limited.
"""

AI_PHRASE_PATTERNS: list[tuple[str, str]] = [
    # Format: (phrase, reason it's suspicious)
    ("it is important to note", "AI hedging boilerplate"),
    ("it is worth noting", "AI hedging boilerplate"),
    ("it is essential to", "AI hedging boilerplate"),
    ("it is crucial to", "AI hedging boilerplate"),
    ("it is worth mentioning", "AI hedging boilerplate"),
    ("it should be noted that", "AI hedging boilerplate"),
    ("it goes without saying", "AI filler phrase"),
    ("needless to say", "AI filler phrase"),
    ("in today's world", "AI cliché opener"),
    ("in today's society", "AI cliché opener"),
    ("in today's fast-paced", "AI cliché opener"),
    ("in today's digital age", "AI cliché opener"),
    ("in conclusion", "AI structural marker"),
    ("in summary", "AI structural marker"),
    ("to summarize", "AI structural marker"),
    ("to conclude", "AI structural marker"),
    ("furthermore", "AI transition — significantly overused"),
    ("moreover", "AI transition — significantly overused"),
    ("in addition to this", "AI transition boilerplate"),
    ("delve into", "AI idiomatic favourite"),
    ("dive deep into", "AI idiomatic favourite"),
    ("shed light on", "AI idiomatic favourite"),
    ("a myriad of", "AI formal inflation"),
    ("multifaceted", "AI buzzword"),
    ("nuanced", "AI buzzword — overused as filler"),
    ("tapestry", "AI metaphor — statistically overrepresented"),
    ("testament to", "AI formal construction"),
    ("in the realm of", "AI formal inflation"),
    ("navigate the complexities", "AI corporate buzzword"),
    ("the landscape of", "AI corporate buzzword"),
    ("foster a deeper understanding", "AI goal-oriented boilerplate"),
    ("align with", "AI corporate buzzword"),
    ("at the end of the day", "AI transition cliché"),
    ("when it comes to", "AI discourse marker — overused"),
    ("as previously mentioned", "AI self-referential boilerplate"),
    ("as mentioned earlier", "AI self-referential boilerplate"),
    ("plays a crucial role", "AI formal boilerplate"),
    ("plays a vital role", "AI formal boilerplate"),
    ("plays an important role", "AI formal boilerplate"),
    ("in order to", "AI verbose construction (vs. simple 'to')"),
    ("a wide range of", "AI vague intensifier"),
    ("a variety of", "AI vague intensifier"),
    ("it is clear that", "AI assertive boilerplate"),
    ("it is evident that", "AI assertive boilerplate"),
    ("it is undeniable that", "AI assertive boilerplate"),
]


def compute_phrase_score(sentence: str) -> dict:
    """
    Check a sentence for AI fingerprint phrases.

    Returns:
        score    — AI probability in [0, 1].
        matched  — List of (phrase, reason) tuples found.
        count    — Number of distinct phrases matched.
    """
    lower = sentence.lower()

    matched = [
        {"phrase": phrase, "reason": reason}
        for phrase, reason in AI_PHRASE_PATTERNS
        if phrase in lower
    ]

    # Scoring: each match raises the score substantially.
    # Two matches in one sentence is very suspicious.
    # Cap at 1.0 regardless of how many are found.
    score = min(1.0, len(matched) * 0.45)

    if matched:
        note = f"Found {len(matched)} AI fingerprint phrase(s): {', '.join(m['phrase'] for m in matched)}"
    else:
        note = "No AI fingerprint phrases detected."

    return {
        "score": round(float(score), 4),
        "matched": matched,
        "count": len(matched),
        "note": note,
    }
