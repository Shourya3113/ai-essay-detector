"""
perplexity.py — GPT-2 based perplexity signal

HOW IT WORKS (plain English):
  We run each sentence through GPT-2, a small language model (124M parameters).
  GPT-2 predicts the probability of each word given the words before it.
  If it assigns HIGH probability to most words → it found the text predictable → low perplexity.
  If it assigns LOW probability → it was surprised → high perplexity.

WHY THIS DETECTS AI:
  AI-generated text is written by models similar to GPT-2. So GPT-2 finds AI text
  predictable — it's essentially "recognising its own kind." Human writing is messier,
  uses unexpected transitions, and surprises the model more.

WHAT WE DON'T DO:
  We do NOT ask GPT-2 "is this AI?" — we only read the raw probability numbers.
  GPT-2 is a measuring instrument here, not a judge.
"""

import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_tokenizer: GPT2TokenizerFast | None = None
_model: GPT2LMHeadModel | None = None


def load_model() -> None:
    """Load GPT-2 weights into memory. Called once at server startup."""
    global _tokenizer, _model
    if _model is None:
        print("[perplexity] Loading GPT-2 (gpt2, 124M params)...")
        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _model = GPT2LMHeadModel.from_pretrained("gpt2")
        _model.eval()
        print("[perplexity] GPT-2 loaded.")


def compute_perplexity(text: str) -> dict:
    """
    Compute the GPT-2 perplexity of a single sentence.

    Returns:
        score  — AI probability in [0, 1]. Higher = more likely AI.
        raw    — Raw perplexity value (for transparency in UI).
        note   — Human-readable interpretation.
    """
    if _model is None or _tokenizer is None:
        load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    n_tokens = inputs["input_ids"].shape[1]
    if n_tokens < 2:
        return {
            "score": 0.5,
            "raw": None,
            "note": "Sentence too short to evaluate perplexity.",
        }

    with torch.no_grad():
        outputs = _model(**inputs, labels=inputs["input_ids"])
        # outputs.loss = mean negative log-likelihood per token
        neg_log_likelihood = outputs.loss.item()

    try:
        raw_perplexity = math.exp(neg_log_likelihood)
    except OverflowError:
        raw_perplexity = 10000.0

    # --- Normalisation ---
    # Empirical ranges from testing on essay corpora:
    #   AI-generated (ChatGPT/Claude output) → perplexity ~20–120
    #   Polished human academic writing       → perplexity ~100–300
    #   Casual / idiosyncratic human writing  → perplexity ~200–600+
    #
    # We use: score = 1 / (1 + (perplexity / 80)^1.5)
    # This gives:
    #   perplexity = 30  → score ≈ 0.80  (very AI-like)
    #   perplexity = 80  → score ≈ 0.50  (uncertain)
    #   perplexity = 200 → score ≈ 0.22  (likely human)
    #   perplexity = 400 → score ≈ 0.10  (very human-like)
    score = 1.0 / (1.0 + (raw_perplexity / 80.0) ** 1.5)
    score = float(max(0.0, min(1.0, score)))

    if raw_perplexity < 60:
        note = f"Perplexity {raw_perplexity:.0f} — very predictable (AI-like)"
    elif raw_perplexity < 150:
        note = f"Perplexity {raw_perplexity:.0f} — moderately predictable (uncertain)"
    else:
        note = f"Perplexity {raw_perplexity:.0f} — model was surprised (human-like)"

    return {
        "score": round(score, 4),
        "raw": round(raw_perplexity, 1),
        "note": note,
    }
