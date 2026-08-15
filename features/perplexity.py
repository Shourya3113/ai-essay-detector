"""Per-sentence log-likelihood / perplexity from a local base LM.

Linguistic reasoning
--------------------
Instruction-tuned chat models are *not* used here. We run a small
open-weight base model (DistilGPT-2) as a measuring instrument: it
assigns a next-token log-probability to each token. Average negative
log-likelihood (NLL) is a surprisal score.

Empirically, many LLM-written passages are more "typical" under a
generic English LM — lower NLL, lower perplexity — than messy human
drafts. That is not a law of nature (a student who writes in a very
standard register can also look typical), so we never threshold NLL
against a global cutoff. We compare each sentence to *this essay's*
own NLL distribution (see burstiness.py).

Fallback: a character trigram trained on public-domain English, used
only when DistilGPT-2 is unavailable. Same quantity, weaker model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from features.base import SentenceSpan
from features.char_lm import char_nll


def sentence_nlls(
    text: str,
    sentences: list[SentenceSpan],
    model=None,
    tokenizer=None,
) -> tuple[list[float], str]:
    """Return per-sentence average NLL and a string naming the instrument."""
    if model is not None and tokenizer is not None:
        nlls = _gpt2_sentence_nlls(text, sentences, model, tokenizer)
        if nlls is not None:
            return nlls, "distilgpt2"
    return [_sentence_char_nll(s.text) for s in sentences], "char_trigram"


def _sentence_char_nll(sentence: str) -> float:
    return float(char_nll(sentence))


def _gpt2_sentence_nlls(
    text: str,
    sentences: list[SentenceSpan],
    model: Any,
    tokenizer: Any,
) -> list[float] | None:
    try:
        import torch
    except Exception:
        return None

    max_len = min(getattr(model.config, "n_positions", 1024), 1024)
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
        )
    except TypeError:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        encoded["offset_mapping"] = None

    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return [_gpt2_isolated_nll(s.text, model, tokenizer) for s in sentences]

    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        targets = input_ids[0, 1:]
        token_nll = -log_probs[torch.arange(targets.shape[0]), targets].cpu().numpy()

    off = offsets[0].tolist()
    # token_nll[i] is the NLL of token i+1 (the predicted token)
    token_offsets = off[1:]
    nlls: list[float] = []
    for sent in sentences:
        bucket: list[float] = []
        for nll, (a, b) in zip(token_nll, token_offsets):
            if b <= a:
                continue
            if a < sent.end and b > sent.start:
                bucket.append(float(nll))
        if bucket:
            nlls.append(float(np.mean(bucket)))
        else:
            nlls.append(_gpt2_isolated_nll(sent.text, model, tokenizer))
    return nlls


def _gpt2_isolated_nll(sentence: str, model: Any, tokenizer: Any) -> float:
    import torch

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < 2:
        return 0.0
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        targets = input_ids[0, 1:]
        nll = -log_probs[torch.arange(targets.shape[0]), targets]
        return float(nll.mean().cpu())
