"""Lazy loaders for optional measurement models.

DistilGPT-2 and MiniLM are instruments: they emit numbers, never a
"this is AI" decision. Set ESSAYTRACE_SKIP_NEURAL=1 to skip downloads
(used by unit tests).
"""

from __future__ import annotations

import os
import threading
from typing import Any

_LOCK = threading.Lock()
_SPACY = None
_GPT2 = None
_GPT2_TOK = None
_EMBEDDER = None
_SPACY_FAILED = False
_GPT2_FAILED = False
_EMBED_FAILED = False


def skip_neural() -> bool:
    return os.environ.get("ESSAYTRACE_SKIP_NEURAL", "").strip() in {"1", "true", "yes"}


def get_spacy() -> Any | None:
    global _SPACY, _SPACY_FAILED
    if _SPACY_FAILED:
        return None
    if _SPACY is not None:
        return _SPACY
    with _LOCK:
        if _SPACY is not None or _SPACY_FAILED:
            return _SPACY
        try:
            import spacy

            try:
                _SPACY = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.blank("en")
                nlp.add_pipe("sentencizer")
                _SPACY = nlp
        except Exception:
            _SPACY_FAILED = True
            _SPACY = None
        return _SPACY


def get_gpt2() -> tuple[Any, Any] | tuple[None, None]:
    global _GPT2, _GPT2_TOK, _GPT2_FAILED
    if skip_neural() or _GPT2_FAILED:
        return None, None
    if _GPT2 is not None:
        return _GPT2, _GPT2_TOK
    with _LOCK:
        if _GPT2 is not None or _GPT2_FAILED:
            return _GPT2, _GPT2_TOK
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = os.environ.get("ESSAYTRACE_LM", "distilgpt2")
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name)
            model.eval()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            _GPT2, _GPT2_TOK = model, tok
            # Keep torch imported for dtype/device later.
            _ = torch
        except Exception:
            _GPT2_FAILED = True
            _GPT2, _GPT2_TOK = None, None
        return _GPT2, _GPT2_TOK


def get_embedder() -> Any | None:
    global _EMBEDDER, _EMBED_FAILED
    if skip_neural() or _EMBED_FAILED:
        return None
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _LOCK:
        if _EMBEDDER is not None or _EMBED_FAILED:
            return _EMBEDDER
        try:
            from sentence_transformers import SentenceTransformer

            name = os.environ.get(
                "ESSAYTRACE_EMBED", "sentence-transformers/all-MiniLM-L6-v2"
            )
            _EMBEDDER = SentenceTransformer(name)
        except Exception:
            _EMBED_FAILED = True
            _EMBEDDER = None
        return _EMBEDDER
