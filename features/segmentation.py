"""Sentence segmentation with original character offsets.

spaCy is preferred (better handling of abbreviations and quotes). The
regex fallback exists so unit tests and a first-run demo still work if
the English model has not been downloaded yet.
"""

from __future__ import annotations

import re

from features.base import SentenceSpan

_ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "st",
    "ave",
    "inc",
    "ltd",
    "e.g",
    "i.e",
}

# Split on end punctuation followed by whitespace and a likely sentence start.
_SPLIT = re.compile(
    r'(?<=[.!?])(?:["”\')\]]*)\s+(?=["“‘(\[]?[A-Z0-9])'
)


def split_sentences(text: str, nlp=None) -> list[SentenceSpan]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []

    if nlp is not None and getattr(nlp, "pipe_names", None) is not None:
        spans = _spacy_split(text, nlp)
        if spans:
            return spans
    return _regex_split(text)


def _spacy_split(text: str, nlp) -> list[SentenceSpan]:
    doc = nlp(text)
    spans: list[SentenceSpan] = []
    for sent in doc.sents:
        raw = text[sent.start_char : sent.end_char]
        stripped = raw.strip()
        if not stripped:
            continue
        lead = raw.index(stripped[0]) if stripped else 0
        start = sent.start_char + lead
        end = start + len(stripped)
        spans.append(SentenceSpan(start=start, end=end, text=stripped))
    return spans


def _regex_split(text: str) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    start = 0
    for match in _SPLIT.finditer(text):
        end = match.start()
        piece = text[start:end]
        _append_span(spans, text, start, piece)
        start = match.end()
    _append_span(spans, text, start, text[start:])
    return [s for s in spans if _looks_like_sentence(s.text)]


def _append_span(spans: list[SentenceSpan], text: str, origin: int, piece: str) -> None:
    stripped = piece.strip()
    if not stripped:
        return
    rel = piece.find(stripped)
    start = origin + rel
    spans.append(SentenceSpan(start=start, end=start + len(stripped), text=stripped))


def _looks_like_sentence(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) == 0:
        return False
    if len(words) == 1 and words[0].lower().rstrip(".") in _ABBREV:
        return False
    return True
