from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SentenceSpan:
    """A sentence as it appears in the original essay, with character offsets."""

    start: int
    end: int
    text: str


@dataclass
class FeatureValue:
    """One measured quantity plus a plain-English reading of it."""

    name: str
    value: float
    explanation: str


@dataclass
class SentenceRecord:
    span: SentenceSpan
    features: dict[str, float]
    explanations: dict[str, str] = field(default_factory=dict)


@dataclass
class EssayRecord:
    text: str
    sentences: list[SentenceRecord]
    essay_features: dict[str, float]
    essay_explanations: dict[str, str]
    neural_used: dict


def safe_var(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def zscores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = safe_mean(values)
    var = safe_var(values)
    std = var**0.5
    if std < 1e-9:
        return [0.0] * len(values)
    return [(x - mean) / std for x in values]
