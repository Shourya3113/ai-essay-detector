from __future__ import annotations

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class FeatureHit(BaseModel):
    feature: str
    value: float
    contribution: float
    plain: str


class SentenceResult(BaseModel):
    text: str
    start: int
    end: int
    p_ai: float
    explanations: list[FeatureHit]


class EssayScore(BaseModel):
    point: float
    low: float
    high: float
    label: str


class DetectResponse(BaseModel):
    essay_score: EssayScore
    caveat: str
    sentences: list[SentenceResult]
    instruments: dict
    model_name: str
    too_short: bool = False
