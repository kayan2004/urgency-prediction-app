from __future__ import annotations

from pydantic import BaseModel, Field


class PriorityPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PriorityPredictionResponse(BaseModel):
    text: str
    clean_text: str
    predicted_priority: str
    confidence: float
    probabilities: dict[str, float]


class LlmPriorityPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)


class LlmPriorityPredictionResponse(BaseModel):
    text: str
    clean_text: str
    predicted_priority: str
    rationale: str
