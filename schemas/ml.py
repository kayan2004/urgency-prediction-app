from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.metrics import RuntimeMetrics


class PriorityPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PriorityPredictionResponse(BaseModel):
    text: str
    clean_text: str
    predicted_priority: str
    confidence: float
    probabilities: dict[str, float]
    reference_accuracy: float | None = None
    metrics: RuntimeMetrics


class LlmPriorityPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)


class LlmPriorityPredictionResponse(BaseModel):
    text: str
    clean_text: str
    predicted_priority: str
    rationale: str
    reference_accuracy: float | None = None
    metrics: RuntimeMetrics
