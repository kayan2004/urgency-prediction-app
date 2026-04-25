from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.metrics import RuntimeMetrics
from schemas.ml import LlmPriorityPredictionResponse, PriorityPredictionResponse
from schemas.retrieve import RetrievedChunk


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AnalyzeResponse(BaseModel):
    query: str
    top_k: int
    sources: list[RetrievedChunk]
    rag_answer: str
    rag_metrics: RuntimeMetrics
    non_rag_answer: str
    non_rag_metrics: RuntimeMetrics
    ml_priority: PriorityPredictionResponse
    llm_priority: LlmPriorityPredictionResponse
