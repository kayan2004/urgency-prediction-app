from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.metrics import RuntimeMetrics
from schemas.retrieve import RetrievedChunk


class RagAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RagAnswerResponse(BaseModel):
    query: str
    top_k: int
    answer: str
    metrics: RuntimeMetrics
    sources: list[RetrievedChunk]


class NonRagAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)


class NonRagAnswerResponse(BaseModel):
    query: str
    answer: str
    metrics: RuntimeMetrics
