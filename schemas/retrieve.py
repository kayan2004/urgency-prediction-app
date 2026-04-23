from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    doc_id: str
    tweet_id: str
    customer_text: str
    company_reply_text: str | None = None
    retrieval_text: str
    similarity: float


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    sources: list[RetrievedChunk]
