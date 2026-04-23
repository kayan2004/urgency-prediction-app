from __future__ import annotations

from fastapi import APIRouter, HTTPException

from schemas.rag import (
    NonRagAnswerRequest,
    NonRagAnswerResponse,
    RagAnswerRequest,
    RagAnswerResponse,
)
from schemas.retrieve import RetrievedChunk
from services.generation_service import generate_non_rag_answer, generate_rag_answer
from services.retrieval_service import retrieve_similar_tickets


router = APIRouter(tags=["generation"])


@router.post("/rag-answer", response_model=RagAnswerResponse)
def rag_answer(request: RagAnswerRequest) -> RagAnswerResponse:
    try:
        results = retrieve_similar_tickets(query=request.query, top_k=request.top_k)
        sources = [
            RetrievedChunk(
                doc_id=str(row.doc_id),
                tweet_id=str(row.tweet_id),
                customer_text=str(row.text),
                company_reply_text=None
                if row.company_reply_text != row.company_reply_text
                else str(row.company_reply_text),
                retrieval_text=str(row.retrieval_text),
                similarity=float(row.similarity),
            )
            for row in results.itertuples(index=False)
        ]
        answer = generate_rag_answer(
            query=request.query,
            sources=sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RagAnswerResponse(
        query=request.query,
        top_k=request.top_k,
        answer=answer,
        sources=sources,
    )


@router.post("/non-rag-answer", response_model=NonRagAnswerResponse)
def non_rag_answer(request: NonRagAnswerRequest) -> NonRagAnswerResponse:
    try:
        answer = generate_non_rag_answer(query=request.query)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return NonRagAnswerResponse(
        query=request.query,
        answer=answer,
    )
