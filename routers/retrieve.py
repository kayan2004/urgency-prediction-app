from __future__ import annotations

from fastapi import APIRouter, HTTPException

from schemas.retrieve import RetrieveRequest, RetrieveResponse, RetrievedChunk
from services.retrieval_service import retrieve_similar_tickets


router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    try:
        results = retrieve_similar_tickets(query=request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources = [
        RetrievedChunk(
            doc_id=str(row.doc_id),
            tweet_id=str(row.tweet_id),
            customer_text=str(row.text),
            company_reply_text=None if row.company_reply_text != row.company_reply_text else str(row.company_reply_text),
            retrieval_text=str(row.retrieval_text),
            similarity=float(row.similarity),
        )
        for row in results.itertuples(index=False)
    ]

    return RetrieveResponse(query=request.query, top_k=request.top_k, sources=sources)
