from __future__ import annotations

from fastapi import APIRouter, HTTPException

from schemas.analyze import AnalyzeRequest, AnalyzeResponse
from services.analysis_service import analyze_query
from services.logging_service import get_logger


router = APIRouter(tags=["analysis"])
logger = get_logger("decision_intelligence.analyze")


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = analyze_query(query=request.query, top_k=request.top_k)
    except ValueError as exc:
        logger.exception(
            "analyze_failed query=%r top_k=%s error=%s",
            request.query,
            request.top_k,
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AnalyzeResponse(**result)
