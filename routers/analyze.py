from __future__ import annotations

from fastapi import APIRouter, HTTPException

from schemas.analyze import AnalyzeRequest, AnalyzeResponse
from services.analysis_service import analyze_query


router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = analyze_query(query=request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AnalyzeResponse(**result)
