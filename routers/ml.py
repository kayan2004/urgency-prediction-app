from __future__ import annotations

from fastapi import APIRouter, HTTPException

from schemas.ml import (
    LlmPriorityPredictionRequest,
    LlmPriorityPredictionResponse,
    PriorityPredictionRequest,
    PriorityPredictionResponse,
)
from services.generation_service import generate_llm_priority_prediction
from services.ml_service import predict_priority


router = APIRouter(tags=["ml"])


@router.post("/priority-predict", response_model=PriorityPredictionResponse)
def priority_predict(request: PriorityPredictionRequest) -> PriorityPredictionResponse:
    try:
        prediction = predict_priority(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PriorityPredictionResponse(**prediction)


@router.post("/priority-predict-llm", response_model=LlmPriorityPredictionResponse)
def priority_predict_llm(request: LlmPriorityPredictionRequest) -> LlmPriorityPredictionResponse:
    try:
        prediction = generate_llm_priority_prediction(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return LlmPriorityPredictionResponse(**prediction)
