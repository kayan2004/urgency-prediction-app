from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from schemas.metrics import RuntimeMetrics
from schemas.ml import (
    LlmPriorityPredictionRequest,
    LlmPriorityPredictionResponse,
    PriorityPredictionRequest,
    PriorityPredictionResponse,
)
from services.evaluation_service import get_llm_reference_accuracy, get_ml_reference_accuracy
from services.generation_service import generate_llm_priority_prediction
from services.ml_service import predict_priority


router = APIRouter(tags=["ml"])


@router.post("/priority-predict", response_model=PriorityPredictionResponse)
def priority_predict(request: PriorityPredictionRequest) -> PriorityPredictionResponse:
    try:
        started_at = time.perf_counter()
        prediction = predict_priority(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    metrics = RuntimeMetrics(
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        prompt_tokens=0,
        output_tokens=0,
        total_tokens=0,
    )
    return PriorityPredictionResponse(
        **prediction,
        reference_accuracy=get_ml_reference_accuracy(),
        metrics=metrics,
    )


@router.post("/priority-predict-llm", response_model=LlmPriorityPredictionResponse)
def priority_predict_llm(request: LlmPriorityPredictionRequest) -> LlmPriorityPredictionResponse:
    try:
        prediction = generate_llm_priority_prediction(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return LlmPriorityPredictionResponse(
        **prediction,
        reference_accuracy=get_llm_reference_accuracy(),
    )
