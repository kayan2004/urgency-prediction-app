from __future__ import annotations

import time
from statistics import mean

from schemas.metrics import RuntimeMetrics
from schemas.ml import LlmPriorityPredictionResponse, PriorityPredictionResponse
from schemas.retrieve import RetrievedChunk
from services.evaluation_service import get_llm_reference_accuracy, get_ml_reference_accuracy
from services.generation_service import generate_llm_priority_prediction, generate_non_rag_answer, generate_rag_answer
from services.logging_service import get_logger
from services.ml_service import predict_priority
from services.retrieval_service import retrieve_similar_tickets

logger = get_logger("decision_intelligence.analysis")


def _build_sources(query: str, top_k: int) -> list[RetrievedChunk]:
    results = retrieve_similar_tickets(query=query, top_k=top_k)
    return [
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


def analyze_query(query: str, top_k: int) -> dict[str, object]:
    logger.info("analysis_started query=%r top_k=%s", query, top_k)
    sources = _build_sources(query=query, top_k=top_k)
    top_similarity = max((source.similarity for source in sources), default=0.0)
    average_similarity = mean(source.similarity for source in sources) if sources else 0.0
    logger.info(
        "retrieval_completed source_count=%s top_similarity=%.4f avg_similarity=%.4f top_doc_ids=%s",
        len(sources),
        top_similarity,
        average_similarity,
        [source.doc_id for source in sources[:3]],
    )

    rag_answer, rag_metrics = generate_rag_answer(query=query, sources=sources)
    non_rag_answer, non_rag_metrics = generate_non_rag_answer(query=query)

    ml_started = time.perf_counter()
    ml_prediction = predict_priority(query)
    ml_metrics = RuntimeMetrics(
        latency_ms=round((time.perf_counter() - ml_started) * 1000, 2),
        prompt_tokens=0,
        output_tokens=0,
        total_tokens=0,
    )
    ml_priority = PriorityPredictionResponse(
        **ml_prediction,
        reference_accuracy=get_ml_reference_accuracy(),
        metrics=ml_metrics,
    )
    llm_priority = LlmPriorityPredictionResponse(
        **generate_llm_priority_prediction(query),
        reference_accuracy=get_llm_reference_accuracy(),
    )
    logger.info(
        "analysis_completed ml_priority=%s llm_priority=%s rag_latency_ms=%.2f non_rag_latency_ms=%.2f ml_latency_ms=%.2f llm_latency_ms=%.2f rag_tokens=%s non_rag_tokens=%s llm_tokens=%s",
        ml_priority.predicted_priority,
        llm_priority.predicted_priority,
        rag_metrics.latency_ms,
        non_rag_metrics.latency_ms,
        ml_metrics.latency_ms,
        llm_priority.metrics.latency_ms,
        rag_metrics.total_tokens,
        non_rag_metrics.total_tokens,
        llm_priority.metrics.total_tokens,
    )

    return {
        "query": query,
        "top_k": top_k,
        "sources": sources,
        "rag_answer": rag_answer,
        "rag_metrics": rag_metrics,
        "non_rag_answer": non_rag_answer,
        "non_rag_metrics": non_rag_metrics,
        "ml_priority": ml_priority,
        "llm_priority": llm_priority,
    }
