from __future__ import annotations

from schemas.ml import LlmPriorityPredictionResponse, PriorityPredictionResponse
from schemas.retrieve import RetrievedChunk
from services.generation_service import generate_llm_priority_prediction, generate_non_rag_answer, generate_rag_answer
from services.ml_service import predict_priority
from services.retrieval_service import retrieve_similar_tickets


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
    sources = _build_sources(query=query, top_k=top_k)

    rag_answer = generate_rag_answer(query=query, sources=sources)
    non_rag_answer = generate_non_rag_answer(query=query)

    ml_priority = PriorityPredictionResponse(**predict_priority(query))
    llm_priority = LlmPriorityPredictionResponse(**generate_llm_priority_prediction(query))

    return {
        "query": query,
        "top_k": top_k,
        "sources": sources,
        "rag_answer": rag_answer,
        "non_rag_answer": non_rag_answer,
        "ml_priority": ml_priority,
        "llm_priority": llm_priority,
    }
