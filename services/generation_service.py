from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from schemas.metrics import RuntimeMetrics
from schemas.retrieve import RetrievedChunk
from services.ml_service import normalize_text

if TYPE_CHECKING:
    from google.genai import Client


load_dotenv()

DEFAULT_RAG_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemma-4-31b-it")
DEFAULT_GENERATION_MAX_RETRIES = 3
RAG_CONTEXT_SOURCE_LIMIT = 3
RAG_CONTEXT_CHAR_LIMIT = 700


def _get_client() -> "Client":
    try:
        from google import genai
    except ImportError as exc:
        raise ValueError(
            "The google-genai package is not installed. Install dependencies before calling /rag-answer."
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "No Gemini API key was provided. Set GEMINI_API_KEY in your environment or .env file."
        )
    return genai.Client(api_key=api_key)


def _is_retryable_generation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "500",
            "internal",
            "503",
            "unavailable",
            "overloaded",
            "rate limit",
            "429",
            "quota",
        ]
    )


def _build_metrics(*, started_at: float, response) -> RuntimeMetrics:
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    return RuntimeMetrics(
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _generate_text(prompt: str, *, context_name: str) -> tuple[str, RuntimeMetrics]:
    client = _get_client()

    last_exception: Exception | None = None
    for attempt in range(DEFAULT_GENERATION_MAX_RETRIES + 1):
        try:
            started_at = time.perf_counter()
            response = client.models.generate_content(
                model=DEFAULT_RAG_MODEL,
                contents=prompt,
            )
            answer = getattr(response, "text", None)
            if not answer:
                raise ValueError(f"Gemini returned an empty response for the {context_name}.")
            return answer.strip(), _build_metrics(started_at=started_at, response=response)
        except ValueError:
            raise
        except Exception as exc:
            last_exception = exc
            if not _is_retryable_generation_error(exc) or attempt == DEFAULT_GENERATION_MAX_RETRIES:
                raise ValueError(
                    f"Gemini failed while generating the {context_name}: {exc}"
                ) from exc

            wait_seconds = 2 * (attempt + 1)
            time.sleep(wait_seconds)

    raise ValueError(
        f"Gemini failed while generating the {context_name}: {last_exception}"
    )


def _build_rag_prompt(query: str, sources: list[RetrievedChunk]) -> str:
    context_blocks = []
    for index, source in enumerate(sources[:RAG_CONTEXT_SOURCE_LIMIT], start=1):
        retrieval_text = source.retrieval_text[:RAG_CONTEXT_CHAR_LIMIT].strip()
        context_blocks.append(
            "\n".join(
                [
                    f"Source {index}",
                    retrieval_text,
                ]
            )
        )

    joined_context = "\n\n".join(context_blocks)

    return "\n\n".join(
        [
            "You are a customer support assistant.",
            "Use the retrieved support cases below to answer the customer's query.",
            "Be concise, helpful, and grounded in the retrieved cases.",
            "Only rely on the retrieved cases when they are relevant.",
            "If the retrieved cases are not enough, say that you are making a best-effort suggestion.",
            f"Customer query: {query}",
            f"Retrieved cases:\n{joined_context}",
            "Answer:",
        ]
    )


def _build_non_rag_prompt(query: str) -> str:
    return "\n\n".join(
        [
            "You are a customer support assistant.",
            "Answer the customer's query directly and concisely.",
            "Be helpful, practical, and honest when you are uncertain.",
            f"Customer query: {query}",
            "Answer:",
        ]
    )


def _build_priority_prompt(text: str) -> str:
    return "\n\n".join(
        [
            "You are classifying customer support messages by urgency.",
            "Classify the message as exactly one of: urgent, normal.",
            "Use urgent for issues like account lockout, billing/charge problems, security problems, broken service, or time-sensitive failures.",
            "Use normal for lower-stakes questions, generic complaints, or non-urgent requests.",
            "Return valid JSON only with this schema:",
            '{"predicted_priority": "urgent|normal", "rationale": "short explanation"}',
            f"Message: {text}",
        ]
    )


def _parse_priority_response(response_text: str) -> dict[str, str]:
    text = response_text.strip()

    if text.startswith("```"):
        parts = text.split("```")
        text = next((part for part in parts if "predicted_priority" in part), text).strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned an invalid priority-classification response."
        ) from exc

    predicted_priority = str(parsed.get("predicted_priority", "")).strip().lower()
    rationale = str(parsed.get("rationale", "")).strip()

    if predicted_priority not in {"urgent", "normal"}:
        raise ValueError(
            "Gemini returned an unexpected priority label. Expected 'urgent' or 'normal'."
        )

    if not rationale:
        rationale = "No rationale provided."

    return {
        "predicted_priority": predicted_priority,
        "rationale": rationale,
    }


def generate_rag_answer(
    query: str,
    sources: list[RetrievedChunk],
) -> tuple[str, RuntimeMetrics]:
    prompt = _build_rag_prompt(query=query, sources=sources)
    return _generate_text(prompt, context_name="RAG answer")


def generate_non_rag_answer(query: str) -> tuple[str, RuntimeMetrics]:
    prompt = _build_non_rag_prompt(query=query)
    return _generate_text(prompt, context_name="non-RAG answer")


def generate_llm_priority_prediction(text: str) -> dict[str, str | RuntimeMetrics]:
    clean_text = normalize_text(text)
    prompt = _build_priority_prompt(text=clean_text)
    answer, metrics = _generate_text(prompt, context_name="LLM priority prediction")

    parsed = _parse_priority_response(answer)
    return {
        "text": text,
        "clean_text": clean_text,
        "predicted_priority": parsed["predicted_priority"],
        "rationale": parsed["rationale"],
        "metrics": metrics,
    }
