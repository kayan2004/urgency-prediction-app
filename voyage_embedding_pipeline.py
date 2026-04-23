from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dotenv import load_dotenv

if TYPE_CHECKING:
    import voyageai


load_dotenv()

VOYAGE_EMBEDDING_MODEL = "voyage-3.5-lite"
DEFAULT_BATCH_SIZE = 25
DEFAULT_MAX_CHARS = 300
DEFAULT_SLEEP_SECONDS = 21.0
DEFAULT_MAX_RETRIES = 5
RATE_LIMIT_RESET_SECONDS = 65.0


def _get_client() -> "voyageai.Client":
    import voyageai

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Voyage API key was provided. Set VOYAGE_API_KEY in your environment or .env file."
        )
    return voyageai.Client(api_key=api_key)


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = str(text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip() or normalized[:max_chars]


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "rate limit" in message
        or "too many requests" in message
        or "429" in message
        or "quota" in message
    )


def _embed_with_retry(
    client: "voyageai.Client",
    texts: list[str],
    input_type: str,
    max_retries: int,
) -> np.ndarray:
    for attempt in range(max_retries + 1):
        try:
            result = client.embed(
                texts=texts,
                model=VOYAGE_EMBEDDING_MODEL,
                input_type=input_type,
                truncation=True,
            )
            return np.array(result.embeddings, dtype=np.float32)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == max_retries:
                raise
            wait_seconds = RATE_LIMIT_RESET_SECONDS * (attempt + 1)
            print(
                f"Embedding batch hit rate limits. "
                f"Retrying in {wait_seconds} seconds (attempt {attempt + 1}/{max_retries})."
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Embedding retry loop exited unexpectedly.")


def build_voyage_embeddings(
    records_path: str | Path,
    embeddings_path: str | Path,
    metadata_path: str | Path,
    text_column: str = "retrieval_clean_text",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_chars: int = DEFAULT_MAX_CHARS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[np.ndarray, dict]:
    records_path = Path(records_path)
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    checkpoint_path = embeddings_path.with_name(f"{embeddings_path.stem}_partial.npy")

    retrieval_df = pd.read_csv(records_path)
    text_values = [
        _truncate_text(text, max_chars=max_chars)
        for text in retrieval_df[text_column].fillna("").astype(str).tolist()
    ]

    client = _get_client()
    embedding_batches: list[np.ndarray] = []
    completed_rows = 0

    if checkpoint_path.exists():
        checkpoint_embeddings = np.load(checkpoint_path)
        if checkpoint_embeddings.ndim == 2 and checkpoint_embeddings.shape[0] <= len(text_values):
            embedding_batches.append(checkpoint_embeddings)
            completed_rows = int(checkpoint_embeddings.shape[0])
            print(f"Resuming from checkpoint with {completed_rows:,} embedded rows.")

    for start_idx in range(completed_rows, len(text_values), batch_size):
        end_idx = min(start_idx + batch_size, len(text_values))
        batch_texts = text_values[start_idx:end_idx]
        embedding_batches.append(
            _embed_with_retry(
                client=client,
                texts=batch_texts,
                input_type="document",
                max_retries=max_retries,
            )
        )
        current_embeddings = np.vstack(embedding_batches)
        np.save(checkpoint_path, current_embeddings)
        print(f"Embedded rows {start_idx:,} to {end_idx:,} of {len(text_values):,}")
        if end_idx < len(text_values):
            time.sleep(max(sleep_seconds, DEFAULT_SLEEP_SECONDS))

    embeddings = np.vstack(embedding_batches) if embedding_batches else np.empty((0, 0), dtype=np.float32)
    np.save(embeddings_path, embeddings)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    metadata = {
        "model": VOYAGE_EMBEDDING_MODEL,
        "batch_size": batch_size,
        "max_chars": max_chars,
        "sleep_seconds": sleep_seconds,
        "max_retries": max_retries,
        "rows_embedded": int(embeddings.shape[0]),
        "vector_width": int(embeddings.shape[1]) if embeddings.size else 0,
        "records_path": str(records_path.resolve()),
        "text_column": text_column,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return embeddings, metadata


def embed_query_text(query: str) -> np.ndarray:
    client = _get_client()
    result = client.embed(
        texts=[str(query).strip()],
        model=VOYAGE_EMBEDDING_MODEL,
        input_type="query",
        truncation=True,
    )
    return np.array(result.embeddings, dtype=np.float32)
