from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai


GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_BATCH_SIZE = 12
DEFAULT_MAX_CHARS = 400
DEFAULT_SLEEP_SECONDS = 2.0
DEFAULT_MAX_RETRIES = 5


def _embed_contents(client: genai.Client, contents: str | list[str]) -> np.ndarray:
    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=contents,
    )
    return np.array([item.values for item in result.embeddings], dtype=np.float32)


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = str(text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip() or normalized[:max_chars]


def _is_resource_exhausted_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "resource exhausted" in message or "429" in message or "quota" in message


def _embed_with_retry(
    client: genai.Client,
    batch_texts: list[str],
    max_retries: int,
) -> np.ndarray:
    for attempt in range(max_retries + 1):
        try:
            return _embed_contents(client, batch_texts)
        except Exception as exc:
            if not _is_resource_exhausted_error(exc) or attempt == max_retries:
                raise
            wait_seconds = 5 * (2**attempt)
            print(
                f"Embedding batch hit quota/resource limits. "
                f"Retrying in {wait_seconds} seconds (attempt {attempt + 1}/{max_retries})."
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Embedding retry loop exited unexpectedly.")


def build_gemini_embeddings(
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

    client = genai.Client()
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
                batch_texts=batch_texts,
                max_retries=max_retries,
            )
        )
        current_embeddings = np.vstack(embedding_batches)
        np.save(checkpoint_path, current_embeddings)
        print(f"Embedded rows {start_idx:,} to {end_idx:,} of {len(text_values):,}")
        if end_idx < len(text_values):
            time.sleep(sleep_seconds)

    embeddings = np.vstack(embedding_batches) if embedding_batches else np.empty((0, 0), dtype=np.float32)
    np.save(embeddings_path, embeddings)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    metadata = {
        "model": GEMINI_EMBEDDING_MODEL,
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
    client = genai.Client()
    embeddings = _embed_contents(client, query)
    return embeddings[0].reshape(1, -1)
