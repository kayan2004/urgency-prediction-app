from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

from voyage_embedding_pipeline import embed_query_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_RECORDS_PATH = PROJECT_ROOT / "dataset" / "retrieval" / "support_tickets_for_rag.csv"
RAG_EMBEDDINGS_PATH = PROJECT_ROOT / "model" / "rag_voyage_embeddings.npy"
CHROMA_DB_PATH = PROJECT_ROOT / "model" / "chroma_db"
CHROMA_COLLECTION_NAME = "support_tickets_rag"


def normalize_query_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _clean_metadata_value(value: object) -> str | float | int:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _build_metadata(row: pd.Series) -> dict[str, str | float | int]:
    return {
        "doc_id": _clean_metadata_value(row["doc_id"]),
        "tweet_id": _clean_metadata_value(row["tweet_id"]),
        "text": _clean_metadata_value(row["text"]),
        "clean_text": _clean_metadata_value(row["clean_text"]),
        "company_reply_text": _clean_metadata_value(row["company_reply_text"]),
        "retrieval_text": _clean_metadata_value(row["retrieval_text"]),
        "label": _clean_metadata_value(row["label"]),
        "priority_score": _clean_metadata_value(row["priority_score"]),
    }


def _batch_upsert(collection, retrieval_df: pd.DataFrame, rag_embeddings: np.ndarray, batch_size: int = 200) -> None:
    for start in range(0, len(retrieval_df), batch_size):
        batch = retrieval_df.iloc[start : start + batch_size]
        embeddings = rag_embeddings[start : start + batch_size]
        ids = [str(row.doc_id) for row in batch.itertuples(index=False)]
        documents = [str(row.retrieval_text) for row in batch.itertuples(index=False)]
        metadatas = [_build_metadata(row) for _, row in batch.iterrows()]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )


@lru_cache(maxsize=1)
def load_retrieval_artifacts() -> tuple[pd.DataFrame, object]:
    retrieval_df = pd.read_csv(RETRIEVAL_RECORDS_PATH)

    if not RAG_EMBEDDINGS_PATH.exists():
        raise ValueError(
            "Voyage embeddings file is missing. Generate model/rag_voyage_embeddings.npy "
            "from the notebook before calling /retrieve."
        )

    rag_embeddings = np.load(RAG_EMBEDDINGS_PATH)

    if len(retrieval_df) != len(rag_embeddings):
        raise ValueError(
            "Retrieval records and embedding matrix have different row counts: "
            f"{len(retrieval_df)} vs {len(rag_embeddings)}. "
            "Regenerate the Voyage embeddings so they match the current retrieval dataset."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() != len(retrieval_df):
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        _batch_upsert(collection, retrieval_df, rag_embeddings)

    return retrieval_df, collection


def retrieve_similar_tickets(query: str, top_k: int = 5) -> pd.DataFrame:
    _, collection = load_retrieval_artifacts()

    query_embedding = embed_query_text(normalize_query_text(query))
    chroma_result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    metadatas = chroma_result.get("metadatas", [[]])[0]
    documents = chroma_result.get("documents", [[]])[0]
    distances = chroma_result.get("distances", [[]])[0]

    rows: list[dict[str, object]] = []
    for metadata, document, distance in zip(metadatas, documents, distances, strict=True):
        company_reply_text = metadata.get("company_reply_text") or None
        rows.append(
            {
                "doc_id": str(metadata.get("doc_id", "")),
                "tweet_id": str(metadata.get("tweet_id", "")),
                "text": str(metadata.get("text", "")),
                "clean_text": str(metadata.get("clean_text", "")),
                "company_reply_text": company_reply_text,
                "retrieval_text": str(document or metadata.get("retrieval_text", "")),
                "label": str(metadata.get("label", "")),
                "priority_score": metadata.get("priority_score", 0),
                "similarity": max(0.0, 1.0 - float(distance)),
            }
        )

    return pd.DataFrame(rows)
