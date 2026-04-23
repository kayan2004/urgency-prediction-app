from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from voyage_embedding_pipeline import embed_query_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_RECORDS_PATH = PROJECT_ROOT / "dataset" / "retrieval" / "support_tickets_for_rag.csv"
RAG_EMBEDDINGS_PATH = PROJECT_ROOT / "model" / "rag_voyage_embeddings.npy"


def normalize_query_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


@lru_cache(maxsize=1)
def load_retrieval_artifacts() -> tuple[pd.DataFrame, np.ndarray]:
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

    return retrieval_df, rag_embeddings


def retrieve_similar_tickets(query: str, top_k: int = 5) -> pd.DataFrame:
    retrieval_df, rag_embeddings = load_retrieval_artifacts()

    query_embedding = embed_query_text(normalize_query_text(query))
    similarity_scores = cosine_similarity(query_embedding, rag_embeddings).ravel()
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = retrieval_df.iloc[top_indices].copy()
    results["similarity"] = similarity_scores[top_indices]

    return results[
        [
            "doc_id",
            "tweet_id",
            "text",
            "clean_text",
            "company_reply_text",
            "retrieval_text",
            "label",
            "priority_score",
            "similarity",
        ]
    ].reset_index(drop=True)
