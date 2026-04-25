from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from services.ml_service import LABEL_MAP, build_feature_frame, load_priority_model, normalize_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_RECORDS_PATH = PROJECT_ROOT / "dataset" / "retrieval" / "support_tickets_for_rag.csv"
PRIORITY_EVALUATION_PATH = PROJECT_ROOT / "model" / "priority_evaluation.json"
REVERSE_LABEL_MAP = {label: code for code, label in LABEL_MAP.items()}


def _resolve_llm_accuracy_from_metadata() -> float | None:
    env_value = os.getenv("LLM_PRIORITY_REFERENCE_ACCURACY", "").strip()
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            return None

    if not PRIORITY_EVALUATION_PATH.exists():
        return None

    try:
        payload = json.loads(PRIORITY_EVALUATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    raw_value = payload.get("llm_priority_accuracy")
    if raw_value is None:
        return None

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_ml_reference_accuracy() -> float | None:
    if not RETRIEVAL_RECORDS_PATH.exists():
        return None

    dataset = pd.read_csv(RETRIEVAL_RECORDS_PATH)
    if "clean_text" not in dataset.columns or "label" not in dataset.columns:
        return None

    evaluation_rows = dataset.dropna(subset=["clean_text", "label"]).copy()
    if evaluation_rows.empty:
        return None

    evaluation_rows["clean_text"] = evaluation_rows["clean_text"].map(normalize_text)
    evaluation_rows["label_code"] = evaluation_rows["label"].map(REVERSE_LABEL_MAP)
    evaluation_rows = evaluation_rows.dropna(subset=["label_code"])
    if evaluation_rows.empty:
        return None

    features = build_feature_frame(evaluation_rows["clean_text"].tolist())
    model = load_priority_model()
    predictions = model.predict(features)
    labels = evaluation_rows["label_code"].astype(int).to_numpy()

    accuracy = float((predictions == labels).mean())
    return round(accuracy, 4)


@lru_cache(maxsize=1)
def get_llm_reference_accuracy() -> float | None:
    value = _resolve_llm_accuracy_from_metadata()
    return None if value is None else round(value, 4)
