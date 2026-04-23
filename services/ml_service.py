from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "best_logistic_regression.joblib"
LABEL_MAP = {0: "normal", 1: "urgent"}


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    text_lower = text.lower()
    letters = [char for char in text if char.isalpha()]
    uppercase_ratio = sum(char.isupper() for char in letters) / max(1, len(letters))

    return {
        "length": len(text),
        "word_count": len(text.split()),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "uppercase_ratio": uppercase_ratio,
        "has_refund": int("refund" in text_lower),
        "has_help": int("help" in text_lower),
        "has_error": int("error" in text_lower),
        "has_not_working": int("not working" in text_lower),
        "has_issue": int("issue" in text_lower),
        "has_problem": int("problem" in text_lower),
        "has_delay": int("delay" in text_lower or "delayed" in text_lower),
        "has_waiting": int("waiting" in text_lower),
        "has_login": int("login" in text_lower or "log in" in text_lower),
        "has_charge": int("charged" in text_lower or "charge" in text_lower),
        "has_cancel": int("cancel" in text_lower),
        "has_delivery": int("delivery" in text_lower),
        "has_still": int("still" in text_lower),
        "has_again": int("again" in text_lower),
    }


def build_feature_frame(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame([extract_features(text) for text in texts]).fillna(0)


@lru_cache(maxsize=1)
def load_priority_model():
    if not MODEL_PATH.exists():
        raise ValueError(
            "ML model file is missing. Generate model/best_logistic_regression.joblib "
            "from the notebook before calling /priority-predict."
        )
    return joblib.load(MODEL_PATH)


def predict_priority(text: str) -> dict[str, object]:
    model = load_priority_model()
    clean_text = normalize_text(text)
    features = build_feature_frame([clean_text])

    predicted_code = int(model.predict(features)[0])
    predicted_label = LABEL_MAP.get(predicted_code, str(predicted_code))

    probabilities: dict[str, float] = {}
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(features)[0]
        for class_code, probability in zip(model.named_steps["model"].classes_, probability_values, strict=True):
            label = LABEL_MAP.get(int(class_code), str(class_code))
            probabilities[label] = float(probability)
        confidence = float(max(probabilities.values(), default=0.0))

    return {
        "text": text,
        "clean_text": clean_text,
        "predicted_priority": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }
