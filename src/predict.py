from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_model(model_path: str | Path):
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the model") from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def predict_values(model, features: pd.DataFrame) -> pd.Series:
    if features.empty:
        raise ValueError("Feature DataFrame is empty")

    predictions = model.predict(features)
    return pd.Series(predictions, index=features.index, name="predicted_value")


def predict_health_risk(predictions: pd.Series) -> pd.Series:
    def classify(value: float) -> str:
        if value <= 12:
            return "Low"
        if value <= 35.4:
            return "Moderate"
        if value <= 55.4:
            return "High"
        return "Very High"

    return predictions.apply(classify).astype("string")
