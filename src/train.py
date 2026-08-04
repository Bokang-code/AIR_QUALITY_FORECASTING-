from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def prepare_training_data(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a simple supervised training dataset from engineered air quality data."""

    if input_path is None:
        input_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "engineered_air_quality_data.csv"
    else:
        input_path = Path(input_path)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Engineered data file not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    if dataframe.empty:
        raise ValueError("No data found in the engineered dataset.")

    dataframe = dataframe.copy()

    feature_columns = ["hour", "day_of_week", "month", "rolling_mean_3"]
    available_features = [column for column in feature_columns if column in dataframe.columns]

    if not available_features:
        raise ValueError("The engineered dataset does not contain the expected feature columns.")

    features = dataframe[available_features].copy()
    target = dataframe["value"].copy() if "value" in dataframe.columns else pd.Series([0] * len(dataframe), index=dataframe.index)

    features = features.fillna(0)
    target = target.fillna(0)

    output_file = output_dir / "training_data.csv"
    training_frame = pd.concat([features, target.rename("target")], axis=1)
    training_frame.to_csv(output_file, index=False)

    return features, target


def train_and_save_model(
    features, target, output_dir: str | Path | None = None, model_name: str = "random_forest.joblib"
):
    """Train a simple RandomForestRegressor and save the fitted model.

    Parameters
    ----------
    features: pd.DataFrame
        Feature matrix.
    target: pd.Series
        Target values.
    output_dir:
        Directory where the model artifact will be saved.
    model_name:
        Filename for the saved model.
    """

    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("scikit-learn is required for training the model") from exc

    try:
        import joblib
    except Exception:
        # joblib is bundled with sklearn but handle explicit import errors
        raise RuntimeError("joblib is required to save the model")

    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(features, target)

    model_path = output_dir / model_name
    joblib.dump(model, model_path)

    return model_path
