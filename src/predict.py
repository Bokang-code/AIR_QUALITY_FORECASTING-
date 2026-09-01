"""
Prediction utilities for the Sensor 218 XGBoost PM2.5 forecasting project.

Supports:
    - 48-hour forecast
    - 72-hour forecast
    - 7-day forecast
    - 14-day forecast
    - 30-day forecast
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"


# =============================================================================
# FORECAST HORIZONS
# =============================================================================

HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(
    model_path: str | Path,
) -> dict[str, Any]:
    """
    Load a saved XGBoost model package.

    The model files created by train.py contain:
        - model
        - features
        - target
        - horizon_hours
        - horizon_name
    """

    model_path = Path(model_path)

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    model_data = joblib.load(
        model_path
    )

    if not isinstance(
        model_data,
        dict,
    ):
        raise ValueError(
            "The saved model does not contain the expected model package."
        )

    required_keys = {
        "model",
        "features",
        "target",
        "horizon_hours",
        "horizon_name",
    }

    missing_keys = (
        required_keys
        - set(model_data.keys())
    )

    if missing_keys:
        raise ValueError(
            "Saved model is missing required information:\n"
            + "\n".join(sorted(missing_keys))
        )

    return model_data


# =============================================================================
# LOAD MODEL BY HORIZON
# =============================================================================

def load_horizon_model(
    horizon_name: str,
) -> dict[str, Any]:
    """
    Load the XGBoost model for a specific forecast horizon.

    Example:
        load_horizon_model("48h")
    """

    if horizon_name not in HORIZONS:
        raise ValueError(
            f"Unknown forecast horizon: {horizon_name}. "
            f"Available horizons: {list(HORIZONS.keys())}"
        )

    model_path = (
        MODEL_DIR
        / f"sensor_218_xgboost_{horizon_name}.pkl"
    )

    return load_model(
        model_path
    )


# =============================================================================
# VALIDATE FEATURES
# =============================================================================

def validate_features(
    features: pd.DataFrame,
    expected_features: list[str],
) -> pd.DataFrame:
    """
    Ensure the prediction data contains exactly the required features.
    """

    if features.empty:
        raise ValueError(
            "Feature DataFrame is empty."
        )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Prediction data is missing required features:\n"
            + "\n".join(missing_features)
        )

    return features[
        expected_features
    ].copy()


# =============================================================================
# CREATE PREDICTIONS
# =============================================================================

def predict_values(
    model_data: dict[str, Any],
    features: pd.DataFrame,
) -> pd.Series:
    """
    Generate PM2.5 predictions using a loaded model package.
    """

    if features.empty:
        raise ValueError(
            "Feature DataFrame is empty."
        )

    model = model_data["model"]

    expected_features = model_data[
        "features"
    ]

    X = validate_features(
        features,
        expected_features,
    )

    predictions = model.predict(
        X
    )

    return pd.Series(
        predictions,
        index=features.index,
        name="predicted_pm25",
    )


# =============================================================================
# PM2.5 HEALTH RISK CLASSIFICATION
# =============================================================================

def predict_health_risk(
    predictions: pd.Series,
) -> pd.Series:
    """
    Classify predicted PM2.5 concentrations.

    Categories are based on the PM2.5 breakpoints used by the project.
    """

    def classify(
        value: float,
    ) -> str:

        if value <= 12:
            return "Low"

        if value <= 35.4:
            return "Moderate"

        if value <= 55.4:
            return "High"

        return "Very High"

    return predictions.apply(
        classify
    ).astype("string")


# =============================================================================
# CREATE FORECAST DATAFRAME
# =============================================================================

def create_forecast(
    model_data: dict[str, Any],
    features: pd.DataFrame,
    timestamps: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Generate predictions and return them in a structured DataFrame.
    """

    predictions = predict_values(
        model_data,
        features,
    )

    risk = predict_health_risk(
        predictions
    )

    forecast = pd.DataFrame(
        {
            "predicted_pm25": predictions,
            "health_risk": risk,
        },
        index=features.index,
    )

    if timestamps is not None:
        forecast.insert(
            0,
            "timestamp",
            pd.to_datetime(
                timestamps,
                utc=True,
                errors="coerce",
            ),
        )

    forecast["forecast_horizon"] = (
        model_data["horizon_name"]
    )

    forecast["horizon_hours"] = (
        model_data["horizon_hours"]
    )

    return forecast


# =============================================================================
# PREDICT ALL HORIZONS
# =============================================================================

def predict_all_horizons(
    features: pd.DataFrame,
    timestamps: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate predictions for all five forecast horizons.

    Returns
    -------
    dict
        Dictionary containing one forecast DataFrame per horizon.
    """

    forecasts = {}

    for horizon_name in HORIZONS:

        print(
            f"Generating {horizon_name} forecast..."
        )

        model_data = load_horizon_model(
            horizon_name
        )

        forecast = create_forecast(
            model_data,
            features,
            timestamps,
        )

        forecasts[
            horizon_name
        ] = forecast

    return forecasts


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Display available prediction horizons.
    """

    print("=" * 80)
    print("SENSOR 218 XGBOOST PREDICTION MODULE")
    print("=" * 80)

    print("\nAvailable forecast horizons:")

    for horizon_name, hours in HORIZONS.items():

        model_path = (
            MODEL_DIR
            / f"sensor_218_xgboost_{horizon_name}.pkl"
        )

        status = (
            "AVAILABLE"
            if model_path.exists()
            else "NOT FOUND"
        )

        print(
            f"  {horizon_name:<6} "
            f"({hours:>3} hours)  "
            f"{status}"
        )


if __name__ == "__main__":
    main()