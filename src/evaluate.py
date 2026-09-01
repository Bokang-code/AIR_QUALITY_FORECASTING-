"""
Evaluation utilities for the Sensor 218 XGBoost PM2.5 forecasting project.

Evaluates the saved XGBoost predictions for:
    - 48 hours
    - 72 hours
    - 7 days
    - 14 days
    - 30 days

Outputs:
    outputs/reports/
    outputs/figures/
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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
# LOAD PREDICTIONS
# =============================================================================

def load_predictions(horizon_name: str) -> pd.DataFrame:
    """
    Load saved XGBoost predictions for a forecast horizon.
    """

    file_path = (
        REPORT_DIR
        / f"sensor_218_xgboost_{horizon_name}_predictions.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "timestamp",
        "actual_pm25",
        "predicted_pm25",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Prediction file is missing required columns:\n"
            + "\n".join(sorted(missing_columns))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["actual_pm25"] = pd.to_numeric(
        df["actual_pm25"],
        errors="coerce",
    )

    df["predicted_pm25"] = pd.to_numeric(
        df["predicted_pm25"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "actual_pm25",
            "predicted_pm25",
        ]
    ).copy()

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# =============================================================================
# CALCULATE METRICS
# =============================================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    nonzero = actual != 0

    if nonzero.sum() > 0:
        mape = (
            np.mean(
                np.abs(
                    (
                        actual[nonzero]
                        - predicted[nonzero]
                    )
                    / actual[nonzero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


# =============================================================================
# EVALUATE ONE HORIZON
# =============================================================================

def evaluate_horizon(
    horizon_name: str,
) -> dict:
    """
    Evaluate one XGBoost forecast horizon.
    """

    print("\n" + "=" * 80)
    print(
        f"EVALUATING {horizon_name.upper()} XGBOOST FORECAST"
    )
    print("=" * 80)

    df = load_predictions(
        horizon_name
    )

    metrics = calculate_metrics(
        df["actual_pm25"],
        df["predicted_pm25"],
    )

    print(
        f"\nRows evaluated: {len(df):,}"
    )

    print(
        f"MAE:  {metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {metrics['rmse']:.4f}"
    )

    print(
        f"R²:   {metrics['r2']:.4f}"
    )

    print(
        f"MAPE: {metrics['mape']:.2f}%"
    )

    return {
        "horizon": horizon_name,
        "horizon_hours": HORIZONS[horizon_name],
        "rows_evaluated": int(len(df)),
        "metrics": metrics,
    }


# =============================================================================
# ACTUAL VS PREDICTED
# =============================================================================

def plot_actual_vs_predicted(
    horizon_name: str,
) -> Path:
    """
    Create actual vs predicted scatter plot.
    """

    df = load_predictions(
        horizon_name
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        df["actual_pm25"],
        df["predicted_pm25"],
        alpha=0.5,
        s=20,
    )

    minimum = min(
        df["actual_pm25"].min(),
        df["predicted_pm25"].min(),
    )

    maximum = max(
        df["actual_pm25"].max(),
        df["predicted_pm25"].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel(
        "Actual PM2.5 (µg/m³)"
    )

    plt.ylabel(
        "Predicted PM2.5 (µg/m³)"
    )

    plt.title(
        f"XGBoost {horizon_name}: Actual vs Predicted PM2.5"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    output_path = (
        FIGURE_DIR
        / f"sensor_218_xgboost_{horizon_name}_actual_vs_predicted.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    return output_path


# =============================================================================
# RESIDUAL PLOT
# =============================================================================

def plot_residuals(
    horizon_name: str,
) -> Path:
    """
    Create residual plot.
    """

    df = load_predictions(
        horizon_name
    )

    residuals = (
        df["actual_pm25"]
        - df["predicted_pm25"]
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        df["predicted_pm25"],
        residuals,
        alpha=0.5,
        s=20,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel(
        "Predicted PM2.5 (µg/m³)"
    )

    plt.ylabel(
        "Residual (Actual - Predicted)"
    )

    plt.title(
        f"XGBoost {horizon_name}: Residuals"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    output_path = (
        FIGURE_DIR
        / f"sensor_218_xgboost_{horizon_name}_residuals.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    return output_path


# =============================================================================
# FORECAST TIME SERIES
# =============================================================================

def plot_forecast(
    horizon_name: str,
) -> Path:
    """
    Plot actual and predicted PM2.5 across the test period.
    """

    df = load_predictions(
        horizon_name
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        df["timestamp"],
        df["actual_pm25"],
        label="Actual",
        linewidth=1.5,
    )

    plt.plot(
        df["timestamp"],
        df["predicted_pm25"],
        label="XGBoost",
        linewidth=1,
    )

    plt.xlabel(
        "Timestamp"
    )

    plt.ylabel(
        "PM2.5 (µg/m³)"
    )

    plt.title(
        f"Sensor 218: {horizon_name} PM2.5 Forecast"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    output_path = (
        FIGURE_DIR
        / f"sensor_218_xgboost_{horizon_name}_forecast.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    return output_path


# =============================================================================
# SUMMARY PLOT
# =============================================================================

def plot_metric_comparison(
    results: list[dict],
) -> Path:
    """
    Compare MAE and RMSE across forecast horizons.
    """

    horizons = [
        result["horizon"]
        for result in results
    ]

    mae_values = [
        result["metrics"]["mae"]
        for result in results
    ]

    rmse_values = [
        result["metrics"]["rmse"]
        for result in results
    ]

    x = np.arange(
        len(horizons)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        mae_values,
        width,
        label="MAE",
    )

    plt.bar(
        x + width / 2,
        rmse_values,
        width,
        label="RMSE",
    )

    plt.xticks(
        x,
        horizons,
    )

    plt.xlabel(
        "Forecast Horizon"
    )

    plt.ylabel(
        "Error"
    )

    plt.title(
        "XGBoost Forecast Error by Horizon"
    )

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    output_path = (
        FIGURE_DIR
        / "sensor_218_xgboost_metric_comparison.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    return output_path


# =============================================================================
# SAVE SUMMARY
# =============================================================================

def save_summary(
    results: list[dict],
) -> Path:
    """
    Save evaluation summary as JSON.
    """

    output_path = (
        REPORT_DIR
        / "sensor_218_xgboost_evaluation_summary.json"
    )

    summary = {
        "model": "XGBoost",
        "sensor": 218,
        "forecast_horizons": HORIZONS,
        "results": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"\nEvaluation summary saved: {output_path}"
    )

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n")
    print("=" * 80)
    print("SENSOR 218 XGBOOST MODEL EVALUATION")
    print("=" * 80)

    results = []

    for horizon_name in HORIZONS:

        result = evaluate_horizon(
            horizon_name
        )

        results.append(
            result
        )

        plot_actual_vs_predicted(
            horizon_name
        )

        plot_residuals(
            horizon_name
        )

        plot_forecast(
            horizon_name
        )

    plot_metric_comparison(
        results
    )

    save_summary(
        results
    )

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()