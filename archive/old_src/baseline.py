"""
Baseline forecasting models for Sensor 218 PM2.5.

Baseline:
    Persistence / naive forecast

The persistence baseline is evaluated on the EXACT SAME TEST
OBSERVATIONS as the XGBoost models by using the timestamps
contained in the XGBoost prediction files.

Forecast horizons:
    - 48 hours
    - 72 hours
    - 7 days (168 hours)
    - 14 days (336 hours)
    - 30 days (720 hours)

Outputs:
    outputs/reports/sensor_218_baseline_comparison.json
    outputs/reports/sensor_218_baseline_<horizon>_predictions.csv
    outputs/figures/sensor_218_baseline_<horizon>_comparison.png
    outputs/figures/sensor_218_baseline_vs_xgboost_mae.png
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

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_sensor_218.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
# LOAD DATA
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Load the engineered Sensor 218 dataset.
    """

    print("=" * 80)
    print("LOADING SENSOR 218 DATA FOR BASELINE")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    required_columns = {
        "timestamp",
        "pm25",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns:\n"
            + "\n".join(sorted(missing_columns))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    ).copy()

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Rows loaded: {len(df):,}"
    )

    return df


# =============================================================================
# LOAD XGBOOST PREDICTIONS
# =============================================================================

def load_xgboost_predictions(
    horizon_name: str,
) -> pd.DataFrame:
    """
    Load predictions produced by train.py.

    These timestamps define the exact observations that were
    evaluated by XGBoost.
    """

    path = (
        REPORT_DIR
        / f"sensor_218_xgboost_{horizon_name}_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"XGBoost prediction file not found:\n{path}\n\n"
            "Run src\\train.py first."
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "timestamp",
        "actual_pm25",
        "predicted_pm25",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "XGBoost prediction file is missing:\n"
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
# CREATE PERSISTENCE BASELINE
# =============================================================================

def create_persistence_baseline(
    df: pd.DataFrame,
    xgb_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the persistence prediction for the EXACT SAME
    timestamps evaluated by XGBoost.

    Persistence rule:

        predicted PM2.5 at t+h
        =
        PM2.5 at time t

    The XGBoost prediction timestamp represents the forecast
    origin time t.

    Therefore, the baseline prediction is simply the PM2.5
    value at the same timestamp.
    """

    baseline_source = df[
        [
            "timestamp",
            "pm25",
        ]
    ].copy()

    baseline_source = baseline_source.rename(
        columns={
            "pm25": "baseline_prediction"
        }
    )

    comparison = xgb_predictions[
        [
            "timestamp",
            "actual_pm25",
            "predicted_pm25",
        ]
    ].merge(
        baseline_source,
        on="timestamp",
        how="left",
    )

    return comparison


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    """
    Calculate regression metrics.
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
# EVALUATE HORIZON
# =============================================================================

def evaluate_horizon(
    df: pd.DataFrame,
    horizon_name: str,
    horizon_hours: int,
) -> dict:
    """
    Evaluate persistence baseline against XGBoost using
    EXACTLY the same test observations.
    """

    print("\n")
    print("#" * 80)
    print(
        f"BASELINE EVALUATION: {horizon_name.upper()}"
    )
    print("#" * 80)

    # -------------------------------------------------------------------------
    # Load exact XGBoost test predictions
    # -------------------------------------------------------------------------

    xgb_test = load_xgboost_predictions(
        horizon_name
    )

    print(
        f"\nXGBoost test observations: {len(xgb_test):,}"
    )

    # -------------------------------------------------------------------------
    # Create persistence baseline using identical timestamps
    # -------------------------------------------------------------------------

    comparison = create_persistence_baseline(
        df,
        xgb_test,
    )

    # -------------------------------------------------------------------------
    # Remove any rows where baseline value is unavailable
    # -------------------------------------------------------------------------

    comparison = comparison.dropna(
        subset=[
            "actual_pm25",
            "predicted_pm25",
            "baseline_prediction",
        ]
    ).copy()

    comparison = comparison.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Verify actual values match
    # -------------------------------------------------------------------------

    # The actual value in the XGBoost prediction file is the
    # future PM2.5 target. We do NOT compare it to the current
    # PM2.5 baseline value because they represent different times.

    # -------------------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------------------

    baseline_metrics = calculate_metrics(
        comparison["actual_pm25"],
        comparison["baseline_prediction"],
    )

    xgb_metrics = calculate_metrics(
        comparison["actual_pm25"],
        comparison["predicted_pm25"],
    )

    # -------------------------------------------------------------------------
    # Calculate improvement
    # -------------------------------------------------------------------------

    if baseline_metrics["mae"] != 0:

        mae_improvement = (
            (
                baseline_metrics["mae"]
                - xgb_metrics["mae"]
            )
            / baseline_metrics["mae"]
            * 100
        )

    else:

        mae_improvement = 0.0

    if baseline_metrics["rmse"] != 0:

        rmse_improvement = (
            (
                baseline_metrics["rmse"]
                - xgb_metrics["rmse"]
            )
            / baseline_metrics["rmse"]
            * 100
        )

    else:

        rmse_improvement = 0.0

    # -------------------------------------------------------------------------
    # Determine whether XGBoost beats baseline
    # -------------------------------------------------------------------------

    xgb_better_mae = (
        xgb_metrics["mae"]
        < baseline_metrics["mae"]
    )

    xgb_better_rmse = (
        xgb_metrics["rmse"]
        < baseline_metrics["rmse"]
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print(
        f"\nRows evaluated: {len(comparison):,}"
    )

    print("\nPersistence Baseline:")
    print(
        f"MAE:  {baseline_metrics['mae']:.4f}"
    )
    print(
        f"RMSE: {baseline_metrics['rmse']:.4f}"
    )
    print(
        f"R²:   {baseline_metrics['r2']:.4f}"
    )
    print(
        f"MAPE: {baseline_metrics['mape']:.2f}%"
    )

    print("\nXGBoost:")
    print(
        f"MAE:  {xgb_metrics['mae']:.4f}"
    )
    print(
        f"RMSE: {xgb_metrics['rmse']:.4f}"
    )
    print(
        f"R²:   {xgb_metrics['r2']:.4f}"
    )
    print(
        f"MAPE: {xgb_metrics['mape']:.2f}%"
    )

    print("\nXGBoost Improvement:")
    print(
        f"MAE improvement:  {mae_improvement:.2f}%"
    )
    print(
        f"RMSE improvement: {rmse_improvement:.2f}%"
    )

    print(
        f"\nXGBoost beats baseline on MAE: "
        f"{xgb_better_mae}"
    )

    print(
        f"XGBoost beats baseline on RMSE: "
        f"{xgb_better_rmse}"
    )

    # -------------------------------------------------------------------------
    # Save aligned predictions
    # -------------------------------------------------------------------------

    prediction_path = (
        REPORT_DIR
        / f"sensor_218_baseline_{horizon_name}_predictions.csv"
    )

    comparison.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"\nPredictions saved: {prediction_path}"
    )

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------

    return {
        "horizon": horizon_name,
        "horizon_hours": horizon_hours,
        "rows_evaluated": int(
            len(comparison)
        ),
        "baseline": baseline_metrics,
        "xgboost": xgb_metrics,
        "xgboost_improvement": {
            "mae_percent": float(
                mae_improvement
            ),
            "rmse_percent": float(
                rmse_improvement
            ),
        },
        "xgboost_beats_baseline": {
            "mae": bool(
                xgb_better_mae
            ),
            "rmse": bool(
                xgb_better_rmse
            ),
        },
    }


# =============================================================================
# COMPARISON PLOT
# =============================================================================

def plot_comparison(
    horizon_name: str,
) -> Path:
    """
    Create actual vs persistence baseline vs XGBoost
    using the aligned prediction file.
    """

    prediction_path = (
        REPORT_DIR
        / f"sensor_218_baseline_{horizon_name}_predictions.csv"
    )

    df = pd.read_csv(
        prediction_path
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
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
        df["baseline_prediction"],
        label="Persistence Baseline",
        linewidth=1,
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
        f"Sensor 218: {horizon_name} "
        "Persistence Baseline vs XGBoost"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    output_path = (
        FIGURE_DIR
        / f"sensor_218_baseline_{horizon_name}_comparison.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Figure saved: {output_path}"
    )

    return output_path


# =============================================================================
# SUMMARY PLOT
# =============================================================================

def plot_summary(
    results: list[dict],
) -> Path:
    """
    Compare XGBoost and persistence baseline MAE
    across forecast horizons.
    """

    horizons = [
        result["horizon"]
        for result in results
    ]

    baseline_mae = [
        result["baseline"]["mae"]
        for result in results
    ]

    xgb_mae = [
        result["xgboost"]["mae"]
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
        baseline_mae,
        width,
        label="Persistence Baseline",
    )

    plt.bar(
        x + width / 2,
        xgb_mae,
        width,
        label="XGBoost",
    )

    plt.xticks(
        x,
        horizons,
    )

    plt.xlabel(
        "Forecast Horizon"
    )

    plt.ylabel(
        "MAE"
    )

    plt.title(
        "Persistence Baseline vs XGBoost MAE"
    )

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    output_path = (
        FIGURE_DIR
        / "sensor_218_baseline_vs_xgboost_mae.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Summary figure saved: {output_path}"
    )

    return output_path


# =============================================================================
# SAVE SUMMARY
# =============================================================================

def save_summary(
    results: list[dict],
) -> Path:
    """
    Save baseline comparison results.
    """

    output_path = (
        REPORT_DIR
        / "sensor_218_baseline_comparison.json"
    )

    summary = {
        "model": "XGBoost",
        "baseline": "Persistence",
        "sensor": 218,
        "evaluation_method": (
            "Baseline and XGBoost evaluated on the "
            "exact same timestamps from XGBoost test predictions."
        ),
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
        f"\nSummary saved: {output_path}"
    )

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")
    print("=" * 80)
    print("SENSOR 218 BASELINE VS XGBOOST")
    print("=" * 80)

    df = load_data()

    results = []

    for horizon_name, horizon_hours in HORIZONS.items():

        result = evaluate_horizon(
            df,
            horizon_name,
            horizon_hours,
        )

        results.append(
            result
        )

        plot_comparison(
            horizon_name
        )

    plot_summary(
        results
    )

    save_summary(
        results
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL BASELINE VS XGBOOST RESULTS")
    print("=" * 80)

    print(
        f"{'HORIZON':<12}"
        f"{'BASELINE MAE':<16}"
        f"{'XGB MAE':<16}"
        f"{'BASELINE R²':<16}"
        f"{'XGB R²':<16}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['horizon']:<12}"
            f"{result['baseline']['mae']:<16.4f}"
            f"{result['xgboost']['mae']:<16.4f}"
            f"{result['baseline']['r2']:<16.4f}"
            f"{result['xgboost']['r2']:<16.4f}"
        )

    print("\n")
    print("=" * 80)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()