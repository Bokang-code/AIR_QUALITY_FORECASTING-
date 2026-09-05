"""
STEP 4: PERSISTENCE + SEASONAL BASELINES VS XGBOOST

Compares:
    1. Persistence baseline
    2. Seasonal baseline
    3. Existing XGBoost predictions

Horizons:
    48h
    72h
    7d
    14d
    30d
"""

from pathlib import Path
import json

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

PROJECT_ROOT = Path(__file__).resolve().parent

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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# HORIZONS
# =============================================================================

HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    # Avoid division by zero
    non_zero = y_true != 0

    if non_zero.sum() > 0:
        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    - predictions[non_zero]
                )
                / y_true[non_zero]
            )
        ) * 100
    else:
        mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 80)
    print("STEP 4: BASELINE COMPARISON")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce"
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"\nRows loaded: {len(df):,}"
    )

    print(
        f"Missing PM2.5: "
        f"{df['pm25'].isna().sum():,}"
    )

    return df


# =============================================================================
# CREATE SEASONAL BASELINE
# =============================================================================

def create_seasonal_baseline(
    train_df,
    test_df
):
    """
    Predict PM2.5 using the historical average
    for the same month + hour.

    Example:
        For a test observation at:
            January, 14:00

        prediction =
            average PM2.5 historically observed
            during January at 14:00 in training data.
    """

    seasonal_means = (
        train_df
        .assign(
            month=train_df["timestamp"].dt.month,
            hour=train_df["timestamp"].dt.hour
        )
        .groupby(
            ["month", "hour"]
        )["pm25"]
        .mean()
    )

    test_month = test_df["timestamp"].dt.month
    test_hour = test_df["timestamp"].dt.hour

    predictions = []

    overall_mean = train_df["pm25"].mean()

    for month, hour in zip(
        test_month,
        test_hour
    ):

        key = (month, hour)

        if key in seasonal_means.index:
            prediction = seasonal_means.loc[key]
        else:
            prediction = overall_mean

        predictions.append(prediction)

    return np.array(predictions)


# =============================================================================
# EVALUATE ONE HORIZON
# =============================================================================

def evaluate_horizon(
    df,
    horizon_name,
    horizon_hours
):

    print()
    print("#" * 80)
    print(
        f"BASELINE EVALUATION: "
        f"{horizon_name.upper()}"
    )
    print("#" * 80)

    # -------------------------------------------------------------------------
    # Create future target
    # -------------------------------------------------------------------------

    target_column = (
        f"target_pm25_{horizon_hours}h"
    )

    working = df.copy()

    working[target_column] = (
        working["pm25"]
        .shift(-horizon_hours)
    )

    # -------------------------------------------------------------------------
    # Match the same modelling observations used by XGBoost
    # -------------------------------------------------------------------------

    required_features = [
        "pm25",
        "pm25_missing",
        "hour",
        "day",
        "month",
        "day_of_week",
        "weekend",
        "season_code",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm25_lag_6h",
        "pm25_lag_12h",
        "pm25_lag_24h",
        "pm25_lag_48h",
        "pm25_lag_168h",
        "pm25_rolling_mean_6h",
        "pm25_rolling_std_6h",
        "pm25_rolling_mean_24h",
        "pm25_rolling_std_24h",
        "pm25_rolling_mean_7d",
        "pm25_rolling_std_7d",
    ]

    working = working.dropna(
        subset=[target_column]
    ).copy()

    working = working.dropna(
        subset=required_features
    ).copy()

    # -------------------------------------------------------------------------
    # Same 70/15/15 chronological split as XGBoost
    # -------------------------------------------------------------------------

    n = len(working)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_df = working.iloc[
        :train_end
    ].copy()

    test_df = working.iloc[
        validation_end:
    ].copy()

    print(
        f"\nXGBoost test observations: "
        f"{len(test_df):,}"
    )

    # -------------------------------------------------------------------------
    # Actual target
    # -------------------------------------------------------------------------

    y_test = test_df[
        target_column
    ].values

    # -------------------------------------------------------------------------
    # Persistence baseline
    #
    # Uses current PM2.5 as the forecast.
    # -------------------------------------------------------------------------

    persistence_predictions = (
        test_df["pm25"].values
    )

    # -------------------------------------------------------------------------
    # Seasonal baseline
    # -------------------------------------------------------------------------

    seasonal_predictions = (
        create_seasonal_baseline(
            train_df,
            test_df
        )
    )

    # -------------------------------------------------------------------------
    # Load XGBoost predictions already generated by train.py
    # -------------------------------------------------------------------------

    prediction_path = (
        REPORT_DIR
        / f"sensor_218_xgboost_"
        f"{horizon_name}_predictions.csv"
    )

    if not prediction_path.exists():

        raise FileNotFoundError(
            f"\nXGBoost prediction file not found:\n"
            f"{prediction_path}\n\n"
            f"Run 'python src\\train.py' first."
        )

    xgb_df = pd.read_csv(
        prediction_path
    )

    xgb_predictions = (
        xgb_df["predicted_pm25"]
        .values
    )

    # -------------------------------------------------------------------------
    # Make sure lengths match
    # -------------------------------------------------------------------------

    if len(xgb_predictions) != len(y_test):

        raise ValueError(
            f"\nPrediction length mismatch for "
            f"{horizon_name}.\n"
            f"XGBoost predictions: "
            f"{len(xgb_predictions)}\n"
            f"Baseline observations: "
            f"{len(y_test)}"
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    persistence_metrics = calculate_metrics(
        y_test,
        persistence_predictions
    )

    seasonal_metrics = calculate_metrics(
        y_test,
        seasonal_predictions
    )

    xgb_metrics = calculate_metrics(
        y_test,
        xgb_predictions
    )

    # -------------------------------------------------------------------------
    # Improvement
    # -------------------------------------------------------------------------

    persistence_mae = (
        persistence_metrics["mae"]
    )

    seasonal_mae = (
        seasonal_metrics["mae"]
    )

    xgb_mae = (
        xgb_metrics["mae"]
    )

    improvement_vs_persistence = (
        (persistence_mae - xgb_mae)
        / persistence_mae
        * 100
    )

    improvement_vs_seasonal = (
        (seasonal_mae - xgb_mae)
        / seasonal_mae
        * 100
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print(
        f"\nRows evaluated: {len(y_test):,}"
    )

    print("\nPersistence Baseline:")

    print(
        f"MAE:  {persistence_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {persistence_metrics['rmse']:.4f}"
    )

    print(
        f"R²:   {persistence_metrics['r2']:.4f}"
    )

    print(
        f"MAPE: {persistence_metrics['mape']:.2f}%"
    )

    print("\nSeasonal Baseline:")

    print(
        f"MAE:  {seasonal_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {seasonal_metrics['rmse']:.4f}"
    )

    print(
        f"R²:   {seasonal_metrics['r2']:.4f}"
    )

    print(
        f"MAPE: {seasonal_metrics['mape']:.2f}%"
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
        f"vs Persistence MAE: "
        f"{improvement_vs_persistence:.2f}%"
    )

    print(
        f"vs Seasonal MAE:    "
        f"{improvement_vs_seasonal:.2f}%"
    )

    print(
        "\nXGBoost beats Persistence on MAE:",
        xgb_mae < persistence_mae
    )

    print(
        "XGBoost beats Seasonal on MAE:",
        xgb_mae < seasonal_mae
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    comparison_df = pd.DataFrame(
        {
            "timestamp": test_df[
                "timestamp"
            ].values,

            "actual_pm25": y_test,

            "persistence_prediction":
                persistence_predictions,

            "seasonal_prediction":
                seasonal_predictions,

            "xgboost_prediction":
                xgb_predictions,
        }
    )

    comparison_path = (
        REPORT_DIR
        / f"sensor_218_baseline_comparison_"
        f"{horizon_name}.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Return summary
    # -------------------------------------------------------------------------

    return {
        "horizon": horizon_name,
        "horizon_hours": horizon_hours,
        "observations": len(y_test),

        "persistence": persistence_metrics,

        "seasonal": seasonal_metrics,

        "xgboost": xgb_metrics,

        "xgboost_improvement_vs_persistence_mae_percent":
            float(improvement_vs_persistence),

        "xgboost_improvement_vs_seasonal_mae_percent":
            float(improvement_vs_seasonal),

        "xgboost_beats_persistence_mae":
            bool(xgb_mae < persistence_mae),

        "xgboost_beats_seasonal_mae":
            bool(xgb_mae < seasonal_mae),

        "comparison_file":
            str(comparison_path),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    df = load_data()

    results = []

    for horizon_name, horizon_hours in HORIZONS.items():

        result = evaluate_horizon(
            df,
            horizon_name,
            horizon_hours
        )

        results.append(result)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 4: FINAL BASELINE COMPARISON")
    print("=" * 80)

    print()

    print(
        f"{'HORIZON':<10}"
        f"{'PERSIST MAE':<15}"
        f"{'SEASONAL MAE':<15}"
        f"{'XGB MAE':<15}"
    )

    print("-" * 55)

    for result in results:

        print(
            f"{result['horizon']:<10}"
            f"{result['persistence']['mae']:<15.4f}"
            f"{result['seasonal']['mae']:<15.4f}"
            f"{result['xgboost']['mae']:<15.4f}"
        )

    # =========================================================================
    # SAVE JSON
    # =========================================================================

    summary_path = (
        REPORT_DIR
        / "sensor_218_step4_baseline_comparison.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        f"Summary saved: {summary_path}"
    )

    print()
    print("=" * 80)
    print("STEP 4 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()