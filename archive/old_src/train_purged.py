"""
Step 4.5 — Purged XGBoost Multi-Horizon PM2.5 Forecasting

Purpose:
    Test whether the original chronological split was affected by
    target-overlap across train/validation/test boundaries.

Split:
    70% training
    horizon-sized purge gap
    15% validation
    horizon-sized purge gap
    15% testing

Forecast horizons:
    48 hours
    72 hours
    7 days (168 hours)
    14 days (336 hours)
    30 days (720 hours)

This script DOES NOT modify the original train.py.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_sensor_218.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# SETTINGS
# =============================================================================

HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# =============================================================================
# FEATURES
# =============================================================================

FEATURE_COLUMNS = [
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


# =============================================================================
# DISPLAY
# =============================================================================

def print_separator(char="=", length=80):
    print(char * length)


def print_header(title):
    print()
    print_separator()
    print(title)
    print_separator()


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print_header(
        "LOADING ENGINEERED SENSOR 218 DATA"
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "timestamp" not in df.columns:
        raise ValueError(
            "Dataset must contain timestamp."
        )

    if "pm25" not in df.columns:
        raise ValueError(
            "Dataset must contain pm25."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns:     {len(df.columns)}"
    )

    print(
        f"Start:       {df['timestamp'].min()}"
    )

    print(
        f"End:         {df['timestamp'].max()}"
    )

    return df


# =============================================================================
# TARGET
# =============================================================================

def create_future_target(
    df,
    horizon_hours
):

    target_column = (
        f"target_pm25_{horizon_hours}h"
    )

    df = df.copy()

    df[target_column] = (
        df["pm25"]
        .shift(-horizon_hours)
    )

    return df, target_column


# =============================================================================
# PREPARE DATA
# =============================================================================

def prepare_data(
    df,
    horizon_hours
):

    print()
    print("-" * 80)
    print(
        f"PREPARING {horizon_hours}h "
        "FORECASTING DATA"
    )
    print("-" * 80)

    df, target_column = (
        create_future_target(
            df,
            horizon_hours
        )
    )

    print(
        f"Target:   {target_column}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    # -------------------------------------------------------------------------
    # Remove unavailable future targets
    # -------------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[target_column]
    ).copy()

    print(
        "Rows removed due to missing "
        f"future target: {before - len(df)}"
    )

    # -------------------------------------------------------------------------
    # Check features
    # -------------------------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features:\n"
            + "\n".join(
                missing_features
            )
        )

    # -------------------------------------------------------------------------
    # Remove rows where engineered features are unavailable
    # -------------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    print(
        "Rows removed due to missing "
        f"features: {before - len(df)}"
    )

    print(
        f"Final modelling rows: {len(df):,}"
    )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        target_column
    ].copy()

    timestamps = df[
        "timestamp"
    ].copy()

    return (
        X,
        y,
        timestamps,
        target_column,
    )


# =============================================================================
# PURGED CHRONOLOGICAL SPLIT
# =============================================================================

def purged_chronological_split(
    X,
    y,
    timestamps,
    horizon_hours,
):

    print_header(
        f"PURGED CHRONOLOGICAL SPLIT "
        f"({horizon_hours}H PURGE)"
    )

    n = len(X)

    # -------------------------------------------------------------------------
    # Initial chronological boundaries
    #
    # These are calculated from the full modelling dataset.
    # -------------------------------------------------------------------------

    train_end = int(
        n * TRAIN_SIZE
    )

    validation_end = int(
        n * (TRAIN_SIZE + VALIDATION_SIZE)
    )

    # -------------------------------------------------------------------------
    # Purge gaps
    #
    # Remove horizon_hours observations immediately before the validation
    # period and immediately before the test period.
    #
    # Example:
    #
    # TRAIN | PURGE | VALIDATION | PURGE | TEST
    # -------------------------------------------------------------------------

    train_end_purged = (
        train_end - horizon_hours
    )

    validation_start = train_end

    validation_end_purged = (
        validation_end - horizon_hours
    )

    test_start = validation_end

    if train_end_purged <= 0:
        raise ValueError(
            "Training set became empty after purge."
        )

    if validation_end_purged <= validation_start:
        raise ValueError(
            "Validation set became empty after purge."
        )

    if test_start >= n:
        raise ValueError(
            "Test set became empty."
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    X_train = X.iloc[
        :train_end_purged
    ].copy()

    y_train = y.iloc[
        :train_end_purged
    ].copy()

    timestamps_train = timestamps.iloc[
        :train_end_purged
    ].copy()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    X_validation = X.iloc[
        validation_start:
        validation_end_purged
    ].copy()

    y_validation = y.iloc[
        validation_start:
        validation_end_purged
    ].copy()

    timestamps_validation = timestamps.iloc[
        validation_start:
        validation_end_purged
    ].copy()

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    X_test = X.iloc[
        test_start:
    ].copy()

    y_test = y.iloc[
        test_start:
    ].copy()

    timestamps_test = timestamps.iloc[
        test_start:
    ].copy()

    # -------------------------------------------------------------------------
    # Purged rows
    # -------------------------------------------------------------------------

    train_validation_purge = (
        validation_start
        - train_end_purged
    )

    validation_test_purge = (
        test_start
        - validation_end_purged
    )

    print(
        f"Original rows:        {n:,}"
    )

    print()

    print(
        f"Training rows:        {len(X_train):,}"
    )

    print(
        f"Train purge gap:      "
        f"{train_validation_purge:,} rows"
    )

    print(
        f"Validation rows:      "
        f"{len(X_validation):,}"
    )

    print(
        f"Validation purge gap: "
        f"{validation_test_purge:,} rows"
    )

    print(
        f"Testing rows:         {len(X_test):,}"
    )

    print()

    if len(timestamps_train) > 0:

        print(
            "Training:"
        )

        print(
            f"  {timestamps_train.iloc[0]}"
            f" → "
            f"{timestamps_train.iloc[-1]}"
        )

    print()

    if len(timestamps_validation) > 0:

        print(
            "Validation:"
        )

        print(
            f"  {timestamps_validation.iloc[0]}"
            f" → "
            f"{timestamps_validation.iloc[-1]}"
        )

    print()

    if len(timestamps_test) > 0:

        print(
            "Testing:"
        )

        print(
            f"  {timestamps_test.iloc[0]}"
            f" → "
            f"{timestamps_test.iloc[-1]}"
        )

    print()

    print(
        f"[PURGED] {horizon_hours} hours removed "
        "at each split boundary."
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        timestamps_test,
    )


# =============================================================================
# MODEL
# =============================================================================

def create_xgboost_model():

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    predictions
):

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

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# =============================================================================
# TRAIN ONE HORIZON
# =============================================================================

def train_horizon(
    df,
    horizon_name,
    horizon_hours,
):

    print()
    print("#" * 80)

    print(
        f"PURGED FORECAST HORIZON: "
        f"{horizon_name.upper()}"
    )

    print("#" * 80)

    # -------------------------------------------------------------------------
    # Prepare
    # -------------------------------------------------------------------------

    (
        X,
        y,
        timestamps,
        target_column,
    ) = prepare_data(
        df,
        horizon_hours
    )

    # -------------------------------------------------------------------------
    # Purged split
    # -------------------------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        timestamps_test,
    ) = purged_chronological_split(
        X,
        y,
        timestamps,
        horizon_hours,
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print_header(
        "TRAINING XGBOOST"
    )

    model = create_xgboost_model()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation
            )
        ],
        verbose=False,
    )

    print(
        "XGBoost training complete."
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print_header(
        "VALIDATION EVALUATION"
    )

    validation_predictions = (
        model.predict(
            X_validation
        )
    )

    validation_metrics = (
        calculate_metrics(
            y_validation,
            validation_predictions
        )
    )

    print(
        f"MAE:  "
        f"{validation_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{validation_metrics['rmse']:.4f}"
    )

    print(
        f"R²:   "
        f"{validation_metrics['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    print_header(
        "TEST EVALUATION"
    )

    test_predictions = (
        model.predict(
            X_test
        )
    )

    test_metrics = (
        calculate_metrics(
            y_test,
            test_predictions
        )
    )

    print(
        f"MAE:  "
        f"{test_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{test_metrics['rmse']:.4f}"
    )

    print(
        f"R²:   "
        f"{test_metrics['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    print_header(
        "TOP FEATURE IMPORTANCE"
    )

    importance = (
        pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        )
        .sort_values(
            ascending=False
        )
    )

    for feature, value in (
        importance.head(15).items()
    ):

        print(
            f"{feature:<30}"
            f"{value:.6f}"
        )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "timestamp":
                timestamps_test.values,

            "actual_pm25":
                y_test.values,

            "predicted_pm25":
                test_predictions,

            "error":
                y_test.values
                - test_predictions,
        }
    )

    prediction_path = (
        REPORT_DIR
        / (
            "sensor_218_xgboost_"
            f"{horizon_name}_purged_predictions.csv"
        )
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / (
            "sensor_218_xgboost_"
            f"{horizon_name}_purged.pkl"
        )
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "target": target_column,
            "horizon_hours": horizon_hours,
            "horizon_name": horizon_name,
            "split_method": "purged_chronological",
            "purge_hours": horizon_hours,
        },
        model_path,
    )

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------

    report = {

        "model": "XGBoost",

        "sensor": 218,

        "forecast_horizon":
            horizon_name,

        "forecast_horizon_hours":
            horizon_hours,

        "split_method":
            "purged_chronological",

        "purge_hours":
            horizon_hours,

        "features":
            FEATURE_COLUMNS,

        "feature_count":
            len(FEATURE_COLUMNS),

        "split": {

            "training_rows":
                len(X_train),

            "validation_rows":
                len(X_validation),

            "testing_rows":
                len(X_test),

            "train_validation_purge_rows":
                horizon_hours,

            "validation_test_purge_rows":
                horizon_hours,
        },

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "top_features": {
            str(feature): float(value)
            for feature, value
            in importance.head(15).items()
        },
    }

    report_path = (
        REPORT_DIR
        / (
            "sensor_218_xgboost_"
            f"{horizon_name}_purged_report.json"
        )
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print_header(
        "PURGED MODEL SAVED"
    )

    print(
        f"Model:       {model_path}"
    )

    print(
        f"Predictions: {prediction_path}"
    )

    print(
        f"Report:      {report_path}"
    )

    return {
        "horizon":
            horizon_name,

        "horizon_hours":
            horizon_hours,

        "mae":
            test_metrics["mae"],

        "rmse":
            test_metrics["rmse"],

        "r2":
            test_metrics["r2"],

        "training_rows":
            len(X_train),

        "validation_rows":
            len(X_validation),

        "testing_rows":
            len(X_test),

        "purge_hours":
            horizon_hours,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)

    print(
        "STEP 4.5 — PURGED "
        "XGBOOST MULTI-HORIZON FORECASTING"
    )

    print("=" * 80)

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a separate experiment."
    )

    print(
        "The original src/train.py is NOT modified."
    )

    print()

    df = load_data()

    results = []

    for (
        horizon_name,
        horizon_hours
    ) in HORIZONS.items():

        result = train_horizon(
            df,
            horizon_name,
            horizon_hours,
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_header(
        "STEP 4.5 — PURGED RESULTS"
    )

    print(
        f"{'HORIZON':<12}"
        f"{'MAE':<15}"
        f"{'RMSE':<15}"
        f"{'R²':<15}"
    )

    print(
        "-" * 57
    )

    for result in results:

        print(
            f"{result['horizon']:<12}"
            f"{result['mae']:<15.4f}"
            f"{result['rmse']:<15.4f}"
            f"{result['r2']:<15.4f}"
        )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "sensor_218_step4_5_purged_summary.json"
    )

    summary = {
        "experiment":
            "Step 4.5 Purged XGBoost",

        "split_method":
            "purged_chronological",

        "results":
            results,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()

    print(
        f"Summary saved: {summary_path}"
    )

    print()

    print("=" * 80)

    print(
        "STEP 4.5 PURGED TRAINING COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()