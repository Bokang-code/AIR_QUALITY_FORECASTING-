"""
STEP 6 — V2 PURGED XGBOOST FORECASTING

Purpose:
    Test whether the expanded V2 feature set improves forecasting
    compared with the existing V1 model.

Important:
    - Existing train.py is NOT modified.
    - Existing train_purged.py is NOT modified.
    - Existing tuned models are NOT modified.
    - V2 features are loaded from sensor_218_hourly_v2.csv.
    - Horizon-specific purging is used.
    - Future PM2.5 values are NOT used as features.
    - XGBoost handles missing feature values natively.
    - Only rows with missing forecast targets are removed.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sensor_218_hourly_v2.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

SENSOR_ID = 218

HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_header(title, width=80):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_section(title, width=80):
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print_header("LOADING V2 ENGINEERED SENSOR 218 DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nV2 dataset not found:\n{DATA_PATH}\n\n"
            "Run feature_engineering_v2.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if "timestamp" not in df.columns:
        raise ValueError(
            "The V2 dataset does not contain a 'timestamp' column."
        )

    if "pm25" not in df.columns:
        raise ValueError(
            "The V2 dataset does not contain the 'pm25' column."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Rows loaded:    {len(df):,}")
    print(f"Columns loaded: {len(df):,}")
    print(f"Start:          {df['timestamp'].min()}")
    print(f"End:            {df['timestamp'].max()}")

    print(f"\nPM2.5 observations: {df['pm25'].notna().sum():,}")
    print(f"PM2.5 missing:      {df['pm25'].isna().sum():,}")

    return df


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def get_feature_columns(df):

    print_header("FEATURE SELECTION")

    # Columns that must NEVER be used as model inputs.
    excluded_exact = {
        "timestamp",
        "pm25",
        "target_pm25",
    }

    # Exclude every future target.
    # Examples:
    # target_pm25_48h
    # target_pm25_72h
    # target_pm25_168h
    # etc.
    excluded_future_targets = {
        column
        for column in df.columns
        if column.startswith("target_")
    }

    excluded = excluded_exact.union(excluded_future_targets)

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns were found.")

    print(f"Total V2 feature columns: {len(feature_columns)}")

    print("\nFirst 20 features:")

    for feature in feature_columns[:20]:
        print(f"  - {feature}")

    if len(feature_columns) > 20:
        print(
            f"  ... and {len(feature_columns) - 20} more"
        )

    # -------------------------------------------------------------------------
    # Leakage check
    # -------------------------------------------------------------------------

    print_section("FEATURE LEAKAGE CHECK")

    suspicious = []

    future_words = [
        "future",
        "forecast",
        "target",
        "lead",
        "ahead",
    ]

    for feature in feature_columns:

        lower = feature.lower()

        if any(word in lower for word in future_words):
            suspicious.append(feature)

    if suspicious:

        print("[WARNING] Potential future-looking feature names detected:")

        for feature in suspicious:
            print(f"  - {feature}")

        raise ValueError(
            "Potential future-looking features were detected. "
            "Review the feature list before training."
        )

    print("[PASS] No obviously future-looking feature names detected.")

    return feature_columns


# =============================================================================
# PREPARE HORIZON DATA
# =============================================================================

def prepare_horizon_data(df, horizon):

    print_section(
        f"PREPARING {horizon}H FORECASTING DATA"
    )

    target_column = f"target_pm25_{horizon}h"

    # -------------------------------------------------------------------------
    # Create target if it does not already exist.
    # -------------------------------------------------------------------------

    if target_column not in df.columns:

        print(
            f"[INFO] {target_column} not found."
        )

        print(
            f"[INFO] Creating target using PM2.5 shifted by {-horizon}."
        )

        df = df.copy()

        df[target_column] = df["pm25"].shift(-horizon)

    else:

        print(f"Target column found: {target_column}")

    # -------------------------------------------------------------------------
    # Do NOT drop rows because of missing FEATURES.
    #
    # XGBoost supports NaN feature values natively.
    # -------------------------------------------------------------------------

    before = len(df)

    target_available = df[target_column].notna()

    df_model = df.loc[target_available].copy()

    removed_target = before - len(df_model)

    print(
        f"Rows removed due to missing future target: "
        f"{removed_target:,}"
    )

    print(
        "Rows removed due to missing features: 0"
    )

    print(
        "Missing feature values are retained for XGBoost "
        "to handle natively."
    )

    print(
        f"Final modelling rows: {len(df_model):,}"
    )

    return df_model, target_column


# =============================================================================
# PURGED CHRONOLOGICAL SPLIT
# =============================================================================

def purged_chronological_split(
    df,
    horizon,
    train_ratio=0.70,
    validation_ratio=0.15,
):

    print_section(
        f"PURGED CHRONOLOGICAL SPLIT ({horizon}H PURGE)"
    )

    n = len(df)

    train_end = int(n * train_ratio)
    validation_end = int(
        n * (train_ratio + validation_ratio)
    )

    # -------------------------------------------------------------------------
    # First determine the chronological boundaries.
    # -------------------------------------------------------------------------

    train_raw = df.iloc[:train_end].copy()

    validation_raw = df.iloc[
        train_end:validation_end
    ].copy()

    test_raw = df.iloc[
        validation_end:
    ].copy()

    # -------------------------------------------------------------------------
    # Apply horizon-specific purge gaps.
    #
    # The purge is applied on BOTH sides of each boundary.
    # -------------------------------------------------------------------------

    train = df.iloc[
        : max(0, train_end - horizon)
    ].copy()

    validation = df.iloc[
        min(n, train_end + horizon):
        max(
            train_end + horizon,
            validation_end - horizon,
        )
    ].copy()

    test = df.iloc[
        min(n, validation_end + horizon):
    ].copy()

    print(f"Original rows:        {n:,}")
    print()
    print(f"Training rows:        {len(train):,}")
    print(f"Train purge gap:      {horizon} rows")
    print(f"Validation rows:      {len(validation):,}")
    print(f"Validation purge gap: {horizon} rows")
    print(f"Testing rows:         {len(test):,}")

    if len(train) > 0:
        print("\nTraining:")
        print(
            f"  {train['timestamp'].min()} "
            f"→ {train['timestamp'].max()}"
        )

    if len(validation) > 0:
        print("\nValidation:")
        print(
            f"  {validation['timestamp'].min()} "
            f"→ {validation['timestamp'].max()}"
        )

    if len(test) > 0:
        print("\nTesting:")
        print(
            f"  {test['timestamp'].min()} "
            f"→ {test['timestamp'].max()}"
        )

    print(
        f"\n[PURGED] {horizon} hours removed "
        "at each split boundary."
    )

    # -------------------------------------------------------------------------
    # Verify chronological ordering.
    # -------------------------------------------------------------------------

    if len(train) > 0 and len(validation) > 0:

        if (
            train["timestamp"].max()
            >= validation["timestamp"].min()
        ):
            raise ValueError(
                "Chronological ordering violation between "
                "training and validation."
            )

    if len(validation) > 0 and len(test) > 0:

        if (
            validation["timestamp"].max()
            >= test["timestamp"].min()
        ):
            raise ValueError(
                "Chronological ordering violation between "
                "validation and testing."
            )

    return train, validation, test


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions,
        )
    )

    r2 = r2_score(
        y,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions": predictions,
    }


# =============================================================================
# TRAIN ONE HORIZON
# =============================================================================

def train_horizon(
    df,
    feature_columns,
    horizon_name,
    horizon_hours,
):

    print("\n" + "#" * 80)
    print(
        f"V2 FORECAST HORIZON: "
        f"{horizon_name.upper()}"
    )
    print("#" * 80)

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------

    df_model, target_column = prepare_horizon_data(
        df,
        horizon_hours,
    )

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

    train_df, validation_df, test_df = (
        purged_chronological_split(
            df_model,
            horizon_hours,
        )
    )

    if len(train_df) == 0:
        raise ValueError(
            f"No training observations available "
            f"for {horizon_name}."
        )

    if len(validation_df) == 0:
        raise ValueError(
            f"No validation observations available "
            f"for {horizon_name}."
        )

    if len(test_df) == 0:
        raise ValueError(
            f"No test observations available "
            f"for {horizon_name}."
        )

    # -------------------------------------------------------------------------
    # Build X/y
    # -------------------------------------------------------------------------

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    # -------------------------------------------------------------------------
    # Convert features to numeric.
    #
    # Any non-numeric values become NaN, which XGBoost can handle.
    # -------------------------------------------------------------------------

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_validation = X_validation.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Check feature availability
    # -------------------------------------------------------------------------

    print_section("FEATURE MISSINGNESS")

    train_missing = int(
        X_train.isna().sum().sum()
    )

    validation_missing = int(
        X_validation.isna().sum().sum()
    )

    test_missing = int(
        X_test.isna().sum().sum()
    )

    print(
        f"Training missing feature cells:   "
        f"{train_missing:,}"
    )

    print(
        f"Validation missing feature cells: "
        f"{validation_missing:,}"
    )

    print(
        f"Testing missing feature cells:     "
        f"{test_missing:,}"
    )

    print(
        "\n[PASS] Missing feature values are retained."
    )

    print(
        "[INFO] XGBoost will handle missing values natively."
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print_header("TRAINING V2 XGBOOST")

    model = XGBRegressor(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    print("V2 XGBoost training complete.")

    # -------------------------------------------------------------------------
    # Validation evaluation
    # -------------------------------------------------------------------------

    print_header("VALIDATION EVALUATION")

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    print(
        f"MAE:  "
        f"{validation_results['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{validation_results['rmse']:.4f}"
    )

    print(
        f"R²:   "
        f"{validation_results['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Test evaluation
    # -------------------------------------------------------------------------

    print_header("TEST EVALUATION")

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print(
        f"MAE:  "
        f"{test_results['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{test_results['rmse']:.4f}"
    )

    print(
        f"R²:   "
        f"{test_results['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    print_header("TOP V2 FEATURE IMPORTANCE")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    )

    importance = importance.sort_values(
        ascending=False
    )

    print(
        importance.head(20).to_string()
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"sensor_{SENSOR_ID}_xgboost_"
          f"{horizon_name}_v2_purged.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].values,
            "actual_pm25": y_test.values,
            "predicted_pm25": test_results[
                "predictions"
            ],
        }
    )

    predictions_path = (
        REPORT_DIR
        / f"sensor_{SENSOR_ID}_xgboost_"
          f"{horizon_name}_v2_purged_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------

    report = {
        "sensor_id": SENSOR_ID,
        "horizon": horizon_name,
        "horizon_hours": horizon_hours,
        "model": "XGBoost",
        "feature_version": "V2",
        "purged": True,
        "feature_count": len(feature_columns),
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "missing_feature_cells": {
            "train": train_missing,
            "validation": validation_missing,
            "test": test_missing,
        },
        "validation": {
            "mae": validation_results["mae"],
            "rmse": validation_results["rmse"],
            "r2": validation_results["r2"],
        },
        "test": {
            "mae": test_results["mae"],
            "rmse": test_results["rmse"],
            "r2": test_results["r2"],
        },
        "xgboost_parameters": XGB_PARAMS,
        "top_features": {
            str(k): float(v)
            for k, v in importance.head(20).items()
        },
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }

    report_path = (
        REPORT_DIR
        / f"sensor_{SENSOR_ID}_xgboost_"
          f"{horizon_name}_v2_purged_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print_header("V2 MODEL SAVED")

    print(
        f"Model:       {model_path}"
    )

    print(
        f"Predictions: {predictions_path}"
    )

    print(
        f"Report:      {report_path}"
    )

    return {
        "horizon": horizon_name,
        "validation_mae": validation_results["mae"],
        "validation_rmse": validation_results["rmse"],
        "validation_r2": validation_results["r2"],
        "test_mae": test_results["mae"],
        "test_rmse": test_results["rmse"],
        "test_r2": test_results["r2"],
        "feature_count": len(feature_columns),
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "STEP 6: V2 PURGED XGBOOST FORECASTING"
    )

    print(
        """
IMPORTANT:
• Existing train.py is NOT modified.
• Existing train_purged.py is NOT modified.
• Existing tuned models are NOT modified.
• V2 features are used from sensor_218_hourly_v2.csv.
• Horizon-specific purging is used.
• No future PM2.5 values are used as features.
• Missing feature values are NOT discarded.
• XGBoost handles missing feature values natively.
• This experiment tests whether V2 features improve forecasting.
"""
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    feature_columns = get_feature_columns(df)

    # -------------------------------------------------------------------------
    # Train all horizons
    # -------------------------------------------------------------------------

    results = []

    for horizon_name, horizon_hours in HORIZONS.items():

        try:

            result = train_horizon(
                df=df,
                feature_columns=feature_columns,
                horizon_name=horizon_name,
                horizon_hours=horizon_hours,
            )

            results.append(result)

        except Exception as e:

            print("\n" + "=" * 80)
            print(
                f"[ERROR] {horizon_name} V2 training failed"
            )
            print("=" * 80)

            print(str(e))

            results.append(
                {
                    "horizon": horizon_name,
                    "error": str(e),
                }
            )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_header(
        "STEP 6: FINAL V2 RESULTS"
    )

    print(
        f"{'HORIZON':<10}"
        f"{'FEATURES':<12}"
        f"{'VAL MAE':<12}"
        f"{'TEST MAE':<12}"
        f"{'TEST RMSE':<14}"
        f"{'TEST R²':<12}"
    )

    print("-" * 80)

    for result in results:

        if "error" in result:

            print(
                f"{result['horizon']:<10}"
                f"ERROR: {result['error']}"
            )

            continue

        print(
            f"{result['horizon']:<10}"
            f"{result['feature_count']:<12}"
            f"{result['validation_mae']:<12.4f}"
            f"{result['test_mae']:<12.4f}"
            f"{result['test_rmse']:<14.4f}"
            f"{result['test_r2']:<12.4f}"
        )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / f"sensor_{SENSOR_ID}_step6_v2_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print_header(
        "STEP 6 V2 PURGED TRAINING COMPLETE"
    )

    print(
        f"\nSummary JSON:\n"
        f"  {summary_path}"
    )

    print(
        "\nExisting V1 files/models were not modified."
    )

    print(
        "\nNext decision:"
    )

    print(
        "Compare V2 against the V1 purged/tuned models."
    )

    print(
        "If V2 improves performance, we can tune the V2 model."
    )


if __name__ == "__main__":
    main()