"""
Final XGBoost Multi-Horizon PM2.5 Forecasting Model

Forecast horizons:
    - 48 hours
    - 72 hours
    - 7 days (168 hours)
    - 14 days (336 hours)
    - 30 days (720 hours)

Uses:
    data/processed/engineered_sensor_218.csv

Outputs:
    models/sensor_218_xgboost_<horizon>.pkl
    outputs/reports/sensor_218_xgboost_<horizon>_report.json
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_sensor_218.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# These are the engineered features already present in
# engineered_sensor_218.csv.

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
# DISPLAY HELPERS
# =============================================================================

def print_separator(char="=", length=80):
    print(char * length)


def print_header(title):
    print()
    print_separator()
    print(title)
    print_separator()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    print_header("LOADING ENGINEERED SENSOR 218 DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a 'timestamp' column.")

    if "pm25" not in df.columns:
        raise ValueError("Dataset must contain a 'pm25' column.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns:     {len(df.columns)}")

    return df


# =============================================================================
# TARGET CREATION
# =============================================================================

def create_future_target(df, horizon_hours):
    """
    Create a future PM2.5 target.

    Example:
        horizon = 48

        target_pm25_48h at time t
        = PM2.5 at time t + 48 hours
    """

    target_column = f"target_pm25_{horizon_hours}h"

    df = df.copy()

    df[target_column] = df["pm25"].shift(-horizon_hours)

    return df, target_column


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(df, horizon_hours):
    print()
    print("-" * 80)
    print(f"PREPARING {horizon_hours}h XGBOOST FORECASTING DATA")
    print("-" * 80)

    df, target_column = create_future_target(
        df,
        horizon_hours,
    )

    print(f"Target:   {target_column}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # -------------------------------------------------------------------------
    # Remove rows where the future target is unavailable.
    # -------------------------------------------------------------------------

    before_target = len(df)

    df = df.dropna(
        subset=[target_column]
    ).copy()

    removed_target = before_target - len(df)

    print(
        f"Rows removed due to missing future target: "
        f"{removed_target}"
    )

    # -------------------------------------------------------------------------
    # Check which requested features actually exist.
    # -------------------------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following required features are missing from the dataset:\n"
            + "\n".join(missing_features)
        )

    # -------------------------------------------------------------------------
    # Remove rows with missing feature values.
    # -------------------------------------------------------------------------

    before_features = len(df)

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    removed_features = before_features - len(df)

    print(
        f"Rows removed due to missing features: "
        f"{removed_features}"
    )

    print(f"Final modelling rows: {len(df):,}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[target_column].copy()

    timestamps = df["timestamp"].copy()

    return X, y, timestamps, target_column


# =============================================================================
# CHRONOLOGICAL SPLIT
# =============================================================================

def chronological_split(X, y, timestamps):
    """
    Split the data chronologically.

    70% training
    15% validation
    15% testing

    No shuffling is used because this is a time-series forecasting problem.
    """

    print_header("CHRONOLOGICAL DATA SPLIT")

    n = len(X)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()

    X_validation = X.iloc[
        train_end:validation_end
    ].copy()

    y_validation = y.iloc[
        train_end:validation_end
    ].copy()

    X_test = X.iloc[
        validation_end:
    ].copy()

    y_test = y.iloc[
        validation_end:
    ].copy()

    timestamps_test = timestamps.iloc[
        validation_end:
    ].copy()

    print(
        f"Training rows:   {len(X_train):,} "
        f"(70%)"
    )

    print(
        f"Validation rows: {len(X_validation):,} "
        f"(15%)"
    )

    print(
        f"Testing rows:    {len(X_test):,} "
        f"(15%)"
    )

    if len(timestamps_test) > 0:
        print(
            f"Test start:      {timestamps_test.iloc[0]}"
        )

        print(
            f"Test end:        {timestamps_test.iloc[-1]}"
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
    """
    Create the final XGBoost regression model.
    """

    model = XGBRegressor(
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

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def calculate_metrics(y_true, predictions):
    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# =============================================================================
# TRAIN ONE HORIZON
# =============================================================================

def train_horizon(df, horizon_name, horizon_hours):
    print()
    print("#" * 80)
    print(f"FORECAST HORIZON: {horizon_name.upper()}")
    print("#" * 80)

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------

    (
        X,
        y,
        timestamps,
        target_column,
    ) = prepare_data(
        df,
        horizon_hours,
    )

    # -------------------------------------------------------------------------
    # Chronological split
    # -------------------------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        timestamps_test,
    ) = chronological_split(
        X,
        y,
        timestamps,
    )

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------

    print_header("TRAINING XGBOOST")

    model = create_xgboost_model()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    print("XGBoost training complete.")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print_header("VALIDATION EVALUATION")

    validation_predictions = model.predict(
        X_validation
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    print(
        f"MAE:  {validation_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {validation_metrics['rmse']:.4f}"
    )

    print(
        f"R²:   {validation_metrics['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    print_header("TEST EVALUATION")

    test_predictions = model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print(
        f"MAE:  {test_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {test_metrics['rmse']:.4f}"
    )

    print(
        f"R²:   {test_metrics['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    print_header("TOP FEATURE IMPORTANCE")

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(
        ascending=False
    )

    top_features = importance.head(15)

    for feature, value in top_features.items():
        print(
            f"{feature:<30} {value:.6f}"
        )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_filename = (
        f"sensor_218_xgboost_{horizon_name}.pkl"
    )

    model_path = MODEL_DIR / model_filename

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "target": target_column,
            "horizon_hours": horizon_hours,
            "horizon_name": horizon_name,
        },
        model_path,
    )

    # -------------------------------------------------------------------------
    # Save predictions for later analysis
    # -------------------------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "timestamp": timestamps_test.values,
            "actual_pm25": y_test.values,
            "predicted_pm25": test_predictions,
            "error": (
                y_test.values
                - test_predictions
            ),
        }
    )

    prediction_path = (
        REPORT_DIR
        / f"sensor_218_xgboost_{horizon_name}_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------

    report = {
        "model": "XGBoost",
        "sensor": 218,
        "forecast_horizon": horizon_name,
        "forecast_horizon_hours": horizon_hours,
        "dataset": str(DATA_PATH),
        "rows_total_after_preparation": int(len(X)),
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "split": {
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "testing_rows": int(len(X_test)),
            "training_percentage": 70,
            "validation_percentage": 15,
            "testing_percentage": 15,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": {
            str(feature): float(value)
            for feature, value in importance.items()
        },
        "top_features": {
            str(feature): float(value)
            for feature, value in top_features.items()
        },
        "model_parameters": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    }

    report_filename = (
        f"sensor_218_xgboost_{horizon_name}_report.json"
    )

    report_path = REPORT_DIR / report_filename

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

    print_header("MODEL SAVED")

    print(
        f"File: {model_path}"
    )

    print(
        f"Predictions saved: {prediction_path}"
    )

    print(
        f"Report saved: {report_path}"
    )

    return {
        "horizon": horizon_name,
        "horizon_hours": horizon_hours,
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],
        "model_path": str(model_path),
        "report_path": str(report_path),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 80)
    print("FINAL XGBOOST MULTI-HORIZON PM2.5 FORECASTING")
    print("=" * 80)

    df = load_data()

    results = []

    for horizon_name, horizon_hours in HORIZONS.items():

        result = train_horizon(
            df,
            horizon_name,
            horizon_hours,
        )

        results.append(result)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_header(
        "FINAL XGBOOST MULTI-HORIZON RESULTS"
    )

    print(
        f"{'HORIZON':<12}"
        f"{'MAE':<15}"
        f"{'RMSE':<15}"
        f"{'R²':<15}"
    )

    print("-" * 57)

    for result in results:

        print(
            f"{result['horizon']:<12}"
            f"{result['mae']:<15.4f}"
            f"{result['rmse']:<15.4f}"
            f"{result['r2']:<15.4f}"
        )

    # =========================================================================
    # SAVE MULTI-HORIZON SUMMARY
    # =========================================================================

    summary_path = (
        REPORT_DIR
        / "sensor_218_xgboost_multi_horizon_summary.json"
    )

    summary = {
        "model": "XGBoost",
        "sensor": 218,
        "results": results,
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
    print("FINAL XGBOOST TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()