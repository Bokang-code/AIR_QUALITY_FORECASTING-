"""
FAIR RANDOM FOREST VS XGBOOST MODEL COMPARISON

Compares Random Forest and XGBoost using:
- Identical engineered features
- Identical future targets
- Identical chronological train/validation/test splits
- Multiple forecast horizons

Horizons:
- 48 hours
- 72 hours
- 7 days
- 14 days
- 30 days

Metrics:
- MAE
- RMSE
- R²

Outputs:
- Console comparison
- JSON report in outputs/reports/
- Trained models in models/
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "engineered_sensor_218.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


# These are the engineered predictor columns.
# target_pm25 is NOT included because it represents the current/target
# pollutant value and is handled separately.
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
# UTILITY FUNCTIONS
# =============================================================================

def print_separator(char="=", length=80):
    print(char * length)


def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE and R².
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    # R² requires at least two observations.
    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2) if not np.isnan(r2) else None,
    }


def create_future_target(df, horizon_hours):
    """
    Create a future PM2.5 target.

    Example:
        horizon_hours = 48

    target_pm25_48h at time t =
        PM2.5 at time t + 48 hours
    """

    data = df.copy()

    target_column = f"target_pm25_{horizon_hours}h"

    # Directly shift the PM2.5 value by the requested
    # forecasting horizon.
    #
    # Therefore:
    # 48h = PM2.5 exactly 48 hours into the future
    # 72h = PM2.5 exactly 72 hours into the future
    # 168h = PM2.5 exactly 7 days into the future
    # 336h = PM2.5 exactly 14 days into the future
    # 720h = PM2.5 exactly 30 days into the future
    data[target_column] = data["pm25"].shift(-horizon_hours)

    return data, target_column


def prepare_data(df, horizon_hours):
    """
    Prepare the exact same modelling dataset for both algorithms.
    """

    print()
    print("-" * 80)
    print(f"PREPARING {horizon_hours}h FORECAST DATA")
    print("-" * 80)

    data, target_column = create_future_target(
        df,
        horizon_hours
    )

    print(f"Target column created: {target_column}")

    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    if "target_pm25" not in data.columns:
        raise ValueError(
            "target_pm25 column is missing from the engineered dataset."
        )

    # -------------------------------------------------------------------------
    # Remove rows with missing future target
    # -------------------------------------------------------------------------

    before_target = len(data)

    data = data.dropna(
        subset=[target_column]
    ).copy()

    removed_target = before_target - len(data)

    print(
        f"Rows removed due to missing future target: "
        f"{removed_target}"
    )

    # -------------------------------------------------------------------------
    # Remove rows with missing features
    # -------------------------------------------------------------------------

    before_features = len(data)

    data = data.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    removed_features = before_features - len(data)

    print(
        f"Rows removed due to missing features: "
        f"{removed_features}"
    )

    # -------------------------------------------------------------------------
    # Sort chronologically
    # -------------------------------------------------------------------------

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Final modelling rows: {len(data):,}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    return data, target_column


def chronological_split(data, target_column):
    """
    70% training
    15% validation
    15% testing

    Chronological split prevents future information leaking into training.
    """

    n = len(data)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = data.iloc[:train_end].copy()

    validation = data.iloc[
        train_end:validation_end
    ].copy()

    test = data.iloc[
        validation_end:
    ].copy()

    print()
    print("=" * 80)
    print("CHRONOLOGICAL DATA SPLIT")
    print("=" * 80)

    print(
        f"Training rows:   {len(train):,} (70%)"
    )

    print(
        f"Validation rows: {len(validation):,} (15%)"
    )

    print(
        f"Testing rows:    {len(test):,} (15%)"
    )

    if len(test) > 0:
        print(
            f"Test start:      {test['timestamp'].iloc[0]}"
        )

        print(
            f"Test end:        {test['timestamp'].iloc[-1]}"
        )

    return train, validation, test


# =============================================================================
# RANDOM FOREST
# =============================================================================

def train_random_forest(X_train, y_train):
    """
    Train Random Forest regression model.
    """

    print()
    print("-" * 80)
    print("TRAINING RANDOM FOREST")
    print("-" * 80)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print("Random Forest training complete.")

    return model


# =============================================================================
# XGBOOST
# =============================================================================

def train_xgboost(X_train, y_train):
    """
    Train XGBoost regression model.
    """

    print()
    print("-" * 80)
    print("TRAINING XGBOOST")
    print("-" * 80)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print("XGBoost training complete.")

    return model


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(model, model_name):
    """
    Return top 15 feature importance values.
    """

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importance,
    })

    feature_importance = feature_importance.sort_values(
        "importance",
        ascending=False
    )

    top_features = feature_importance.head(15)

    print()
    print(f"TOP {model_name.upper()} FEATURE IMPORTANCE")

    for _, row in top_features.iterrows():
        print(
            f"{row['feature']:<30} "
            f"{row['importance']:.6f}"
        )

    return [
        {
            "feature": row["feature"],
            "importance": float(row["importance"]),
        }
        for _, row in top_features.iterrows()
    ]


# =============================================================================
# SINGLE HORIZON EXPERIMENT
# =============================================================================

def run_horizon(df, horizon_name, horizon_hours):
    """
    Run a fair RF vs XGBoost comparison for one horizon.
    """

    print()
    print("#" * 80)
    print(f"FORECAST HORIZON: {horizon_name.upper()}")
    print("#" * 80)

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------

    data, target_column = prepare_data(
        df,
        horizon_hours
    )

    # -------------------------------------------------------------------------
    # Chronological split
    # -------------------------------------------------------------------------

    train, validation, test = chronological_split(
        data,
        target_column
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train[target_column]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[target_column]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[target_column]

    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    rf_validation_predictions = rf_model.predict(
        X_validation
    )

    rf_test_predictions = rf_model.predict(
        X_test
    )

    rf_validation_metrics = calculate_metrics(
        y_validation,
        rf_validation_predictions
    )

    rf_test_metrics = calculate_metrics(
        y_test,
        rf_test_predictions
    )

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    xgb_validation_predictions = xgb_model.predict(
        X_validation
    )

    xgb_test_predictions = xgb_model.predict(
        X_test
    )

    xgb_validation_metrics = calculate_metrics(
        y_validation,
        xgb_validation_predictions
    )

    xgb_test_metrics = calculate_metrics(
        y_test,
        xgb_test_predictions
    )

    # -------------------------------------------------------------------------
    # Print validation results
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    print(
        f"Random Forest  MAE:  "
        f"{rf_validation_metrics['mae']:.4f}"
    )

    print(
        f"Random Forest  RMSE: "
        f"{rf_validation_metrics['rmse']:.4f}"
    )

    print(
        f"Random Forest  R²:   "
        f"{rf_validation_metrics['r2']:.4f}"
    )

    print()

    print(
        f"XGBoost        MAE:  "
        f"{xgb_validation_metrics['mae']:.4f}"
    )

    print(
        f"XGBoost        RMSE: "
        f"{xgb_validation_metrics['rmse']:.4f}"
    )

    print(
        f"XGBoost        R²:   "
        f"{xgb_validation_metrics['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Print test results
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print(
        f"Random Forest  MAE:  "
        f"{rf_test_metrics['mae']:.4f}"
    )

    print(
        f"Random Forest  RMSE: "
        f"{rf_test_metrics['rmse']:.4f}"
    )

    print(
        f"Random Forest  R²:   "
        f"{rf_test_metrics['r2']:.4f}"
    )

    print()

    print(
        f"XGBoost        MAE:  "
        f"{xgb_test_metrics['mae']:.4f}"
    )

    print(
        f"XGBoost        RMSE: "
        f"{xgb_test_metrics['rmse']:.4f}"
    )

    print(
        f"XGBoost        R²:   "
        f"{xgb_test_metrics['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Determine winner
    # -------------------------------------------------------------------------

    rf_mae = rf_test_metrics["mae"]
    xgb_mae = xgb_test_metrics["mae"]

    if rf_mae < xgb_mae:
        winner = "Random Forest"
        best_mae = rf_mae
        other_mae = xgb_mae
    elif xgb_mae < rf_mae:
        winner = "XGBoost"
        best_mae = xgb_mae
        other_mae = rf_mae
    else:
        winner = "Tie"
        best_mae = rf_mae
        other_mae = xgb_mae

    if other_mae != 0:
        improvement = (
            (other_mae - best_mae)
            / other_mae
        ) * 100
    else:
        improvement = 0.0

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        f"Random Forest MAE: {rf_mae:.4f}"
    )

    print(
        f"XGBoost MAE:       {xgb_mae:.4f}"
    )

    print(
        f"WINNER: {winner}"
    )

    if winner != "Tie":
        print(
            f"Winner improvement: "
            f"{improvement:.2f}%"
        )

    # -------------------------------------------------------------------------
    # Save models
    # -------------------------------------------------------------------------

    rf_filename = (
        f"sensor_218_random_forest_{horizon_name}.pkl"
    )

    xgb_filename = (
        f"sensor_218_xgboost_comparison_{horizon_name}.pkl"
    )

    rf_path = MODEL_DIR / rf_filename
    xgb_path = MODEL_DIR / xgb_filename

    joblib.dump(
        rf_model,
        rf_path
    )

    joblib.dump(
        xgb_model,
        xgb_path
    )

    print()
    print("MODELS SAVED")

    print(
        f"Random Forest: {rf_path}"
    )

    print(
        f"XGBoost:       {xgb_path}"
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    print()

    rf_importance = get_feature_importance(
        rf_model,
        "Random Forest"
    )

    print()

    xgb_importance = get_feature_importance(
        xgb_model,
        "XGBoost"
    )

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------

    return {
        "horizon": horizon_name,
        "horizon_hours": horizon_hours,
        "rows": {
            "total_modelling_rows": int(len(data)),
            "training": int(len(train)),
            "validation": int(len(validation)),
            "testing": int(len(test)),
        },
        "test_period": {
            "start": str(test["timestamp"].iloc[0])
            if len(test) > 0 else None,
            "end": str(test["timestamp"].iloc[-1])
            if len(test) > 0 else None,
        },
        "random_forest": {
            "validation": rf_validation_metrics,
            "test": rf_test_metrics,
            "top_features": rf_importance,
            "model_file": str(rf_path),
        },
        "xgboost": {
            "validation": xgb_validation_metrics,
            "test": xgb_test_metrics,
            "top_features": xgb_importance,
            "model_file": str(xgb_path),
        },
        "winner": winner,
        "winner_mae_improvement_percent": float(improvement),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("FAIR RANDOM FOREST VS XGBOOST MODEL COMPARISON")
    print("=" * 80)

    print()
    print(
        "Both models will use the same data, features and "
        "chronological test splits."
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING ENGINEERED SENSOR 218 DATA")
    print("=" * 80)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns:     {len(df.columns)}"
    )

    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------

    required_columns = [
        "timestamp",
        "target_pm25",
    ] + FEATURE_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # -------------------------------------------------------------------------
    # Run all horizons
    # -------------------------------------------------------------------------

    results = {}

    for horizon_name, horizon_hours in HORIZONS.items():

        result = run_horizon(
            df,
            horizon_name,
            horizon_hours
        )

        results[horizon_name] = result

    # -------------------------------------------------------------------------
    # Final comparison
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL RANDOM FOREST VS XGBOOST COMPARISON")
    print("=" * 80)

    rf_wins = 0
    xgb_wins = 0

    for horizon_name, result in results.items():

        rf_mae = result["random_forest"]["test"]["mae"]
        xgb_mae = result["xgboost"]["test"]["mae"]

        print()
        print(
            f"{horizon_name.upper()} FORECAST"
        )

        print(
            f"Random Forest MAE: {rf_mae:.4f}"
        )

        print(
            f"XGBoost MAE:       {xgb_mae:.4f}"
        )

        print(
            f"Random Forest RMSE: "
            f"{result['random_forest']['test']['rmse']:.4f}"
        )

        print(
            f"XGBoost RMSE:       "
            f"{result['xgboost']['test']['rmse']:.4f}"
        )

        print(
            f"Random Forest R²:   "
            f"{result['random_forest']['test']['r2']:.4f}"
        )

        print(
            f"XGBoost R²:         "
            f"{result['xgboost']['test']['r2']:.4f}"
        )

        print(
            f"WINNER: {result['winner']}"
        )

        if result["winner"] == "Random Forest":
            rf_wins += 1

        elif result["winner"] == "XGBoost":
            xgb_wins += 1

    # -------------------------------------------------------------------------
    # Overall winner
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 80)

    print(
        f"Random Forest wins: {rf_wins}"
    )

    print(
        f"XGBoost wins:       {xgb_wins}"
    )

    if rf_wins > xgb_wins:
        overall_winner = "Random Forest"

    elif xgb_wins > rf_wins:
        overall_winner = "XGBoost"

    else:
        overall_winner = "Tie"

    print()
    print(
        f"OVERALL WINNER: {overall_winner}"
    )

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------

    report = {
        "experiment": "Fair Random Forest vs XGBoost Model Comparison",
        "dataset": str(DATA_FILE),
        "features": FEATURE_COLUMNS,
        "split": {
            "training": 0.70,
            "validation": 0.15,
            "testing": 0.15,
            "method": "chronological",
        },
        "horizons": results,
        "overall": {
            "random_forest_wins": rf_wins,
            "xgboost_wins": xgb_wins,
            "winner": overall_winner,
        },
    }

    report_path = (
        REPORT_DIR /
        "random_forest_vs_xgboost_comparison.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Report saved: {report_path}"
    )


if __name__ == "__main__":
    main()