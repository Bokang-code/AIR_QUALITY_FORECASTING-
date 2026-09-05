import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = "data/processed/v3/engineered_sensor_218_v3.csv"

MODEL_DIR = "models/v3"
REPORT_DIR = "reports/v3"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


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
# SPLIT SETTINGS
# =============================================================================

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# =============================================================================
# TARGET
# =============================================================================

def create_target(df, horizon):
    df = df.copy()

    df["target_pm25"] = df["pm25"].shift(-horizon)

    return df


# =============================================================================
# PREPARE DATA
# =============================================================================

def prepare_data(df, horizon):

    df = create_target(df, horizon)

    # Only remove rows where the future target is unavailable.
    # XGBoost can handle missing feature values natively.

    df = df.dropna(
        subset=["target_pm25"]
    ).copy()

    exclude_columns = [
        "timestamp",
        "pm25",
        "pm25_missing",
        "target_pm25",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in exclude_columns
    ]

    X = df[feature_columns].copy()
    y = df["target_pm25"].copy()
    timestamps = df["timestamp"].copy()

    return (
        X,
        y,
        timestamps,
        feature_columns,
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

    n = len(X)

    # Initial chronological boundaries

    train_end = int(
        n * TRAIN_SIZE
    )

    validation_end = int(
        n * (TRAIN_SIZE + VALIDATION_SIZE)
    )

    # -------------------------------------------------------------------------
    # PURGE
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
    # TRAIN
    # -------------------------------------------------------------------------

    X_train = X.iloc[
        :train_end_purged
    ].copy()

    y_train = y.iloc[
        :train_end_purged
    ].copy()

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    X_validation = X.iloc[
        validation_start:validation_end_purged
    ].copy()

    y_validation = y.iloc[
        validation_start:validation_end_purged
    ].copy()

    # -------------------------------------------------------------------------
    # TEST
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
    # DISPLAY
    # -------------------------------------------------------------------------

    train_purge = (
        validation_start
        - train_end_purged
    )

    validation_purge = (
        test_start
        - validation_end_purged
    )

    print()
    print("-" * 70)
    print("PURGED CHRONOLOGICAL SPLIT")
    print("-" * 70)

    print(
        f"Original target-valid rows: {n:,}"
    )

    print(
        f"Training rows:              {len(X_train):,}"
    )

    print(
        f"Train purge:                {train_purge:,}"
    )

    print(
        f"Validation rows:            {len(X_validation):,}"
    )

    print(
        f"Validation purge:           {validation_purge:,}"
    )

    print(
        f"Test rows:                  {len(X_test):,}"
    )

    print()

    print(
        "[PURGED] "
        f"{horizon_hours} hours removed "
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

def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
):

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

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model,
    X,
    y,
):

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

    return (
        mae,
        rmse,
        r2,
        predictions,
    )


# =============================================================================
# TRAIN ONE HORIZON
# =============================================================================

def train_horizon(
    df,
    horizon_name,
    horizon,
):

    print()
    print("=" * 70)
    print(
        f"FORECAST HORIZON: "
        f"{horizon_name} ({horizon} hours)"
    )
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Prepare
    # -------------------------------------------------------------------------

    (
        X,
        y,
        timestamps,
        feature_columns,
    ) = prepare_data(
        df,
        horizon,
    )

    print()
    print(
        f"Target-valid rows: {len(X):,}"
    )

    print(
        f"Features:          {len(feature_columns)}"
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
        horizon,
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print()
    print("Training XGBoost...")

    model = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    print(
        "XGBoost training complete."
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    (
        val_mae,
        val_rmse,
        val_r2,
        val_predictions,
    ) = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    print()
    print("Validation:")

    print(
        f"MAE:  {val_mae:.4f}"
    )

    print(
        f"RMSE: {val_rmse:.4f}"
    )

    print(
        f"R²:   {val_r2:.4f}"
    )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    (
        test_mae,
        test_rmse,
        test_r2,
        test_predictions,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print()
    print("Test:")

    print(
        f"MAE:  {test_mae:.4f}"
    )

    print(
        f"RMSE: {test_rmse:.4f}"
    )

    print(
        f"R²:   {test_r2:.4f}"
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    importance = (
        pd.Series(
            model.feature_importances_,
            index=feature_columns,
        )
        .sort_values(
            ascending=False
        )
    )

    print()
    print("Top 15 features:")

    for feature, value in (
        importance.head(15).items()
    ):

        print(
            f"{feature:<35} "
            f"{value:.6f}"
        )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_file = (
        f"{MODEL_DIR}/"
        f"xgboost_v3_{horizon_name}_purged.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "horizon_hours": horizon,
            "horizon_name": horizon_name,
            "split_method": "purged_chronological",
            "purge_hours": horizon,
        },
        model_file,
    )

    print()
    print(
        f"Model saved: {model_file}"
    )

    # -------------------------------------------------------------------------
    # Save predictions
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

    prediction_file = (
        f"{REPORT_DIR}/"
        f"v3_{horizon_name}_purged_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_file,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------

    return {
        "horizon": horizon_name,
        "horizon_hours": horizon,
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
        "features": len(feature_columns),
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_r2": val_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "purge_hours": horizon,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("V3 PM2.5 FORECASTING — PURGED EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Loading dataset:\n{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print()
    print(
        f"Dataset shape: {df.shape}"
    )

    results = []

    for (
        horizon_name,
        horizon,
    ) in HORIZONS.items():

        result = train_horizon(
            df,
            horizon_name,
            horizon,
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_file = (
        f"{REPORT_DIR}/"
        "v3_purged_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("V3 PURGED RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "horizon",
                "train_rows",
                "validation_rows",
                "test_rows",
                "test_mae",
                "test_rmse",
                "test_r2",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved: {results_file}"
    )

    print()
    print("=" * 70)
    print("V3 PURGED TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()