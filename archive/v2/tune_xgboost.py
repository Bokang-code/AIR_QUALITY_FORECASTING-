"""
STEP 5: XGBOOST HYPERPARAMETER TUNING

Purpose:
    Test whether XGBoost model limitations are contributing to poor
    forecasting performance.

Important:
    - Uses the same engineered dataset as the previous experiments.
    - Uses chronological train/validation/test splits.
    - Uses a horizon-specific purge gap.
    - Hyperparameters are selected using VALIDATION performance only.
    - The TEST set is used only once for the selected configuration.
    - Existing train.py and train_purged.py are NOT modified.

Forecast horizons:
    - 48 hours
    - 72 hours
    - 7 days
    - 14 days
    - 30 days

Outputs:
    outputs/reports/sensor_218_step5_tuning_results.csv
    outputs/reports/sensor_218_step5_best_models.json
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

PROJECT_ROOT = Path(__file__).resolve().parent

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
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# TUNING GRID
# =============================================================================

# Small, controlled grid.
#
# We are NOT doing an enormous grid search because:
# 1. The dataset is relatively small.
# 2. We want a defensible thesis experiment.
# 3. We want to determine whether tuning materially improves performance.
#
# The baseline model was:
#
# n_estimators = 500
# learning_rate = 0.05
# max_depth = 6
# min_child_weight = 3
# subsample = 0.8
# colsample_bytree = 0.8
# reg_alpha = 0
# reg_lambda = 1
#
# The configurations below vary the most important parameters.

PARAMETER_GRID = [
    {
        "name": "baseline",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "shallower",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "medium_depth",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "deeper",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "stronger_regularization",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "name": "more_trees",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "lower_learning_rate",
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "subsample_regularized",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
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

    df = pd.read_csv(
        DATA_PATH
    )

    if "timestamp" not in df.columns:

        raise ValueError(
            "Dataset must contain 'timestamp'."
        )

    if "pm25" not in df.columns:

        raise ValueError(
            "Dataset must contain 'pm25'."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = (
        df
        .sort_values("timestamp")
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
# TARGET CREATION
# =============================================================================

def create_target(
    df,
    horizon_hours,
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
    horizon_hours,
):

    print()
    print("-" * 80)

    print(
        f"PREPARING {horizon_hours}H DATA"
    )

    print("-" * 80)

    df, target_column = create_target(
        df,
        horizon_hours,
    )

    # Check features.

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features:\n"
            + "\n".join(
                missing_features
            )
        )

    # Remove unavailable targets.

    before = len(df)

    df = df.dropna(
        subset=[target_column]
    ).copy()

    removed_target = (
        before - len(df)
    )

    print(
        f"Rows removed due to missing target: "
        f"{removed_target}"
    )

    # Remove rows with missing feature values.

    before = len(df)

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    removed_features = (
        before - len(df)
    )

    print(
        f"Rows removed due to missing features: "
        f"{removed_features}"
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
# PURGED SPLIT
# =============================================================================

def purged_split(
    X,
    y,
    timestamps,
    purge_hours,
):

    print()
    print("-" * 80)

    print(
        f"PURGED CHRONOLOGICAL SPLIT "
        f"({purge_hours}H PURGE)"
    )

    print("-" * 80)

    n = len(X)

    # -------------------------------------------------------------------------
    # The split is based on the original chronological ordering.
    #
    # First 70% = training region
    # Next 15%  = validation region
    # Final 15% = test region
    #
    # A purge gap equal to the forecasting horizon is inserted between
    # train/validation and validation/test.
    # -------------------------------------------------------------------------

    train_boundary = int(
        n * 0.70
    )

    test_boundary = int(
        n * 0.85
    )

    train_end = (
        train_boundary
        - purge_hours // 2
    )

    validation_start = (
        train_boundary
        + purge_hours // 2
    )

    validation_end = (
        test_boundary
        - purge_hours // 2
    )

    test_start = (
        test_boundary
        + purge_hours // 2
    )

    # Safety checks.

    if train_end <= 0:
        raise ValueError(
            "Training set became empty."
        )

    if validation_start >= validation_end:
        raise ValueError(
            "Validation set became empty. "
            f"Purging {purge_hours} rows is too large "
            "for this dataset."
        )

    if test_start >= n:
        raise ValueError(
            "Test set became empty."
        )

    X_train = X.iloc[
        :train_end
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    X_validation = X.iloc[
        validation_start:validation_end
    ].copy()

    y_validation = y.iloc[
        validation_start:validation_end
    ].copy()

    X_test = X.iloc[
        test_start:
    ].copy()

    y_test = y.iloc[
        test_start:
    ].copy()

    timestamps_test = timestamps.iloc[
        test_start:
    ].copy()

    print(
        f"Original rows:        {n:,}"
    )

    print()

    print(
        f"Training rows:        {len(X_train):,}"
    )

    print(
        f"Train purge gap:      "
        f"{validation_start - train_end:,} rows"
    )

    print(
        f"Validation rows:      "
        f"{len(X_validation):,}"
    )

    print(
        f"Validation purge gap: "
        f"{test_start - validation_end:,} rows"
    )

    print(
        f"Testing rows:         "
        f"{len(X_test):,}"
    )

    print()

    print("Training:")

    print(
        f"  {timestamps.iloc[0]}"
        f" → "
        f"{timestamps.iloc[train_end - 1]}"
    )

    print()

    print("Validation:")

    print(
        f"  {timestamps.iloc[validation_start]}"
        f" → "
        f"{timestamps.iloc[validation_end - 1]}"
    )

    print()

    print("Testing:")

    print(
        f"  {timestamps.iloc[test_start]}"
        f" → "
        f"{timestamps.iloc[-1]}"
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
# MODEL CREATION
# =============================================================================

def create_model(params):

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=params[
            "n_estimators"
        ],

        learning_rate=params[
            "learning_rate"
        ],

        max_depth=params[
            "max_depth"
        ],

        min_child_weight=params[
            "min_child_weight"
        ],

        subsample=params[
            "subsample"
        ],

        colsample_bytree=params[
            "colsample_bytree"
        ],

        reg_alpha=params[
            "reg_alpha"
        ],

        reg_lambda=params[
            "reg_lambda"
        ],

        random_state=42,

        n_jobs=-1,

        tree_method="hist",
    )

    return model


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    predictions,
):

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
# TUNE ONE HORIZON
# =============================================================================

def tune_horizon(
    df,
    horizon_name,
    horizon_hours,
):

    print()
    print("#" * 80)

    print(
        f"STEP 5 TUNING: "
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
        horizon_hours,
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
    ) = purged_split(
        X,
        y,
        timestamps,
        horizon_hours,
    )

    # -------------------------------------------------------------------------
    # Run tuning configurations
    # -------------------------------------------------------------------------

    tuning_results = []

    print()
    print("=" * 80)

    print(
        f"TESTING {len(PARAMETER_GRID)} "
        f"XGBOOST CONFIGURATIONS"
    )

    print("=" * 80)

    for index, params in enumerate(
        PARAMETER_GRID,
        start=1,
    ):

        name = params["name"]

        print()
        print(
            f"[{index}/{len(PARAMETER_GRID)}] "
            f"Testing: {name}"
        )

        model = create_model(
            params
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

        validation_predictions = (
            model.predict(
                X_validation
            )
        )

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        print(
            f"Validation MAE:  "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"Validation RMSE: "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"Validation R²:   "
            f"{metrics['r2']:.4f}"
        )

        tuning_results.append(
            {
                "horizon": horizon_name,
                "horizon_hours": horizon_hours,
                "configuration": name,
                "validation_mae": metrics[
                    "mae"
                ],
                "validation_rmse": metrics[
                    "rmse"
                ],
                "validation_r2": metrics[
                    "r2"
                ],
                "parameters": params,
            }
        )

    # -------------------------------------------------------------------------
    # Select best configuration using validation MAE.
    #
    # MAE is the primary selection metric because it is easy to interpret
    # in the same units as PM2.5.
    # -------------------------------------------------------------------------

    tuning_results_sorted = sorted(
        tuning_results,
        key=lambda x: x[
            "validation_mae"
        ],
    )

    best_result = (
        tuning_results_sorted[0]
    )

    best_params = (
        best_result["parameters"]
    )

    print()
    print("=" * 80)

    print(
        "BEST CONFIGURATION "
        "BASED ON VALIDATION MAE"
    )

    print("=" * 80)

    print(
        f"Configuration: "
        f"{best_result['configuration']}"
    )

    print(
        f"Validation MAE: "
        f"{best_result['validation_mae']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{best_result['validation_rmse']:.4f}"
    )

    print(
        f"Validation R²: "
        f"{best_result['validation_r2']:.4f}"
    )

    print()

    for key, value in best_params.items():

        if key != "name":

            print(
                f"{key}: {value}"
            )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # The test set is now evaluated using ONLY the selected configuration.
    #
    # We do not select the model based on test performance.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "FINAL TEST EVALUATION "
        "OF SELECTED CONFIGURATION"
    )

    print("=" * 80)

    best_model = create_model(
        best_params
    )

    best_model.fit(
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

    test_predictions = (
        best_model.predict(
            X_test
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print(
        f"Test MAE:  "
        f"{test_metrics['mae']:.4f}"
    )

    print(
        f"Test RMSE: "
        f"{test_metrics['rmse']:.4f}"
    )

    print(
        f"Test R²:   "
        f"{test_metrics['r2']:.4f}"
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

    prediction_path = (
        REPORT_DIR
        / (
            f"sensor_218_xgboost_"
            f"{horizon_name}_tuned_predictions.csv"
        )
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / (
            f"sensor_218_xgboost_"
            f"{horizon_name}_tuned.pkl"
        )
    )

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURE_COLUMNS,
            "target": target_column,
            "horizon_hours": horizon_hours,
            "horizon_name": horizon_name,
            "configuration": best_result[
                "configuration"
            ],
            "parameters": best_params,
        },
        model_path,
    )

    print()
    print(
        f"Model saved: "
        f"{model_path}"
    )

    print(
        f"Predictions saved: "
        f"{prediction_path}"
    )

    return {
        "horizon": horizon_name,
        "horizon_hours": horizon_hours,
        "best_configuration": best_result[
            "configuration"
        ],
        "best_parameters": best_params,

        "validation_mae": best_result[
            "validation_mae"
        ],

        "validation_rmse": best_result[
            "validation_rmse"
        ],

        "validation_r2": best_result[
            "validation_r2"
        ],

        "test_mae": test_metrics[
            "mae"
        ],

        "test_rmse": test_metrics[
            "rmse"
        ],

        "test_r2": test_metrics[
            "r2"
        ],

        "model_path": str(
            model_path
        ),

        "prediction_path": str(
            prediction_path
        ),

        "all_tuning_results": (
            tuning_results_sorted
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)

    print(
        "STEP 5: XGBOOST HYPERPARAMETER TUNING"
    )

    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "• Existing train.py is NOT modified."
    )

    print(
        "• Existing train_purged.py is NOT modified."
    )

    print(
        "• Horizon-specific purging is used."
    )

    print(
        "• Validation determines the best configuration."
    )

    print(
        "• Test data is evaluated only after selection."
    )

    print()

    df = load_data()

    all_results = []

    for (
        horizon_name,
        horizon_hours,
    ) in HORIZONS.items():

        result = tune_horizon(
            df,
            horizon_name,
            horizon_hours,
        )

        all_results.append(
            result
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_header(
        "STEP 5: FINAL TUNING RESULTS"
    )

    print(
        f"{'HORIZON':<10}"
        f"{'CONFIGURATION':<25}"
        f"{'VAL MAE':<12}"
        f"{'TEST MAE':<12}"
        f"{'TEST RMSE':<12}"
        f"{'TEST R²':<12}"
    )

    print("-" * 83)

    for result in all_results:

        print(
            f"{result['horizon']:<10}"
            f"{result['best_configuration']:<25}"
            f"{result['validation_mae']:<12.4f}"
            f"{result['test_mae']:<12.4f}"
            f"{result['test_rmse']:<12.4f}"
            f"{result['test_r2']:<12.4f}"
        )

    # =========================================================================
    # SAVE CSV SUMMARY
    # =========================================================================

    csv_rows = []

    for result in all_results:

        csv_rows.append(
            {
                "horizon": result[
                    "horizon"
                ],

                "horizon_hours": result[
                    "horizon_hours"
                ],

                "best_configuration": result[
                    "best_configuration"
                ],

                "validation_mae": result[
                    "validation_mae"
                ],

                "validation_rmse": result[
                    "validation_rmse"
                ],

                "validation_r2": result[
                    "validation_r2"
                ],

                "test_mae": result[
                    "test_mae"
                ],

                "test_rmse": result[
                    "test_rmse"
                ],

                "test_r2": result[
                    "test_r2"
                ],
            }
        )

    summary_df = pd.DataFrame(
        csv_rows
    )

    summary_path = (
        REPORT_DIR
        / "sensor_218_step5_tuning_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # =========================================================================
    # SAVE JSON
    # =========================================================================

    json_path = (
        REPORT_DIR
        / "sensor_218_step5_best_models.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
        )

    print()
    print(
        f"Tuning summary saved: "
        f"{summary_path}"
    )

    print(
        f"Detailed results saved: "
        f"{json_path}"
    )

    print()
    print("=" * 80)

    print(
        "STEP 5 XGBOOST TUNING COMPLETE"
    )

    print("=" * 80)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()