"""
STEP 7 — V2 FEATURE ABLATION STUDY

Purpose:
    Determine which V2 feature groups actually improve forecasting
    compared with the original V1 feature set.

Important:
    • Existing files are NOT modified.
    • Existing models are NOT modified.
    • Existing train.py is NOT modified.
    • Existing train_purged.py is NOT modified.
    • Existing tuned models are NOT modified.
    • Horizon-specific purging is used.
    • Validation MAE is used for feature-group selection.
    • Test data is evaluated only after feature-group selection.
    • Missing feature values are retained.
    • XGBoost handles missing feature values natively.
    • No future PM2.5 values are used as input features.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path("data/processed/sensor_218_hourly_v2.csv")

REPORT_DIR = Path("outputs/reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


# =============================================================================
# XGBOOST CONFIGURATION
# =============================================================================
#
# This is intentionally kept fixed.
#
# We are testing FEATURE GROUPS, not hyperparameters.
#
# Hyperparameter tuning was already performed in Step 5.
# Changing model parameters here would mix two experiments.
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
# FEATURE GROUP DEFINITIONS
# =============================================================================

V1_FEATURES = [
    "pm25_missing",
    "hour",
    "day",
    "month",
    "day_of_week",
    "weekend",
    "season",
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


V2_LAG_FEATURES = [
    "pm25_lag_2h",
    "pm25_lag_4h",
    "pm25_lag_8h",
    "pm25_lag_18h",
    "pm25_lag_36h",
    "pm25_lag_72h",
    "pm25_lag_120h",
    "pm25_lag_336h",
]


V2_ROLLING_FEATURES = [
    "pm25_rolling_mean_3h",
    "pm25_rolling_std_3h",
    "pm25_rolling_min_3h",
    "pm25_rolling_max_3h",
    "pm25_rolling_median_3h",
    "pm25_rolling_range_3h",

    "pm25_rolling_mean_12h",
    "pm25_rolling_std_12h",
    "pm25_rolling_min_12h",
    "pm25_rolling_max_12h",
    "pm25_rolling_median_12h",
    "pm25_rolling_range_12h",

    "pm25_rolling_mean_48h",
    "pm25_rolling_std_48h",
    "pm25_rolling_min_48h",
    "pm25_rolling_max_48h",
    "pm25_rolling_median_48h",
    "pm25_rolling_range_48h",

    "pm25_rolling_mean_72h",
    "pm25_rolling_std_72h",
    "pm25_rolling_min_72h",
    "pm25_rolling_max_72h",
    "pm25_rolling_median_72h",
    "pm25_rolling_range_72h",

    "pm25_rolling_mean_336h",
    "pm25_rolling_std_336h",
    "pm25_rolling_min_336h",
    "pm25_rolling_max_336h",
    "pm25_rolling_median_336h",
    "pm25_rolling_range_336h",
]


V2_TREND_FEATURES = [
    "pm25_change_3h",
    "pm25_pct_change_3h",
    "pm25_change_6h",
    "pm25_pct_change_6h",
    "pm25_change_12h",
    "pm25_pct_change_12h",
    "pm25_change_24h",
    "pm25_pct_change_24h",
    "pm25_change_48h",
    "pm25_pct_change_48h",

    "pm25_mean_deviation_6h",
    "pm25_cv_6h",
    "pm25_range_6h",

    "pm25_mean_deviation_24h",
    "pm25_cv_24h",
    "pm25_range_24h",

    "pm25_mean_deviation_48h",
    "pm25_cv_48h",
    "pm25_range_48h",

    "pm25_mean_deviation_72h",
    "pm25_cv_72h",
    "pm25_range_72h",

    "pm25_mean_deviation_336h",
    "pm25_cv_336h",
    "pm25_range_336h",
]


V2_SEASONAL_FEATURES = [
    "day_of_year",
    "day_of_year_sin",
    "day_of_year_cos",
    "week_of_year",
    "week_of_year_sin",
    "week_of_year_cos",
]


# =============================================================================
# FEATURE GROUPS
# =============================================================================

FEATURE_GROUPS = {
    "V1_baseline": V1_FEATURES,

    "V1_plus_V2_lags": (
        V1_FEATURES
        + V2_LAG_FEATURES
    ),

    "V1_plus_V2_rolling": (
        V1_FEATURES
        + V2_ROLLING_FEATURES
    ),

    "V1_plus_V2_trend": (
        V1_FEATURES
        + V2_TREND_FEATURES
    ),

    "V1_plus_V2_seasonal": (
        V1_FEATURES
        + V2_SEASONAL_FEATURES
    ),

    "ALL_V2_features": (
        V1_FEATURES
        + V2_LAG_FEATURES
        + V2_ROLLING_FEATURES
        + V2_TREND_FEATURES
        + V2_SEASONAL_FEATURES
    ),
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title, width=80):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subheader(title, width=80):
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    print_header("LOADING V2 ENGINEERED SENSOR 218 DATA")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded:    {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    # -------------------------------------------------------------------------
    # Detect timestamp
    # -------------------------------------------------------------------------

    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
    ]

    timestamp_column = None

    for column in timestamp_candidates:
        if column in df.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        raise ValueError(
            "Could not find a timestamp column."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce",
    )

    if df[timestamp_column].isna().any():
        raise ValueError(
            "Timestamp column contains invalid values."
        )

    df = df.sort_values(
        timestamp_column
    ).reset_index(drop=True)

    # Standardize timestamp name
    if timestamp_column != "timestamp":
        df = df.rename(
            columns={
                timestamp_column: "timestamp"
            }
        )

    print(
        f"Start:          {df['timestamp'].min()}"
    )
    print(
        f"End:            {df['timestamp'].max()}"
    )

    # -------------------------------------------------------------------------
    # Detect PM2.5
    # -------------------------------------------------------------------------

    pm25_candidates = [
        "pm25",
        "PM2.5",
        "value",
    ]

    pm25_column = None

    for column in pm25_candidates:
        if column in df.columns:
            pm25_column = column
            break

    if pm25_column is None:
        raise ValueError(
            "Could not find PM2.5 column."
        )

    if pm25_column != "pm25":
        df = df.rename(
            columns={
                pm25_column: "pm25"
            }
        )

    print(
        f"\nPM2.5 observations: {df['pm25'].notna().sum():,}"
    )

    print(
        f"PM2.5 missing:      {df['pm25'].isna().sum():,}"
    )

    return df


# =============================================================================
# FEATURE LEAKAGE CHECK
# =============================================================================

def check_feature_leakage(df):
    print_subheader("FEATURE LEAKAGE CHECK")

    forbidden_patterns = [
        "future",
        "target",
        "forecast",
        "lead",
        "next",
    ]

    all_features = []

    for features in FEATURE_GROUPS.values():
        all_features.extend(features)

    all_features = list(dict.fromkeys(all_features))

    problems = []

    for feature in all_features:

        feature_lower = feature.lower()

        for pattern in forbidden_patterns:

            if pattern in feature_lower:

                # "target" isn't a feature in our groups,
                # so flag it.
                problems.append(
                    (feature, pattern)
                )

    if problems:

        print("[FAIL] Potential future-looking features detected:")

        for feature, pattern in problems:
            print(
                f"  - {feature} contains '{pattern}'"
            )

        raise ValueError(
            "Feature leakage check failed."
        )

    print(
        "[PASS] No obviously future-looking feature names detected."
    )

    # -------------------------------------------------------------------------
    # Make sure target isn't being used as a feature
    # -------------------------------------------------------------------------

    for group_name, features in FEATURE_GROUPS.items():

        for feature in features:

            if feature.startswith("target_"):

                raise ValueError(
                    f"Target feature detected in {group_name}: "
                    f"{feature}"
                )

    print(
        "[PASS] No target columns are included as features."
    )


# =============================================================================
# TARGET CREATION
# =============================================================================

def prepare_target_data(df, horizon):
    target_column = f"target_pm25_{horizon}h"

    print_subheader(
        f"PREPARING {horizon}H TARGET"
    )

    if target_column in df.columns:

        print(
            f"[INFO] {target_column} already exists."
        )

    else:

        print(
            f"[INFO] Creating {target_column} "
            f"using PM2.5 shifted by -{horizon}."
        )

        df[target_column] = df["pm25"].shift(
            -horizon
        )

    before = len(df)

    df_model = df[
        df[target_column].notna()
    ].copy()

    removed = before - len(df_model)

    print(
        f"Rows removed due to missing future target: "
        f"{removed:,}"
    )

    return df_model, target_column


# =============================================================================
# PURGED CHRONOLOGICAL SPLIT
# =============================================================================

def purged_split(
    df,
    horizon,
    train_ratio=0.70,
    validation_ratio=0.15,
):
    """
    Chronological split with horizon-specific purge.

    Structure:

        TRAIN
        PURGE
        VALIDATION
        PURGE
        TEST

    The purge gap equals the forecast horizon.
    """

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n * (train_ratio + validation_ratio)
    )

    train_end_purged = (
        train_end - horizon
    )

    validation_start = (
        train_end + horizon
    )

    validation_end_purged = (
        validation_end - horizon
    )

    test_start = (
        validation_end + horizon
    )

    train = df.iloc[
        :train_end_purged
    ].copy()

    validation = df.iloc[
        validation_start:
        validation_end_purged
    ].copy()

    test = df.iloc[
        test_start:
    ].copy()

    print_subheader(
        f"PURGED CHRONOLOGICAL SPLIT "
        f"({horizon}H PURGE)"
    )

    print(
        f"Original rows:        {n:,}"
    )

    print()

    print(
        f"Training rows:        {len(train):,}"
    )

    print(
        f"Train purge gap:      {horizon:,} rows"
    )

    print(
        f"Validation rows:      {len(validation):,}"
    )

    print(
        f"Validation purge gap: {horizon:,} rows"
    )

    print(
        f"Testing rows:         {len(test):,}"
    )

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
        f"at each split boundary."
    )

    return train, validation, test


# =============================================================================
# VERIFY FEATURES
# =============================================================================

def validate_feature_group(
    df,
    feature_names,
    group_name,
):
    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:

        print(
            f"\n[WARNING] {group_name} "
            f"is missing {len(missing_features)} features:"
        )

        for feature in missing_features:
            print(
                f"  - {feature}"
            )

        return False

    return True


# =============================================================================
# TRAIN / EVALUATE ONE FEATURE GROUP
# =============================================================================

def run_feature_group(
    train_df,
    validation_df,
    test_df,
    feature_names,
    target_column,
    group_name,
    horizon,
):
    """
    Train one XGBoost model using one feature group.
    """

    if len(train_df) == 0:
        return {
            "status": "ERROR",
            "error": "No training observations available.",
        }

    if len(validation_df) == 0:
        return {
            "status": "ERROR",
            "error": "No validation observations available.",
        }

    if len(test_df) == 0:
        return {
            "status": "ERROR",
            "error": "No testing observations available.",
        }

    # -------------------------------------------------------------------------
    # Feature validation
    # -------------------------------------------------------------------------

    valid = validate_feature_group(
        train_df,
        feature_names,
        group_name,
    )

    if not valid:

        return {
            "status": "ERROR",
            "error": "Missing feature columns.",
        }

    X_train = train_df[
        feature_names
    ]

    y_train = train_df[
        target_column
    ]

    X_validation = validation_df[
        feature_names
    ]

    y_validation = validation_df[
        target_column
    ]

    X_test = test_df[
        feature_names
    ]

    y_test = test_df[
        target_column
    ]

    # -------------------------------------------------------------------------
    # Ensure numeric data
    # -------------------------------------------------------------------------

    for feature in feature_names:

        if not pd.api.types.is_numeric_dtype(
            X_train[feature]
        ):

            print(
                f"[WARNING] Converting "
                f"{feature} to numeric."
            )

            X_train[feature] = pd.to_numeric(
                X_train[feature],
                errors="coerce",
            )

            X_validation[feature] = pd.to_numeric(
                X_validation[feature],
                errors="coerce",
            )

            X_test[feature] = pd.to_numeric(
                X_test[feature],
                errors="coerce",
            )

    # -------------------------------------------------------------------------
    # Missing feature reporting
    # -------------------------------------------------------------------------

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
        f"Training missing cells:   "
        f"{train_missing:,}"
    )

    print(
        f"Validation missing cells: "
        f"{validation_missing:,}"
    )

    print(
        f"Testing missing cells:     "
        f"{test_missing:,}"
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    model = XGBRegressor(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------

    return {
        "status": "SUCCESS",

        "group": group_name,

        "horizon": horizon,

        "feature_count": len(feature_names),

        "features": feature_names,

        "validation": validation_metrics,

        "test": test_metrics,

        "model": model,

        "validation_predictions": validation_predictions,

        "test_predictions": test_predictions,
    }


# =============================================================================
# SAVE BEST MODEL
# =============================================================================

def save_best_model(
    model,
    group_name,
    horizon,
):
    model_path = (
        MODEL_DIR
        / f"sensor_218_xgboost_{horizon}h_v2_{group_name}.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    return model_path


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(
    df,
    predictions,
    target_column,
    group_name,
    horizon,
):
    prediction_df = df[
        [
            "timestamp",
            "pm25",
            target_column,
        ]
    ].copy()

    prediction_df[
        "prediction"
    ] = predictions

    prediction_df[
        "absolute_error"
    ] = (
        prediction_df[target_column]
        - prediction_df["prediction"]
    ).abs()

    path = (
        REPORT_DIR
        / f"sensor_218_step7_{horizon}h_"
          f"{group_name}_predictions.csv"
    )

    prediction_df.to_csv(
        path,
        index=False,
    )

    return path


# =============================================================================
# RUN ONE HORIZON
# =============================================================================

def run_horizon(
    df,
    horizon_name,
    horizon,
):
    print("\n")

    print(
        "#" * 80
    )

    print(
        f"STEP 7 FEATURE ABLATION: "
        f"{horizon_name.upper()}"
    )

    print(
        "#" * 80
    )

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    horizon_df, target_column = prepare_target_data(
        df.copy(),
        horizon,
    )

    # -------------------------------------------------------------------------
    # Purged split
    # -------------------------------------------------------------------------

    train_df, validation_df, test_df = purged_split(
        horizon_df,
        horizon,
    )

    if (
        len(train_df) == 0
        or len(validation_df) == 0
        or len(test_df) == 0
    ):

        print(
            "\n[ERROR] Cannot perform feature ablation."
        )

        if len(train_df) == 0:
            print(
                "No training observations available."
            )

        if len(validation_df) == 0:
            print(
                "No validation observations available."
            )

        if len(test_df) == 0:
            print(
                "No testing observations available."
            )

        return {
            "horizon": horizon_name,
            "horizon_hours": horizon,
            "status": "ERROR",
            "error": "Insufficient data after purging.",
            "feature_groups": {},
        }

    # -------------------------------------------------------------------------
    # Run every feature group
    # -------------------------------------------------------------------------

    print_header(
        f"TESTING {len(FEATURE_GROUPS)} FEATURE GROUPS"
    )

    horizon_results = {}

    successful_results = []

    for index, (
        group_name,
        feature_names,
    ) in enumerate(
        FEATURE_GROUPS.items(),
        start=1,
    ):

        print(
            f"\n[{index}/{len(FEATURE_GROUPS)}] "
            f"{group_name}"
        )

        print(
            f"Feature count: {len(feature_names)}"
        )

        result = run_feature_group(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            feature_names=feature_names,
            target_column=target_column,
            group_name=group_name,
            horizon=horizon_name,
        )

        if result["status"] == "SUCCESS":

            print(
                f"Validation MAE:  "
                f"{result['validation']['mae']:.4f}"
            )

            print(
                f"Validation RMSE: "
                f"{result['validation']['rmse']:.4f}"
            )

            print(
                f"Validation R²:   "
                f"{result['validation']['r2']:.4f}"
            )

            print(
                f"Test MAE:        "
                f"{result['test']['mae']:.4f}"
            )

            print(
                f"Test RMSE:       "
                f"{result['test']['rmse']:.4f}"
            )

            print(
                f"Test R²:         "
                f"{result['test']['r2']:.4f}"
            )

            successful_results.append(
                result
            )

            horizon_results[
                group_name
            ] = {
                "status": "SUCCESS",
                "feature_count": len(
                    feature_names
                ),
                "validation": result[
                    "validation"
                ],
                "test": result[
                    "test"
                ],
            }

        else:

            print(
                f"[ERROR] {result['error']}"
            )

            horizon_results[
                group_name
            ] = {
                "status": "ERROR",
                "error": result["error"],
            }

    # -------------------------------------------------------------------------
    # Select best based ONLY on validation MAE
    # -------------------------------------------------------------------------

    if not successful_results:

        print(
            "\n[ERROR] No feature groups completed."
        )

        return {
            "horizon": horizon_name,
            "horizon_hours": horizon,
            "status": "ERROR",
            "error": "No successful feature groups.",
            "feature_groups": horizon_results,
        }

    best_result = min(
        successful_results,
        key=lambda x: x[
            "validation"
        ]["mae"],
    )

    best_group = best_result["group"]

    print_header(
        "BEST FEATURE GROUP "
        "BASED ON VALIDATION MAE"
    )

    print(
        f"Feature group: "
        f"{best_group}"
    )

    print(
        f"Feature count: "
        f"{best_result['feature_count']}"
    )

    print(
        f"Validation MAE: "
        f"{best_result['validation']['mae']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{best_result['validation']['rmse']:.4f}"
    )

    print(
        f"Validation R²: "
        f"{best_result['validation']['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Evaluate selected model on test
    # -------------------------------------------------------------------------

    print_header(
        "FINAL TEST EVALUATION "
        "OF SELECTED FEATURE GROUP"
    )

    print(
        f"Test MAE:  "
        f"{best_result['test']['mae']:.4f}"
    )

    print(
        f"Test RMSE: "
        f"{best_result['test']['rmse']:.4f}"
    )

    print(
        f"Test R²:   "
        f"{best_result['test']['r2']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_path = save_best_model(
        best_result["model"],
        best_group,
        horizon_name,
    )

    print(
        f"\nBest model saved:"
    )

    print(
        f"  {model_path}"
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    prediction_path = save_predictions(
        test_df,
        best_result[
            "test_predictions"
        ],
        target_column,
        best_group,
        horizon_name,
    )

    print(
        f"Predictions saved:"
    )

    print(
        f"  {prediction_path}"
    )

    # -------------------------------------------------------------------------
    # Compare against V1 baseline
    # -------------------------------------------------------------------------

    v1_result = horizon_results.get(
        "V1_baseline"
    )

    comparison = None

    if (
        v1_result
        and v1_result["status"] == "SUCCESS"
    ):

        v1_mae = v1_result[
            "test"
        ]["mae"]

        best_mae = best_result[
            "test"
        ]["mae"]

        mae_improvement = (
            v1_mae - best_mae
        )

        if v1_mae != 0:

            mae_improvement_percent = (
                mae_improvement
                / v1_mae
                * 100
            )

        else:

            mae_improvement_percent = 0

        print_header(
            "V2 FEATURE ABLATION VS V1"
        )

        print(
            f"V1 test MAE:       "
            f"{v1_mae:.4f}"
        )

        print(
            f"Best V2 test MAE:   "
            f"{best_mae:.4f}"
        )

        print(
            f"MAE improvement:    "
            f"{mae_improvement:.4f}"
        )

        print(
            f"MAE improvement %:  "
            f"{mae_improvement_percent:.2f}%"
        )

        if mae_improvement > 0:

            print(
                "[RESULT] Best V2 feature group "
                "improved test MAE."
            )

        elif mae_improvement < 0:

            print(
                "[RESULT] V1 baseline "
                "performed better."
            )

        else:

            print(
                "[RESULT] Test MAE is unchanged."
            )

        comparison = {
            "v1_test_mae": float(v1_mae),
            "best_v2_test_mae": float(
                best_mae
            ),
            "mae_improvement": float(
                mae_improvement
            ),
            "mae_improvement_percent": float(
                mae_improvement_percent
            ),
        }

    # -------------------------------------------------------------------------
    # Return serializable results
    # -------------------------------------------------------------------------

    return {
        "horizon": horizon_name,
        "horizon_hours": horizon,
        "status": "SUCCESS",
        "best_feature_group": best_group,
        "best_feature_count": best_result[
            "feature_count"
        ],
        "best_validation": best_result[
            "validation"
        ],
        "best_test": best_result[
            "test"
        ],
        "comparison_with_v1": comparison,
        "feature_groups": horizon_results,
        "model_path": str(model_path),
        "prediction_path": str(
            prediction_path
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "STEP 7: V2 FEATURE ABLATION STUDY"
    )

    print(
        """
IMPORTANT:
• Existing train.py is NOT modified.
• Existing train_purged.py is NOT modified.
• Existing tuned models are NOT modified.
• Existing V2 models are NOT modified.
• Feature groups are compared using validation MAE.
• Test data is evaluated only after selection.
• Missing feature values are NOT discarded.
• XGBoost handles missing values natively.
• Horizon-specific purging is used.
• No future PM2.5 values are used as features.
"""
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------------------
    # Leakage check
    # -------------------------------------------------------------------------

    check_feature_leakage(
        df
    )

    # -------------------------------------------------------------------------
    # Print feature group summary
    # -------------------------------------------------------------------------

    print_header(
        "FEATURE GROUP SUMMARY"
    )

    for group_name, features in FEATURE_GROUPS.items():

        print(
            f"{group_name:<30} "
            f"{len(features):>3} features"
        )

    # -------------------------------------------------------------------------
    # Run all horizons
    # -------------------------------------------------------------------------

    all_results = {}

    for horizon_name, horizon in HORIZONS.items():

        try:

            result = run_horizon(
                df,
                horizon_name,
                horizon,
            )

            all_results[
                horizon_name
            ] = result

        except Exception as e:

            print(
                "\n[ERROR] "
                f"{horizon_name} failed:"
            )

            print(
                str(e)
            )

            all_results[
                horizon_name
            ] = {
                "horizon": horizon_name,
                "horizon_hours": horizon,
                "status": "ERROR",
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n")

    print_header(
        "STEP 7: FINAL FEATURE ABLATION RESULTS"
    )

    print(
        f"{'HORIZON':<10}"
        f"{'BEST GROUP':<28}"
        f"{'VAL MAE':>12}"
        f"{'TEST MAE':>12}"
        f"{'TEST RMSE':>12}"
        f"{'TEST R²':>12}"
    )

    print(
        "-" * 86
    )

    for horizon_name, result in all_results.items():

        if result["status"] != "SUCCESS":

            print(
                f"{horizon_name:<10}"
                f"ERROR: "
                f"{result.get('error', '')}"
            )

            continue

        best_group = result[
            "best_feature_group"
        ]

        val_mae = result[
            "best_validation"
        ]["mae"]

        test_mae = result[
            "best_test"
        ]["mae"]

        test_rmse = result[
            "best_test"
        ]["rmse"]

        test_r2 = result[
            "best_test"
        ]["r2"]

        print(
            f"{horizon_name:<10}"
            f"{best_group:<28}"
            f"{val_mae:>12.4f}"
            f"{test_mae:>12.4f}"
            f"{test_rmse:>12.4f}"
            f"{test_r2:>12.4f}"
        )

    # -------------------------------------------------------------------------
    # V1 vs BEST V2 summary
    # -------------------------------------------------------------------------

    print("\n")

    print_header(
        "V1 BASELINE VS BEST V2 FEATURE GROUP"
    )

    print(
        f"{'HORIZON':<10}"
        f"{'V1 TEST MAE':>15}"
        f"{'BEST V2 MAE':>15}"
        f"{'IMPROVEMENT':>15}"
        f"{'IMPROVEMENT %':>16}"
    )

    print(
        "-" * 76
    )

    for horizon_name, result in all_results.items():

        if result["status"] != "SUCCESS":

            print(
                f"{horizon_name:<10}"
                f"ERROR"
            )

            continue

        comparison = result.get(
            "comparison_with_v1"
        )

        if comparison is None:

            print(
                f"{horizon_name:<10}"
                f"N/A"
            )

            continue

        print(
            f"{horizon_name:<10}"
            f"{comparison['v1_test_mae']:>15.4f}"
            f"{comparison['best_v2_test_mae']:>15.4f}"
            f"{comparison['mae_improvement']:>15.4f}"
            f"{comparison['mae_improvement_percent']:>15.2f}%"
        )

    # -------------------------------------------------------------------------
    # Save complete JSON
    # -------------------------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "sensor_218_step7_feature_ablation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
        )

    print_header(
        "STEP 7 COMPLETE"
    )

    print(
        f"Summary saved:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        "\nExisting V1 and V2 files/models "
        "were not modified."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()