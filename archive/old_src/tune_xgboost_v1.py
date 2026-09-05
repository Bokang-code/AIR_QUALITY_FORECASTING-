"""
STEP 6 — V1 XGBOOST HYPERPARAMETER TUNING

Purpose:
    Tune XGBoost hyperparameters using ONLY the original V1
    feature set.

Important:
    • Existing files are NOT modified.
    • Existing models are NOT modified.
    • V1 features only.
    • Chronological splitting is preserved.
    • Horizon-specific purging is used.
    • Validation MAE selects the best parameters.
    • Test data is evaluated only after tuning.
    • Missing feature values are retained.
    • XGBoost handles missing feature values natively.
    • Tuned models are saved separately.
"""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path("data/processed/engineered_sensor_218.csv")

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


# Number of random hyperparameter combinations
N_RANDOM_TRIALS = 40

RANDOM_STATE = 42


# =============================================================================
# V1 FEATURE SET
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


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

PARAMETER_SPACE = {
    "n_estimators": [
        300,
        500,
        700,
        900,
        1200,
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.07,
        0.1,
    ],

    "max_depth": [
        3,
        4,
        5,
        6,
        7,
    ],

    "min_child_weight": [
        1,
        3,
        5,
        7,
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "reg_alpha": [
        0.0,
        0.01,
        0.1,
        1.0,
    ],

    "reg_lambda": [
        0.5,
        1.0,
        2.0,
        5.0,
    ],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title, width=90):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subheader(title, width=90):
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print_header(
        "LOADING DATA FOR V1 XGBOOST TUNING"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded:    {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns):,}"
    )

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
    # Detect PM2.5 column
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
        f"\nPM2.5 observations: "
        f"{df['pm25'].notna().sum():,}"
    )

    print(
        f"PM2.5 missing:      "
        f"{df['pm25'].isna().sum():,}"
    )

    return df


# =============================================================================
# VALIDATE V1 FEATURES
# =============================================================================

def validate_v1_features(df):

    print_header(
        "VALIDATING V1 FEATURE SET"
    )

    missing_features = [
        feature
        for feature in V1_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        print(
            "[FAIL] Missing V1 features:"
        )

        for feature in missing_features:

            print(
                f"  - {feature}"
            )

        raise ValueError(
            "V1 feature validation failed."
        )

    print(
        f"[PASS] All {len(V1_FEATURES)} "
        f"V1 features are present."
    )


# =============================================================================
# TARGET CREATION
# =============================================================================

def prepare_target_data(
    df,
    horizon,
):

    target_column = (
        f"target_pm25_{horizon}h"
    )

    print_subheader(
        f"PREPARING {horizon}H TARGET"
    )

    if target_column in df.columns:

        print(
            f"[INFO] {target_column}"
        )

    else:

        print(
            f"[INFO] Creating "
            f"{target_column}"
        )

        df[target_column] = (
            df["pm25"]
            .shift(-horizon)
        )

    before = len(df)

    df_model = df[
        df[target_column].notna()
    ].copy()

    removed = (
        before
        - len(df_model)
    )

    print(
        "Rows removed because future "
        f"target is unavailable: {removed:,}"
    )

    return (
        df_model,
        target_column,
    )


# =============================================================================
# PURGED CHRONOLOGICAL SPLIT
# =============================================================================

def purged_split(
    df,
    horizon,
    train_ratio=0.70,
    validation_ratio=0.15,
):

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n
        * (
            train_ratio
            + validation_ratio
        )
    )

    # -------------------------------------------------------------------------
    # Horizon-specific purge
    # -------------------------------------------------------------------------

    train_end_purged = (
        train_end
        - horizon
    )

    validation_start = (
        train_end
        + horizon
    )

    validation_end_purged = (
        validation_end
        - horizon
    )

    test_start = (
        validation_end
        + horizon
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
        f"Original rows:       {n:,}"
    )

    print(
        f"Training rows:       {len(train):,}"
    )

    print(
        f"Validation rows:     {len(validation):,}"
    )

    print(
        f"Testing rows:        {len(test):,}"
    )

    print(
        f"Train purge:         {horizon} hours"
    )

    print(
        f"Validation purge:    {horizon} hours"
    )

    if len(train) > 0:

        print("\nTraining:")

        print(
            f"  {train['timestamp'].min()} "
            f"→ "
            f"{train['timestamp'].max()}"
        )

    if len(validation) > 0:

        print("\nValidation:")

        print(
            f"  {validation['timestamp'].min()} "
            f"→ "
            f"{validation['timestamp'].max()}"
        )

    if len(test) > 0:

        print("\nTesting:")

        print(
            f"  {test['timestamp'].min()} "
            f"→ "
            f"{test['timestamp'].max()}"
        )

    return (
        train,
        validation,
        test,
    )


# =============================================================================
# GENERATE RANDOM PARAMETER COMBINATIONS
# =============================================================================

def generate_random_parameters():

    random.seed(
        RANDOM_STATE
    )

    combinations = []

    seen = set()

    keys = list(
        PARAMETER_SPACE.keys()
    )

    max_possible = np.prod(
        [
            len(
                PARAMETER_SPACE[key]
            )
            for key in keys
        ]
    )

    trials = min(
        N_RANDOM_TRIALS,
        int(max_possible),
    )

    while len(combinations) < trials:

        params = {
            key: random.choice(
                PARAMETER_SPACE[key]
            )
            for key in keys
        }

        signature = tuple(
            params[key]
            for key in keys
        )

        if signature not in seen:

            seen.add(
                signature
            )

            combinations.append(
                params
            )

    return combinations


# =============================================================================
# TUNE ONE HORIZON
# =============================================================================

def tune_horizon(
    train_df,
    validation_df,
    test_df,
    target_column,
    horizon_name,
    horizon,
):

    print_header(
        f"XGBOOST HYPERPARAMETER TUNING — "
        f"{horizon_name}"
    )

    X_train = train_df[
        V1_FEATURES
    ].copy()

    y_train = train_df[
        target_column
    ]

    X_validation = validation_df[
        V1_FEATURES
    ].copy()

    y_validation = validation_df[
        target_column
    ]

    X_test = test_df[
        V1_FEATURES
    ].copy()

    y_test = test_df[
        target_column
    ]

    # -------------------------------------------------------------------------
    # Convert non-numeric features if required
    # -------------------------------------------------------------------------

    for feature in V1_FEATURES:

        if not pd.api.types.is_numeric_dtype(
            X_train[feature]
        ):

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

    print_subheader(
        "MISSING FEATURE VALUES"
    )

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
        f"Test missing cells:       "
        f"{test_missing:,}"
    )

    print(
        "\nXGBoost will handle these "
        "missing values natively."
    )

    # -------------------------------------------------------------------------
    # Generate parameter combinations
    # -------------------------------------------------------------------------

    parameter_combinations = (
        generate_random_parameters()
    )

    print_subheader(
        f"TUNING {horizon_name}"
    )

    print(
        f"Feature count:       "
        f"{len(V1_FEATURES)}"
    )

    print(
        f"Random combinations: "
        f"{len(parameter_combinations)}"
    )

    print(
        "Selection metric:    "
        "Validation MAE"
    )

    print(
        "Test set:            NOT USED"
    )

    # -------------------------------------------------------------------------
    # Track best model
    # -------------------------------------------------------------------------

    best_mae = float("inf")

    best_params = None

    best_model = None

    tuning_results = []

    # -------------------------------------------------------------------------
    # Run trials
    # -------------------------------------------------------------------------

    for index, params in enumerate(
        parameter_combinations,
        start=1,
    ):

        print(
            f"\n[{index}/"
            f"{len(parameter_combinations)}]"
            f" Testing parameters..."
        )

        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        validation_predictions = (
            model.predict(
                X_validation
            )
        )

        validation_mae = (
            mean_absolute_error(
                y_validation,
                validation_predictions,
            )
        )

        print(
            f"Validation MAE: "
            f"{validation_mae:.4f}"
        )

        tuning_results.append(
            {
                "trial": index,
                "validation_mae": float(
                    validation_mae
                ),
                "parameters": params,
            }
        )

        # ---------------------------------------------------------------------
        # New best model
        # ---------------------------------------------------------------------

        if validation_mae < best_mae:

            best_mae = validation_mae

            best_params = params.copy()

            best_model = model

            print(
                ">>> NEW BEST MODEL"
            )

            print(
                f">>> Best validation MAE: "
                f"{best_mae:.4f}"
            )

    # -------------------------------------------------------------------------
    # Best parameters
    # -------------------------------------------------------------------------

    print_header(
        f"BEST V1 XGBOOST PARAMETERS — "
        f"{horizon_name}"
    )

    print(
        f"Best validation MAE: "
        f"{best_mae:.4f}"
    )

    print(
        "\nBest parameters:"
    )

    for key, value in best_params.items():

        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------------------
    # FINAL TEST EVALUATION
    # -------------------------------------------------------------------------

    print_header(
        f"FINAL TEST EVALUATION — "
        f"{horizon_name}"
    )

    print(
        "IMPORTANT: Hyperparameters were "
        "selected using validation only."
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
    # Save tuned model
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"sensor_218_xgboost_v1_tuned_"
          f"{horizon_name}.pkl"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    print(
        "\nModel saved:"
    )

    print(
        f"  {model_path}"
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    prediction_df = test_df[
        [
            "timestamp",
            "pm25",
            target_column,
        ]
    ].copy()

    prediction_df[
        "prediction"
    ] = test_predictions

    prediction_df[
        "absolute_error"
    ] = (
        prediction_df[target_column]
        - prediction_df["prediction"]
    ).abs()

    prediction_path = (
        REPORT_DIR
        / f"sensor_218_v1_tuned_"
          f"{horizon_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        "Predictions saved:"
    )

    print(
        f"  {prediction_path}"
    )

    # -------------------------------------------------------------------------
    # Save tuning results
    # -------------------------------------------------------------------------

    result = {
        "horizon": horizon_name,
        "horizon_hours": horizon,
        "feature_count": len(V1_FEATURES),
        "features": V1_FEATURES,
        "random_trials": len(
            parameter_combinations
        ),
        "selection_metric": (
            "validation_mae"
        ),
        "best_validation_mae": float(
            best_mae
        ),
        "best_parameters": best_params,
        "test_metrics": test_metrics,
        "model_path": str(
            model_path
        ),
        "prediction_path": str(
            prediction_path
        ),
        "tuning_trials": tuning_results,
    }

    result_path = (
        REPORT_DIR
        / f"sensor_218_v1_tuning_"
          f"{horizon_name}.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    print(
        "Results saved:"
    )

    print(
        f"  {result_path}"
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "STEP 6 — V1 XGBOOST "
        "HYPERPARAMETER TUNING"
    )

    print(
        """
IMPORTANT:

• V1 feature set only.
• XGBoost hyperparameters are being tuned.
• Chronological splitting is preserved.
• Horizon-specific purging is used.
• Validation MAE selects the best parameters.
• TEST DATA IS NOT USED FOR TUNING.
• Missing feature values are retained.
• XGBoost handles missing values natively.
• Existing V1 files are NOT modified.
• Existing models are NOT modified.
• Tuned models are saved separately.
"""
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------------------
    # Validate V1 features
    # -------------------------------------------------------------------------

    validate_v1_features(
        df
    )

    # -------------------------------------------------------------------------
    # Print configuration
    # -------------------------------------------------------------------------

    print_header(
        "TUNING CONFIGURATION"
    )

    print(
        f"V1 features:       "
        f"{len(V1_FEATURES)}"
    )

    print(
        f"Random trials:     "
        f"{N_RANDOM_TRIALS}"
    )

    print(
        "Selection metric:  "
        "Validation MAE"
    )

    print(
        f"Random state:      "
        f"{RANDOM_STATE}"
    )

    print(
        "\nHorizons:"
    )

    for horizon_name, horizon in HORIZONS.items():

        print(
            f"  {horizon_name}: "
            f"{horizon} hours"
        )

    # -------------------------------------------------------------------------
    # Run all horizons
    # -------------------------------------------------------------------------

    all_results = {}

    for horizon_name, horizon in HORIZONS.items():

        print("\n")

        print(
            "#" * 90
        )

        print(
            f"V1 XGBOOST TUNING — "
            f"{horizon_name.upper()}"
        )

        print(
            "#" * 90
        )

        try:

            # -------------------------------------------------------------
            # Target
            # -------------------------------------------------------------

            horizon_df, target_column = (
                prepare_target_data(
                    df.copy(),
                    horizon,
                )
            )

            # -------------------------------------------------------------
            # Split
            # -------------------------------------------------------------

            train_df, validation_df, test_df = (
                purged_split(
                    horizon_df,
                    horizon,
                )
            )

            # -------------------------------------------------------------
            # Check data availability
            # -------------------------------------------------------------

            if (
                len(train_df) == 0
                or len(validation_df) == 0
                or len(test_df) == 0
            ):

                raise ValueError(
                    f"Insufficient data for "
                    f"{horizon_name}."
                )

            # -------------------------------------------------------------
            # Tune
            # -------------------------------------------------------------

            result = tune_horizon(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                target_column=target_column,
                horizon_name=horizon_name,
                horizon=horizon,
            )

            all_results[
                horizon_name
            ] = result

        except Exception as e:

            print(
                f"\n[ERROR] "
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
        "FINAL V1 XGBOOST TUNING RESULTS"
    )

    print(
        f"{'HORIZON':<12}"
        f"{'VAL MAE':>14}"
        f"{'TEST MAE':>14}"
        f"{'TEST RMSE':>16}"
        f"{'TEST R²':>14}"
    )

    print(
        "-" * 70
    )

    for horizon_name, result in (
        all_results.items()
    ):

        if result.get("status") == "ERROR":

            print(
                f"{horizon_name:<12}"
                f"ERROR"
            )

            continue

        print(
            f"{horizon_name:<12}"
            f"{result['best_validation_mae']:>14.4f}"
            f"{result['test_metrics']['mae']:>14.4f}"
            f"{result['test_metrics']['rmse']:>16.4f}"
            f"{result['test_metrics']['r2']:>14.4f}"
        )

    # -------------------------------------------------------------------------
    # Save overall summary
    # -------------------------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "sensor_218_v1_xgboost_tuning_summary.json"
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
        "STEP 6 COMPLETE"
    )

    print(
        "Overall summary saved:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        "\nV1 XGBoost tuning completed."
    )

    print(
        "The test set was used only for "
        "final evaluation after "
        "hyperparameter selection."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()