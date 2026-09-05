from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/v4/chengdu_features_v4.csv"
)

MODEL_DIR = Path(
    "models/v4/random_forest"
)

REPORT_DIR = Path(
    "reports/v4/random_forest"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Forecast horizons
HORIZONS = {
    "24h": 24,
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


# ============================================================
# FEATURE GROUPS
# ============================================================

TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


PM25_FEATURES = [
    "pm25_lag_1h",
    "pm25_lag_2h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
    "pm25_lag_48h",
    "pm25_lag_72h",
    "pm25_lag_168h",
    "pm25_lag_336h",
    "pm25_lag_720h",
    "pm25_rolling_mean_3h",
    "pm25_rolling_std_3h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_std_6h",
    "pm25_rolling_mean_12h",
    "pm25_rolling_std_12h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_std_24h",
    "pm25_rolling_mean_48h",
    "pm25_rolling_std_48h",
    "pm25_rolling_mean_72h",
    "pm25_rolling_std_72h",
    "pm25_rolling_mean_168h",
    "pm25_rolling_std_168h",
]


FEATURE_COLUMNS = (
    TEMPORAL_FEATURES
    + PM25_FEATURES
)


# ============================================================
# RANDOM FOREST SETTINGS
# ============================================================

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[mask]
    predicted = predicted[mask]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RANDOM FOREST V4 - PURGED FORECASTING")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Stations: "
        f"{df['station_code'].nunique()}"
    )

    print(
        f"Features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    # ========================================================
    # CHECK FEATURES
    # ========================================================

    missing_features = [
        col
        for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features:\n"
            + "\n".join(
                missing_features
            )
        )

    # ========================================================
    # RESULTS
    # ========================================================

    overall_results = []

    # ========================================================
    # TRAIN EACH HORIZON
    # ========================================================

    for horizon_name, horizon in HORIZONS.items():

        print("\n" + "=" * 70)
        print(
            f"HORIZON: {horizon_name}"
        )
        print("=" * 70)

        target_column = (
            f"target_pm25_{horizon_name}"
        )

        if target_column not in df.columns:

            raise ValueError(
                f"Missing target: "
                f"{target_column}"
            )

        # ----------------------------------------------------
        # CREATE DATASET
        # ----------------------------------------------------

        required_columns = (
            FEATURE_COLUMNS
            + [target_column]
        )

        data = df[
            [
                "timestamp",
                "station_code",
                "pm25",
            ]
            + required_columns
        ].copy()

        # Target must exist
        data = data[
            data[target_column].notna()
        ].copy()

        # ----------------------------------------------------
        # PURGED TIME SPLIT
        # ----------------------------------------------------

        train_cutoff = (
            pd.Timestamp("2020-01-01")
            - pd.Timedelta(
                hours=horizon
            )
        )

        validation_start = pd.Timestamp(
            "2020-01-01"
        )

        validation_end = (
            pd.Timestamp("2021-01-01")
            - pd.Timedelta(
                hours=horizon
            )
        )

        test_start = pd.Timestamp(
            "2021-01-01"
        )

        train_mask = (
            data["timestamp"]
            < train_cutoff
        )

        validation_mask = (
            (data["timestamp"] >= validation_start)
            &
            (data["timestamp"] < validation_end)
        )

        test_mask = (
            data["timestamp"]
            >= test_start
        )

        train = data[
            train_mask
        ].copy()

        validation = data[
            validation_mask
        ].copy()

        test = data[
            test_mask
        ].copy()

        print(
            f"Train rows: "
            f"{len(train):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        print(
            f"Test rows: "
            f"{len(test):,}"
        )

        # ----------------------------------------------------
        # FEATURES / TARGET
        # ----------------------------------------------------

        X_train = train[
            FEATURE_COLUMNS
        ]

        y_train = train[
            target_column
        ]

        X_validation = validation[
            FEATURE_COLUMNS
        ]

        y_validation = validation[
            target_column
        ]

        X_test = test[
            FEATURE_COLUMNS
        ]

        y_test = test[
            target_column
        ]

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        print(
            "\nTraining Random Forest..."
        )

        model = RandomForestRegressor(
            **RF_PARAMS
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "Training complete."
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_predictions = (
            model.predict(
                X_validation
            )
        )

        val_mae, val_rmse, val_r2 = (
            calculate_metrics(
                y_validation,
                validation_predictions
            )
        )

        print(
            "\nValidation:"
        )

        print(
            f"MAE  : {val_mae:.4f}"
        )

        print(
            f"RMSE : {val_rmse:.4f}"
        )

        print(
            f"R²   : {val_r2:.4f}"
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        print(
            "\nGenerating test predictions..."
        )

        test_predictions = model.predict(
            X_test
        )

        test_mae, test_rmse, test_r2 = (
            calculate_metrics(
                y_test,
                test_predictions
            )
        )

        print(
            "\nTest:"
        )

        print(
            f"MAE  : {test_mae:.4f}"
        )

        print(
            f"RMSE : {test_rmse:.4f}"
        )

        print(
            f"R²   : {test_r2:.4f}"
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_file = (
            MODEL_DIR
            / f"random_forest_{horizon_name}_purged.joblib"
        )

        joblib.dump(
            model,
            model_file
        )

        print(
            f"\nModel saved: "
            f"{model_file}"
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        predictions_df = test[
            [
                "timestamp",
                "station_code",
                "pm25",
                target_column,
            ]
        ].copy()

        predictions_df[
            "prediction"
        ] = test_predictions

        predictions_df[
            "horizon"
        ] = horizon_name

        predictions_file = (
            REPORT_DIR
            / f"random_forest_{horizon_name}_purged_predictions.csv"
        )

        predictions_df.to_csv(
            predictions_file,
            index=False
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        overall_results.append(
            {
                "horizon":
                    horizon_name,

                "horizon_hours":
                    horizon,

                "train_samples":
                    len(train),

                "validation_samples":
                    len(validation),

                "test_samples":
                    len(test),

                "validation_mae":
                    val_mae,

                "validation_rmse":
                    val_rmse,

                "validation_r2":
                    val_r2,

                "test_mae":
                    test_mae,

                "test_rmse":
                    test_rmse,

                "test_r2":
                    test_r2,
            }
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        overall_results
    )

    results_file = (
        REPORT_DIR
        / "random_forest_purged_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("RANDOM FOREST FINAL RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "horizon",
                "test_mae",
                "test_rmse",
                "test_r2",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nResults saved:")
    print(results_file)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()