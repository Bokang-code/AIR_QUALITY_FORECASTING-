from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    from config import FORECAST_HORIZONS, TRAIN_END, TEST_START, VALIDATION_END
except ImportError:  # pragma: no cover
    from src.v4.config import FORECAST_HORIZONS, TRAIN_END, TEST_START, VALIDATION_END


# ============================================================
# V4-C: TEMPORAL + METEOROLOGY + CO-POLLUTANTS XGBOOST
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/processed/v4/chengdu_features_v4.csv")
MODEL_DIR = Path("models/v4/temporal_meteorology_copollutants")
REPORT_DIR = Path("reports/v4/temporal_meteorology_copollutants")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values(["station_code", "timestamp"]).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Number of stations: {df['station_code'].nunique()}")
print()


# ============================================================
# FEATURE GROUP 1 — TEMPORAL
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


# ============================================================
# FEATURE GROUP 2 — PM2.5 HISTORY
# ============================================================

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


# ============================================================
# FEATURE GROUP 3 — METEOROLOGY
# ============================================================

METEOROLOGY_FEATURES = [
    "temperature",
    "temperature_lag_1h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_12h",
    "temperature_lag_24h",
    "temperature_rolling_mean_6h",
    "temperature_rolling_mean_12h",
    "temperature_rolling_mean_24h",
    "temperature_rolling_mean_48h",
    "temperature_rolling_mean_72h",
    "relative_humidity",
    "relative_humidity_lag_1h",
    "relative_humidity_lag_3h",
    "relative_humidity_lag_6h",
    "relative_humidity_lag_12h",
    "relative_humidity_lag_24h",
    "relative_humidity_rolling_mean_6h",
    "relative_humidity_rolling_mean_12h",
    "relative_humidity_rolling_mean_24h",
    "relative_humidity_rolling_mean_48h",
    "relative_humidity_rolling_mean_72h",
    "u10",
    "u10_lag_1h",
    "u10_lag_3h",
    "u10_lag_6h",
    "u10_lag_12h",
    "u10_lag_24h",
    "u10_rolling_mean_6h",
    "u10_rolling_mean_12h",
    "u10_rolling_mean_24h",
    "u10_rolling_mean_48h",
    "u10_rolling_mean_72h",
    "v10",
    "v10_lag_1h",
    "v10_lag_3h",
    "v10_lag_6h",
    "v10_lag_12h",
    "v10_lag_24h",
    "v10_rolling_mean_6h",
    "v10_rolling_mean_12h",
    "v10_rolling_mean_24h",
    "v10_rolling_mean_48h",
    "v10_rolling_mean_72h",
    "wind_speed",
    "wind_direction_sin",
    "wind_direction_cos",
    "boundary_layer_height",
    "boundary_layer_height_lag_1h",
    "boundary_layer_height_lag_3h",
    "boundary_layer_height_lag_6h",
    "boundary_layer_height_lag_12h",
    "boundary_layer_height_lag_24h",
    "boundary_layer_height_rolling_mean_6h",
    "boundary_layer_height_rolling_mean_12h",
    "boundary_layer_height_rolling_mean_24h",
    "boundary_layer_height_rolling_mean_48h",
    "boundary_layer_height_rolling_mean_72h",
]


# ============================================================
# FEATURE GROUP 4 — CO-POLLUTANTS
# ============================================================

COPOLLUTANT_FEATURES = [
    "pm10_lag_1h",
    "pm10_lag_3h",
    "pm10_lag_6h",
    "pm10_lag_12h",
    "pm10_lag_24h",
    "pm10_rolling_mean_6h",
    "pm10_rolling_mean_12h",
    "pm10_rolling_mean_24h",
    "pm10_rolling_mean_48h",
    "so2_lag_1h",
    "so2_lag_3h",
    "so2_lag_6h",
    "so2_lag_12h",
    "so2_lag_24h",
    "so2_rolling_mean_6h",
    "so2_rolling_mean_12h",
    "so2_rolling_mean_24h",
    "so2_rolling_mean_48h",
    "no2_lag_1h",
    "no2_lag_3h",
    "no2_lag_6h",
    "no2_lag_12h",
    "no2_lag_24h",
    "no2_rolling_mean_6h",
    "no2_rolling_mean_12h",
    "no2_rolling_mean_24h",
    "no2_rolling_mean_48h",
    "o3_8h_lag_1h",
    "o3_8h_lag_3h",
    "o3_8h_lag_6h",
    "o3_8h_lag_12h",
    "o3_8h_lag_24h",
    "o3_8h_rolling_mean_6h",
    "o3_8h_rolling_mean_12h",
    "o3_8h_rolling_mean_24h",
    "o3_8h_rolling_mean_48h",
    "co_lag_1h",
    "co_lag_3h",
    "co_lag_6h",
    "co_lag_12h",
    "co_lag_24h",
    "co_rolling_mean_6h",
    "co_rolling_mean_12h",
    "co_rolling_mean_24h",
    "co_rolling_mean_48h",
]


FEATURE_COLUMNS = TEMPORAL_FEATURES + PM25_FEATURES + METEOROLOGY_FEATURES + COPOLLUTANT_FEATURES


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_required_features(feature_columns):
    missing_features = [feature for feature in feature_columns if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "The following required features are missing from the dataset:\n"
            f"{missing_features}"
        )


def prepare_horizon_data(target_column, horizon_hours):
    working = df.dropna(subset=[target_column]).copy()

    train_cutoff = TRAIN_END - pd.Timedelta(hours=horizon_hours)
    validation_cutoff = VALIDATION_END - pd.Timedelta(hours=horizon_hours)

    train_mask = working["timestamp"] < train_cutoff
    validation_mask = (working["timestamp"] >= TRAIN_END) & (working["timestamp"] < validation_cutoff)
    test_mask = working["timestamp"] >= TEST_START

    train_df = working.loc[train_mask].copy()
    validation_df = working.loc[validation_mask].copy()
    test_df = working.loc[test_mask].copy()

    if len(train_df) == 0:
        raise ValueError(f"No training data for {target_column}")
    if len(validation_df) == 0:
        raise ValueError(f"No validation data for {target_column}")
    if len(test_df) == 0:
        raise ValueError(f"No test data for {target_column}")

    return train_df, validation_df, test_df


def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

check_required_features(FEATURE_COLUMNS)

print(f"Temporal features: {len(TEMPORAL_FEATURES)}")
print(f"PM2.5 history features: {len(PM25_FEATURES)}")
print(f"Meteorological features: {len(METEOROLOGY_FEATURES)}")
print(f"Co-pollutant features: {len(COPOLLUTANT_FEATURES)}")
print(f"Total features: {len(FEATURE_COLUMNS)}")
print()

print("Co-pollutant features included:")
for feature in COPOLLUTANT_FEATURES:
    print(f"  - {feature}")
print()


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


# ============================================================
# TRAINING
# ============================================================

results = []

for horizon_name, horizon_hours in FORECAST_HORIZONS.items():
    print("=" * 70)
    print(f"TRAINING TEMPORAL + METEOROLOGY + CO-POLLUTANTS XGBOOST — {horizon_name}")
    print("=" * 70)

    target_column = f"target_pm25_{horizon_name}"

    print(f"Forecast horizon: {horizon_hours} hours")
    print(f"Training origins end before: {TRAIN_END - pd.Timedelta(hours=horizon_hours)}")
    print(f"Validation origins: {TRAIN_END} to {VALIDATION_END - pd.Timedelta(hours=horizon_hours)}")
    print(f"Test origins start: {TEST_START}")
    print()

    train_df, validation_df, test_df = prepare_horizon_data(target_column, horizon_hours)

    print("Split sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")
    print()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_column]
    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[target_column]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_column]

    print("Training XGBoost...")
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)

    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    validation_metrics = compute_metrics(y_validation, validation_predictions)
    test_metrics = compute_metrics(y_test, test_predictions)

    model_path = MODEL_DIR / f"xgboost_temporal_meteorology_copollutants_{horizon_name}_purged.joblib"
    joblib.dump(model, model_path)

    predictions = test_df[["timestamp", "station_code", "station_name", "pm25", target_column]].copy()
    predictions["prediction"] = test_predictions
    predictions_path = REPORT_DIR / f"temporal_meteorology_copollutants_{horizon_name}_purged_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    results.append(
        {
            "horizon": horizon_name,
            "horizon_hours": horizon_hours,
            "train_samples": len(train_df),
            "validation_samples": len(validation_df),
            "test_samples": len(test_df),
            "n_features": len(FEATURE_COLUMNS),
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_r2": validation_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }
    )

    print()
    print("Validation results:")
    print(f"  MAE:  {validation_metrics['mae']:.4f}")
    print(f"  RMSE: {validation_metrics['rmse']:.4f}")
    print(f"  R²:   {validation_metrics['r2']:.4f}")

    print()
    print("Test results:")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")

    print()
    print(f"Model saved to: {model_path}")
    print(f"Predictions saved to: {predictions_path}")
    print()


results_df = pd.DataFrame(results)
results_path = REPORT_DIR / "temporal_meteorology_copollutants_xgboost_purged_results.csv"
results_df.to_csv(results_path, index=False)

print("=" * 70)
print("FINAL PURGED TEMPORAL + METEOROLOGY + CO-POLLUTANTS XGBOOST RESULTS")
print("=" * 70)
print(
    results_df[
        [
            "horizon",
            "train_samples",
            "validation_samples",
            "test_samples",
            "n_features",
            "test_mae",
            "test_rmse",
            "test_r2",
        ]
    ].to_string(index=False)
)
print()
print(f"Results saved to: {results_path}")
print()