from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (
        CO_POLLUTANTS,
        FORECAST_HORIZONS,
        PM25_LAGS,
        PM25_ROLLING_WINDOWS,
        WEATHER_LAGS,
        WEATHER_ROLLING_WINDOWS,
        WEATHER_VARIABLES,
    )
except ImportError:  # pragma: no cover
    from src.v4.config import (
        CO_POLLUTANTS,
        FORECAST_HORIZONS,
        PM25_LAGS,
        PM25_ROLLING_WINDOWS,
        WEATHER_LAGS,
        WEATHER_ROLLING_WINDOWS,
        WEATHER_VARIABLES,
    )


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "v4" / "chengdu_master_2015_2021.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "v4"
OUTPUT_FILE = OUTPUT_DIR / "chengdu_features_v4.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_master_dataset():
    print("=" * 70)
    print("LOADING CHENGDU MASTER DATASET")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find master dataset:\n{INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# BASIC TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df):
    print("\nAdding temporal features...")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["month"] = df["timestamp"].dt.month
    df["quarter"] = df["timestamp"].dt.quarter
    df["year"] = df["timestamp"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


# ============================================================
# REUSABLE FEATURE HELPERS
# ============================================================

def grouped_station_series(df, column_name):
    return df.groupby("station_code", group_keys=False)[column_name]


def add_lag_features(df, column_name, lags, prefix):
    grouped = grouped_station_series(df, column_name)

    for lag in lags:
        df[f"{prefix}_lag_{lag}h"] = grouped.shift(lag)

    return df


def add_rolling_features(df, column_name, windows, prefix, include_std=True):
    past_values = grouped_station_series(df, column_name).shift(1)

    for window in windows:
        grouped_past = past_values.groupby(df["station_code"])

        df[f"{prefix}_rolling_mean_{window}h"] = (
            grouped_past.rolling(
                window=window,
                min_periods=max(3, window // 2),
            )
            .mean()
            .reset_index(level=0, drop=True)
        )

        if include_std:
            df[f"{prefix}_rolling_std_{window}h"] = (
                grouped_past.rolling(
                    window=window,
                    min_periods=max(3, window // 2),
                )
                .std()
                .reset_index(level=0, drop=True)
            )

    return df


# ============================================================
# PM2.5 FEATURES
# ============================================================

def add_pm25_lag_features(df):
    print("Adding PM2.5 lag features...")
    return add_lag_features(df, "pm25", PM25_LAGS, "pm25")


def add_pm25_rolling_features(df):
    print("Adding PM2.5 rolling features...")
    return add_rolling_features(df, "pm25", PM25_ROLLING_WINDOWS, "pm25", include_std=True)


# ============================================================
# WEATHER FEATURES
# ============================================================

def add_weather_features(df):
    print("Adding meteorological features...")

    for variable in WEATHER_VARIABLES:
        df = add_lag_features(df, variable, WEATHER_LAGS, variable)
        df = add_rolling_features(
            df,
            variable,
            WEATHER_ROLLING_WINDOWS,
            variable,
            include_std=False,
        )

    return df


# ============================================================
# WIND FEATURES
# ============================================================

def add_wind_features(df):
    print("Adding wind features...")

    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["wind_direction"] = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) % 360)
    df["wind_direction_sin"] = np.sin(np.radians(df["wind_direction"]))
    df["wind_direction_cos"] = np.cos(np.radians(df["wind_direction"]))

    return df


# ============================================================
# CO-POLLUTANT FEATURES
# ============================================================

def add_copollutant_features(df):
    print("Adding co-pollutant features...")

    for variable in CO_POLLUTANTS:
        df = add_lag_features(df, variable, [1, 3, 6, 12, 24], variable)
        df = add_rolling_features(
            df,
            variable,
            [6, 12, 24, 48],
            variable,
            include_std=False,
        )

    return df


# ============================================================
# FORECAST TARGETS
# ============================================================

def add_targets(df):
    print("Adding forecast targets...")

    grouped = grouped_station_series(df, "pm25")

    for horizon_name, horizon_hours in FORECAST_HORIZONS.items():
        df[f"target_pm25_{horizon_name}"] = grouped.shift(-horizon_hours)

    return df


# ============================================================
# FEATURE AVAILABILITY FLAGS
# ============================================================

def add_missingness_flags(df):
    print("Adding missingness indicators...")

    df["pm25_missing"] = df["pm25"].isna().astype(int)

    for variable in CO_POLLUTANTS:
        df[f"{variable}_missing"] = df[variable].isna().astype(int)

    for variable in WEATHER_VARIABLES:
        df[f"{variable}_missing"] = df[variable].isna().astype(int)

    return df


# ============================================================
# VALIDATE TIME ORDER
# ============================================================

def validate_time_order(df):
    print("\nValidating station time ordering...")

    duplicate_count = df.duplicated(subset=["station_code", "timestamp"]).sum()
    print(f"Duplicate station timestamps: {duplicate_count:,}")

    if duplicate_count > 0:
        raise ValueError("Duplicate station timestamps detected.")

    out_of_order = 0
    for station in df["station_code"].unique():
        station_df = df[df["station_code"] == station]
        if not station_df["timestamp"].is_monotonic_increasing:
            out_of_order += 1

    print(f"Stations with incorrect time ordering: {out_of_order}")
    if out_of_order > 0:
        raise ValueError("At least one station is not chronologically ordered.")


# ============================================================
# FINAL COLUMN ORGANISATION
# ============================================================

def organise_columns(df):
    metadata = [
        "timestamp",
        "station_code",
        "station_name",
        "longitude",
        "latitude",
    ]

    current_observations = [
        "pm25",
        "pm10",
        "so2",
        "no2",
        "o3_8h",
        "co",
        "aqi",
        "u10",
        "v10",
        "temperature",
        "relative_humidity",
        "boundary_layer_height",
        "wind_speed",
        "wind_direction",
    ]

    temporal = [
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
        "wind_direction_sin",
        "wind_direction_cos",
    ]

    missingness = [column for column in df.columns if column.endswith("_missing")]
    targets = [column for column in df.columns if column.startswith("target_pm25_")]

    already_used = set(metadata + current_observations + temporal + missingness + targets)
    engineered = [column for column in df.columns if column not in already_used]

    ordered_columns = (
        metadata
        + current_observations
        + temporal
        + engineered
        + missingness
        + targets
    )

    return df[ordered_columns]


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_master_dataset()

    df = df.sort_values(["station_code", "timestamp"]).reset_index(drop=True)
    validate_time_order(df)

    df = add_temporal_features(df)
    df = add_pm25_lag_features(df)
    df = add_pm25_rolling_features(df)
    df = add_weather_features(df)
    df = add_wind_features(df)
    df = add_copollutant_features(df)
    df = add_missingness_flags(df)
    df = add_targets(df)
    df = organise_columns(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("V4 FEATURE DATASET CREATED")
    print("=" * 70)

    print(f"\nOutput:")
    print(OUTPUT_FILE)
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nMemory estimate: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print("\nTarget availability:")
    for horizon_name in FORECAST_HORIZONS:
        target = f"target_pm25_{horizon_name}"
        available = df[target].notna().sum()
        percentage = available / len(df) * 100
        print(f"  {horizon_name:>4}: {available:,} ({percentage:.2f}%)")

    print("\nMissing values in major current variables:")
    major_variables = [
        "pm25",
        "pm10",
        "so2",
        "no2",
        "o3_8h",
        "co",
        "temperature",
        "relative_humidity",
        "u10",
        "v10",
        "boundary_layer_height",
    ]

    for variable in major_variables:
        missing = df[variable].isna().sum()
        percentage = missing / len(df) * 100
        print(f"  {variable:25s}: {missing:8,} ({percentage:6.2f}%)")

    print("\nFeature groups:")
    print(f"  PM2.5 lags: {len(PM25_LAGS)}")
    print(f"  PM2.5 rolling windows: {len(PM25_ROLLING_WINDOWS)}")
    print(f"  Weather variables: {len(WEATHER_VARIABLES)}")
    print(f"  Co-pollutants: {len(CO_POLLUTANTS)}")
    print(f"  Forecast horizons: {len(FORECAST_HORIZONS)}")

    print("\nDone.")


if __name__ == "__main__":
    main()