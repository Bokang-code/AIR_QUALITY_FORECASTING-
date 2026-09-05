from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "chengdu_master_2015_2021.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "chengdu_features_v4.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FORECAST_HORIZONS = {
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

PM25_LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
    336,
    720,
]

PM25_ROLLING_WINDOWS = [
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

WEATHER_VARIABLES = [
    "temperature",
    "relative_humidity",
    "u10",
    "v10",
    "boundary_layer_height",
]

WEATHER_LAGS = [
    1,
    3,
    6,
    12,
    24,
]

WEATHER_ROLLING_WINDOWS = [
    6,
    12,
    24,
    48,
    72,
]

CO_POLLUTANTS = [
    "pm10",
    "so2",
    "no2",
    "o3_8h",
    "co",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_master_dataset():
    print("=" * 70)
    print("LOADING CHENGDU MASTER DATASET")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find master dataset:\n{INPUT_FILE}"
        )

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

    df["week_of_year"] = (
        df["timestamp"].dt.isocalendar().week.astype(int)
    )

    df["month"] = df["timestamp"].dt.month

    df["quarter"] = df["timestamp"].dt.quarter

    df["year"] = df["timestamp"].dt.year

    # Weekend indicator
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical encoding
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    return df


# ============================================================
# PM2.5 LAG FEATURES
# ============================================================

def add_pm25_lag_features(df):
    print("Adding PM2.5 lag features...")

    grouped = df.groupby(
        "station_code",
        group_keys=False,
    )["pm25"]

    for lag in PM25_LAGS:

        column_name = f"pm25_lag_{lag}h"

        df[column_name] = grouped.shift(lag)

    return df


# ============================================================
# PM2.5 ROLLING FEATURES
# ============================================================

def add_pm25_rolling_features(df):
    print("Adding PM2.5 rolling features...")

    # IMPORTANT:
    # Shift first so the current PM2.5 value is NOT included.
    #
    # This means:
    #
    # rolling_24h at time t
    #
    # uses:
    # t-1 ... t-24
    #
    # and NOT pm25(t).

    past_pm25 = (
        df.groupby(
            "station_code",
            group_keys=False,
        )["pm25"]
        .shift(1)
    )

    for window in PM25_ROLLING_WINDOWS:

        grouped_past = past_pm25.groupby(
            df["station_code"]
        )

        df[f"pm25_rolling_mean_{window}h"] = (
            grouped_past
            .rolling(
                window=window,
                min_periods=max(3, window // 2),
            )
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"pm25_rolling_std_{window}h"] = (
            grouped_past
            .rolling(
                window=window,
                min_periods=max(3, window // 2),
            )
            .std()
            .reset_index(level=0, drop=True)
        )

    return df


# ============================================================
# WEATHER FEATURES
# ============================================================

def add_weather_features(df):
    print("Adding meteorological features...")

    for variable in WEATHER_VARIABLES:

        grouped = df.groupby(
            "station_code",
            group_keys=False,
        )[variable]

        # ----------------------------------------------
        # Lagged weather
        # ----------------------------------------------

        for lag in WEATHER_LAGS:

            df[f"{variable}_lag_{lag}h"] = (
                grouped.shift(lag)
            )

        # ----------------------------------------------
        # Rolling weather
        #
        # Shift first to avoid using the current value.
        # ----------------------------------------------

        past_weather = grouped.shift(1)

        for window in WEATHER_ROLLING_WINDOWS:

            rolling = (
                past_weather.groupby(
                    df["station_code"]
                )
                .rolling(
                    window=window,
                    min_periods=max(3, window // 2),
                )
            )

            df[
                f"{variable}_rolling_mean_{window}h"
            ] = (
                rolling
                .mean()
                .reset_index(level=0, drop=True)
            )

    return df


# ============================================================
# WIND FEATURES
# ============================================================

def add_wind_features(df):
    print("Adding wind features...")

    # Wind speed reconstructed from U/V components.
    df["wind_speed"] = np.sqrt(
        df["u10"] ** 2 +
        df["v10"] ** 2
    )

    # Wind direction.
    #
    # atan2 gives direction in radians.
    df["wind_direction"] = (
        np.degrees(
            np.arctan2(
                -df["u10"],
                -df["v10"],
            )
        )
        % 360
    )

    # Cyclical representation of direction.
    df["wind_direction_sin"] = np.sin(
        np.radians(df["wind_direction"])
    )

    df["wind_direction_cos"] = np.cos(
        np.radians(df["wind_direction"])
    )

    return df


# ============================================================
# CO-POLLUTANT FEATURES
# ============================================================

def add_copollutant_features(df):
    print("Adding co-pollutant features...")

    for variable in CO_POLLUTANTS:

        grouped = df.groupby(
            "station_code",
            group_keys=False,
        )[variable]

        # ------------------------------------------------
        # Lagged co-pollutants
        # ------------------------------------------------

        for lag in [1, 3, 6, 12, 24]:

            df[
                f"{variable}_lag_{lag}h"
            ] = grouped.shift(lag)

        # ------------------------------------------------
        # Rolling co-pollutants
        # ------------------------------------------------

        past_variable = grouped.shift(1)

        for window in [6, 12, 24, 48]:

            rolling = (
                past_variable.groupby(
                    df["station_code"]
                )
                .rolling(
                    window=window,
                    min_periods=max(3, window // 2),
                )
            )

            df[
                f"{variable}_rolling_mean_{window}h"
            ] = (
                rolling
                .mean()
                .reset_index(level=0, drop=True)
            )

    return df


# ============================================================
# FORECAST TARGETS
# ============================================================

def add_targets(df):
    print("Adding forecast targets...")

    grouped = df.groupby(
        "station_code",
        group_keys=False,
    )["pm25"]

    for horizon_name, horizon_hours in FORECAST_HORIZONS.items():

        target_column = (
            f"target_pm25_{horizon_name}"
        )

        df[target_column] = grouped.shift(
            -horizon_hours
        )

    return df


# ============================================================
# FEATURE AVAILABILITY FLAGS
# ============================================================

def add_missingness_flags(df):
    print("Adding missingness indicators...")

    # Current PM2.5 missingness
    df["pm25_missing"] = (
        df["pm25"].isna().astype(int)
    )

    # Current co-pollutant missingness
    for variable in CO_POLLUTANTS:

        df[f"{variable}_missing"] = (
            df[variable].isna().astype(int)
        )

    # Current weather missingness
    for variable in WEATHER_VARIABLES:

        df[f"{variable}_missing"] = (
            df[variable].isna().astype(int)
        )

    return df


# ============================================================
# VALIDATE TIME ORDER
# ============================================================

def validate_time_order(df):

    print("\nValidating station time ordering...")

    duplicate_count = df.duplicated(
        subset=["station_code", "timestamp"]
    ).sum()

    print(
        f"Duplicate station timestamps: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate station timestamps detected."
        )

    out_of_order = 0

    for station in df["station_code"].unique():

        station_df = df[
            df["station_code"] == station
        ]

        if not station_df[
            "timestamp"
        ].is_monotonic_increasing:

            out_of_order += 1

    print(
        f"Stations with incorrect time ordering: "
        f"{out_of_order}"
    )

    if out_of_order > 0:
        raise ValueError(
            "At least one station is not chronologically ordered."
        )


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

    missingness = [
        column
        for column in df.columns
        if column.endswith("_missing")
    ]

    targets = [
        column
        for column in df.columns
        if column.startswith("target_pm25_")
    ]

    already_used = set(
        metadata
        + current_observations
        + temporal
        + missingness
        + targets
    )

    engineered = [
        column
        for column in df.columns
        if column not in already_used
    ]

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

    # --------------------------------------------------------
    # Sort BEFORE creating lags/rolling features.
    # --------------------------------------------------------

    df = df.sort_values(
        ["station_code", "timestamp"]
    ).reset_index(drop=True)

    validate_time_order(df)

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = add_temporal_features(df)

    df = add_pm25_lag_features(df)

    df = add_pm25_rolling_features(df)

    df = add_weather_features(df)

    df = add_wind_features(df)

    df = add_copollutant_features(df)

    df = add_missingness_flags(df)

    df = add_targets(df)

    # --------------------------------------------------------
    # Organise
    # --------------------------------------------------------

    df = organise_columns(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("V4 FEATURE DATASET CREATED")
    print("=" * 70)

    print(f"\nOutput:")
    print(OUTPUT_FILE)

    print(f"\nRows: {len(df):,}")

    print(f"Columns: {len(df.columns)}")

    print(
        f"\nMemory estimate: "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

    print("\nTarget availability:")

    for horizon_name in FORECAST_HORIZONS:

        target = f"target_pm25_{horizon_name}"

        available = df[target].notna().sum()

        percentage = (
            available / len(df) * 100
        )

        print(
            f"  {horizon_name:>4}: "
            f"{available:,} "
            f"({percentage:.2f}%)"
        )

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

        percentage = (
            missing / len(df) * 100
        )

        print(
            f"  {variable:25s}: "
            f"{missing:8,} "
            f"({percentage:6.2f}%)"
        )

    print("\nFeature groups:")

    print(
        f"  PM2.5 lags: "
        f"{len(PM25_LAGS)}"
    )

    print(
        f"  PM2.5 rolling windows: "
        f"{len(PM25_ROLLING_WINDOWS)}"
    )

    print(
        f"  Weather variables: "
        f"{len(WEATHER_VARIABLES)}"
    )

    print(
        f"  Co-pollutants: "
        f"{len(CO_POLLUTANTS)}"
    )

    print(
        f"  Forecast horizons: "
        f"{len(FORECAST_HORIZONS)}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()