import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/sensor_218_hourly.csv"
OUTPUT_FILE = "data/processed/engineered_sensor_218.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("LOADING SENSOR 218 PROCESSED DATA")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # Convert PM2.5 to numeric
    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce"
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Create missing-value indicator BEFORE any
    # missing-value handling.
    #
    # 1 = PM2.5 was genuinely missing
    # 0 = PM2.5 was observed
    df["pm25_missing"] = (
        df["pm25"].isna().astype(int)
    )

    print(f"\nRows loaded: {len(df):,}")

    print(
        f"Missing PM2.5 values: "
        f"{df['pm25'].isna().sum():,}"
    )

    print(
        f"Missing PM2.5 percentage: "
        f"{df['pm25'].isna().mean() * 100:.2f}%"
    )

    return df


# ============================================================
# HANDLE MISSING PM2.5 VALUES
# ============================================================

def handle_missing_pm25(df):

    print("\n" + "=" * 80)
    print("HANDLING MISSING PM2.5 VALUES")
    print("=" * 80)

    missing_count = df["pm25"].isna().sum()

    print(
        f"Missing PM2.5 values retained: "
        f"{missing_count:,}"
    )

    print(
        "\nNo interpolation is performed."
    )

    print(
        "Genuine missing PM2.5 observations are "
        "kept as NaN so that XGBoost can handle them."
    )

    print(
        "The pm25_missing feature records whether "
        "the original PM2.5 value was missing."
    )

    return df


# ============================================================
# TIME FEATURES
# ============================================================

def create_time_features(df):

    print("\n" + "=" * 80)
    print("CREATING TIME FEATURES")
    print("=" * 80)

    # Basic calendar features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Weekend indicator
    df["weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Season
    def get_season(month):

        if month in [12, 1, 2]:
            return "winter"

        elif month in [3, 4, 5]:
            return "spring"

        elif month in [6, 7, 8]:
            return "summer"

        else:
            return "autumn"

    df["season"] = df["month"].apply(
        get_season
    )

    # Numeric season code for ML models
    season_mapping = {
        "winter": 0,
        "spring": 1,
        "summer": 2,
        "autumn": 3
    }

    df["season_code"] = df["season"].map(
        season_mapping
    )

    print(
        "Created: hour, day, month, day_of_week, "
        "weekend, season, season_code"
    )

    return df


# ============================================================
# CYCLICAL TIME FEATURES
# ============================================================

def create_cyclical_features(df):

    print("\n" + "=" * 80)
    print("CREATING CYCLICAL TIME FEATURES")
    print("=" * 80)

    # Hour of day
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Day of week
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Month
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    print(
        "Created cyclical encodings for "
        "hour, day-of-week and month."
    )

    return df


# ============================================================
# LAG FEATURES
# ============================================================

def create_lag_features(df):

    print("\n" + "=" * 80)
    print("CREATING LAG FEATURES")
    print("=" * 80)

    # Previous hour
    df["pm25_lag_1h"] = (
        df["pm25"].shift(1)
    )

    # Previous 3 hours
    df["pm25_lag_3h"] = (
        df["pm25"].shift(3)
    )

    # Previous 6 hours
    df["pm25_lag_6h"] = (
        df["pm25"].shift(6)
    )

    # Previous 12 hours
    df["pm25_lag_12h"] = (
        df["pm25"].shift(12)
    )

    # Same hour previous day
    df["pm25_lag_24h"] = (
        df["pm25"].shift(24)
    )

    # Same hour previous 2 days
    df["pm25_lag_48h"] = (
        df["pm25"].shift(48)
    )

    # Same hour previous week
    df["pm25_lag_168h"] = (
        df["pm25"].shift(168)
    )

    print(
        "Created lags: 1h, 3h, 6h, 12h, "
        "24h, 48h and 168h."
    )

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def create_rolling_features(df):

    print("\n" + "=" * 80)
    print("CREATING ROLLING FEATURES")
    print("=" * 80)

    # Use only historical PM2.5 values.
    #
    # shift(1) ensures the current observation
    # is not included in the rolling calculation.
    previous = df["pm25"].shift(1)

    # Previous 6 hours
    df["pm25_rolling_mean_6h"] = (
        previous
        .rolling(
            window=6,
            min_periods=3
        )
        .mean()
    )

    df["pm25_rolling_std_6h"] = (
        previous
        .rolling(
            window=6,
            min_periods=3
        )
        .std()
    )

    # Previous 24 hours
    df["pm25_rolling_mean_24h"] = (
        previous
        .rolling(
            window=24,
            min_periods=12
        )
        .mean()
    )

    df["pm25_rolling_std_24h"] = (
        previous
        .rolling(
            window=24,
            min_periods=12
        )
        .std()
    )

    # Previous 7 days
    df["pm25_rolling_mean_7d"] = (
        previous
        .rolling(
            window=168,
            min_periods=84
        )
        .mean()
    )

    df["pm25_rolling_std_7d"] = (
        previous
        .rolling(
            window=168,
            min_periods=84
        )
        .std()
    )

    print(
        "Created rolling mean/std features "
        "for 6h, 24h and 7d."
    )

    return df


# ============================================================
# FUTURE TARGET
# ============================================================

def create_target(df):

    print("\n" + "=" * 80)
    print("CREATING FORECASTING TARGET")
    print("=" * 80)

    # Predict PM2.5 one hour into the future.
    df["target_pm25"] = (
        df["pm25"].shift(-1)
    )

    print(
        "Target: PM2.5 concentration 1 hour ahead."
    )

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(df):

    print("\n" + "=" * 80)
    print("VALIDATING ENGINEERED FEATURES")
    print("=" * 80)

    print(
        f"\nTotal rows: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print("\nMissing values by column:")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing.to_string()
    )

    print("\nOriginal PM2.5 missing values:")

    print(
        f"pm25_missing = "
        f"{df['pm25_missing'].sum():,}"
    )

    print("\nTarget missing values:")

    print(
        f"target_pm25: "
        f"{df['target_pm25'].isna().sum():,}"
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save_data(df):

    print("\n" + "=" * 80)
    print("SAVING ENGINEERED DATA")
    print("=" * 80)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nFile: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = handle_missing_pm25(df)

    df = create_time_features(df)

    df = create_cyclical_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_target(df)

    validate_features(df)

    save_data(df)

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()