import numpy as np
import pandas as pd


INPUT_FILE = (
    "data/processed/v3/"
    "sensor_218_hourly_weather.csv"
)

OUTPUT_FILE = (
    "data/processed/v3/"
    "engineered_sensor_218_v3.csv"
)


def create_features(df):

    df = df.copy()

    df = df.sort_values(
        "timestamp"
    )

    # -------------------------------------------------
    # Existing PM2.5 temporal features
    # -------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclical time features

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # -------------------------------------------------
    # PM2.5 lag features
    # -------------------------------------------------

    pm25_lags = [
        1,
        3,
        6,
        12,
        24,
        48,
        168,
    ]

    for lag in pm25_lags:

        df[f"pm25_lag_{lag}h"] = (
            df["pm25"].shift(lag)
        )

    # -------------------------------------------------
    # PM2.5 rolling features
    # -------------------------------------------------

    for window in [6, 24, 168]:

        df[f"pm25_rolling_mean_{window}h"] = (
            df["pm25"]
            .rolling(window)
            .mean()
        )

        df[f"pm25_rolling_std_{window}h"] = (
            df["pm25"]
            .rolling(window)
            .std()
        )

    # -------------------------------------------------
    # Weather variables
    # -------------------------------------------------

    weather_variables = [
        "temperature",
        "relative_humidity",
        "surface_pressure",
        "precipitation",
        "wind_speed",
    ]

    # Only use PAST weather.
    weather_lags = [
        1,
        6,
        12,
        24,
    ]

    for variable in weather_variables:

        for lag in weather_lags:

            df[
                f"{variable}_lag_{lag}h"
            ] = df[variable].shift(lag)

    # -------------------------------------------------
    # Weather rolling statistics
    # -------------------------------------------------

    for variable in weather_variables:

        for window in [6, 24]:

            df[
                f"{variable}_rolling_mean_{window}h"
            ] = (
                df[variable]
                .shift(1)
                .rolling(window)
                .mean()
            )

    # -------------------------------------------------
    # Wind direction
    #
    # Direction is circular:
    # 359° and 1° are close together.
    # -------------------------------------------------

    radians = np.deg2rad(
        df["wind_direction"]
    )

    df["wind_direction_sin"] = (
        np.sin(radians)
    )

    df["wind_direction_cos"] = (
        np.cos(radians)
    )

    # Past wind direction only

    df["wind_direction_sin_lag_1h"] = (
        df["wind_direction_sin"].shift(1)
    )

    df["wind_direction_cos_lag_1h"] = (
        df["wind_direction_cos"].shift(1)
    )

    return df


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = create_features(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("V3 dataset created:")
    print(OUTPUT_FILE)

    print("\nShape:")
    print(df.shape)

    print("\nNumber of features:")
    print(len(df.columns))

    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()