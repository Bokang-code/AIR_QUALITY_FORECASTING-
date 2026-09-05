import pandas as pd

INPUT_FILE = "data/processed/v3/engineered_sensor_218_v3.csv"


def main():

    print("=" * 70)
    print("V3 DATA AUDIT")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("\nOriginal dataset:")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # ---------------------------------------------------------
    # 1. Missing PM2.5
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("1. PM2.5 MISSINGNESS")
    print("-" * 70)

    print("Missing PM2.5:", df["pm25"].isna().sum())
    print("Available PM2.5:", df["pm25"].notna().sum())

    # ---------------------------------------------------------
    # 2. Missing values by feature
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("2. MISSING VALUES BY FEATURE")
    print("-" * 70)

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing[missing > 0].to_string())

    # ---------------------------------------------------------
    # 3. Rows with complete features
    # ---------------------------------------------------------

    exclude = [
        "timestamp",
        "pm25",
        "pm25_missing",
    ]

    feature_columns = [
        col for col in df.columns
        if col not in exclude
    ]

    complete_features = df[feature_columns].notna().all(axis=1)

    print("\n" + "-" * 70)
    print("3. COMPLETE FEATURE ROWS")
    print("-" * 70)

    print("Feature columns:", len(feature_columns))
    print("Rows with all features available:", complete_features.sum())
    print("Rows with at least one missing feature:", (~complete_features).sum())

    # ---------------------------------------------------------
    # 4. Which features cause the most row loss?
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. ROWS LOST BY FEATURE")
    print("-" * 70)

    loss = (
        df[feature_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(loss[loss > 0].to_string())

    # ---------------------------------------------------------
    # 5. PM2.5 lag/rolling vs weather
    # ---------------------------------------------------------

    pm25_features = [
        col for col in feature_columns
        if col.startswith("pm25_")
    ]

    weather_features = [
        col for col in feature_columns
        if any(
            col.startswith(prefix)
            for prefix in [
                "temperature",
                "relative_humidity",
                "surface_pressure",
                "precipitation",
                "wind_speed",
                "wind_direction",
            ]
        )
    ]

    print("\n" + "-" * 70)
    print("5. PM2.5 vs WEATHER FEATURES")
    print("-" * 70)

    pm25_complete = df[pm25_features].notna().all(axis=1)
    weather_complete = df[weather_features].notna().all(axis=1)

    print("PM2.5-derived features:", len(pm25_features))
    print("Rows with complete PM2.5 features:", pm25_complete.sum())

    print("\nWeather-derived features:", len(weather_features))
    print("Rows with complete weather features:", weather_complete.sum())

    # ---------------------------------------------------------
    # 6. Time coverage
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("6. TIME COVERAGE")
    print("-" * 70)

    print("First timestamp:", df["timestamp"].min())
    print("Last timestamp:", df["timestamp"].max())

    # ---------------------------------------------------------
    # 7. Continuous PM2.5 periods
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("7. PM2.5 AVAILABILITY")
    print("-" * 70)

    available = df["pm25"].notna()

    print(
        "Rows where PM2.5 is available:",
        available.sum()
    )

    print(
        "Rows where PM2.5 is missing:",
        (~available).sum()
    )

    # ---------------------------------------------------------
    # 8. Final simulated training rows for each horizon
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("8. USABLE ROWS BY FORECAST HORIZON")
    print("-" * 70)

    horizons = {
        "48h": 48,
        "72h": 72,
        "7d": 168,
        "14d": 336,
        "30d": 720,
    }

    for name, horizon in horizons.items():

        temp = df.copy()

        # Future target
        temp["target_pm25"] = temp["pm25"].shift(-horizon)

        # Require target
        target_available = temp["target_pm25"].notna()

        # Require all model features
        features_available = (
            temp[feature_columns]
            .notna()
            .all(axis=1)
        )

        usable = target_available & features_available

        print(
            f"{name:5s}: {usable.sum():5d} usable rows"
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()