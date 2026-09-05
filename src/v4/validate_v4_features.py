from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "chengdu_features_v4.csv"
)


def main():

    print("=" * 70)
    print("VALIDATING V4 FEATURE DATASET")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Station check
    # --------------------------------------------------------

    print("\nStations:")

    print(
        df[
            [
                "station_code",
                "station_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("station_code")
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 3136A 2016 check
    # --------------------------------------------------------

    station_3136_2016 = df[
        (df["station_code"] == "3136A")
        & (df["timestamp"].dt.year == 2016)
    ]

    print(
        "\n3136A rows during 2016:",
        len(station_3136_2016),
    )

    if len(station_3136_2016) == 0:
        print("PASS: 3136A 2016 is excluded.")
    else:
        print("FAIL: 3136A 2016 is still present.")

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "station_code",
            "timestamp",
        ]
    ).sum()

    print(
        f"\nDuplicate station timestamps: {duplicates:,}"
    )

    if duplicates == 0:
        print("PASS: No duplicate station timestamps.")
    else:
        print("FAIL: Duplicate station timestamps detected.")

    # --------------------------------------------------------
    # Target check
    # --------------------------------------------------------

    target_columns = [
        "target_pm25_48h",
        "target_pm25_72h",
        "target_pm25_7d",
        "target_pm25_14d",
        "target_pm25_30d",
    ]

    print("\nTarget columns:")

    for column in target_columns:

        exists = column in df.columns

        print(
            f"  {column}: "
            f"{'PASS' if exists else 'MISSING'}"
        )

    # --------------------------------------------------------
    # Feature check
    # --------------------------------------------------------

    required_features = [
        "pm25_lag_1h",
        "pm25_lag_24h",
        "pm25_lag_48h",
        "pm25_lag_168h",
        "pm25_rolling_mean_24h",
        "pm25_rolling_std_24h",
        "temperature_lag_1h",
        "temperature_rolling_mean_24h",
        "pm10_lag_1h",
        "no2_lag_24h",
        "wind_speed",
        "wind_direction",
        "wind_direction_sin",
        "wind_direction_cos",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    print("\nRequired feature columns:")

    missing_features = []

    for column in required_features:

        if column in df.columns:
            print(f"  PASS: {column}")
        else:
            print(f"  FAIL: {column}")
            missing_features.append(column)

    # --------------------------------------------------------
    # Check first observations of each station
    # --------------------------------------------------------

    print("\nFirst observations by station:")

    for station in sorted(
        df["station_code"].unique()
    ):

        station_df = (
            df[df["station_code"] == station]
            .sort_values("timestamp")
            .head(3)
        )

        print(
            f"\n{station}:"
        )

        print(
            station_df[
                [
                    "timestamp",
                    "pm25",
                    "pm25_lag_1h",
                    "pm25_lag_24h",
                    "target_pm25_48h",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Check target alignment manually
    # --------------------------------------------------------

    print("\nChecking target alignment...")

    test_station = (
        df["station_code"]
        .drop_duplicates()
        .iloc[0]
    )

    station_df = (
        df[
            df["station_code"] == test_station
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Find a row where both current and 48h target exist.
    valid = station_df[
        station_df["pm25"].notna()
        & station_df["target_pm25_48h"].notna()
    ]

    if len(valid) > 0:

        row_index = valid.index[0]

        current_row = station_df.iloc[row_index]

        future_index = row_index + 48

        if future_index < len(station_df):

            future_row = station_df.iloc[future_index]

            print(
                "\nExample target alignment:"
            )

            print(
                f"Station: "
                f"{test_station}"
            )

            print(
                f"Current timestamp: "
                f"{current_row['timestamp']}"
            )

            print(
                f"Current PM2.5: "
                f"{current_row['pm25']}"
            )

            print(
                f"Target timestamp: "
                f"{future_row['timestamp']}"
            )

            print(
                f"Target PM2.5 from dataset: "
                f"{current_row['target_pm25_48h']}"
            )

            print(
                f"Actual PM2.5 at +48h: "
                f"{future_row['pm25']}"
            )

            if (
                current_row["target_pm25_48h"]
                == future_row["pm25"]
            ):
                print(
                    "PASS: 48-hour target alignment is correct."
                )
            else:
                print(
                    "WARNING: 48-hour target does not match."
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    if missing_features:
        print(
            "\nMissing required features:"
        )

        for column in missing_features:
            print(f"  - {column}")

    else:
        print(
            "\nAll required feature columns are present."
        )


if __name__ == "__main__":
    main()