import pandas as pd


PM25_FILE = "data/processed/sensor_218_hourly.csv"
WEATHER_FILE = "data/processed/v3/weather_2016_2017.csv"

OUTPUT_FILE = (
    "data/processed/v3/"
    "sensor_218_hourly_weather.csv"
)


def main():

    pm25 = pd.read_csv(PM25_FILE)

    weather = pd.read_csv(WEATHER_FILE)

    pm25["timestamp"] = pd.to_datetime(
        pm25["timestamp"],
        utc=True,
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        utc=True,
    )

    merged = pd.merge(
        pm25,
        weather,
        on="timestamp",
        how="left",
    )

    merged = merged.sort_values(
        "timestamp"
    )

    print("PM2.5 rows:", len(pm25))
    print("Weather rows:", len(weather))
    print("Merged rows:", len(merged))

    print("\nMerged columns:")
    print(merged.columns.tolist())

    print("\nMissing values:")
    print(merged.isna().sum())

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()