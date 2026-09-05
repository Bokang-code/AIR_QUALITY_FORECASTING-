import requests
import pandas as pd


LATITUDE = 30.63
LONGITUDE = 104.07

START_DATE = "2016-02-09"
END_DATE = "2017-02-08"

OUTPUT_FILE = "data/processed/v3/weather_2016_2017.csv"


def download_weather():

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),

        # Keep timestamps in UTC to match OpenAQ.
        "timezone": "GMT",

        # ERA5 provides consistent historical reanalysis.
        "models": "era5",
    }

    print("Downloading historical weather data...")
    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print(f"Period: {START_DATE} to {END_DATE}")

    response = requests.get(
        url,
        params=params,
        timeout=120,
    )

    print("Status:", response.status_code)

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            "No hourly weather data returned."
        )

    hourly = data["hourly"]

    weather = pd.DataFrame(hourly)

    weather = weather.rename(
        columns={
            "time": "timestamp",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "relative_humidity",
            "surface_pressure": "surface_pressure",
            "precipitation": "precipitation",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
        }
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        utc=True,
    )

    weather.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nWeather data saved to:")
    print(OUTPUT_FILE)

    print("\nShape:")
    print(weather.shape)

    print("\nColumns:")
    print(weather.columns.tolist())

    print("\nMissing values:")
    print(weather.isna().sum())

    print("\nFirst rows:")
    print(weather.head())


if __name__ == "__main__":
    download_weather()