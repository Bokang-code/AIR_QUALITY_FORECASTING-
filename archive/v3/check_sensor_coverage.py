import os
import requests
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_FILE = "data/processed/v3/nearby_pm25_sensors.csv"
OUTPUT_FILE = "data/processed/v3/nearby_pm25_sensor_coverage.csv"

START_DATE = "2016-02-09T00:00:00Z"
END_DATE = "2017-02-08T23:59:59Z"

EXPECTED_HOURS = 8760

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAQ_API_KEY was not found in your .env file."
    )


# =============================================================================
# GET SENSOR MEASUREMENTS
# =============================================================================

def get_sensor_measurements(sensor_id):

    url = (
        f"https://api.openaq.org/v3/sensors/"
        f"{sensor_id}/measurements"
    )

    headers = {
        "X-API-Key": API_KEY,
        "Accept": "application/json",
    }

    all_results = []

    page = 1
    limit = 1000

    while True:

        params = {
            "datetime_from": START_DATE,
            "datetime_to": END_DATE,
            "limit": limit,
            "page": page,
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=60,
        )

        print(
            f"    HTTP status: {response.status_code}"
        )

        if response.status_code != 200:

            print(
                f"    API response: "
                f"{response.text[:500]}"
            )

            return []

        data = response.json()

        results = data.get(
            "results",
            []
        )

        if not results:
            break

        all_results.extend(results)

        print(
            f"    Page {page}: "
            f"{len(results)} measurements"
        )

        if len(results) < limit:
            break

        page += 1

    return all_results


# =============================================================================
# EXTRACT TIMESTAMP
# =============================================================================

def extract_timestamp(measurement):

    period = measurement.get(
        "period"
    )

    if not isinstance(
        period,
        dict
    ):
        return None

    datetime_from = period.get(
        "datetimeFrom"
    )

    if not isinstance(
        datetime_from,
        dict
    ):
        return None

    utc_time = datetime_from.get(
        "utc"
    )

    return utc_time


# =============================================================================
# ANALYSE SENSOR
# =============================================================================

def analyse_sensor(sensor):

    sensor_id = int(
        sensor["sensor_id"]
    )

    print()
    print("-" * 70)

    print(
        f"Checking sensor {sensor_id} "
        f"({sensor['location_name']})"
    )

    print("-" * 70)

    measurements = get_sensor_measurements(
        sensor_id
    )

    if not measurements:

        print(
            "    No measurements returned."
        )

        return {
            **sensor,
            "measurement_count": 0,
            "unique_hours": 0,
            "coverage_percent": 0.0,
            "first_measurement": None,
            "last_measurement": None,
        }

    timestamps = []

    for measurement in measurements:

        timestamp = extract_timestamp(
            measurement
        )

        if timestamp:
            timestamps.append(
                timestamp
            )

    if not timestamps:

        print(
            "    WARNING: Could not extract timestamps."
        )

        print()
        print(
            "    Example measurement:"
        )

        print(
            measurements[0]
        )

        return {
            **sensor,
            "measurement_count": len(
                measurements
            ),
            "unique_hours": 0,
            "coverage_percent": 0.0,
            "first_measurement": None,
            "last_measurement": None,
        }

    timestamps = pd.to_datetime(
        timestamps,
        utc=True,
        errors="coerce",
    )

    timestamps = timestamps[
        ~timestamps.isna()
    ]

    hourly_timestamps = (
        timestamps
        .floor("h")
        .drop_duplicates()
    )

    unique_hours = len(
        hourly_timestamps
    )

    coverage_percent = (
        unique_hours
        / EXPECTED_HOURS
        * 100
    )

    print(
        f"    Unique hours: "
        f"{unique_hours:,}"
    )

    print(
        f"    Coverage: "
        f"{coverage_percent:.2f}%"
    )

    print(
        f"    First: "
        f"{timestamps.min()}"
    )

    print(
        f"    Last: "
        f"{timestamps.max()}"
    )

    return {
        **sensor,
        "measurement_count": len(
            measurements
        ),
        "unique_hours": unique_hours,
        "coverage_percent": round(
            coverage_percent,
            2,
        ),
        "first_measurement": timestamps.min(),
        "last_measurement": timestamps.max(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("CHECKING HISTORICAL PM2.5 SENSOR COVERAGE")
    print("=" * 70)

    print()
    print(
        f"Period: "
        f"{START_DATE} → {END_DATE}"
    )

    print(
        f"Expected hourly observations: "
        f"{EXPECTED_HOURS:,}"
    )

    sensors = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Sensors to check: "
        f"{len(sensors)}"
    )

    results = []

    for _, sensor in sensors.iterrows():

        result = analyse_sensor(
            sensor
        )

        results.append(
            result
        )

    coverage = pd.DataFrame(
        results
    )

    coverage = (
        coverage
        .sort_values(
            [
                "coverage_percent",
                "distance_km",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    coverage.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("SENSOR COVERAGE RESULTS")
    print("=" * 70)

    display_columns = [
        "sensor_id",
        "location_name",
        "distance_km",
        "measurement_count",
        "unique_hours",
        "coverage_percent",
        "first_measurement",
        "last_measurement",
    ]

    print(
        coverage[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("COVERAGE CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()