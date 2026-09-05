import os
import requests
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_FILE = "data/processed/v3/nearby_pm25_sensors.csv"
OUTPUT_FILE = "data/processed/v3/nearby_pm25_sensor_date_ranges.csv"

STUDY_START = pd.Timestamp("2016-02-09", tz="UTC")
STUDY_END = pd.Timestamp("2017-02-08 23:59:59", tz="UTC")

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAQ_API_KEY was not found in your .env file."
    )


# =============================================================================
# GET SENSOR METADATA
# =============================================================================

def get_sensor_metadata(sensor_id):

    url = f"https://api.openaq.org/v3/sensors/{sensor_id}"

    headers = {
        "X-API-Key": API_KEY,
        "Accept": "application/json",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60,
    )

    print(f"    HTTP status: {response.status_code}")

    if response.status_code != 200:
        print(f"    API response: {response.text[:500]}")
        return None

    data = response.json()

    results = data.get("results", [])

    if not results:
        print("    No sensor metadata returned.")
        return None

    return results[0]


# =============================================================================
# CHECK WHETHER DATE RANGE OVERLAPS STUDY PERIOD
# =============================================================================

def check_overlap(first_date, last_date):

    if pd.isna(first_date) or pd.isna(last_date):
        return False

    return (
        first_date <= STUDY_END
        and last_date >= STUDY_START
    )


# =============================================================================
# CHECK ONE SENSOR
# =============================================================================

def check_sensor(sensor):

    sensor_id = int(sensor["sensor_id"])

    print()
    print("-" * 70)
    print(
        f"Checking sensor {sensor_id} "
        f"({sensor['location_name']})"
    )
    print(
        f"Distance: {sensor['distance_km']} km"
    )
    print("-" * 70)

    metadata = get_sensor_metadata(sensor_id)

    if metadata is None:

        return {
            **sensor,
            "first_available": pd.NaT,
            "last_available": pd.NaT,
            "observed_count": None,
            "percent_coverage": None,
            "overlaps_study_period": False,
        }

    # OpenAQ uses datetimeFirst / datetimeLast

    first_raw = metadata.get("datetimeFirst")
    last_raw = metadata.get("datetimeLast")

    first_date = pd.NaT
    last_date = pd.NaT

    if isinstance(first_raw, dict):
        first_date = pd.to_datetime(
            first_raw.get("utc"),
            utc=True,
            errors="coerce",
        )

    if isinstance(last_raw, dict):
        last_date = pd.to_datetime(
            last_raw.get("utc"),
            utc=True,
            errors="coerce",
        )

    # Coverage information

    coverage = metadata.get("coverage", {})

    observed_count = coverage.get(
        "observedCount"
    )

    percent_coverage = coverage.get(
        "percentCoverage"
    )

    overlaps = check_overlap(
        first_date,
        last_date,
    )

    print(
        f"    First available: {first_date}"
    )

    print(
        f"    Last available:  {last_date}"
    )

    print(
        f"    Observed count:  {observed_count}"
    )

    print(
        f"    Coverage:        {percent_coverage}%"
    )

    if overlaps:
        print(
            "    *** OVERLAPS 2016-2017 STUDY PERIOD ***"
        )
    else:
        print(
            "    No overlap with 2016-2017 study period."
        )

    return {
        **sensor,
        "first_available": first_date,
        "last_available": last_date,
        "observed_count": observed_count,
        "percent_coverage": percent_coverage,
        "overlaps_study_period": overlaps,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("CHECKING ACTUAL SENSOR DATE RANGES")
    print("=" * 70)

    print()
    print(
        f"Study period: "
        f"{STUDY_START} -> {STUDY_END}"
    )

    sensors = pd.read_csv(
        INPUT_FILE
    )

    # Exclude our primary sensor 218

    nearby_sensors = sensors[
        sensors["sensor_id"] != 218
    ].copy()

    print()
    print(
        f"Nearby sensors to check: "
        f"{len(nearby_sensors)}"
    )

    results = []

    for _, sensor in nearby_sensors.iterrows():

        result = check_sensor(sensor)

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    # Sort by distance

    results_df = (
        results_df
        .sort_values("distance_km")
        .reset_index(drop=True)
    )

    # Save results

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # Display summary

    print()
    print("=" * 70)
    print("SENSOR DATE RANGE SUMMARY")
    print("=" * 70)

    display_columns = [
        "sensor_id",
        "location_name",
        "distance_km",
        "first_available",
        "last_available",
        "observed_count",
        "percent_coverage",
        "overlaps_study_period",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # Show sensors that overlap

    overlapping = results_df[
        results_df["overlaps_study_period"]
    ]

    print()
    print("=" * 70)
    print("SENSORS OVERLAPPING 2016-2017")
    print("=" * 70)

    if overlapping.empty:

        print(
            "No nearby PM2.5 sensors overlap "
            "the 2016-02-09 to 2017-02-08 study period."
        )

    else:

        print(
            overlapping[
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
    print("DATE RANGE CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()