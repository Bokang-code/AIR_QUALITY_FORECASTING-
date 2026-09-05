import os
import requests
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_FILE = (
    "data/processed/v3/"
    "historical_pollution_sensors.csv"
)

OUTPUT_FILE = (
    "data/processed/v3/"
    "historical_pollution_sensor_ranges.csv"
)

STUDY_START = pd.Timestamp(
    "2016-02-09",
    tz="UTC"
)

STUDY_END = pd.Timestamp(
    "2017-02-08 23:59:59",
    tz="UTC"
)


# =============================================================================
# LOAD API KEY
# =============================================================================

load_dotenv()

API_KEY = os.getenv(
    "OPENAQ_API_KEY"
)

if not API_KEY:
    raise ValueError(
        "OPENAQ_API_KEY was not found in your .env file."
    )


HEADERS = {
    "X-API-Key": API_KEY,
    "Accept": "application/json",
}


# =============================================================================
# GET SENSOR METADATA
# =============================================================================

def get_sensor_metadata(sensor_id):

    url = (
        f"https://api.openaq.org/v3/"
        f"sensors/{sensor_id}"
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
        )

    except requests.RequestException as e:

        print(
            f"    Request error: {e}"
        )

        return None

    if response.status_code != 200:

        print(
            f"    HTTP {response.status_code}"
        )

        return None

    results = response.json().get(
        "results",
        []
    )

    if not results:
        return None

    return results[0]


# =============================================================================
# CHECK OVERLAP
# =============================================================================

def check_overlap(
    first_date,
    last_date
):

    if pd.isna(first_date):
        return False

    if pd.isna(last_date):
        return False

    return (
        first_date <= STUDY_END
        and last_date >= STUDY_START
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print(
        "CHECKING HISTORICAL POLLUTION "
        "SENSOR DATE RANGES"
    )
    print("=" * 70)

    sensors = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Pollution sensors to check: "
        f"{len(sensors)}"
    )

    results = []

    for index, sensor in sensors.iterrows():

        sensor_id = int(
            sensor["sensor_id"]
        )

        pollutant = sensor[
            "pollutant"
        ]

        print()
        print(
            f"[{index + 1}/{len(sensors)}] "
            f"Sensor {sensor_id} "
            f"({pollutant})"
        )

        metadata = get_sensor_metadata(
            sensor_id
        )

        if metadata is None:

            print(
                "    No metadata returned."
            )

            results.append({
                **sensor.to_dict(),
                "datetime_first": pd.NaT,
                "datetime_last": pd.NaT,
                "observed_count": None,
                "overlaps_study_period": False,
            })

            continue

        # -------------------------------------------------------------
        # Extract actual OpenAQ datetimeFirst / datetimeLast
        # -------------------------------------------------------------

        first = metadata.get(
            "datetimeFirst"
        )

        last = metadata.get(
            "datetimeLast"
        )

        first_date = pd.NaT
        last_date = pd.NaT

        if isinstance(first, dict):

            first_date = pd.to_datetime(
                first.get("utc"),
                utc=True,
                errors="coerce",
            )

        if isinstance(last, dict):

            last_date = pd.to_datetime(
                last.get("utc"),
                utc=True,
                errors="coerce",
            )

        # -------------------------------------------------------------
        # Coverage
        # -------------------------------------------------------------

        coverage = metadata.get(
            "coverage",
            {}
        )

        observed_count = coverage.get(
            "observedCount"
        )

        overlap = check_overlap(
            first_date,
            last_date
        )

        print(
            f"    First: {first_date}"
        )

        print(
            f"    Last:  {last_date}"
        )

        print(
            f"    Observed: {observed_count}"
        )

        if overlap:

            print(
                "    *** OVERLAPS 2016-2017 ***"
            )

        else:

            print(
                "    No overlap"
            )

        results.append({
            **sensor.to_dict(),
            "datetime_first": first_date,
            "datetime_last": last_date,
            "observed_count": observed_count,
            "overlaps_study_period": overlap,
        })

    # =============================================================================
    # CREATE DATAFRAME
    # =============================================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        [
            "pollutant",
            "distance_km",
        ]
    ).reset_index(
        drop=True
    )

    # Save

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =============================================================================
    # DISPLAY
    # =============================================================================

    print()
    print("=" * 70)
    print(
        "POLLUTION SENSOR DATE RANGES"
    )
    print("=" * 70)

    display_columns = [
        "sensor_id",
        "location_name",
        "pollutant",
        "distance_km",
        "datetime_first",
        "datetime_last",
        "observed_count",
        "overlaps_study_period",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # =============================================================================
    # OVERLAPPING SENSORS
    # =============================================================================

    overlapping = results_df[
        results_df[
            "overlaps_study_period"
        ]
    ]

    print()
    print("=" * 70)
    print(
        "SENSORS WITH 2016-2017 OVERLAP"
    )
    print("=" * 70)

    if overlapping.empty:

        print(
            "NONE"
        )

    else:

        print(
            overlapping[
                display_columns
            ].to_string(
                index=False
            )
        )

    # =============================================================================
    # SUMMARY BY POLLUTANT
    # =============================================================================

    print()
    print("=" * 70)
    print(
        "OVERLAP SUMMARY"
    )
    print("=" * 70)

    summary = (
        results_df
        .groupby("pollutant")
        ["overlaps_study_period"]
        .agg(
            sensors="count",
            overlapping="sum",
        )
    )

    print(
        summary.to_string()
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print(
        "CHECK COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()