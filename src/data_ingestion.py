import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

SENSOR_ID = 218

START = "2016-02-09T04:00:00Z"
END = "2017-02-08T03:00:00Z"

BASE_URL = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements"

LIMIT = 1000
MAX_PAGES = 15

OUTPUT_FILE = "data/raw/sensor_218_raw.csv"


# =============================================================================
# API HEADERS
# =============================================================================

HEADERS = {
    "Accept": "application/json",
    "X-API-Key": API_KEY,
}


# =============================================================================
# FETCH DATA
# =============================================================================

def fetch_sensor_data():
    """
    Fetch PM2.5 measurements for Sensor 218 from OpenAQ.

    Returns
    -------
    list
        Raw OpenAQ measurement records.
    """

    print("=" * 80)
    print("OPENAQ SENSOR 218 DATA INGESTION")
    print("=" * 80)

    if not API_KEY:
        raise ValueError(
            "OPENAQ_API_KEY was not found. "
            "Make sure it is defined in the .env file."
        )

    all_rows = []

    for page in range(1, MAX_PAGES + 1):

        params = {
            "datetime_from": START,
            "datetime_to": END,
            "limit": LIMIT,
            "page": page,
        }

        response = None

        for attempt in range(3):

            try:
                response = requests.get(
                    BASE_URL,
                    headers=HEADERS,
                    params=params,
                    timeout=60,
                )

                if response.status_code == 200:
                    break

                print(
                    f"Page {page}: HTTP {response.status_code} "
                    f"(attempt {attempt + 1}/3)"
                )

                time.sleep(3)

            except requests.RequestException as error:

                print(
                    f"Page {page}: request error "
                    f"(attempt {attempt + 1}/3): {error}"
                )

                time.sleep(3)

        else:
            print(f"Could not retrieve page {page}.")
            break

        data = response.json()
        rows = data.get("results", [])

        print(
            f"Page {page}: {len(rows)} rows "
            f"(total {len(all_rows) + len(rows)})"
        )

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < LIMIT:
            break

        time.sleep(0.5)

    return all_rows


# =============================================================================
# PREPARE DATA
# =============================================================================

def prepare_data(rows):
    """
    Convert raw OpenAQ responses into a simple timestamp/PM2.5 dataframe.
    """

    records = []

    for row in rows:

        timestamp = (
            row.get("period", {})
            .get("datetimeFrom", {})
            .get("utc")
        )

        value = row.get("value")

        records.append(
            {
                "timestamp": timestamp,
                "pm25": value,
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce",
    )

    df = df.sort_values("timestamp")

    return df


# =============================================================================
# CLEAN DATA
# =============================================================================

def clean_data(df):
    """
    Clean raw Sensor 218 data while preserving missing PM2.5 values.
    """

    print("\n" + "=" * 80)
    print("CLEANING SENSOR 218 DATA")
    print("=" * 80)

    if df.empty:
        return df

    print(f"\nRows before cleaning: {len(df):,}")

    # -------------------------------------------------------------------------
    # Remove invalid timestamps
    # -------------------------------------------------------------------------

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:

        print(
            f"Removing {invalid_timestamps:,} rows "
            "with invalid timestamps."
        )

        df = df.dropna(subset=["timestamp"])

    # -------------------------------------------------------------------------
    # Remove duplicate timestamps
    # -------------------------------------------------------------------------

    duplicates = df["timestamp"].duplicated().sum()

    print(f"Duplicate timestamps: {duplicates:,}")

    if duplicates > 0:

        df = df.drop_duplicates(
            subset="timestamp",
            keep="first",
        )

    # -------------------------------------------------------------------------
    # Restrict to project date range
    # -------------------------------------------------------------------------

    start = pd.Timestamp(START)
    end = pd.Timestamp(END)

    df = df[
        (df["timestamp"] >= start)
        & (df["timestamp"] < end)
    ].copy()

    # -------------------------------------------------------------------------
    # Sort chronologically
    # -------------------------------------------------------------------------

    df = df.sort_values("timestamp").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Keep missing PM2.5 values as NaN
    # -------------------------------------------------------------------------

    missing = df["pm25"].isna().sum()

    print(f"Missing PM2.5 values: {missing:,}")

    print(f"\nRows after cleaning: {len(df):,}")

    return df


# =============================================================================
# SAVE DATA
# =============================================================================

def save_data(df, output_file=OUTPUT_FILE):
    """
    Save cleaned raw data to CSV.
    """

    directory = os.path.dirname(output_file)

    if directory:
        os.makedirs(directory, exist_ok=True)

    df.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 80)
    print("DATA SAVED")
    print("=" * 80)

    print(f"\nFile: {output_file}")
    print(f"Rows: {len(df):,}")

    if not df.empty:

        print(f"Start: {df['timestamp'].min()}")
        print(f"End:   {df['timestamp'].max()}")

    print("\nColumns:")
    print(df.columns.tolist())


# =============================================================================
# COMPLETE INGESTION PIPELINE
# =============================================================================

def run_ingestion(output_file=OUTPUT_FILE):
    """
    Run the complete data ingestion process.

    Returns
    -------
    pandas.DataFrame
        Cleaned Sensor 218 dataframe.
    """

    rows = fetch_sensor_data()

    if not rows:
        raise RuntimeError("No data was retrieved from OpenAQ.")

    df = prepare_data(rows)

    df = clean_data(df)

    save_data(df, output_file)

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    run_ingestion()


if __name__ == "__main__":
    main()