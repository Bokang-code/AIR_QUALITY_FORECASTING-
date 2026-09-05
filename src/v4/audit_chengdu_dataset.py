from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/chengdu")
REPORT_DIR = Path("reports/v4")

STATIONS_FILE = DATA_DIR / "stations.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# FULL DATASET — NO DATE RESTRICTION
START_DATE = None
END_DATE = None


POLLUTANT_COLUMNS = [
    "AQI",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "O3-8h",
    "CO",
]

WEATHER_COLUMNS = [
    "U10",
    "V10",
    "T2",
    "RH2",
    "PBLH",
]

ALL_ANALYSIS_COLUMNS = POLLUTANT_COLUMNS + WEATHER_COLUMNS


# Station code → station name
STATION_NAMES = {
    "1431A": "JQLH",
    "1432A": "SLD",
    "1433A": "SWY",
    "1434A": "SHP",
    "1437A": "JPJ",
    "1438A": "LYS",
    "2880A": "DSXL",
    "3136A": "LQXQ",
    "3358A": "LJL",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates in kilometres.
    """

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


def load_station_coordinates():

    with open(
        STATIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        stations = json.load(f)

    return stations


def build_timestamp(df):

    timestamp = pd.to_datetime(
        {
            "year": pd.to_numeric(
                df["year"],
                errors="coerce"
            ),
            "month": pd.to_numeric(
                df["month"],
                errors="coerce"
            ),
            "day": pd.to_numeric(
                df["day"],
                errors="coerce"
            ),
            "hour": pd.to_numeric(
                df["hour"],
                errors="coerce"
            ),
        },
        errors="coerce",
    )

    return timestamp


# ============================================================
# AUDIT ONE STATION
# ============================================================

def audit_station(file_path, station_code):

    print("\n" + "=" * 70)
    print(
        f"STATION: {station_code} "
        f"({STATION_NAMES.get(station_code, 'Unknown')})"
    )
    print(
        f"FILE: {file_path}"
    )
    print("=" * 70)

    df = pd.read_excel(file_path)

    original_rows = len(df)

    print(
        f"Original rows: {original_rows:,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # --------------------------------------------------------
    # Build timestamp
    # --------------------------------------------------------

    df["timestamp"] = build_timestamp(df)

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:

        print(
            f"Invalid timestamps: "
            f"{invalid_timestamps:,}"
        )

    df = df.dropna(
        subset=["timestamp"]
    ).copy()

    # --------------------------------------------------------
    # Optional date filtering
    # --------------------------------------------------------

    if START_DATE is not None:

        df = df[
            df["timestamp"] >= pd.Timestamp(START_DATE)
        ]

    if END_DATE is not None:

        df = df[
            df["timestamp"] <= pd.Timestamp(END_DATE)
        ]

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(df) == 0:

        print(
            "WARNING: No rows remain."
        )

        return None, None, None, None

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    first_timestamp = df["timestamp"].min()
    last_timestamp = df["timestamp"].max()

    expected_hours = (
        int(
            (
                last_timestamp
                - first_timestamp
            ).total_seconds()
            / 3600
        )
        + 1
    )

    actual_rows = len(df)

    coverage_pct = (
        actual_rows
        / expected_hours
        * 100
        if expected_hours > 0
        else np.nan
    )

    duplicate_timestamps = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        f"Earliest timestamp: {first_timestamp}"
    )

    print(
        f"Latest timestamp:   {last_timestamp}"
    )

    print(
        f"Rows:                {actual_rows:,}"
    )

    print(
        f"Expected hourly rows: {expected_hours:,}"
    )

    print(
        f"Timestamp coverage:  {coverage_pct:.2f}%"
    )

    print(
        f"Duplicate timestamps: "
        f"{duplicate_timestamps:,}"
    )

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    df["year_from_timestamp"] = (
        df["timestamp"].dt.year
    )

    year_counts = (
        df.groupby("year_from_timestamp")
        .size()
        .sort_index()
    )

    print("\nRows by year:")

    for year, count in year_counts.items():

        print(
            f"  {year}: {count:,}"
        )

    # ========================================================
    # OVERALL MISSINGNESS
    # ========================================================

    missing_records = []

    for column in ALL_ANALYSIS_COLUMNS:

        if column not in df.columns:

            missing_count = np.nan
            missing_pct = np.nan

        else:

            missing_count = (
                df[column]
                .isna()
                .sum()
            )

            missing_pct = (
                missing_count
                / len(df)
                * 100
            )

        missing_records.append(
            {
                "station_code": station_code,
                "station_name": STATION_NAMES.get(
                    station_code,
                    "Unknown"
                ),
                "variable": column,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
            }
        )

    missing_df = pd.DataFrame(
        missing_records
    )

    print("\nOverall missingness:")

    for _, row in missing_df.iterrows():

        if pd.isna(row["missing_pct"]):

            print(
                f"  {row['variable']}: "
                f"COLUMN NOT FOUND"
            )

        else:

            print(
                f"  {row['variable']}: "
                f"{int(row['missing_count']):,} missing "
                f"({row['missing_pct']:.2f}%)"
            )

    # ========================================================
    # MISSINGNESS BY YEAR
    # ========================================================

    yearly_missing_records = []

    print("\nMissingness by year:")

    for year, year_df in (
        df.groupby("year_from_timestamp")
    ):

        print(
            f"\n  YEAR {year}"
        )

        print(
            "  " + "-" * 50
        )

        for column in ALL_ANALYSIS_COLUMNS:

            if column not in year_df.columns:

                missing_count = np.nan
                missing_pct = np.nan

            else:

                missing_count = (
                    year_df[column]
                    .isna()
                    .sum()
                )

                missing_pct = (
                    missing_count
                    / len(year_df)
                    * 100
                )

            yearly_missing_records.append(
                {
                    "station_code": station_code,
                    "station_name": STATION_NAMES.get(
                        station_code,
                        "Unknown"
                    ),
                    "year": year,
                    "rows": len(year_df),
                    "variable": column,
                    "missing_count": missing_count,
                    "missing_pct": missing_pct,
                }
            )

            if pd.isna(missing_pct):

                print(
                    f"    {column}: "
                    f"COLUMN NOT FOUND"
                )

            else:

                print(
                    f"    {column}: "
                    f"{int(missing_count):,} "
                    f"({missing_pct:.2f}%)"
                )

    yearly_missing_df = pd.DataFrame(
        yearly_missing_records
    )

    # ========================================================
    # PM2.5 SUMMARY
    # ========================================================

    pm25_summary = None

    if "PM2.5" in df.columns:

        pm25 = pd.to_numeric(
            df["PM2.5"],
            errors="coerce"
        ).dropna()

        if len(pm25) > 0:

            q1 = pm25.quantile(0.25)
            q3 = pm25.quantile(0.75)

            iqr = q3 - q1

            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )

            iqr_outliers = (
                (pm25 < lower_bound)
                | (pm25 > upper_bound)
            ).sum()

            pm25_summary = {
                "station_code": station_code,
                "station_name": STATION_NAMES.get(
                    station_code,
                    "Unknown"
                ),
                "count": len(pm25),
                "mean": pm25.mean(),
                "median": pm25.median(),
                "std": pm25.std(),
                "min": pm25.min(),
                "q1": q1,
                "q3": q3,
                "max": pm25.max(),
                "iqr": iqr,
                "lower_iqr_bound": lower_bound,
                "upper_iqr_bound": upper_bound,
                "iqr_outliers": iqr_outliers,
            }

            print("\nPM2.5 summary:")

            print(
                f"  Valid observations: "
                f"{len(pm25):,}"
            )

            print(
                f"  Mean:               "
                f"{pm25.mean():.2f}"
            )

            print(
                f"  Median:             "
                f"{pm25.median():.2f}"
            )

            print(
                f"  Std:                "
                f"{pm25.std():.2f}"
            )

            print(
                f"  Min:                "
                f"{pm25.min():.2f}"
            )

            print(
                f"  Q1:                 "
                f"{q1:.2f}"
            )

            print(
                f"  Q3:                 "
                f"{q3:.2f}"
            )

            print(
                f"  Max:                "
                f"{pm25.max():.2f}"
            )

            print(
                f"  IQR outliers:       "
                f"{iqr_outliers:,}"
            )

    # ========================================================
    # STATION SUMMARY
    # ========================================================

    station_summary = {
        "station_code": station_code,
        "station_name": STATION_NAMES.get(
            station_code,
            "Unknown"
        ),
        "original_rows": original_rows,
        "rows_after_filter": actual_rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "expected_hourly_rows": expected_hours,
        "timestamp_coverage_pct": coverage_pct,
        "duplicate_timestamps": duplicate_timestamps,
        "invalid_timestamps": invalid_timestamps,
    }

    return (
        station_summary,
        missing_df,
        yearly_missing_df,
        pm25_summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CHENGDU DATASET FULL-PERIOD AUDIT"
    )
    print("=" * 70)

    print(
        f"Data directory: {DATA_DIR}"
    )

    print(
        f"Report directory: {REPORT_DIR}"
    )

    print(
        "Date restriction: NONE — "
        "auditing FULL available period"
    )

    # --------------------------------------------------------
    # Find Excel files
    # --------------------------------------------------------

    excel_files = sorted(
        DATA_DIR.glob("*.xlsx")
    )

    if not excel_files:

        print(
            "\nERROR: No Excel files found."
        )

        print(
            f"Expected files inside: "
            f"{DATA_DIR}"
        )

        return

    print(
        f"\nExcel files found: "
        f"{len(excel_files)}"
    )

    for file in excel_files:

        print(
            f"  - {file.name}"
        )

    # --------------------------------------------------------
    # Load coordinates
    # --------------------------------------------------------

    try:

        station_coordinates = (
            load_station_coordinates()
        )

    except Exception as e:

        print(
            f"\nERROR loading stations.json: {e}"
        )

        return

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    station_results = []
    all_missingness = []
    all_yearly_missingness = []
    all_pm25_summaries = []

    # --------------------------------------------------------
    # Audit stations
    # --------------------------------------------------------

    for file_path in excel_files:

        station_code = file_path.stem

        try:

            (
                station_summary,
                missing_df,
                yearly_missing_df,
                pm25_summary,
            ) = audit_station(
                file_path,
                station_code,
            )

            if station_summary is not None:

                station_results.append(
                    station_summary
                )

            if missing_df is not None:

                all_missingness.append(
                    missing_df
                )

            if yearly_missing_df is not None:

                all_yearly_missingness.append(
                    yearly_missing_df
                )

            if pm25_summary is not None:

                all_pm25_summaries.append(
                    pm25_summary
                )

        except Exception as e:

            print(
                "\nERROR processing station:"
            )

            print(
                f"  {file_path}"
            )

            print(
                f"  {e}"
            )

    # ========================================================
    # SAVE STATION SUMMARY
    # ========================================================

    if station_results:

        station_summary_df = pd.DataFrame(
            station_results
        )

        path = (
            REPORT_DIR
            / "station_summary.csv"
        )

        station_summary_df.to_csv(
            path,
            index=False
        )

        print(
            f"\nSaved: {path}"
        )

    # ========================================================
    # SAVE OVERALL MISSINGNESS
    # ========================================================

    if all_missingness:

        missingness_df = pd.concat(
            all_missingness,
            ignore_index=True
        )

        path = (
            REPORT_DIR
            / "missingness.csv"
        )

        missingness_df.to_csv(
            path,
            index=False
        )

        print(
            f"Saved: {path}"
        )

    # ========================================================
    # SAVE YEARLY MISSINGNESS
    # ========================================================

    if all_yearly_missingness:

        yearly_missingness_df = pd.concat(
            all_yearly_missingness,
            ignore_index=True
        )

        path = (
            REPORT_DIR
            / "missingness_by_year.csv"
        )

        yearly_missingness_df.to_csv(
            path,
            index=False
        )

        print(
            f"Saved: {path}"
        )

    # ========================================================
    # SAVE PM2.5 SUMMARY
    # ========================================================

    if all_pm25_summaries:

        pm25_summary_df = pd.DataFrame(
            all_pm25_summaries
        )

        path = (
            REPORT_DIR
            / "pm25_summary.csv"
        )

        pm25_summary_df.to_csv(
            path,
            index=False
        )

        print(
            f"Saved: {path}"
        )

    # ========================================================
    # STATION DISTANCES
    # ========================================================

    distance_records = []

    station_codes = [
        s["station_code"]
        for s in station_results
        if s["station_code"]
        in station_coordinates
    ]

    for i in range(
        len(station_codes)
    ):

        for j in range(
            i + 1,
            len(station_codes)
        ):

            station_a = station_codes[i]
            station_b = station_codes[j]

            lon_a, lat_a = (
                station_coordinates[
                    station_a
                ]
            )

            lon_b, lat_b = (
                station_coordinates[
                    station_b
                ]
            )

            distance = haversine_km(
                lat_a,
                lon_a,
                lat_b,
                lon_b
            )

            distance_records.append(
                {
                    "station_a": station_a,
                    "station_a_name": STATION_NAMES.get(
                        station_a,
                        "Unknown"
                    ),
                    "station_b": station_b,
                    "station_b_name": STATION_NAMES.get(
                        station_b,
                        "Unknown"
                    ),
                    "distance_km": distance,
                }
            )

    if distance_records:

        distance_df = (
            pd.DataFrame(
                distance_records
            )
            .sort_values(
                "distance_km"
            )
        )

        path = (
            REPORT_DIR
            / "station_distances.csv"
        )

        distance_df.to_csv(
            path,
            index=False
        )

        print(
            f"Saved: {path}"
        )

        print(
            "\nClosest station pairs:"
        )

        for _, row in (
            distance_df.head(15).iterrows()
        ):

            print(
                f"  {row['station_a']} "
                f"({row['station_a_name']}) ↔ "
                f"{row['station_b']} "
                f"({row['station_b_name']}): "
                f"{row['distance_km']:.3f} km"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Stations successfully audited: "
        f"{len(station_results)}"
    )

    if station_results:

        earliest = min(
            x["first_timestamp"]
            for x in station_results
        )

        latest = max(
            x["last_timestamp"]
            for x in station_results
        )

        print(
            f"Overall earliest timestamp: "
            f"{earliest}"
        )

        print(
            f"Overall latest timestamp:   "
            f"{latest}"
        )

    print(
        f"\nReports saved in: "
        f"{REPORT_DIR}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No stations or observations were "
        "removed during this audit."
    )

    print(
        "The next step is to use "
        "missingness_by_year.csv to "
        "select the final modelling period "
        "and station set."
    )


if __name__ == "__main__":
    main()