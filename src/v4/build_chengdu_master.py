from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

# If your Excel files are currently somewhere else, change this.
CHENGDU_DIR = DATA_DIR / "chengdu"

OUTPUT_DIR = DATA_DIR / "processed" / "v4"

OUTPUT_FILE = OUTPUT_DIR / "chengdu_master_2015_2021.csv"


# ============================================================
# STATION INFORMATION
# ============================================================

STATIONS = {
    "1431A": {
        "name": "JQLH",
        "longitude": 103.9728,
        "latitude": 30.7236,
    },
    "1432A": {
        "name": "SLD",
        "longitude": 104.1419,
        "latitude": 30.6764,
    },
    "1433A": {
        "name": "SWY",
        "longitude": 104.0594,
        "latitude": 30.5767,
    },
    "1434A": {
        "name": "SHP",
        "longitude": 104.1122,
        "latitude": 30.6306,
    },
    "1437A": {
        "name": "JPJ",
        "longitude": 104.0431,
        "latitude": 30.6556,
    },
    "1438A": {
        "name": "LYS",
        "longitude": 103.6202,
        "latitude": 31.0201,
    },
    "2880A": {
        "name": "DSXL",
        "longitude": 104.0219,
        "latitude": 30.6558,
    },
    "3136A": {
        "name": "LQXQ",
        "longitude": 104.2725,
        "latitude": 30.5589,
    },
    "3358A": {
        "name": "LJL",
        "longitude": 103.8458,
        "latitude": 30.6994,
    },
}


# ============================================================
# SOURCE COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "AQI",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "O3-8h",
    "CO",
    "U10",
    "V10",
    "T2",
    "RH2",
    "PBLH",
]


# ============================================================
# COLUMN RENAMING
# ============================================================

COLUMN_RENAME = {
    "AQI": "aqi",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "O3-8h": "o3_8h",
    "CO": "co",
    "U10": "u10",
    "V10": "v10",
    "T2": "temperature",
    "RH2": "relative_humidity",
    "PBLH": "boundary_layer_height",
}


# ============================================================
# LOAD ONE STATION
# ============================================================

def load_station(station_code: str, station_info: dict) -> pd.DataFrame:
    """
    Load one Chengdu station Excel file and standardise its columns.
    """

    file_path = CHENGDU_DIR / f"{station_code}.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{file_path}\n\n"
            f"Expected Excel file: {station_code}.xlsx"
        )

    print(f"Reading {file_path.name}...")

    df = pd.read_excel(file_path)

    # --------------------------------------------------------
    # Check source columns
    # --------------------------------------------------------

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{station_code} is missing expected columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Keep only expected source columns
    # --------------------------------------------------------

    df = df[EXPECTED_COLUMNS].copy()

    # --------------------------------------------------------
    # Build local timestamp
    #
    # The source data does not contain timezone information.
    # We treat the recorded hours as China Standard Time
    # (Asia/Shanghai, UTC+8).
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
        },
        errors="coerce",
    )

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        print(
            f"  WARNING: {invalid_timestamps} invalid timestamps "
            f"found in {station_code}; these rows will be removed."
        )

        df = df.dropna(subset=["timestamp"]).copy()

    # --------------------------------------------------------
    # Add station metadata
    # --------------------------------------------------------

    df["station_code"] = station_code
    df["station_name"] = station_info["name"]
    df["longitude"] = station_info["longitude"]
    df["latitude"] = station_info["latitude"]

    # --------------------------------------------------------
    # Rename variables
    # --------------------------------------------------------

    df = df.rename(columns=COLUMN_RENAME)

    # --------------------------------------------------------
    # Drop original date components
    # --------------------------------------------------------

    df = df.drop(
        columns=["year", "month", "day", "hour"]
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    column_order = [
        "timestamp",
        "station_code",
        "station_name",
        "longitude",
        "latitude",
        "aqi",
        "pm25",
        "pm10",
        "so2",
        "no2",
        "o3_8h",
        "co",
        "u10",
        "v10",
        "temperature",
        "relative_humidity",
        "boundary_layer_height",
    ]

    df = df[column_order]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# HELPERS
# ============================================================

def exclude_station_year(df, station_code):
    if station_code != "3136A":
        return df

    exclusion_mask = df["timestamp"].dt.year == 2016
    excluded_rows = int(exclusion_mask.sum())

    if excluded_rows > 0:
        print(f"  Excluding {excluded_rows:,} rows from {station_code} for 2016.")
        df = df.loc[~exclusion_mask].copy()

    return df


def print_station_summary(df, station_code):
    print(f"  {station_code} ({STATIONS[station_code]['name']}): {len(df):,} rows")
    print(f"    {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"    PM2.5 missing: {df['pm25'].isna().sum():,} ({df['pm25'].isna().mean() * 100:.2f}%)")


def drop_duplicate_station_timestamps(df):
    duplicate_mask = df.duplicated(subset=["station_code", "timestamp"], keep=False)
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        print(f"\nWARNING: Found {duplicate_count:,} duplicate station timestamps.")
        print("Keeping the first occurrence.")
        df = df.drop_duplicates(subset=["station_code", "timestamp"], keep="first").reset_index(drop=True)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING CHENGDU MASTER DATASET")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stations = []
    total_rows_before_exclusion = 0
    total_rows_excluded = 0

    for station_code, station_info in STATIONS.items():
        df = load_station(station_code, station_info)

        total_rows_before_exclusion += len(df)
        rows_before = len(df)
        df = exclude_station_year(df, station_code)
        total_rows_excluded += rows_before - len(df)
        print_station_summary(df, station_code)
        all_stations.append(df)

    master = pd.concat(all_stations, ignore_index=True)
    master = master.sort_values(["timestamp", "station_code"]).reset_index(drop=True)
    master = drop_duplicate_station_timestamps(master)
    master.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("MASTER DATASET CREATED")
    print("=" * 70)

    print(f"\nOutput:")
    print(OUTPUT_FILE)
    print(f"\nRows before special exclusion: {total_rows_before_exclusion:,}")
    print(f"Rows excluded:                  {total_rows_excluded:,}")
    print(f"Rows in master dataset:         {len(master):,}")
    print(f"\nColumns: {len(master.columns)}")

    print("\nStations:")
    station_summary = (
        master.groupby(["station_code", "station_name"])
        .agg(rows=("timestamp", "size"), start=("timestamp", "min"), end=("timestamp", "max"), pm25_missing=("pm25", lambda x: x.isna().sum()))
        .reset_index()
    )
    station_summary["pm25_missing_pct"] = station_summary["pm25_missing"] / station_summary["rows"] * 100
    print(station_summary.to_string(index=False))

    print("\nMissing values:")
    missing = master.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("No missing values.")
    else:
        missing_table = pd.DataFrame({"missing": missing, "percentage": missing / len(master) * 100})
        print(missing_table.to_string())

    print("\nTimestamp range:")
    print(f"Start: {master['timestamp'].min()}")
    print(f"End:   {master['timestamp'].max()}")
    print("\nDataset shape:")
    print(master.shape)
    print("\nFirst 5 rows:")
    print(master.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(master.tail().to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()