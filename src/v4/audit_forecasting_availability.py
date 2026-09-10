from pathlib import Path

import pandas as pd
import numpy as np

try:
    from config import FORECAST_HORIZONS
except ImportError:  # pragma: no cover
    from src.v4.config import FORECAST_HORIZONS


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/chengdu")
REPORT_DIR = Path("reports/v4")

START_DATE = "2015-01-01"
END_DATE = "2021-12-02 08:00:00"

TARGET_COLUMN = "PM2.5"

# Forecast horizons
HORIZONS = {name: hours for name, hours in FORECAST_HORIZONS.items() if name != "24h"}


# 3136A has a very large PM2.5 outage during 2016.
# We will exclude ONLY that station/year.
EXCLUDE_STATION_YEAR = {
    ("3136A", 2016)
}


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
# BUILD TIMESTAMP
# ============================================================

def build_timestamp(df):

    return pd.to_datetime(
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
        errors="coerce"
    )


# ============================================================
# AUDIT ONE STATION
# ============================================================

def audit_station(file_path, station_code):

    df = pd.read_excel(file_path)

    # Build timestamp
    df["timestamp"] = build_timestamp(df)

    # Remove invalid timestamps
    df = df.dropna(
        subset=["timestamp"]
    ).copy()

    # Restrict to study period
    df = df[
        (df["timestamp"] >= pd.Timestamp(START_DATE))
        & (df["timestamp"] <= pd.Timestamp(END_DATE))
    ].copy()

    # Sort chronologically
    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Convert PM2.5 to numeric
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # Add year
    df["study_year"] = df["timestamp"].dt.year

    results = []

    # --------------------------------------------------------
    # Check each forecasting horizon
    # --------------------------------------------------------

    for horizon_name, horizon_hours in HORIZONS.items():

        # Future PM2.5 target
        df["future_pm25"] = (
            df[TARGET_COLUMN]
            .shift(-horizon_hours)
        )

        # A forecasting sample is usable when:
        # 1. Current PM2.5 exists
        # 2. Future PM2.5 target exists
        #
        # We are NOT requiring other features yet.
        # This audit focuses on target availability.

        df["usable"] = (
            df[TARGET_COLUMN].notna()
            & df["future_pm25"].notna()
        )

        for year, year_df in df.groupby(
            "study_year"
        ):

            # Special exclusion:
            # 3136A in 2016
            if (
                station_code,
                int(year)
            ) in EXCLUDE_STATION_YEAR:

                continue

            total_rows = len(year_df)

            usable_samples = (
                year_df["usable"].sum()
            )

            usable_pct = (
                usable_samples
                / total_rows
                * 100
                if total_rows > 0
                else 0
            )

            results.append(
                {
                    "station_code": station_code,
                    "station_name": STATION_NAMES.get(
                        station_code,
                        "Unknown"
                    ),
                    "year": int(year),
                    "horizon": horizon_name,
                    "horizon_hours": horizon_hours,
                    "total_rows": total_rows,
                    "usable_samples": int(
                        usable_samples
                    ),
                    "unusable_samples": int(
                        total_rows
                        - usable_samples
                    ),
                    "usable_pct": usable_pct,
                }
            )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def summarize_station_availability(summary):
    for station_code in sorted(summary["station_code"].unique()):
        station_summary = summary[summary["station_code"] == station_code]
        station_name = STATION_NAMES.get(station_code, "Unknown")
        print(f"\n{station_code} ({station_name})")
        for _, row in station_summary.iterrows():
            print(f"  {row['horizon']:>4}: {int(row['usable_samples']):,} / {int(row['total_rows']):,} ({row['usable_pct']:.2f}%)")


def summarize_yearly_availability(results_df):
    yearly_summary = results_df.groupby(["year", "horizon"], as_index=False).agg(total_samples=("total_rows", "sum"), usable_samples=("usable_samples", "sum"))
    yearly_summary["usable_pct"] = yearly_summary["usable_samples"] / yearly_summary["total_samples"] * 100

    print("\n" + "=" * 80)
    print("USABLE SAMPLES ACROSS ALL AVAILABLE STATIONS")
    print("=" * 80)

    for year in sorted(yearly_summary["year"].unique()):
        print(f"\n{int(year)}")
        year_df = yearly_summary[yearly_summary["year"] == year]
        for _, row in year_df.iterrows():
            print(f"  {row['horizon']:>4}: {int(row['usable_samples']):,} / {int(row['total_samples']):,} ({row['usable_pct']:.2f}%)")


def main():

    print("\n" + "=" * 80)
    print("FORECASTING AVAILABILITY AUDIT")
    print("=" * 80)
    print(f"Study period: {START_DATE} → {END_DATE}")
    print("\nForecast horizons:")
    for name, hours in HORIZONS.items():
        print(f"  {name}: {hours} hours")

    print("\nSpecial exclusion:")
    print("  3136A (LQXQ) — 2016 excluded")
    print("\nDefinition of usable sample:")
    print("  Current PM2.5 exists AND future PM2.5 target exists.")
    print("\nNo observations are being removed.")

    files = sorted(DATA_DIR.glob("*.xlsx"))
    if not files:
        print("\nERROR: No Excel files found.")
        return

    all_results = []
    for file_path in files:
        station_code = file_path.stem
        print(f"\nProcessing {station_code} ({STATION_NAMES.get(station_code, 'Unknown')})...")
        try:
            result = audit_station(file_path, station_code)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_results:
        print("\nNo results generated.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    detailed_path = REPORT_DIR / "forecasting_availability_by_year.csv"
    results_df.to_csv(detailed_path, index=False)
    print(f"\nSaved: {detailed_path}")

    summary = results_df.groupby(["station_code", "station_name", "horizon", "horizon_hours"], as_index=False).agg(total_rows=("total_rows", "sum"), usable_samples=("usable_samples", "sum"), unusable_samples=("unusable_samples", "sum"))
    summary["usable_pct"] = summary["usable_samples"] / summary["total_rows"] * 100
    summary_path = REPORT_DIR / "forecasting_availability_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("USABLE FORECASTING SAMPLES")
    print("=" * 80)
    summarize_station_availability(summary)

    summarize_yearly_availability(results_df)

    print("\n" + "=" * 80)
    print("STATION COVERAGE BY YEAR")
    print("=" * 80)
    station_coverage = results_df.groupby(["year", "horizon"]).agg(stations_available=("station_code", "nunique")).reset_index()
    for year in sorted(station_coverage["year"].unique()):
        year_df = station_coverage[station_coverage["year"] == year]
        print(f"\n{int(year)}")
        for _, row in year_df.iterrows():
            print(f"  {row['horizon']:>4}: {int(row['stations_available'])} stations")

    print("\nDone.")


if __name__ == "__main__":
    main()