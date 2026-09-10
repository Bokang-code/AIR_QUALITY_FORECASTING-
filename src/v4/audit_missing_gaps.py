from pathlib import Path

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/chengdu")
REPORT_DIR = Path("reports/v4")

START_DATE = "2015-01-01"
END_DATE = "2021-12-02 08:00:00"

TARGET_COLUMN = "PM2.5"

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
# FIND CONSECUTIVE MISSING GAPS
# ============================================================

def find_missing_gaps(df, station_code):

    df = df.copy()

    df["timestamp"] = build_timestamp(df)

    df = df.dropna(
        subset=["timestamp"]
    )

    # Restrict to the chosen study period
    df = df[
        (df["timestamp"] >= pd.Timestamp(START_DATE))
        & (df["timestamp"] <= pd.Timestamp(END_DATE))
    ].copy()

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Create missing flag
    # --------------------------------------------------------

    df["missing"] = df[TARGET_COLUMN].isna()

    # A new group starts whenever missing/non-missing changes
    df["group"] = (
        df["missing"]
        .ne(df["missing"].shift())
        .cumsum()
    )

    # --------------------------------------------------------
    # Extract only missing groups
    # --------------------------------------------------------

    gaps = []

    for _, group in df.groupby("group"):

        if not group["missing"].iloc[0]:
            continue

        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1]

        hours = len(group)

        gaps.append(
            {
                "station_code": station_code,
                "station_name": STATION_NAMES.get(
                    station_code,
                    "Unknown"
                ),
                "start": start,
                "end": end,
                "hours_missing": hours,
            }
        )

    return pd.DataFrame(gaps)


# ============================================================
# REPORT HELPERS
# ============================================================

def print_station_gap_summary(gaps_df):
    for station_code in sorted(gaps_df["station_code"].unique()):
        station_gaps = gaps_df[gaps_df["station_code"] == station_code]
        station_name = STATION_NAMES.get(station_code, "Unknown")

        total_missing_hours = station_gaps["hours_missing"].sum()
        longest_gap = station_gaps["hours_missing"].max()
        average_gap = station_gaps["hours_missing"].mean()
        median_gap = station_gaps["hours_missing"].median()

        gaps_1h = (station_gaps["hours_missing"] == 1).sum()
        gaps_2_6h = ((station_gaps["hours_missing"] >= 2) & (station_gaps["hours_missing"] <= 6)).sum()
        gaps_7_23h = ((station_gaps["hours_missing"] >= 7) & (station_gaps["hours_missing"] <= 23)).sum()
        gaps_24h_plus = (station_gaps["hours_missing"] >= 24).sum()

        print(f"\n{station_code} ({station_name})")
        print(f"  Total missing hours: {total_missing_hours:,}")
        print(f"  Number of gaps: {len(station_gaps):,}")
        print(f"  Longest gap: {longest_gap:,} hours")
        print(f"  Average gap: {average_gap:.2f} hours")
        print(f"  Median gap: {median_gap:.2f} hours")
        print(f"  1-hour gaps: {gaps_1h:,}")
        print(f"  2–6 hour gaps: {gaps_2_6h:,}")
        print(f"  7–23 hour gaps: {gaps_7_23h:,}")
        print(f"  ≥24 hour gaps: {gaps_24h_plus:,}")


def print_longest_gaps(gaps_df):
    print("\n" + "=" * 80)
    print("10 LONGEST PM2.5 MISSING GAPS")
    print("=" * 80)

    longest = gaps_df.sort_values("hours_missing", ascending=False).head(10)
    for _, gap in longest.iterrows():
        print(f"{gap['station_code']} ({gap['station_name']}): {gap['start']} → {gap['end']} | {int(gap['hours_missing'])} hours")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("PM2.5 MISSING-GAP AUDIT")
    print("=" * 80)
    print(f"Study period: {START_DATE} → {END_DATE}")
    print("Target: PM2.5")
    print("\nIMPORTANT: This audit does NOT remove or modify data.")

    excel_files = sorted(DATA_DIR.glob("*.xlsx"))
    if not excel_files:
        print("\nERROR: No Excel files found.")
        return

    all_gaps = []

    for file_path in excel_files:
        station_code = file_path.stem
        print(f"\nProcessing {station_code} ({STATION_NAMES.get(station_code, 'Unknown')})...")

        try:
            df = pd.read_excel(file_path)
            gaps = find_missing_gaps(df, station_code)

            if len(gaps) == 0:
                print("  No missing PM2.5 gaps found.")
                continue

            all_gaps.append(gaps)
            print(f"  Number of missing gaps: {len(gaps):,}")
            print(f"  Longest missing gap: {gaps['hours_missing'].max():,} hours")
            print(f"  Average missing gap: {gaps['hours_missing'].mean():.2f} hours")
            print(f"  Median missing gap: {gaps['hours_missing'].median():.2f} hours")

            long_gaps = gaps[gaps["hours_missing"] >= 24].sort_values("hours_missing", ascending=False).head(10)
            print("\n  Long gaps:")
            if len(long_gaps) == 0:
                print("    None ≥ 24 hours")
            else:
                for _, gap in long_gaps.iterrows():
                    print(f"    {gap['start']} → {gap['end']} | {int(gap['hours_missing'])} hours")

        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_gaps:
        print("\nNo missing PM2.5 gaps found.")
        return

    gaps_df = pd.concat(all_gaps, ignore_index=True).sort_values(["station_code", "hours_missing"], ascending=[True, False])
    output_file = REPORT_DIR / "pm25_missing_gaps.csv"
    gaps_df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    print("\n" + "=" * 80)
    print("MISSING GAP SUMMARY")
    print("=" * 80)
    print_station_gap_summary(gaps_df)
    print_longest_gaps(gaps_df)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Full gap report: {output_file}")
    print("\nNo data has been removed.")


if __name__ == "__main__":
    main()