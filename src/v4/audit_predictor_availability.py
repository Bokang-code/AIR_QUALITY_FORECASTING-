from pathlib import Path

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/chengdu")
REPORT_DIR = Path("reports/v4")

START_DATE = "2015-01-01"
END_DATE = "2021-12-02 08:00:00"

TARGET = "PM2.5"

# All variables we may use as predictors
PREDICTORS = [
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

# Forecast horizons
HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}

# 3136A has a major PM2.5 outage during 2016.
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

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Convert variables to numeric
    for column in [TARGET] + PREDICTORS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["study_year"] = df["timestamp"].dt.year

    results = []

    # ========================================================
    # CHECK EACH HORIZON
    # ========================================================

    for horizon_name, horizon_hours in HORIZONS.items():

        # Future PM2.5 target
        df["future_pm25"] = (
            df[TARGET]
            .shift(-horizon_hours)
        )

        # ----------------------------------------------------
        # Target-only availability
        # ----------------------------------------------------

        target_available = (
            df[TARGET].notna()
            & df["future_pm25"].notna()
        )

        # ----------------------------------------------------
        # Predictor availability
        # ----------------------------------------------------

        for predictor in PREDICTORS:

            predictor_available = (
                df[predictor].notna()
            )

            usable = (
                target_available
                & predictor_available
            )

            for year, year_df in df.groupby(
                "study_year"
            ):

                # Exclude 3136A 2016
                if (
                    station_code,
                    int(year)
                ) in EXCLUDE_STATION_YEAR:

                    continue

                total_rows = len(year_df)

                target_samples = (
                    target_available[
                        year_df.index
                    ].sum()
                )

                predictor_samples = (
                    predictor_available[
                        year_df.index
                    ].sum()
                )

                usable_samples = (
                    usable[
                        year_df.index
                    ].sum()
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
                        "predictor": predictor,
                        "total_rows": total_rows,
                        "target_available_samples": int(
                            target_samples
                        ),
                        "predictor_available_samples": int(
                            predictor_samples
                        ),
                        "usable_samples": int(
                            usable_samples
                        ),
                        "usable_pct": (
                            usable_samples
                            / total_rows
                            * 100
                            if total_rows > 0
                            else 0
                        ),
                    }
                )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("PREDICTOR AVAILABILITY AUDIT")
    print("=" * 80)

    print(
        f"Study period: "
        f"{START_DATE} → {END_DATE}"
    )

    print("\nPredictors:")

    for predictor in PREDICTORS:

        print(
            f"  - {predictor}"
        )

    print("\nHorizons:")

    for name, hours in HORIZONS.items():

        print(
            f"  - {name}: {hours} hours"
        )

    print(
        "\nSpecial exclusion:"
    )

    print(
        "  3136A (LQXQ) — 2016"
    )

    print(
        "\nThis audit does NOT modify the raw data."
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = sorted(
        DATA_DIR.glob("*.xlsx")
    )

    if not files:

        print(
            "\nERROR: No Excel files found."
        )

        return

    all_results = []

    # --------------------------------------------------------
    # Process stations
    # --------------------------------------------------------

    for file_path in files:

        station_code = file_path.stem

        print(
            f"\nProcessing "
            f"{station_code} "
            f"({STATION_NAMES.get(station_code, 'Unknown')})..."
        )

        try:

            result = audit_station(
                file_path,
                station_code
            )

            all_results.append(result)

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    if not all_results:

        print(
            "\nNo results generated."
        )

        return

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    detailed_path = (
        REPORT_DIR
        / "predictor_availability_by_year.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False
    )

    print(
        f"\nSaved: {detailed_path}"
    )

    # ========================================================
    # OVERALL PREDICTOR MISSINGNESS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "OVERALL PREDICTOR MISSINGNESS"
    )

    print(
        "=" * 80
    )

    overall = (
        results_df[
            [
                "station_code",
                "station_name",
                "predictor",
                "total_rows",
                "predictor_available_samples",
            ]
        ]
        .drop_duplicates()
    )

    # Aggregate carefully because each horizon/year
    # repeats the same predictor observations.
    overall_simple = (
        results_df
        .groupby(
            [
                "station_code",
                "station_name",
                "predictor",
            ]
        )
        .agg(
            total_rows=(
                "total_rows",
                "sum"
            ),
            predictor_available=(
                "predictor_available_samples",
                "sum"
            )
        )
        .reset_index()
    )

    overall_simple["missing_pct"] = (
        1
        - (
            overall_simple["predictor_available"]
            / overall_simple["total_rows"]
        )
    ) * 100

    # Because each observation is repeated for every horizon/year,
    # calculate the actual missingness directly from the source
    # data instead for display.

    direct_records = []

    for file_path in files:

        station_code = file_path.stem

        df = pd.read_excel(file_path)

        df["timestamp"] = build_timestamp(df)

        df = df.dropna(
            subset=["timestamp"]
        ).copy()

        df = df[
            (df["timestamp"] >= pd.Timestamp(START_DATE))
            & (df["timestamp"] <= pd.Timestamp(END_DATE))
        ].copy()

        # Exclude 3136A 2016
        if station_code == "3136A":

            df = df[
                df["timestamp"].dt.year != 2016
            ].copy()

        for predictor in PREDICTORS:

            missing = df[predictor].isna().sum()

            total = len(df)

            direct_records.append(
                {
                    "station_code": station_code,
                    "station_name": STATION_NAMES.get(
                        station_code,
                        "Unknown"
                    ),
                    "predictor": predictor,
                    "total_rows": total,
                    "missing": int(missing),
                    "missing_pct": (
                        missing
                        / total
                        * 100
                        if total > 0
                        else 0
                    ),
                }
            )

    direct_missingness = pd.DataFrame(
        direct_records
    )

    print(
        "\nMissingness after excluding "
        "3136A-2016:"
    )

    for station_code in sorted(
        direct_missingness["station_code"].unique()
    ):

        station_df = direct_missingness[
            direct_missingness["station_code"]
            == station_code
        ]

        station_name = station_df[
            "station_name"
        ].iloc[0]

        print(
            f"\n{station_code} ({station_name})"
        )

        for _, row in station_df.iterrows():

            print(
                f"  {row['predictor']:>6}: "
                f"{row['missing_pct']:.2f}%"
            )

    # ========================================================
    # FULL FEATURE SET AVAILABILITY
    # ========================================================
    #
    # This is the most important part.
    #
    # A sample is "full-feature usable" if:
    #
    #   current PM2.5 exists
    #   future PM2.5 exists
    #   ALL predictors exist
    #
    # This tells us how many samples survive when we use
    # every pollutant + every weather variable together.
    # ========================================================

    full_feature_records = []

    print(
        "\n" + "=" * 80
    )

    print(
        "FULL FEATURE-SET AVAILABILITY"
    )

    print(
        "=" * 80
    )

    print(
        "Required predictors:"
    )

    print(
        ", ".join(PREDICTORS)
    )

    for file_path in files:

        station_code = file_path.stem

        df = pd.read_excel(file_path)

        df["timestamp"] = build_timestamp(df)

        df = df.dropna(
            subset=["timestamp"]
        ).copy()

        df = df[
            (df["timestamp"] >= pd.Timestamp(START_DATE))
            & (df["timestamp"] <= pd.Timestamp(END_DATE))
        ].copy()

        # Exclude 3136A 2016
        if station_code == "3136A":

            df = df[
                df["timestamp"].dt.year != 2016
            ].copy()

        df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Numeric conversion
        for column in [TARGET] + PREDICTORS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        for horizon_name, horizon_hours in HORIZONS.items():

            df["future_pm25"] = (
                df[TARGET]
                .shift(-horizon_hours)
            )

            target_available = (
                df[TARGET].notna()
                & df["future_pm25"].notna()
            )

            all_predictors_available = (
                df[PREDICTORS]
                .notna()
                .all(axis=1)
            )

            usable = (
                target_available
                & all_predictors_available
            )

            total = len(df)

            usable_count = usable.sum()

            usable_pct = (
                usable_count
                / total
                * 100
                if total > 0
                else 0
            )

            full_feature_records.append(
                {
                    "station_code": station_code,
                    "station_name": STATION_NAMES.get(
                        station_code,
                        "Unknown"
                    ),
                    "horizon": horizon_name,
                    "horizon_hours": horizon_hours,
                    "total_rows": total,
                    "usable_samples": int(
                        usable_count
                    ),
                    "usable_pct": usable_pct,
                }
            )

    full_feature_df = pd.DataFrame(
        full_feature_records
    )

    full_feature_path = (
        REPORT_DIR
        / "full_feature_availability.csv"
    )

    full_feature_df.to_csv(
        full_feature_path,
        index=False
    )

    print(
        f"\nSaved: {full_feature_path}"
    )

    # --------------------------------------------------------
    # Print concise summary
    # --------------------------------------------------------

    for station_code in sorted(
        full_feature_df["station_code"].unique()
    ):

        station_df = full_feature_df[
            full_feature_df["station_code"]
            == station_code
        ]

        station_name = station_df[
            "station_name"
        ].iloc[0]

        print(
            f"\n{station_code} ({station_name})"
        )

        for _, row in station_df.iterrows():

            print(
                f"  {row['horizon']:>4}: "
                f"{int(row['usable_samples']):,} / "
                f"{int(row['total_rows']):,} "
                f"({row['usable_pct']:.2f}%)"
            )

    # ========================================================
    # BY YEAR — FULL FEATURE SET
    # ========================================================

    yearly_records = []

    for file_path in files:

        station_code = file_path.stem

        df = pd.read_excel(file_path)

        df["timestamp"] = build_timestamp(df)

        df = df.dropna(
            subset=["timestamp"]
        ).copy()

        df = df[
            (df["timestamp"] >= pd.Timestamp(START_DATE))
            & (df["timestamp"] <= pd.Timestamp(END_DATE))
        ].copy()

        if station_code == "3136A":

            df = df[
                df["timestamp"].dt.year != 2016
            ].copy()

        df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        for column in [TARGET] + PREDICTORS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        for horizon_name, horizon_hours in HORIZONS.items():

            df["future_pm25"] = (
                df[TARGET]
                .shift(-horizon_hours)
            )

            target_available = (
                df[TARGET].notna()
                & df["future_pm25"].notna()
            )

            all_predictors_available = (
                df[PREDICTORS]
                .notna()
                .all(axis=1)
            )

            usable = (
                target_available
                & all_predictors_available
            )

            for year, year_df in df.groupby(
                df["timestamp"].dt.year
            ):

                usable_year = usable[
                    year_df.index
                ]

                total_year = len(year_df)

                usable_count = (
                    usable_year.sum()
                )

                yearly_records.append(
                    {
                        "station_code": station_code,
                        "station_name": STATION_NAMES.get(
                            station_code,
                            "Unknown"
                        ),
                        "year": int(year),
                        "horizon": horizon_name,
                        "horizon_hours": horizon_hours,
                        "total_rows": total_year,
                        "usable_samples": int(
                            usable_count
                        ),
                        "usable_pct": (
                            usable_count
                            / total_year
                            * 100
                            if total_year > 0
                            else 0
                        ),
                    }
                )

    yearly_df = pd.DataFrame(
        yearly_records
    )

    yearly_path = (
        REPORT_DIR
        / "full_feature_availability_by_year.csv"
    )

    yearly_df.to_csv(
        yearly_path,
        index=False
    )

    print(
        f"\nSaved: {yearly_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "No raw data was changed."
    )

    print(
        "\nReports created:"
    )

    print(
        f"  {detailed_path}"
    )

    print(
        f"  {full_feature_path}"
    )

    print(
        f"  {yearly_path}"
    )


if __name__ == "__main__":
    main()