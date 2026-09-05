import os

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "data/raw/sensor_218_raw.csv"
OUTPUT_FILE = "data/processed/sensor_218_hourly.csv"

START = "2016-02-09 04:00:00+00:00"
END = "2017-02-08 03:00:00+00:00"


# =============================================================================
# LOAD RAW DATA
# =============================================================================

def load_raw_data(input_file=INPUT_FILE):
    """
    Load the raw Sensor 218 dataset.
    """

    print("=" * 80)
    print("LOADING SENSOR 218 RAW DATA")
    print("=" * 80)

    df = pd.read_csv(input_file)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    required_columns = {"timestamp", "pm25"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce",
    )

    return df


# =============================================================================
# REMOVE DUPLICATES
# =============================================================================

def remove_duplicates(df):
    """
    Remove duplicate timestamps while keeping the first observation.
    """

    print("\n" + "=" * 80)
    print("REMOVING DUPLICATE TIMESTAMPS")
    print("=" * 80)

    duplicates = df["timestamp"].duplicated().sum()

    print(f"\nDuplicate timestamps found: {duplicates:,}")

    if duplicates > 0:

        df = df.drop_duplicates(
            subset="timestamp",
            keep="first",
        ).copy()

        print(
            f"Rows after duplicate removal: "
            f"{len(df):,}"
        )

    else:

        print("No duplicates removed.")

    return df


# =============================================================================
# CREATE COMPLETE HOURLY TIMELINE
# =============================================================================

def create_hourly_dataset(df):
    """
    Create a complete hourly timeline for the study period.

    Missing measurement hours are explicitly represented as NaN.
    """

    print("\n" + "=" * 80)
    print("CREATING COMPLETE HOURLY TIMELINE")
    print("=" * 80)

    start = pd.Timestamp(START)
    end = pd.Timestamp(END)

    # END is exclusive.
    hourly_index = pd.date_range(
        start=start,
        end=end - pd.Timedelta(hours=1),
        freq="h",
        tz="UTC",
    )

    print(
        f"\nExpected hourly timestamps: "
        f"{len(hourly_index):,}"
    )

    df = df.set_index("timestamp")

    # Reindex creates explicit rows for hours
    # where no measurement was recorded.
    df = df.reindex(hourly_index)

    df.index.name = "timestamp"

    df = df.reset_index()

    return df


# =============================================================================
# ADD MISSINGNESS FLAGS
# =============================================================================

def add_missingness_flags(df):
    """
    Add an indicator showing whether PM2.5 is missing for an hour.
    """

    print("\n" + "=" * 80)
    print("ADDING MISSINGNESS INFORMATION")
    print("=" * 80)

    df["pm25_missing"] = df["pm25"].isna()

    missing = int(df["pm25_missing"].sum())

    print(
        f"\nTotal missing PM2.5 hours: "
        f"{missing:,}"
    )

    print(
        f"Missing PM2.5 rate: "
        f"{missing / len(df) * 100:.2f}%"
    )

    return df


# =============================================================================
# VALIDATE PM2.5 VALUES
# =============================================================================

def validate_values(df):
    """
    Check PM2.5 values for potentially invalid observations.

    Values above 250 are preserved because they are not automatically
    treated as invalid for this project.
    """

    print("\n" + "=" * 80)
    print("VALIDATING PM2.5 VALUES")
    print("=" * 80)

    valid = df["pm25"].dropna()

    negative = int((valid < 0).sum())
    zero = int((valid == 0).sum())
    extreme = int((valid > 250).sum())

    print(f"\nValid PM2.5 observations: {len(valid):,}")
    print(f"Negative values:          {negative:,}")
    print(f"Zero values:              {zero:,}")
    print(f">250 ug/m3:               {extreme:,}")

    if negative > 0:
        print(
            "\nWARNING: Negative PM2.5 values found."
        )
    else:
        print(
            "\nNo negative PM2.5 values found."
        )

    print(
        "\nPM2.5 values above 250 ug/m3 "
        "are being preserved."
    )

    return df


# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

def save_processed_data(
    df,
    output_file=OUTPUT_FILE,
):
    """
    Save the hourly processed dataset.
    """

    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)

    directory = os.path.dirname(output_file)

    if directory:
        os.makedirs(directory, exist_ok=True)

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"\nFile: {output_file}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    print(
        f"\nStart: {df['timestamp'].min()}"
    )

    print(
        f"End:   {df['timestamp'].max()}"
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_summary(df):
    """
    Print a final summary of the preprocessed dataset.
    """

    print("\n" + "=" * 80)
    print("FINAL PREPROCESSED DATA SUMMARY")
    print("=" * 80)

    expected = len(
        pd.date_range(
            start=pd.Timestamp(START),
            end=pd.Timestamp(END) - pd.Timedelta(hours=1),
            freq="h",
            tz="UTC",
        )
    )

    actual = len(df)

    missing = int(df["pm25"].isna().sum())

    print(
        f"\nExpected hourly rows:     {expected:,}"
    )

    print(
        f"Actual rows:              {actual:,}"
    )

    print(
        f"Missing PM2.5 values:     {missing:,}"
    )

    print(
        f"Missing PM2.5 rate:       "
        f"{missing / actual * 100:.2f}%"
    )

    print(
        f"Duplicate timestamps:     "
        f"{df['timestamp'].duplicated().sum():,}"
    )

    print(
        f"Timestamp range:          "
        f"{df['timestamp'].min()} -> "
        f"{df['timestamp'].max()}"
    )

    if actual == expected:
        print("\nHourly timeline is complete.")
    else:
        print(
            "\nWARNING: Hourly timeline does not contain "
            "the expected number of rows."
        )

    print(
        "\nDataset is ready for feature engineering."
    )


# =============================================================================
# COMPLETE PREPROCESSING PIPELINE
# =============================================================================

def run_preprocessing(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
):
    """
    Run the complete preprocessing pipeline.

    Returns
    -------
    pandas.DataFrame
        Processed hourly dataset.
    """

    df = load_raw_data(input_file)

    df = remove_duplicates(df)

    df = create_hourly_dataset(df)

    df = add_missingness_flags(df)

    df = validate_values(df)

    save_processed_data(
        df,
        output_file,
    )

    print_summary(df)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    run_preprocessing()


if __name__ == "__main__":
    main()