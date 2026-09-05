import os
import numpy as np
import pandas as pd


# =============================================================================
# STEP 6: V2 FEATURE ENGINEERING
# =============================================================================

INPUT_FILE = "data/processed/engineered_sensor_218.csv"
OUTPUT_FILE = "data/processed/sensor_218_hourly_v2.csv"


print("\n" + "=" * 80)
print("STEP 6: V2 FEATURE ENGINEERING")
print("=" * 80)

print(
    "IMPORTANT:\n"
    "• Existing V1 engineered features are preserved.\n"
    "• New V2 features use only current/historical information.\n"
    "• No future PM2.5 values are used to create features."
)

# =============================================================================
# LOAD V1 ENGINEERED DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING V1 ENGINEERED SENSOR 218 DATA")
print("=" * 80)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Make sure engineered_sensor_218.csv exists in data/processed/."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df):,}")

# =============================================================================
# IDENTIFY TIMESTAMP
# =============================================================================

timestamp_candidates = [
    "timestamp",
    "datetime",
    "date",
    "time",
]

timestamp_column = None

for column in timestamp_candidates:
    if column in df.columns:
        timestamp_column = column
        break

if timestamp_column is None:
    raise ValueError(
        "Could not find a timestamp column. "
        f"Available columns: {list(df.columns)}"
    )

df[timestamp_column] = pd.to_datetime(
    df[timestamp_column],
    errors="coerce",
    utc=True
)

if df[timestamp_column].isna().any():
    raise ValueError("Some timestamps could not be converted.")

df = df.sort_values(timestamp_column).reset_index(drop=True)

print(f"Start:          {df[timestamp_column].min()}")
print(f"End:            {df[timestamp_column].max()}")

# =============================================================================
# IDENTIFY PM2.5 COLUMN
# =============================================================================

if "pm25" not in df.columns:
    raise ValueError(
        "PM2.5 column 'pm25' was not found in the V1 dataset."
    )

df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

print(f"\nPM2.5 observations: {df['pm25'].notna().sum():,}")
print(f"PM2.5 missing:      {df['pm25'].isna().sum():,}")

# Keep a copy of the original V1 columns
original_columns = list(df.columns)

# Keep track of newly created V2 features
new_features = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_feature(name, values):
    """Add a feature while tracking it."""
    df[name] = values
    new_features.append(name)


# =============================================================================
# ADDITIONAL LAG FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("ADDING V2 LAG FEATURES")
print("=" * 80)

lag_hours = [
    2,
    4,
    8,
    18,
    36,
    72,
    120,
    336,
]

for lag in lag_hours:
    name = f"pm25_lag_{lag}h"

    add_feature(
        name,
        df["pm25"].shift(lag)
    )

    print(f"[ADDED] {name}")


# =============================================================================
# ADDITIONAL ROLLING FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("ADDING V2 ROLLING FEATURES")
print("=" * 80)

rolling_windows = [
    3,
    12,
    48,
    72,
    336,
]

for window in rolling_windows:

    # IMPORTANT:
    # shift(1) ensures the rolling window only contains information
    # available BEFORE the prediction timestamp.
    historical = df["pm25"].shift(1).rolling(
        window=window,
        min_periods=window
    )

    add_feature(
        f"pm25_rolling_mean_{window}h",
        historical.mean()
    )

    add_feature(
        f"pm25_rolling_std_{window}h",
        historical.std()
    )

    add_feature(
        f"pm25_rolling_min_{window}h",
        historical.min()
    )

    add_feature(
        f"pm25_rolling_max_{window}h",
        historical.max()
    )

    add_feature(
        f"pm25_rolling_median_{window}h",
        historical.median()
    )

    add_feature(
        f"pm25_rolling_range_{window}h",
        historical.max() - historical.min()
    )

    print(
        f"[ADDED] {window}h rolling "
        "mean/std/min/max/median/range"
    )


# =============================================================================
# PM2.5 TREND FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("ADDING PM2.5 TREND FEATURES")
print("=" * 80)

trend_windows = [
    3,
    6,
    12,
    24,
    48,
]

for window in trend_windows:

    change_name = f"pm25_change_{window}h"
    pct_name = f"pm25_pct_change_{window}h"

    add_feature(
        change_name,
        df["pm25"] - df["pm25"].shift(window)
    )

    previous = df["pm25"].shift(window)

    # Avoid division by zero
    pct_change = np.where(
        previous != 0,
        (df["pm25"] - previous) / previous * 100,
        np.nan
    )

    add_feature(
        pct_name,
        pct_change
    )

    print(f"[ADDED] {change_name}")
    print(f"[ADDED] {pct_name}")


# =============================================================================
# ROLLING TREND / VOLATILITY FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("ADDING ROLLING TREND / VOLATILITY FEATURES")
print("=" * 80)

volatility_windows = [
    6,
    24,
    48,
    72,
    336,
]

for window in volatility_windows:

    historical = df["pm25"].shift(1).rolling(
        window=window,
        min_periods=window
    )

    rolling_mean = historical.mean()
    rolling_std = historical.std()
    rolling_max = historical.max()
    rolling_min = historical.min()

    mean_deviation = (
        df["pm25"].shift(1) - rolling_mean
    )

    cv = np.where(
        rolling_mean != 0,
        rolling_std / rolling_mean,
        np.nan
    )

    rolling_range = rolling_max - rolling_min

    add_feature(
        f"pm25_mean_deviation_{window}h",
        mean_deviation
    )

    add_feature(
        f"pm25_cv_{window}h",
        cv
    )

    add_feature(
        f"pm25_range_{window}h",
        rolling_range
    )

    print(
        f"[ADDED] {window}h "
        "mean deviation / coefficient of variation / range"
    )


# =============================================================================
# ADDITIONAL SEASONAL FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("ADDING ADDITIONAL SEASONAL FEATURES")
print("=" * 80)

timestamp = df[timestamp_column]

# Day of year
day_of_year = timestamp.dt.dayofyear

add_feature(
    "day_of_year",
    day_of_year
)

add_feature(
    "day_of_year_sin",
    np.sin(2 * np.pi * day_of_year / 365.25)
)

add_feature(
    "day_of_year_cos",
    np.cos(2 * np.pi * day_of_year / 365.25)
)

# ISO week number
week_of_year = timestamp.dt.isocalendar().week.astype(int)

add_feature(
    "week_of_year",
    week_of_year
)

add_feature(
    "week_of_year_sin",
    np.sin(2 * np.pi * week_of_year / 52.18)
)

add_feature(
    "week_of_year_cos",
    np.cos(2 * np.pi * week_of_year / 52.18)
)

print("[ADDED] day_of_year")
print("[ADDED] day_of_year_sin")
print("[ADDED] day_of_year_cos")
print("[ADDED] week_of_year")
print("[ADDED] week_of_year_sin")
print("[ADDED] week_of_year_cos")


# =============================================================================
# FEATURE LEAKAGE CHECK
# =============================================================================

print("\n" + "=" * 80)
print("V2 FEATURE LEAKAGE CHECK")
print("=" * 80)

future_keywords = [
    "future",
    "target",
    "lead",
    "next",
    "ahead",
]

future_features = []

for feature in new_features:

    feature_lower = feature.lower()

    if any(keyword in feature_lower for keyword in future_keywords):
        future_features.append(feature)

if future_features:

    print("[WARNING] Potential future-looking V2 feature names detected:")

    for feature in future_features:
        print(f"  - {feature}")

else:

    print(
        "[PASS] No obviously future-looking V2 feature names detected."
    )


# =============================================================================
# VERIFY ORIGINAL V1 FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("V1 FEATURE PRESERVATION CHECK")
print("=" * 80)

missing_original = [
    column
    for column in original_columns
    if column not in df.columns
]

if missing_original:

    print("[FAIL] Original V1 columns were lost:")

    for column in missing_original:
        print(f"  - {column}")

    raise ValueError(
        "V1 feature preservation check failed."
    )

else:

    print(
        f"[PASS] All {len(original_columns)} original V1 columns "
        "are preserved."
    )


# =============================================================================
# TARGET CHECK
# =============================================================================

print("\n" + "=" * 80)
print("TARGET COLUMN CHECK")
print("=" * 80)

target_columns = [
    column
    for column in df.columns
    if "target" in column.lower()
]

if target_columns:

    print("Target columns detected:")

    for column in target_columns:
        print(f"  - {column}")

else:

    print("Target columns detected:")
    print("  None")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("V2 DATASET SUMMARY")
print("=" * 80)

print(f"Rows:              {len(df):,}")
print(f"Original V1 columns: {len(original_columns):,}")
print(f"New V2 features:     {len(new_features):,}")
print(f"Final columns:        {len(df.columns):,}")

print(f"\nPM2.5 missing: {df['pm25'].isna().sum():,}")


# =============================================================================
# MISSING VALUES IN NEW FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("MISSING VALUES IN NEW V2 FEATURES")
print("=" * 80)

missing_v2 = (
    df[new_features]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

if missing_v2.sum() == 0:

    print("No missing values in V2 features.")

else:

    print(missing_v2.to_string())


# =============================================================================
# LIST NEW FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("NEW V2 FEATURE NAMES")
print("=" * 80)

for feature in new_features:
    print(f"  - {feature}")


# =============================================================================
# SAVE V2 DATASET
# =============================================================================

print("\n" + "=" * 80)
print("SAVING CORRECTED V2 DATASET")
print("=" * 80)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("V2 dataset saved:")
print(f"  {OUTPUT_FILE}")


# =============================================================================
# FINAL VERIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

saved_df = pd.read_csv(OUTPUT_FILE)

print(f"Saved rows:       {len(saved_df):,}")
print(f"Saved columns:    {len(saved_df.columns):,}")

if len(saved_df) == len(df):

    print("[PASS] Row count preserved.")

else:

    print("[FAIL] Row count changed.")

if "pm25" in saved_df.columns:

    print("[PASS] PM2.5 column preserved.")

else:

    print("[FAIL] PM2.5 column missing.")

missing_after_save = [
    column
    for column in original_columns
    if column not in saved_df.columns
]

if not missing_after_save:

    print(
        "[PASS] All original V1 features preserved after saving."
    )

else:

    print(
        "[FAIL] Some original V1 features disappeared after saving:"
    )

    for column in missing_after_save:
        print(f"  - {column}")


# =============================================================================
# FINAL MESSAGE
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6 V2 FEATURE ENGINEERING COMPLETE")
print("=" * 80)

print(
    "\nCorrected V2 dataset:"
    f"\n  {OUTPUT_FILE}"
)

print(
    f"\nV1 columns preserved: {len(original_columns)}"
    f"\nNew V2 features:      {len(new_features)}"
    f"\nFinal columns:        {len(df.columns)}"
)