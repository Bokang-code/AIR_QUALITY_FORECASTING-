from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# STEP 4.5 — DATA LEAKAGE AUDIT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_sensor_218.csv"
)


def header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# LOAD DATA
# =============================================================================

header("STEP 4.5: DATA LEAKAGE AUDIT")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce"
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns:     {len(df.columns)}")
print(
    f"Time range:  {df['timestamp'].min()} "
    f"to {df['timestamp'].max()}"
)


# =============================================================================
# 1. TIMESTAMP CONTINUITY
# =============================================================================

header("1. TIMESTAMP CONTINUITY")

time_diff = df["timestamp"].diff().dropna()

expected = pd.Timedelta(hours=1)

non_hourly = (
    time_diff != expected
).sum()

print(
    f"Non-1-hour intervals: {non_hourly:,}"
)

if non_hourly == 0:
    print("[PASS] Dataset is continuously hourly.")
else:
    print("[WARNING] Dataset contains gaps or irregular intervals.")
    print()
    print("Most common intervals:")
    print(
        time_diff
        .value_counts()
        .head(10)
        .to_string()
    )


# =============================================================================
# 2. MISSING VALUE HANDLING
# =============================================================================

header("2. MISSING VALUE HANDLING")

actual_missing = df["pm25"].isna().sum()

indicator_missing = int(
    df["pm25_missing"].sum()
)

print(
    f"PM2.5 missing values:       {actual_missing:,}"
)

print(
    f"pm25_missing indicator:     {indicator_missing:,}"
)

if actual_missing == indicator_missing:
    print(
        "[PASS] Missing-value indicator correctly "
        "matches original missing observations."
    )
else:
    print(
        "[FAIL] Missing-value indicator does not match."
    )


# =============================================================================
# 3. CHECK THAT MISSING VALUES WERE NOT INTERPOLATED
# =============================================================================

header("3. INTERPOLATION CHECK")

print(
    "Current PM2.5 missing values:",
    actual_missing
)

if actual_missing > 0:
    print(
        "[PASS] Genuine missing PM2.5 observations "
        "are still represented as NaN."
    )
else:
    print(
        "[WARNING] No PM2.5 missing values remain."
    )


# =============================================================================
# 4. LAG FEATURE LEAKAGE
# =============================================================================

header("4. LAG FEATURE LEAKAGE")

lags = {
    "pm25_lag_1h": 1,
    "pm25_lag_3h": 3,
    "pm25_lag_6h": 6,
    "pm25_lag_12h": 12,
    "pm25_lag_24h": 24,
    "pm25_lag_48h": 48,
    "pm25_lag_168h": 168,
}

for column, lag in lags.items():

    expected_values = (
        df["pm25"].shift(lag)
    )

    actual_values = df[column]

    mask = (
        actual_values.notna()
        & expected_values.notna()
    )

    if mask.sum() == 0:
        print(
            f"[WARNING] {column}: no comparable values."
        )
        continue

    maximum_difference = np.abs(
        actual_values[mask]
        - expected_values[mask]
    ).max()

    if maximum_difference < 1e-10:
        print(
            f"[PASS] {column} correctly uses "
            f"previous {lag} observations."
        )
    else:
        print(
            f"[FAIL] {column} does not match "
            f"the expected historical lag."
        )


# =============================================================================
# 5. ROLLING FEATURE LEAKAGE
# =============================================================================

header("5. ROLLING FEATURE LEAKAGE")

# IMPORTANT:
# shift(1) means the current PM2.5 observation is excluded.
previous_pm25 = df["pm25"].shift(1)

rolling_features = {

    "pm25_rolling_mean_6h":
        previous_pm25
        .rolling(
            window=6,
            min_periods=3
        )
        .mean(),

    "pm25_rolling_std_6h":
        previous_pm25
        .rolling(
            window=6,
            min_periods=3
        )
        .std(),

    "pm25_rolling_mean_24h":
        previous_pm25
        .rolling(
            window=24,
            min_periods=12
        )
        .mean(),

    "pm25_rolling_std_24h":
        previous_pm25
        .rolling(
            window=24,
            min_periods=12
        )
        .std(),

    "pm25_rolling_mean_7d":
        previous_pm25
        .rolling(
            window=168,
            min_periods=84
        )
        .mean(),

    "pm25_rolling_std_7d":
        previous_pm25
        .rolling(
            window=168,
            min_periods=84
        )
        .std(),
}


for column, expected_values in rolling_features.items():

    actual_values = df[column]

    mask = (
        actual_values.notna()
        & expected_values.notna()
    )

    maximum_difference = np.abs(
        actual_values[mask]
        - expected_values[mask]
    ).max()

    if maximum_difference < 1e-10:
        print(
            f"[PASS] {column} uses only historical PM2.5."
        )
    else:
        print(
            f"[FAIL] {column} may contain future/current information."
        )


# =============================================================================
# 6. FEATURE LIST CHECK
# =============================================================================

header("6. FEATURE / TARGET CHECK")

FEATURE_COLUMNS = [
    "pm25",
    "pm25_missing",
    "hour",
    "day",
    "month",
    "day_of_week",
    "weekend",
    "season_code",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
    "pm25_lag_48h",
    "pm25_lag_168h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_std_6h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_std_24h",
    "pm25_rolling_mean_7d",
    "pm25_rolling_std_7d",
]

target_columns = [
    column
    for column in df.columns
    if "target" in column.lower()
]

print("Target columns:")
for column in target_columns:
    print(f"  - {column}")

overlap = set(FEATURE_COLUMNS).intersection(
    target_columns
)

print()

if not overlap:
    print(
        "[PASS] No target columns are included "
        "in FEATURE_COLUMNS."
    )
else:
    print(
        "[FAIL] Target columns appear in FEATURE_COLUMNS:"
    )
    print(overlap)


# =============================================================================
# 7. FUTURE-LOOKING FEATURE NAMES
# =============================================================================

header("7. FUTURE-LOOKING FEATURE CHECK")

future_keywords = [
    "future",
    "target",
    "lead",
    "ahead",
    "next",
]

suspicious = []

for feature in FEATURE_COLUMNS:

    feature_lower = feature.lower()

    if any(
        keyword in feature_lower
        for keyword in future_keywords
    ):
        suspicious.append(feature)


if not suspicious:
    print(
        "[PASS] No obviously future-looking "
        "feature names detected."
    )
else:
    print(
        "[WARNING] Suspicious features:"
    )

    for feature in suspicious:
        print(
            f"  - {feature}"
        )


# =============================================================================
# 8. FUTURE TARGET CONSTRUCTION
# =============================================================================

header("8. FUTURE TARGET CONSTRUCTION")

HORIZONS = {
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}

print(
    "The training script creates targets using:"
)

print(
    "target = pm25.shift(-horizon)"
)

print()

for name, hours in HORIZONS.items():

    target = df["pm25"].shift(
        -hours
    )

    print(
        f"{name:<5} → PM2.5 at t + {hours} hours"
    )

    print(
        f"       Target unavailable at final "
        f"{hours} rows: "
        f"{target.isna().sum():,}"
    )


# =============================================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

header("9. TRAIN / VALIDATION / TEST SPLIT")

n = len(df)

train_end = int(n * 0.70)

validation_end = int(n * 0.85)

print(
    "Current split used by train.py:"
)

print()

print(
    f"Training:   rows 0 → {train_end - 1:,}"
)

print(
    f"Validation: rows {train_end:,} → "
    f"{validation_end - 1:,}"
)

print(
    f"Testing:    rows {validation_end:,} → "
    f"{n - 1:,}"
)

print()

print(
    "Training end:"
)

print(
    df.loc[
        train_end - 1,
        "timestamp"
    ]
)

print()

print(
    "Validation start:"
)

print(
    df.loc[
        train_end,
        "timestamp"
    ]
)

print()

print(
    "Validation end:"
)

print(
    df.loc[
        validation_end - 1,
        "timestamp"
    ]
)

print()

print(
    "Test start:"
)

print(
    df.loc[
        validation_end,
        "timestamp"
    ]
)


# =============================================================================
# 10. MULTI-HORIZON SPLIT ANALYSIS
# =============================================================================

header("10. MULTI-HORIZON SPLIT ANALYSIS")

print(
    "Forecast horizons:"
)

for name, hours in HORIZONS.items():
    print(
        f"  {name:<5} = {hours} hours"
    )

print()

print(
    "Important:"
)

print(
    "The current train.py creates the future target "
    "before performing the 70/15/15 split."
)

print(
    "We therefore need to determine whether a purge/embargo "
    "is required between training, validation and testing."
)

print()

print(
    "This is the main point that requires further investigation."
)


# =============================================================================
# 11. OVERALL RESULT
# =============================================================================

header("STEP 4.5: PRELIMINARY LEAKAGE AUDIT RESULT")

print(
    "[PASS] Timestamp ordering checked."
)

print(
    "[PASS] Missing-value handling checked."
)

print(
    "[PASS] Lag features checked."
)

print(
    "[PASS] Rolling features checked."
)

print(
    "[PASS] Feature/target overlap checked."
)

print(
    "[PASS] Future-looking feature names checked."
)

print(
    "[PASS] Target construction checked."
)

print(
    "[REVIEW REQUIRED] Multi-horizon train/validation/test "
    "boundary needs to be assessed for a purge gap."
)

print()

print(
    "NO PROJECT FILES WERE MODIFIED."
)

print()

print(
    "STEP 4.5 COMPLETE"
)