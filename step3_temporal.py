import pandas as pd
import numpy as np


# =============================================================================
# STEP 3: TEMPORAL PREDICTABILITY ANALYSIS
# =============================================================================

INPUT_FILE = "data/processed/sensor_218_hourly.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("STEP 3: PM2.5 TEMPORAL PREDICTABILITY")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce"
)

df["pm25"] = pd.to_numeric(
    df["pm25"],
    errors="coerce"
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

print(f"\nRows loaded: {len(df):,}")
print(f"Missing PM2.5: {df['pm25'].isna().sum():,}")


# =============================================================================
# AUTOCORRELATION
# =============================================================================

print("\n" + "=" * 80)
print("AUTOCORRELATION")
print("=" * 80)

print(
    "\nHow strongly PM2.5 at time t is related to PM2.5 "
    "at previous time steps."
)

lags = [
    1,
    3,
    6,
    12,
    24,
    48,
    72,
    168
]

print()
print(f"{'LAG':<12}{'HOURS':<12}{'AUTOCORRELATION':<20}")
print("-" * 44)

for lag in lags:

    correlation = df["pm25"].autocorr(
        lag=lag
    )

    print(
        f"{lag:<12}"
        f"{lag:<12}"
        f"{correlation:<20.4f}"
    )


# =============================================================================
# FUTURE CORRELATION
# =============================================================================

print("\n" + "=" * 80)
print("PAST PM2.5 VS FUTURE PM2.5")
print("=" * 80)

print(
    "\nCorrelation between PM2.5 at time t and "
    "PM2.5 at future horizons."
)

horizons = {
    "1h": 1,
    "3h": 3,
    "6h": 6,
    "12h": 12,
    "24h": 24,
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}

print()
print(f"{'HORIZON':<12}{'CORRELATION':<20}")
print("-" * 32)

for name, horizon in horizons.items():

    future_pm25 = df["pm25"].shift(
        -horizon
    )

    valid = pd.concat(
        [
            df["pm25"],
            future_pm25
        ],
        axis=1
    ).dropna()

    correlation = valid.iloc[:, 0].corr(
        valid.iloc[:, 1]
    )

    print(
        f"{name:<12}"
        f"{correlation:<20.4f}"
    )


# =============================================================================
# MEAN / STANDARD DEVIATION BY HOUR
# =============================================================================

print("\n" + "=" * 80)
print("HOURLY PM2.5 PATTERN")
print("=" * 80)

hourly = (
    df.groupby("hour" if "hour" in df.columns else df["timestamp"].dt.hour)
)

# Recalculate hour directly to avoid depending on engineered data
df["hour_analysis"] = df["timestamp"].dt.hour

hourly_stats = (
    df.groupby("hour_analysis")["pm25"]
    .agg(["mean", "median", "std", "count"])
)

print(
    hourly_stats.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# =============================================================================
# DAILY PATTERN
# =============================================================================

print("\n" + "=" * 80)
print("DAY-OF-WEEK PM2.5 PATTERN")
print("=" * 80)

df["day_of_week_analysis"] = (
    df["timestamp"].dt.dayofweek
)

day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

daily_stats = (
    df.groupby("day_of_week_analysis")["pm25"]
    .agg(["mean", "median", "std", "count"])
)

daily_stats.index = [
    day_names[i]
    for i in daily_stats.index
]

print(
    daily_stats.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# =============================================================================
# MONTHLY PATTERN
# =============================================================================

print("\n" + "=" * 80)
print("MONTHLY PM2.5 PATTERN")
print("=" * 80)

df["month_analysis"] = (
    df["timestamp"].dt.month
)

monthly_stats = (
    df.groupby("month_analysis")["pm25"]
    .agg(["mean", "median", "std", "count"])
)

print(
    monthly_stats.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# =============================================================================
# VARIABILITY
# =============================================================================

print("\n" + "=" * 80)
print("TEMPORAL VARIABILITY")
print("=" * 80)

valid_pm25 = df["pm25"].dropna()

print(
    f"\nOverall mean:       {valid_pm25.mean():.4f}"
)

print(
    f"Overall std:        {valid_pm25.std():.4f}"
)

print(
    f"Coefficient of variation: "
    f"{valid_pm25.std() / valid_pm25.mean():.4f}"
)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TEMPORAL PREDICTABILITY SUMMARY")
print("=" * 80)

print(
    """
Interpretation guide:

Autocorrelation:
  > 0.70  = strong temporal persistence
  0.40-0.70 = moderate temporal persistence
  0.20-0.40 = weak/moderate persistence
  < 0.20  = weak temporal persistence

Future correlation:
  Higher correlation means past PM2.5 contains useful
  information about future PM2.5.

The important horizons for this project are:

  48h
  72h
  168h (7d)
  336h (14d)
  720h (30d)

We will use these results to determine whether the
negative R² values are primarily caused by:

  1. Weak temporal predictability,
  2. Insufficient forecasting features,
  3. Model limitations,
  4. Or some combination of these.
"""
)

print("\n" + "=" * 80)
print("STEP 3 COMPLETE")
print("=" * 80)