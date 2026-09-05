import pandas as pd
from scipy.stats import skew

# Load hourly PM2.5 data
df = pd.read_csv(
    "data/processed/sensor_218_hourly.csv"
)

# Convert PM2.5 to numeric
df["pm25"] = pd.to_numeric(
    df["pm25"],
    errors="coerce"
)

# Remove missing values for distribution analysis
x = df["pm25"].dropna()

print("=" * 80)
print("STEP 2: PM2.5 TARGET DISTRIBUTION")
print("=" * 80)

print(f"\nCount:       {len(x):,}")
print(f"Missing:     {df['pm25'].isna().sum():,}")
print(f"Mean:        {x.mean():.4f}")
print(f"Median:      {x.median():.4f}")
print(f"Std:         {x.std():.4f}")
print(f"Min:         {x.min():.4f}")
print(f"Max:         {x.max():.4f}")
print(f"Skewness:    {skew(x):.4f}")

print("\n" + "=" * 80)
print("PERCENTILES")
print("=" * 80)

percentiles = x.quantile([
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00
])

print(percentiles.to_string())

# IQR outlier analysis
q1 = x.quantile(0.25)
q3 = x.quantile(0.75)

iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = (
    (x < lower_fence) |
    (x > upper_fence)
)

print("\n" + "=" * 80)
print("OUTLIER ANALYSIS")
print("=" * 80)

print(f"\nQ1:           {q1:.4f}")
print(f"Q3:           {q3:.4f}")
print(f"IQR:          {iqr:.4f}")
print(f"Lower fence:  {lower_fence:.4f}")
print(f"Upper fence:  {upper_fence:.4f}")

print(
    f"IQR outliers: {outliers.sum():,} "
    f"({outliers.mean() * 100:.2f}%)"
)

print("\nLargest PM2.5 values:")
print(
    x.nlargest(20).to_string(index=False)
)

print("\nSmallest PM2.5 values:")
print(
    x.nsmallest(20).to_string(index=False)
)

print("\n" + "=" * 80)
print("STEP 2 COMPLETE")
print("=" * 80)