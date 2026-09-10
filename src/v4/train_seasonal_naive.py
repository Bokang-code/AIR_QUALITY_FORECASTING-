from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import FORECAST_HORIZONS, TEST_START
except ImportError:  # pragma: no cover
    from src.v4.config import FORECAST_HORIZONS, TEST_START


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/v4/chengdu_features_v4.csv"
)

OUTPUT_DIR = Path(
    "reports/v4/seasonal_naive"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZONS = FORECAST_HORIZONS

# Seasonal periods in hours
DAILY_SEASON = 24
WEEKLY_SEASON = 168

STATIONS = [
    "1431A",
    "1432A",
    "1433A",
    "1434A",
    "1437A",
    "1438A",
    "2880A",
    "3136A",
    "3358A",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return np.nan, np.nan, np.nan

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (
            ss_res / ss_tot
        )

    return mae, rmse, r2


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_station_predictions(df, station_code, horizon_name, horizon_hours):
    station = df[df["station_code"] == station_code].copy().sort_values("timestamp")
    station = station.set_index("timestamp")
    pm25 = station["pm25"].astype(float)
    test_times = pm25.loc[pm25.index >= "2021-01-01"].index

    rows = []
    for target_time in test_times:
        reference_time = target_time - pd.Timedelta(hours=horizon_hours)
        actual = pm25.get(target_time, np.nan)
        prediction = pm25.get(reference_time, np.nan)

        if pd.isna(actual) or pd.isna(prediction):
            continue

        rows.append(
            {
                "station_code": station_code,
                "target_time": target_time,
                "horizon": horizon_name,
                "horizon_hours": horizon_hours,
                "reference_time": reference_time,
                "actual_pm25": actual,
                "seasonal_naive_prediction": prediction,
            }
        )

    return rows


def summarize_predictions(predictions_df):
    rows = []
    for horizon_name, horizon_hours in HORIZONS.items():
        subset = predictions_df[predictions_df["horizon"] == horizon_name]
        mae, rmse, r2 = calculate_metrics(subset["actual_pm25"], subset["seasonal_naive_prediction"])
        rows.append(
            {
                "horizon": horizon_name,
                "horizon_hours": horizon_hours,
                "n_predictions": len(subset),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def summarize_station_predictions(predictions_df):
    rows = []
    for station_code in STATIONS:
        for horizon_name, horizon_hours in HORIZONS.items():
            subset = predictions_df[(predictions_df["station_code"] == station_code) & (predictions_df["horizon"] == horizon_name)]
            if len(subset) == 0:
                continue
            mae, rmse, r2 = calculate_metrics(subset["actual_pm25"], subset["seasonal_naive_prediction"])
            rows.append(
                {
                    "station_code": station_code,
                    "horizon": horizon_name,
                    "horizon_hours": horizon_hours,
                    "n_predictions": len(subset),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEASONAL NAIVE BENCHMARK")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Dataset shape: {df.shape}")
    print(f"Stations: {df['station_code'].nunique()}")

    all_predictions = []
    for station_code in STATIONS:
        print("\n" + "=" * 70)
        print(f"STATION: {station_code}")
        print("=" * 70)

        station = df[df["station_code"] == station_code].copy()
        station = station.sort_values("timestamp")
        pm25 = station.set_index("timestamp")["pm25"].astype(float)
        test_times = pm25.loc[pm25.index >= "2021-01-01"].index
        print(f"Test observations: {len(test_times):,}")

        for horizon_name, horizon_hours in HORIZONS.items():
            print(f"  Horizon: {horizon_name}")
            all_predictions.extend(generate_station_predictions(df, station_code, horizon_name, horizon_hours))

    predictions_df = pd.DataFrame(all_predictions)
    predictions_file = OUTPUT_DIR / "seasonal_naive_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)

    overall_df = summarize_predictions(predictions_df)
    overall_file = OUTPUT_DIR / "seasonal_naive_results.csv"
    overall_df.to_csv(overall_file, index=False)

    station_df = summarize_station_predictions(predictions_df)
    station_file = OUTPUT_DIR / "seasonal_naive_station_results.csv"
    station_df.to_csv(station_file, index=False)

    print("\n" + "=" * 70)
    print("SEASONAL NAIVE RESULTS")
    print("=" * 70)
    print(overall_df[["horizon", "n_predictions", "mae", "rmse", "r2"]].round(4).to_string(index=False))
    print("\nSaved:")
    print(overall_file)
    print(station_file)
    print(predictions_file)
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()