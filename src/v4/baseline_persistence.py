from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from config import FORECAST_HORIZONS, TRAIN_END, VALIDATION_END
except ImportError:  # pragma: no cover
    from src.v4.config import FORECAST_HORIZONS, TRAIN_END, VALIDATION_END


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "chengdu_features_v4.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "v4"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "persistence_results.csv"
)

STATION_FILE = (
    REPORT_DIR
    / "persistence_results_by_station.csv"
)


# ============================================================
# FORECAST HORIZONS / DATA SPLIT
# ============================================================

HORIZONS = FORECAST_HORIZONS


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_persistence_subset(df, target_column):
    subset = df[["timestamp", "station_code", "pm25", target_column]].copy()
    subset = subset.rename(columns={target_column: "actual", "pm25": "prediction"})
    return subset.dropna(subset=["prediction", "actual"]).copy()


def create_time_splits(subset):
    train = subset[subset["timestamp"] < TRAIN_END].copy()
    validation = subset[(subset["timestamp"] >= TRAIN_END) & (subset["timestamp"] < VALIDATION_END)].copy()
    test = subset[subset["timestamp"] >= VALIDATION_END].copy()
    return train, validation, test


def append_station_metrics(station_results, horizon_name, horizon_hours, test_df):
    for station, station_test in test_df.groupby("station_code"):
        if len(station_test) < 2:
            continue

        station_metrics = calculate_metrics(station_test["actual"], station_test["prediction"])
        station_results.append(
            {
                "horizon": horizon_name,
                "horizon_hours": horizon_hours,
                "station_code": station,
                "station_name": station_test["station_code"].iloc[0],
                "test_samples": len(station_test),
                "mae": station_metrics["mae"],
                "rmse": station_metrics["rmse"],
                "r2": station_metrics["r2"],
            }
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V4 PERSISTENCE BASELINE")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find:\n{INPUT_FILE}")

    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    print(f"Rows loaded: {len(df):,}")

    df = df.sort_values(["station_code", "timestamp"]).reset_index(drop=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overall_results = []
    station_results = []

    for horizon_name, horizon_hours in HORIZONS.items():
        print(f"\n{'-' * 70}")
        print(f"HORIZON: {horizon_name}")
        print(f"{'-' * 70}")

        target_column = f"target_pm25_{horizon_name}"
        subset = prepare_persistence_subset(df, target_column)
        train, validation, test = create_time_splits(subset)

        print(f"Train samples:      {len(train):,}")
        print(f"Validation samples: {len(validation):,}")
        print(f"Test samples:       {len(test):,}")

        metrics = calculate_metrics(test["actual"], test["prediction"])
        print(f"\nTest MAE:  {metrics['mae']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"Test R²:   {metrics['r2']:.4f}")

        overall_results.append(
            {
                "horizon": horizon_name,
                "horizon_hours": horizon_hours,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "test_samples": len(test),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )

        append_station_metrics(station_results, horizon_name, horizon_hours, test)

    results_df = pd.DataFrame(overall_results)
    station_results_df = pd.DataFrame(station_results)
    results_df.to_csv(SUMMARY_FILE, index=False)
    station_results_df.to_csv(STATION_FILE, index=False)

    print("\n" + "=" * 70)
    print("PERSISTENCE BASELINE COMPLETE")
    print("=" * 70)
    print("\nOverall results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved:")
    print(SUMMARY_FILE)
    print(STATION_FILE)
    print("\nDone.")


if __name__ == "__main__":
    main()