from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def engineer_features(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save_format: str = "csv",
) -> pd.DataFrame:
    """Create simple time-based features from the processed air quality dataset."""

    if input_path is None:
        input_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_air_quality_data.csv"
    else:
        input_path = Path(input_path)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    if dataframe.empty:
        raise ValueError("No data found in the processed dataset.")

    dataframe = dataframe.copy()

    if "date_utc" in dataframe.columns:
        dataframe["date_utc"] = pd.to_datetime(dataframe["date_utc"], errors="coerce", utc=True)
        dataframe = dataframe.sort_values("date_utc").reset_index(drop=True)

    if "value" in dataframe.columns:
        dataframe["value"] = pd.to_numeric(dataframe["value"], errors="coerce")

    if "date_utc" in dataframe.columns:
        dataframe["hour"] = dataframe["date_utc"].dt.hour
        dataframe["day_of_week"] = dataframe["date_utc"].dt.dayofweek
        dataframe["month"] = dataframe["date_utc"].dt.month

    if "value" in dataframe.columns:
        dataframe["rolling_mean_3"] = dataframe["value"].rolling(window=3, min_periods=1).mean()

    dataframe = dataframe.reset_index(drop=True)

    if save_format.lower() == "json":
        output_file = output_dir / "engineered_air_quality_data.json"
        dataframe.to_json(output_file, orient="records", indent=2)
    else:
        output_file = output_dir / "engineered_air_quality_data.csv"
        dataframe.to_csv(output_file, index=False)

    return dataframe
