from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def preprocess_air_quality_data(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save_format: str = "csv",
) -> pd.DataFrame:
    """Load raw air quality data, clean it, and save a processed dataset."""

    if input_path is None:
        input_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "air_quality_data.csv"
    else:
        input_path = Path(input_path)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError("No data found in the raw dataset.")

    dataframe = dataframe.copy()
    dataframe.columns = [column.strip().lower() for column in dataframe.columns]

    if "date_utc" in dataframe.columns:
        dataframe["date_utc"] = pd.to_datetime(dataframe["date_utc"], errors="coerce", utc=True)
        dataframe = dataframe.sort_values("date_utc").reset_index(drop=True)

    for column in ["location", "city", "country", "parameter", "unit"]:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("string").fillna("unknown")

    if "value" in dataframe.columns:
        dataframe["value"] = pd.to_numeric(dataframe["value"], errors="coerce")
        dataframe = dataframe.dropna(subset=["value"])

    dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    if save_format.lower() == "json":
        output_file = output_dir / "processed_air_quality_data.json"
        dataframe.to_json(output_file, orient="records", indent=2)
    else:
        output_file = output_dir / "processed_air_quality_data.csv"
        dataframe.to_csv(output_file, index=False)

    return dataframe
