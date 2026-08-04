from pathlib import Path

import pandas as pd

from src.feature_engineering import engineer_features


def test_engineer_features_creates_output(tmp_path):
    processed_file = tmp_path / "processed_air_quality_data.csv"
    pd.DataFrame(
        [
            {"location": "Test Station", "city": "Test City", "parameter": "pm25", "value": 10.0, "date_utc": "2024-01-01T00:00:00Z"},
            {"location": "Test Station", "city": "Test City", "parameter": "pm25", "value": 12.0, "date_utc": "2024-01-01T01:00:00Z"},
            {"location": "Test Station", "city": "Test City", "parameter": "pm25", "value": 14.0, "date_utc": "2024-01-01T02:00:00Z"},
        ]
    ).to_csv(processed_file, index=False)

    result = engineer_features(input_path=processed_file, output_dir=tmp_path, save_format="csv")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "hour" in result.columns
    assert "rolling_mean_3" in result.columns
    assert (tmp_path / "engineered_air_quality_data.csv").exists()
