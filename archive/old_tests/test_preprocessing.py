from pathlib import Path

import pandas as pd

from src.preprocessing import preprocess_air_quality_data


def test_preprocess_air_quality_data_creates_processed_file(tmp_path):
    raw_file = tmp_path / "air_quality_data.csv"
    pd.DataFrame(
        [
            {
                "location": "Test Station",
                "city": "Test City",
                "country": "US",
                "parameter": "pm25",
                "value": 12.5,
                "unit": "µg/m³",
                "date_utc": "2024-01-01T00:00:00Z",
            },
            {
                "location": "Test Station",
                "city": "Test City",
                "country": "US",
                "parameter": "pm25",
                "value": 12.5,
                "unit": "µg/m³",
                "date_utc": "2024-01-01T00:00:00Z",
            },
        ]
    ).to_csv(raw_file, index=False)

    result = preprocess_air_quality_data(input_path=raw_file, output_dir=tmp_path, save_format="csv")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert (tmp_path / "processed_air_quality_data.csv").exists()
