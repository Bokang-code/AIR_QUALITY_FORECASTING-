import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_ingestion import ingest_air_quality_data


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_ingest_air_quality_data_creates_csv_and_dataframe(tmp_path, monkeypatch):
    sample_payload = {
        "results": [
            {
                "location": "Test Station",
                "city": "Test City",
                "country": "US",
                "coordinates": {"latitude": 40.0, "longitude": -74.0},
                "measurements": [
                    {"parameter": "pm25", "value": 12.5, "unit": "µg/m³", "date": {"utc": "2024-01-01T00:00:00Z"}}
                ],
            }
        ]
    }

    monkeypatch.setattr("src.data_ingestion.requests.get", lambda *args, **kwargs: DummyResponse(sample_payload))

    result = ingest_air_quality_data(output_dir=tmp_path, api_key="test-key", save_format="csv")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "city" in result.columns
    assert (tmp_path / "air_quality_data.csv").exists()
