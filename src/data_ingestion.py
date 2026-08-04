from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - graceful fallback for minimal environments
    def load_dotenv() -> bool:
        return False


load_dotenv()


def ingest_air_quality_data(
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    save_format: str = "csv",
    endpoint: str = "/v3/parameters/2/latest",
    base_url: str = "https://api.openaq.org",
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch air quality data from the OpenAQ API and persist it locally.

    Parameters
    ----------
    output_dir:
        Directory where the downloaded files will be stored. Defaults to the
        repository's data/raw folder.
    api_key:
        Optional OpenAQ API key. If omitted, the module will fall back to the
        OPENAQ_API_KEY environment variable.
    save_format:
        Output file format, either "csv" or "json".
    endpoint:
        OpenAQ API endpoint to request.
    base_url:
        Base URL for the OpenAQ API.
    params:
        Optional query parameters for the API request.
    """

    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "data" / "raw"
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_api_key = api_key or os.getenv("OPENAQ_API_KEY")
    headers = {"Accept": "application/json"}
    if resolved_api_key:
        headers["X-API-Key"] = resolved_api_key

    response = requests.get(
        f"{base_url.rstrip('/')}{endpoint}",
        headers=headers,
        params=params or {},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Unexpected API response format. Expected a JSON object.")

    results = payload.get("results") or payload.get("data") or []
    if not isinstance(results, list):
        raise ValueError("Unexpected API response format. Expected a list of results.")

    parameter_map: dict[str, dict[str, str]] = {
        "1": {"name": "pm10", "unit": "µg/m³"},
        "2": {"name": "pm25", "unit": "µg/m³"},
    }

    def infer_endpoint_parameter() -> tuple[str | None, str | None]:
        match = re.search(r"/v3/parameters/(?P<param_id>\d+)/latest", endpoint)
        if not match:
            return None, None
        param_id = match.group("param_id")
        mapping = parameter_map.get(param_id)
        if mapping:
            return mapping["name"], mapping["unit"]
        return None, None

    inferred_parameter, inferred_unit = infer_endpoint_parameter()

    def extract_parameter(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("name")
        if value is None:
            return inferred_parameter
        return value

    def extract_unit(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("units")
        if value is None:
            return inferred_unit
        return value

    def extract_datetime(value: Any) -> Any:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("utc") or value.get("local")
        return None

    rows: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        latitude = None
        longitude = None
        if isinstance(item.get("coordinates"), dict):
            latitude = item["coordinates"].get("latitude")
            longitude = item["coordinates"].get("longitude")

        base_record = {
            "location": item.get("location"),
            "city": item.get("city"),
            "country": item.get("country"),
            "latitude": latitude,
            "longitude": longitude,
        }

        measurements = item.get("measurements")
        if isinstance(measurements, list) and measurements:
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    continue

                date_info = measurement.get("date") or {}
                rows.append(
                    {
                        **base_record,
                        "parameter": extract_parameter(measurement.get("parameter")),
                        "value": measurement.get("value"),
                        "unit": extract_unit(measurement.get("unit")),
                        "date_utc": extract_datetime(date_info),
                        "raw_measurement": json.dumps(measurement, ensure_ascii=False),
                    }
                )
            continue

        if item.get("value") is not None:
            rows.append(
                {
                    **base_record,
                    "parameter": extract_parameter(item.get("parameter")),
                    "value": item.get("value"),
                    "unit": extract_unit(item.get("unit") or item.get("parameter")),
                    "date_utc": extract_datetime(item.get("datetime") or item.get("date") or item.get("date_utc")),
                    "raw_measurement": json.dumps(item, ensure_ascii=False),
                }
            )
            continue

        rows.append({**base_record, "parameter": None, "value": None, "unit": None, "date_utc": None})

    dataframe = pd.DataFrame(rows)

    if save_format.lower() == "json":
        output_file = output_path / "air_quality_data.json"
        dataframe.to_json(output_file, orient="records", indent=2)
    else:
        output_file = output_path / "air_quality_data.csv"
        dataframe.to_csv(output_file, index=False)

    return dataframe
