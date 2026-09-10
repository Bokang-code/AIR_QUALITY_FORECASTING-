from pathlib import Path

import pandas as pd

# ============================================================
# Shared V4 Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "v4"

FORECAST_HORIZONS = {
    "24h": 24,
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}

PM25_LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
    336,
    720,
]

PM25_ROLLING_WINDOWS = [
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

WEATHER_VARIABLES = [
    "temperature",
    "relative_humidity",
    "u10",
    "v10",
    "boundary_layer_height",
]

WEATHER_LAGS = [
    1,
    3,
    6,
    12,
    24,
]

WEATHER_ROLLING_WINDOWS = [
    6,
    12,
    24,
    48,
    72,
]

CO_POLLUTANTS = [
    "pm10",
    "so2",
    "no2",
    "o3_8h",
    "co",
]

TRAIN_END = pd.Timestamp("2020-01-01")
VALIDATION_END = pd.Timestamp("2021-01-01")
TEST_START = pd.Timestamp("2021-01-01")
