from pathlib import Path

import pandas as pd

from src.train import prepare_training_data


def test_prepare_training_data_creates_training_dataset(tmp_path):
    engineered_file = tmp_path / "engineered_air_quality_data.csv"
    pd.DataFrame(
        [
            {"hour": 0, "day_of_week": 1, "month": 1, "rolling_mean_3": 10.0, "value": 12.0},
            {"hour": 1, "day_of_week": 1, "month": 1, "rolling_mean_3": 12.0, "value": 14.0},
            {"hour": 2, "day_of_week": 2, "month": 1, "rolling_mean_3": 14.0, "value": 16.0},
        ]
    ).to_csv(engineered_file, index=False)

    features, target = prepare_training_data(input_path=engineered_file, output_dir=tmp_path)

    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert not features.empty
    assert (tmp_path / "training_data.csv").exists()
