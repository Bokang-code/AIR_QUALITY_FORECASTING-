from pathlib import Path

import pandas as pd

from src.train import train_and_save_model


def test_train_and_save_model_creates_model_file(tmp_path):
    features = pd.DataFrame({"hour": [0, 1, 2], "day_of_week": [1, 1, 2], "month": [1, 1, 1], "rolling_mean_3": [10.0, 12.0, 14.0]})
    target = pd.Series([12.0, 14.0, 16.0])

    model_path = train_and_save_model(features, target, output_dir=tmp_path, model_name="test_model.joblib")

    assert Path(model_path).exists()
