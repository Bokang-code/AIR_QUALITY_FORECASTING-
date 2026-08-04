import joblib
import pandas as pd

from src.predict import load_model, predict_health_risk, predict_values


class DummyModel:
    def predict(self, features):
        return [10.0 for _ in range(len(features))]


def test_predict_functions(tmp_path):
    dummy_model = DummyModel()
    features = pd.DataFrame({"hour": [0, 1, 2], "day_of_week": [1, 1, 2], "month": [1, 1, 1], "rolling_mean_3": [10.0, 12.0, 14.0]})

    predictions = predict_values(dummy_model, features)
    assert all(predictions == 10.0)

    risks = predict_health_risk(predictions)
    assert all(risks == "Low")

    dummy_model_path = tmp_path / "dummy_model.joblib"
    joblib.dump(dummy_model, dummy_model_path)

    loaded_model = load_model(dummy_model_path)
    assert hasattr(loaded_model, "predict")
    assert list(loaded_model.predict(features)) == [10.0, 10.0, 10.0]
