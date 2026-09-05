import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.evaluate import (
    evaluate_regression,
    load_evaluation_report,
    load_model,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
)


def test_evaluate_regression_saves_metrics(tmp_path):
    y_true = pd.Series([10.0, 12.0, 14.0])
    y_pred = pd.Series([9.0, 11.0, 15.0])

    metrics = evaluate_regression(
        y_true, y_pred, output_dir=tmp_path, file_name="metrics.json"
    )

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["r2"] == pytest.approx(0.625, rel=1e-6)
    assert (tmp_path / "metrics.json").exists()


def test_load_evaluation_report(tmp_path):
    report = {"mae": 1.0, "rmse": 1.0, "r2": 0.625, "mape": 8.0}
    report_path = tmp_path / "metrics.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded_report = load_evaluation_report(report_path)

    assert loaded_report == report


def test_plot_generation(tmp_path):
    y_true = pd.Series([10.0, 12.0, 14.0])
    y_pred = pd.Series([9.0, 11.0, 15.0])

    actual_vs_predicted_path = plot_actual_vs_predicted(y_true, y_pred, output_dir=tmp_path)
    residuals_path = plot_residuals(y_true, y_pred, output_dir=tmp_path)

    assert actual_vs_predicted_path.exists()
    assert residuals_path.exists()


def test_plot_feature_importance(tmp_path):
    X = pd.DataFrame(
        {
            "hour": [0, 1, 2],
            "day_of_week": [1, 1, 2],
            "month": [1, 1, 1],
            "rolling_mean_3": [10.0, 12.0, 14.0],
        }
    )
    y = pd.Series([10.0, 12.0, 14.0])
    model = RandomForestRegressor(n_estimators=1, random_state=42)
    model.fit(X, y)

    output_path = plot_feature_importance(model, list(X.columns), output_dir=tmp_path)

    assert output_path is not None
    assert output_path.exists()


def test_load_model(tmp_path):
    X = pd.DataFrame({"hour": [0, 1], "day_of_week": [1, 2], "month": [1, 1], "rolling_mean_3": [10.0, 12.0]})
    y = pd.Series([10.0, 11.0])
    model = RandomForestRegressor(n_estimators=1, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    loaded_model = load_model(model_path)
    assert hasattr(loaded_model, "predict")
    assert list(loaded_model.predict(X)) == list(model.predict(X))


def test_invalid_input_raises_errors():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_regression([], [])

    with pytest.raises(ValueError, match="must have the same length"):
        evaluate_regression([1.0, 2.0], [1.0])
