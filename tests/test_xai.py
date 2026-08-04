import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.xai import calculate_shap_values, plot_shap_dependence, plot_shap_summary


def test_shap_explainability(tmp_path):
    X = pd.DataFrame(
        {
            "hour": [0, 1, 2, 3],
            "day_of_week": [1, 1, 2, 2],
            "month": [1, 1, 1, 1],
            "rolling_mean_3": [10.0, 12.0, 14.0, 16.0],
        }
    )
    y = pd.Series([10.0, 12.0, 14.0, 16.0])
    model = RandomForestRegressor(n_estimators=1, random_state=42)
    model.fit(X, y)

    shap_values, feature_names = calculate_shap_values(model, X)
    assert shap_values is not None
    assert feature_names == list(X.columns)

    summary_path = plot_shap_summary(shap_values, feature_names, output_dir=tmp_path)
    assert summary_path.exists()

    dependence_path = plot_shap_dependence(shap_values, "hour", X, output_dir=tmp_path)
    assert dependence_path.exists()


def test_shap_invalid_feature_raises(tmp_path):
    X = pd.DataFrame(
        {
            "hour": [0, 1],
            "day_of_week": [1, 2],
            "month": [1, 1],
            "rolling_mean_3": [10.0, 12.0],
        }
    )
    y = pd.Series([10.0, 11.0])
    model = RandomForestRegressor(n_estimators=1, random_state=42)
    model.fit(X, y)

    shap_values, feature_names = calculate_shap_values(model, X)

    with pytest.raises(ValueError, match="Feature 'invalid_feature' was not found"):
        plot_shap_dependence(shap_values, "invalid_feature", X, output_dir=tmp_path)
