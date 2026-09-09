from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


def load_model(model_path):
    """Load a trained XGBoost model."""
    return joblib.load(model_path)


def create_explainer(model):
    """Create a SHAP TreeExplainer for a tree-based model."""
    return shap.TreeExplainer(model)


def explain(model, X):
    """
    Generate SHAP values for one or more observations.

    Parameters
    ----------
    model : trained model
    X : pandas.DataFrame
        Model input features.

    Returns
    -------
    shap.Explanation
        SHAP explanation object.
    """
    explainer = create_explainer(model)
    return explainer(X)


def get_feature_importance(model, X):
    """
    Calculate global SHAP feature importance.

    Importance is calculated as the mean absolute SHAP
    value across the supplied observations.
    """
    shap_values = explain(model, X)

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        }
    )

    return (
        importance
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def get_local_explanation(model, X_row):
    """
    Explain one individual prediction.

    Returns the feature value and its SHAP contribution.
    Positive SHAP values push the prediction higher.
    Negative SHAP values push the prediction lower.
    """
    if len(X_row) != 1:
        raise ValueError(
            "X_row must contain exactly one observation."
        )

    shap_values = explain(model, X_row)

    explanation = pd.DataFrame(
        {
            "feature": X_row.columns,
            "value": X_row.iloc[0].values,
            "shap_value": shap_values.values[0],
        }
    )

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation["direction"] = np.where(
        explanation["shap_value"] > 0,
        "increases",
        "decreases",
    )

    return (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )


def get_top_drivers(model, X_row, n=5):
    """
    Return the strongest positive and negative drivers
    for one prediction.
    """
    explanation = get_local_explanation(
        model,
        X_row
    )

    positive = (
        explanation[
            explanation["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(n)
        .copy()
    )

    negative = (
        explanation[
            explanation["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(n)
        .copy()
    )

    return positive, negative


def humanize_feature_name(feature):
    """
    Convert technical model feature names into
    readable names for the application.
    """

    direct_names = {
        "pm25_lag_1h": "PM2.5 — previous hour",
        "pm25_lag_2h": "PM2.5 — 2 hours ago",
        "pm25_lag_3h": "PM2.5 — 3 hours ago",
        "pm25_lag_6h": "PM2.5 — 6 hours ago",
        "pm25_lag_12h": "PM2.5 — 12 hours ago",
        "pm25_lag_24h": "PM2.5 — 24 hours ago",
        "pm25_lag_48h": "PM2.5 — 48 hours ago",
        "pm25_lag_72h": "PM2.5 — 72 hours ago",
        "pm25_lag_168h": "PM2.5 — 7 days ago",
        "pm25_lag_336h": "PM2.5 — 14 days ago",
        "pm25_lag_720h": "PM2.5 — 30 days ago",

        "temperature": "Temperature",
        "relative_humidity": "Relative humidity",
        "wind_speed": "Wind speed",
        "wind_direction": "Wind direction",
        "boundary_layer_height": "Boundary-layer height",

        "pm10": "PM10",
        "so2": "SO₂",
        "no2": "NO₂",
        "o3_8h": "O₃",
        "co": "CO",

        "hour": "Hour of day",
        "day_of_week": "Day of week",
        "day_of_month": "Day of month",
        "day_of_year": "Day of year",
        "week_of_year": "Week of year",
        "month": "Month",
        "quarter": "Quarter",
        "year": "Year",
        "is_weekend": "Weekend",

        "hour_sin": "Time of day",
        "hour_cos": "Time of day",
        "day_of_week_sin": "Day of week",
        "day_of_week_cos": "Day of week",
        "month_sin": "Seasonality",
        "month_cos": "Seasonality",
        "day_of_year_sin": "Seasonality",
        "day_of_year_cos": "Seasonality",
    }

    if feature in direct_names:
        return direct_names[feature]

    # Rolling means
    if "_rolling_mean_" in feature:
        variable, window = feature.split(
            "_rolling_mean_"
        )

        variable_names = {
            "pm25": "PM2.5",
            "temperature": "Temperature",
            "relative_humidity": "Relative humidity",
            "u10": "U-component wind",
            "v10": "V-component wind",
            "wind_speed": "Wind speed",
            "boundary_layer_height":
                "Boundary-layer height",
            "pm10": "PM10",
            "so2": "SO₂",
            "no2": "NO₂",
            "o3_8h": "O₃",
            "co": "CO",
        }

        variable = variable_names.get(
            variable,
            variable
        )

        window = window.replace(
            "h",
            " hours"
        )

        return (
            f"{variable} — "
            f"{window} rolling average"
        )

    # Rolling standard deviation
    if "_rolling_std_" in feature:
        variable, window = feature.split(
            "_rolling_std_"
        )

        variable = {
            "pm25": "PM2.5"
        }.get(
            variable,
            variable
        )

        window = window.replace(
            "h",
            " hours"
        )

        return (
            f"{variable} — "
            f"{window} variability"
        )

    # Lagged features
    if "_lag_" in feature:
        variable, lag = feature.rsplit(
            "_lag_",
            1
        )

        variable_names = {
            "pm25": "PM2.5",
            "temperature": "Temperature",
            "relative_humidity": "Relative humidity",
            "u10": "U-component wind",
            "v10": "V-component wind",
            "boundary_layer_height":
                "Boundary-layer height",
            "pm10": "PM10",
            "so2": "SO₂",
            "no2": "NO₂",
            "o3_8h": "O₃",
            "co": "CO",
        }

        variable = variable_names.get(
            variable,
            variable
        )

        lag = lag.replace(
            "h",
            " hours"
        )

        return (
            f"{variable} — "
            f"{lag} ago"
        )

    return (
        feature
        .replace("_", " ")
        .title()
    )


def prepare_local_explanation(model, X_row):
    """
    Create a dashboard-ready explanation
    for one prediction.
    """
    explanation = get_local_explanation(
        model,
        X_row
    )

    explanation[
        "display_feature"
    ] = explanation[
        "feature"
    ].apply(
        humanize_feature_name
    )

    explanation["impact"] = np.where(
        explanation["shap_value"] > 0,
        "↑",
        "↓",
    )

    explanation[
        "impact_text"
    ] = np.where(
        explanation["shap_value"] > 0,
        "Pushes PM2.5 higher",
        "Pushes PM2.5 lower",
    )

    return explanation


def save_feature_importance(
    model,
    X,
    output_path
):
    """Save global SHAP importance to CSV."""
    importance = get_feature_importance(
        model,
        X
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    importance.to_csv(
        output_path,
        index=False
    )

    return importance