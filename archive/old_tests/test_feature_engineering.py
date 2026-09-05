import pandas as pd

from src.feature_engineering import (
    create_time_features,
    create_cyclical_features,
    create_lag_features,
    create_rolling_features,
    create_target,
)


def make_test_data():
    """Create a small hourly PM2.5 dataset for testing."""

    timestamps = pd.date_range(
        start="2016-02-09 04:00:00",
        periods=200,
        freq="h",
        tz="UTC",
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pm25": range(200),
            "pm25_missing": [False] * 200,
        }
    )

    return df


def test_time_features():
    df = make_test_data()
    df = create_time_features(df)

    required_columns = [
        "hour",
        "day",
        "month",
        "day_of_week",
        "weekend",
        "season",
        "season_code",
    ]

    for column in required_columns:
        assert column in df.columns

    assert df["hour"].between(0, 23).all()
    assert df["day"].between(1, 31).all()
    assert df["month"].between(1, 12).all()
    assert df["day_of_week"].between(0, 6).all()


def test_cyclical_features():
    df = make_test_data()

    df = create_time_features(df)
    df = create_cyclical_features(df)

    required_columns = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]

    for column in required_columns:
        assert column in df.columns

    for column in required_columns:
        assert df[column].between(-1, 1).all()


def test_lag_features():
    df = make_test_data()
    df = create_lag_features(df)

    required_columns = [
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm25_lag_6h",
        "pm25_lag_12h",
        "pm25_lag_24h",
        "pm25_lag_48h",
        "pm25_lag_168h",
    ]

    for column in required_columns:
        assert column in df.columns

    assert pd.isna(df.loc[0, "pm25_lag_1h"])
    assert df.loc[1, "pm25_lag_1h"] == df.loc[0, "pm25"]


def test_rolling_features():
    df = make_test_data()
    df = create_rolling_features(df)

    required_columns = [
        "pm25_rolling_mean_6h",
        "pm25_rolling_std_6h",
        "pm25_rolling_mean_24h",
        "pm25_rolling_std_24h",
        "pm25_rolling_mean_7d",
        "pm25_rolling_std_7d",
    ]

    for column in required_columns:
        assert column in df.columns


def test_forecasting_target():
    df = make_test_data()
    df = create_target(df)

    assert "target_pm25" in df.columns
    assert df.loc[0, "target_pm25"] == df.loc[1, "pm25"]
    assert pd.isna(df.loc[len(df) - 1, "target_pm25"])


def test_feature_engineering_pipeline():
    df = make_test_data()

    df = create_time_features(df)
    df = create_cyclical_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_target(df)

    expected_columns = [
        "timestamp",
        "pm25",
        "pm25_missing",
        "hour",
        "day",
        "month",
        "day_of_week",
        "weekend",
        "season",
        "season_code",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm25_lag_6h",
        "pm25_lag_12h",
        "pm25_lag_24h",
        "pm25_lag_48h",
        "pm25_lag_168h",
        "pm25_rolling_mean_6h",
        "pm25_rolling_std_6h",
        "pm25_rolling_mean_24h",
        "pm25_rolling_std_24h",
        "pm25_rolling_mean_7d",
        "pm25_rolling_std_7d",
        "target_pm25",
    ]

    for column in expected_columns:
        assert column in df.columns

    assert len(df) == 200
    assert df["timestamp"].is_monotonic_increasing