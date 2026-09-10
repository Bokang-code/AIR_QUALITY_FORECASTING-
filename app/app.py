from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from src/
sys.path.insert(0, str(PROJECT_ROOT))

from src.xai import (
    load_model,
    prepare_local_explanation,
)

from src.health_risk import classify_pm25
from app.dashboard_helpers import (
    HORIZONS,
    STATIONS,
    STATION_COORDINATES,
    build_history_figure,
    build_map_figure,
    build_outlook_figure,
    build_station_figure,
    get_map_legend_html,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "chengdu_features_v4.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "v4"
    / "temporal_meteorology_copollutants"
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AIR / Intelligence",
    page_icon="◉",
    layout="wide",
)


# --------------------------------------------------
# Premium visual styling
# --------------------------------------------------

style_path = PROJECT_ROOT / "styles" / "style.css"

st.markdown(
    f"<style>{style_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Data
# --------------------------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH
    )

    metadata_columns = [
        "timestamp",
        "station_code",
        "station_name",
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in metadata_columns
    ]

    data[feature_columns] = (
        data[feature_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    return data


# --------------------------------------------------
# Model
# --------------------------------------------------

@st.cache_resource
def load_forecast_model(
    model_suffix
):

    model_path = (
        MODEL_DIR
        / (
            "xgboost_temporal_meteorology_copollutants_"
            f"{model_suffix}_purged.joblib"
        )
    )

    return load_model(
        model_path
    )


df = load_data()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="air-eyebrow">
        Environmental intelligence platform
    </div>

    <div class="air-title">
        AIR /<br>
        INTELLIGENCE
    </div>

    <div class="air-subtitle">
        Chengdu · PM2.5 forecasting · atmospheric analysis
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Controls
# --------------------------------------------------

st.markdown(
    "<br>",
    unsafe_allow_html=True,
)


control_col1, control_col2, control_col3 = (
    st.columns(3)
)


with control_col1:

    station_label = st.selectbox(
        "Monitoring station",
        list(STATIONS.keys()),
    )


with control_col2:

    horizon_label = st.selectbox(
        "Forecast horizon",
        list(HORIZONS.keys()),
    )


station_code = STATIONS[
    station_label
]

model_suffix, horizon = HORIZONS[
    horizon_label
]


# --------------------------------------------------
# Station data
# --------------------------------------------------

station_df = (
    df[
        df["station_code"]
        == station_code
    ]
    .sort_values(
        "timestamp"
    )
    .copy()
)


if station_df.empty:

    st.error(
        "No data is available for this "
        "monitoring station."
    )

    st.stop()


usable = (
    station_df
    .dropna(
        subset=["pm25"]
    )
    .copy()
)


if usable.empty:

    st.error(
        "No usable PM2.5 observations are "
        "available for this monitoring station."
    )

    st.stop()


# --------------------------------------------------
# Forecast origin
# --------------------------------------------------


available_dates = (
    usable["timestamp"]
    .dt.date
    .drop_duplicates()
    .tolist()
)


selected_date = st.date_input(
    "Select historical forecast date",
    value=available_dates[-1],
    min_value=available_dates[0],
    max_value=available_dates[-1],
)


date_rows = usable[
    usable["timestamp"].dt.date
    == selected_date
].copy()


if date_rows.empty:

    st.warning(
        "No PM2.5 observation is available "
        "on this date. Please select another date."
    )

    st.stop()


# Use the latest available observation
# on the selected day.
latest = date_rows.iloc[-1]


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = load_forecast_model(
    model_suffix
)


model_features = (
    model
    .get_booster()
    .feature_names
)


# --------------------------------------------------
# Prepare model input
# --------------------------------------------------

X_latest = (
    latest[model_features]
    .to_frame()
    .T
)


X_latest = X_latest.apply(
    pd.to_numeric,
    errors="coerce",
)


# --------------------------------------------------
# Forecast
# --------------------------------------------------

prediction = float(
    model.predict(
        X_latest
    )[0]
)


risk = classify_pm25(
    prediction
)


# --------------------------------------------------
# XAI
# --------------------------------------------------

local_explanation = (
    prepare_local_explanation(
        model,
        X_latest,
    )
)


# Get strongest SHAP drivers.
top_drivers = (
    local_explanation
    .assign(
        absolute_shap=lambda x:
            x["shap_value"].abs()
    )
    .sort_values(
        "absolute_shap",
        ascending=False,
    )
    .head(8)
)


# --------------------------------------------------
# Main forecast hero
# --------------------------------------------------

st.html(
    f"""
    <div class="hero-card">

        <div class="hero-label">
            Predicted PM2.5 · {horizon_label}
        </div>

        <div>
            <span class="hero-number">
                {prediction:.1f}
            </span>

            <span class="hero-unit">
                µg/m³
            </span>
        </div>

        <div class="hero-status">
            {risk["short_label"]}
        </div>

    </div>
    """
)


# --------------------------------------------------
# Main metrics
# --------------------------------------------------

st.divider()


metric1, metric2, metric3 = (
    st.columns(3)
)


with metric1:

    st.metric(
        "Observed PM2.5",
        f"{latest['pm25']:.1f} µg/m³",
    )


with metric2:

    st.metric(
        f"Predicted PM2.5 • {horizon_label}",
        f"{prediction:.1f} µg/m³",
    )


with metric3:

    st.metric(
        "WHO interpretation",
        risk["short_label"],
    )


# --------------------------------------------------
# Methodology note
# --------------------------------------------------

st.caption(
    "Forecasts are generated from historical "
    "observations in the Chengdu dataset. "
    "WHO interpretation is based on the 2021 "
    "24-hour PM2.5 guideline and interim targets. "
    "Longer forecast horizons are not separate "
    "WHO averaging-period standards."
)

# --------------------------------------------------
# Historical PM2.5
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="section-label">
        PM2.5 HISTORY
    </div>

    <div class="section-title">
        How has the air been recently?
    </div>

    <div class="section-description">
        See how pollution levels have changed over time
        leading up to this prediction.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# History controls
# --------------------------------------------------

history_col1, _ = st.columns([1, 3])


with history_col1:

    history_window = st.selectbox(
        "Time window",
        [
            "7 days",
            "30 days",
            "90 days",
            "1 year",
        ],
        index=1,
        key="history_window",
    )


history_days = {
    "7 days": 7,
    "30 days": 30,
    "90 days": 90,
    "1 year": 365,
}


days = history_days[
    history_window
]


# --------------------------------------------------
# Historical data
# --------------------------------------------------

origin_timestamp = latest["timestamp"]


history_start = (
    origin_timestamp
    - pd.Timedelta(days=days)
)


history_df = station_df[
    (
        station_df["timestamp"]
        >= history_start
    )
    &
    (
        station_df["timestamp"]
        <= origin_timestamp
    )
].copy()


history_df = history_df.dropna(
    subset=["pm25"]
)


# --------------------------------------------------
# Historical chart
# --------------------------------------------------

if history_df.empty:

    st.info(
        "No historical PM2.5 data is available "
        "for the selected period."
    )

else:

    observed_pm25 = float(
        latest["pm25"]
    )

    fig = build_history_figure(
        history_df,
        origin_timestamp,
        observed_pm25,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


    # --------------------------------------------------
    # Historical summary
    # --------------------------------------------------

    history_metric1, history_metric2, history_metric3 = (
        st.columns(3)
    )


    with history_metric1:

        st.metric(
            "Historical average",
            f"{history_df['pm25'].mean():.1f} µg/m³",
        )


    with history_metric2:

        st.metric(
            "Historical maximum",
            f"{history_df['pm25'].max():.1f} µg/m³",
        )


    with history_metric3:

        st.metric(
            "Observations",
            f"{len(history_df):,}",
        )

# --------------------------------------------------
# Multi-horizon forecast outlook
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="section-label">
        FORECAST OUTLOOK
    </div>

    <div class="section-title">
        What might the air be like in the future?
    </div>

    <div class="section-description">
        See what the model expects pollution levels
        to look like from tomorrow through the next 30 days.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Generate predictions for all horizons
# --------------------------------------------------

outlook_labels = [
    "24h",
    "48h",
    "72h",
    "7d",
    "14d",
    "30d",
]

outlook_predictions = []


for label, (suffix, hours) in HORIZONS.items():

    try:

        outlook_model = load_forecast_model(
            suffix
        )

        outlook_features = (
            outlook_model
            .get_booster()
            .feature_names
        )

        X_outlook = (
            latest[outlook_features]
            .to_frame()
            .T
        )

        X_outlook = X_outlook.apply(
            pd.to_numeric,
            errors="coerce",
        )

        outlook_prediction = float(
            outlook_model.predict(
                X_outlook
            )[0]
        )

        outlook_predictions.append(
            {
                "horizon": label,
                "short_horizon": (
                    "24h"
                    if hours == 24
                    else
                    "48h"
                    if hours == 48
                    else
                    "72h"
                    if hours == 72
                    else
                    "7d"
                    if hours == 168
                    else
                    "14d"
                    if hours == 336
                    else
                    "30d"
                ),
                "hours": hours,
                "prediction": outlook_prediction,
            }
        )

    except Exception as error:

        st.warning(
            f"Could not generate the "
            f"{label} forecast: {error}"
        )


# --------------------------------------------------
# Forecast outlook chart
# --------------------------------------------------

if outlook_predictions:

    outlook_df = pd.DataFrame(
        outlook_predictions
    )


    fig_outlook = build_outlook_figure(
        outlook_df
    )


    st.plotly_chart(
        fig_outlook,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


    # --------------------------------------------------
    # Forecast cards
    # --------------------------------------------------

    st.markdown(
        "### Horizon breakdown"
    )


    forecast_cards = st.columns(6)


    for column, forecast in zip(
        forecast_cards,
        outlook_predictions,
    ):

        forecast_risk = classify_pm25(
            forecast["prediction"]
        )


        with column:

            st.html(
                f"""
                <div class="glass-card"
                     style="
                        min-height: 145px;
                        padding: 1rem;
                     ">

                    <div style="
                        color: #69736f;
                        font-size: 0.65rem;
                        text-transform: uppercase;
                        letter-spacing: 0.12em;
                    ">
                        {forecast["short_horizon"]}
                    </div>

                    <div style="
                        margin-top: 0.65rem;
                        font-size: 1.8rem;
                        font-weight: 800;
                        color: #f4f6f5;
                        letter-spacing: -0.04em;
                    ">
                        {forecast["prediction"]:.1f}
                    </div>

                    <div style="
                        margin-top: 0.15rem;
                        font-size: 0.68rem;
                        color: #69736f;
                    ">
                        µg/m³
                    </div>

                    <div style="
                        margin-top: 0.75rem;
                        font-size: 0.63rem;
                        color: #68e0c5;
                        text-transform: uppercase;
                        letter-spacing: 0.08em;
                        font-weight: 700;
                    ">
                        {forecast_risk["short_label"]}
                    </div>

                </div>
                """
            )

# --------------------------------------------------
# Station intelligence
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="section-label">
        STATION INTELLIGENCE
    </div>

    <div class="section-title">
        What is happening across Chengdu?
    </div>

    <div class="section-description">
        Compare current PM2.5 conditions across the
        monitoring stations used in this project.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Station coordinates
# --------------------------------------------------

STATION_COORDINATES = {

    "1431A": {
        "name": "JQLH",
        "lat": 30.7236,
        "lon": 103.9728,
    },

    "1432A": {
        "name": "SLD",
        "lat": 30.6764,
        "lon": 104.1419,
    },

    "1433A": {
        "name": "SWY",
        "lat": 30.5767,
        "lon": 104.0594,
    },

    "1434A": {
        "name": "SHP",
        "lat": 30.6306,
        "lon": 104.1122,
    },

    "1437A": {
        "name": "JPJ",
        "lat": 30.6556,
        "lon": 104.0431,
    },

    "1438A": {
        "name": "LYS",
        "lat": 31.0201,
        "lon": 103.6202,
    },

    "2880A": {
        "name": "DSXL",
        "lat": 30.6558,
        "lon": 104.0219,
    },

    "3136A": {
        "name": "LQXQ",
        "lat": 30.5589,
        "lon": 104.2725,
    },

    "3358A": {
        "name": "LJL",
        "lat": 30.6994,
        "lon": 103.8458,
    },
}


# --------------------------------------------------
# Get station observations
# --------------------------------------------------

station_snapshot = []


for code, info in STATION_COORDINATES.items():

    station_rows = (
        df[
            df["station_code"]
            == code
        ]
        .copy()
    )


    station_rows = station_rows[
        station_rows["timestamp"].dt.date
        == selected_date
    ]


    station_rows = (
        station_rows
        .dropna(
            subset=["pm25"]
        )
        .sort_values(
            "timestamp"
        )
    )


    if station_rows.empty:

        continue


    observation = (
        station_rows.iloc[-1]
    )


    station_snapshot.append(
        {
            "station_code": code,
            "station_name": info["name"],
            "latitude": info["lat"],
            "longitude": info["lon"],
            "pm25": float(
                observation["pm25"]
            ),
            "timestamp": observation["timestamp"],
        }
    )


station_snapshot_df = pd.DataFrame(
    station_snapshot
)


# --------------------------------------------------
# Station comparison
# --------------------------------------------------

if station_snapshot_df.empty:

    st.info(
        "No station observations are available "
        "for the selected date."
    )

else:

    station_snapshot_df = (
        station_snapshot_df
        .sort_values(
            "pm25",
            ascending=True,
        )
    )


    # --------------------------------------------------
    # City-wide metrics
    # --------------------------------------------------

    station_metric1, station_metric2, station_metric3 = (
        st.columns(3)
    )


    with station_metric1:

        st.metric(
            "Network average",
            (
                f"{station_snapshot_df['pm25'].mean():.1f} "
                "µg/m³"
            ),
        )


    with station_metric2:

        highest_station = (
            station_snapshot_df
            .iloc[-1]
        )


        st.metric(
            "Highest station",
            (
                f"{highest_station['station_code']} · "
                f"{highest_station['pm25']:.1f}"
            ),
        )


    with station_metric3:

        lowest_station = (
            station_snapshot_df
            .iloc[0]
        )


        st.metric(
            "Lowest station",
            (
                f"{lowest_station['station_code']} · "
                f"{lowest_station['pm25']:.1f}"
            ),
        )


    fig_station = build_station_figure(
        station_snapshot_df,
        station_code,
    )


    st.plotly_chart(
        fig_station,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


   # --------------------------------------------------
# Station map
# --------------------------------------------------

st.markdown(
    "### Monitoring network"
)

st.caption(
    "Station markers are coloured according to the "
    "application's PM2.5 interpretation of the WHO "
    "2021 thresholds."
)


station_status_lookup = {
    row["station_code"]: classify_pm25(row["pm25"])["short_label"]
    for _, row in station_snapshot_df.iterrows()
}

fig_map = build_map_figure(
    station_snapshot_df,
    station_code,
    station_status_lookup,
)


st.plotly_chart(
    fig_map,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "scrollZoom": False,
    },
)


# --------------------------------------------------
# Map legend
# --------------------------------------------------

st.html(get_map_legend_html())

# --------------------------------------------------
# Why this forecast?
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="section-label">
        WHY THIS FORECAST?
    </div>

    <div class="section-title">
        Why did the model make this prediction?
    </div>

    <div class="section-description">
        See which factors had the biggest influence on
        the prediction, such as recent pollution levels,
        weather and past patterns.
    </div>
    """,
    unsafe_allow_html=True,
)


if top_drivers.empty:

    st.info(
        "No explainability information is available "
        "for this prediction."
    )

else:

    max_shap = (
        top_drivers["absolute_shap"].max()
    )

    for _, row in (
        top_drivers.iterrows()
    ):

        feature_name = (
            row["display_feature"]
        )

        feature_value = (
            row["value"]
        )

        shap_value = (
            row["shap_value"]
        )


        if max_shap > 0:

            bar_width = (
                abs(shap_value)
                / max_shap
                * 100
            )

        else:

            bar_width = 0


        if shap_value >= 0:

            bar_class = (
                "shap-positive"
            )

            direction = "↑"

            direction_text = (
                "increases PM2.5"
            )

        else:

            bar_class = (
                "shap-negative"
            )

            direction = "↓"

            direction_text = (
                "reduces PM2.5"
            )


        st.html(
            f"""
            <div class="shap-row">

                <div class="shap-header">

                    <div class="shap-feature">
                        {direction}
                        {feature_name}
                    </div>

                    <div class="shap-value">
                        {shap_value:+.2f}
                    </div>

                </div>

                <div class="shap-meta">
                    Observed value:
                    {feature_value:.2f}
                    · {direction_text}
                </div>

                <div class="shap-track">

                    <div
                        class="shap-bar {bar_class}"
                        style="
                            width:
                            {bar_width:.1f}%;
                        "
                    ></div>

                </div>

            </div>
            """
        )


# --------------------------------------------------
# Health impact
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="section-label">
        HEALTH IMPACT
    </div>

    <div class="section-title">
        What does this pollution level mean?
    </div>

    <div class="section-description">
        See how the predicted pollution level compares
        with recommended air-quality limits and what
        the level means for health.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Health summary
# --------------------------------------------------

health_col1, health_col2 = (
    st.columns([1, 2])
)


with health_col1:

    st.html(
        f"""
        <div class="health-card">

            <div class="section-label">
                Predicted concentration
            </div>

            <div style="
                margin-top: 0.8rem;
                display: flex;
                align-items: baseline;
                gap: 0.5rem;
            ">

                <span class="health-number">
                    {prediction:.1f}
                </span>

                <span class="health-unit">
                    µg/m³
                </span>

            </div>

            <div style="
                margin-top: 1.2rem;
                font-size: 0.75rem;
                color: #68e0c5;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-weight: 700;
            ">
                {risk["short_label"]}
            </div>

            <div style="
                margin-top: 0.7rem;
                font-size: 0.85rem;
                line-height: 1.6;
                color: #8e9995;
            ">
                {risk["description"]}
            </div>

        </div>
        """
    )


# --------------------------------------------------
# WHO visual scale
# --------------------------------------------------

with health_col2:

    st.markdown(
        "### WHO threshold scale"
    )


    # Keep the gauge visually capped at 75.
    # Values above 75 appear in the final IT1+
    # section.
    gauge_value = min(
        max(prediction, 0),
        75,
    )


    gauge_position = (
        gauge_value / 75 * 100
    )


    st.html(
        f"""
        <div style="
            margin-top: 2rem;
            padding: 1.5rem 0.5rem 0.8rem 0.5rem;
        ">

            <!-- Gauge -->

            <div style="
                position: relative;
                height: 16px;
                width: 100%;
                border-radius: 999px;
                overflow: visible;
                display: flex;
            ">

                <!-- WHO AQG: 0–15 -->

                <div style="
                    width: 20%;
                    background: #4bd8bd;
                    border-radius:
                        999px 0 0 999px;
                "></div>


                <!-- IT4: 15–25 -->

                <div style="
                    width: 13.333%;
                    background: #68e0c5;
                "></div>


                <!-- IT3: 25–37.5 -->

                <div style="
                    width: 16.667%;
                    background: #f4c95d;
                "></div>


                <!-- IT2: 37.5–50 -->

                <div style="
                    width: 16.667%;
                    background: #f39c52;
                "></div>


                <!-- IT1+: 50–75+ -->

                <div style="
                    width: 33.333%;
                    background: #ff6f61;
                    border-radius:
                        0 999px 999px 0;
                "></div>


                <!-- Prediction marker -->

                <div style="
                    position: absolute;
                    left: {gauge_position:.2f}%;
                    top: -11px;
                    transform: translateX(-50%);
                    width: 4px;
                    height: 38px;
                    background: #f5f7f6;
                    border-radius: 999px;
                    box-shadow:
                        0 0 0 3px
                        rgba(255,255,255,0.10),
                        0 0 18px
                        rgba(255,255,255,0.35);
                "></div>


                <!-- Prediction value -->

                <div style="
                    position: absolute;
                    left: {gauge_position:.2f}%;
                    top: -43px;
                    transform: translateX(-50%);
                    white-space: nowrap;
                    padding: 0.3rem 0.55rem;
                    border-radius: 7px;
                    background: #f4f6f5;
                    color: #0a0d0d;
                    font-size: 0.72rem;
                    font-weight: 800;
                    letter-spacing: 0.02em;
                ">
                    {prediction:.1f}
                </div>

            </div>


            <!-- Threshold labels -->

            <div style="
                position: relative;
                height: 55px;
                margin-top: 0.7rem;
            ">


                <!-- 15 -->

                <div style="
                    position: absolute;
                    left: 20%;
                    transform: translateX(-50%);
                    text-align: center;
                ">

                    <div style="
                        width: 1px;
                        height: 10px;
                        background: #78827f;
                        margin:
                            0 auto 0.35rem auto;
                    "></div>

                    <div style="
                        color: #d8dddb;
                        font-size: 0.72rem;
                        font-weight: 700;
                    ">
                        15
                    </div>

                    <div style="
                        color: #69736f;
                        font-size: 0.62rem;
                        margin-top: 0.15rem;
                    ">
                        WHO AQG
                    </div>

                </div>


                <!-- 25 -->

                <div style="
                    position: absolute;
                    left: 33.333%;
                    transform: translateX(-50%);
                    text-align: center;
                ">

                    <div style="
                        width: 1px;
                        height: 10px;
                        background: #78827f;
                        margin:
                            0 auto 0.35rem auto;
                    "></div>

                    <div style="
                        color: #d8dddb;
                        font-size: 0.72rem;
                        font-weight: 700;
                    ">
                        25
                    </div>

                    <div style="
                        color: #69736f;
                        font-size: 0.62rem;
                        margin-top: 0.15rem;
                    ">
                        IT4
                    </div>

                </div>


                <!-- 37.5 -->

                <div style="
                    position: absolute;
                    left: 50%;
                    transform: translateX(-50%);
                    text-align: center;
                ">

                    <div style="
                        width: 1px;
                        height: 10px;
                        background: #78827f;
                        margin:
                            0 auto 0.35rem auto;
                    "></div>

                    <div style="
                        color: #d8dddb;
                        font-size: 0.72rem;
                        font-weight: 700;
                    ">
                        37.5
                    </div>

                    <div style="
                        color: #69736f;
                        font-size: 0.62rem;
                        margin-top: 0.15rem;
                    ">
                        IT3
                    </div>

                </div>


                <!-- 50 -->

                <div style="
                    position: absolute;
                    left: 66.667%;
                    transform: translateX(-50%);
                    text-align: center;
                ">

                    <div style="
                        width: 1px;
                        height: 10px;
                        background: #78827f;
                        margin:
                            0 auto 0.35rem auto;
                    "></div>

                    <div style="
                        color: #d8dddb;
                        font-size: 0.72rem;
                        font-weight: 700;
                    ">
                        50
                    </div>

                    <div style="
                        color: #69736f;
                        font-size: 0.62rem;
                        margin-top: 0.15rem;
                    ">
                        IT2
                    </div>

                </div>


                <!-- 75 -->

                <div style="
                    position: absolute;
                    left: 100%;
                    transform: translateX(-100%);
                    text-align: right;
                ">

                    <div style="
                        width: 1px;
                        height: 10px;
                        background: #78827f;
                        margin-left: auto;
                        margin-bottom: 0.35rem;
                    "></div>

                    <div style="
                        color: #d8dddb;
                        font-size: 0.72rem;
                        font-weight: 700;
                    ">
                        75+
                    </div>

                    <div style="
                        color: #69736f;
                        font-size: 0.62rem;
                        margin-top: 0.15rem;
                    ">
                        IT1+
                    </div>

                </div>

            </div>

        </div>
        """
    )


# --------------------------------------------------
# Interpretation
# --------------------------------------------------

st.html(
    f"""
    <div
        class="glass-card"
        style="
            margin-top: 1rem;
            padding: 1.2rem 1.4rem;
        "
    >

        <div class="section-label">
            Interpretation
        </div>

        <div
            style="
                margin-top: 0.55rem;
                color: #c1c9c6;
                font-size: 0.88rem;
                line-height: 1.65;
            "
        >
            The predicted concentration of

            <strong style="color:#f4f6f5;">
                {prediction:.1f} µg/m³
            </strong>

            is classified by the application as

            <strong style="color:#68e0c5;">
                {risk["short_label"]}
            </strong>.

            This classification is based on the WHO 2021
            24-hour PM2.5 guideline and interim targets.
        </div>

    </div>
    """
)


# --------------------------------------------------
# Scientific note
# --------------------------------------------------

st.caption(
    "WHO 2021 provides a 24-hour PM2.5 air-quality "
    "guideline of 15 µg/m³ and interim targets at "
    "25, 37.5, 50 and 75 µg/m³. The labels used by "
    "this application are application-defined "
    "interpretations of these thresholds and are "
    "not official WHO risk-category names. For "
    "forecast horizons longer than 24 hours, these "
    "thresholds should not be interpreted as separate "
    "WHO standards for those averaging periods."
)