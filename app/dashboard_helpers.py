from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


STATIONS = {
    "1431A — JQLH": "1431A",
    "1432A — SLD": "1432A",
    "1433A — SWY": "1433A",
    "1434A — SHP": "1434A",
    "1437A — JPJ": "1437A",
    "1438A — LYS": "1438A",
    "2880A — DSXL": "2880A",
    "3136A — LQXQ": "3136A",
    "3358A — LJL": "3358A",
}


HORIZONS = {
    "24 hours": ("24h", 24),
    "48 hours": ("48h", 48),
    "72 hours": ("72h", 72),
    "7 days": ("7d", 168),
    "14 days": ("14d", 336),
    "30 days": ("30d", 720),
}


STATION_COORDINATES = {
    "1431A": {"name": "JQLH", "lat": 30.7236, "lon": 103.9728},
    "1432A": {"name": "SLD", "lat": 30.6764, "lon": 104.1419},
    "1433A": {"name": "SWY", "lat": 30.5767, "lon": 104.0594},
    "1434A": {"name": "SHP", "lat": 30.6306, "lon": 104.1122},
    "1437A": {"name": "JPJ", "lat": 30.6556, "lon": 104.0431},
    "1438A": {"name": "LYS", "lat": 31.0201, "lon": 103.6202},
    "2880A": {"name": "DSXL", "lat": 30.6558, "lon": 104.0219},
    "3136A": {"name": "LQXQ", "lat": 30.5589, "lon": 104.2725},
    "3358A": {"name": "LJL", "lat": 30.6994, "lon": 103.8458},
}


def get_station_colour(pm25_value):
    if pm25_value <= 15:
        return "#4bd8bd"
    elif pm25_value <= 25:
        return "#68e0c5"
    elif pm25_value <= 37.5:
        return "#f4c95d"
    elif pm25_value <= 50:
        return "#f39c52"
    elif pm25_value <= 75:
        return "#ff6f61"
    return "#d94f70"


def build_history_figure(history_df, origin_timestamp, observed_pm25):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["pm25"],
            mode="lines",
            name="PM2.5",
            line=dict(width=1.7),
            hovertemplate=(
                "<b>%{x|%d %b %Y %H:%M}</b>"
                "<br>"
                "PM2.5: %{y:.1f} µg/m³"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[origin_timestamp],
            y=[observed_pm25],
            mode="markers",
            name="Forecast origin",
            marker=dict(size=10, symbol="circle"),
            hovertemplate=(
                "<b>Forecast origin</b>"
                "<br>"
                "%{x|%d %b %Y %H:%M}"
                "<br>"
                "Observed PM2.5: %{y:.1f} µg/m³"
                "<extra></extra>"
            ),
        )
    )

    for threshold, label in [
        (15, "WHO AQG"),
        (25, "IT4"),
        (37.5, "IT3"),
        (50, "IT2"),
        (75, "IT1"),
    ]:
        fig.add_hline(
            y=threshold,
            line_width=1,
            line_dash="dot",
            annotation_text=f"{label} · {threshold:g}",
            annotation_position="top left",
        )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=20, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="PM2.5 · µg/m³",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=False,
            rangemode="tozero",
            tickfont=dict(size=11),
        ),
    )

    fig.add_vline(
        x=origin_timestamp.timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        annotation_text="Forecast origin",
        annotation_position="top right",
    )

    return fig


def build_outlook_figure(outlook_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=outlook_df["short_horizon"],
            y=outlook_df["prediction"],
            mode="lines+markers",
            name="Forecast",
            line=dict(width=3, shape="spline"),
            marker=dict(size=9),
            hovertemplate=(
                "<b>%{x}</b>"
                "<br>"
                "Predicted PM2.5: %{y:.1f} µg/m³"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=15,
        line_width=1,
        line_dash="dot",
        annotation_text="WHO AQG · 15",
        annotation_position="top left",
    )

    for threshold, label in [(25, "IT4 · 25"), (37.5, "IT3 · 37.5"), (50, "IT2 · 50"), (75, "IT1 · 75")]:
        fig.add_hline(
            y=threshold,
            line_width=1,
            line_dash="dot",
            annotation_text=label,
            annotation_position="top left",
        )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        hovermode="x",
        showlegend=False,
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            showline=False,
            categoryorder="array",
            categoryarray=["24h", "48h", "72h", "7d", "14d", "30d"],
        ),
        yaxis=dict(
            title="PM2.5 · µg/m³",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=False,
            rangemode="tozero",
        ),
    )

    return fig


def build_station_figure(station_snapshot_df, station_code):
    fig = go.Figure()

    for _, station in station_snapshot_df.iterrows():
        is_selected = station["station_code"] == station_code
        fig.add_trace(
            go.Bar(
                x=[station["station_code"]],
                y=[station["pm25"]],
                name=station["station_code"],
                showlegend=False,
                marker=dict(opacity=1.0 if is_selected else 0.55),
                hovertemplate=(
                    f"<b>{station['station_code']} · {station['station_name']}</b>"
                    "<br>"
                    "PM2.5: %{y:.1f} µg/m³"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=15, line_width=1, line_dash="dot", annotation_text="WHO AQG · 15", annotation_position="top left")
    fig.add_hline(y=25, line_width=1, line_dash="dot", annotation_text="IT4 · 25", annotation_position="top left")

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        showlegend=False,
        xaxis=dict(title=None, showgrid=False, zeroline=False, showline=False),
        yaxis=dict(
            title="PM2.5 · µg/m³",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=False,
            rangemode="tozero",
        ),
    )

    return fig


def build_map_figure(station_snapshot_df, station_code, station_status_lookup=None):
    if station_status_lookup is None:
        station_status_lookup = {}

    fig = go.Figure()

    for _, station in station_snapshot_df.iterrows():
        is_selected = station["station_code"] == station_code
        marker_size = 20 if is_selected else 13
        marker_colour = get_station_colour(station["pm25"])
        status_label = station_status_lookup.get(station["station_code"], "")

        if is_selected:
            fig.add_trace(
                go.Scattermap(
                    lat=[station["latitude"]],
                    lon=[station["longitude"]],
                    mode="markers",
                    marker=dict(size=34, color="rgba(255,255,255,0.12)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scattermap(
                lat=[station["latitude"]],
                lon=[station["longitude"]],
                mode="markers+text",
                text=[station["station_code"]],
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(size=marker_size, color=marker_colour),
                customdata=[[station["station_name"], station["pm25"], status_label]],
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<br>"
                    "%{customdata[0]}"
                    "<br>"
                    "PM2.5: %{customdata[1]:.1f} µg/m³"
                    "<br>"
                    "Status: %{customdata[2]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        map=dict(style="carto-darkmatter", center=dict(lat=30.68, lon=104.06), zoom=9),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig


def get_map_legend_html():
    return """
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        align-items: center;
        margin-top: 0.4rem;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
    ">

        <div style="
            font-size: 0.65rem;
            color: #69736f;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
        ">
            PM2.5
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:0.4rem;
            font-size:0.68rem;
            color:#9ba5a1;
        ">
            <span style="
                width:9px;
                height:9px;
                border-radius:50%;
                background:#4bd8bd;
                display:inline-block;
            "></span>
            ≤15
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:0.4rem;
            font-size:0.68rem;
            color:#9ba5a1;
        ">
            <span style="
                width:9px;
                height:9px;
                border-radius:50%;
                background:#68e0c5;
                display:inline-block;
            "></span>
            15–25
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:0.4rem;
            font-size:0.68rem;
            color:#9ba5a1;
        ">
            <span style="
                width:9px;
                height:9px;
                border-radius:50%;
                background:#f4c95d;
                display:inline-block;
            "></span>
            25–37.5
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:0.4rem;
            font-size:0.68rem;
            color:#9ba5a1;
        ">
            <span style="
                width:9px;
                height:9px;
                border-radius:50%;
                background:#f39c52;
                display:inline-block;
            "></span>
            37.5–50
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:0.4rem;
            font-size:0.68rem;
            color:#9ba5a1;
        ">
            <span style="
                width:9px;
                height:9px;
                border-radius:50%;
                background:#ff6f61;
                display:inline-block;
            "></span>
            50–75
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:0.4rem;
            font-size:0.68rem;
            color:#9ba5a1;
        ">
            <span style="
                width:9px;
                height:9px;
                border-radius:50%;
                background:#d94f70;
                display:inline-block;
            "></span>
            75+
        </div>

    </div>
    """
