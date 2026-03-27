from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from wordcloud import WordCloud


INDIA_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson",
    "https://raw.githubusercontent.com/imdevskp/india_map/master/india_state_geo.json",
]

STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.74),
    "Arunachal Pradesh": (28.218, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Delhi": (28.7041, 77.1025),
    "Goa": (15.2993, 74.124),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jammu and Kashmir": (34.0837, 74.7973),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.855),
}


def _normalize_state(value: str) -> str:
    return " ".join(str(value).lower().replace("&", "and").replace("-", " ").split())


def _state_alias_map() -> dict[str, str]:
    return {
        "orissa": "Odisha",
        "uttaranchal": "Uttarakhand",
        "nct of delhi": "Delhi",
    }


def _extract_feature_key(geojson_obj: dict[str, Any]) -> str:
    sample_props = geojson_obj["features"][0]["properties"]
    candidates = ["st_nm", "state_name", "STATE", "name", "NAME_1"]
    for candidate in candidates:
        if candidate in sample_props:
            return f"properties.{candidate}"

    first_key = list(sample_props.keys())[0]
    return f"properties.{first_key}"


def load_india_geojson() -> tuple[dict[str, Any] | None, str | None]:
    for url in INDIA_GEOJSON_URLS:
        try:
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            geojson_obj = response.json()
            feature_key = _extract_feature_key(geojson_obj)
            return geojson_obj, feature_key
        except Exception:
            continue
    return None, None


def event_participation_chart(df: pd.DataFrame) -> go.Figure:
    event_counts = (
        df.groupby("event_name", as_index=False)
        .size()
        .rename(columns={"size": "participants"})
        .sort_values("participants", ascending=False)
    )

    fig = px.bar(
        event_counts,
        x="participants",
        y="event_name",
        orientation="h",
        color="participants",
        color_continuous_scale="Sunset",
        title="Event-wise Participation",
        hover_data={"participants": True, "event_name": True},
    )
    fig.update_layout(yaxis_title="Event", xaxis_title="Participants", coloraxis_showscale=False)
    return fig


def college_top10_chart(df: pd.DataFrame) -> go.Figure:
    college_counts = (
        df.groupby("college_name", as_index=False)
        .size()
        .rename(columns={"size": "participants"})
        .sort_values("participants", ascending=False)
        .head(10)
    )

    fig = px.bar(
        college_counts,
        x="participants",
        y="college_name",
        orientation="h",
        color="participants",
        color_continuous_scale="Tealgrn",
        title="Top 10 College Leaderboard",
    )
    fig.update_layout(yaxis_title="College", xaxis_title="Participants", coloraxis_showscale=False)
    return fig


def registration_trend_chart(df: pd.DataFrame) -> go.Figure:
    trend_df = (
        df.assign(registration_date=pd.to_datetime(df["registration_date"]))
        .groupby("registration_date", as_index=False)
        .size()
        .rename(columns={"size": "registrations"})
        .sort_values("registration_date")
    )

    fig = px.line(
        trend_df,
        x="registration_date",
        y="registrations",
        markers=True,
        title="Registration Trend Over Time",
    )
    fig.update_traces(line=dict(color="#0f766e", width=3))
    fig.update_layout(xaxis_title="Registration Date", yaxis_title="Registrations")
    return fig


def sentiment_pie_chart(df: pd.DataFrame) -> go.Figure:
    sent_df = (
        df.groupby("sentiment", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )

    fig = px.pie(
        sent_df,
        values="count",
        names="sentiment",
        color="sentiment",
        color_discrete_map={"Positive": "#16a34a", "Neutral": "#0ea5e9", "Negative": "#dc2626"},
        hole=0.45,
        title="Sentiment Mix",
    )
    return fig


def rating_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="rating",
        nbins=5,
        title="Rating Distribution",
        color_discrete_sequence=["#f97316"],
    )
    fig.update_layout(xaxis=dict(dtick=1), xaxis_title="Rating", yaxis_title="Responses")
    return fig


def avg_rating_event_chart(df: pd.DataFrame) -> go.Figure:
    avg_df = (
        df.groupby("event_name", as_index=False)["rating"]
        .mean()
        .sort_values("rating", ascending=False)
    )
    fig = px.bar(
        avg_df,
        x="rating",
        y="event_name",
        orientation="h",
        color="rating",
        color_continuous_scale="Bluered",
        title="Average Rating per Event",
    )
    fig.update_layout(xaxis_title="Average Rating", yaxis_title="Event", coloraxis_showscale=False)
    return fig


def keyword_bar_chart(keyword_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        keyword_df,
        x="count",
        y="keyword",
        orientation="h",
        color="count",
        color_continuous_scale="Oranges",
        title="Most Common Feedback Keywords",
    )
    fig.update_layout(xaxis_title="Frequency", yaxis_title="Keyword", coloraxis_showscale=False)
    return fig


def wordcloud_figure(text_series: pd.Series):
    text_blob = " ".join(str(text) for text in text_series.fillna(""))
    if not text_blob.strip():
        text_blob = "no feedback available"

    cloud = WordCloud(width=1100, height=450, background_color="white", colormap="magma").generate(text_blob)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _india_scatter_fallback(state_df: pd.DataFrame) -> go.Figure:
    scatter_df = state_df.copy()
    scatter_df["lat"] = scatter_df["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[0])
    scatter_df["lon"] = scatter_df["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[1])
    scatter_df = scatter_df.dropna(subset=["lat", "lon"])

    fig = px.scatter_geo(
        scatter_df,
        lat="lat",
        lon="lon",
        size="participants",
        color="participants",
        color_continuous_scale="YlOrRd",
        hover_name="state",
        hover_data={"participants": True, "lat": False, "lon": False},
        title="India State-wise Participation Map (Fallback)",
        scope="asia",
    )
    fig.update_geos(
        center={"lat": 22.5, "lon": 79.0},
        projection_scale=4.2,
        showcountries=True,
        countrycolor="#94a3b8",
        showcoastlines=True,
        coastlinecolor="#64748b",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    return fig


def india_choropleth(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    state_df = (
        df.groupby("state", as_index=False)
        .size()
        .rename(columns={"size": "participants"})
        .sort_values("participants", ascending=False)
    )

    geojson_obj, feature_key = load_india_geojson()
    if geojson_obj is None or feature_key is None:
        return _india_scatter_fallback(state_df), state_df

    feature_state_key = feature_key.split(".", 1)[1]
    geo_states = {
        _normalize_state(feature["properties"].get(feature_state_key, "")): feature["properties"].get(feature_state_key, "")
        for feature in geojson_obj["features"]
    }

    alias = _state_alias_map()

    def resolve_geo_state(raw_state: str) -> str | None:
        norm = _normalize_state(raw_state)
        if norm in alias:
            norm = _normalize_state(alias[norm])
        return geo_states.get(norm)

    map_df = state_df.copy()
    map_df["geo_state"] = map_df["state"].apply(resolve_geo_state)
    map_df = map_df.dropna(subset=["geo_state"])

    if map_df.empty:
        return _india_scatter_fallback(state_df), state_df

    fig = px.choropleth(
        map_df,
        geojson=geojson_obj,
        locations="geo_state",
        featureidkey=feature_key,
        color="participants",
        color_continuous_scale="YlOrRd",
        hover_name="state",
        hover_data={"participants": True, "geo_state": False},
        title="India State-wise Participation Heatmap",
    )

    top3 = map_df.sort_values("participants", ascending=False).head(3)
    if not top3.empty:
        marker_lats = [STATE_COORDS.get(state, (None, None))[0] for state in top3["state"]]
        marker_lons = [STATE_COORDS.get(state, (None, None))[1] for state in top3["state"]]
        marker_text = [
            f"Top State: {state} ({count})"
            for state, count in zip(top3["state"], top3["participants"])
        ]
        fig.add_trace(
            go.Scattergeo(
                lon=marker_lons,
                lat=marker_lats,
                mode="markers+text",
                text=["Top 1", "Top 2", "Top 3"][: len(top3)],
                textposition="top center",
                marker=dict(size=16, color="#1d4ed8", symbol="star"),
                hovertext=marker_text,
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    return fig, state_df
