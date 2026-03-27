from __future__ import annotations

import pandas as pd
import streamlit as st

from data_loader import load_dataset
from utils import auto_insights, extract_top_feedback_samples, keyword_frequency, preprocess_feedback
from visuals import (
    avg_rating_event_chart,
    college_top10_chart,
    event_participation_chart,
    india_choropleth,
    keyword_bar_chart,
    rating_histogram,
    registration_trend_chart,
    sentiment_pie_chart,
    wordcloud_figure,
)


st.set_page_config(
    page_title="GATEWAYS-2025 Analytics Dashboard",
    page_icon="🎯",
    layout="wide",
)


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Manrope:wght@400;600&display=swap');

            .stApp {
                font-family: 'Manrope', sans-serif;
                background:
                    radial-gradient(circle at 8% 12%, rgba(251, 146, 60, 0.18), transparent 32%),
                    radial-gradient(circle at 86% 20%, rgba(14, 165, 233, 0.18), transparent 35%),
                    linear-gradient(160deg, #fffdf8 0%, #f7fcff 44%, #f5fff8 100%);
            }

            section[data-testid="stSidebar"] {
                background:
                    radial-gradient(circle at 12% 6%, rgba(125, 211, 252, 0.2), transparent 34%),
                    linear-gradient(180deg, #0b1324 0%, #14213d 55%, #1e293b 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.24);
            }

            section[data-testid="stSidebar"] * {
                color: #e6edf6;
            }

            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                color: #ffffff;
                letter-spacing: 0.25px;
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #bfdbfe;
            }

            section[data-testid="stSidebar"] .stMultiSelect,
            section[data-testid="stSidebar"] [data-baseweb="select"],
            section[data-testid="stSidebar"] [data-baseweb="popover"] {
                font-size: 1rem;
            }

            section[data-testid="stSidebar"] [data-baseweb="select"] > div {
                min-height: 48px;
                border-radius: 12px;
                border: 1px solid rgba(125, 211, 252, 0.4);
                background: rgba(15, 23, 42, 0.52);
            }

            section[data-testid="stSidebar"] [data-baseweb="tag"] {
                background: rgba(34, 211, 238, 0.2) !important;
                border: 1px solid rgba(34, 211, 238, 0.45) !important;
                color: #e0f2fe !important;
                font-weight: 700 !important;
                border-radius: 999px !important;
            }

            section[data-testid="stSidebar"] label {
                font-size: 0.98rem !important;
                font-weight: 700 !important;
                color: #f8fafc !important;
            }

            .sidebar-panel {
                border-radius: 14px;
                padding: 0.8rem 0.9rem;
                margin: 0.25rem 0 0.95rem 0;
                background: linear-gradient(140deg, rgba(34, 211, 238, 0.15), rgba(37, 99, 235, 0.14));
                border: 1px solid rgba(125, 211, 252, 0.35);
            }

            .sidebar-panel h4 {
                margin: 0;
                font-size: 1rem;
                color: #f8fafc;
            }

            .sidebar-panel p {
                margin: 0.25rem 0 0 0;
                font-size: 0.84rem;
                color: #dbeafe;
            }

            h1, h2, h3 {
                font-family: 'Sora', sans-serif;
                letter-spacing: 0.3px;
            }

            .hero {
                border-radius: 18px;
                padding: 1.25rem 1.4rem;
                background: linear-gradient(120deg, rgba(255, 161, 90, 0.96), rgba(251, 191, 36, 0.9));
                border: 1px solid rgba(194, 65, 12, 0.18);
                box-shadow: 0 16px 30px rgba(251, 146, 60, 0.24);
                animation: fadeInUp 0.8s ease-out;
            }

            .kpi-card {
                border-radius: 14px;
                padding: 0.85rem 1rem;
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(15, 23, 42, 0.08);
                box-shadow: 0 10px 22px rgba(2, 6, 23, 0.08);
                animation: fadeInUp 0.7s ease-out;
            }

            .kpi-title {
                font-size: 0.86rem;
                color: #334155;
                margin-bottom: 2px;
                font-weight: 600;
            }

            .kpi-value {
                font-size: 1.55rem;
                color: #0f172a;
                font-weight: 800;
            }

            .insight-box {
                border-left: 5px solid #0284c7;
                border-radius: 10px;
                padding: 0.8rem 1rem;
                background: rgba(14, 165, 233, 0.08);
                margin: 0.3rem 0;
            }

            .section-title {
                background: linear-gradient(90deg, rgba(251, 146, 60, 0.18), rgba(14, 165, 233, 0.15));
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 12px;
                padding: 0.65rem 0.95rem;
                margin-bottom: 0.65rem;
                font-weight: 800;
                color: #0f172a;
            }

            button[kind="secondary"] {
                border-radius: 12px !important;
                border: 1px solid rgba(14, 165, 233, 0.42) !important;
                font-weight: 700 !important;
                font-size: 0.95rem !important;
            }

            [data-testid="stDownloadButton"] button {
                width: 100%;
                min-height: 3.1rem;
                border-radius: 14px !important;
                border: none !important;
                background: linear-gradient(135deg, #0284c7, #0369a1) !important;
                color: #f8fafc !important;
                font-size: 1rem !important;
                font-weight: 800 !important;
                box-shadow: 0 10px 22px rgba(2, 132, 199, 0.34);
            }

            [data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 10px;
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 14px;
                padding: 8px;
            }

            [data-testid="stTabs"] [data-baseweb="tab"] {
                height: 54px;
                padding: 0 20px;
                border-radius: 11px;
                border: 1px solid transparent;
                font-size: 1.02rem;
                font-weight: 800;
                color: #334155;
                background: rgba(255, 255, 255, 0.65);
                transition: all 0.2s ease;
            }

            [data-testid="stTabs"] [aria-selected="true"] {
                background: linear-gradient(120deg, #f97316, #f59e0b) !important;
                color: #0b1120 !important;
                border-color: rgba(234, 88, 12, 0.45) !important;
                box-shadow: 0 8px 18px rgba(249, 115, 22, 0.34);
            }

            [data-testid="stTabs"] [data-baseweb="tab"]:hover {
                border-color: rgba(2, 132, 199, 0.4);
                transform: translateY(-1px);
            }

            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(14px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(icon: str, title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{icon} {title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_filtered_data(
    df: pd.DataFrame,
    selected_events: list[str],
    selected_states: list[str],
    selected_colleges: list[str],
) -> pd.DataFrame:
    mask = (
        df["event_name"].isin(selected_events)
        & df["state"].isin(selected_states)
        & df["college_name"].isin(selected_colleges)
    )
    return df.loc[mask].copy()


@st.cache_data(show_spinner=False)
def build_csv_bytes(df: pd.DataFrame) -> bytes:
    # UTF-8 with BOM helps Excel open CSV without garbled characters.
    return df.to_csv(index=False).encode("utf-8-sig")


def render() -> None:
    apply_custom_theme()

    with st.spinner("Loading and analyzing GATEWAYS-2025 data..."):
        base_df = load_dataset()
        df = preprocess_feedback(base_df)

    st.markdown(
        """
        <div class="hero">
            <h1>GATEWAYS-2025 Analytics Dashboard</h1>
            <p>
                A professional insights engine for participation intelligence, state outreach, and feedback quality.
                Use filters to explore patterns, compare outcomes, and generate action-ready decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-panel">
            <h4>🎛️ Insight Navigator</h4>
            <p>Adjust filters to update the full dashboard in real time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Tip: Narrow down one event + one state to present focused insights.")
    all_events = sorted(df["event_name"].dropna().unique().tolist())
    all_states = sorted(df["state"].dropna().unique().tolist())
    all_colleges = sorted(df["college_name"].dropna().unique().tolist())

    selected_events = st.sidebar.multiselect("Select Event", all_events, default=all_events)
    selected_states = st.sidebar.multiselect("Select State", all_states, default=all_states)
    selected_colleges = st.sidebar.multiselect("Select College", all_colleges, default=all_colleges)

    filtered_df = get_filtered_data(df, selected_events, selected_states, selected_colleges)

    if filtered_df.empty:
        st.warning("No records match current filters. Please broaden your selections.")
        return

    csv_bytes = build_csv_bytes(filtered_df)
    st.sidebar.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_bytes,
        file_name="gateways_2025_filtered_insights.csv",
        mime="text/csv",
        key="sidebar_csv_download",
        use_container_width=True,
    )

    total_participants = filtered_df["participant_id"].nunique()
    total_events = filtered_df["event_name"].nunique()
    total_colleges = filtered_df["college_name"].nunique()
    avg_rating = filtered_df["rating"].mean()

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi_card("👥", "Total Participants", f"{total_participants}")
    with kpi_cols[1]:
        render_kpi_card("🎯", "Total Events", f"{total_events}")
    with kpi_cols[2]:
        render_kpi_card("🏫", "Total Colleges", f"{total_colleges}")
    with kpi_cols[3]:
        render_kpi_card("⭐", "Average Rating", f"{avg_rating:.2f}")

    most_popular_event = filtered_df["event_name"].value_counts().idxmax()
    top_state = filtered_df["state"].value_counts().idxmax()
    highest_rated_event = (
        filtered_df.groupby("event_name", as_index=False)["rating"].mean().sort_values("rating", ascending=False).iloc[0]["event_name"]
    )

    insight_row = st.columns(3)
    insight_row[0].info(f"🔥 Most Popular Event: {most_popular_event}")
    insight_row[1].info(f"🗺️ Top Participating State: {top_state}")
    insight_row[2].info(f"🏆 Highest Rated Event: {highest_rated_event}")

    tab_home, tab_participation, tab_map, tab_feedback, tab_rating = st.tabs(
        [
            "🏠 Home Story",
            "📊 Participation Intelligence",
            "🗺️ India Heatmap",
            "💬 Smart Feedback",
            "⭐ Rating Analytics",
        ]
    )

    with tab_home:
        st.markdown('<div class="section-title">🏠 Home Story</div>', unsafe_allow_html=True)
        insights = auto_insights(filtered_df)
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.markdown("### Key Insights Generated Automatically")
        st.write("- Participation concentration reveals where outreach should scale next.")
        st.write("- Sentiment shifts highlight quality moments and operational gaps.")
        st.write("- Event-level ratings indicate where to allocate mentors and logistics focus.")

        st.download_button(
            label="⬇️ Download Insights Report (CSV)",
            data=csv_bytes,
            file_name="gateways_2025_filtered_insights.csv",
            mime="text/csv",
            key="home_csv_download",
            use_container_width=True,
        )

    with tab_participation:
        st.markdown('<div class="section-title">📊 Participation Intelligence</div>', unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            event_fig = event_participation_chart(filtered_df)
            event_selection = st.plotly_chart(
                event_fig,
                use_container_width=True,
                key="event_participation_chart",
                on_select="rerun",
            )

        with col_right:
            st.plotly_chart(college_top10_chart(filtered_df), use_container_width=True)
            leaderboard = (
                filtered_df.groupby("college_name", as_index=False)
                .size()
                .rename(columns={"size": "participants"})
                .sort_values("participants", ascending=False)
                .head(3)
            )
            medals = ["🥇", "🥈", "🥉"]
            for idx, row in leaderboard.reset_index(drop=True).iterrows():
                st.write(f"{medals[idx]} {row['college_name']} - {row['participants']} participants")

        st.plotly_chart(registration_trend_chart(filtered_df), use_container_width=True)

        selected_event = None
        points = event_selection.get("selection", {}).get("points", []) if event_selection else []
        if points:
            selected_event = points[0].get("y")

        if not selected_event:
            selected_event = st.selectbox("Drill-down Event", sorted(filtered_df["event_name"].unique()))

        event_drill = filtered_df[filtered_df["event_name"] == selected_event]
        drill_cols = st.columns(4)
        drill_cols[0].metric("Event Participants", int(event_drill["participant_id"].nunique()))
        drill_cols[1].metric("Avg Rating", f"{event_drill['rating'].mean():.2f}")
        drill_cols[2].metric("States Reached", int(event_drill["state"].nunique()))
        drill_cols[3].metric("Colleges Involved", int(event_drill["college_name"].nunique()))

    with tab_map:
        st.markdown('<div class="section-title">🗺️ India Heatmap</div>', unsafe_allow_html=True)
        map_fig, state_summary = india_choropleth(filtered_df)
        st.plotly_chart(map_fig, use_container_width=True)
        st.caption("Top 3 states are highlighted on the map where geospatial points are available.")

        st.dataframe(state_summary, use_container_width=True, height=320)

    with tab_feedback:
        st.markdown('<div class="section-title">💬 Smart Feedback Analysis</div>', unsafe_allow_html=True)

        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            st.plotly_chart(sentiment_pie_chart(filtered_df), use_container_width=True)
        with feedback_col2:
            keyword_df = keyword_frequency(filtered_df["feedback_clean"], top_n=12)
            st.plotly_chart(keyword_bar_chart(keyword_df), use_container_width=True)

        cloud_fig = wordcloud_figure(filtered_df["feedback_clean"])
        st.pyplot(cloud_fig, clear_figure=True)

        pos_samples, neg_samples = extract_top_feedback_samples(filtered_df, top_n=3)
        sample_cols = st.columns(2)
        with sample_cols[0]:
            st.markdown("### What participants loved ❤️")
            if pos_samples.empty:
                st.info("No strong positive feedback found for current filters.")
            else:
                st.dataframe(pos_samples, use_container_width=True)

        with sample_cols[1]:
            st.markdown("### Areas to improve ⚠️")
            if neg_samples.empty:
                st.info("No negative feedback found for current filters.")
            else:
                st.dataframe(neg_samples, use_container_width=True)

    with tab_rating:
        st.markdown('<div class="section-title">⭐ Rating Analytics</div>', unsafe_allow_html=True)
        rate_col1, rate_col2 = st.columns(2)

        with rate_col1:
            st.plotly_chart(rating_histogram(filtered_df), use_container_width=True)
        with rate_col2:
            st.plotly_chart(avg_rating_event_chart(filtered_df), use_container_width=True)

        avg_by_event = filtered_df.groupby("event_name", as_index=False)["rating"].mean().sort_values("rating")
        worst_event = avg_by_event.iloc[0]
        best_event = avg_by_event.iloc[-1]

        result_cols = st.columns(2)
        result_cols[0].success(
            f"Best event: {best_event['event_name']} ({best_event['rating']:.2f}/5)"
        )
        result_cols[1].warning(
            f"Needs improvement: {worst_event['event_name']} ({worst_event['rating']:.2f}/5)"
        )


if __name__ == "__main__":
    render()
