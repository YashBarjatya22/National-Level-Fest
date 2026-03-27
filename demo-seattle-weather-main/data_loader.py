from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import streamlit as st


CANONICAL_COLUMNS: Dict[str, Iterable[str]] = {
    "participant_id": ["participant_id", "participant id", "student_id", "student id", "id"],
    "name": ["name", "student name", "participant_name", "participant name"],
    "college_name": ["college_name", "college name", "college", "institute", "institution"],
    "state": ["state", "region", "state_name", "state name"],
    "event_name": ["event_name", "event name", "event", "competition"],
    "rating": ["rating", "score", "stars", "review_rating"],
    "feedback_text": ["feedback_text", "feedback text", "feedback", "feedback on fest", "review"],
    "registration_date": [
        "registration_date",
        "registration date",
        "date",
        "created_at",
        "created at",
        "timestamp",
    ],
}


def _normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().replace("_", " ").split())


def _find_column_match(columns: list[str], aliases: Iterable[str]) -> str | None:
    normalized_map = {_normalize_header(col): col for col in columns}
    for alias in aliases:
        if alias in normalized_map:
            return normalized_map[alias]

    for norm_col, raw_col in normalized_map.items():
        for alias in aliases:
            if alias in norm_col or norm_col in alias:
                return raw_col
    return None


def _resolve_dataset_path() -> Path:
    app_dir = Path(__file__).resolve().parent
    candidate_paths = [
        app_dir / "C5-FestDataset - fest_dataset.csv",
        app_dir.parent / "C5-FestDataset - fest_dataset.csv",
        app_dir / "fest_dataset.csv",
        app_dir.parent / "fest_dataset.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    csv_files = list(app_dir.glob("*.csv")) + list(app_dir.parent.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError("No CSV file found. Place your dataset in the project directory.")


def _generate_sample_data(rows: int = 200) -> pd.DataFrame:
    names = [f"Participant {i}" for i in range(1, rows + 1)]
    states = ["Karnataka", "Tamil Nadu", "Delhi", "Maharashtra", "Gujarat", "Kerala"]
    colleges = [
        "Anna University",
        "IIT Madras",
        "PES University",
        "BITS Pilani",
        "Christ University",
        "Delhi University",
    ]
    events = ["Hackathon", "Coding Challenge", "Project Expo", "Paper Presentation", "UI Design"]
    ratings = [3, 4, 5, 2, 5, 4]

    df = pd.DataFrame(
        {
            "participant_id": [f"P{i:04d}" for i in range(1, rows + 1)],
            "name": names,
            "college_name": [colleges[i % len(colleges)] for i in range(rows)],
            "state": [states[i % len(states)] for i in range(rows)],
            "event_name": [events[i % len(events)] for i in range(rows)],
            "rating": [ratings[i % len(ratings)] for i in range(rows)],
            "feedback_text": [
                "Great event and very informative" if i % 4 else "Needs better scheduling"
                for i in range(rows)
            ],
            "registration_date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        }
    )
    return df


def _to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    mapped_cols: Dict[str, str] = {}
    for canonical_name, aliases in CANONICAL_COLUMNS.items():
        matched = _find_column_match(df.columns.tolist(), aliases)
        if matched is not None:
            mapped_cols[canonical_name] = matched

    canon_df = pd.DataFrame()

    if "participant_id" in mapped_cols:
        canon_df["participant_id"] = df[mapped_cols["participant_id"]].astype(str)
    else:
        canon_df["participant_id"] = [f"P{i:04d}" for i in range(1, len(df) + 1)]

    if "name" in mapped_cols:
        canon_df["name"] = df[mapped_cols["name"]].fillna("Unknown Participant").astype(str)
    else:
        canon_df["name"] = [f"Participant {i}" for i in range(1, len(df) + 1)]

    if "college_name" in mapped_cols:
        canon_df["college_name"] = df[mapped_cols["college_name"]].fillna("Unknown College").astype(str)
    else:
        canon_df["college_name"] = "Unknown College"

    if "state" in mapped_cols:
        canon_df["state"] = df[mapped_cols["state"]].fillna("Unknown").astype(str)
    else:
        canon_df["state"] = "Unknown"

    if "event_name" in mapped_cols:
        canon_df["event_name"] = df[mapped_cols["event_name"]].fillna("Unknown Event").astype(str)
    else:
        canon_df["event_name"] = "Unknown Event"

    if "rating" in mapped_cols:
        canon_df["rating"] = pd.to_numeric(df[mapped_cols["rating"]], errors="coerce").fillna(0)
    else:
        canon_df["rating"] = 0

    if "feedback_text" in mapped_cols:
        canon_df["feedback_text"] = df[mapped_cols["feedback_text"]].fillna("No feedback").astype(str)
    else:
        canon_df["feedback_text"] = "No feedback"

    if "registration_date" in mapped_cols:
        canon_df["registration_date"] = pd.to_datetime(
            df[mapped_cols["registration_date"]], errors="coerce"
        )
    else:
        canon_df["registration_date"] = pd.date_range(
            "2025-01-01", periods=len(df), freq="D"
        )

    canon_df["registration_date"] = canon_df["registration_date"].fillna(pd.Timestamp("2025-01-01"))
    canon_df["rating"] = canon_df["rating"].clip(1, 5)

    for col in ["name", "college_name", "state", "event_name", "feedback_text"]:
        canon_df[col] = canon_df[col].astype(str).str.strip()

    return canon_df


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    try:
        csv_path = _resolve_dataset_path()
        raw_df = pd.read_csv(csv_path)
        if raw_df.empty:
            return _generate_sample_data()
        return _to_canonical(raw_df)
    except Exception:
        # Fallback sample ensures the app remains demonstrable even if user data has issues.
        return _generate_sample_data()
