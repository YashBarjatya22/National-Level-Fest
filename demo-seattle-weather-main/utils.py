from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from textblob import TextBlob


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "very",
    "event",
    "fest",
    "session",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 2]
    return " ".join(tokens)


def sentiment_label(text: str) -> str:
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


def sentiment_polarity(text: str) -> float:
    return TextBlob(str(text)).sentiment.polarity


def preprocess_feedback(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df["feedback_clean"] = work_df["feedback_text"].fillna("").apply(clean_text)
    work_df["sentiment"] = work_df["feedback_text"].fillna("").apply(sentiment_label)
    work_df["polarity"] = work_df["feedback_text"].fillna("").apply(sentiment_polarity)
    return work_df


def keyword_frequency(series: pd.Series, top_n: int = 15) -> pd.DataFrame:
    words: list[str] = []
    for text in series.fillna(""):
        words.extend(clean_text(text).split())

    counter = Counter(words)
    common = counter.most_common(top_n)
    return pd.DataFrame(common, columns=["keyword", "count"])


def extract_top_feedback_samples(df: pd.DataFrame, top_n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_samples = (
        df[df["sentiment"] == "Positive"]
        .sort_values("polarity", ascending=False)
        [["event_name", "state", "feedback_text", "rating"]]
        .head(top_n)
    )

    negative_samples = (
        df[df["sentiment"] == "Negative"]
        .sort_values("polarity", ascending=True)
        [["event_name", "state", "feedback_text", "rating"]]
        .head(top_n)
    )

    return positive_samples, negative_samples


def auto_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if df.empty:
        return ["No data available for current filters."]

    event_counts = df["event_name"].value_counts()
    state_counts = df["state"].value_counts()
    rating_by_event = df.groupby("event_name", as_index=True)["rating"].mean().sort_values(ascending=False)

    top_event = event_counts.index[0]
    top_state = state_counts.index[0]
    highest_rated = rating_by_event.index[0]

    insights.append(f"Event {top_event} has the highest engagement with {event_counts.iloc[0]} participants.")
    insights.append(f"State {top_state} dominates participation with {state_counts.iloc[0]} registrations.")
    insights.append(f"Feedback indicates strong satisfaction in {highest_rated} with average rating {rating_by_event.iloc[0]:.2f}.")

    return insights
