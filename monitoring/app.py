"""Database-backed production monitoring dashboard."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

import boto3
import numpy as np
import pandas as pd
import streamlit as st

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("DYNAMODB_TABLE", "comp4450-toxic-comment-predictions")
ENDPOINT_URL = os.getenv("DYNAMODB_ENDPOINT_URL", "").strip() or None
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def python_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: python_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [python_value(item) for item in value]
    return value


@st.cache_data(ttl=30)
def load_predictions() -> list[dict[str, Any]]:
    resource = boto3.resource(
        "dynamodb",
        region_name=AWS_REGION,
        endpoint_url=ENDPOINT_URL,
    )
    table = resource.Table(TABLE_NAME)
    response = table.scan()
    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response.get("Items", []))
    return [python_value(item) for item in items]


def label_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    counts = {label: 0 for label in [*LABELS, "non_toxic"]}
    for labels in frame["predicted_labels"]:
        if labels:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        else:
            counts["non_toxic"] += 1
    return pd.DataFrame(
        {"label": list(counts.keys()), "count": list(counts.values())}
    ).set_index("label")


def histogram_frame(values: pd.Series, bins: np.ndarray, column_name: str) -> pd.DataFrame:
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bins)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"bucket": labels, column_name: counts}).set_index("bucket")


st.set_page_config(page_title="Toxicity Model Monitoring", page_icon="📈", layout="wide")
st.title("📈 Toxic Comment Model Monitoring")
st.caption("Production metrics read from Amazon DynamoDB — no shared JSON log files")

if st.button("Refresh data"):
    load_predictions.clear()

try:
    records = load_predictions()
except Exception as exc:  # Streamlit should show a useful deployment error rather than a blank page.
    st.error(f"Could not read DynamoDB table '{TABLE_NAME}': {exc}")
    st.stop()

if not records:
    st.info("No production predictions have been logged yet.")
    st.stop()

frame = pd.DataFrame(records)
frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
frame = frame.sort_values("timestamp").reset_index(drop=True)
frame["inference_latency_ms"] = pd.to_numeric(frame["inference_latency_ms"], errors="coerce")
frame["text_length"] = pd.to_numeric(frame["text_length"], errors="coerce")

if "feedback_received" in frame.columns:
    feedback_mask = frame["feedback_received"].fillna(False).astype(bool)
    feedback = frame[feedback_mask].copy()
else:
    feedback = frame.iloc[0:0].copy()
live_accuracy = float(feedback["is_correct"].mean()) if not feedback.empty else None

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Predictions", f"{len(frame):,}")
col2.metric("Average latency", f"{frame['inference_latency_ms'].mean():.1f} ms")
col3.metric("P95 latency", f"{frame['inference_latency_ms'].quantile(0.95):.1f} ms")
col4.metric("Reviewed", f"{len(feedback):,}")
col5.metric("Live accuracy", "N/A" if live_accuracy is None else f"{live_accuracy:.1%}")

st.subheader("Prediction latency over time")
latency_chart = frame.set_index("timestamp")[["inference_latency_ms"]]
st.line_chart(latency_chart)

st.subheader("Predicted-label distribution (target/prediction drift view)")
st.bar_chart(label_distribution(frame))

if live_accuracy is None:
    st.info("No feedback has been collected yet, so live accuracy is not available.")
else:
    st.subheader("Feedback-derived accuracy over time")
    feedback_chart = feedback[["timestamp", "is_correct"]].copy()
    feedback_chart["is_correct"] = feedback_chart["is_correct"].astype(float)
    feedback_chart["rolling_accuracy"] = feedback_chart["is_correct"].expanding().mean()
    st.line_chart(feedback_chart.set_index("timestamp")[["rolling_accuracy"]])

st.subheader("Input drift: earlier vs recent comment-length distribution")
if len(frame) < 20:
    st.info("At least 20 predictions are recommended before interpreting this drift chart.")
else:
    split_index = max(1, int(len(frame) * 0.8))
    earlier = frame.iloc[:split_index]["text_length"].dropna()
    recent = frame.iloc[split_index:]["text_length"].dropna()
    maximum = max(float(frame["text_length"].max()), 1.0)
    bins = np.linspace(0, maximum, num=11)
    earlier_hist = histogram_frame(earlier, bins, "earlier")
    recent_hist = histogram_frame(recent, bins, "recent")
    st.bar_chart(earlier_hist.join(recent_hist, how="outer").fillna(0))

st.subheader("Model versions observed in production")
version_counts = frame["model_version"].fillna("unknown").value_counts().rename("count")
st.bar_chart(version_counts)

with st.expander("Recent prediction records"):
    columns = [
        column
        for column in [
            "timestamp",
            "prediction_id",
            "predicted_labels",
            "inference_latency_ms",
            "model_version",
            "feedback_received",
            "is_correct",
        ]
        if column in frame.columns
    ]
    st.dataframe(frame[columns].tail(100), hide_index=True, use_container_width=True)
