"""User-facing Streamlit application."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
LABEL_DISPLAY = {
    "toxic": "Toxic",
    "severe_toxic": "Severe toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
}

st.set_page_config(page_title="Toxic Comment Moderation", page_icon="🛡️")
st.title("🛡️ Toxic Comment Moderation")
st.caption("COMP-4450 Final Course Project")

text = st.text_area("Enter a comment to moderate", height=160, max_chars=20_000)

if st.button("Analyze", type="primary", disabled=not text.strip()):
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        response.raise_for_status()
        st.session_state["prediction"] = response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")

prediction = st.session_state.get("prediction")
if prediction:
    st.subheader("Prediction")
    if prediction["predicted_labels"]:
        st.warning(
            "Detected: "
            + ", ".join(LABEL_DISPLAY[label] for label in prediction["predicted_labels"])
        )
    else:
        st.success("No toxicity label exceeded the model threshold.")

    probability_rows = [
        {"Label": LABEL_DISPLAY[label], "Probability": probability}
        for label, probability in prediction["probabilities"].items()
    ]
    st.dataframe(probability_rows, hide_index=True, width="stretch")
    st.caption(
        f"Prediction ID: {prediction['prediction_id']} · "
        f"Inference latency: {prediction['inference_latency_ms']} ms · "
        f"Model version: {prediction['model_version']}"
    )

    st.subheader("Human review")
    feedback_choice = st.radio(
        "Was the moderation prediction correct?",
        ["Select", "Yes", "No"],
        horizontal=True,
        key=f"feedback_{prediction['prediction_id']}",
    )

    corrected_labels: list[str] | None = None
    if feedback_choice == "No":
        corrected_labels = st.multiselect(
            "Select the correct toxicity labels. Leave all unselected for non-toxic.",
            options=list(LABEL_DISPLAY),
            format_func=lambda label: LABEL_DISPLAY[label],
        )

    if feedback_choice in {"Yes", "No"} and st.button("Submit feedback"):
        payload = {
            "prediction_id": prediction["prediction_id"],
            "is_correct": feedback_choice == "Yes",
        }
        if feedback_choice == "No":
            payload["corrected_labels"] = corrected_labels or []
        try:
            response = requests.post(f"{API_URL}/feedback", json=payload, timeout=30)
            response.raise_for_status()
            st.success("Feedback saved.")
        except requests.RequestException as exc:
            st.error(f"Feedback request failed: {exc}")
