"""Text validation and post-processing helpers."""

from __future__ import annotations

TOXICITY_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = " ".join(text.strip().split())
    if not normalized:
        raise ValueError("text must not be empty")
    return normalized


def labels_from_probabilities(
    probabilities: dict[str, float], threshold: float
) -> list[str]:
    return [
        label
        for label in TOXICITY_LABELS
        if float(probabilities.get(label, 0.0)) >= threshold
    ]
