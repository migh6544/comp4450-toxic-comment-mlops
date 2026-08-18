"""Pydantic request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from .preprocessing import TOXICITY_LABELS, normalize_text


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return normalize_text(value)


class PredictResponse(BaseModel):
    prediction_id: str
    predicted_labels: list[str]
    is_toxic: bool
    probabilities: dict[str, float]
    inference_latency_ms: float
    model_ref: str
    model_version: str


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(min_length=1)
    is_correct: bool
    corrected_labels: list[str] | None = None

    @field_validator("corrected_labels")
    @classmethod
    def validate_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        unknown = sorted(set(value) - set(TOXICITY_LABELS))
        if unknown:
            raise ValueError(f"Unknown corrected labels: {unknown}")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_feedback_shape(self) -> FeedbackRequest:
        if self.is_correct and self.corrected_labels is not None:
            raise ValueError("corrected_labels must be omitted when is_correct=true")
        if not self.is_correct and self.corrected_labels is None:
            raise ValueError(
                "corrected_labels is required when is_correct=false; use [] for non-toxic"
            )
        return self


class FeedbackResponse(BaseModel):
    prediction_id: str
    status: str
