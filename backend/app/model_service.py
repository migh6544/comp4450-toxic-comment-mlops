"""Model loading and inference service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .config import Settings
from .preprocessing import TOXICITY_LABELS, labels_from_probabilities, normalize_text


@dataclass(frozen=True)
class ModelPrediction:
    probabilities: dict[str, float]
    predicted_labels: list[str]


class ModelService:
    def __init__(
        self,
        bundle: dict,
        model_ref: str,
        model_version: str,
    ) -> None:
        required = {"pipeline", "labels", "threshold"}
        missing = required - set(bundle)
        if missing:
            raise ValueError(f"Model bundle is missing required keys: {sorted(missing)}")
        labels = list(bundle["labels"])
        if labels != TOXICITY_LABELS:
            raise ValueError(f"Unexpected model labels: {labels}")
        self.pipeline = bundle["pipeline"]
        self.labels = labels
        self.threshold = float(bundle["threshold"])
        self.model_ref = model_ref
        self.model_version = model_version

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelService:
        if settings.model_source == "local":
            return cls.from_local(settings.local_model_path)
        return cls.from_wandb(settings)

    @classmethod
    def from_local(cls, path: str) -> ModelService:
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Local model bundle not found: {model_path}")
        bundle = joblib.load(model_path)
        return cls(bundle=bundle, model_ref=str(model_path), model_version="local")

    @classmethod
    def from_wandb(cls, settings: Settings) -> ModelService:
        import wandb

        artifact_ref = (
            f"wandb-registry-{settings.wandb_registry}/"
            f"{settings.wandb_collection}:{settings.wandb_model_alias}"
        )
        api = wandb.Api(overrides={"entity": settings.wandb_entity})
        artifact = api.artifact(name=artifact_ref)
        download_dir = artifact.download(root=settings.model_cache_dir)
        model_path = Path(download_dir) / "toxic_comment_model.joblib"
        if not model_path.exists():
            candidates = list(Path(download_dir).glob("*.joblib"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "Could not identify the .joblib model file in the downloaded Registry artifact."
                )
            model_path = candidates[0]
        bundle = joblib.load(model_path)
        return cls(
            bundle=bundle,
            model_ref=artifact_ref,
            model_version=getattr(artifact, "version", "unknown"),
        )

    def predict(self, text: str) -> ModelPrediction:
        normalized = normalize_text(text)
        raw = np.asarray(self.pipeline.predict_proba([normalized]), dtype=float)[0]
        probabilities = {
            label: round(float(raw[index]), 6) for index, label in enumerate(self.labels)
        }
        predicted_labels = labels_from_probabilities(probabilities, self.threshold)
        return ModelPrediction(
            probabilities=probabilities,
            predicted_labels=predicted_labels,
        )
