"""Environment-based backend configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    aws_region: str
    dynamodb_table: str
    dynamodb_endpoint_url: str | None
    model_source: str
    local_model_path: str
    model_cache_dir: str
    wandb_entity: str | None
    wandb_registry: str
    wandb_collection: str
    wandb_model_alias: str
    store_raw_text: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        endpoint = os.getenv("DYNAMODB_ENDPOINT_URL", "").strip() or None
        settings = cls(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            dynamodb_table=os.getenv(
                "DYNAMODB_TABLE", "comp4450-toxic-comment-predictions"
            ),
            dynamodb_endpoint_url=endpoint,
            model_source=os.getenv("MODEL_SOURCE", "wandb").strip().lower(),
            local_model_path=os.getenv(
                "LOCAL_MODEL_PATH", "artifacts/toxic_comment_model.joblib"
            ),
            model_cache_dir=os.getenv("MODEL_CACHE_DIR", ".model_cache"),
            wandb_entity=os.getenv("WANDB_ENTITY"),
            wandb_registry=os.getenv("WANDB_REGISTRY", "Models"),
            wandb_collection=os.getenv("WANDB_COLLECTION", "toxic-comment-classifier"),
            wandb_model_alias=os.getenv("WANDB_MODEL_ALIAS", "production"),
            store_raw_text=_bool_env("STORE_RAW_TEXT", True),
        )
        if settings.model_source not in {"wandb", "local"}:
            raise ValueError("MODEL_SOURCE must be either 'wandb' or 'local'.")
        if settings.model_source == "wandb" and not settings.wandb_entity:
            raise ValueError("WANDB_ENTITY is required when MODEL_SOURCE=wandb.")
        return settings
