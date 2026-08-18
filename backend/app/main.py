"""FastAPI inference service for the COMP-4450 Toxic Comment Moderation project."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .config import Settings
from .database import PredictionRepository
from .model_service import ModelService
from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    PredictRequest,
    PredictResponse,
)


def create_app(
    model_service: ModelService | None = None,
    repository: PredictionRepository | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    injected = model_service is not None and repository is not None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if injected:
            app.state.settings = settings
            app.state.model_service = model_service
            app.state.repository = repository
        else:
            runtime_settings = settings or Settings.from_env()
            app.state.settings = runtime_settings
            app.state.model_service = ModelService.from_settings(runtime_settings)
            app.state.repository = PredictionRepository(
                table_name=runtime_settings.dynamodb_table,
                region_name=runtime_settings.aws_region,
                endpoint_url=runtime_settings.dynamodb_endpoint_url,
            )
        yield

    app = FastAPI(
        title="COMP-4450 Toxic Comment Moderation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        service = app.state.model_service
        runtime_settings = app.state.settings
        return {
            "status": "healthy",
            "model_ref": service.model_ref,
            "model_version": service.model_version,
            "database_table": (
                runtime_settings.dynamodb_table
                if runtime_settings is not None
                else getattr(app.state.repository, "table_name", "injected-test-repository")
            ),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        service = app.state.model_service
        repository_instance = app.state.repository
        runtime_settings = app.state.settings

        started = perf_counter()
        prediction = service.predict(request.text)
        inference_latency_ms = round((perf_counter() - started) * 1000, 3)

        prediction_id = str(uuid4())
        timestamp = datetime.now(UTC).isoformat()
        record = {
            "prediction_id": prediction_id,
            "timestamp": timestamp,
            "text_length": len(request.text),
            "predicted_labels": prediction.predicted_labels,
            "is_toxic": bool(prediction.predicted_labels),
            "probabilities": prediction.probabilities,
            "inference_latency_ms": inference_latency_ms,
            "model_ref": service.model_ref,
            "model_version": service.model_version,
            "feedback_received": False,
        }
        if runtime_settings is None or runtime_settings.store_raw_text:
            record["request_text"] = request.text

        repository_instance.save_prediction(record)
        return PredictResponse(
            prediction_id=prediction_id,
            predicted_labels=prediction.predicted_labels,
            is_toxic=bool(prediction.predicted_labels),
            probabilities=prediction.probabilities,
            inference_latency_ms=inference_latency_ms,
            model_ref=service.model_ref,
            model_version=service.model_version,
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        try:
            app.state.repository.add_feedback(
                prediction_id=request.prediction_id,
                is_correct=request.is_correct,
                corrected_labels=request.corrected_labels,
                feedback_timestamp=datetime.now(UTC).isoformat(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Prediction not found") from exc
        return FeedbackResponse(prediction_id=request.prediction_id, status="feedback_saved")

    return app


app = create_app()
