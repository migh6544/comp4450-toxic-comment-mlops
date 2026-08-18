from dataclasses import dataclass

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.model_service import ModelPrediction


@dataclass
class FakeModelService:
    model_ref: str = "wandb-registry-Models/toxic-comment-classifier:production"
    model_version: str = "v3"

    def predict(self, text: str) -> ModelPrediction:
        return ModelPrediction(
            probabilities={
                "toxic": 0.90,
                "severe_toxic": 0.03,
                "obscene": 0.10,
                "threat": 0.01,
                "insult": 0.75,
                "identity_hate": 0.02,
            },
            predicted_labels=["toxic", "insult"],
        )


class FakeRepository:
    table_name = "fake-predictions"

    def __init__(self):
        self.predictions = []
        self.feedback = {}

    def save_prediction(self, record):
        self.predictions.append(record)

    def add_feedback(self, prediction_id, is_correct, corrected_labels, feedback_timestamp):
        if prediction_id == "missing":
            raise KeyError(prediction_id)
        self.feedback[prediction_id] = {
            "is_correct": is_correct,
            "corrected_labels": corrected_labels,
            "feedback_timestamp": feedback_timestamp,
        }


def make_client():
    repository = FakeRepository()
    app = create_app(model_service=FakeModelService(), repository=repository)
    return TestClient(app), repository


def test_health_endpoint():
    client, _ = make_client()
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "v3"


def test_predict_endpoint_logs_exactly_one_prediction():
    client, repository = make_client()
    with client:
        response = client.post("/predict", json={"text": "You are useless."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_labels"] == ["toxic", "insult"]
    assert len(repository.predictions) == 1
    assert repository.predictions[0]["prediction_id"] == payload["prediction_id"]
    assert "timestamp" in repository.predictions[0]


def test_predict_rejects_blank_text():
    client, _ = make_client()
    with client:
        response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_feedback_endpoint():
    client, repository = make_client()
    with client:
        prediction = client.post("/predict", json={"text": "You are useless."}).json()
        response = client.post(
            "/feedback",
            json={
                "prediction_id": prediction["prediction_id"],
                "is_correct": False,
                "corrected_labels": ["insult"],
            },
        )
    assert response.status_code == 200
    assert repository.feedback[prediction["prediction_id"]]["corrected_labels"] == ["insult"]


def test_feedback_rejects_unknown_label():
    client, _ = make_client()
    with client:
        response = client.post(
            "/feedback",
            json={
                "prediction_id": "abc",
                "is_correct": False,
                "corrected_labels": ["not_a_real_label"],
            },
        )
    assert response.status_code == 422


def test_feedback_missing_prediction_returns_404():
    client, _ = make_client()
    with client:
        response = client.post(
            "/feedback",
            json={
                "prediction_id": "missing",
                "is_correct": True,
            },
        )
    assert response.status_code == 404
