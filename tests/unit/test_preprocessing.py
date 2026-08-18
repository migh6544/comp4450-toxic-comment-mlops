import pytest

from backend.app.preprocessing import labels_from_probabilities, normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello   world\n") == "hello world"


def test_normalize_text_rejects_blank_text():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_text("   \n\t")


def test_labels_from_probabilities_uses_threshold():
    probabilities = {
        "toxic": 0.9,
        "severe_toxic": 0.1,
        "obscene": 0.2,
        "threat": 0.01,
        "insult": 0.75,
        "identity_hate": 0.05,
    }
    assert labels_from_probabilities(probabilities, 0.5) == ["toxic", "insult"]
