"""Generate small demonstration traffic for dashboard screenshots."""

from __future__ import annotations

import os
import random
import time

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
COMMENTS = [
    "Thank you for the helpful explanation.",
    "I disagree with your argument, but I understand your point.",
    "That was a terrible and useless response.",
    "You are acting like an idiot.",
    "This is a normal sentence for testing the service.",
    "Please stop posting insulting comments.",
    "Your answer is awful.",
    "I appreciate the update.",
]


def main() -> None:
    count = int(os.getenv("DEMO_TRAFFIC_COUNT", "40"))
    for index in range(count):
        text = random.choice(COMMENTS)
        response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if index % 3 == 0:
            feedback = {
                "prediction_id": payload["prediction_id"],
                "is_correct": True,
            }
            requests.post(f"{API_URL}/feedback", json=feedback, timeout=30).raise_for_status()

        print(index + 1, payload["predicted_labels"], payload["inference_latency_ms"])
        time.sleep(0.15)


if __name__ == "__main__":
    main()
