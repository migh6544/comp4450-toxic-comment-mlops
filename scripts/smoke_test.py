"""Simple live API smoke test."""

from __future__ import annotations

import os

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

health = requests.get(f"{API_URL}/health", timeout=15)
health.raise_for_status()
print("HEALTH:", health.json())

prediction = requests.post(
    f"{API_URL}/predict",
    json={"text": "This is a deliberately rude example for a moderation test."},
    timeout=30,
)
prediction.raise_for_status()
print("PREDICTION:", prediction.json())
