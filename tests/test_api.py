"""API tests with neural models skipped."""

from __future__ import annotations

import os

os.environ["ESSAYTRACE_SKIP_NEURAL"] = "1"

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE = (
    "I burned the first tray of baklava on a Thursday that already smelled like bleach. "
    "My uncle swore in two languages and then handed me a cracked plastic scraper. "
    "Anyway, I kept the scraper in my backpack for a year. "
    "It made a stupid clicking sound against my chemistry notes. "
    "I still don't think of that year as a lesson about resilience."
)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_detect_returns_range_and_explanations():
    r = client.post("/api/detect", json={"text": SAMPLE})
    assert r.status_code == 200
    data = r.json()
    assert "caveat" in data
    assert "not a verdict" in data["caveat"].lower()
    score = data["essay_score"]
    assert 0 <= score["low"] <= score["point"] <= score["high"] <= 1
    assert data["sentences"]
    hit = data["sentences"][0]
    assert "plain" in hit["explanations"][0]
    assert hit["start"] < hit["end"]


def test_reject_emptyish():
    r = client.post("/api/detect", json={"text": "Hi."})
    assert r.status_code == 400
