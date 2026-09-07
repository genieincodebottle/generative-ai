"""Routing-layer tests. No key, no model download, no network."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from services import config


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["backends"], list)


def test_catalogue_lists_every_backend_even_without_a_key(client, monkeypatch):
    # The reader needs to know what they are missing and where to get it.
    monkeypatch.setattr(config, "api_key_for", lambda b: None)
    body = client.get("/catalogue").json()
    assert body["backends"] == []
    assert {b["id"] for b in body["all_backends"]} == {"groq", "huggingface"}
    assert all(b["key_url"] for b in body["all_backends"])


def test_catalogue_reports_both_model_sizes(client):
    body = client.get("/catalogue").json()
    for backend in body["all_backends"]:
        assert backend["sizes"] == ["22M", "86M"]


def test_classify_rejects_blank_text(client):
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_classify_rejects_overlong_text(client):
    response = client.post("/classify", json={"text": "x" * 20000})
    assert response.status_code == 422


def test_classify_rejects_out_of_range_threshold(client):
    response = client.post("/classify", json={"text": "hello", "threshold": 5.0})
    assert response.status_code == 422


def test_unknown_backend_is_503_not_a_verdict(client):
    # Critically: never a 200 with "BENIGN".
    response = client.post("/classify", json={"text": "hello", "backend": "nope"})
    assert response.status_code == 503
    assert "label" not in response.json()


def test_missing_key_is_503_not_a_verdict(client, monkeypatch):
    monkeypatch.setattr("services.guard_service.api_key_for", lambda b: None)
    response = client.post("/classify", json={"text": "hello", "backend": "groq"})
    assert response.status_code == 503
    assert "label" not in response.json()


def test_an_unusable_score_is_502_and_never_benign(client, monkeypatch):
    """The whole point of this project's error handling.

    If the classifier answers with something that is not a score, the API
    must not return a 200 saying the text is safe.
    """
    import services.guard_service as guard

    def broken(*_args, **_kwargs):
        raise guard.ClassifierError("returned 'oops', which is not a score")

    monkeypatch.setattr(guard, "classify_with_groq", broken)
    monkeypatch.setattr(guard, "api_key_for", lambda b: "key")

    response = client.post("/classify", json={"text": "hello", "backend": "groq"})
    assert response.status_code == 502
    body = response.json()
    assert "label" not in body
    assert "is_malicious" not in body
