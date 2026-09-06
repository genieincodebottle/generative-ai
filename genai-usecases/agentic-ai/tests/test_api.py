"""Routing-layer tests. No API key, no network."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from services import providers


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["ollama_reachable"], bool)


def test_catalogue_returns_providers_and_apps(client):
    body = client.get("/catalogue").json()
    assert body["apps"]
    assert all({"id", "label", "family", "blurb", "fields"} <= set(a)
               for a in body["apps"])


def test_catalogue_only_lists_providers_that_are_usable(client, monkeypatch):
    monkeypatch.setattr(providers, "available_providers", lambda: ["Ollama"])
    body = client.get("/catalogue").json()
    assert [p["id"] for p in body["providers"]] == ["Ollama"]
    assert body["providers"][0]["needs_key"] is False


def test_unknown_app_is_404(client):
    response = client.post("/apps/nope/run", json={
        "provider": "Gemini", "model": "gemini-flash-latest", "inputs": {},
    })
    assert response.status_code == 404
    assert "GET /catalogue" in response.json()["detail"]


def test_unknown_provider_is_400(client):
    response = client.post("/apps/query_routing/run", json={
        "provider": "Nope", "model": "x", "inputs": {"query": "hi"},
    })
    assert response.status_code == 400


def test_missing_required_input_is_422_and_names_the_field(client):
    response = client.post("/apps/query_routing/run", json={
        "provider": "Gemini", "model": "gemini-flash-latest", "inputs": {},
    })
    assert response.status_code == 422
    assert "query" in response.json()["detail"]


def test_blank_required_input_is_also_rejected(client):
    # Whitespace is not an input.
    response = client.post("/apps/query_routing/run", json={
        "provider": "Gemini", "model": "gemini-flash-latest",
        "inputs": {"query": "   "},
    })
    assert response.status_code == 422


def test_optional_fields_are_not_required(client):
    # document_processing has optional filename/output_fmt; only `content`
    # is required, so omitting the optional ones must not 422.
    response = client.post("/apps/document_processing/run", json={
        "provider": "Gemini", "model": "gemini-flash-latest", "inputs": {},
    })
    assert response.status_code == 422
    assert "content" in response.json()["detail"]
    assert "filename" not in response.json()["detail"]
