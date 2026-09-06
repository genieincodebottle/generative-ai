"""Routing-layer tests. No API key, no network."""

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
    assert set(body) >= {"google_key", "tavily_key", "configured"}


def test_models_lists_rolling_aliases_first(client):
    body = client.get("/models").json()
    # Pinned IDs rot; the alias is what keeps a fresh clone working.
    assert body["llm_models"][0] == "gemini-flash-latest"
    assert "models/gemini-embedding-001" in body["embedding_models"]


def test_models_reports_web_search_availability(client, monkeypatch):
    monkeypatch.setattr(config, "has_tavily_key", lambda: False)
    assert client.get("/models").json()["web_search_available"] is False


def test_configure_rejects_overlap_not_smaller_than_chunk_size(client):
    response = client.post("/configure", json={"chunk_size": 500, "chunk_overlap": 500})
    assert response.status_code == 422
    assert "chunk_overlap" in response.json()["detail"]


def test_configure_rejects_out_of_range_temperature(client):
    assert client.post("/configure", json={"temperature": 5.0}).status_code == 422


def test_configure_rejects_out_of_range_k(client):
    assert client.post("/configure", json={"k_retrieval": 0}).status_code == 422


def test_configure_without_a_key_is_503(client, monkeypatch):
    monkeypatch.setattr("services.system_manager.google_api_key", lambda: None)
    response = client.post("/configure", json={})
    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents", files=[("files", ("notes.txt", b"hi", "text/plain"))]
    )
    assert response.status_code in (422, 503)


def test_status_is_reachable_before_configuration(client):
    assert "configured" in client.get("/status").json()
