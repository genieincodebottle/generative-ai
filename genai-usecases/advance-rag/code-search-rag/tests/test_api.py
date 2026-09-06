"""Routing-layer tests. No API key, no network."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body) >= {"google_key", "ready"}


def test_catalogue_lists_rolling_alias_first(client):
    body = client.get("/catalogue").json()
    assert body["llm_models"][0] == "gemini-flash-latest"
    assert "python" in body["languages"]


def test_status_before_indexing(client):
    assert client.get("/status").json()["ready"] is False


def test_search_before_indexing_is_409(client):
    assert client.post("/search", json={"query": "hi"}).status_code == 409


def test_search_rejects_blank_query(client):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_configure_rejects_final_greater_than_rerank(client):
    # Asking for more final results than survive filtering is incoherent.
    response = client.post("/configure", json={"top_k_rerank": 3, "top_k_final": 10})
    assert response.status_code == 422
    assert "top_k_final" in response.json()["detail"]


@pytest.mark.parametrize("field, value", [("top_k_initial", 0), ("top_k_rerank", 99)])
def test_configure_rejects_out_of_range(client, field, value):
    assert client.post("/configure", json={field: value}).status_code == 422


def test_index_rejects_files_with_no_supported_language(client):
    response = client.post(
        "/index", files=[("files", ("notes.txt", b"hello", "text/plain"))]
    )
    assert response.status_code in (422, 503)
