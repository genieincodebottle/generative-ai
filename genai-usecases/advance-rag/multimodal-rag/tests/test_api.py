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


def test_catalogue_lists_rolling_aliases_first(client):
    body = client.get("/catalogue").json()
    assert body["main_models"][0] == "gemini-pro-latest"
    assert body["vision_models"][0] == "gemini-flash-latest"


def test_status_before_indexing(client):
    assert client.get("/status").json()["ready"] is False


def test_query_before_indexing_is_409(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 409


def test_query_rejects_blank_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


@pytest.mark.parametrize("k", [0, 99])
def test_query_rejects_out_of_range_k(client, k):
    assert client.post("/query", json={"question": "hi", "k": k}).status_code == 422


def test_configure_rejects_out_of_range_temperature(client):
    assert client.post("/configure", json={"temperature": 5.0}).status_code == 422


def test_documents_rejects_unsupported_type(client):
    response = client.post(
        "/documents", files=[("files", ("notes.txt", b"x", "text/plain"))]
    )
    assert response.status_code in (422, 503)
