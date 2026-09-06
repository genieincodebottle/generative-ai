"""Routing-layer tests. No API key, no network."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_reports_keys_and_features(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["keys"]) == {"groq", "google"}


def test_catalogue_lists_models_and_key_urls(client):
    body = client.get("/catalogue").json()
    assert body["models"][0].startswith("meta-llama/llama-4")
    # Only vision-capable models may be offered for images.
    assert all("llama-4" in m for m in body["vision_models"])
    assert body["gemini_models"][0] == "gemini-flash-latest"
    assert set(body["key_urls"]) == {"groq", "google"}


def test_chat_rejects_empty_messages(client):
    assert client.post("/chat", json={"messages": []}).status_code == 422


def test_chat_rejects_an_invalid_role(client):
    response = client.post(
        "/chat", json={"messages": [{"role": "wizard", "content": "hi"}]}
    )
    assert response.status_code == 422


def test_chat_rejects_out_of_range_temperature(client):
    response = client.post("/chat", json={
        "messages": [{"role": "user", "content": "hi"}], "temperature": 9,
    })
    assert response.status_code == 422


def test_chat_without_a_key_is_503_and_carries_no_answer(client, monkeypatch):
    monkeypatch.setattr("services.llama_service.api_key", lambda n: None)
    response = client.post(
        "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 503
    assert "text" not in response.json()


def test_search_before_indexing_is_409(client):
    assert client.post("/search", json={"query": "hi"}).status_code == 409


def test_search_rejects_out_of_range_top_k(client):
    response = client.post("/search", json={"query": "hi", "top_k": 99})
    assert response.status_code == 422


def test_documents_status_is_safe_when_empty(client):
    body = client.get("/documents").json()
    assert body["indexed"] is False
    assert body["total_chunks"] == 0


def test_vision_rejects_a_non_image(client):
    response = client.post(
        "/vision",
        files=[("file", ("notes.txt", b"hello", "text/plain"))],
        data={"prompt": "read this"},
    )
    assert response.status_code in (422, 503)
