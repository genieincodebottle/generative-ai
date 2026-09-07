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
    assert "sessions" in body and "max_sessions" in body


def test_providers_only_lists_configured_keys(client, monkeypatch):
    monkeypatch.setattr(config, "api_key_for", lambda p: None)
    assert client.get("/providers").json()["providers"] == []


def test_create_session_rejects_unknown_provider(client):
    response = client.post(
        "/sessions",
        files=[("files", ("a.pdf", b"%PDF-1.4", "application/pdf"))],
        data={"provider": "Nope", "model": "x"},
    )
    assert response.status_code == 400


def test_create_session_rejects_overlap_not_smaller_than_chunk_size(client):
    # Overlap >= chunk size makes the splitter loop or produce nonsense, so it
    # is rejected up front rather than surfacing as a confusing failure later.
    response = client.post(
        "/sessions",
        files=[("files", ("a.pdf", b"%PDF-1.4", "application/pdf"))],
        data={"provider": "Gemini", "model": "gemini-flash-latest",
              "chunk_size": 500, "chunk_overlap": 500},
    )
    assert response.status_code == 422
    assert "chunk_overlap" in response.json()["detail"]


def test_create_session_rejects_non_pdf(client):
    response = client.post(
        "/sessions",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
        data={"provider": "Gemini", "model": "gemini-flash-latest"},
    )
    assert response.status_code == 422
    assert "not a PDF" in response.json()["detail"]


def test_chat_on_unknown_session_is_404(client):
    response = client.post("/sessions/does-not-exist/chat", json={"question": "hi"})
    assert response.status_code == 404


def test_chat_rejects_empty_question(client):
    response = client.post("/sessions/whatever/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_rejects_out_of_range_retriever_k(client):
    response = client.post(
        "/sessions/whatever/chat", json={"question": "hi", "retriever_k": 99}
    )
    assert response.status_code == 422


def test_history_on_unknown_session_is_404(client):
    assert client.get("/sessions/nope/history").status_code == 404


def test_delete_unknown_session_is_404(client):
    assert client.delete("/sessions/nope").status_code == 404


def test_clear_history_on_unknown_session_is_404(client):
    assert client.delete("/sessions/nope/history").status_code == 404
