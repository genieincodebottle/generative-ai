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
    assert "sessions" in body


def test_catalogue_lists_all_five_techniques(client):
    body = client.get("/catalogue").json()
    ids = [t["id"] for t in body["techniques"]]
    assert ids == ["basic", "adaptive", "corrective", "hybrid", "reranking"]


def test_catalogue_declares_which_options_each_technique_takes(client):
    body = client.get("/catalogue").json()
    by_id = {t["id"]: t for t in body["techniques"]}
    assert by_id["basic"]["options"] == ["top_k"]
    assert by_id["hybrid"]["options"] == ["bm25_weight", "vector_weight"]
    assert by_id["adaptive"]["options"] == []


def test_catalogue_only_lists_providers_with_keys(client, monkeypatch):
    monkeypatch.setattr(config, "api_key_for", lambda p: None)
    body = client.get("/catalogue").json()
    assert body["providers"] == []
    # Key URLs are still returned so the UI can tell the user where to go.
    assert body["key_urls"]


def test_create_session_rejects_unknown_provider(client):
    response = client.post("/sessions", data={"provider": "Nope"})
    assert response.status_code == 400


def test_create_session_rejects_bad_file_type(client):
    response = client.post(
        "/sessions",
        data={"provider": "Gemini (Google)"},
        files=[("files", ("notes.docx", b"x", "application/octet-stream"))],
    )
    assert response.status_code in (422, 503)


def test_query_on_unknown_session_is_404(client):
    response = client.post("/sessions/nope/query", json={
        "query": "hi", "technique": "basic", "model": "gemini-flash-latest",
    })
    assert response.status_code == 404


def test_query_rejects_unknown_technique(client):
    response = client.post("/sessions/whatever/query", json={
        "query": "hi", "technique": "magic", "model": "gemini-flash-latest",
    })
    assert response.status_code == 400


def test_query_rejects_blank_query(client):
    response = client.post("/sessions/whatever/query", json={
        "query": "", "technique": "basic", "model": "gemini-flash-latest",
    })
    assert response.status_code == 422


@pytest.mark.parametrize("field, value", [("top_k", 0), ("top_k", 99),
                                          ("temperature", 5.0), ("bm25_weight", 2.0)])
def test_query_rejects_out_of_range_options(client, field, value):
    payload = {"query": "hi", "technique": "basic", "model": "gemini-flash-latest",
               field: value}
    assert client.post("/sessions/whatever/query", json=payload).status_code == 422


def test_delete_unknown_session_is_404(client):
    assert client.delete("/sessions/nope").status_code == 404
