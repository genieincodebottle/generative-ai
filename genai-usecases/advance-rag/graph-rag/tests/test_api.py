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


def test_catalogue_lists_the_three_retrievers(client):
    body = client.get("/catalogue").json()
    ids = [r["id"] for r in body["retrievers"]]
    assert ids == ["traversal", "standard", "hybrid"]


def test_catalogue_lists_rolling_alias_first(client):
    # Pinned IDs rot; the alias is what keeps a fresh clone working.
    assert client.get("/catalogue").json()["llm_models"][0] == "gemini-flash-latest"


def test_status_is_reachable_before_anything_is_loaded(client):
    body = client.get("/status").json()
    assert body["ready"] is False


def test_configure_rejects_overlap_not_smaller_than_chunk_size(client):
    response = client.post("/configure", json={"chunk_size": 500, "chunk_overlap": 500})
    assert response.status_code == 422
    assert "chunk_overlap" in response.json()["detail"]


@pytest.mark.parametrize("field, value", [("k_retrieval", 0), ("max_depth", 9),
                                          ("chunk_size", 10)])
def test_configure_rejects_out_of_range(client, field, value):
    assert client.post("/configure", json={field: value}).status_code == 422


def test_query_before_loading_is_409(client):
    # The user has not loaded documents. That is their state, not a fault.
    response = client.post("/query", json={"question": "hi", "retriever": "traversal"})
    assert response.status_code == 409


def test_query_rejects_unknown_retriever(client):
    response = client.post("/query", json={"question": "hi", "retriever": "magic"})
    assert response.status_code == 400


def test_query_rejects_blank_question(client):
    response = client.post("/query", json={"question": "", "retriever": "traversal"})
    assert response.status_code == 422


def test_documents_rejects_unsupported_type(client):
    response = client.post(
        "/documents", files=[("files", ("slides.pptx", b"x", "application/octet-stream"))]
    )
    assert response.status_code in (422, 503)
