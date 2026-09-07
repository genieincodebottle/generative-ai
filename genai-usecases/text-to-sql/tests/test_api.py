"""Tests for the routing layer, using FastAPI's TestClient.

These exercise the real database (it ships with the repo) but never call an
LLM, so they run offline and in about a second.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_reports_database_and_providers():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert isinstance(body["providers"], list)


def test_database_lists_chinook_tables():
    body = client.get("/database").json()
    assert body["dialect"] == "sqlite"
    assert "artists" in body["tables"]
    assert "invoices" in body["tables"]


def test_table_preview_returns_rows():
    body = client.get("/database/tables/artists", params={"limit": 3}).json()
    assert body["columns"] == ["ArtistId", "Name"]
    assert len(body["rows"]) == 3


def test_table_preview_rejects_unknown_table():
    # The table name is validated against the real schema, so it cannot be
    # used to smuggle SQL into the preview query.
    assert client.get("/database/tables/artists;DROP").status_code == 404


@pytest.mark.parametrize("limit", [0, 101])
def test_table_preview_rejects_out_of_range_limit(limit):
    response = client.get("/database/tables/artists", params={"limit": limit})
    assert response.status_code == 422


def test_query_rejects_unknown_provider():
    response = client.post("/query", json={
        "question": "How many artists?", "provider": "Nope", "model": "x",
    })
    assert response.status_code == 400


def test_query_rejects_empty_question():
    response = client.post("/query", json={
        "question": "", "provider": "Google Gemini", "model": "gemini-flash-latest",
    })
    assert response.status_code == 422


def test_query_rejects_out_of_range_temperature():
    response = client.post("/query", json={
        "question": "How many artists?", "provider": "Google Gemini",
        "model": "gemini-flash-latest", "temperature": 5.0,
    })
    assert response.status_code == 422
