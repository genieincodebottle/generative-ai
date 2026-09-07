"""Routing-layer tests. The database is redirected to a temp file per test."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from services import config, database


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point every module that resolves DB_PATH at a throwaway file."""
    path = tmp_path / "api-test.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(database, "DB_PATH", path)
    return path


@pytest.fixture
def client():
    return TestClient(app)


def test_health_is_ok_even_without_a_database(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database_ready"] is False


def test_database_reports_not_ready_before_init(client):
    assert client.get("/database").json()["ready"] is False


def test_init_then_status(client):
    init = client.post("/database/init").json()
    assert init["seeded"] == 10
    status = client.get("/database").json()
    assert status == {"ready": True, "calls": 10, "tagged": 0}


def test_calls_before_init_is_409_not_500(client):
    # An uninitialized database is the user's state, not a server fault.
    assert client.get("/calls").status_code == 409


def test_calls_after_init(client):
    client.post("/database/init")
    calls = client.get("/calls").json()
    assert len(calls) == 10
    assert calls[0]["customer_id"] == 101


def test_reset_removes_database(client):
    client.post("/database/init")
    assert client.delete("/database").status_code == 204
    assert client.get("/database").json()["ready"] is False


def test_stats_empty_after_init(client):
    client.post("/database/init")
    assert client.get("/stats").json()["total"] == 0


def test_analyze_rejects_blank_text(client):
    response = client.post("/analyze", json={
        "text": "", "provider": "gemini", "model": "gemini-flash-latest",
    })
    assert response.status_code == 422


def test_analyze_rejects_out_of_range_temperature(client):
    response = client.post("/analyze", json={
        "text": "a long enough transcript here", "provider": "gemini",
        "model": "gemini-flash-latest", "temperature": 3.0,
    })
    assert response.status_code == 422


def test_tag_all_rejects_unknown_provider(client):
    response = client.post("/taggings/run", json={"provider": "nope", "model": "x"})
    assert response.status_code == 400


def test_providers_only_lists_configured_keys(client, monkeypatch):
    monkeypatch.setattr(config, "api_key_for", lambda p: None)
    body = client.get("/providers").json()
    assert body["providers"] == []
    assert body["default_provider"] is None
