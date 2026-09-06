"""Routing layer and Cypher cleaning. No API key, no database, no network."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from services import config
from services.qa_service import clean_cypher


@pytest.fixture
def client():
    return TestClient(app)


class TestCleanCypher:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("MATCH (m) RETURN m", "MATCH (m) RETURN m"),
            ("```cypher\nMATCH (m) RETURN m\n```", "MATCH (m) RETURN m"),
            ("```\nMATCH (m) RETURN m\n```", "MATCH (m) RETURN m"),
            ("Cypher: MATCH (m) RETURN m", "MATCH (m) RETURN m"),
            ("cypher:\nMATCH (m) RETURN m", "MATCH (m) RETURN m"),
            ("MATCH (m) RETURN m;", "MATCH (m) RETURN m"),
            ("  MATCH (m) RETURN m  ", "MATCH (m) RETURN m"),
        ],
    )
    def test_strips_model_scaffolding(self, raw, expected):
        assert clean_cypher(raw) == expected


class TestRoutes:
    def test_health_reports_providers_and_graph(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "graph" in body

    def test_catalogue_lists_a_rolling_alias_first(self, client):
        body = client.get("/catalogue").json()
        if "Gemini" in body["models"]:
            assert body["models"]["Gemini"][0] == "gemini-flash-latest"
        # Key URLs are returned even for providers with no key, so the UI can
        # tell the reader where to go.
        assert body["key_urls"]

    def test_catalogue_only_lists_configured_providers(self, client, monkeypatch):
        monkeypatch.setattr(config, "api_key_for", lambda p: None)
        assert client.get("/catalogue").json()["providers"] == []

    def test_ask_rejects_unknown_provider(self, client):
        response = client.post("/ask", json={
            "question": "hi", "provider": "Nope", "model": "x",
        })
        assert response.status_code == 400

    def test_ask_rejects_blank_question(self, client):
        response = client.post("/ask", json={
            "question": "", "provider": "Gemini", "model": "gemini-flash-latest",
        })
        assert response.status_code == 422

    def test_ask_rejects_an_overlong_question(self, client):
        response = client.post("/ask", json={
            "question": "x" * 5000, "provider": "Gemini",
            "model": "gemini-flash-latest",
        })
        assert response.status_code == 422


class TestTheGateNotJustThePrompt:
    """The prompt asks for read-only Cypher; the gate enforces it.

    A prompt instruction is a request the model may decline. This drives the
    service with a model that returns a DELETE and checks that nothing ever
    reaches the database.
    """

    def test_a_write_query_is_refused_before_execution(self, monkeypatch):
        import services.qa_service as qa

        executed = []

        from langchain_core.runnables import Runnable

        class Reply:
            content = "MATCH (s:Supplier) DETACH DELETE s"

        class WritingLLM(Runnable):
            """Returns a destructive query no matter what it is asked.

            A real Runnable: LangChain validates the right-hand side of `|`
            and rejects anything that is not one.
            """

            def invoke(self, input, config=None, **kwargs):
                return Reply()

        class SpyGraph:
            def query(self, cypher, *a, **k):
                executed.append(cypher)
                return []

        monkeypatch.setattr(qa, "get_llm", lambda *a, **k: WritingLLM())
        monkeypatch.setattr(qa, "graph_schema", lambda *a, **k: "(schema)")
        monkeypatch.setattr(qa, "get_graph", lambda *a, **k: SpyGraph())

        result = qa.ask("delete everything", "Gemini", "gemini-flash-latest")

        assert result.success is False
        assert "write operation" in result.error
        assert executed == [], (
            f"the gate let a query through to the database: {executed}"
        )
