"""Upload validation, config guards, and the key check. No API key needed."""

import pytest

from services.graph_rag import MissingAPIKey, require_api_key
from services.manager import IngestError, Manager, NotReady, Upload, validate


class TestRequireApiKey:
    def test_raises_a_named_error_when_unset(self, monkeypatch):
        """The module must stay importable without a key.

        It used to raise at import time, which meant the API could not start
        to report the problem and no test could import the module at all.
        """
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(MissingAPIKey, match="GOOGLE_API_KEY"):
            require_api_key()

    def test_returns_the_key_when_set(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
        assert require_api_key() == "abc123"

    def test_error_message_tells_you_where_to_get_one(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(MissingAPIKey, match="aistudio.google.com"):
            require_api_key()


class TestValidate:
    def test_accepts_supported_types(self):
        validate([Upload("a.pdf", b"x"), Upload("b.txt", b"y"), Upload("c.csv", b"z")])

    def test_rejects_nothing(self):
        with pytest.raises(IngestError, match="No files"):
            validate([])

    def test_rejects_unsupported_type(self):
        with pytest.raises(IngestError, match="not a .pdf, .txt or .csv"):
            validate([Upload("slides.pptx", b"x")])

    def test_rejects_empty_file(self):
        with pytest.raises(IngestError, match="empty"):
            validate([Upload("a.txt", b"")])

    def test_rejects_too_many_files(self, monkeypatch):
        monkeypatch.setattr("services.manager.MAX_FILES", 2)
        with pytest.raises(IngestError, match="Too many files"):
            validate([Upload(f"{i}.txt", b"x") for i in range(3)])

    def test_size_limit_is_across_all_files(self, monkeypatch):
        monkeypatch.setattr("services.manager.MAX_UPLOAD_BYTES", 250)
        with pytest.raises(IngestError, match="too large"):
            validate([Upload(f"{i}.txt", b"x" * 100) for i in range(3)])


class TestManagerGuards:
    def test_query_before_loading_raises(self):
        manager = Manager()
        with pytest.raises(NotReady, match="No documents are loaded"):
            manager.query("anything", "traversal")

    def test_configure_rejects_unknown_keys(self):
        # A typo would otherwise be dropped silently by the dataclass and
        # leave the caller believing it took effect.
        manager = Manager()
        with pytest.raises(NotReady, match="Unknown configuration keys"):
            manager.configure(chunk_sze=500)

    def test_status_is_safe_before_anything_is_built(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        status = Manager().status()
        assert status["configured"] is False
        assert status["ready"] is False
        assert status["google_key"] is False
        assert status["config"] == {}
