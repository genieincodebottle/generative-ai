"""Upload validation and configuration guards. No API key, no network."""

import pytest

from services.system_manager import IngestError, SystemManager, SystemNotReady, Upload


@pytest.fixture
def manager():
    return SystemManager()


class TestValidateUploads:
    def test_accepts_a_pdf(self, manager):
        manager.validate([Upload("a.pdf", b"%PDF-1.4 content")])

    def test_rejects_no_files(self, manager):
        with pytest.raises(IngestError, match="No files"):
            manager.validate([])

    def test_rejects_non_pdf(self, manager):
        with pytest.raises(IngestError, match="not a PDF"):
            manager.validate([Upload("notes.txt", b"hello")])

    def test_rejects_empty_file(self, manager):
        with pytest.raises(IngestError, match="empty"):
            manager.validate([Upload("a.pdf", b"")])

    def test_rejects_too_many_files(self, manager, monkeypatch):
        monkeypatch.setattr("services.system_manager.MAX_FILES", 2)
        with pytest.raises(IngestError, match="Too many files"):
            manager.validate([Upload(f"{i}.pdf", b"x") for i in range(3)])

    def test_size_limit_is_across_all_files(self, manager, monkeypatch):
        # Files that each fit can still exceed the cap together.
        monkeypatch.setattr("services.system_manager.MAX_UPLOAD_BYTES", 250)
        with pytest.raises(IngestError, match="too large"):
            manager.validate([Upload(f"{i}.pdf", b"x" * 100) for i in range(3)])


class TestConfigure:
    def test_requires_a_google_key(self, manager, monkeypatch):
        monkeypatch.setattr("services.system_manager.google_api_key", lambda: None)
        with pytest.raises(SystemNotReady, match="GOOGLE_API_KEY"):
            manager.configure()

    def test_rejects_unknown_config_keys(self, manager, monkeypatch):
        # A typo in a config key would otherwise be silently ignored by the
        # dataclass and leave the caller believing it took effect.
        monkeypatch.setattr("services.system_manager.google_api_key", lambda: "k")
        with pytest.raises(SystemNotReady, match="Unknown configuration keys"):
            manager.configure(temprature=0.5)


class TestStatus:
    def test_reports_unconfigured_before_first_use(self, manager, monkeypatch):
        monkeypatch.setattr("services.system_manager.google_api_key", lambda: None)
        monkeypatch.setattr("services.system_manager.tavily_api_key", lambda: None)
        status = manager.status()
        assert status["configured"] is False
        assert status["google_key"] is False
        assert status["documents"] == []

    def test_reports_key_presence_without_building_the_system(self, manager, monkeypatch):
        monkeypatch.setattr("services.system_manager.google_api_key", lambda: "k")
        monkeypatch.setattr("services.system_manager.tavily_api_key", lambda: "t")
        status = manager.status()
        assert status["google_key"] is True
        assert status["tavily_key"] is True
        assert status["configured"] is False


class TestQueryGuards:
    def test_query_without_documents_raises(self, manager, monkeypatch):
        class StubSystem:
            retriever = None

        monkeypatch.setattr(manager, "ensure", lambda: StubSystem())
        with pytest.raises(SystemNotReady, match="No documents"):
            manager.query("anything")
