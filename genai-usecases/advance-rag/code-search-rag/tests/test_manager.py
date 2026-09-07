"""Upload classification, validation, and guards. No API key, no network."""

import pytest

from services.manager import (
    IngestError,
    Manager,
    MissingAPIKey,
    NotReady,
    Upload,
    require_api_key,
    validate,
)


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "name, language",
        [("a.py", "python"), ("b.JS", "javascript"), ("c.tsx", "typescript"),
         ("d.java", "java"), ("e.go", "go"), ("f.rs", "rust"),
         ("g.cpp", "cpp"), ("h.c", "c")],
    )
    def test_recognises_supported_extensions(self, name, language):
        assert Upload(name, b"x").language == language

    @pytest.mark.parametrize("name", ["notes.txt", "README.md", "data.csv", "noext"])
    def test_unsupported_extensions_have_no_language(self, name):
        # These are skipped rather than indexed as unstructured text, which
        # would pollute retrieval with things that are not code.
        assert Upload(name, b"x").language is None


class TestValidate:
    def test_accepts_supported_code(self):
        validate([Upload("a.py", b"def f(): pass")])

    def test_rejects_nothing(self):
        with pytest.raises(IngestError, match="No files"):
            validate([])

    def test_rejects_a_batch_with_no_supported_language(self):
        with pytest.raises(IngestError, match="supported language"):
            validate([Upload("notes.txt", b"x"), Upload("data.csv", b"y")])

    def test_accepts_a_mixed_batch_if_any_file_is_code(self):
        validate([Upload("notes.txt", b"x"), Upload("a.py", b"def f(): pass")])

    def test_rejects_too_many_files(self, monkeypatch):
        monkeypatch.setattr("services.manager.MAX_FILES", 2)
        with pytest.raises(IngestError, match="Too many files"):
            validate([Upload(f"{i}.py", b"x") for i in range(3)])

    def test_size_limit_is_across_all_files(self, monkeypatch):
        monkeypatch.setattr("services.manager.MAX_UPLOAD_BYTES", 250)
        with pytest.raises(IngestError, match="too large"):
            validate([Upload(f"{i}.py", b"x" * 100) for i in range(3)])


class TestKeyHandling:
    def test_missing_key_raises_a_named_error(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(MissingAPIKey, match="GOOGLE_API_KEY"):
            require_api_key()

    def test_message_points_at_where_to_get_one(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(MissingAPIKey, match="aistudio.google.com"):
            require_api_key()


class TestManagerGuards:
    def test_search_before_indexing_raises(self):
        with pytest.raises(NotReady, match="Nothing has been indexed"):
            Manager().search("anything")

    def test_status_is_safe_before_anything_is_built(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        status = Manager().status()
        assert status["configured"] is False
        assert status["ready"] is False
        assert status["chunks"] == 0
        assert "python" in status["languages"]


class TestBundledSamples:
    def test_sample_files_exist_and_are_python(self):
        # The UI offers a "Use samples" button; without these it is a dead end.
        from services.manager import SAMPLES_DIR
        samples = sorted(SAMPLES_DIR.glob("*.py"))
        assert samples, "no sample code bundled"
        assert all(s.read_bytes() for s in samples)

    def test_sample_corpus_is_not_named_like_a_test(self):
        """The OAuth2 corpus used to be called `test_oauth2_examples.py`.

        It contains no tests and no assertions, so pytest collected a sample
        file as a test module. Renaming it into `samples/` fixed that.
        """
        from services.manager import SAMPLES_DIR
        assert not list(SAMPLES_DIR.glob("test_*.py"))
