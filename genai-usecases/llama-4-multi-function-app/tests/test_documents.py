"""Document ingestion guards. No API key, no network."""

import pytest

from services.documents import (
    IngestError,
    NotIndexed,
    Upload,
    add_document,
    search,
    validate,
)


class TestValidate:
    @pytest.mark.parametrize("name", ["a.pdf", "b.txt", "c.csv", "D.TXT"])
    def test_accepts_supported_types(self, name):
        validate(Upload(name, b"content"))

    def test_rejects_unsupported_type(self):
        with pytest.raises(IngestError, match="not a supported type"):
            validate(Upload("slides.pptx", b"x"))

    def test_rejects_empty(self):
        with pytest.raises(IngestError, match="empty"):
            validate(Upload("a.txt", b""))

    def test_rejects_oversized(self, monkeypatch):
        monkeypatch.setattr("services.documents.MAX_UPLOAD_BYTES", 10)
        with pytest.raises(IngestError, match="too large"):
            validate(Upload("a.txt", b"x" * 20))


class TestSearchGuard:
    def test_search_before_indexing_raises(self):
        with pytest.raises(NotIndexed, match="No documents"):
            search("anything")


class TestChunkGuards:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        # Validated before the embedding model is loaded, so this test does
        # not download 90 MB of weights to discover a config mistake.
        with pytest.raises(IngestError, match="chunk_overlap must be smaller"):
            add_document(Upload("a.txt", b"hello"),
                         chunk_size=100, chunk_overlap=100)

    def test_validation_runs_before_chunking(self):
        with pytest.raises(IngestError, match="not a supported type"):
            add_document(Upload("a.pptx", b"hello"),
                         chunk_size=100, chunk_overlap=100)
