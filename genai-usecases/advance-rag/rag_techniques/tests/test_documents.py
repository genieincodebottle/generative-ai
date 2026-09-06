"""Ingestion: validation, parsing, chunking. No API key, no network."""

import pytest
from langchain_core.documents import Document as LCDocument

from services.documents import IngestError, Upload, chunk, load, sample_uploads, validate


class TestValidate:
    def test_accepts_pdf_and_txt(self):
        validate([Upload("a.pdf", b"%PDF-1.4"), Upload("b.txt", b"hello")])

    def test_rejects_nothing(self):
        with pytest.raises(IngestError, match="No files"):
            validate([])

    def test_rejects_other_extensions(self):
        with pytest.raises(IngestError, match="not a .pdf or .txt"):
            validate([Upload("notes.docx", b"x")])

    def test_rejects_empty_file(self):
        with pytest.raises(IngestError, match="empty"):
            validate([Upload("a.txt", b"")])

    def test_rejects_too_many_files(self, monkeypatch):
        monkeypatch.setattr("services.documents.MAX_FILES", 2)
        with pytest.raises(IngestError, match="Too many files"):
            validate([Upload(f"{i}.txt", b"x") for i in range(3)])

    def test_size_limit_is_across_all_files(self, monkeypatch):
        # Each file fits; together they do not.
        monkeypatch.setattr("services.documents.MAX_UPLOAD_BYTES", 250)
        with pytest.raises(IngestError, match="too large"):
            validate([Upload(f"{i}.txt", b"x" * 100) for i in range(3)])


class TestLoad:
    def test_reads_a_text_file(self):
        pages = load([Upload("notes.txt", b"hello from a text file")])
        assert "hello" in pages[0].page_content

    def test_source_metadata_uses_the_uploaded_name(self):
        # The loader records a temp path the reader has never seen.
        pages = load([Upload("quarterly.txt", b"content")])
        assert pages[0].metadata["source"] == "quarterly.txt"

    def test_unreadable_pdf_raises_ingest_error(self):
        with pytest.raises(IngestError, match="Could not read"):
            load([Upload("broken.pdf", b"not a pdf at all")])


class TestChunk:
    def make(self, text="word " * 500):
        return [LCDocument(page_content=text, metadata={"source": "a.txt"})]

    def test_splits_and_numbers_chunks(self):
        chunks = chunk(self.make(), 200, 20)
        assert len(chunks) > 1
        assert [c.metadata["id"] for c in chunks] == list(range(len(chunks)))

    def test_rejects_overlap_not_smaller_than_size(self):
        with pytest.raises(IngestError, match="chunk_overlap must be smaller"):
            chunk(self.make(), 200, 200)

    def test_rejects_documents_with_no_text(self):
        # A scanned PDF parses fine and yields nothing. An empty index would
        # answer "I don't know" to everything, which hides the real problem.
        empty = [LCDocument(page_content="   ", metadata={})]
        with pytest.raises(IngestError, match="No extractable text"):
            chunk(empty, 500, 50)

    def test_rejects_no_documents(self):
        with pytest.raises(IngestError, match="No extractable text"):
            chunk([], 500, 50)


class TestSampleUploads:
    def test_bundled_samples_are_present_and_readable(self):
        # The UI offers a "Use samples" button, so these must exist in a
        # fresh clone or that button is a dead end.
        uploads = sample_uploads()
        assert len(uploads) >= 2
        assert all(u.content for u in uploads)
        assert all(u.filename.endswith(".txt") for u in uploads)
