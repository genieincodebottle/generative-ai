"""Upload validation and guards. No API key, no network."""

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


class TestUploadClassification:
    def test_pdf_is_a_document(self):
        assert Upload("report.PDF", b"x").is_document is True
        assert Upload("report.PDF", b"x").is_image is False

    @pytest.mark.parametrize("name", ["a.png", "b.JPG", "c.jpeg", "d.webp", "e.gif", "f.bmp"])
    def test_image_suffixes_are_recognised(self, name):
        assert Upload(name, b"x").is_image is True

    def test_unknown_suffix_is_neither(self):
        upload = Upload("notes.txt", b"x")
        assert not upload.is_document and not upload.is_image


class TestValidate:
    def test_accepts_pdfs_and_images_together(self):
        validate([Upload("a.pdf", b"x"), Upload("b.png", b"y")])

    def test_rejects_nothing(self):
        with pytest.raises(IngestError, match="No files"):
            validate([])

    def test_rejects_unsupported_type(self):
        with pytest.raises(IngestError, match="not a PDF or an image"):
            validate([Upload("notes.txt", b"x")])

    def test_rejects_empty_file(self):
        with pytest.raises(IngestError, match="empty"):
            validate([Upload("a.pdf", b"")])

    def test_rejects_too_many_files(self, monkeypatch):
        monkeypatch.setattr("services.manager.MAX_FILES", 2)
        with pytest.raises(IngestError, match="Too many files"):
            validate([Upload(f"{i}.pdf", b"x") for i in range(3)])

    def test_size_limit_is_across_all_files(self, monkeypatch):
        monkeypatch.setattr("services.manager.MAX_UPLOAD_BYTES", 250)
        with pytest.raises(IngestError, match="too large"):
            validate([Upload(f"{i}.pdf", b"x" * 100) for i in range(3)])


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
    def test_query_before_indexing_raises(self):
        with pytest.raises(NotReady, match="Nothing has been indexed"):
            Manager().query("anything")

    def test_status_is_safe_before_anything_is_built(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        status = Manager().status()
        assert status == {"configured": False, "ready": False, "google_key": False,
                          "documents": [], "images": []}
