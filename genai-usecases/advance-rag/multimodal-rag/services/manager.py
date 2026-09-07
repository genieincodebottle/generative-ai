"""Owns the multimodal system and validates what gets fed into it."""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from services.multimodal_rag import MultimodalRAGSystem

load_dotenv()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_FILES = int(os.getenv("MAX_FILES", "15"))

DOCUMENT_SUFFIXES = (".pdf",)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

# Rolling aliases first: they track Google's current generation, so they do not
# 404 the way a retired pinned ID does.
MAIN_MODELS = [
    "gemini-pro-latest",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
]
VISION_MODELS = [
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-pro-latest",
]


class MissingAPIKey(RuntimeError):
    """GOOGLE_API_KEY is not configured."""


class IngestError(RuntimeError):
    """The uploads could not be indexed."""


class NotReady(RuntimeError):
    """Nothing has been indexed yet."""


@dataclass
class Upload:
    filename: str
    content: bytes

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()

    @property
    def is_image(self) -> bool:
        return self.suffix in IMAGE_SUFFIXES

    @property
    def is_document(self) -> bool:
        return self.suffix in DOCUMENT_SUFFIXES


@dataclass
class IngestResult:
    documents: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


def api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY")


def require_api_key() -> str:
    key = api_key()
    if not key:
        raise MissingAPIKey(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env, add your "
            "key from https://aistudio.google.com/app/apikey, then restart."
        )
    return key


def validate(uploads: list[Upload]) -> None:
    if not uploads:
        raise IngestError("No files were provided.")
    if len(uploads) > MAX_FILES:
        raise IngestError(f"Too many files (maximum {MAX_FILES}).")

    total = sum(len(u.content) for u in uploads)
    if total > MAX_UPLOAD_BYTES:
        raise IngestError(
            f"Upload is too large: {total / 1e6:.1f} MB, "
            f"maximum {MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )
    for upload in uploads:
        if not (upload.is_document or upload.is_image):
            raise IngestError(
                f"{upload.filename} is not a PDF or an image "
                f"({', '.join(IMAGE_SUFFIXES)})."
            )
        if not upload.content:
            raise IngestError(f"{upload.filename} is empty.")


class Manager:
    def __init__(self) -> None:
        self._system: MultimodalRAGSystem | None = None
        self._lock = threading.Lock()
        self._ingested = IngestResult()
        self._built = False

    def configure(self, **overrides) -> MultimodalRAGSystem:
        key = require_api_key()
        with self._lock:
            self._system = MultimodalRAGSystem(google_api_key=key, **overrides)
            self._ingested = IngestResult()
            self._built = False
            return self._system

    def ensure(self) -> MultimodalRAGSystem:
        if self._system is None:
            return self.configure()
        return self._system

    @property
    def is_ready(self) -> bool:
        return self._built

    def ingest(self, uploads: list[Upload]) -> IngestResult:
        """Write uploads to temp files, index them, then clean up.

        PDFs and images take different paths through the system, so they are
        separated here rather than inside the indexing call.
        """
        validate(uploads)
        system = self.ensure()

        doc_paths: list[str] = []
        image_paths: list[str] = []
        written: list[str] = []
        try:
            for upload in uploads:
                handle, path = tempfile.mkstemp(suffix=upload.suffix)
                with os.fdopen(handle, "wb") as tmp:
                    tmp.write(upload.content)
                written.append(path)
                (image_paths if upload.is_image else doc_paths).append(path)

            system.build_enhanced_vector_database(doc_paths, image_paths)
            self._built = True
        except Exception as exc:
            raise IngestError(f"Indexing failed: {exc}") from exc
        finally:
            for path in written:
                try:
                    Path(path).unlink()
                except OSError:
                    pass

        self._ingested = IngestResult(
            documents=[u.filename for u in uploads if u.is_document],
            images=[u.filename for u in uploads if u.is_image],
        )
        return self._ingested

    def query(self, question: str, k: int = 6) -> dict:
        if not self.is_ready:
            raise NotReady(
                "Nothing has been indexed yet. Upload a PDF or an image first."
            )
        return self._system.query(question, k=k)

    def status(self) -> dict:
        return {
            "configured": self._system is not None,
            "ready": self.is_ready,
            "google_key": bool(api_key()),
            "documents": self._ingested.documents,
            "images": self._ingested.images,
        }


manager = Manager()
