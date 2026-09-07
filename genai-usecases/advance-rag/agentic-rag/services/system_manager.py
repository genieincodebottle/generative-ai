"""Owns the single AgenticRAGSystem instance and the uploaded documents.

Building the system opens a persistent Chroma collection and constructs the
LangGraph workflow, so it is built once and reused rather than per request.
Changing the configuration rebuilds it deliberately.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from services.agentic_rag_system import AgenticRAGConfig, AgenticRAGSystem
from services.config import (
    MAX_FILES,
    MAX_UPLOAD_BYTES,
    google_api_key,
    tavily_api_key,
)


class SystemNotReady(RuntimeError):
    """The system has not been initialized, or its API key is missing."""


class IngestError(RuntimeError):
    """The uploaded files could not be indexed."""


@dataclass
class Upload:
    filename: str
    content: bytes


@dataclass
class IngestResult:
    filenames: list[str] = field(default_factory=list)
    loaded: int = 0


class SystemManager:
    """Thread-safe holder for one configured system."""

    def __init__(self) -> None:
        self._system: AgenticRAGSystem | None = None
        self._lock = threading.Lock()
        self._documents: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def configure(self, **overrides) -> AgenticRAGSystem:
        """Build or rebuild the system. Raises :class:`SystemNotReady`."""
        if not google_api_key():
            raise SystemNotReady(
                "GOOGLE_API_KEY is not set. Add it to your .env file and "
                "restart the API."
            )

        valid = {f for f in AgenticRAGConfig.__dataclass_fields__}
        unknown = set(overrides) - valid
        if unknown:
            raise SystemNotReady(f"Unknown configuration keys: {sorted(unknown)}")

        config = AgenticRAGConfig(**overrides)
        with self._lock:
            self._system = AgenticRAGSystem(
                config=config,
                google_api_key=google_api_key(),
                tavily_api_key=tavily_api_key(),
            )
            return self._system

    def ensure(self) -> AgenticRAGSystem:
        """Return the system, building it with defaults on first use."""
        if self._system is None:
            return self.configure()
        return self._system

    @property
    def is_configured(self) -> bool:
        return self._system is not None

    # -- documents ---------------------------------------------------------

    def validate(self, uploads: list[Upload]) -> None:
        if not uploads:
            raise IngestError("No files were uploaded.")
        if len(uploads) > MAX_FILES:
            raise IngestError(f"Too many files (maximum {MAX_FILES}).")
        total = sum(len(u.content) for u in uploads)
        if total > MAX_UPLOAD_BYTES:
            raise IngestError(
                f"Upload is too large: {total / 1e6:.1f} MB, "
                f"maximum {MAX_UPLOAD_BYTES / 1e6:.0f} MB."
            )
        for upload in uploads:
            if not upload.filename.lower().endswith(".pdf"):
                raise IngestError(f"{upload.filename} is not a PDF.")
            if not upload.content:
                raise IngestError(f"{upload.filename} is empty.")

    def load_documents(self, uploads: list[Upload]) -> IngestResult:
        """Write uploads to temp files and index them. Cleans up either way."""
        self.validate(uploads)
        system = self.ensure()

        paths: list[str] = []
        try:
            for upload in uploads:
                handle, path = tempfile.mkstemp(suffix=".pdf")
                with os.fdopen(handle, "wb") as tmp:
                    tmp.write(upload.content)
                paths.append(path)

            if not system.load_documents(paths):
                raise IngestError(
                    "The documents could not be indexed. They may be scanned "
                    "images rather than text PDFs; this app does not run OCR."
                )
        finally:
            for path in paths:
                try:
                    Path(path).unlink()
                except OSError:
                    pass

        names = [u.filename for u in uploads]
        self._documents.extend(names)
        return IngestResult(filenames=names, loaded=len(names))

    @property
    def documents(self) -> list[str]:
        return list(self._documents)

    # -- use ---------------------------------------------------------------

    def query(self, question: str, thread_id: str | None = None) -> dict:
        system = self.ensure()
        if system.retriever is None:
            raise SystemNotReady(
                "No documents have been indexed yet. Upload a PDF first."
            )
        return system.query(question, thread_id)

    def status(self) -> dict:
        if self._system is None:
            return {
                "configured": False,
                "google_key": bool(google_api_key()),
                "tavily_key": bool(tavily_api_key()),
                "documents": [],
            }
        status = self._system.get_system_status()
        status.update({
            "configured": True,
            "google_key": bool(google_api_key()),
            "tavily_key": bool(tavily_api_key()),
            "documents": self.documents,
        })
        return status


manager = SystemManager()
