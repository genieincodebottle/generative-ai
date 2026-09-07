"""Owns the GraphRAG system and the uploaded documents.

Building the system downloads a sentence-transformers model and builds the
graph, so it is built once and reused. Uploads arrive as ``(filename, bytes)``
pairs rather than Streamlit objects.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from services.graph_rag import (
    GraphRAGConfig,
    GraphRAGSystem,
    MissingAPIKey,
    RetrieverType,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))
MAX_FILES = int(os.getenv("MAX_FILES", "10"))
ALLOWED_SUFFIXES = (".pdf", ".txt", ".csv")

LLM_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-pro-latest",
]

RETRIEVERS = [
    {
        "id": "traversal",
        "label": "Graph traversal",
        "blurb": "Start from the best vector matches, then walk the edges "
                 "between documents to pull in what they connect to.",
    },
    {
        "id": "standard",
        "label": "Standard vector",
        "blurb": "Plain similarity search. No edges. The baseline.",
    },
    {
        "id": "hybrid",
        "label": "Agentic routing",
        "blurb": "A router reads the question and picks traversal or standard, "
                 "and tells you why.",
    },
]


class IngestError(RuntimeError):
    """The uploaded files could not be indexed."""


class NotReady(RuntimeError):
    """No documents have been loaded yet."""


@dataclass
class Upload:
    filename: str
    content: bytes


@dataclass
class LoadResult:
    source: str
    filenames: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)


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
        if not upload.filename.lower().endswith(ALLOWED_SUFFIXES):
            raise IngestError(
                f"{upload.filename} is not a .pdf, .txt or .csv file."
            )
        if not upload.content:
            raise IngestError(f"{upload.filename} is empty.")


class Manager:
    def __init__(self) -> None:
        self._system: GraphRAGSystem | None = None
        self._lock = threading.Lock()
        self._loaded = LoadResult(source="none")

    def configure(self, **overrides) -> GraphRAGSystem:
        valid = set(GraphRAGConfig.__dataclass_fields__)
        unknown = set(overrides) - valid
        if unknown:
            raise NotReady(f"Unknown configuration keys: {sorted(unknown)}")
        with self._lock:
            self._system = GraphRAGSystem(config=GraphRAGConfig(**overrides))
            self._loaded = LoadResult(source="none")
            return self._system

    def ensure(self) -> GraphRAGSystem:
        if self._system is None:
            return self.configure()
        return self._system

    @property
    def is_ready(self) -> bool:
        return self._system is not None and self._system.vector_store is not None

    def load_sample(self) -> LoadResult:
        """Load the bundled animals dataset, so the app works with no files."""
        system = self.ensure()
        if not system.initialize_with_default_data():
            raise IngestError("The sample dataset could not be loaded.")
        self._loaded = LoadResult(
            source="sample animals dataset",
            relationships=system.get_detected_relationships(),
        )
        return self._loaded

    def load_uploads(self, uploads: list[Upload]) -> LoadResult:
        validate(uploads)
        system = self.ensure()

        paths: list[str] = []
        try:
            for upload in uploads:
                suffix = Path(upload.filename).suffix or ".txt"
                handle, path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(handle, "wb") as tmp:
                    tmp.write(upload.content)
                paths.append(path)
            if not system.initialize_with_files(paths):
                raise IngestError(
                    "The documents could not be indexed. They may be scanned "
                    "images rather than text PDFs; this project does not OCR."
                )
        finally:
            for path in paths:
                try:
                    Path(path).unlink()
                except OSError:
                    pass

        self._loaded = LoadResult(
            source="uploaded documents",
            filenames=[u.filename for u in uploads],
            relationships=system.get_detected_relationships(),
        )
        return self._loaded

    def query(self, question: str, retriever: str) -> dict:
        if not self.is_ready:
            raise NotReady(
                "No documents are loaded. Upload files or load the sample "
                "dataset first."
            )
        try:
            kind = RetrieverType(retriever)
        except ValueError as exc:
            raise NotReady(f"Unknown retriever: {retriever}") from exc

        result = self._system.query(question, kind, return_details=True)
        if isinstance(result, str):
            result = {"answer": result}
        return result

    def routing_explanation(self, question: str) -> dict:
        if not self.is_ready:
            raise NotReady("No documents are loaded.")
        return self._system.get_routing_explanation(question)

    def status(self) -> dict:
        configured = self._system is not None
        return {
            "configured": configured,
            "ready": self.is_ready,
            "google_key": bool(os.getenv("GOOGLE_API_KEY")),
            "source": self._loaded.source,
            "filenames": self._loaded.filenames,
            "relationships": self._loaded.relationships,
            "config": (
                {k: str(v) for k, v in self._system.config.__dict__.items()}
                if configured else {}
            ),
        }


manager = Manager()

__all__ = [
    "IngestError", "LoadResult", "MissingAPIKey", "Manager", "NotReady",
    "Upload", "LLM_MODELS", "RETRIEVERS", "manager", "validate",
]
