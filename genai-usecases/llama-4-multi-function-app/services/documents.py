"""Document ingestion and retrieval for the RAG tab.

The index lives here rather than in ``st.session_state``, which is what tied
the original to a running Streamlit session.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from services.config import EMBEDDING_MODEL, MAX_UPLOAD_BYTES

LOADERS = {".pdf": "PyPDFLoader", ".txt": "TextLoader", ".csv": "CSVLoader"}


class IngestError(RuntimeError):
    """The upload could not be indexed."""


class NotIndexed(RuntimeError):
    """Nothing has been indexed yet."""


@dataclass
class Upload:
    filename: str
    content: bytes

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()


@dataclass
class IndexedDocument:
    name: str
    chunks: int
    uploaded_at: str


@dataclass
class Store:
    vector_store: object | None = None
    documents: list[IndexedDocument] = field(default_factory=list)


_store = Store()
_lock = threading.Lock()
_embeddings = None


def get_embeddings():
    """Load the local embedding model once. Downloads on first use."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise IngestError(
            "langchain-huggingface is required for retrieval. "
            "Run: pip install -r requirements.txt"
        ) from exc
    _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def validate(upload: Upload) -> None:
    if upload.suffix not in LOADERS:
        raise IngestError(
            f"{upload.filename} is not a supported type "
            f"({', '.join(sorted(LOADERS))})."
        )
    if not upload.content:
        raise IngestError(f"{upload.filename} is empty.")
    if len(upload.content) > MAX_UPLOAD_BYTES:
        raise IngestError(
            f"{upload.filename} is too large "
            f"({len(upload.content) / 1e6:.1f} MB, "
            f"maximum {MAX_UPLOAD_BYTES / 1e6:.0f} MB)."
        )


def add_document(upload: Upload, chunk_size: int = 1000,
                 chunk_overlap: int = 200) -> IndexedDocument:
    """Parse, chunk and index one upload. Cleans up its temp file either way."""
    validate(upload)
    if chunk_overlap >= chunk_size:
        raise IngestError("chunk_overlap must be smaller than chunk_size.")

    from langchain_community.document_loaders import (
        CSVLoader,
        PyPDFLoader,
        TextLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    handle, path = tempfile.mkstemp(suffix=upload.suffix)
    try:
        with os.fdopen(handle, "wb") as tmp:
            tmp.write(upload.content)

        loader = {
            ".pdf": PyPDFLoader,
            ".txt": lambda p: TextLoader(p, encoding="utf-8"),
            ".csv": CSVLoader,
        }[upload.suffix](path)

        try:
            documents = loader.load()
        except Exception as exc:
            raise IngestError(f"Could not read {upload.filename}: {exc}") from exc

        # A scanned PDF parses fine and yields nothing. An empty index answers
        # "I don't know" to everything, which hides the real problem.
        if not documents or not any(d.page_content.strip() for d in documents):
            raise IngestError(
                f"No extractable text found in {upload.filename}. It may be a "
                f"scanned image rather than a text PDF; this app does not OCR."
            )

        for document in documents:
            document.metadata["source"] = upload.filename

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        ).split_documents(documents)
        if not chunks:
            raise IngestError(f"{upload.filename} produced no chunks to index.")

        from langchain_community.vectorstores import FAISS

        embeddings = get_embeddings()
        with _lock:
            if _store.vector_store is None:
                _store.vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                _store.vector_store.add_documents(chunks)

            record = IndexedDocument(
                name=upload.filename, chunks=len(chunks),
                uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            _store.documents.append(record)
        return record
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


def search(query: str, top_k: int = 3) -> list[dict]:
    if _store.vector_store is None:
        raise NotIndexed("No documents have been indexed yet.")
    hits = _store.vector_store.similarity_search(query, k=top_k)
    return [
        {"content": hit.page_content,
         "metadata": {k: str(v) for k, v in (hit.metadata or {}).items()}}
        for hit in hits
    ]


def status() -> dict:
    return {
        "indexed": _store.vector_store is not None,
        "documents": [d.__dict__ for d in _store.documents],
        "total_chunks": sum(d.chunks for d in _store.documents),
    }


def reset() -> None:
    with _lock:
        _store.vector_store = None
        _store.documents = []
