"""Upload validation, parsing, and chunking - shared by every technique.

Documents arrive as ``(filename, bytes)`` pairs, not as Streamlit upload
objects, so this is callable from a test or a notebook.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.config import MAX_FILES, MAX_UPLOAD_BYTES, SAMPLE_DOCS

ALLOWED_SUFFIXES = (".pdf", ".txt")


class IngestError(RuntimeError):
    """The uploaded files could not be turned into chunks."""


@dataclass
class Upload:
    filename: str
    content: bytes


def validate(uploads: list[Upload]) -> None:
    """Reject the whole batch before any of it is parsed."""
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
                f"{upload.filename} is not a .pdf or .txt file."
            )
        if not upload.content:
            raise IngestError(f"{upload.filename} is empty.")


def load(uploads: list[Upload]) -> list[LCDocument]:
    """Parse uploads into LangChain documents via short-lived temp files."""
    loaded: list[LCDocument] = []
    for upload in uploads:
        is_txt = upload.filename.lower().endswith(".txt")
        handle, path = tempfile.mkstemp(suffix=".txt" if is_txt else ".pdf")
        try:
            with os.fdopen(handle, "wb") as tmp:
                tmp.write(upload.content)
            loader = TextLoader(path, encoding="utf-8") if is_txt else PyPDFLoader(path)
            pages = loader.load()
            for page in pages:
                # The loader records the temp path, which means nothing to the
                # reader. Replace it with the name they uploaded.
                page.metadata["source"] = upload.filename
            loaded.extend(pages)
        except Exception as exc:
            raise IngestError(f"Could not read {upload.filename}: {exc}") from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    return loaded


def chunk(documents: list[LCDocument], chunk_size: int,
          chunk_overlap: int) -> list[LCDocument]:
    """Split documents, and refuse to return an index of nothing."""
    if chunk_overlap >= chunk_size:
        raise IngestError("chunk_overlap must be smaller than chunk_size.")

    # A scanned PDF parses fine and yields no text. An empty index answers
    # "I don't know" to everything, which is a much worse failure than saying
    # so here.
    if not documents or not any(d.page_content.strip() for d in documents):
        raise IngestError(
            "No extractable text found. These may be scanned images rather "
            "than text PDFs; this project does not run OCR."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(documents)

    if not chunks:
        raise IngestError("The documents produced no chunks to index.")

    for index, piece in enumerate(chunks):
        piece.metadata["id"] = index
    return chunks


def prepare(uploads: list[Upload], chunk_size: int,
            chunk_overlap: int) -> list[LCDocument]:
    validate(uploads)
    return chunk(load(uploads), chunk_size, chunk_overlap)


def sample_uploads() -> list[Upload]:
    """The two bundled sample documents, so the app is usable with no files."""
    uploads = []
    for path in sorted(SAMPLE_DOCS.glob("*.txt")):
        uploads.append(Upload(filename=path.name, content=path.read_bytes()))
    if not uploads:
        raise IngestError("No sample documents are bundled with this project.")
    return uploads


def chunk_stats(chunks: list[LCDocument], chunk_size: int,
                chunk_overlap: int) -> dict:
    lengths = [len(c.page_content) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chunk_length": int(sum(lengths) / len(lengths)) if lengths else 0,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
