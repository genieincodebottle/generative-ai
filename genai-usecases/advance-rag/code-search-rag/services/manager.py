"""Owns the code-search system and validates what gets indexed."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from services.rag import Config, RAGCodeSearchSystem

load_dotenv()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_FILES = int(os.getenv("MAX_FILES", "200"))

# Extensions the parser understands. Anything else is skipped rather than
# indexed as unstructured text, which would pollute retrieval.
CODE_SUFFIXES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".cpp": "cpp", ".cc": "cpp",
    ".c": "c", ".h": "c", ".hpp": "cpp",
}

LLM_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-pro-latest",
]

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


class MissingAPIKey(RuntimeError):
    """GOOGLE_API_KEY is not configured."""


class IngestError(RuntimeError):
    """The code could not be indexed."""


class NotReady(RuntimeError):
    """Nothing has been indexed yet."""


@dataclass
class Upload:
    filename: str
    content: bytes

    @property
    def language(self) -> str | None:
        return CODE_SUFFIXES.get(Path(self.filename).suffix.lower())


@dataclass
class IndexResult:
    chunks: int = 0
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


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
    if not any(u.language for u in uploads):
        raise IngestError(
            "None of these files are in a supported language "
            f"({', '.join(sorted(CODE_SUFFIXES))})."
        )


class Manager:
    def __init__(self) -> None:
        self._system: RAGCodeSearchSystem | None = None
        self._lock = threading.Lock()
        self._indexed = IndexResult()

    def configure(self, **overrides) -> RAGCodeSearchSystem:
        require_api_key()
        valid = set(Config.__dataclass_fields__)
        unknown = set(overrides) - valid
        if unknown:
            raise IngestError(f"Unknown configuration keys: {sorted(unknown)}")
        with self._lock:
            self._system = RAGCodeSearchSystem(config=Config(**overrides))
            self._indexed = IndexResult()
            return self._system

    def ensure(self) -> RAGCodeSearchSystem:
        if self._system is None:
            return self.configure()
        return self._system

    @property
    def is_ready(self) -> bool:
        return self._indexed.chunks > 0

    def index_uploads(self, uploads: list[Upload]) -> IndexResult:
        """Index source files, skipping anything the parser cannot read."""
        validate(uploads)
        system = self.ensure()

        indexed, skipped, chunks = [], [], 0
        for upload in uploads:
            if not upload.language:
                skipped.append(upload.filename)
                continue
            try:
                code = upload.content.decode("utf-8")
            except UnicodeDecodeError:
                # A binary file with a code extension. Skip it loudly rather
                # than indexing mojibake.
                skipped.append(upload.filename)
                continue
            chunks += system.index_code(
                code, upload.language, repo_name="upload",
                file_name=upload.filename,
            )
            indexed.append(upload.filename)

        if not chunks:
            raise IngestError(
                "Nothing could be indexed. The files may be empty, or contain "
                "no functions or classes for the parser to find."
            )

        self._indexed = IndexResult(
            chunks=self._indexed.chunks + chunks,
            files=self._indexed.files + indexed,
            skipped=skipped,
        )
        return self._indexed

    def index_samples(self) -> IndexResult:
        """Index the bundled sample code, so the demo works with no files."""
        uploads = [
            Upload(filename=path.name, content=path.read_bytes())
            for path in sorted(SAMPLES_DIR.glob("*.py"))
        ]
        if not uploads:
            raise IngestError("No sample code is bundled with this project.")
        return self.index_uploads(uploads)

    def index_repository(self, uploads: list[Upload], repo_name: str) -> IndexResult:
        """Write uploads into a temp tree and index it as one repository."""
        validate(uploads)
        system = self.ensure()

        root = tempfile.mkdtemp(prefix="code-search-")
        try:
            for upload in uploads:
                target = Path(root) / Path(upload.filename).name
                target.write_bytes(upload.content)
            chunks = system.index_repository(root, repo_name)
        finally:
            shutil.rmtree(root, ignore_errors=True)

        if not chunks:
            raise IngestError("The repository produced no indexable chunks.")

        self._indexed = IndexResult(
            chunks=self._indexed.chunks + chunks,
            files=self._indexed.files + [u.filename for u in uploads],
        )
        return self._indexed

    def search(self, query: str, filters: dict | None = None) -> str:
        if not self.is_ready:
            raise NotReady(
                "Nothing has been indexed yet. Upload code or index the "
                "bundled samples first."
            )
        return self._system.search(query, filters or None)

    def status(self) -> dict:
        return {
            "configured": self._system is not None,
            "ready": self.is_ready,
            "google_key": bool(api_key()),
            "chunks": self._indexed.chunks,
            "files": self._indexed.files,
            "skipped": self._indexed.skipped,
            "languages": sorted(set(CODE_SUFFIXES.values())),
        }


manager = Manager()
