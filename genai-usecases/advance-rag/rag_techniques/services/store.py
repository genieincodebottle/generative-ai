"""Session store: one built index per session, LRU bounded.

Building an index costs an embedding call per chunk, so it is built once when
documents are uploaded and then reused by every technique. That is also what
makes the comparison fair: all five techniques query the *same* index.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_chroma import Chroma

from services import documents
from services.config import MAX_SESSIONS
from services.documents import Upload
from services.llm import get_embeddings
from services.techniques import Index


class SessionNotFound(KeyError):
    """Unknown session id, or the session has been evicted."""


@dataclass
class Session:
    id: str
    provider: str
    index: Index
    filenames: list[str]
    stats: dict
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(self, uploads: list[Upload], provider: str,
               chunk_size: int, chunk_overlap: int) -> Session:
        chunks = documents.prepare(uploads, chunk_size, chunk_overlap)
        embeddings = get_embeddings(provider)

        # An in-memory Chroma collection per session, so two sessions cannot
        # read each other's documents.
        vectorstore = Chroma.from_documents(
            chunks, embeddings, collection_name=f"rag_{uuid.uuid4().hex[:12]}"
        )

        session = Session(
            id=uuid.uuid4().hex,
            provider=provider,
            index=Index(chunks=chunks, vectorstore=vectorstore, embeddings=embeddings),
            filenames=[u.filename for u in uploads],
            stats=documents.chunk_stats(chunks, chunk_size, chunk_overlap),
        )
        with self._lock:
            self._sessions[session.id] = session
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFound(session_id)
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise SessionNotFound(session_id)

    def stats(self) -> dict:
        with self._lock:
            return {"sessions": len(self._sessions), "max_sessions": self._max}


store = SessionStore()
