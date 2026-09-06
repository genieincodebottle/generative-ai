"""Server-side chat sessions: the index, the chain, and the conversation.

This is where the memory actually lives. The original app rebuilt its history
dict every time the chain was rebuilt, so changing the temperature slider threw
away the conversation without saying so. Here the history belongs to the
session and outlives any chain built from it.

In-process and single-node on purpose: it is a demo. Swapping this for Redis is
a change to one file.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from services.config import MAX_SESSIONS
from services.llm_text import message_text
from services.rag_service import Document, IngestResult, build_chain, build_index, get_llm


class SessionNotFound(KeyError):
    """The session id is unknown, or its session has been evicted."""


@dataclass
class Session:
    id: str
    provider: str
    model: str
    temperature: float
    retriever_k: int
    index: object
    ingest: IngestResult
    history: BaseChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def messages(self) -> list[dict]:
        """The conversation so far, in a shape the UI can render directly."""
        out = []
        for message in self.history.messages:
            role = {"human": "user", "ai": "assistant"}.get(message.type, message.type)
            out.append({"role": role, "content": message_text(message)})
        return out


class SessionStore:
    """LRU-bounded, thread-safe session registry."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(self, documents: list[Document], provider: str, model: str,
               temperature: float, chunk_size: int, chunk_overlap: int,
               retriever_k: int) -> Session:
        index, ingest = build_index(documents, provider, chunk_size, chunk_overlap)
        session = Session(
            id=uuid.uuid4().hex,
            provider=provider, model=model, temperature=temperature,
            retriever_k=retriever_k, index=index, ingest=ingest,
        )
        with self._lock:
            self._sessions[session.id] = session
            # Each session holds a FAISS index in memory; evict the oldest
            # rather than growing without bound.
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

    def clear_history(self, session_id: str) -> None:
        self.get(session_id).history.clear()

    def ask(self, session_id: str, question: str,
            temperature: float | None = None,
            retriever_k: int | None = None) -> str:
        """Answer within a session, keeping its conversation intact.

        Changing the temperature or k rebuilds the chain but reuses the same
        history object, so tuning a slider mid-conversation no longer wipes it.
        """
        session = self.get(session_id)
        if temperature is not None:
            session.temperature = temperature
        if retriever_k is not None:
            session.retriever_k = retriever_k

        llm = get_llm(session.provider, session.model, session.temperature)
        chain = build_chain(
            llm, session.index, session.retriever_k, lambda _sid: session.history
        )
        return chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session.id}},
        )

    def stats(self) -> dict:
        with self._lock:
            return {"sessions": len(self._sessions), "max_sessions": self._max}


store = SessionStore()
