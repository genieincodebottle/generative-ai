"""Session lifecycle and memory retention. No API key, no network.

The store is exercised with a stub index and a stub chain, so these tests
cover the behaviour that actually regressed - conversation memory surviving a
config change - without calling a model.
"""

import pytest

from services.rag_service import IngestResult
from services.session_store import Session, SessionNotFound, SessionStore


@pytest.fixture
def store(monkeypatch):
    """A store whose create() skips embedding and indexing."""
    s = SessionStore(max_sessions=3)

    def fake_create(documents, provider, model, temperature, chunk_size,
                    chunk_overlap, retriever_k):
        import uuid
        session = Session(
            id=uuid.uuid4().hex, provider=provider, model=model,
            temperature=temperature, retriever_k=retriever_k,
            index=object(),
            ingest=IngestResult(filenames=[d.filename for d in documents],
                                pages=1, chunks=1),
        )
        with s._lock:
            s._sessions[session.id] = session
            while len(s._sessions) > s._max:
                s._sessions.popitem(last=False)
        return session

    monkeypatch.setattr(s, "create", fake_create)
    return s


def make(store, name="doc.pdf"):
    from services.rag_service import Document
    return store.create([Document(name, b"x")], "Gemini", "gemini-flash-latest",
                        0.3, 2000, 200, 3)


def test_create_returns_a_session_with_ingest_details(store):
    session = make(store)
    assert session.ingest.filenames == ["doc.pdf"]
    assert store.get(session.id) is session


def test_get_unknown_session_raises(store):
    with pytest.raises(SessionNotFound):
        store.get("nope")


def test_delete_removes_the_session(store):
    session = make(store)
    store.delete(session.id)
    with pytest.raises(SessionNotFound):
        store.get(session.id)


def test_delete_unknown_session_raises(store):
    with pytest.raises(SessionNotFound):
        store.delete("nope")


def test_history_starts_empty_and_records_turns(store):
    session = make(store)
    assert session.messages() == []
    session.history.add_user_message("What is this about?")
    session.history.add_ai_message("A test document.")
    assert session.messages() == [
        {"role": "user", "content": "What is this about?"},
        {"role": "assistant", "content": "A test document."},
    ]


def test_clear_history_keeps_the_session_and_its_index(store):
    session = make(store)
    session.history.add_user_message("hello")
    store.clear_history(session.id)
    assert session.messages() == []
    assert store.get(session.id) is session  # documents are still indexed


def test_history_object_is_owned_by_the_session(store):
    """The regression this store exists to prevent.

    The original app built its history dict inside the chain factory, so every
    rebuild - triggered by moving the temperature slider - silently discarded
    the conversation. Here the history belongs to the session, and rebuilding
    a chain cannot reach it.
    """
    session = make(store)
    session.history.add_user_message("remember this")
    before = session.history

    session.temperature = 0.9      # what a slider change does
    session.retriever_k = 7

    assert store.get(session.id).history is before
    assert len(store.get(session.id).messages()) == 1


def test_lru_eviction_drops_the_oldest_session(store):
    first = make(store, "a.pdf")
    second = make(store, "b.pdf")
    third = make(store, "c.pdf")
    fourth = make(store, "d.pdf")      # exceeds max_sessions=3

    with pytest.raises(SessionNotFound):
        store.get(first.id)
    for session in (second, third, fourth):
        assert store.get(session.id) is session


def test_access_refreshes_lru_position(store):
    first = make(store, "a.pdf")
    second = make(store, "b.pdf")
    make(store, "c.pdf")

    store.get(first.id)            # first is now the most recently used
    make(store, "d.pdf")           # so second should be evicted, not first

    assert store.get(first.id) is first
    with pytest.raises(SessionNotFound):
        store.get(second.id)


def test_stats_reports_occupancy(store):
    make(store)
    assert store.stats() == {"sessions": 1, "max_sessions": 3}
