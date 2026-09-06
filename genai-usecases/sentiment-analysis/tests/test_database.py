"""Database tests against a temporary file, so they never touch your real DB."""

import pytest

from services import database


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    database.initialize(path)
    return path


def test_initialize_creates_tables_and_seeds(tmp_path):
    path = tmp_path / "fresh.db"
    assert not database.exists(path)
    result = database.initialize(path)
    assert result["seeded"] == len(database.SAMPLE_CALLS)
    assert database.is_ready(path)


def test_initialize_is_idempotent(db):
    # Running it twice must not duplicate the sample calls.
    result = database.initialize(db)
    assert result["seeded"] == 0
    assert database.counts(db)["calls"] == len(database.SAMPLE_CALLS)


def test_is_ready_false_for_missing_file(tmp_path):
    assert database.is_ready(tmp_path / "nope.db") is False


def test_is_ready_false_when_file_exists_but_tables_do_not(tmp_path):
    # A file with no tables is a real state after a partial reset, and it
    # looks like a working database to a plain os.path.exists check.
    path = tmp_path / "empty.db"
    with database.connection(path):
        pass
    assert database.exists(path) is True
    assert database.is_ready(path) is False


def test_fetch_calls_returns_seeded_rows(db):
    calls = database.fetch_calls(db)
    assert len(calls) == len(database.SAMPLE_CALLS)
    assert calls[0]["customer_id"] == 101


def test_fetch_calls_on_uninitialized_db_raises(tmp_path):
    path = tmp_path / "empty.db"
    with database.connection(path):
        pass
    with pytest.raises(database.DatabaseError):
        database.fetch_calls(path)


def test_save_tagging_then_fetch(db):
    call = database.fetch_calls(db)[0]
    database.save_tagging(call["id"], call["call_details"], "Positive", 2, db)
    rows = database.fetch_taggings(db)
    assert len(rows) == 1
    assert rows[0]["sentiment"] == "Positive"
    assert rows[0]["customer_id"] == 101


def test_save_tagging_upserts_rather_than_duplicating(db):
    call = database.fetch_calls(db)[0]
    database.save_tagging(call["id"], call["call_details"], "Positive", 2, db)
    database.save_tagging(call["id"], call["call_details"], "Negative", 9, db)
    rows = database.fetch_taggings(db)
    assert len(rows) == 1
    assert rows[0]["sentiment"] == "Negative"
    assert rows[0]["aggressiveness"] == 9


def test_counts_tracks_tagging(db):
    assert database.counts(db) == {"calls": 10, "tagged": 0}
    call = database.fetch_calls(db)[0]
    database.save_tagging(call["id"], call["call_details"], "Neutral", 1, db)
    assert database.counts(db)["tagged"] == 1


def test_reset_removes_the_file(db):
    database.reset(db)
    assert not database.exists(db)


def test_reset_on_missing_file_is_not_an_error(tmp_path):
    database.reset(tmp_path / "never-existed.db")
