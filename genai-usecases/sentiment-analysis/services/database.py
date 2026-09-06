"""SQLite persistence for calls and their tagging results.

Raises :class:`DatabaseError` instead of writing to the screen. The original
version called ``st.error`` from here, which is what made this logic
untestable and unusable outside Streamlit.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from services.config import DB_PATH

SAMPLE_CALLS: list[tuple[int, str]] = [
    (101, "Customer expressed happiness about the quick resolution of their billing issue."),
    (102, "Customer was delighted with the advanced features of our new product."),
    (103, "Customer was upset about the delayed shipment and requested an expedited delivery."),
    (104, "Customer felt neutral and inquired about the specifications of various models."),
    (105, "Customer expressed dissatisfaction with the recent service, noting multiple unresolved issues."),
    (106, "Customer joyfully reported that our product exceeded their expectations."),
    (107, "Customer was frustrated due to a misunderstanding about warranty coverage."),
    (108, "Customer was indifferent when discussing the upcoming software update details."),
    (109, "Customer was angry about receiving the wrong order and demanded a prompt resolution."),
    (110, "Customer was thrilled to hear about our loyalty program upgrades."),
]


class DatabaseError(RuntimeError):
    """Raised when the database cannot be read or written."""


def _path(db_path: str | Path | None = None) -> Path:
    return Path(db_path) if db_path else DB_PATH


@contextmanager
def connection(db_path: str | Path | None = None):
    conn = sqlite3.connect(_path(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def exists(db_path: str | Path | None = None) -> bool:
    return _path(db_path).exists()


def initialize(db_path: str | Path | None = None) -> dict:
    """Create the tables and seed sample calls if the table is empty."""
    try:
        with connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS customer_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    call_details TEXT NOT NULL,
                    call_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS customer_tagging (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id INTEGER UNIQUE,
                    call_details TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    aggressiveness INTEGER NOT NULL,
                    tagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (call_id) REFERENCES customer_calls (id)
                )
            """)
            cur.execute("SELECT COUNT(*) FROM customer_calls")
            seeded = 0
            if cur.fetchone()[0] == 0:
                cur.executemany(
                    "INSERT INTO customer_calls (customer_id, call_details) VALUES (?, ?)",
                    SAMPLE_CALLS,
                )
                seeded = len(SAMPLE_CALLS)
            conn.commit()
        return {"initialized": True, "seeded": seeded}
    except sqlite3.Error as exc:
        raise DatabaseError(f"Database initialization failed: {exc}") from exc


def reset(db_path: str | Path | None = None) -> None:
    """Delete the database file. The next initialize() rebuilds it."""
    path = _path(db_path)
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            raise DatabaseError(f"Could not delete {path}: {exc}") from exc


def is_ready(db_path: str | Path | None = None) -> bool:
    """True when the file exists *and* the tables are there.

    A file with no tables is a real state - it happens after a partial reset -
    and it looked like a working database to the original code.
    """
    if not exists(db_path):
        return False
    try:
        with connection(db_path) as conn:
            conn.execute("SELECT 1 FROM customer_calls LIMIT 1")
            conn.execute("SELECT 1 FROM customer_tagging LIMIT 1")
        return True
    except sqlite3.OperationalError:
        return False


def fetch_calls(db_path: str | Path | None = None) -> list[dict]:
    try:
        with connection(db_path) as conn:
            rows = conn.execute("SELECT * FROM customer_calls ORDER BY id").fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError as exc:
        raise DatabaseError("The database is not initialized yet.") from exc


def save_tagging(call_id: int, call_details: str, sentiment: str,
                 aggressiveness: int, db_path: str | Path | None = None) -> None:
    try:
        with connection(db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO customer_tagging
                   (call_id, call_details, sentiment, aggressiveness, tagged_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (call_id, call_details, sentiment, aggressiveness,
                 datetime.now().isoformat()),
            )
            conn.commit()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not save the tagging result: {exc}") from exc


def fetch_taggings(db_path: str | Path | None = None) -> list[dict]:
    try:
        with connection(db_path) as conn:
            rows = conn.execute("""
                SELECT ct.*, cc.customer_id, cc.call_time
                FROM customer_tagging ct
                JOIN customer_calls cc ON ct.call_id = cc.id
                ORDER BY ct.tagged_at DESC
            """).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError as exc:
        raise DatabaseError("The database is not initialized yet.") from exc


def counts(db_path: str | Path | None = None) -> dict:
    try:
        with connection(db_path) as conn:
            calls = conn.execute("SELECT COUNT(*) FROM customer_calls").fetchone()[0]
            tagged = conn.execute("SELECT COUNT(*) FROM customer_tagging").fetchone()[0]
        return {"calls": calls, "tagged": tagged}
    except sqlite3.OperationalError as exc:
        raise DatabaseError("The database is not initialized yet.") from exc
