"""Database access: connection, schema inspection, and query execution.

Knows nothing about LLMs, Streamlit, or FastAPI.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd
from langchain_community.utilities import SQLDatabase

from services.config import DB_PATH


class DatabaseError(RuntimeError):
    """Raised when the database is missing or a query cannot be executed."""


@lru_cache(maxsize=4)
def get_database(db_path: str | None = None) -> SQLDatabase:
    """Return a cached LangChain SQLDatabase handle.

    Cached because building it inspects the whole schema, which is wasted work
    on every request.
    """
    path = Path(db_path) if db_path else DB_PATH
    if not path.exists():
        raise DatabaseError(f"Database file not found at: {path}")
    return SQLDatabase.from_uri(f"sqlite:///{path}")


def table_names(db_path: str | None = None) -> list[str]:
    return list(get_database(db_path).get_usable_table_names())


def dialect(db_path: str | None = None) -> str:
    return get_database(db_path).dialect


def run_sql(sql: str, db_path: str | None = None) -> str:
    """Execute SQL and return LangChain's string rendering of the rows."""
    try:
        return get_database(db_path).run(sql)
    except Exception as exc:  # surfaced to the caller as a clean error
        raise DatabaseError(str(exc)) from exc


def run_sql_dataframe(sql: str, db_path: str | None = None) -> pd.DataFrame:
    """Execute SQL and return a DataFrame, for tabular display and CSV export."""
    path = Path(db_path) if db_path else DB_PATH
    if not path.exists():
        raise DatabaseError(f"Database file not found at: {path}")
    try:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query(sql, conn)
    except Exception as exc:
        raise DatabaseError(str(exc)) from exc


def preview_table(table: str, limit: int = 5, db_path: str | None = None) -> pd.DataFrame:
    """Return the first ``limit`` rows of ``table``.

    ``table`` is validated against the real table list rather than interpolated
    blindly, so a crafted table name cannot turn this into arbitrary SQL.
    """
    names = table_names(db_path)
    if table not in names:
        raise DatabaseError(f"Unknown table: {table}")
    return run_sql_dataframe(f'SELECT * FROM "{table}" LIMIT {int(limit)}', db_path)
