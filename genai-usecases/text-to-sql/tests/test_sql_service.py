"""Tests for the business logic. No API key and no network required.

The point of the layering is that the interesting logic - cleaning model
output and refusing to run anything that writes - is testable without an LLM.
"""

import pytest

from services.sql_service import UnsafeQueryError, assert_read_only, clean_sql_query


class TestCleanSqlQuery:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("SELECT 1", "SELECT 1"),
            ("```sql\nSELECT 1\n```", "SELECT 1"),
            ("```\nSELECT 1\n```", "SELECT 1"),
            ("```sqlite\nSELECT 1\n```", "SELECT 1"),
            ("`SELECT 1`", "SELECT 1"),
            ("SELECT 1;", "SELECT 1"),
            ("  SELECT 1  ", "SELECT 1"),
            # LangChain's SQL chain prefixes its output and may append commentary.
            ("SQLQuery: SELECT * FROM artists", "SELECT * FROM artists"),
            ("sqlquery: SELECT 1", "SELECT 1"),
            ("SELECT 1\nSQLResult: [(1,)]\nAnswer: one", "SELECT 1"),
        ],
    )
    def test_strips_scaffolding(self, raw, expected):
        assert clean_sql_query(raw) == expected

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_empty_input_gives_empty_string(self, empty):
        assert clean_sql_query(empty) == ""

    def test_preserves_inner_semicolons_free_query(self):
        sql = "SELECT Name FROM artists WHERE Name LIKE 'A%'"
        assert clean_sql_query(sql) == sql


class TestAssertReadOnly:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM artists",
            "select 1",
            "WITH x AS (SELECT 1) SELECT * FROM x",
        ],
    )
    def test_allows_reads(self, sql):
        assert_read_only(sql)  # must not raise

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE artists",
            "DELETE FROM artists",
            "UPDATE artists SET Name = 'x'",
            "INSERT INTO artists VALUES (1, 'x')",
            "ALTER TABLE artists ADD COLUMN x INT",
            "PRAGMA table_info(artists)",
            "ATTACH DATABASE 'other.db' AS other",
        ],
    )
    def test_rejects_writes(self, sql):
        with pytest.raises(UnsafeQueryError):
            assert_read_only(sql)

    def test_rejects_stacked_statements(self):
        with pytest.raises(UnsafeQueryError, match="Multiple SQL statements"):
            assert_read_only("SELECT 1; DROP TABLE artists")

    def test_rejects_write_hidden_after_a_select(self):
        with pytest.raises(UnsafeQueryError):
            assert_read_only("SELECT * FROM artists UNION SELECT 1; DELETE FROM artists")

    def test_rejects_empty(self):
        with pytest.raises(UnsafeQueryError):
            assert_read_only("")
