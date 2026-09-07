"""The business logic: natural language in, SQL and an answer out.

Deliberately free of Streamlit and FastAPI imports. Everything here can be
called from a notebook, a test, or a different UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# LangChain 1.x moved the legacy chains into `langchain_classic`. Try the new
# home first so an up-to-date install works, and fall back to the old path so
# a 0.3.x environment keeps working too.
try:
    from langchain_classic.chains import create_sql_query_chain
except ImportError:  # langchain < 1.0
    from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from services import database
from services.llm import get_llm

# Only these statements are allowed to run. The model is asked for SELECTs, but
# "asked for" is not a guarantee, and this service is reachable over HTTP, so
# the check belongs here rather than in the prompt.
_READ_ONLY_PREFIXES = ("SELECT", "WITH")
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)

_ANSWER_PROMPT = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)


class UnsafeQueryError(RuntimeError):
    """Raised when the generated SQL is not a read-only statement."""


@dataclass
class QueryResult:
    """One end-to-end run. ``success`` tells you which half of this is filled in."""

    success: bool
    question: str
    sql_query: str | None = None
    raw_result: str | None = None
    answer: str | None = None
    error: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)


def clean_sql_query(query: str | None) -> str:
    """Strip the markdown and chat scaffolding models wrap around SQL.

    Handles ```sql fences, stray backticks, and the ``SQLQuery:`` prefix that
    LangChain's SQL chain emits.
    """
    if not query:
        return ""

    clean = str(query).strip()

    # ```sql ... ``` or ``` ... ```
    clean = re.sub(r"^```[a-zA-Z]*\s*\n?", "", clean)
    clean = re.sub(r"\n?```\s*$", "", clean)
    clean = clean.strip().strip("`").strip()

    # LangChain's SQL chain prefixes its output; later text is commentary.
    clean = re.sub(r"^SQLQuery:\s*", "", clean, flags=re.IGNORECASE)
    clean = re.split(r"\n\s*(?:SQLResult|Answer)\s*:", clean, flags=re.IGNORECASE)[0]

    return clean.strip().rstrip(";").strip()


def assert_read_only(sql: str) -> None:
    """Raise :class:`UnsafeQueryError` unless ``sql`` is a single read query."""
    if not sql:
        raise UnsafeQueryError("The model did not return a SQL statement.")
    if ";" in sql.strip().rstrip(";"):
        raise UnsafeQueryError("Multiple SQL statements are not allowed.")
    if not sql.upper().startswith(_READ_ONLY_PREFIXES):
        raise UnsafeQueryError("Only SELECT and WITH queries may be executed.")
    if _FORBIDDEN.search(sql):
        raise UnsafeQueryError("The generated SQL contains a write operation.")


def database_info(db_path: str | None = None) -> dict:
    return {
        "dialect": database.dialect(db_path),
        "tables": database.table_names(db_path),
    }


def generate_sql(question: str, provider: str, model: str, temperature: float = 0.0,
                 db_path: str | None = None) -> str:
    """Turn a question into a cleaned, validated SQL string."""
    llm = get_llm(provider, model, temperature)
    db = database.get_database(db_path)
    raw = create_sql_query_chain(llm, db).invoke({"question": question})
    sql = clean_sql_query(raw)
    assert_read_only(sql)
    return sql


def answer_question(question: str, provider: str, model: str, temperature: float = 0.0,
                    db_path: str | None = None) -> QueryResult:
    """Generate SQL, run it, and explain the result in plain language."""
    sql: str | None = None
    try:
        sql = generate_sql(question, provider, model, temperature, db_path)
        raw_result = database.run_sql(sql, db_path)

        llm = get_llm(provider, model, temperature)
        answer = (_ANSWER_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question, "query": sql, "result": raw_result}
        )

        # A DataFrame view as well, for the table and the CSV download. A failure
        # here must not lose the answer we already have.
        columns: list[str] = []
        rows: list[list] = []
        try:
            frame = database.run_sql_dataframe(sql, db_path)
            columns = [str(c) for c in frame.columns]
            rows = frame.astype(object).where(frame.notna(), None).values.tolist()
        except Exception:
            pass

        return QueryResult(
            success=True, question=question, sql_query=sql,
            raw_result=raw_result, answer=answer, columns=columns, rows=rows,
        )

    except Exception as exc:
        return QueryResult(
            success=False, question=question, sql_query=sql, error=str(exc)
        )
