"""Natural language in, Cypher and an answer out.

Imports no web framework. The Neo4j connection is cached here rather than
with Streamlit's ``@st.cache_resource``, which is what tied the original
version to a running Streamlit session.

The chain is assembled by hand rather than with ``GraphCypherQAChain``, for
one reason: that chain generates Cypher and executes it in a single call, so
there is no point at which you can inspect the query before it runs. Splitting
generate / validate / execute is what turns the read-only check into an actual
gate instead of an after-the-fact report.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from services.config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    PROVIDERS,
    api_key_for,
)
from services.cypher_guard import UnsafeCypherError, assert_read_only
from services.llm_text import message_text

MAX_ROWS = 50


class GraphUnavailable(RuntimeError):
    """Neo4j could not be reached, or refused the credentials."""


class ProviderError(RuntimeError):
    """Unknown provider, or its API key is missing."""


@dataclass
class Answer:
    question: str
    answer: str | None = None
    cypher: str | None = None
    rows: list = field(default_factory=list)
    error: str | None = None
    success: bool = True


CYPHER_PROMPT = ChatPromptTemplate.from_template(
    "You are a Neo4j expert. Write ONE Cypher query that answers the question.\n\n"
    "Schema:\n{schema}\n\n"
    "Rules:\n"
    "- Read only. Never use CREATE, MERGE, DELETE, SET, REMOVE or DROP.\n"
    "- Return only the Cypher. No explanation, no markdown fences.\n"
    "- Always include a RETURN clause.\n"
    "- Add LIMIT {limit} unless the question asks for a single value.\n\n"
    "Question: {question}\n"
    "Cypher:"
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only these query results.\n"
    "If the results are empty, say the graph has no matching data.\n\n"
    "Question: {question}\n"
    "Cypher: {cypher}\n"
    "Results: {rows}\n\n"
    "Answer:"
)

_graph = None
_graph_lock = threading.Lock()


def get_graph(refresh: bool = False):
    """Connect to Neo4j once and reuse it. Raises :class:`GraphUnavailable`."""
    global _graph
    with _graph_lock:
        if _graph is not None and not refresh:
            return _graph
        try:
            from langchain_neo4j import Neo4jGraph

            # refresh_schema=False: Neo4jGraph calls apoc.meta.data() on
            # construction, and APOC is a plugin the stock community image
            # does not ship. services/schema.py derives the schema from
            # built-in procedures instead.
            graph = Neo4jGraph(
                url=NEO4J_URI, username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD, refresh_schema=False,
            )
            graph.query("RETURN 1")          # prove the connection works
        except Exception as exc:
            raise GraphUnavailable(
                f"Could not connect to Neo4j at {NEO4J_URI}: {exc}\n"
                f"Start one with:\n"
                f"  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 "
                f"-e NEO4J_AUTH=neo4j/your-password neo4j:5-community\n"
                f"then set NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD in .env."
            ) from exc
        _graph = graph
        return _graph


_schema_cache: str | None = None


def graph_schema(refresh: bool = False) -> str:
    """The schema the model is shown. Cached; it is several queries."""
    global _schema_cache
    if _schema_cache is not None and not refresh:
        return _schema_cache
    from services.schema import schema_for

    _schema_cache = schema_for(get_graph())
    return _schema_cache


def get_llm(provider: str, model: str):
    if provider not in PROVIDERS:
        raise ProviderError(f"Unknown provider: {provider}")
    key = api_key_for(provider)
    if not key:
        raise ProviderError(
            f"{PROVIDERS[provider]['env_key']} is not set. "
            f"Add it to your .env file and restart the API."
        )

    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=key, temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)


def clean_cypher(raw: str) -> str:
    """Strip the markdown fences and prose models wrap around Cypher."""
    text = str(raw).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"^cypher\s*:?\s*", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(";").strip()


def generate_cypher(question: str, provider: str, model: str) -> str:
    """Ask the model for Cypher, clean it, and refuse it if it writes."""
    llm = get_llm(provider, model)
    raw = message_text((CYPHER_PROMPT | llm).invoke({
        "schema": graph_schema(), "question": question, "limit": MAX_ROWS,
    }))
    cypher = clean_cypher(raw)
    assert_read_only(cypher)          # before anything touches the database
    return cypher


def ask(question: str, provider: str, model: str) -> Answer:
    """Answer a question about the graph. Nothing that writes ever executes."""
    if not question or not question.strip():
        return Answer(question=question, success=False,
                      error="The question is empty.")
    question = question.strip()

    try:
        cypher = generate_cypher(question, provider, model)
    except UnsafeCypherError as exc:
        return Answer(question=question, success=False, error=str(exc))
    except (GraphUnavailable, ProviderError):
        raise
    except Exception as exc:
        return Answer(question=question, success=False,
                      error=f"Could not generate Cypher: {exc}")

    try:
        rows = get_graph().query(cypher)[:MAX_ROWS]
    except Exception as exc:
        return Answer(question=question, cypher=cypher, success=False,
                      error=f"The query failed: {exc}")

    try:
        llm = get_llm(provider, model)
        answer = message_text((ANSWER_PROMPT | llm).invoke({
            "question": question, "cypher": cypher, "rows": rows,
        }))
    except Exception as exc:
        return Answer(question=question, cypher=cypher, rows=rows,
                      success=False, error=f"Could not phrase an answer: {exc}")

    return Answer(question=question, answer=answer, cypher=cypher, rows=rows)
