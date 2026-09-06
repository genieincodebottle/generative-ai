"""Web search providers, with no MCP or web framework in sight.

Import nothing from `fastmcp`, `streamlit` or `fastapi` here. That rule is what
lets the whole search layer be tested with no server running, and it is why the
MCP tool in `web_search_mcp_server.py` is a thin wrapper over these functions.

Three providers, tried in order of how much they ask of you:

    DuckDuckGo   no key at all        <- the default
    Tavily       free key             <- better snippets, built for RAG
    SerpApi      paid key             <- real Google SERP features

The original server supported SerpApi only, and refused to do anything without
`SERPAPI_API_KEY`. That is a hard stop before a reader has seen the project
work even once, which is the wrong first experience for a demo about MCP
rather than about search.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field


class SearchError(RuntimeError):
    """No provider could answer, with a reason worth reading."""


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str


@dataclass
class SearchResponse:
    query: str
    provider: str
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""


# --------------------------------------------------------------------------
# Provider availability
# --------------------------------------------------------------------------

def tavily_key() -> str | None:
    return os.getenv("TAVILY_API_KEY") or None


def serpapi_key() -> str | None:
    return os.getenv("SERPAPI_API_KEY") or None


def available_providers() -> list[str]:
    """Which providers can actually run right now.

    DuckDuckGo is always listed: it needs no key. That is the point.
    """
    providers = ["duckduckgo"]
    if tavily_key():
        providers.append("tavily")
    if serpapi_key():
        providers.append("serpapi")
    return providers


def default_provider() -> str:
    """Prefer a real search API when a key exists, else DuckDuckGo."""
    for name in ("tavily", "serpapi"):
        if name in available_providers():
            return name
    return "duckduckgo"


# --------------------------------------------------------------------------
# Providers. Each is synchronous and blocking; `search()` runs them off-thread.
# --------------------------------------------------------------------------

def _search_duckduckgo(query: str, num_results: int) -> SearchResponse:
    try:
        from ddgs import DDGS
    except ImportError as exc:                       # pragma: no cover
        raise SearchError(
            "DuckDuckGo search needs the `ddgs` package: pip install ddgs"
        ) from exc

    hits = list(DDGS().text(query, max_results=num_results))
    return SearchResponse(
        query=query,
        provider="duckduckgo",
        results=[
            SearchResult(
                title=h.get("title", ""),
                link=h.get("href", "") or h.get("link", ""),
                snippet=h.get("body", ""),
            )
            for h in hits
        ],
    )


def _search_tavily(query: str, num_results: int) -> SearchResponse:
    import requests

    key = tavily_key()
    if not key:
        raise SearchError("TAVILY_API_KEY is not set.")

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": key,
            "query": query,
            "max_results": num_results,
            "include_answer": True,
        },
        timeout=30,
    )
    if response.status_code == 401:
        raise SearchError(
            "Tavily rejected the key (401). Check TAVILY_API_KEY, or drop it "
            "entirely and use the duckduckgo provider, which needs no key."
        )
    response.raise_for_status()
    payload = response.json()

    return SearchResponse(
        query=query,
        provider="tavily",
        answer=payload.get("answer") or "",
        results=[
            SearchResult(
                title=r.get("title", ""),
                link=r.get("url", ""),
                snippet=r.get("content", ""),
            )
            for r in payload.get("results", [])
        ],
    )


def _search_serpapi(query: str, num_results: int,
                    location: str | None = None) -> SearchResponse:
    import requests

    key = serpapi_key()
    if not key:
        raise SearchError("SERPAPI_API_KEY is not set.")

    params = {
        "q": query, "api_key": key, "hl": "en", "gl": "us",
        "google_domain": "google.com", "device": "desktop", "safe": "active",
        "num": min(num_results, 100), "output": "json",
    }
    if location:
        params["location"] = location

    response = requests.get("https://serpapi.com/search", params=params,
                            timeout=30)
    if response.status_code == 401:
        raise SearchError(
            "SerpApi rejected the key (401). Keys are revoked when "
            "regenerated. Use the duckduckgo provider, which needs no key."
        )
    response.raise_for_status()
    payload = response.json()

    answer = ""
    if "answer_box" in payload:
        box = payload["answer_box"]
        answer = box.get("answer") or box.get("snippet") or ""

    return SearchResponse(
        query=query,
        provider="serpapi",
        answer=answer,
        results=[
            SearchResult(
                title=r.get("title", ""),
                link=r.get("link", ""),
                snippet=r.get("snippet", ""),
            )
            for r in payload.get("organic_results", [])[:num_results]
        ],
    )


# --------------------------------------------------------------------------
# The public surface
# --------------------------------------------------------------------------

MAX_RESULTS = 25


def validate(query: str, num_results: int) -> tuple[str, int]:
    text = (query or "").strip()
    if not text:
        raise SearchError("The query is empty.")
    if not 1 <= num_results <= MAX_RESULTS:
        raise SearchError(
            f"num_results must be between 1 and {MAX_RESULTS}, got {num_results}."
        )
    return text, num_results


def search_sync(query: str, num_results: int = 5, provider: str | None = None,
                location: str | None = None) -> SearchResponse:
    """Run one search. Blocking; `search()` is the async wrapper."""
    query, num_results = validate(query, num_results)
    provider = provider or default_provider()

    if provider == "duckduckgo":
        return _search_duckduckgo(query, num_results)
    if provider == "tavily":
        return _search_tavily(query, num_results)
    if provider == "serpapi":
        return _search_serpapi(query, num_results, location)
    raise SearchError(
        f"Unknown provider: {provider}. "
        f"Available: {', '.join(available_providers())}"
    )


async def search(query: str, num_results: int = 5, provider: str | None = None,
                 location: str | None = None) -> SearchResponse:
    """Async wrapper, so the MCP tool never blocks its event loop."""
    return await asyncio.to_thread(search_sync, query, num_results, provider,
                                   location)


def format_results(response: SearchResponse) -> str:
    """Render a response as the plain text an LLM reads well."""
    lines = [f"Search results for: '{response.query}'  (via {response.provider})",
             "=" * 50]

    if response.answer:
        lines += ["", "Answer:", response.answer, ""]

    if not response.results:
        lines.append("No results.")
        return "\n".join(lines)

    for i, result in enumerate(response.results, 1):
        lines.append(f"\n{i}. {result.title}")
        lines.append(f"   Link: {result.link}")
        lines.append(f"   Description: {result.snippet}")

    return "\n".join(lines)
