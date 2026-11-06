"""Web search providers, with no MCP or web framework in sight.

Import nothing from `fastmcp`, `streamlit` or `fastapi` here. That rule is what
lets the whole search layer be tested with no server running, and it is why the
MCP tool in `web_search_mcp_server.py` is a thin wrapper over these functions.

Four providers, with different setup requirements:

    DuckDuckGo   no key at all        <- the default
    Parallel     no key, opt-in       <- Unified MCP search
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
    providers = ["duckduckgo", "parallel"]
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


# The endpoint is fixed: caller-supplied URLs or transports are not accepted.
PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"
PARALLEL_USER_AGENT = (
    "generative-ai-web-search-mcp/1.0 "
    "(https://github.com/genieincodebottle/generative-ai)"
)


def _parallel_rpc(response, request_id: int) -> dict:
    """Read JSON or finite SSE replies. Only CR/LF delimit SSE lines."""
    import json
    import re

    response.raise_for_status()
    if response.is_redirect:
        raise SearchError("Parallel MCP redirects are not supported.")
    if "text/event-stream" in response.headers.get("Content-Type", ""):
        messages = []
        data = []
        # str.splitlines() would corrupt legal Unicode separators in JSON.
        for line in re.split(r"\r\n|\r|\n", response.content.decode("utf-8")) + [""]:
            if not line:
                if data:
                    messages.append(json.loads("\n".join(data)))
                    data = []
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
    else:
        messages = [response.json()]
    for message in messages:
        if isinstance(message, dict) and message.get("id") == request_id:
            if "error" in message:
                raise SearchError(f"Parallel MCP error: {message['error']}")
            if isinstance(message.get("result"), dict):
                return message["result"]
    raise SearchError("Parallel MCP returned no matching result.")


def _parallel_results(payload: dict, query: str, num_results: int) -> SearchResponse:
    import json

    if payload.get("isError"):
        detail = " ".join(c.get("text", "") for c in payload.get("content", [])
                          if c.get("type") == "text")
        raise SearchError(f"Parallel search failed: {detail}")
    data = payload.get("structuredContent")
    if data is None:
        # Older MCP servers return their JSON in a text content block.
        for block in payload.get("content", []):
            if block.get("type") == "text":
                data = json.loads(block["text"])
                break
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SearchError("Parallel search returned an invalid results payload.")
    results = []
    for hit in data["results"][:num_results]:
        if (not isinstance(hit, dict) or not isinstance(hit.get("url"), str)
                or not hit["url"].startswith(("https://", "http://"))
                or not isinstance(hit.get("excerpts"), list)
                or not all(isinstance(s, str) for s in hit["excerpts"])
                or not isinstance(hit.get("title") or "", str)):
            raise SearchError("Parallel search returned an invalid result.")
        results.append(SearchResult(title=hit.get("title") or hit["url"],
                                    link=hit["url"],
                                    snippet="\n".join(hit["excerpts"])))
    return SearchResponse(query=query, provider="parallel", results=results)


def _search_parallel(query: str, num_results: int) -> SearchResponse:
    """Opt-in Unified MCP search, anonymous unless a key is explicitly set.

    This small blocking client uses only request/reply POSTs, not a background
    GET stream. It exposes search only, not the remote server's other tools.
    """
    import requests

    try:
        with requests.Session() as session:
            # Session defaults cover initialization, notification and teardown too.
            session.headers.update({"User-Agent": PARALLEL_USER_AGENT,
                                    "Accept": "application/json, text/event-stream"})
            if os.getenv("PARALLEL_API_KEY"):
                session.headers["Authorization"] = f"Bearer {os.environ['PARALLEL_API_KEY']}"

            def post(method, request_id=None, params=None):
                body = {"jsonrpc": "2.0", "method": method}
                if request_id is not None:
                    body["id"] = request_id
                if params is not None:
                    body["params"] = params
                return session.post(PARALLEL_MCP_URL, json=body,
                                    timeout=(10, 60), allow_redirects=False)

            initialized = post("initialize", 1, {
                "protocolVersion": "2025-03-26", "capabilities": {},
                "clientInfo": {"name": "generative-ai-web-search-mcp", "version": "1.0"},
            })
            session_id = initialized.headers.get("Mcp-Session-Id")
            if session_id:
                session.headers["Mcp-Session-Id"] = session_id
            failed = True
            try:
                init = _parallel_rpc(initialized, 1)
                if init.get("protocolVersion") != "2025-03-26":
                    raise SearchError("Parallel MCP returned an unsupported protocol version.")
                session.headers["MCP-Protocol-Version"] = init["protocolVersion"]
                notice = post("notifications/initialized")
                if notice.status_code != 202:
                    notice.raise_for_status()
                    raise SearchError("Parallel MCP did not accept initialization.")
                tools = _parallel_rpc(post("tools/list", 2), 2)
                if not any(t.get("name") == "web_search" for t in tools.get("tools", [])):
                    raise SearchError("Parallel MCP does not offer web_search.")
                payload = _parallel_rpc(post("tools/call", 3, {
                    "name": "web_search",
                    "arguments": {"objective": query, "search_queries": [query]},
                }), 3)
                result = _parallel_results(payload, query, num_results)
                failed = False
                return result
            finally:
                if session_id:
                    try:
                        closed = session.delete(PARALLEL_MCP_URL, timeout=(10, 30),
                                                allow_redirects=False)
                        # The MCP spec permits servers without session termination.
                        if closed.status_code not in (404, 405):
                            closed.raise_for_status()
                            if closed.is_redirect:
                                raise SearchError("Parallel MCP teardown redirected.")
                    except (requests.RequestException, SearchError):
                        if not failed:
                            raise
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        raise SearchError(f"Parallel MCP request failed: {exc}") from exc


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
    if provider == "parallel":
        return _search_parallel(query, num_results)
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
