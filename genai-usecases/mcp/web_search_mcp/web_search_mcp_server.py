"""Web search MCP server, over HTTP streamable transport.

This file is deliberately thin. Every MCP tool below is a wrapper over
`services/search.py`, which imports no web framework at all - so the search
logic can be tested with nothing running, and swapping MCP for another
transport would not touch it.

Providers, in order of what they ask of you:

    duckduckgo   no key at all   <- works out of the box
    tavily       free key
    serpapi      paid key

Run it:  python web_search_mcp_server.py --port 8000
"""

import argparse
import logging

from dotenv import load_dotenv
from fastmcp import FastMCP

from services import search as search_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

mcp = FastMCP(name="Web Search Server")


@mcp.tool()
async def web_search(query: str, num_results: int = 5,
                     provider: str = None, location: str = None) -> str:
    """Search the web.

    Args:
        query: The search query.
        num_results: How many results to return (1-25).
        provider: "duckduckgo" (no key), "tavily", or "serpapi". Defaults to
            the best one your environment can actually use.
        location: Localised results. SerpApi only; ignored elsewhere.

    Returns:
        Formatted search results as text.
    """
    try:
        response = await search_service.search(query, num_results, provider,
                                               location)
        return search_service.format_results(response)
    except search_service.SearchError as exc:
        # A bad query or an unusable provider is the caller's problem to fix,
        # and the message says which. It is returned as a RESULT rather than
        # raised, so an agent can read it and try something else.
        return f"Search error: {exc}"
    except Exception as exc:                       # pragma: no cover
        logger.exception("search failed")
        return f"Search error: {type(exc).__name__}: {exc}"


@mcp.tool()
async def list_providers() -> str:
    """Which search providers this server can currently use, and why."""
    available = search_service.available_providers()
    lines = [f"Default provider: {search_service.default_provider()}", ""]
    for name, needs in (("duckduckgo", "no key"),
                        ("tavily", "TAVILY_API_KEY"),
                        ("serpapi", "SERPAPI_API_KEY")):
        mark = "available" if name in available else f"needs {needs}"
        lines.append(f"  {name:12s} {mark}")
    return "\n".join(lines)


@mcp.tool()
async def health_check() -> str:
    """Health check, naming the provider that would actually be used."""
    return (f"Web Search MCP server is healthy. "
            f"Default provider: {search_service.default_provider()}. "
            f"Available: {', '.join(search_service.available_providers())}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the web search MCP server (HTTP streamable transport)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    print(f"Starting Web Search MCP Server on {args.host}:{args.port}")
    print(f"Default provider : {search_service.default_provider()}")
    print(f"Available        : {', '.join(search_service.available_providers())}")
    print("No API key is required - DuckDuckGo is used when no key is set.")
    print(f"Server will be available at http://{args.host}:{args.port}/mcp")

    mcp.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        log_level="info"
    )
