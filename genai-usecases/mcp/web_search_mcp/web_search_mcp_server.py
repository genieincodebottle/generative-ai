"""
SerpApi Remote MCP Server with HTTP Streamable Transport
"""
import os
import asyncio
import argparse
import logging
from typing import Any, Dict
from dotenv import load_dotenv

from fastmcp import FastMCP
from serpapi.google_search import GoogleSearch


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create the MCP server
mcp = FastMCP(name="SerpApi Search Server")

# SerpApi API configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    logger.warning("SERPAPI_API_KEY environment variable not set. Server will not function properly.")

class SerpApiClient:
    """Client for interacting with SerpApi"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def search(self, query: str, num_results: int = 10, location: str = None) -> Dict[str, Any]:
        """Perform a search using SerpApi"""
        params = {
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us",
            "google_domain": "google.com",
            "device": "desktop",
            "safe": "active",
            "num": min(num_results, 100),
            "output": "json"
        }
        
        if location:
            params["location"] = location
        
        try:
            search = GoogleSearch(params)
            results = await asyncio.to_thread(search.get_dict)
            return results
        except Exception as e:
            logger.error(f"SerpApi search error: {e}")
            raise

# Initialize SerpApi client
serpapi_client = SerpApiClient(SERPAPI_API_KEY) if SERPAPI_API_KEY else None

@mcp.tool()
async def web_search(query: str, num_results: int = 10, location: str = None) -> str:
    """
    Search the web using SerpApi
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 10, max: 100)
        location: Optional location for localized results
    
    Returns:
        Formatted search results as text
    """
    if not serpapi_client:
        return "WEB SEARCH Error: SERPAPI_API_KEY not configured"
    
    if not query.strip():
        return "WEB SEARCH Error: Query cannot be empty"
    
    if num_results < 1 or num_results > 100:
        return "WEB SEARCH Error: num_results must be between 1 and 100"
    
    try:
        results = await serpapi_client.search(query, num_results, location)
        
        # Format the results for better Lambda client consumption
        formatted_results = []
        formatted_results.append(f"Search Results for: '{query}'")
        if location:
            formatted_results.append(f"Location: {location}")
        formatted_results.append("=" * 50)
        
        # Add organic results
        if "organic_results" in results:
            for i, result in enumerate(results["organic_results"][:num_results], 1):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No description")
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   Link: {link}")
                formatted_results.append(f"   Description: {snippet}")
        
        # Add answer box if available
        if "answer_box" in results:
            answer_box = results["answer_box"]
            formatted_results.append(f"\n\nAnswer Box:")
            formatted_results.append(f"Title: {answer_box.get('title', 'N/A')}")
            formatted_results.append(f"Answer: {answer_box.get('answer', answer_box.get('snippet', 'N/A'))}")
            formatted_results.append(f"Source: {answer_box.get('link', 'N/A')}")
        
        # Add knowledge graph if available
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            formatted_results.append(f"\n\nKnowledge Graph:")
            formatted_results.append(f"Title: {kg.get('title', 'N/A')}")
            formatted_results.append(f"Type: {kg.get('type', 'N/A')}")
            formatted_results.append(f"Description: {kg.get('description', 'N/A')}")
        
        # Add related questions if available
        if "related_questions" in results:
            formatted_results.append(f"\n\nRelated Questions:")
            for i, question in enumerate(results["related_questions"][:5], 1):
                formatted_results.append(f"{i}. {question.get('question', 'N/A')}")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error performing search: {str(e)}"

@mcp.tool()
async def health_check() -> str:
    """Health check endpoint for Lambda client testing"""
    return "MCP SerpApi Server is healthy and ready"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP SerpApi server with HTTP streamable transport")
    parser.add_argument("--port", type=int, default=8000, help="Localhost port to listen on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to listen on")
    args = parser.parse_args()
    
    print(f"Starting SerpApi MCP Server with HTTP streamable transport on {args.host}:{args.port}...")
    print("Make sure to set SERPAPI_API_KEY environment variable")
    print("Server will be available at: http://{}:{}/mcp".format(args.host, args.port))
    
    mcp.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        log_level="info"
    )