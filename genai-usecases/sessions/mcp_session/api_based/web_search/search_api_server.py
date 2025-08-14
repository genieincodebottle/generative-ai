# fastapi_search_server.py
import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from serpapi.google_search import GoogleSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Search API Server",
    description="Web search API using SerpApi with Google Gemini integration support",
    version="1.0.0"
)

# SerpApi configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    logger.warning("SERPAPI_API_KEY environment variable not set. Search functionality will not work.")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    num_results: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    location: Optional[str] = Field(default=None, description="Location for localized results")

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: int

class SearchResponse(BaseModel):
    query: str
    location: Optional[str]
    total_results: int
    organic_results: List[SearchResult]
    answer_box: Optional[Dict[str, Any]] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    related_questions: Optional[List[str]] = None
    success: bool
    error: Optional[str] = None

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

@app.post("/search", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    """
    Perform web search using SerpApi
    """
    if not serpapi_client:
        raise HTTPException(
            status_code=500, 
            detail="SERPAPI_API_KEY not configured"
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400, 
            detail="Query cannot be empty"
        )
    
    try:
        results = await serpapi_client.search(
            request.query, 
            request.num_results, 
            request.location
        )
        
        # Parse organic results
        organic_results = []
        if "organic_results" in results:
            for i, result in enumerate(results["organic_results"][:request.num_results], 1):
                organic_results.append(SearchResult(
                    title=result.get("title", "No title"),
                    link=result.get("link", "No link"),
                    snippet=result.get("snippet", "No description"),
                    position=i
                ))
        
        # Extract answer box
        answer_box = None
        if "answer_box" in results:
            answer_box = {
                "title": results["answer_box"].get("title"),
                "answer": results["answer_box"].get("answer", results["answer_box"].get("snippet")),
                "source": results["answer_box"].get("link")
            }
        
        # Extract knowledge graph
        knowledge_graph = None
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            knowledge_graph = {
                "title": kg.get("title"),
                "type": kg.get("type"),
                "description": kg.get("description")
            }
        
        # Extract related questions
        related_questions = None
        if "related_questions" in results:
            related_questions = [
                q.get("question") for q in results["related_questions"][:5] 
                if q.get("question")
            ]
        
        return SearchResponse(
            query=request.query,
            location=request.location,
            total_results=len(organic_results),
            organic_results=organic_results,
            answer_box=answer_box,
            knowledge_graph=knowledge_graph,
            related_questions=related_questions,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/search")
async def web_search_get(
    q: str = Query(..., description="Search query"),
    num_results: int = Query(default=10, ge=1, le=100, description="Number of results"),
    location: Optional[str] = Query(default=None, description="Location filter")
):
    """
    Perform web search using GET method (for simple queries)
    """
    request = SearchRequest(
        query=q,
        num_results=num_results,
        location=location
    )
    return await web_search(request)

@app.get("/search/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    num_results: int = Query(default=5, ge=1, le=20, description="Number of results")
):
    """
    Simple search endpoint returning formatted text (for LLM consumption)
    """
    if not serpapi_client:
        return {"error": "SERPAPI_API_KEY not configured"}
    
    try:
        results = await serpapi_client.search(q, num_results)
        
        # Format as simple text for LLM
        formatted_results = []
        formatted_results.append(f"Search Results for: '{q}'")
        formatted_results.append("=" * 50)
        
        if "organic_results" in results:
            for i, result in enumerate(results["organic_results"][:num_results], 1):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No description")
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   {snippet}")
                formatted_results.append(f"   Source: {link}")
        
        # Add answer box if available
        if "answer_box" in results:
            answer_box = results["answer_box"]
            formatted_results.append(f"\n\nDirect Answer:")
            formatted_results.append(f"{answer_box.get('answer', answer_box.get('snippet', 'N/A'))}")
        
        return {"result": "\n".join(formatted_results)}
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_status = "configured" if SERPAPI_API_KEY else "not configured"
    return {
        "status": "healthy",
        "service": "Search API Server",
        "serpapi_key": api_status,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Search API Server",
        "version": "1.0.0",
        "description": "Web search API using SerpApi with Google Gemini integration support",
        "endpoints": {
            "POST /search": "Structured web search with full response",
            "GET /search": "Web search via GET parameters", 
            "GET /search/simple": "Simple text-formatted search for LLMs",
            "GET /health": "Health check endpoint",
            "GET /": "This information"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Search API Server...")
    print("Make sure to set SERPAPI_API_KEY environment variable")
    print("API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)