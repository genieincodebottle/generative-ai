"""
MCP Client 
"""
import os
import asyncio
import argparse
import logging
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GeminiMCPClient:
    """Gemini client with MCP HTTP Streamable Integration"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000/mcp"):
        # Initialize Gemini
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Gemini configuration constants
        #self.model_name = "gemini-2.0-flash"
        self.model_name = "gemini-2.0-flash-lite"
        self.temperature = 0.7
        self.top_p = 1.0
        self.top_k = 1
        self.max_output_tokens = 3000
        
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._streams_context = None
        self._session_context = None
    
    async def connect_to_streamable_http_server(self, headers: Optional[dict] = None):
        """Connect to an MCP server running with HTTP Streamable transport"""
        try:
            self._streams_context = streamablehttp_client(
                url=self.server_url,
                headers=headers or {},
            )
            read_stream, write_stream, _ = await self._streams_context.__aenter__()

            self._session_context = ClientSession(read_stream, write_stream)
            self.session: ClientSession = await self._session_context.__aenter__()

            await self.session.initialize()
            logger.info(f"Successfully connected to MCP server at {self.server_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def cleanup(self):
        """Properly clean up the session and streams"""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def search_and_analyze(self, query: str, search_type: str = "web", max_results: int = 10) -> Dict[str, Any]:
        """Search and analyze results with Gemini"""
        try:
            logger.info(f"Searching for: {query} (type: {search_type})")
            
            if not self.session:
                raise Exception("Not connected to MCP server. Call connect_to_streamable_http_server() first.")
            
            search_data = await self._perform_search(query, max_results)
            
            if not search_data:
                return {
                    "query": query,
                    "search_type": search_type,
                    "error": "No search results returned",
                    "search_results": [],
                    "analysis": None
                }
            
            logger.info(f"Search completed successfully")
            
            # Analyze with Gemini
            analysis = await self._analyze_with_gemini(query, search_data)
            
            return {
                "query": query,
                "search_type": search_type,
                "search_results": search_data,
                "analysis": analysis,
                "total_results": len(search_data) if isinstance(search_data, list) else 1
            }
                    
        except Exception as e:
            logger.error(f"Search and analysis failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "query": query,
                "search_type": search_type,
                "error": str(e),
                "search_results": [],
                "analysis": None
            }
    
    async def _perform_search(self, query: str, max_results: int) -> Any:
        """Perform the actual search with better error handling"""
        try:
            # Call the search tool with timeout
            result = await asyncio.wait_for(
                self.session.call_tool("web_search", {
                    "query": query,
                    "num_results": max_results
                }),
                timeout=60.0
            )
            
            # Extract search results from response
            return self._extract_tool_result(result)
                    
        except asyncio.TimeoutError:
            logger.error("Search operation timed out")
            raise Exception("Search operation timed out")
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            raise
    
    def _extract_tool_result(self, result) -> Any:
        """Extract tool result from MCP response"""
        try:
            # MCP tool results come in result.content
            if hasattr(result, 'content') and result.content:
                # Get the text content from the first content item
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    return content_item.text
                else:
                    return str(content_item)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error extracting tool result: {e}")
            return None
    
    async def _analyze_with_gemini(self, query: str, search_results: Any) -> str:
        """Analyze search results using Gemini"""
        try:
            prompt_parts = [
                f"You are an AI research assistant analyzing search results.",
                f"User Query: {query}",
                "",
                f"Search Results:",
                "```",
                str(search_results)[:4000],  # Limit the length to avoid token limits
                "```",
                "",
                "Please provide a comprehensive analysis that includes:",
                "1. **Summary**: Key findings and main themes",
                "2. **Insights**: Important insights drawn from the sources",
                "3. **Analysis**: Detailed analysis of the information",
                "4. **Conclusions**: Your conclusions based on the evidence",
                "5. **Follow-up**: Suggested follow-up questions or research areas",
                "",
                "Format your response in clear sections with headers."
            ]
            
            prompt = "\n".join(prompt_parts)
            
            # Create configuration for Gemini
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                response_mime_type='text/plain'
            )
            
            # Generate response with Gemini API using asyncio.to_thread for better async handling
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
            except AttributeError:
                # Fallback for older asyncio versions
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                )
            
            if response and response.text:
                return response.text
            else:
                logger.warning("Gemini returned no content")
                return "Analysis unavailable: No content returned from Gemini."
                
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return f"Analysis failed: {str(e)}"
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools with better error handling"""
        try:
            if not self.session:
                raise Exception("Not connected to MCP server")
            
            tools = await asyncio.wait_for(self.session.list_tools(), timeout=30.0)
            
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                }
                for tool in tools.tools
            ]
                    
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if MCP server is healthy with timeout"""
        try:
            if not self.session:
                return False
            
            # Try to list tools as a health check
            await asyncio.wait_for(self.session.list_tools(), timeout=10.0)
            return True
                    
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    async def process_query_with_tools(self, query: str) -> str:
        """Process a query using available tools"""
        try:
            if not self.session:
                raise Exception("Not connected to MCP server")
            
            # Get available tools
            response = await self.session.list_tools()
            available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]
            
            logger.info(f"Available tools: {[tool['name'] for tool in available_tools]}")
            
            # For this example, we'll directly call the web_search tool
            # In a more sophisticated implementation, you could use an LLM to decide which tools to call
            if "web_search" in [tool['name'] for tool in available_tools]:
                result = await self.session.call_tool("web_search", {"query": query})
                search_results = self._extract_tool_result(result)
                
                # Analyze with Gemini
                analysis = await self._analyze_with_gemini(query, search_results)
                
                return f"Search Results:\n{search_results}\n\nAnalysis:\n{analysis}"
            else:
                return "No suitable tools available for this query."
                
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"Error processing query: {str(e)}"
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nGemini MCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query_with_tools(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

async def handle_search(query: str, client):
    """Handle search with better error reporting"""
    try:
        result = await client.search_and_analyze(query)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Display search results
        print(f"\n📋 Search Results ({result.get('total_results', 0)} found):")
        print("-" * 40)
        search_data = result['search_results']
        
        # Display raw results (first 500 chars)
        if isinstance(search_data, str):
            display_data = search_data[:500] + "..." if len(search_data) > 500 else search_data
            print(display_data)
        else:
            print(str(search_data)[:500] + "..." if len(str(search_data)) > 500 else str(search_data))
        
        # Display analysis
        if result['analysis']:
            print(f"\n🤖 Gemini Analysis:")
            print("-" * 30)
            print(result['analysis'])
    
    except Exception as e:
        print(f"❌ Search handling error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

async def main():
    """Main function with better error handling"""
    parser = argparse.ArgumentParser(description="Run Gemini MCP Streamable HTTP Client")
    parser.add_argument(
        "--mcp-localhost-port", type=int, default=8000, help="Localhost port for MCP server"
    )
    parser.add_argument(
        "--mode", choices=["search", "chat"], default="search", 
        help="Run mode: 'search' for single search, 'chat' for interactive mode"
    )
    args = parser.parse_args()
    
    print("🚀 Starting Gemini + MCP HTTP Streamable Client...")
    server_url = f"http://localhost:{args.mcp_localhost_port}/mcp"
    
    client = GeminiMCPClient(server_url)
    
    try:
        # Connect to server
        print("Connecting to server...")
        await client.connect_to_streamable_http_server()
        
        # Test server connection with timeout
        print("Testing server connection...")
        is_healthy = await asyncio.wait_for(client.health_check(), timeout=15.0)
        
        if not is_healthy:
            print(f"❌ Cannot connect to MCP server at {server_url}")
            print("Make sure the MCP server is running with HTTP streamable transport.")
            return
        
        print("✅ Server connection successful!")
        
        # List available tools
        tools = await client.list_available_tools()
        logger.info(f"Connected to MCP server. Available tools: {len(tools)}")
        print(f"Available tools: {[tool['name'] for tool in tools]}")
        
        if args.mode == "search":
            # Perform single search
            await handle_search("What is the current weather of Bengaluru?", client)
        else:
            # Start interactive chat loop
            await client.chat_loop()
        
    except asyncio.TimeoutError:
        print("❌ Connection timeout. Make sure the MCP server is running and accessible.")
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        print(f"❌ Connection error: {e}")
        print(f"Make sure the MCP server is running on {server_url}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        # For Python 3.11+, you might need to set the event loop policy
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")