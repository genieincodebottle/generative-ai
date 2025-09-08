## 🔍 Gemini LLM-powered MCP Client integrated with Web Search Remote MCP Server

A local web search system that combines Google's **Gemini LLM** with a **FastMCP** tool-calling interface and a **Streamlit UI**. Uses **SerpApi** for live, real-time search results via a custom MCP server to show Remote MCP Server capabilities with Gemini LLM.

![alt text](images/mcp_flow.png)

### Features

- ✅ Google's Gemini API integration with Remote FastMCP Server
- ✅ Tool calling using [FastMCP based Remote MCP Server](https://github.com/jlowin/fastmcp)
- ✅ Custom MCP Server with live web search via SerpApi

### [MCP Official Doc](https://modelcontextprotocol.io/docs/getting-started/intro)

### Installation

   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\mcp\web_search_mcp
      ```
   2. Open the Project in VS Code or any code editor.
   3. Create a virtual environment:

      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. Create a requirements.txt file and add the following libraries:
      
      ```bash
        asyncio>=3.4.3
        aiohttp>=3.12.13
        python-dotenv>=1.1.0
        google-search-results>=2.4.2
        fastmcp>=2.8.1
        mcp>=1.9.4
        streamlit>=1.45.1
        requests>=2.32.4
        google-genai>=1.20.0
        anthropic>=0.54.0
        openai>=1.88.0
      ```
   5. Install dependencies:
      ```bash
      uv pip install -r requirements.txt
      ```
   6. Set up environment variables
      * Rename .env.example to .env
      * Update the file with your API keys:
      
      ```bash
      GOOGLE_API_KEY=your_key_here # Using the free-tier API 
      SERPAPI_API_KEY=your_serpapi_key # Using free-tier
      ```
      * 🔑 Get your API keys:
      
        For **GOOGLE_API_KEY** follow this -> https://aistudio.google.com/app/apikey

        For **SERPAPI_API_KEY** follow this -> https://serpapi.com/manage-api-key

   7. Run the MCP Server at one terminal

        ```bash
        cd genai-usecases\mcp\web_search_mcp
        .venv\Scripts\activate
        python web_search_mcp_server.py --host localhost --port 8000
        ```

   8.  Run the Streamlit Client App in another terminal (Split VSCode terminal or Open window's terminal)

        ```bash
        cd genai-usecases\mcp\web_search_mcp
        .venv\Scripts\activate
        streamlit run gemini_mcp_client.py
        ```

   9. Once started, go to:

        http://localhost:8501

        Use the sidebar to view Gemini API status, adjust model/temperature, and run queries with real-time web search using SerpApi.

        ![alt text](images/app.png)

### ⚙️ How It Works
- The Remote MCP server exposes tools (like web_search) via FastMCP over Streamable HTTP.
- The Streamlit app uses Gemini's function calling to request results from the Remote Web Saerch MCP server.
- Gemini combines LLM reasoning with live data via SerpApi using tool calling.