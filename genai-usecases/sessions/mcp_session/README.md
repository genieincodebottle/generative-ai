## Experiment with Agentic Tool Calls, API Calls, and MCP-based LLM Integration

### References

* [MCP Official Doc](https://modelcontextprotocol.io/docs/getting-started/intro)
* [FastMCP Official Doc](https://gofastmcp.com/getting-started/welcome)
* [FastAPI Official Doc](https://fastapi.tiangolo.com/)
* [FastAPI MCP GitHub](https://github.com/tadata-org/fastapi_mcp)
* [Streamlit UI Official Doc](https://streamlit.io/)

### Installation

```bash
cd mcp_session
pip install uv # If uv doesn't exist in your system
uv venv
.venv\Scripts\activate   # Linux: source .venv/bin/activate
uv pip install -r requirements.txt
```

### Environment Setup
Create a .env file and set the following keys:

```bash
GEMINI_API_KEY=your_google_genai_api_key
SERPAPI_API_KEY=your_serpapi_key
```

**For free tier Gemini API Key** -> https://aistudio.google.com/apikey 

**For free-tier SerpAPI API Key** -> https://serpapi.com/manage-api-key


## ⚡API based LLM Integration
### 1️⃣ Run the FastAPI Web Search Server (Terminal 1)

```bash
cd mcp_session
.venv\Scripts\activate
cd api_based\web_search
python search_api_server.py
```

2️⃣ Run the Streamlit Search Client (Terminal 2 or Split IDE Terminal)

```bash
cd mcp_session
.venv\Scripts\activate
cd api_based\web_search
python streamlit run search_api_client.py
```

## 🔗 MCP-based LLM Integration
### 1️⃣ Run the MCP Server (Terminal 1)

```bash
cd mcp_session
.venv\Scripts\activate
cd mcp_based
python web_search_mcp_server.py
```

### 2️⃣ Run the Streamlit MCP Client (Terminal 2)

```bash
cd mcp_session
.venv\Scripts\activate
cd mcp_based
streamlit run mcp_client.py
```
* How It Works
    - The Remote MCP server exposes tools (like web_search) via FastMCP over Streamable HTTP.
    - The Streamlit app uses Gemini's function calling to request results from the Remote Web Saerch MCP server.
    - Gemini combines LLM reasoning with live data via SerpApi using tool calling.