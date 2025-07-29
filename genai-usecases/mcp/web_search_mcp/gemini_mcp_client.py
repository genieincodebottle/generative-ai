"""
Streamlit UI for MCP Client with Google's Gemini API Response Integration
"""
import os
import logging
import asyncio
import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client
from google import genai


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemini MCP Client",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .response-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-info {
        color: #17a2b8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = None
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None

async def initialize_clients():
    """Initialize MCP and Gemini clients"""
    try:
        # Initialize MCP client
        mcp_client = Client("http://localhost:8000/mcp")
        
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ GEMINI_API_KEY environment variable not set")
            return None, None
        
        # Initialize Gemini client
        gemini_client = genai.Client(api_key=api_key)
        
        return mcp_client, gemini_client
    
    except Exception as e:
        st.error(f"❌ Error initializing clients: {str(e)}")
        return None, None

async def perform_search(query, model_name, temperature, mcp_client, gemini_client):
    """Perform search using MCP and Gemini"""
    try:
        async with mcp_client:
            # Test MCP connection
            tools = await mcp_client.list_tools()
            st.sidebar.success(f"✅ Connected to MCP server ({len(tools)} tools available)")
            
            # Generate response with Gemini
            response = await gemini_client.aio.models.generate_content(
                model=model_name,
                contents=f"Search for: {query}. Use web search MCP Server to find comprehensive and up-to-date information and synthesise details.",
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    tools=[mcp_client.session],
                ),
            )
            return response.text
            
    except Exception as e:
        raise Exception(f"Search failed: {str(e)}")

def main():
    # Header
    st.markdown('<h2>🔍 MCP Client</h2>', unsafe_allow_html=True)
    st.markdown('(Google\'s Gemini API Response API with FastMCP Integration)', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration and status
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Server status
        mcp_url = st.text_input("MCP Server URL", value="http://localhost:8000/mcp/", disabled=True)
        
        # API Key status
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.markdown('<p class="status-success">✅ Gemini API Key: Configured</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ Gemini API Key: Not Set</p>', unsafe_allow_html=True)
            st.error("Please set GEMINI_API_KEY environment variable")
        
        # Model settings
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.0-flash-exp", "gemini-2.0-flash-lite", "gemini-1.5-pro"],
            index=0
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        st.markdown("---")
        
    # Main search interface
    
    search_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., Who won the 2025 WTC Cricket trophy?",
        help="Enter any question or topic you'd like to search for"
    )
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    
    # Quick search suggestions
    st.subheader("💡 Quick Search Suggestions")
    suggestions = [
        "Latest tech news today",
        "Current weather in Bengaluru",
        "Recent developments in AI",
    ]
    
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                search_query = suggestion
                search_button = True
    
    # Process search
    if search_button and search_query:
        if not api_key:
            st.error("❌ Please set your GEMINI_API_KEY environment variable first")
            return
        
        with st.spinner("🔄 Initializing clients and performing search..."):
            try:
                # Initialize clients
                mcp_client, gemini_client = asyncio.run(initialize_clients())
                
                if mcp_client and gemini_client:
                    # Perform search
                    result = asyncio.run(perform_search(search_query, model_name, temperature, mcp_client, gemini_client))
                    
                    # Display results
                    st.subheader(f"🎯 Results for: {search_query}")
                    st.markdown(result)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Search completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Search error: {e}")
    
    elif search_button and not search_query:
        st.warning("⚠️ Please enter a search query")
    

if __name__ == "__main__":
    main()