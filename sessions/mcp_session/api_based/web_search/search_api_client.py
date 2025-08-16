"""
Streamlit UI for Search API with LangChain Google Gemini Integration
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemini Search Client",
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
    .search-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def search_web(query: str, num_results: int = 5, api_url: str = "http://localhost:8000"):
    """Search the web using FastAPI backend"""
    try:
        response = requests.get(
            f"{api_url}/search/simple",
            params={"q": query, "num_results": num_results},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            return f"Search Error: {data['error']}"
        
        return data.get("result", "No results found")
        
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

def get_gemini_response(query: str, search_results: str, model_name: str, temperature: float):
    """Get response from Google Gemini using LangChain with search context"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not set in environment variables"
        
        # Initialize LangChain Gemini model
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=2048
        )
        
        # Create messages for the conversation
        system_message = SystemMessage(content="""
        You are a helpful AI assistant that provides comprehensive and accurate answers based on web search results.
        
        Your task:
        1. Analyze the provided search results carefully
        2. Provide a well-structured, informative response
        3. Cite relevant sources when possible
        4. Be clear and easy to understand
        5. If search results are insufficient, mention what additional information might be helpful
        """)
        
        human_message = HumanMessage(content=f"""
        Based on the following web search results, please answer the user's question comprehensively.

        User Question: {query}

        Search Results:
        {search_results}

        Please provide a detailed answer that directly addresses the question using the search results.
        """)
        
        # Get response from Gemini
        response = llm.invoke([system_message, human_message])
        
        return response.content
        
    except Exception as e:
        return f"LangChain Gemini API Error: {str(e)}"

def check_api_status(api_url: str = "http://localhost:8000"):
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        response.raise_for_status()
        return True, response.json()
    except:
        return False, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Search Client</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Web Search with LLM Integration</p>', unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API URL
        api_url = st.text_input("FastAPI Server URL", value="http://localhost:8000")
        
        # Check API status
        api_online, api_info = check_api_status(api_url)
        if api_online:
            st.markdown('<p class="status-success">✅ Search API: Online</p>', unsafe_allow_html=True)
            if api_info:
                st.json(api_info)
        else:
            st.markdown('<p class="status-error">❌ Search API: Offline</p>', unsafe_allow_html=True)
            st.error("Please start the FastAPI server")
        
        # Gemini API Key status
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            st.markdown('<p class="status-success">✅ Google API: Configured</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ Google API: Not Set</p>', unsafe_allow_html=True)
            st.error("Please set GOOGLE_API_KEY environment variable")
        
        st.markdown("---")
        
        # Model settings
        st.subheader("🤖 Gemini Settings")
        model_name = st.selectbox(
            "Model",
            ["gemini-2.0-flash","gemini-1.5-pro", "gemini-1.5-flash"],
            index=0
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        
        num_results = st.slider("Search Results", 3, 15, 5, 1)
        
    search_query = st.text_input("Enter your search query",
        placeholder="e.g., What are the latest developments in AI?"
    )
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Quick suggestions
    st.subheader("💡 Quick Questions")
    suggestions = [
        "Latest news in technology",
        "Current weather in Bengaluru", 
        "Recent AI breakthroughs 2025",
    ]
    
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                search_query = suggestion
                search_button = True
    
    # Process search
    if search_button and search_query:
        if not api_online:
            st.error("❌ Search API is not available. Please start the FastAPI server.")
            return
        
        if not google_api_key:
            st.error("❌ Google API key is not configured. Please set GOOGLE_API_KEY environment variable.")
            return
        
        # Show search progress
        with st.spinner("🔄 Searching the web..."):
            search_results = search_web(search_query, num_results, api_url)
        
        if search_results and not search_results.startswith("Error"):
            # Show search results in expandable section
            with st.expander("🌐 Raw Search Results", expanded=False):
                st.text(search_results)
            
            # Generate AI response
            with st.spinner("🤖 Generating Gemini response..."):
                ai_response = get_gemini_response(search_query, search_results, model_name, temperature)
            
            # Display AI response
            st.subheader(f"Answer for: {search_query}")
            st.markdown(ai_response)
            
            st.success("✅ Search and analysis completed!")
            
        else:
            st.error(f"❌ Search failed: {search_results}")
    
    elif search_button and not search_query:
        st.warning("⚠️ Please enter a search query")
    
if __name__ == "__main__":
    main()