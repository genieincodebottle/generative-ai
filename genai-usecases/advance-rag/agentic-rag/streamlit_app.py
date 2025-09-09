"""
Streamlit App for Latest Agentic RAG System

This app provides an interactive interface for the state-of-the-art agentic RAG system
featuring multi-agent workflows, advanced reasoning, and web-augmented retrieval.
"""

import streamlit as st
import time
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
# Import our agentic RAG system
from agentic_rag_system import AgenticRAGSystem, AgenticRAGConfig
from dotenv import load_dotenv
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.error("❌ Google API Key is missing. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

google_api_key = os.environ["GOOGLE_API_KEY"]
tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .agent-card {
        background-color: #000000;
        color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #000000 0%, #000000 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .execution-step {
        background-color: #000000;
        color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #17a2b8;
    }
    
    .source-item {
        background-color: #000000;
        color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = None

def initialize_rag_system_sync(config, google_api_key, tavily_api_key):
    """Initialize RAG system in a synchronous context with proper event loop handling"""
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize the system
            rag_system = AgenticRAGSystem(
                config=config,
                google_api_key=google_api_key,
                tavily_api_key=tavily_api_key
            )
            return rag_system
        except Exception as e:
            raise e
        finally:
            loop.close()
    
    # Run initialization in a separate thread with its own event loop
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()

def create_system_status_dashboard():
    """Create system status dashboard"""
    if not st.session_state.system_initialized:
        st.warning("🔧 System not initialized. Please configure and initialize the system first.")
        return
    
    status = st.session_state.rag_system.get_system_status()
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Vector Store</h3>
            <h2>{'✅' if status['vector_store_initialized'] else '❌'}</h2>
            <p>{'Ready' if status['vector_store_initialized'] else 'Not Ready'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Web Search</h3>
            <h2>{'🌐' if status['web_search_enabled'] else '🚫'}</h2>
            <p>{'Enabled' if status['web_search_enabled'] else 'Disabled'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Tools</h3>
            <h2>{status['tools_available']}</h2>
            <p>Available Tools</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model</h3>
            <h2>🤖</h2>
            <p>{status['model']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add informational message if vector store is not ready
    if not status['vector_store_initialized']:
        st.info("📄 **Vector Store Status**: No documents loaded yet. Upload and process documents to enable document-based retrieval. The system can still answer questions using web search if enabled.")

def display_agent_workflow():
    """Display the agent workflow visualization"""
    st.subheader("🔄 Agent Workflow")
    
    agents = [
        {"name": "Planner", "description": "Analyzes query and creates execution plan", "icon": "🎯"},
        {"name": "Retriever", "description": "Retrieves relevant documents from vector store", "icon": "📚"},
        {"name": "Researcher", "description": "Performs web search for current information", "icon": "🔍"},
        {"name": "Synthesizer", "description": "Combines information to generate answer", "icon": "⚡"},
        {"name": "Validator", "description": "Validates and refines the final answer", "icon": "✅"}
    ]
    
    cols = st.columns(5)
    for i, agent in enumerate(agents):
        with cols[i]:
            st.markdown(f"""
            <div class="agent-card">
                <h6>{agent['icon']} {agent['name']}</h6>
                <p>{agent['description']}</p>
            </div>
            """, unsafe_allow_html=True)

def display_query_analysis(response_data: Dict[str, Any]):
    """Display detailed query analysis"""
    if 'query_plan' not in response_data or not response_data['query_plan']:
        return
    
    plan = response_data['query_plan']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Query Complexity:**")
        complexity = plan['complexity']
        color_map = {
            'simple': 'green',
            'moderate': 'orange', 
            'complex': 'red',
            'research': 'purple'
        }
        st.markdown(f"<span style='color:{color_map.get(complexity, 'gray')}'>{complexity.upper()}</span>", unsafe_allow_html=True)
        
        st.markdown("**Sub-queries Generated:**")
        for i, sub_q in enumerate(plan.get('sub_queries', []), 1):
            st.write(f"{i}. {sub_q}")
    
    with col2:
        st.markdown("**Execution Steps:**")
        st.write(f"Estimated: {plan.get('estimated_steps', 1)} steps")
        
        if response_data.get('execution_log'):
            st.markdown("**Execution Log:**")
            for step in response_data['execution_log']:
                st.markdown(f"""
                <div class="execution-step">
                    {step}
                </div>
                """, unsafe_allow_html=True)

def display_sources(sources: List[Dict[str, Any]]):
    """Display sources with improved formatting"""
    if not sources:
        return
    
    doc_sources = [s for s in sources if s.get('type') == 'document']
    web_sources = [s for s in sources if s.get('type') == 'web']
    
    if doc_sources:
        st.markdown("**📄 Document Sources:**")
        for source in doc_sources:
            st.markdown(f"""
            <div class="source-item">
                <strong>📄 {source.get('source', 'Unknown')}</strong><br>
                <small>{source.get('content_preview', 'No preview available')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    if web_sources:
        st.markdown("**🌐 Web Sources:**")
        for source in web_sources:
            title = source.get('title', 'Unknown Title')
            url = source.get('source', '#')
            preview = source.get('content_preview', 'No preview available')
            
            st.markdown(f"""
            <div class="source-item">
                <strong>🌐 <a href="{url}" target="_blank">{title}</a></strong><br>
                <small>{preview}</small>
            </div>
            """, unsafe_allow_html=True)

def create_confidence_gauge(confidence: float):
    """Create a confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.header("🤖 Agentic RAG System")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        # System Configuration
        st.subheader("🛠️ System Settings")
        
        llm_model = st.selectbox(
            "LLM Model",
            ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.5-pro", "gemini-2.5-flash"],
            help="Choose the Gemini model for reasoning"
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Max Tokens", 1000, 16000, 8192, 1000)
        k_retrieval = st.slider("Retrieval K", 1, 20, 8, 1)
        
        enable_web_search = st.checkbox("Enable Web Search", value=True)
        
        # Initialize System
        if st.button("🚀 Initialize System"):
            
            with st.spinner("Initializing Agentic RAG System..."):
                try:
                    config = AgenticRAGConfig(
                        llm_model=llm_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        k_retrieval=k_retrieval,
                        enable_web_search=enable_web_search
                    )
                    
                    st.session_state.rag_system = initialize_rag_system_sync(
                        config=config,
                        google_api_key=google_api_key,
                        tavily_api_key=tavily_api_key if tavily_api_key else None
                    )
                    
                    st.session_state.system_initialized = True
                    st.success("✅ System initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
    
    # Create tabs for different sections (always shown)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Document Management", 
        "💬 Ask Questions", 
        "🧠 Query Analysis", 
        "📋 Sources & History",
        "🔄 System Overview"
    ])
    
    # Check if system is initialized for conditional content
    system_ready = st.session_state.system_initialized
    
    with tab1:
        st.subheader("📁 Document Management")
        
        if not system_ready:
            st.warning("🚫 System not initialized. Please initialize the system from the sidebar first.")
            st.info("📋 This tab will allow you to:")
            st.write("• Upload PDF, TXT, and CSV documents")
            st.write("• Process documents into the vector store")
            st.write("• View vector store status")
        else:
            uploaded_files = st.file_uploader(
                "Upload documents to add to knowledge base",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'csv']
            )
            
            if uploaded_files and st.button("📚 Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        # Save uploaded files temporarily
                        temp_dir = Path("temp_uploads")
                        temp_dir.mkdir(exist_ok=True)
                        
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = temp_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(str(file_path))
                        
                        # Load documents into the system
                        success = st.session_state.rag_system.load_documents(file_paths)
                        
                        # Cleanup
                        for file_path in file_paths:
                            os.unlink(file_path)
                        
                        if success:
                            st.success(f"✅ Successfully processed {len(uploaded_files)} documents!")
                        else:
                            st.error("❌ Failed to process documents.")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
            
            # Show current vector store status
            status = st.session_state.rag_system.get_system_status()
            if status['vector_store_initialized']:
                st.success("✅ Vector store is ready with documents")
            else:
                st.info("📄 Upload documents above to create the vector store")
        
    with tab2:
        st.subheader("💬 Ask a Question")
        
        if not system_ready:
            st.warning("🚫 System not initialized. Please initialize the system from the sidebar first.")
            st.info("📋 This tab will allow you to:")
            st.write("• Ask questions using natural language")
            st.write("• Get answers from documents and web search")
            st.write("• Start new conversation threads")
            st.write("• View confidence scores and metrics")
        else:
            # Query input
            user_query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="Ask anything! The agentic system will plan the best approach to answer your question."
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                ask_button = st.button("🤖 Ask Question", type="primary")
            
            with col2:
                new_conversation = st.button("🆕 New Conversation")
            
            if new_conversation:
                st.session_state.current_thread_id = None
                st.success("Started new conversation thread!")
            
            # Process query
            if ask_button and user_query.strip():
                with st.spinner("🧠 Agentic system is thinking..."):
                    try:
                        start_time = time.time()
                        
                        # Query the system
                        response = st.session_state.rag_system.query(
                            user_query,
                            thread_id=st.session_state.current_thread_id
                        )
                        
                        end_time = time.time()
                        
                        # Update session state
                        st.session_state.current_thread_id = response.get('thread_id')
                        st.session_state.query_history.append({
                            'query': user_query,
                            'response': response,
                            'timestamp': datetime.now(),
                            'processing_time': end_time - start_time
                        })
                        
                        # Display results
                        st.success("✅ Question processed successfully!")
                        
                        # Main answer
                        st.subheader("💡 Answer")
                        st.write(response['answer'])
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Processing Time", f"{end_time - start_time:.2f}s")
                        
                        with col2:
                            st.metric("Confidence", f"{response.get('confidence', 0):.2%}")
                        
                        with col3:
                            st.metric("Documents Used", response.get('retrieved_documents', 0))
                        
                        with col4:
                            st.metric("Web Results", response.get('web_results', 0))
                        
                        # Confidence gauge
                        if response.get('confidence'):
                            fig = create_confidence_gauge(response['confidence'])
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
        
    with tab3:
        st.subheader("🧠 Query Analysis")
        
        if not system_ready:
            st.warning("🚫 System not initialized. Please initialize the system from the sidebar first.")
            st.info("📋 This tab will show:")
            st.write("• Query complexity analysis")
            st.write("• Sub-questions generated by the planner")
            st.write("• Execution steps and logs")
            st.write("• Processing strategy details")
        else:
            if st.session_state.query_history:
                # Show analysis for the latest query
                latest_response = st.session_state.query_history[-1]['response']
                
                st.write(f"**Latest Query:** {st.session_state.query_history[-1]['query']}")
                
                # Display detailed analysis
                display_query_analysis(latest_response)
                
            else:
                st.info("📝 Ask a question first to see detailed query analysis here.")
        
    with tab4:
        if not system_ready:
            st.warning("🚫 System not initialized. Please initialize the system from the sidebar first.")
            st.info("📋 This tab will show:")
            st.write("• Document sources used in answers")
            st.write("• Web search results and links")
            st.write("• Complete query history with metrics")
            st.write("• Source previews and citations")
        else:
            # Sources from latest query
            if st.session_state.query_history:
                st.write("**Sources from Latest Query:**")
                latest_response = st.session_state.query_history[-1]['response']
                display_sources(latest_response.get('sources', []))
                
                st.markdown("---")
            
            # Query History
            if st.session_state.query_history:
                st.subheader("📈 Query History")
                
                # Create history dataframe
                history_data = []
                for item in st.session_state.query_history:
                    history_data.append({
                        'Timestamp': item['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                        'Query': item['query'][:100] + "..." if len(item['query']) > 100 else item['query'],
                        'Confidence': f"{item['response'].get('confidence', 0):.2%}",
                        'Processing Time': f"{item['processing_time']:.2f}s",
                        'Sources': len(item['response'].get('sources', []))
                    })
                
                df = pd.DataFrame(history_data)
                st.dataframe(df, width="stretch")
                
                # Clear history button
                if st.button("🗑️ Clear History"):
                    st.session_state.query_history = []
                    st.rerun()
            else:
                st.info("📝 No queries yet. Ask questions to build your query history.")
        
    with tab5:
        
        if not system_ready:
            st.warning("🚫 System not initialized. Please initialize the system from the sidebar first.")
            st.info("📋 This tab will show:")
            st.write("• Agent workflow visualization")
            st.write("• System configuration details")
            st.write("• Current settings and parameters")
            st.write("• Architecture overview")
        else:
            # Agent Workflow Visualization
            display_agent_workflow()
            
            st.markdown("---")
            
            # System Configuration
            st.subheader("⚙️ Current Configuration")
            
            config_data = {
                "Setting": [
                    "LLM Model",
                    "Temperature", 
                    "Max Tokens",
                    "Retrieval K",
                    "Web Search",
                    "Chunk Size",
                    "Chunk Overlap"
                ],
                "Value": [
                    str(st.session_state.rag_system.config.llm_model),
                    str(st.session_state.rag_system.config.temperature),
                    str(st.session_state.rag_system.config.max_tokens),
                    str(st.session_state.rag_system.config.k_retrieval),
                    "Enabled" if st.session_state.rag_system.config.enable_web_search else "Disabled",
                    str(st.session_state.rag_system.config.chunk_size),
                    str(st.session_state.rag_system.config.chunk_overlap)
                ]
            }
            
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, width="stretch")

if __name__ == "__main__":
    main()