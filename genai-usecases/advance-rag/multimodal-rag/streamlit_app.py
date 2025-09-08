import streamlit as st
import os
import tempfile
from pathlib import Path

# Import the RAG system
from app import MultimodalRAGSystem
from dotenv import load_dotenv
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🖼️",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'database_built' not in st.session_state:
        st.session_state.database_built = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory."""
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    return saved_paths

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.title("Multimodal RAG System")
    st.markdown("Use Google Gemini’s LLM API to upload documents and images, then ask questions")
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        # Model Selection
        st.markdown("**🤖 Model Selection**")
        main_model = st.selectbox(
            "Main LLM Model",
            ["gemini-2.0-flash", "gemini-2.5-pro","gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
            help="Model for text generation and reasoning"
        )
        
        vision_model = st.selectbox(
            "Vision Model", 
            ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="Model for image processing and analysis"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["models/text-embedding-004", "models/text-embedding-003"],
            index=0,
            help="Model for creating vector embeddings"
        )
        
        if st.button("🔄 Initialize System", type="primary"):
            try:
                with st.spinner("Initializing..."):
                    rag_system = MultimodalRAGSystem(
                        google_api_key=GOOGLE_API_KEY,
                        persist_directory="./chroma_db",
                        main_model=main_model,
                        vision_model=vision_model,
                        embedding_model=embedding_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    st.session_state.rag_system = rag_system
                st.success("✅ System initialized!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Model Parameters
        st.markdown("**⚙️ Model Parameters**")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, help="Creativity level (0=focused, 1=creative)")
        max_tokens = st.slider("Max Tokens", 1024, 8192, 4096, 256, help="Maximum response length")
        
        # Processing Settings
        st.markdown("**📝 Processing Settings**")
        chunk_size = st.slider("Chunk Size", 500, 3000, 1500, help="Size of text chunks for processing")
        retrieval_k = st.slider("Retrieval Count", 3, 15, 6, help="Number of relevant chunks to retrieve")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📁 Setup", "💬 Query", "📜 History"])
    
    with tab1:
        st.subheader("Setup Your RAG System")
        
        uploaded_files = st.file_uploader(
            "Upload documents and images (PDF, TXT, PNG, JPG, JPEG)",
            type=['pdf', 'txt', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Mix documents and images - the system will automatically process each type appropriately"
        )
        
        if uploaded_files:
            # Categorize files
            docs = [f for f in uploaded_files if f.type.startswith('application/') or f.type.startswith('text/')]
            images = [f for f in uploaded_files if f.type.startswith('image/')]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Documents", len(docs))
            with col2:
                st.metric("🖼️ Images", len(images))
            with col3:
                st.metric("📁 Total Files", len(uploaded_files))
            
            # Show file details
            with st.expander("📋 File Details"):
                for file in uploaded_files:
                    file_type = "📄" if file.type.startswith(('application/', 'text/')) else "🖼️"
                    st.write(f"{file_type} {file.name} ({file.size} bytes)")
        
        uploaded_docs = [f for f in uploaded_files if f.type.startswith('application/') or f.type.startswith('text/')] if uploaded_files else []
        uploaded_images = [f for f in uploaded_files if f.type.startswith('image/')] if uploaded_files else []
        
        if st.button("🔨 Build Database", type="primary"):
            if not st.session_state.rag_system:
                st.error("Please initialize system first")
            elif not uploaded_files:
                st.error("Please upload files first")
            else:
                try:
                    with st.spinner("Building database..."):
                        # Save files
                        doc_paths = save_uploaded_files(uploaded_docs) if uploaded_docs else []
                        image_paths = save_uploaded_files(uploaded_images) if uploaded_images else []
                        
                        # Build database
                        st.session_state.rag_system.build_enhanced_vector_database(
                            doc_paths, image_paths
                        )
                        st.session_state.database_built = True
                    st.success("✅ Database built!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Ask Questions")
        
        if not st.session_state.rag_system:
            st.warning("Please initialize the system first")
        elif not st.session_state.database_built:
            st.warning("Please build the database first")
        else:
            # Sample questions
            with st.expander("💡 Sample Questions"):
                sample_questions = [
                    "What are the key findings from the documents?",
                    "Describe the content of the uploaded images",
                    "What data is shown in any charts or tables?",
                    "Summarize the main points across all materials"
                ]
                for q in sample_questions:
                    st.write(f"• {q}")
            
            # Query input
            query = st.text_area("Your Question:", placeholder="Ask about your documents and images...")
            
            if st.button("🚀 Ask Question", type="primary"):
                if not query.strip():
                    st.warning("Please enter a question")
                else:
                    try:
                        with st.spinner("Searching..."):
                            result = st.session_state.rag_system.query(query, k=retrieval_k)
                        
                        # Store in history
                        st.session_state.query_history.append({
                            'query': query,
                            'response': result['response'],
                            'sources': result.get('sources', []),
                            'summary': result.get('multimodal_summary', {})
                        })
                        
                        # Display results
                        st.markdown("### 🤖 Response")
                        st.write(result['response'])
                        
                        # Show metrics
                        if 'multimodal_summary' in result:
                            summary = result['multimodal_summary']
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("📄 Text Sources", summary.get('text_sources', 0))
                            with col2:
                                st.metric("📊 Table Sources", summary.get('table_sources', 0))
                            with col3:
                                st.metric("🖼️ Image Sources", summary.get('image_sources', 0))
                        
                        # Show sources
                        if result.get('sources'):
                            with st.expander("📚 Sources Used"):
                                for i, source in enumerate(result['sources'], 1):
                                    st.write(f"**{i}. {source.get('source', 'Unknown')}** ({source.get('type', 'unknown')})")
                                    if source.get('preview'):
                                        st.write(source['preview'])
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("Query History")
        
        if st.session_state.query_history:
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()
            
            for i, item in enumerate(reversed(st.session_state.query_history), 1):
                with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {item['query'][:50]}..."):
                    st.markdown(f"**Question:** {item['query']}")
                    st.markdown(f"**Response:** {item['response']}")
                    
                    if item.get('summary'):
                        summary = item['summary']
                        st.markdown(f"**Sources:** {summary.get('text_sources', 0)} text, "
                                  f"{summary.get('table_sources', 0)} tables, "
                                  f"{summary.get('image_sources', 0)} images")
        else:
            st.info("No queries yet. Ask some questions in the Query tab!")
    
    # Status bar
    status_text = []
    if st.session_state.rag_system:
        status_text.append("✅ System Ready")
    else:
        status_text.append("❌ System Not Initialized")
    
    if st.session_state.database_built:
        status_text.append("✅ Database Built")
    else:
        status_text.append("❌ Database Not Built")
    
    status_text.append(f"📜 {len(st.session_state.query_history)} Queries")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:**")
    for status in status_text:
        st.sidebar.markdown(status)
    
    # Current Configuration Display
    if st.session_state.rag_system:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Current Config:**")
        st.sidebar.markdown(f"🤖 Main: {main_model}")
        st.sidebar.markdown(f"👁️ Vision: {vision_model}")  
        st.sidebar.markdown(f"📊 Embedding: {embedding_model}")
        st.sidebar.markdown(f"🌡️ Temperature: {temperature}")
        st.sidebar.markdown(f"📝 Max Tokens: {max_tokens:,}")

if __name__ == "__main__":
    main()