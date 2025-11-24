"""
Streamlit UI for RAG Code Search System
========================================
A web interface for indexing code repositories and searching for code examples.

Features:
- Index code from multiple sources (strings, files, directories, GitHub)
- Search and retrieve relevant code examples
- View system statistics
- Filter by language and repository
"""

import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the RAG implementation
from rag import RAGCodeSearchSystem, Config

# Page configuration
st.set_page_config(
    page_title="RAG Code Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .code-snippet {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        overflow-x: auto;
    }
    .stat-card {
        background: linear-gradient(135deg, #66c0ea 0%, #86c0c4 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    # Initialize advanced settings defaults
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 500
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "models/embedding-001"
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = "gemini-2.0-flash"

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

    if 'index_history' not in st.session_state:
        st.session_state.index_history = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

init_session_state()

# Function to initialize or reinitialize RAG system with current settings
def get_or_create_rag_system():
    """Get existing RAG system or create new one with current settings"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    # Check if we need to reinitialize (model changed)
    if st.session_state.rag_system is None or \
       st.session_state.rag_system.config.llm_model != st.session_state.llm_model or \
       st.session_state.rag_system.config.embedding_model != st.session_state.embedding_model:
        try:
            config = Config()
            config.llm_model = st.session_state.llm_model
            config.embedding_model = st.session_state.embedding_model
            config.chunk_size = st.session_state.chunk_size
            config.top_k_final = st.session_state.top_k
            st.session_state.rag_system = RAGCodeSearchSystem(config)
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            return None

    return st.session_state.rag_system

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Check API key status
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        st.success("✅ API Key loaded from .env")
    else:
        st.error("❌ API Key not found in .env file")
        st.info("Please add GOOGLE_API_KEY to your .env file")

    st.divider()

    # Advanced Settings
    st.markdown("### 🔧 Advanced Settings")
    chunk_size = st.number_input(
        "Chunk Size",
        value=st.session_state.chunk_size,
        min_value=100,
        max_value=2000,
        step=50,
        help="Size of code chunks for embedding"
    )
    top_k = st.number_input(
        "Top K Results",
        value=st.session_state.top_k,
        min_value=1,
        max_value=5,
        help="Number of results to return"
    )
    embedding_model = st.selectbox(
        "Embedding Model",
        ["models/gemini-embedding-001"],
        index=0,
        help="Model for generating embeddings"
    )
    llm_model = st.selectbox(
        "LLM Model",
        ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-pro-preview","gemini-2.5-flash"],
        index=0,
        help="Model for generating responses"
    )

    # Update session state if changed
    if chunk_size != st.session_state.chunk_size:
        st.session_state.chunk_size = chunk_size
    if top_k != st.session_state.top_k:
        st.session_state.top_k = top_k
    if embedding_model != st.session_state.embedding_model:
        st.session_state.embedding_model = embedding_model
    if llm_model != st.session_state.llm_model:
            st.session_state.llm_model = llm_model

    st.divider()

    # System Status
    st.markdown("### 📊 System Status")

    # Initialize system with current settings
    rag_system = get_or_create_rag_system()

    if rag_system:
        try:
            stats = rag_system.get_stats()
            st.metric("Total Documents", stats['total_documents'])
            st.caption(f"🤖 Model: {st.session_state.llm_model}")
            st.caption(f"📊 Top K: {st.session_state.top_k}")
        except Exception as e:
            st.error(f"Error fetching stats: {str(e)}")
    else:
        st.warning("System not initialized. Check API key in .env file.")

    

    
# Main content
st.markdown('<h2 class="main-header">🔍 RAG Code Search System</h2>', unsafe_allow_html=True)
st.markdown("**Index code repositories and search for relevant code examples using AI-powered semantic search**")

# Get RAG system with current settings
rag_system = get_or_create_rag_system()

# Check if system is initialized
if not rag_system:
    st.markdown('<div class="error-box">❌ System not initialized. Please check that your GOOGLE_API_KEY is set in the .env file.</div>', unsafe_allow_html=True)

    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Add API Key to .env**: Make sure your `.env` file contains `GOOGLE_API_KEY=your_key_here`
    2. **Restart the app**: Run `streamlit run app.py` again
    3. **Index Code**: Use the "Index Code" tab to add code to the vector database
    4. **Search**: Use the "Search Code" tab to find relevant code examples
    """)

    st.markdown("### 📚 Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Indexing Options:**
        - Direct code input
        - Upload files
        - Index local directories
        - Clone from GitHub
        - Bulk import from JSON
        """)
    with col2:
        st.markdown("""
        **Search Features:**
        - Semantic code search
        - Language filtering
        - Repository filtering
        - AI-powered explanations
        - Code examples with context
        """)

    st.stop()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📥 Index Code", "🔎 Search Code", "📈 Analytics"])

# ============================================================================
# TAB 1: INDEX CODE
# ============================================================================
with tab1:
    st.markdown('<p class="sub-header">Index Code to Vector Database</p>', unsafe_allow_html=True)
    st.markdown("Add code to the vector database for semantic search")

    # Indexing method selection
    index_method = st.radio(
        "Choose indexing method:",
        ["Direct Code Input", "Upload File", "Index Directory", "Clone from GitHub", "Import from JSON"],
        horizontal=True
    )

    st.divider()

    # Method 1: Direct Code Input
    if index_method == "Direct Code Input":
        st.markdown("### 📝 Enter Code Directly")

        col1, col2 = st.columns([2, 1])

        with col1:
            code_input = st.text_area(
                "Code",
                height=300,
                placeholder="Paste your code here...",
                help="Enter the code you want to index"
            )

        with col2:
            language = st.selectbox(
                "Language",
                ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"],
                help="Select the programming language"
            )

            repo_name = st.text_input(
                "Repository Name",
                value="inline",
                help="Name for organizing this code"
            )

            file_name = st.text_input(
                "File Name",
                value="example",
                help="Logical file name"
            )

        if st.button("🚀 Index Code", type="primary", key="index_direct"):
            if code_input:
                with st.spinner("Indexing code..."):
                    try:
                        print(f"\n[APP] Indexing code - Language: {language}, Repo: {repo_name}, File: {file_name}")
                        count = rag_system.index_code(
                            code_input,
                            language,
                            repo_name,
                            file_name
                        )
                        print(f"[APP] Indexing complete - {count} chunks indexed")

                        st.markdown(f'<div class="success-box">✅ Successfully indexed {count} code chunks!</div>', unsafe_allow_html=True)

                        # Add to history
                        st.session_state.index_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'method': 'Direct Input',
                            'details': f"{repo_name}/{file_name}",
                            'chunks': count
                        })

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter some code to index")

    # Method 2: Upload File
    elif index_method == "Upload File":
        st.markdown("### 📁 Upload Code Files")

        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'go', 'rs', 'cpp', 'c', 'h'],
            help="Upload one or more code files"
        )

        repo_name = st.text_input(
            "Repository Name",
            value="uploaded_files",
            help="Name for organizing these files"
        )

        if st.button("🚀 Index Files", type="primary", key="index_upload"):
            if uploaded_files:
                total_chunks = 0
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    try:
                        # Read file content
                        content = uploaded_file.read().decode('utf-8')

                        # Detect language from extension
                        ext = Path(uploaded_file.name).suffix.lower()
                        lang_map = {
                            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                            '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
                            '.go': 'go', '.rs': 'rust', '.cpp': 'cpp', '.c': 'c'
                        }
                        language = lang_map.get(ext, 'python')

                        # Index the code
                        count = rag_system.index_code(
                            content,
                            language,
                            repo_name,
                            uploaded_file.name
                        )

                        total_chunks += count

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.empty()
                progress_bar.empty()

                st.markdown(f'<div class="success-box">✅ Successfully indexed {total_chunks} code chunks from {len(uploaded_files)} files!</div>', unsafe_allow_html=True)

                # Add to history
                st.session_state.index_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'method': 'File Upload',
                    'details': f"{len(uploaded_files)} files to {repo_name}",
                    'chunks': total_chunks
                })
            else:
                st.warning("Please upload at least one file")

    # Method 3: Index Directory
    elif index_method == "Index Directory":
        st.markdown("### 📂 Index Local Directory")

        st.info("Enter the path to a local directory containing code files")

        dir_path = st.text_input(
            "Directory Path",
            placeholder="e.g., C:\\projects\\my-app or /home/user/projects/my-app",
            help="Absolute path to the directory"
        )

        repo_name = st.text_input(
            "Repository Name (optional)",
            help="Leave empty to use directory name"
        )

        if st.button("🚀 Index Directory", type="primary", key="index_dir"):
            if dir_path:
                if Path(dir_path).exists():
                    with st.spinner(f"Indexing directory: {dir_path}..."):
                        try:
                            count = rag_system.index_repository(
                                dir_path,
                                repo_name if repo_name else None
                            )

                            st.markdown(f'<div class="success-box">✅ Successfully indexed {count} code chunks from {dir_path}!</div>', unsafe_allow_html=True)

                            # Add to history
                            st.session_state.index_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'method': 'Directory',
                                'details': dir_path,
                                'chunks': count
                            })

                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Directory not found: {dir_path}")
            else:
                st.warning("Please enter a directory path")

    # Method 4: Clone from GitHub
    elif index_method == "Clone from GitHub":
        st.markdown("### 🌐 Clone and Index GitHub Repository")

        st.info("Enter a GitHub repository URL to clone and index")

        github_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/username/repository",
            help="Full URL to the GitHub repository"
        )

        temp_dir = st.text_input(
            "Temporary Directory",
            value="./temp_repo",
            help="Where to temporarily clone the repository"
        )

        if st.button("🚀 Clone and Index", type="primary", key="index_github"):
            if github_url:
                with st.spinner(f"Cloning and indexing {github_url}..."):
                    try:
                        import subprocess
                        import shutil

                        # Clone repository
                        if Path(temp_dir).exists():
                            shutil.rmtree(temp_dir)

                        st.info(f"Cloning repository to {temp_dir}...")
                        subprocess.run(
                            ["git", "clone", "--depth", "1", github_url, temp_dir],
                            check=True,
                            capture_output=True
                        )

                        # Index repository
                        st.info("Indexing code...")
                        count = rag_system.index_repository(temp_dir)

                        # Cleanup
                        shutil.rmtree(temp_dir)

                        st.markdown(f'<div class="success-box">✅ Successfully indexed {count} code chunks from {github_url}!</div>', unsafe_allow_html=True)

                        # Add to history
                        st.session_state.index_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'method': 'GitHub',
                            'details': github_url,
                            'chunks': count
                        })

                    except subprocess.CalledProcessError as e:
                        st.error(f"Git clone failed: {e.stderr.decode()}")
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a GitHub URL")

    # Method 5: Import from JSON
    elif index_method == "Import from JSON":
        st.markdown("### 📋 Import from JSON File")

        st.info("Upload a JSON file with code samples in the expected format")

        with st.expander("📖 View JSON Format"):
            st.json({
                "examples": [
                    {
                        "code": "def hello(): print('Hello')",
                        "name": "hello",
                        "language": "python",
                        "repo": "my-repo",
                        "file_path": "src/hello.py",
                        "description": "A hello function"
                    }
                ]
            })

        json_file = st.file_uploader(
            "Upload JSON File",
            type=['json'],
            help="JSON file with code samples"
        )

        if st.button("🚀 Import from JSON", type="primary", key="index_json"):
            if json_file:
                try:
                    data = json.load(json_file)
                    samples = data if isinstance(data, list) else data.get('examples', [])

                    total_chunks = 0
                    progress_bar = st.progress(0)

                    for i, sample in enumerate(samples):
                        count = rag_system.index_code(
                            sample['code'],
                            sample.get('language', 'python'),
                            sample.get('repo', 'imported'),
                            sample.get('file_path', f'sample_{i}')
                        )
                        total_chunks += count
                        progress_bar.progress((i + 1) / len(samples))

                    progress_bar.empty()

                    st.markdown(f'<div class="success-box">✅ Successfully indexed {total_chunks} code chunks from {len(samples)} samples!</div>', unsafe_allow_html=True)

                    # Add to history
                    st.session_state.index_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'method': 'JSON Import',
                        'details': f"{len(samples)} samples",
                        'chunks': total_chunks
                    })

                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please upload a JSON file")

    # Show indexing history
    if st.session_state.index_history:
        st.divider()
        st.markdown("### 📜 Indexing History")

        for entry in reversed(st.session_state.index_history[-5:]):
            with st.expander(f"{entry['method']} - {entry['timestamp'][:19]}"):
                st.write(f"**Details:** {entry['details']}")
                st.write(f"**Chunks Indexed:** {entry['chunks']}")

# ============================================================================
# TAB 2: SEARCH CODE
# ============================================================================
with tab2:
    st.markdown('<p class="sub-header">Search for Code Examples</p>', unsafe_allow_html=True)
    st.markdown("Use natural language to find relevant code examples from the indexed repositories")

    # Search input
    query = st.text_area(
        "What are you looking for?",
        placeholder="e.g., How do I authenticate with OAuth2?\nShow me JWT token validation\nDatabase connection with pooling",
        height=100,
        help="Enter your search query in natural language"
    )

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_language = st.selectbox(
            "Filter by Language",
            ["Any", "python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"],
            help="Filter results by programming language"
        )

    with col2:
        filter_repo = st.text_input(
            "Filter by Repository",
            placeholder="e.g., my-repo",
            help="Filter results by repository name (optional)"
        )

    with col3:
        show_details = st.checkbox(
            "Show Details",
            value=True,
            help="Show detailed information about retrieved documents"
        )

    # Search button
    if st.button("🔍 Search", type="primary", key="search_btn"):
        if query:
            with st.spinner("Searching for relevant code..."):
                try:
                    print(f"\n[APP] Search button clicked")
                    print(f"[APP] Query: {query}")
                    print(f"[APP] Show details: {show_details}")

                    # Build filters
                    filters = {}
                    if filter_language != "Any":
                        filters['language'] = filter_language
                    if filter_repo:
                        filters['repo_name'] = filter_repo

                    print(f"[APP] Filters: {filters}")

                    # Perform search
                    if show_details:
                        print(f"[APP] Calling rag_system.search_with_details()...")
                        result = rag_system.search_with_details(
                            query,
                            filters if filters else None
                        )
                        print(f"[APP] Search completed successfully")

                        # Display response
                        st.markdown("### 💬 AI Response")
                        st.markdown(result['response'])

                        # Display retrieved documents
                        st.divider()
                        st.markdown(f"### 📚 Retrieved Code Examples ({result['num_results']})")

                        for i, doc in enumerate(result['documents'], 1):
                            with st.expander(f"📄 {doc['name']} - {doc['language']} ({doc['repo']})"):
                                st.markdown(f"**File:** `{doc['file']}`")
                                st.markdown(f"**Repository:** `{doc['repo']}`")
                                st.markdown(f"**Language:** `{doc['language']}`")
                                st.markdown("**Code:**")
                                st.code(doc['content'], language=doc['language'])

                    else:
                        print(f"[APP] Calling rag_system.search()...")
                        response = rag_system.search(
                            query,
                            filters if filters else None
                        )
                        print(f"[APP] Search completed successfully")

                        st.markdown("### 💬 AI Response")
                        st.markdown(response)

                    # Add to search history
                    st.session_state.search_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'query': query,
                        'filters': filters,
                        'results': result['num_results'] if show_details else 'N/A'
                    })

                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a search query")

    # Example queries
    st.divider()
    st.markdown("### 💡 Example Queries")

    example_queries = [
        "How do I authenticate with OAuth2?",
        "Show me JWT token validation",
        "How to implement API key authentication?",
        "Database connection with pooling",
        "Async function examples in JavaScript",
        "Error handling best practices"
    ]

    cols = st.columns(3)
    for i, example in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.example_query = example
                st.rerun()

    # Show search history
    if st.session_state.search_history:
        st.divider()
        st.markdown("### 🕐 Search History")

        for entry in reversed(st.session_state.search_history[-5:]):
            with st.expander(f"{entry['query'][:50]}... - {entry['timestamp'][:19]}"):
                st.write(f"**Query:** {entry['query']}")
                st.write(f"**Filters:** {entry['filters']}")
                st.write(f"**Results:** {entry['results']}")

# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================
with tab3:
    st.markdown('<p class="sub-header">System Analytics</p>', unsafe_allow_html=True)

    try:
        stats = rag_system.get_stats() if rag_system else {'total_documents': 0, 'collection_name': 'N/A', 'persist_directory': 'N/A'}

        # Key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h1>{stats['total_documents']}</h1>
                <p>Total Documents</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h1>{len(st.session_state.search_history)}</h1>
                <p>Total Searches</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <h1>{len(st.session_state.index_history)}</h1>
                <p>Indexing Operations</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Database info
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🗄️ Database Information")
            st.info(f"""
            **Collection Name:** {stats['collection_name']}
            **Storage Path:** {stats['persist_directory']}
            **Total Documents:** {stats['total_documents']}
            """)

        with col2:
            st.markdown("### 📊 Activity Summary")
            st.info(f"""
            **Total Searches:** {len(st.session_state.search_history)}
            **Total Indexing Ops:** {len(st.session_state.index_history)}
            **Session Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)

        # Recent activity
        st.divider()
        st.markdown("### 🔄 Recent Activity")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Recent Indexing**")
            if st.session_state.index_history:
                for entry in reversed(st.session_state.index_history[-3:]):
                    st.caption(f"✓ {entry['method']} - {entry['chunks']} chunks - {entry['timestamp'][:19]}")
            else:
                st.caption("No indexing activity yet")

        with col2:
            st.markdown("**Recent Searches**")
            if st.session_state.search_history:
                for entry in reversed(st.session_state.search_history[-3:]):
                    st.caption(f"🔍 {entry['query'][:40]}... - {entry['timestamp'][:19]}")
            else:
                st.caption("No search activity yet")

    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>RAG Code Search System | Powered by Google Gemini & ChromaDB</p>
</div>
""", unsafe_allow_html=True)
