import streamlit as st
import tempfile
import os
from typing import List
import pandas as pd

from graph_rag import GraphRAGSystem, GraphRAGConfig, RetrieverType

st.set_page_config(
    page_title="Graph RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def save_uploaded_files(uploaded_files) -> List[str]:
    file_paths = []
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        file_paths.append(file_path)
    
    return file_paths

def display_retriever_comparison(rag: GraphRAGSystem, question: str):
    st.subheader("🔍 Retriever Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌐 Traversal Retriever**")
        st.markdown("*Explores graph relationships*")
        try:
            with st.spinner("Processing with Traversal Retriever..."):
                answer_traversal = rag.query(question, RetrieverType.TRAVERSAL)
            st.markdown(f"**Answer:** {answer_traversal}")
        except Exception as e:
            st.error(f"Error with traversal retriever: {e}")
    
    with col2:
        st.markdown("**📊 Standard Retriever**")
        st.markdown("*Direct vector search*")
        try:
            with st.spinner("Processing with Standard Retriever..."):
                answer_standard = rag.query(question, RetrieverType.STANDARD)
            st.markdown(f"**Answer:** {answer_standard}")
        except Exception as e:
            st.error(f"Error with standard retriever: {e}")
    
    with col3:
        st.markdown("**🤖 Smart Router**")
        st.markdown("*Automatic selection*")
        try:
            with st.spinner("Processing with Smart Router..."):
                answer_smart = rag.query(question, RetrieverType.HYBRID)
            st.markdown(f"**Answer:** {answer_smart}")
        except Exception as e:
            st.error(f"Error with smart router: {e}")

def main():
    initialize_session_state()
    
    st.title("Graph RAG System")
    st.markdown("Advanced Retrieval-Augmented Generation with Graph-based Document Retrieval")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Configuration
        st.subheader("🤖 Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ]
        )
        
        llm_model = st.selectbox(
            "LLM Model",
            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", ]
        )
        
        llm_provider = st.selectbox(
            "LLM Provider",
            ["google_genai"]
        )
        
        # Retrieval Configuration
        st.subheader("🔍 Retrieval Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        k_retrieval = st.slider("Number of Retrieved Documents", 1, 10, 5)
        max_depth = st.slider("Max Graph Traversal Depth", 0, 5, 2)
        
        config = GraphRAGConfig(
            embedding_model=embedding_model,
            llm_model=llm_model,
            llm_provider=llm_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k_retrieval=k_retrieval,
            max_depth=max_depth
        )
    
    # Main Content Area
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Chat Interface", "📊 Analytics"])
    
    with tab1:
        st.subheader("📄 Document Management")
        
        # Data Source Selection
        data_source = st.radio(
            "Choose Data Source:",
            ["Default Animal Dataset", "Upload Custom Documents", "Sample Text Input"]
        )
        
        if data_source == "Default Animal Dataset":
            if st.button("Load Default Dataset", type="primary"):
                try:
                    with st.spinner("Initializing with default animal dataset..."):
                        st.session_state.rag = GraphRAGSystem(config)
                        st.session_state.rag.initialize_with_default_data()
                        
                    st.session_state.documents_loaded = True
                    st.success("✅ Default dataset loaded successfully!")
                    st.info("The system is now ready for querying with animal-related data.")
                    
                    # Show detected relationships
                    relationships = st.session_state.rag.get_detected_relationships()
                    if relationships:
                        st.info(f"🔗 **Detected Relationships**: {', '.join(relationships)}")
                        
                except Exception as e:
                    st.error(f"❌ Error loading default dataset: {e}")
        
        elif data_source == "Upload Custom Documents":
            st.subheader("📁 Upload Your Documents")
            
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=['pdf', 'txt', 'csv', 'doc', 'docx'],
                accept_multiple_files=True,
                help="Supported formats: PDF, TXT, CSV, DOC, DOCX"
            )
            
            if uploaded_files:
                st.write(f"📋 **{len(uploaded_files)} file(s) selected:**")
                
                # Display file information
                file_info = []
                for file in uploaded_files:
                    file_info.append({
                        "File Name": file.name,
                        "File Size": f"{file.size / 1024:.2f} KB",
                        "File Type": file.type
                    })
                
                df = pd.DataFrame(file_info)
                st.dataframe(df, use_container_width=True)
                
                if st.button("Process Uploaded Files", type="primary"):
                    try:
                        with st.spinner("Processing uploaded files..."):
                            file_paths = save_uploaded_files(uploaded_files)
                            st.session_state.rag = GraphRAGSystem(config)
                            st.session_state.rag.initialize_with_files(file_paths)
                            
                        st.session_state.documents_loaded = True
                        st.success("✅ Files processed and indexed successfully!")
                        st.info("The system is now ready for querying with your uploaded documents.")
                        
                        # Show detected relationships
                        relationships = st.session_state.rag.get_detected_relationships()
                        if relationships:
                            st.info(f"🔗 **Auto-detected Relationships**: {', '.join(relationships)}")
                        else:
                            st.warning("⚠️ No specific relationships detected. Using basic content similarity.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing files: {e}")
        
        elif data_source == "Sample Text Input":
            st.subheader("✏️ Enter Sample Text")
            
            sample_text = st.text_area(
                "Enter your text content:",
                height=200,
                placeholder="Paste or type your text here..."
            )
            
            if sample_text and st.button("Process Text", type="primary"):
                try:
                    with st.spinner("Processing text input..."):
                        st.session_state.rag = GraphRAGSystem(config)
                        st.session_state.rag.initialize_with_text(sample_text)
                        
                    st.session_state.documents_loaded = True
                    st.success("✅ Text processed and indexed successfully!")
                    st.info("The system is now ready for querying with your text content.")
                    
                    # Show detected relationships
                    relationships = st.session_state.rag.get_detected_relationships()
                    if relationships:
                        st.info(f"🔗 **Auto-detected Relationships**: {', '.join(relationships)}")
                    else:
                        st.info("🔗 Using content-based similarity for relationships.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing text: {e}")
    
    with tab2:
        st.subheader("💬 Chat with Your Data")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please load documents first in the Document Upload tab.")
            return
        
        # Query modes
        query_mode = st.radio(
            "Select Query Mode:",
            ["Single Answer", "Retriever Comparison", "Interactive Chat"]
        )
        
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            key="main_question"
        )
        
        if question:
            if query_mode == "Single Answer":
                retriever_choice = st.selectbox(
                    "Choose Retriever:",
                    ["Smart Router (Recommended)", "Traversal Retriever", "Standard Retriever"]
                )
                
                retriever_map = {
                    "Smart Router (Recommended)": RetrieverType.HYBRID,
                    "Traversal Retriever": RetrieverType.TRAVERSAL,
                    "Standard Retriever": RetrieverType.STANDARD
                }
                
                if st.button("Get Answer", type="primary"):
                    try:
                        with st.spinner("Generating answer..."):
                            # Add progress indicators
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Initializing query...")
                            progress_bar.progress(25)
                            
                            answer = st.session_state.rag.query(
                                question, 
                                retriever_map[retriever_choice]
                            )
                            
                            progress_bar.progress(100)
                            status_text.text("Complete!")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.markdown("### 📝 Answer")
                        st.markdown(answer)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "retriever": retriever_choice
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {e}")
                        st.error(f"Details: {str(e)}")
            
            elif query_mode == "Retriever Comparison":
                if st.button("Compare Retrievers", type="primary"):
                    display_retriever_comparison(st.session_state.rag, question)
            
            elif query_mode == "Interactive Chat":
                if st.button("Add to Chat", type="primary"):
                    try:
                        with st.spinner("Generating response..."):
                            answer = st.session_state.rag.query(question, RetrieverType.HYBRID)
                        
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "retriever": "Smart Router"
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error in chat: {e}")
        
        # Chat History
        if st.session_state.chat_history:
            st.subheader("📜 Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"💭 Q{len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Retriever:** {chat['retriever']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab3:
        st.subheader("📊 System Analytics")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please load documents first to view analytics.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Usage Statistics")
            st.metric("Total Queries", len(st.session_state.chat_history))
            
            if st.session_state.chat_history:
                retriever_usage = {}
                for chat in st.session_state.chat_history:
                    retriever = chat.get('retriever', 'Unknown')
                    retriever_usage[retriever] = retriever_usage.get(retriever, 0) + 1
                
                st.subheader("🔍 Retriever Usage")
                for retriever, count in retriever_usage.items():
                    st.metric(retriever, count)
        
        with col2:
            st.subheader("⚙️ Current Configuration")
            st.json({
                "Embedding Model": config.embedding_model,
                "LLM Model": config.llm_model,
                "LLM Provider": config.llm_provider,
                "Chunk Size": config.chunk_size,
                "Chunk Overlap": config.chunk_overlap,
                "K Retrieval": config.k_retrieval,
                "Max Depth": config.max_depth
            })
    
if __name__ == "__main__":
    main()