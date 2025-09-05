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
    if 'latest_query_details' not in st.session_state:
        st.session_state.latest_query_details = None

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
                result_traversal = rag.query(question, RetrieverType.TRAVERSAL, return_details=True)
            
            st.markdown(f"**Answer:** {result_traversal['answer']}")
            st.markdown("---")
            st.markdown("📊 **Context Details:**")
            st.metric("Documents", result_traversal["retrieval_info"]["num_documents"])
            st.metric("Context Length", f"{result_traversal['retrieval_info']['context_length']} chars")
            
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(result_traversal["retrieved_documents"], 1):
                    st.markdown(f"**Doc {i}:** {doc['source']}")
                    st.text(doc['content_preview'])
                    
        except Exception as e:
            st.error(f"Error with traversal retriever: {e}")
    
    with col2:
        st.markdown("**📊 Standard Retriever**")
        st.markdown("*Direct vector search*")
        try:
            with st.spinner("Processing with Standard Retriever..."):
                result_standard = rag.query(question, RetrieverType.STANDARD, return_details=True)
            
            st.markdown(f"**Answer:** {result_standard['answer']}")
            st.markdown("---")
            st.markdown("📊 **Context Details:**")
            st.metric("Documents", result_standard["retrieval_info"]["num_documents"])
            st.metric("Context Length", f"{result_standard['retrieval_info']['context_length']} chars")
            
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(result_standard["retrieved_documents"], 1):
                    st.markdown(f"**Doc {i}:** {doc['source']}")
                    st.text(doc['content_preview'])
                    
        except Exception as e:
            st.error(f"Error with standard retriever: {e}")
    
    with col3:
        st.markdown("**🤖 Smart Router**")
        st.markdown("*Automatic selection*")
        try:
            with st.spinner("Processing with Smart Router..."):
                result_smart = rag.query(question, RetrieverType.HYBRID, return_details=True)
            
            st.markdown(f"**Answer:** {result_smart['answer']}")
            st.markdown("---")
            st.markdown("📊 **Context Details:**")
            st.metric("Documents", result_smart["retrieval_info"]["num_documents"])
            st.metric("Context Length", f"{result_smart['retrieval_info']['context_length']} chars")
            
            if "routing_info" in result_smart:
                st.markdown("🤖 **Router Decision:**")
                st.metric("Strategy", result_smart["routing_info"]["strategy"])
                st.metric("Confidence", f"{result_smart['routing_info']['confidence']:.2f}")
            
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(result_smart["retrieved_documents"], 1):
                    st.markdown(f"**Doc {i}:** {doc['source']}")
                    st.text(doc['content_preview'])
                    
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
                "Qwen/Qwen3-Embedding-0.6B",
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
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "💬 Chat Interface", "🔍 Retrieved Context Details", "📊 Analytics"])
    
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
                st.dataframe(df, width="stretch")
                
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
                            
                            # Get detailed query results
                            query_result = st.session_state.rag.query(
                                question, 
                                retriever_map[retriever_choice],
                                return_details=True
                            )
                            
                            progress_bar.progress(100)
                            status_text.text("Complete!")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display the answer
                        st.markdown("### 📝 Answer")
                        st.markdown(query_result["answer"])
                        
                        # Store query details in session state for the context details tab
                        st.session_state.latest_query_details = query_result
                        
                        # Show brief summary and link to context details
                        st.markdown("---")
                        st.info(f"📊 Retrieved {query_result['retrieval_info']['num_documents']} documents using {query_result['retrieval_info']['retriever_type']} retriever. **View full context details in the 'Retrieved Context Details' tab.**")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": query_result["answer"],
                            "retriever": retriever_choice,
                            "details": query_result
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
                            query_result = st.session_state.rag.query(question, RetrieverType.HYBRID, return_details=True)
                        
                        # Store latest query details
                        st.session_state.latest_query_details = query_result
                        
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": query_result["answer"],
                            "retriever": "Smart Router",
                            "details": query_result
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
                    
                    # Show context details if available
                    if "details" in chat:
                        st.markdown("---")
                        st.markdown("**📊 Retrieval Details:**")
                        details = chat["details"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents", details["retrieval_info"]["num_documents"])
                        with col2:
                            st.metric("Context Length", f"{details['retrieval_info']['context_length']} chars")
                        with col3:
                            st.metric("Strategy", details["retrieval_info"]["retriever_type"])
                        
                        if "routing_info" in details:
                            st.markdown(f"**🤖 Router Confidence:** {details['routing_info']['confidence']:.2f}")
                            st.markdown(f"**💭 Router Reasoning:** {details['routing_info']['reasoning']}")
                        
                        # Show document sources
                        sources = [doc['source'] for doc in details['retrieved_documents']]
                        st.markdown(f"**📄 Sources:** {', '.join(set(sources))}")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab3:
        st.subheader("🔍 Retrieved Context Details")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please load documents first to view context details.")
            return
        
        if st.session_state.latest_query_details is None:
            st.info("📋 No query has been executed yet. Run a query in the Chat Interface to see context details here.")
            return
        
        query_details = st.session_state.latest_query_details
        
        # === SUMMARY DASHBOARD ===
        st.markdown("---")
        
        # Key metrics at the top
        dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)
        
        with dashboard_col1:
            st.metric(
                "📄 Documents", 
                query_details["retrieval_info"]["num_documents"],
                help="Number of documents retrieved from the knowledge base"
            )
        
        with dashboard_col2:
            context_length = query_details['retrieval_info']['context_length']
            st.metric(
                "📝 Context Size", 
                f"{context_length:,} chars",
                help="Total characters in retrieved context"
            )
        
        with dashboard_col3:
            st.metric(
                "🔧 Retriever", 
                query_details["retrieval_info"]["retriever_type"].title(),
                help="Retrieval strategy used for this query"
            )
        
        with dashboard_col4:
            if "routing_info" in query_details:
                confidence = query_details["routing_info"]["confidence"]
                st.metric(
                    "🎯 Confidence", 
                    f"{confidence:.0%}",
                    help="Router confidence in strategy selection"
                )
            else:
                st.metric("🎯 Confidence", "N/A", help="Direct retriever selection")
        
        st.markdown("---")
        
        # === QUERY OVERVIEW ===
        with st.expander("📋 Query Overview", expanded=True):
            query_col1, query_col2 = st.columns([2, 3])
            
            with query_col1:
                st.markdown("**❓ Question:**")
                st.info(query_details['question'])
                
            with query_col2:
                st.markdown("**💬 Answer Preview:**")
                answer_preview = query_details['answer'][:150] + "..." if len(query_details['answer']) > 150 else query_details['answer']
                st.success(answer_preview)
        
        # === SMART ROUTER INSIGHTS ===
        if "routing_info" in query_details:
            with st.expander("🤖 Smart Router Decision", expanded=False):
                router_col1, router_col2 = st.columns([1, 2])
                
                with router_col1:
                    st.markdown("**📊 Decision Metrics**")
                    st.metric("Selected Strategy", query_details["routing_info"]["strategy"].title())
                    st.metric("Confidence Score", f"{query_details['routing_info']['confidence']:.2f}")
                
                with router_col2:
                    st.markdown("**🧠 Decision Reasoning**")
                    st.write(query_details["routing_info"]["reasoning"])
                    
                    if "analysis" in query_details["routing_info"]:
                        st.markdown("**📝 Query Analysis**")
                        st.text(query_details["routing_info"]["analysis"])
        
        # === RETRIEVED DOCUMENTS ===
        with st.expander("📄 Retrieved Documents", expanded=True):
            if query_details["retrieved_documents"]:
                # Quick overview table
                st.markdown("#### 📊 Documents Overview")
                doc_overview = []
                for i, doc in enumerate(query_details["retrieved_documents"], 1):
                    doc_overview.append({
                        "📄 Doc": f"#{i}",
                        "📁 Source": doc['source'].split('/')[-1] if '/' in doc['source'] else doc['source'],
                        "📏 Length": f"{len(doc['content']):,} chars",
                        "🔍 Preview": doc['content_preview'][:60] + "..." if len(doc['content_preview']) > 60 else doc['content_preview']
                    })
                
                st.dataframe(pd.DataFrame(doc_overview), width="stretch", hide_index=True)
                
                st.markdown("---")
                
                # Document selector and detailed view
                st.markdown("#### 🔍 Document Inspector")
                
                doc_selector_col1, doc_selector_col2 = st.columns([2, 1])
                
                with doc_selector_col1:
                    selected_doc = st.selectbox(
                        "Select document to inspect:",
                        range(len(query_details["retrieved_documents"])),
                        format_func=lambda x: f"Document #{x+1}: {query_details['retrieved_documents'][x]['source'].split('/')[-1]}",
                        key="doc_selector"
                    )
                
                with doc_selector_col2:
                    if selected_doc is not None:
                        doc = query_details["retrieved_documents"][selected_doc]
                        st.metric("Word Count", len(doc['content'].split()))
                
                if selected_doc is not None:
                    doc = query_details["retrieved_documents"][selected_doc]
                    
                    # Create tabs for different views of the document
                    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["📄 Content", "🏷️ Metadata", "📊 Analysis"])
                    
                    with doc_tab1:
                        st.markdown("**📍 Source Path:**")
                        st.code(doc['source'], language="text")
                        
                        st.markdown("**📖 Full Content:**")
                        st.text_area(
                            "Document Content",
                            doc['content'],
                            height=350,
                            key=f"content_view_{selected_doc}",
                            help="Complete document content used for retrieval"
                        )
                    
                    with doc_tab2:
                        st.markdown("**🏷️ Document Metadata:**")
                        if doc['metadata']:
                            st.json(doc['metadata'])
                        else:
                            st.info("No metadata available for this document")
                    
                    with doc_tab3:
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            st.markdown("**📊 Content Statistics**")
                            content = doc['content']
                            st.metric("Characters", len(content))
                            st.metric("Words", len(content.split()))
                            st.metric("Lines", len(content.split('\n')))
                        
                        with analysis_col2:
                            st.markdown("**🔤 Content Preview**")
                            sentences = content.split('. ')[:3]
                            preview = '. '.join(sentences) + ('.' if len(sentences) > 0 else '')
                            st.text_area("First 3 sentences:", preview, height=100)
                            
            else:
                st.warning("❌ No documents were retrieved for this query.")
        
        # === CONTEXT ANALYSIS ===
        with st.expander("📈 Context Analysis", expanded=False):
            if query_details["retrieved_documents"]:
                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                
                with analysis_col1:
                    st.markdown("**📏 Content Metrics**")
                    word_counts = [len(doc['content'].split()) for doc in query_details["retrieved_documents"]]
                    st.metric("Total Words", f"{sum(word_counts):,}")
                    st.metric("Avg Words/Doc", f"{sum(word_counts) // len(word_counts):,}")
                    st.metric("Max Words", f"{max(word_counts):,}")
                    st.metric("Min Words", f"{min(word_counts):,}")
                
                with analysis_col2:
                    st.markdown("**📚 Source Distribution**")
                    sources = [doc['source'] for doc in query_details["retrieved_documents"]]
                    unique_sources = list(set(sources))
                    st.metric("Unique Sources", len(unique_sources))
                    
                    if len(unique_sources) > 1:
                        for i, source in enumerate(unique_sources[:5], 1):
                            count = sources.count(source)
                            source_name = source.split('/')[-1] if '/' in source else source
                            st.text(f"{i}. {source_name}: {count} doc(s)")
                
                with analysis_col3:
                    st.markdown("**🎯 Retrieval Quality**")
                    total_chars = sum(len(doc['content']) for doc in query_details["retrieved_documents"])
                    st.metric("Total Characters", f"{total_chars:,}")
                    
                    # Calculate diversity score based on unique sources
                    diversity_score = len(unique_sources) / len(query_details["retrieved_documents"])
                    st.metric("Source Diversity", f"{diversity_score:.1%}")
                    
                    # Context efficiency (context length vs total content)
                    efficiency = query_details['retrieval_info']['context_length'] / total_chars if total_chars > 0 else 0
                    st.metric("Context Efficiency", f"{efficiency:.1%}")
            else:
                st.info("No context analysis available - no documents retrieved.")

    with tab4:
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