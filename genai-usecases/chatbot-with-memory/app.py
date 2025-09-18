"""
PDF Chat Bot with Ollama based local LLM
=====================================
A simple Streamlit application for chatting with PDF documents using local Ollama.
Configuration is done through the UI sidebar

"""

import streamlit as st
import os
import requests
from typing import Optional

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================================
# DEFAULT CONFIGURATION VALUES
# ================================
DEFAULT_CONFIG = {
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "llama3.2:1b",
    "ollama_temperature": 0.3,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "max_history_length": 50,
    "retriever_k": 2
}


# ================================
# BACKEND CODE - BUSINESS LOGIC
# ================================

def check_ollama_status(base_url: str) -> bool:
    """BACKEND: Check if Ollama server is running and accessible"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def initialize_llm(base_url: str, model: str, temperature: float) -> Optional[OllamaLLM]:
    """BACKEND: Initialize Ollama LLM with given configuration"""
    try:
        llm = OllamaLLM(
            base_url=base_url,
            model=model,
            temperature=temperature
        )
        # Test the connection
        llm.invoke("Hello")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None


def process_pdfs(pdf_files, embedding_model: str, chunk_size: int, chunk_overlap: int) -> Optional[FAISS]:
    """BACKEND: Process uploaded PDF files and create vector store"""
    try:
        documents = []

        # Process each PDF file
        for pdf_file in pdf_files:
            # Save uploaded file temporarily
            temp_path = f"temp_{pdf_file.name}"
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())

            # Load and process PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

            # Clean up temp file
            os.remove(temp_path)

        if not documents:
            st.error("No content found in the uploaded PDFs")
            return None

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        return vectorstore

    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return None


def create_chat_chain(llm, vectorstore, max_history: int, retriever_k: int):
    """BACKEND: Create conversational RAG chain with LangChain memory patterns"""
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": retriever_k}
        )

        # Create contextualized question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create contextualized question chain
        history_aware_retriever = (
            contextualize_q_prompt | llm | StrOutputParser() | retriever
        )

        # Create QA prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create the chain that formats documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the conversational RAG chain
        rag_chain = (
            RunnablePassthrough.assign(
                context=history_aware_retriever | format_docs
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # Store chat histories
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        # Create conversational RAG chain with memory
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        return conversational_rag_chain, store

    except Exception as e:
        st.error(f"Error creating chat chain: {str(e)}")
        return None, None


# ================================
# UI CODE - USER INTERFACE
# ================================

def setup_page():
    """UI: Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="PDF Chat Bot",
        page_icon="💬",
        layout="wide"
    )

    # Hide Streamlit style
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def render_header():
    """UI: Render page header"""
    st.title("💬 PDF Chat Bot")
    st.markdown("Upload PDFs and ask questions about their content")

def render_sidebar():
    """UI: Render sidebar with essential configuration"""
    st.sidebar.title("⚙️ Settings")

    # Initialize session state for config
    if 'config' not in st.session_state:
        st.session_state.config = DEFAULT_CONFIG.copy()

    # Status section
    st.sidebar.header("📊 Status")

    # Check Ollama status and show in sidebar
    if check_ollama_status(st.session_state.config.get("ollama_base_url", DEFAULT_CONFIG["ollama_base_url"])):
        st.sidebar.success("✅ Ollama running")
    else:
        st.sidebar.error("❌ Ollama not running")
        with st.sidebar.expander("🚀 Setup Instructions"):
            st.markdown("""
            **Quick setup:**
            1. Install from [ollama.com](https://ollama.com)
            2. Run: `ollama serve`
            3. Pull model: `ollama pull llama3.2:1b`
            4. Refresh this page
            """)

    # Essential settings only
    st.session_state.config["ollama_model"] = st.sidebar.selectbox(
        "🤖 Model",
        options=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "gemma3:1b", "gemma3:4b"],
        index=0,
        help="Choose model based on your RAM"
    )

    st.session_state.config["ollama_temperature"] = st.sidebar.slider(
        "🎛️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.config["ollama_temperature"],
        step=0.1,
        help="0 = focused, 1 = creative"
    )

    # Advanced settings in expander
    with st.sidebar.expander("🔧 Advanced Settings"):
        st.session_state.config["ollama_base_url"] = st.text_input(
            "Ollama URL",
            value=st.session_state.config["ollama_base_url"]
        )

        st.session_state.config["chunk_size"] = st.number_input(
            "Chunk Size",
            min_value=1000,
            max_value=8000,
            value=st.session_state.config["chunk_size"],
            step=1000
        )

        st.session_state.config["chunk_overlap"] = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=st.session_state.config["chunk_overlap"],
            step=50,
            help="Overlap between text chunks"
        )

    # PDF upload in sidebar
    st.sidebar.header("📄 Upload PDFs")
    pdf_files = st.sidebar.file_uploader(
        "Choose PDF files",
        accept_multiple_files=True,
        type=['pdf'],
        help="Upload PDF documents to chat with"
    )

    return st.session_state.config, pdf_files


def main():
    """UI: Main application function - orchestrates the entire app"""
    # Setup page
    setup_page()
    render_header()

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'chat_store' not in st.session_state:
        st.session_state.chat_store = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = "default_session"
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    # Render sidebar and get configuration and uploaded files
    config, pdf_files = render_sidebar()

    # Check Ollama status (stop if not running, but don't show main page message)
    if not check_ollama_status(config["ollama_base_url"]):
        st.error("❌ Ollama not running. Check sidebar for setup instructions.")
        st.stop()

    # Initialize LLM if not done or config changed
    current_llm_config = f"{config['ollama_base_url']}_{config['ollama_model']}_{config['ollama_temperature']}"
    if (st.session_state.llm is None or
        st.session_state.get('llm_config') != current_llm_config):
        with st.spinner(f"⚡ Initializing {config['ollama_model']}..."):
            st.session_state.llm = initialize_llm(
                config["ollama_base_url"],
                config["ollama_model"],
                config["ollama_temperature"]
            )
            if st.session_state.llm is None:
                st.stop()
            st.session_state.llm_config = current_llm_config
        st.toast("✅ Model initialized!", icon="🤖")

    # Process PDFs if uploaded
    if pdf_files:
        # Check if we need to reprocess (new files uploaded)
        current_files = [f.name for f in pdf_files]
        if ('processed_files' not in st.session_state or
            st.session_state.processed_files != current_files):

            with st.spinner(f"📄 Processing {len(pdf_files)} PDF file(s)..."):
                st.session_state.vectorstore = process_pdfs(
                    pdf_files,
                    config["embedding_model"],
                    config["chunk_size"],
                    config["chunk_overlap"]
                )
                st.session_state.processed_files = current_files

                if st.session_state.vectorstore is None:
                    st.stop()

            # Create chat chain
            with st.spinner("🔗 Setting up chat system..."):
                chain_result = create_chat_chain(
                    st.session_state.llm,
                    st.session_state.vectorstore,
                    config["max_history_length"],
                    config["retriever_k"]
                )
                if chain_result[0] is None:
                    st.stop()
                st.session_state.chain, st.session_state.chat_store = chain_result

            st.toast(f"✅ Processed {len(pdf_files)} PDF(s)!", icon="📄")

        # Chat interface
        st.subheader("💬 Chat with your documents")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Use the conversational RAG chain with memory
                        response = st.session_state.chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": st.session_state.session_id}}
                        )

                        answer = response
                        st.write(answer)

                        # Add assistant response to chat history for UI display
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        # Clear chat button
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            # Clear the LangChain memory store as well
            if st.session_state.chat_store and st.session_state.session_id in st.session_state.chat_store:
                st.session_state.chat_store[st.session_state.session_id].clear()
            st.rerun()

    else:
        # Simple instructions when no files uploaded
        st.info("📄 Upload PDF files in the sidebar to start chatting!")


if __name__ == "__main__":
    main()