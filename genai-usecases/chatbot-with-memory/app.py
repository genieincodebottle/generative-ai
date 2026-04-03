"""
PDF Chat Bot with Memory
========================
A Streamlit application for chatting with PDF documents using Groq or Gemini LLMs.
Supports dual LLM providers with RAG (Retrieval-Augmented Generation) and
conversational memory using LangChain 0.3+ patterns.

Providers:
  - Primary:   Groq (free API) + HuggingFace Embeddings + FAISS
  - Secondary: Gemini (free API) + Google Embeddings + FAISS
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain core imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Provider imports (graceful — app works even if one provider isn't installed)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Search upward for .env (works whether run from this folder or repo root)
# A local .env in this folder takes priority over the repo-root .env
load_dotenv()                          # local .env (if present)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=False)

# ================================
# CONFIGURATION
# ================================

# Provider-specific settings — eliminates scattered if/else branching
PROVIDERS = {
    "Groq": {
        "available": GROQ_AVAILABLE,
        "env_key": "GROQ_API_KEY",
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b"],
        "model_help": "llama-3.1-8b is fast; llama-3.3-70b & llama-4-scout are more capable",
    },
    "Gemini": {
        "available": GEMINI_AVAILABLE,
        "env_key": "GOOGLE_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
        "model_help": "flash models are fast; pro is more capable",
    },
}

DEFAULT_CONFIG = {
    "provider": "Groq",
    "temperature": 0.3,
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "retriever_k": 3,
}

# Prompts — extracted here so they're easy to find and modify
CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

QA_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the answer concise (3 sentences max).\n\n"
    "{context}"
)


# ================================
# BACKEND — BUSINESS LOGIC
# ================================

def get_api_key(provider: str) -> str:
    """Get API key for the given provider from environment. Returns empty string if missing."""
    return os.getenv(PROVIDERS[provider]["env_key"], "")


def initialize_llm(provider: str, model: str, temperature: float):
    """Initialize LLM based on the selected provider."""
    api_key = get_api_key(provider)
    if not api_key:
        st.error(f"{PROVIDERS[provider]['env_key']} not found. Add it to your `.env` file.")
        return None

    try:
        if provider == "Groq":
            llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        else:
            llm = ChatGoogleGenerativeAI(
                model=model, temperature=temperature, google_api_key=api_key,
            )
        llm.invoke("Hi")  # connectivity check
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None


def get_embeddings(provider: str):
    """Return embeddings model based on the selected provider."""
    if provider == "Groq":
        if not HF_AVAILABLE:
            st.error("Install `langchain-huggingface` for Groq embeddings.")
            return None
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    # Gemini
    api_key = get_api_key(provider)
    if not api_key:
        st.error(f"{PROVIDERS[provider]['env_key']} not found. Add it to your `.env` file.")
        return None
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=api_key,
    )


def load_pdfs(pdf_files) -> list:
    """Load text from uploaded PDF files using temp files for safe handling."""
    documents = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name
        try:
            documents.extend(PyPDFLoader(tmp_path).load())
        finally:
            os.unlink(tmp_path)
    return documents


def process_pdfs(pdf_files, provider: str, chunk_size: int, chunk_overlap: int):
    """Process uploaded PDFs: extract text → chunk → embed → build FAISS index."""
    try:
        documents = load_pdfs(pdf_files)
        if not documents:
            st.error("No content found in the uploaded PDFs.")
            return None

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        ).split_documents(documents)

        embeddings = get_embeddings(provider)
        if embeddings is None:
            return None

        return FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        return None


def create_chat_chain(llm, vectorstore, retriever_k: int):
    """Create a conversational RAG chain with LangChain memory."""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Reformulate user question using history, then retrieve relevant docs
        history_aware_retriever = (
            contextualize_q_prompt | llm | StrOutputParser() | retriever
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(context=history_aware_retriever | format_docs)
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # Session-based memory store
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return conversational_rag_chain, store

    except Exception as e:
        st.error(f"Error creating chat chain: {e}")
        return None, None


# ================================
# UI — USER INTERFACE
# ================================

def setup_page():
    """Configure Streamlit page."""
    st.set_page_config(page_title="PDF Chat Bot", page_icon="💬", layout="wide")
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


def render_header():
    """Render page header."""
    st.title("💬 PDF Chat Bot with Memory")
    st.markdown("Upload PDFs and ask questions — the bot remembers your conversation!")


def render_sidebar():
    """Render sidebar: provider selection, model config, advanced settings, PDF upload."""
    st.sidebar.title("⚙️ Settings")

    if "config" not in st.session_state:
        st.session_state.config = DEFAULT_CONFIG.copy()
    config = st.session_state.config

    # --- Provider selection ---
    st.sidebar.header("🔌 LLM Provider")
    available_providers = [name for name, info in PROVIDERS.items() if info["available"]]
    if not available_providers:
        st.sidebar.error("No LLM provider installed! Install langchain-groq or langchain-google-genai.")
        st.stop()

    config["provider"] = st.sidebar.radio(
        "Choose Provider", options=available_providers, index=0,
        help="Groq is free & fast. Gemini is Google's free-tier model.",
    )

    # --- API key status ---
    provider_info = PROVIDERS[config["provider"]]
    has_key = bool(get_api_key(config["provider"]))
    if has_key:
        st.sidebar.success(f"✅ {config['provider']} API key loaded")
    else:
        st.sidebar.error(f"❌ {provider_info['env_key']} missing — check your `.env` file")

    # --- Model & temperature ---
    config["model"] = st.sidebar.selectbox(
        "🤖 Model", options=provider_info["models"], index=0,
        help=provider_info["model_help"],
    )
    config["temperature"] = st.sidebar.slider(
        "🎛️ Temperature", 0.0, 1.0, value=config["temperature"], step=0.1,
        help="0 = focused, 1 = creative",
    )

    # --- Advanced settings ---
    with st.sidebar.expander("🔧 Advanced Settings"):
        config["chunk_size"] = st.number_input(
            "Chunk Size", min_value=500, max_value=8000,
            value=config["chunk_size"], step=500,
        )
        config["chunk_overlap"] = st.number_input(
            "Chunk Overlap", min_value=0, max_value=500,
            value=config["chunk_overlap"], step=50,
        )
        config["retriever_k"] = st.number_input(
            "Retriever K (docs to fetch)", min_value=1, max_value=10,
            value=config["retriever_k"], step=1,
        )

    # --- PDF upload ---
    st.sidebar.header("📄 Upload PDFs")
    pdf_files = st.sidebar.file_uploader(
        "Choose PDF files", accept_multiple_files=True, type=["pdf"],
        help="Upload one or more PDF documents to chat with",
    )
    return config, pdf_files


def init_session_state():
    """Initialize all session state keys with defaults."""
    defaults = {
        "chat_history": [],
        "vectorstore": None,
        "chain": None,
        "chat_store": None,
        "session_id": "default_session",
        "llm": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def ensure_llm_ready(config):
    """Initialize or re-initialize the LLM when config changes."""
    if not get_api_key(config["provider"]):
        env_key = PROVIDERS[config["provider"]]["env_key"]
        st.error(f"❌ {env_key} is missing. Add it to your `.env` file and restart.")
        st.stop()

    llm_fingerprint = f"{config['provider']}_{config['model']}_{config['temperature']}"
    if st.session_state.llm is None or st.session_state.get("llm_fingerprint") != llm_fingerprint:
        with st.spinner(f"⚡ Initializing {config['model']}..."):
            st.session_state.llm = initialize_llm(
                config["provider"], config["model"], config["temperature"],
            )
            if st.session_state.llm is None:
                st.stop()
            st.session_state.llm_fingerprint = llm_fingerprint
        st.toast("✅ Model ready!", icon="🤖")


def ensure_pdfs_indexed(config, pdf_files):
    """Process and index PDFs if new files are uploaded or provider changed."""
    current_files = sorted(f.name for f in pdf_files)
    needs_reprocess = (
        "processed_files" not in st.session_state
        or st.session_state.processed_files != current_files
        or st.session_state.get("processed_provider") != config["provider"]
    )
    if not needs_reprocess:
        return

    with st.spinner(f"📄 Processing {len(pdf_files)} PDF(s) with {config['provider']}..."):
        st.session_state.vectorstore = process_pdfs(
            pdf_files, config["provider"], config["chunk_size"], config["chunk_overlap"],
        )
        if st.session_state.vectorstore is None:
            st.stop()
        st.session_state.processed_files = current_files
        st.session_state.processed_provider = config["provider"]

    with st.spinner("🔗 Setting up chat chain..."):
        chain, store = create_chat_chain(
            st.session_state.llm, st.session_state.vectorstore, config["retriever_k"],
        )
        if chain is None:
            st.stop()
        st.session_state.chain = chain
        st.session_state.chat_store = store

    st.toast(f"✅ {len(pdf_files)} PDF(s) indexed!", icon="📄")


def render_chat():
    """Render chat history, handle user input, and display responses."""
    st.subheader("💬 Chat with your documents")

    # Display existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": st.session_state.session_id}},
                    )
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    # Clear history button
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.chat_store and st.session_state.session_id in st.session_state.chat_store:
            st.session_state.chat_store[st.session_state.session_id].clear()
        st.rerun()


def main():
    """Main application entry point."""
    setup_page()
    render_header()
    init_session_state()

    config, pdf_files = render_sidebar()

    ensure_llm_ready(config)

    if not pdf_files:
        st.info("📄 Upload PDF files in the sidebar to start chatting!")
        return

    ensure_pdfs_indexed(config, pdf_files)
    render_chat()


if __name__ == "__main__":
    main()
