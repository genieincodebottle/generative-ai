import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
import tempfile
import os
import nest_asyncio

# Apply the patch to allow nested event loops
nest_asyncio.apply()

# Environment Variables
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Optional provider imports — fail gracefully so the app still starts
# even if the user has only one provider's dependencies installed.
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def show_error(e):
    """Display a helpful, specific error message based on the exception type."""
    import traceback
    msg = str(e)
    if any(k in msg for k in ["API_KEY", "api_key", "API key", "credentials", "authentication", "UNAUTHENTICATED"]):
        st.error("❌ API Key Error — your key is missing or invalid.")
        st.info("Get a Gemini key: https://aistudio.google.com/app/apikey  |  Groq key: https://console.groq.com/keys")
    elif any(k in msg.lower() for k in ["quota", "rate limit", "resource exhausted", "429"]):
        st.error("❌ Rate Limit — too many requests. Wait a moment and try again.")
    elif any(k in msg.lower() for k in ["pdf", "pypdf", "cannot read", "no text"]):
        st.error("❌ PDF Error — make sure the file has selectable text (not a scanned-only image PDF).")
    elif any(k in msg.lower() for k in ["no module", "importerror", "cannot import"]):
        st.error(f"❌ Missing dependency: {msg}")
        st.info("Fix: run `uv pip install -r requirements.txt` inside the rag_techniques folder.")
    else:
        st.error(f"❌ {msg}")
    with st.expander("🔍 Full error details"):
        st.code(traceback.format_exc())


class ReRankingRAG:
    def __init__(self, model_name, temperature, chunk_size, chunk_overlap, reranker_type, provider="Gemini (Google)"):
        self.provider = provider
        if provider == "Groq (Open Source)":
            self.llm = ChatGroq(model=model_name, temperature=temperature)
            # Groq has no embedding model — use HuggingFace locally
            self.embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                model_kwargs={"trust_remote_code": True}
            )
            self.embedding_label = "nomic-embed-text-v1.5 (HuggingFace, local)"
        else:
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=google_api_key
            )
            self.embedding_label = "gemini-embedding-001 (Google)"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reranker_type = reranker_type

    def load_pdfs(self, pdf_files):
        all_docs = []
        for pdf_file in pdf_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            all_docs.extend([doc.page_content for doc in docs])

            # Clean up temp file
            os.unlink(tmp_file_path)

        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.create_documents(all_docs)

        # Add metadata IDs to chunks
        for idx, chunk in enumerate(texts):
            chunk.metadata["id"] = idx

        self.vectorstore = Chroma.from_documents(texts, self.embeddings)

    def get_reranker_compressor(self):
        """Get the appropriate reranker based on selection"""
        try:
            if self.reranker_type == "FlashRank":
                try:
                    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
                    # Try to rebuild the model to fix Pydantic issues
                    try:
                        FlashrankRerank.model_rebuild()
                    except Exception:
                        pass
                    return FlashrankRerank(top_n=5), "FlashRank"
                except ImportError:
                    st.warning("FlashRank not available. Install with: pip install flashrank")
                    return None, None
                except Exception as e:
                    st.warning(f"FlashRank initialization failed: {str(e)}. This is a known Pydantic model issue.")
                    return None, None

            elif self.reranker_type == "Cross-Encoder (BGE)":
                try:
                    from langchain.retrievers.document_compressors import CrossEncoderReranker
                    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
                    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
                    return CrossEncoderReranker(model=model, top_n=5), "Cross-Encoder (BAAI/bge-reranker-base)"
                except ImportError:
                    st.warning("Cross-Encoder dependencies not available. Install with: pip install sentence-transformers")
                    return None, None

            elif self.reranker_type == "Cohere Rerank":
                try:
                    from langchain.retrievers.document_compressors import CohereRerank
                    cohere_api_key = os.getenv("COHERE_API_KEY")
                    if not cohere_api_key:
                        st.warning("COHERE_API_KEY environment variable not set")
                        return None, None
                    return CohereRerank(model="rerank-english-v3.0", top_n=5), "Cohere Rerank v3.0"
                except ImportError:
                    st.warning("Cohere dependencies not available. Install with: pip install cohere")
                    return None, None

            elif self.reranker_type == "LLM Listwise Rerank":
                try:
                    from langchain.retrievers.document_compressors import LLMListwiseRerank
                    label = f"LLM Listwise Rerank ({self.provider.split(' ')[0]})"
                    return LLMListwiseRerank.from_llm(self.llm, top_n=5), label
                except Exception as e:
                    st.warning(f"LLM Listwise Rerank not available: {str(e)}")
                    return None, None

            elif self.reranker_type == "LLM Chain Extractor":
                try:
                    from langchain.retrievers.document_compressors import LLMChainExtractor
                    label = f"LLM Chain Extractor ({self.provider.split(' ')[0]})"
                    return LLMChainExtractor.from_llm(self.llm), label
                except Exception as e:
                    st.warning(f"LLM Chain Extractor not available: {str(e)}")
                    return None, None

            elif self.reranker_type == "Embeddings Filter":
                try:
                    from langchain.retrievers.document_compressors import EmbeddingsFilter
                    return EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.76), "Embeddings Filter"
                except Exception as e:
                    st.warning(f"Embeddings Filter not available: {str(e)}")
                    return None, None

        except Exception as e:
            st.error(f"Error initializing reranker: {str(e)}")
            return None, None

    def reranking_rag(self, query):
        # Initial retrieval without re-ranking
        initial_docs = self.vectorstore.similarity_search(query, k=8)
        initial_context = "\n".join([doc.page_content for doc in initial_docs])

        # Generate initial response
        initial_prompt = ChatPromptTemplate.from_template(
            "Based on the following context, please answer the query:\nContext: {context}\nQuery: {query}"
        )
        initial_chain = initial_prompt | self.llm
        initial_response = initial_chain.invoke({"context": initial_context, "query": query})

        # Get reranker and apply re-ranking
        compressor, reranker_name = self.get_reranker_compressor()

        if compressor is None:
            # Fallback to regular retrieval if reranker fails
            reranked_docs = initial_docs[:5]
            reranker_name = "Similarity Search (Fallback)"
        else:
            try:
                # Create compression retriever with re-ranking
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": 8})
                )

                # Retrieve and re-rank documents
                reranked_docs = compression_retriever.get_relevant_documents(query)
            except Exception as e:
                st.warning(f"Re-ranking failed: {str(e)}. Using similarity search fallback.")
                reranked_docs = initial_docs[:5]
                reranker_name = "Similarity Search (Fallback)"

        # Generate final response using re-ranked documents
        reranked_context = "\n".join([doc.page_content for doc in reranked_docs])
        final_prompt = ChatPromptTemplate.from_template(
            "Based on the following re-ranked and optimized context, please provide a comprehensive answer to the query:\nContext: {context}\nQuery: {query}"
        )
        final_chain = final_prompt | self.llm
        final_response = final_chain.invoke({"context": reranked_context, "query": query})

        return {
            "initial_response": initial_response.content,
            "final_response": final_response.content,
            "initial_docs": initial_docs,
            "reranked_docs": reranked_docs,
            "retrieval_method": f"Re-ranking with {reranker_name}"
        }

    def run(self, query):
        return self.reranking_rag(query)


# Helper function for displaying documents
def display_docs(docs, title):
    st.subheader(title)
    for i, doc in enumerate(docs):
        with st.expander(f"Document {i + 1}"):
            st.write(f"**Content:**\n{doc.page_content}")
            if hasattr(doc, 'metadata') and doc.metadata:
                st.write("**Metadata:**")
                for key, value in doc.metadata.items():
                    st.write(f"  {key}: {value}")


# Streamlit App
st.set_page_config(page_title="Re-ranking RAG", page_icon="🔄", layout="wide")
st.title("Re-ranking RAG")
st.markdown(
    "Re-ranking RAG refines and reorders the initially retrieved documents before passing them to the LLM. "
    "It first retrieves a larger set of candidate chunks, then uses a second model (the re-ranker) to score "
    "and reorder them by relevance — so the LLM sees only the most relevant content."
)

# Add information about the process
with st.expander("🔄 How Re-ranking RAG Works"):
    st.markdown("""
    **Re-ranking RAG follows these steps:**

    1. **Initial Retrieval**: Retrieve the top 8 most relevant documents using similarity search
    2. **Initial Response Generation**: Generate a response using the initially retrieved context
    3. **Re-ranking Process**: Use selected reranker to intelligently reorder documents based on relevance
    4. **Optimized Retrieval**: Get the top 5 re-ranked documents that are most relevant to the query
    5. **Final Response Generation**: Generate an improved response using the re-ranked, high-quality context

    **How it differs from Basic RAG:**
    Basic RAG uses the top-K results from similarity search directly. Re-ranking RAG retrieves more candidates
    (top-8) then applies a smarter scoring model to pick the best 5 — improving precision.

    **Available Rerankers:**
    - **FlashRank**: Ultra-fast, lightweight — best for speed (no extra API needed)
    - **Cross-Encoder (BGE)**: BAAI's powerful cross-encoder model — best quality (no API needed)
    - **Cohere Rerank**: Commercial-grade reranking service (requires free COHERE_API_KEY)
    - **LLM Listwise Rerank**: Uses the selected LLM for zero-shot reranking
    - **LLM Chain Extractor**: Extracts relevant content using the selected LLM
    - **Embeddings Filter**: Filters based on embedding similarity threshold
    """)

# Additional Information Section
with st.expander("📚 About Re-ranking Options"):
    st.markdown("""
    **Available Reranking Methods:**

    1. **FlashRank** *(recommended for beginners)*: Ultra-lite & super-fast. Install: `pip install flashrank`
    2. **Cross-Encoder (BGE)**: Uses BAAI's bge-reranker-base. Install: `pip install sentence-transformers`
    3. **Cohere Rerank**: Commercial API. Install: `pip install cohere` + add `COHERE_API_KEY` to `.env`
    4. **LLM Listwise Rerank**: Zero-shot listwise reranking — uses your selected LLM (no extra install)
    5. **LLM Chain Extractor**: Extracts only relevant passages — uses your selected LLM (no extra install)
    6. **Embeddings Filter**: Filters by embedding similarity threshold (no extra install)

    > **Tip:** Start with **LLM Listwise Rerank** or **Embeddings Filter** — they work with your existing setup.
    > Try **FlashRank** or **Cross-Encoder (BGE)** for better quality after installing their packages.
    """)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Provider Selection
    provider_options = ["Gemini (Google)"]
    if GROQ_AVAILABLE:
        provider_options.append("Groq (Open Source)")
    provider = st.selectbox(
        "LLM Provider",
        provider_options,
        help="Gemini requires GOOGLE_API_KEY; Groq requires GROQ_API_KEY and uses HuggingFace embeddings locally"
    )

    # Model Selection (varies by provider)
    if provider == "Groq (Open Source)":
        model_name = st.selectbox(
            "Model Name",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            help="llama-3.1-8b-instant is faster; llama-3.3-70b-versatile gives higher quality answers"
        )
        st.markdown("Free API Key: https://console.groq.com/keys")
        if not groq_api_key:
            st.error("GROQ_API_KEY not set in .env file")
        if not HF_AVAILABLE:
            st.error("Run: pip install langchain-huggingface sentence-transformers")
        else:
            st.info("Embeddings: HuggingFace nomic-embed-text-v1.5 (~270 MB downloaded on first run)")
    else:
        model_name = st.selectbox(
            "Model Name",
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
            help="gemini-2.5-flash-lite is fastest; gemini-2.5-pro gives highest quality answers"
        )
        st.markdown("Free API Key: https://aistudio.google.com/app/apikey")
        if not google_api_key:
            st.error("GOOGLE_API_KEY not set in .env file")

    # Reranker Selection
    st.markdown("---")
    st.subheader("🔄 Reranker Selection")
    reranker_type = st.selectbox(
        "Reranker Type",
        ["FlashRank", "Cross-Encoder (BGE)", "Cohere Rerank", "LLM Listwise Rerank", "LLM Chain Extractor", "Embeddings Filter"],
        help="LLM Listwise Rerank and Embeddings Filter need no extra packages. FlashRank and Cross-Encoder (BGE) need pip install."
    )

    # Add API key info for Cohere
    if reranker_type == "Cohere Rerank":
        st.info("Requires COHERE_API_KEY in .env file")
        st.markdown("Get API key: https://dashboard.cohere.ai/")

    # Temperature Configuration
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. 0 = deterministic/factual, 1 = creative/varied. Use 0–0.3 for factual Q&A."
    )

    st.markdown("---")
    st.subheader("📝 Chunking Settings")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=4000,
        value=2000,
        step=100,
        help="Characters per chunk. Smaller (500–1000) = more precise retrieval. Larger (1500–3000) = more context per chunk."
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50,
        help="Characters shared between adjacent chunks. Prevents sentences from being cut at chunk boundaries. ~10% of chunk size is a good default."
    )

# PDF Upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Query
query = st.text_input(
    "Ask a question about your PDFs",
    placeholder="e.g. What are the key recommendations in this document?"
)

with st.expander("💡 Example questions to try"):
    st.markdown("""
    - *What are the main recommendations or conclusions?*
    - *Describe the process or methodology used.*
    - *What evidence supports the main argument?*
    - *What are the most important facts mentioned?*
    - *How does the document define [specific concept]?*
    """)

# Search
if st.button("Ask"):
    if model_name and uploaded_files and query:
        try:
            with st.spinner(f"Processing PDFs and running Re-ranking RAG with {reranker_type}... (this may take 20–45 seconds)"):
                rag = ReRankingRAG(model_name, temperature, chunk_size, chunk_overlap, reranker_type, provider)
                rag.load_pdfs(uploaded_files)
                result = rag.run(query)

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Final Response",
                "📝 Initial Response",
                "📊 Document Comparison",
                "ℹ️ Retrieval Info"
            ])

            with tab1:
                st.subheader("Final Re-ranked Response")
                st.write(result["final_response"])

            with tab2:
                st.subheader("Initial Response")
                st.caption("Generated before re-ranking — compare with the final response to see the improvement")
                st.write(result["initial_response"])

            with tab3:
                col1, col2 = st.columns(2)

                with col1:
                    display_docs(result["initial_docs"], "📄 Initial Retrieved Documents")

                with col2:
                    display_docs(result["reranked_docs"], "🔄 Re-ranked Documents")

            with tab4:
                st.subheader("Retrieval Method Information")
                st.write(f"**Method:** {result['retrieval_method']}")
                st.write(f"**Initial Documents Retrieved:** {len(result['initial_docs'])}")
                st.write(f"**Re-ranked Documents:** {len(result['reranked_docs'])}")
                st.write(f"**Embedding Model:** {rag.embedding_label}")
                st.write(f"**LLM Provider:** {provider}")
                st.write(f"**LLM Model:** {model_name}")
                st.write(f"**Selected Reranker:** {reranker_type}")

                st.markdown(f"""
                **{reranker_type} Re-ranking Process:**
                - Processes initially retrieved documents for relevance scoring
                - Considers query intent and semantic similarity
                - Reorders documents to prioritize most relevant content
                - Improves overall response quality and accuracy
                - Reduces noise and irrelevant information
                """)

        except Exception as e:
            show_error(e)
    else:
        st.warning("Please upload at least one PDF and enter a question before clicking Ask.")
