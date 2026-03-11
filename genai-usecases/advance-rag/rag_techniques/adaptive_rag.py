import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
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


class PDFRAG:
    def __init__(self, model_name, temperature, chunk_size, chunk_overlap, provider="Gemini (Google)"):
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.create_documents(all_docs)

        self.vectorstore = Chroma.from_documents(texts, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def generate_answer(self, query, context):
        prompt = f"Answer this query based on the PDF content:\nQuery: {query}\nContext: {context}\nAnswer:"
        response = self.llm.invoke(prompt)
        return response.content

    def run(self, query):
        docs = self.retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])
        answer = self.generate_answer(query, context)
        return answer


# Streamlit App
st.set_page_config(page_title="Adaptive-RAG", page_icon="📚", layout="wide")
st.title("Adaptive-RAG")
st.markdown(
    "Adaptive RAG improves upon traditional RAG systems by dynamically adapting the retrieval process "
    "based on the specific query and context. This approach allows for more relevant and context-aware "
    "responses by routing queries to the most appropriate retrieval strategy."
)

# Add information about the process
with st.expander("🧠 How Adaptive RAG Works"):
    st.markdown("""
    **Adaptive RAG dynamically adjusts its retrieval strategy based on query complexity:**

    1. **Query Analysis**: Classify the query — is it a simple fact lookup, an analytical question, or a multi-hop reasoning task?
    2. **Strategy Selection**: Choose the appropriate retrieval depth and method for the query type
    3. **Context Retrieval**: Retrieve the right amount of context (simple queries need less; complex queries need more)
    4. **Response Generation**: Generate an answer tuned to the query complexity

    **Key Benefits:**
    - Avoids over-retrieving for simple questions (faster, cheaper)
    - Retrieves more context for complex multi-part questions (higher quality)
    - Adapts to different document types and content structures

    **How it differs from Basic RAG:**
    Basic RAG always retrieves the same number of chunks regardless of the query.
    Adaptive RAG decides *how much* and *what kind* of retrieval is needed based on the question.
    """)

with st.expander("⚡ Performance Tips"):
    st.markdown("""
    - **Temperature 0.0–0.3**: Best for factual Q&A
    - **Temperature 0.5–0.7**: Good for summaries and explanations
    - **Chunk Size 1000–1500**: Works well for most document types
    - **Model**: Use `gemini-2.5-flash-lite` or `llama-3.1-8b-instant` for fastest responses
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

    # Text Chunking Configuration
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. 0 = deterministic/factual, 1 = creative/varied. Use 0–0.3 for Q&A."
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
    placeholder="e.g. What is the main argument of this document?"
)

with st.expander("💡 Example questions to try"):
    st.markdown("""
    - *What are the main topics covered in this document?*
    - *Summarize the key points in 3 bullet points.*
    - *What does the document say about [specific topic]?*
    - *What are the conclusions or recommendations?*
    - *Compare the approaches described in the document.*
    """)

# Search
if st.button("Ask"):
    if model_name and uploaded_files and query:
        try:
            with st.spinner("Processing PDFs and running Adaptive RAG... (this may take 15–30 seconds)"):
                rag = PDFRAG(model_name, temperature, chunk_size, chunk_overlap, provider)
                rag.load_pdfs(uploaded_files)
                answer = rag.run(query)

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            show_error(e)
    else:
        st.warning("Please upload at least one PDF and enter a question before clicking Ask.")
