import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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


class CorrectiveRAG:
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
        for uploaded_file in pdf_files:
            is_txt = uploaded_file.name.lower().endswith(".txt")
            suffix = ".txt" if is_txt else ".pdf"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            if is_txt:
                loader = TextLoader(tmp_file_path, encoding="utf-8")
            else:
                loader = PyPDFLoader(tmp_file_path)

            docs = loader.load()
            all_docs.extend([doc.page_content for doc in docs])
            os.unlink(tmp_file_path)

        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.create_documents(all_docs)

        self.vectorstore = Chroma.from_documents(texts, self.embeddings)

    def corrective_rag(self, query):
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=3)
        initial_context = "\n".join([doc.page_content for doc in initial_docs])

        # Generate initial response
        initial_prompt = ChatPromptTemplate.from_template(
            "Based on the following context, please answer the query:\nContext: {context}\nQuery: {query}"
        )
        initial_chain = initial_prompt | self.llm
        initial_response = initial_chain.invoke({"context": initial_context, "query": query})

        # Generate critique
        critique_prompt = ChatPromptTemplate.from_template(
            "Please critique the following response to the query. Identify any potential errors or missing information:\nQuery: {query}\nResponse: {response}"
        )
        critique_chain = critique_prompt | self.llm
        critique = critique_chain.invoke({"response": initial_response.content, "query": query})

        # Retrieve additional information based on critique
        additional_docs = self.vectorstore.similarity_search(critique.content, k=2)
        additional_context = "\n".join([doc.page_content for doc in additional_docs])

        # Generate final response
        final_prompt = ChatPromptTemplate.from_template(
            "Based on the initial response, critique, and additional context, please provide an improved answer to the query:\nInitial Response: {initial_response}\nCritique: {critique}\nAdditional Context: {additional_context}\nQuery: {query}"
        )
        final_chain = final_prompt | self.llm
        final_response = final_chain.invoke({
            "initial_response": initial_response.content,
            "critique": critique.content,
            "additional_context": additional_context,
            "query": query
        })

        return {
            "initial_response": initial_response.content,
            "critique": critique.content,
            "final_response": final_response.content
        }

    def run(self, query):
        return self.corrective_rag(query)


# Streamlit App
st.set_page_config(page_title="Corrective-RAG", page_icon="🔧", layout="wide")
st.title("Corrective-RAG")
st.markdown(
    "Corrective RAG introduces a self-critique step to verify and improve the information retrieved "
    "before generating the final response. It generates an initial answer, critiques it for errors or gaps, "
    "retrieves additional context based on the critique, then generates a refined final response."
)

# Add information about the process
with st.expander("🔧 How Corrective RAG Works"):
    st.markdown("""
    **Corrective RAG follows these steps:**

    1. **Initial Retrieval**: Retrieve the top 3 most relevant documents for the query
    2. **Initial Response Generation**: Generate a response using the retrieved context
    3. **Critique Generation**: The model critiques its own response, identifying potential errors or missing information
    4. **Additional Retrieval**: Based on the critique, retrieve additional relevant documents
    5. **Final Response Generation**: Generate an improved response considering the initial response, critique, and additional context

    **How it differs from Basic RAG:**
    Basic RAG generates one answer and stops. Corrective RAG reflects on that answer and improves it —
    similar to how a human writer would draft, review, and revise.

    **Best for:** Questions where accuracy matters and you want the model to double-check itself.

    ---
    **⬆️ How this improves on Re-ranking RAG:**
    Re-ranking improves *which chunks* go into the context. Corrective RAG goes further — it also checks
    *the generated answer* for errors or missing information, then retrieves additional context to fix them.
    Next: [Adaptive RAG](adaptive_rag.py) — classifies the query first, then chooses the right retrieval depth automatically.
    """)

with st.expander("⚡ Performance Tips"):
    st.markdown("""
    - Corrective RAG makes **3 LLM calls** per query (initial + critique + final) — expect ~3× the latency of Basic RAG
    - Use `gemini-2.5-flash` or `llama-3.1-8b-instant` to keep it fast
    - **Temperature 0.3–0.5**: Balanced for self-critique tasks
    - If the critique is too vague, try lowering temperature for more focused self-criticism
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

# Document Upload
uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Sample TXT documents for testing are in the rag_techniques/sample_docs/ folder — no PDF needed to get started."
)

# Query
query = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What are the key conclusions of this document?"
)

with st.expander("💡 Example questions to try"):
    st.markdown("""
    - *What are the main findings or conclusions?*
    - *What does the document recommend?*
    - *Explain the methodology described in the document.*
    - *What are the limitations mentioned?*
    - *How does this document define [specific term or concept]?*
    """)

# Search
if st.button("Ask"):
    if model_name and uploaded_files and query:
        try:
            with st.spinner("Processing documents and running Corrective RAG... (makes 3 LLM calls — may take 30–60 seconds)"):
                rag = CorrectiveRAG(model_name, temperature, chunk_size, chunk_overlap, provider)
                rag.load_pdfs(uploaded_files)
                result = rag.run(query)

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["🎯 Final Response", "📝 Initial Response", "🔍 Critique"])

            with tab1:
                st.subheader("Final Improved Response")
                st.write(result["final_response"])

            with tab2:
                st.subheader("Initial Response")
                st.write(result["initial_response"])

            with tab3:
                st.subheader("Self-Critique")
                st.caption("The model's own assessment of what was missing or incorrect in the initial response")
                st.write(result["critique"])

        except Exception as e:
            show_error(e)
    else:
        st.warning("Please upload at least one document and enter a question before clicking Ask.")
