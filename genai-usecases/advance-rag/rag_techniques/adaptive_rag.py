import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


class AdaptiveRAG:
    """
    Adaptive RAG: classifies each query as simple / moderate / complex,
    then adapts the number of retrieved chunks (k) and the prompt style
    to match the query complexity — avoiding over-retrieval for simple
    questions and under-retrieval for complex ones.
    """

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

        # How many chunks to retrieve per complexity level
        self.k_map = {"simple": 2, "moderate": 5, "complex": 8}

        # Prompt style per complexity level
        self.prompt_templates = {
            "simple": (
                "Answer the following simple, factual question in 1-3 sentences. "
                "Be direct and concise.\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Answer:"
            ),
            "moderate": (
                "Answer the following question with a clear explanation. "
                "Provide enough detail to be useful, but stay focused.\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Answer:"
            ),
            "complex": (
                "Answer the following complex question comprehensively. "
                "Synthesize information from multiple parts of the context, "
                "compare different perspectives if relevant, and structure your "
                "response with clear sections if needed.\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Comprehensive Answer:"
            ),
        }

    def load_documents(self, uploaded_files):
        all_docs = []
        for uploaded_file in uploaded_files:
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.create_documents(all_docs)
        self.vectorstore = Chroma.from_documents(texts, self.embeddings)

    def classify_query(self, query):
        """
        Use the LLM to classify the query as simple, moderate, or complex.
        Falls back to 'moderate' if the response is unclear.
        """
        classification_prompt = (
            "Classify the following question into exactly one complexity category.\n\n"
            "Categories:\n"
            "- simple: a direct factual lookup; a short, specific answer is expected "
            "(e.g. 'What is X?', 'When did Y happen?', 'What does Z stand for?')\n"
            "- moderate: requires explanation, context, or basic analysis "
            "(e.g. 'How does X work?', 'Why is Y important?', 'What are the benefits of Z?')\n"
            "- complex: requires synthesis from multiple sources, comparison of approaches, "
            "or multi-step reasoning "
            "(e.g. 'Compare X and Y', 'What are the trade-offs between...?', 'Analyze the relationship between...')\n\n"
            f"Question: {query}\n\n"
            "Respond with a single word only (simple / moderate / complex):"
        )
        response = self.llm.invoke(classification_prompt)
        level = response.content.strip().lower().split()[0]
        return level if level in ("simple", "moderate", "complex") else "moderate"

    def run(self, query):
        # Step 1: classify the query
        complexity_level = self.classify_query(query)

        # Step 2: retrieve k chunks based on complexity
        k = self.k_map[complexity_level]
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: generate answer with a prompt style matched to complexity
        prompt = self.prompt_templates[complexity_level].format(
            context=context, query=query
        )
        response = self.llm.invoke(prompt)

        strategy_descriptions = {
            "simple": f"Retrieved {k} chunks · concise prompt · direct factual answer",
            "moderate": f"Retrieved {k} chunks · explanatory prompt · structured answer",
            "complex": f"Retrieved {k} chunks · comprehensive prompt · synthesized multi-source answer",
        }

        return {
            "answer": response.content,
            "complexity_level": complexity_level,
            "k_used": k,
            "docs_retrieved": len(docs),
            "strategy": strategy_descriptions[complexity_level],
        }


# Streamlit App
st.set_page_config(page_title="Adaptive-RAG", page_icon="🧠", layout="wide")
st.title("Adaptive-RAG")
st.markdown(
    "Adaptive RAG classifies each query as **simple**, **moderate**, or **complex** using the LLM, "
    "then adapts the number of retrieved chunks and the prompt style to match. "
    "Simple questions get concise, fast answers. Complex questions trigger deeper retrieval and richer prompts."
)

# Add information about the process
with st.expander("🧠 How Adaptive RAG Works"):
    st.markdown("""
    **Adaptive RAG dynamically adjusts its retrieval strategy based on query complexity:**

    1. **Query Classification**: The LLM reads the question and classifies it as `simple`, `moderate`, or `complex`
    2. **Adaptive Retrieval**: Retrieves a different number of chunks based on complexity:
       - 🟢 **Simple** → 2 chunks (direct fact lookup — no need to flood the context window)
       - 🟡 **Moderate** → 5 chunks (explanation/analysis needs more context)
       - 🔴 **Complex** → 8 chunks (synthesis across multiple sources)
    3. **Adaptive Prompt**: Uses a different prompt style per level:
       - Simple → "Answer in 1-3 sentences, be direct"
       - Moderate → "Explain clearly with enough detail"
       - Complex → "Synthesize comprehensively, compare perspectives, structure if needed"
    4. **Response Generation**: Generates an answer tuned to the query complexity

    **Why this matters:**
    - Basic RAG always retrieves the same k chunks regardless of the question
    - Adaptive RAG avoids over-retrieving for simple questions (faster, cheaper, less noise in context)
    - And under-retrieving for complex questions (higher quality, more complete answers)

    **⬆️ How this improves on Corrective RAG:**
    Corrective RAG always uses the same retrieval depth and then self-critiques the answer.
    Adaptive RAG decides *upfront* how much context is needed based on the question type —
    avoiding unnecessary LLM calls for simple questions while going deeper for complex ones.
    """)

with st.expander("⚡ Performance Tips"):
    st.markdown("""
    - Adaptive RAG makes **2 LLM calls** per query (classification + answer generation)
    - Use `gemini-2.5-flash` or `llama-3.1-8b-instant` for fast, cheap classification
    - **Temperature 0.0–0.2 for classification accuracy** (the classify step uses your selected temperature — lower is better)
    - **Try these query types to see the badge change:**
      - 🟢 Simple: *"What is supervised learning?"*
      - 🟡 Moderate: *"How does gradient boosting work?"*
      - 🔴 Complex: *"Compare the trade-offs between solar and wind energy for grid stability"*
    - Sample documents are in `rag_techniques/sample_docs/` — no PDF needed to get started
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
        value=0.2,
        step=0.1,
        help="Controls randomness. Lower (0–0.2) is recommended for Adaptive RAG — it improves classification accuracy and factual answers."
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
    placeholder="e.g. What is the main argument of this document?"
)

with st.expander("💡 Example questions to try (with expected complexity badge)"):
    st.markdown("""
    **🟢 Simple** (2 chunks retrieved):
    - *What is supervised learning?*
    - *What are the main renewable energy sources?*
    - *What does ROC-AUC measure?*

    **🟡 Moderate** (5 chunks retrieved):
    - *How does gradient boosting work?*
    - *What are the challenges of solar energy?*
    - *Why is overfitting a problem and how do you fix it?*

    **🔴 Complex** (8 chunks retrieved):
    - *Compare the trade-offs between solar and wind energy for grid stability*
    - *Analyze the relationship between bias, fairness, and model accuracy in machine learning*
    - *What are the economic and policy factors that determine the pace of renewable energy adoption?*
    """)

# Search
if st.button("Ask"):
    if model_name and uploaded_files and query:
        try:
            with st.spinner("Processing documents and classifying query... (2 LLM calls — may take 15–30 seconds)"):
                rag = AdaptiveRAG(model_name, temperature, chunk_size, chunk_overlap, provider)
                rag.load_documents(uploaded_files)
                result = rag.run(query)

            # Show complexity badge
            badge_map = {"simple": "🟢 Simple", "moderate": "🟡 Moderate", "complex": "🔴 Complex"}
            badge = badge_map.get(result["complexity_level"], "🟡 Moderate")
            st.markdown(f"**Query Complexity:** {badge}")
            st.caption(f"Strategy: {result['strategy']}")

            st.markdown("---")
            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("🔍 Retrieval details"):
                st.markdown(f"""
                - **Complexity level:** `{result['complexity_level']}`
                - **Chunks retrieved (k):** `{result['k_used']}`
                - **Chunks actually found:** `{result['docs_retrieved']}`
                """)

        except Exception as e:
            show_error(e)
    else:
        st.warning("Please upload at least one document and enter a question before clicking Ask.")
