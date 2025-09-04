import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import tempfile
import os
import nest_asyncio

# Apply the patch to allow nested event loops
nest_asyncio.apply()

# Environment Variables
from dotenv import load_dotenv

load_dotenv()

# Ensure the GOOGLE_API_KEY is set in the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it at .env file before running the app.")
os.environ["GOOGLE_API_KEY"] = api_key

class HybridSearchRAG:
    def __init__(self, model_name, temperature, chunk_size, chunk_overlap, bm25_weight, vector_weight):
        # Using Gemini models
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
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
        
        # Create chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.chunks = text_splitter.create_documents(all_docs)
        
        # Add metadata IDs to chunks
        for idx, chunk in enumerate(self.chunks):
            chunk.metadata["id"] = idx
        
        # Create retrievers
        self.create_retrievers()

    def create_retrievers(self):
        """Create vector store and BM25 retrievers, then combine them"""
        # Create vector store retriever
        vectorstore = Chroma.from_documents(self.chunks, self.embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(self.chunks)
        bm25_retriever.k = 5  # Set number of documents to retrieve
        
        # Create ensemble retriever with weighted combination
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[self.bm25_weight, self.vector_weight]
        )
        
        # Store individual retrievers for comparison
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def safe_llm_call(self, prompt, **kwargs):
        """Helper function for safe LLM calls"""
        try:
            response = self.llm.invoke(prompt.format(**kwargs))
            return response.content if response else "No response generated."
        except Exception as e:
            st.error(f"Error in LLM call: {e}")
            return "An error occurred while generating the response."

    def hybrid_search_rag(self, query):
        """Main hybrid search RAG implementation"""
        # 1. Individual retrievals for comparison
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 2. Hybrid retrieval using ensemble
        hybrid_docs = self.ensemble_retriever.invoke(query)
        
        # Create contexts
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs])
        bm25_context = "\n\n".join([doc.page_content for doc in bm25_docs])
        hybrid_context = "\n\n".join([doc.page_content for doc in hybrid_docs])

        # 3. Generate response using hybrid context
        response_prompt = PromptTemplate.from_template(
            "You are an AI assistant tasked with answering questions based on the provided context. "
            "The context contains information retrieved using a hybrid search method combining keyword-based and semantic search. "
            "Please provide a comprehensive answer to the question, using the context when relevant "
            "and your general knowledge when necessary.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n"
            "Answer:"
        )
        final_answer = self.safe_llm_call(response_prompt, context=hybrid_context, query=query)

        # 4. Generate explanation of hybrid search process
        explanation_prompt = PromptTemplate.from_template(
            "Explain how the hybrid search process, combining keyword-based (BM25) and semantic (vector) search, "
            "might have improved the retrieval of relevant information for answering the given query. "
            "Consider the potential benefits of this approach compared to using only one search method.\n\n"
            "Query: {query}\n"
            "BM25 Weight: {bm25_weight}\n"
            "Vector Weight: {vector_weight}\n"
            "Explanation:"
        )
        hybrid_search_explanation = self.safe_llm_call(
            explanation_prompt, 
            query=query, 
            bm25_weight=self.bm25_weight, 
            vector_weight=self.vector_weight
        )

        return {
            "query": query,
            "final_answer": final_answer,
            "hybrid_search_explanation": hybrid_search_explanation,
            "vector_docs": vector_docs,
            "bm25_docs": bm25_docs,
            "hybrid_docs": hybrid_docs,
            "vector_context": vector_context,
            "bm25_context": bm25_context,
            "hybrid_context": hybrid_context,
            "retrieval_method": f"Hybrid Search (BM25: {self.bm25_weight}, Vector: {self.vector_weight})"
        }

    def run(self, query):
        return self.hybrid_search_rag(query)

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
st.set_page_config(page_title="Hybrid Search RAG", page_icon="🔀", layout="wide")
st.title("Hybrid Search RAG")
st.markdown("""`Hybrid Search RAG is a technique that combines multiple search methods to improve retrieval performance. 
It combines traditional keyword-based search (BM25) with semantic search (using embeddings). This approach can provide 
better results than either method alone, especially for queries that have both keyword-specific and semantic aspects.`""")

# Add information about the process
with st.expander("🔀 How Hybrid Search RAG Works"):
    st.markdown("""
    **Hybrid Search RAG follows these steps:**
    
    1. **Document Processing**: Split documents into chunks and prepare for different search methods
    2. **Multiple Retrievers**: Create both BM25 (keyword-based) and Vector (semantic) retrievers
    3. **Ensemble Retrieval**: Use EnsembleRetriever to combine results from both methods with configurable weights
    4. **Response Generation**: Generate answer using the hybrid-retrieved context
    5. **Process Explanation**: Provide explanation of how hybrid search improved retrieval
    
    **Key Benefits:**
    - **Best of Both Worlds**: Combines exact keyword matching with semantic understanding
    - **Improved Recall**: Captures documents that might be missed by single methods
    - **Configurable Weights**: Adjust the balance between keyword and semantic search
    - **Robust Performance**: Works well across different types of queries
    """)

# Additional Information Section
with st.expander("📚 About Hybrid Search"):
    st.markdown("""
    **What is Hybrid Search?**
    
    Hybrid Search combines multiple retrieval methods to leverage the strengths of each approach:
    
    **BM25 (Keyword-based Search):**
    - Excellent for exact keyword matches
    - Good for technical terms and specific phrases
    - Fast and interpretable
    - Works well for factual queries
    
    **Vector Search (Semantic Search):**
    - Understands meaning and context
    - Good for conceptual queries
    - Handles synonyms and paraphrasing
    - Captures semantic relationships
    
    **Ensemble Approach:**
    - **Weighted Combination**: Balances results from both methods
    - **Diversity**: Reduces bias from single retrieval methods
    - **Adaptability**: Can be tuned for different use cases
    - **Robustness**: Handles various query types effectively
    
    **Use Cases:**
    - Document search with mixed query types
    - Question answering systems
    - Legal and medical document retrieval
    - Academic research assistance
    """)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model Selection
    model_name = st.selectbox("Model Name", [
        "gemini-2.5-pro", 
        "gemini-2.5-flash", 
        "gemini-2.0-pro"
    ])
    st.markdown("Free-tier API Key: https://aistudio.google.com/apikey")
    
    # Temperature Configuration
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    st.markdown("---")
    st.subheader("🔀 Hybrid Search Settings")
    
    # Weight configuration
    bm25_weight = st.slider(
        "BM25 Weight (Keyword Search)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Weight for keyword-based search (BM25)"
    )
    
    vector_weight = st.slider(
        "Vector Weight (Semantic Search)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Weight for semantic search (vector similarity)"
    )
    
    # Normalize weights
    total_weight = bm25_weight + vector_weight
    if total_weight > 0:
        bm25_weight_norm = bm25_weight / total_weight
        vector_weight_norm = vector_weight / total_weight
        st.info(f"Normalized weights - BM25: {bm25_weight_norm:.2f}, Vector: {vector_weight_norm:.2f}")
    else:
        st.warning("At least one weight should be greater than 0")
    
    st.markdown("---")
    st.subheader("📝 Chunking Settings")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=4000,
        value=2000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50
    )

# PDF Upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Query
query = st.text_input("Ask a question about your PDFs")

# Search
if st.button("Ask"):
    if model_name and uploaded_files and query and (bm25_weight + vector_weight > 0):
        try:
            with st.spinner("Processing PDFs and running Hybrid Search RAG..."):
                rag = HybridSearchRAG(model_name, temperature, chunk_size, chunk_overlap, bm25_weight, vector_weight)
                rag.load_pdfs(uploaded_files)
                result = rag.run(query)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 Final Answer", 
                "🔍 Search Comparison", 
                "📊 Retrieved Documents",
                "💡 Hybrid Explanation",
                "ℹ️ Retrieval Info"
            ])
            
            with tab1:
                st.subheader("Hybrid Search Answer")
                st.write(result["final_answer"])
            
            with tab2:
                st.subheader("Search Method Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔤 BM25 (Keyword) Results**")
                    st.write(f"Weight: {bm25_weight}")
                    with st.expander("View BM25 Context"):
                        st.write(result["bm25_context"][:500] + "..." if len(result["bm25_context"]) > 500 else result["bm25_context"])
                
                with col2:
                    st.markdown("**🧠 Vector (Semantic) Results**")
                    st.write(f"Weight: {vector_weight}")
                    with st.expander("View Vector Context"):
                        st.write(result["vector_context"][:500] + "..." if len(result["vector_context"]) > 500 else result["vector_context"])
                
                with col3:
                    st.markdown("**🔀 Hybrid (Combined) Results**")
                    st.write(f"BM25: {bm25_weight}, Vector: {vector_weight}")
                    with st.expander("View Hybrid Context"):
                        st.write(result["hybrid_context"][:500] + "..." if len(result["hybrid_context"]) > 500 else result["hybrid_context"])
            
            with tab3:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    display_docs(result["bm25_docs"], "🔤 BM25 Retrieved Documents")
                
                with col2:
                    display_docs(result["vector_docs"], "🧠 Vector Retrieved Documents")
                
                with col3:
                    display_docs(result["hybrid_docs"], "🔀 Hybrid Retrieved Documents")
            
            with tab4:
                st.subheader("Hybrid Search Process Explanation")
                st.write(result["hybrid_search_explanation"])
            
            with tab5:
                st.subheader("Retrieval Method Information")
                st.write(f"**Method:** {result['retrieval_method']}")
                st.write(f"**BM25 Documents Retrieved:** {len(result['bm25_docs'])}")
                st.write(f"**Vector Documents Retrieved:** {len(result['vector_docs'])}")
                st.write(f"**Hybrid Documents Retrieved:** {len(result['hybrid_docs'])}")
                st.write("**Embedding Model:** models/gemini-embedding-001")
                st.write(f"**LLM Model:** {model_name}")
                st.write(f"**BM25 Weight:** {bm25_weight}")
                st.write(f"**Vector Weight:** {vector_weight}")
                
                st.markdown("""
                **Hybrid Search Process:**
                - **BM25 Retrieval**: Uses keyword matching and term frequency analysis
                - **Vector Retrieval**: Uses semantic similarity in embedding space
                - **Ensemble Combination**: Weighted combination of both retrieval methods
                - **Deduplication**: Removes duplicate documents from combined results
                - **Final Ranking**: Produces unified ranking based on weighted scores
                """)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure you have set GOOGLE_API_KEY in your .env file and installed required dependencies")
    else:
        if bm25_weight + vector_weight == 0:
            st.error("Please set at least one weight to be greater than 0")
        else:
            st.error("Please upload PDFs and enter a question")

