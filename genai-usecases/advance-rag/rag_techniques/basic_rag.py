import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
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

class BasicRAG:
    def __init__(self, model_name, temperature, chunk_size, chunk_overlap, top_k):
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
        self.top_k = top_k
        
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
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.create_documents(all_docs)
        
        # Add unique IDs to each text chunk
        for idx, text in enumerate(texts):
            text.metadata["id"] = idx
        
        # Create vector store and retriever
        self.vectorstore = Chroma.from_documents(texts, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        # Store chunks for display
        self.chunks = texts

    def basic_rag(self, query):
        """Basic RAG implementation"""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate response using retrieved context
        response_prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant tasked with answering questions based on the provided context. "
            "Please provide a comprehensive and accurate answer to the question using the context provided. "
            "If the context doesn't contain enough information to fully answer the question, "
            "please indicate what information is missing.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n"
            "Answer:"
        )
        
        chain = response_prompt | self.llm
        response = chain.invoke({"context": context, "query": query})

        return {
            "query": query,
            "answer": response.content,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "retrieval_method": f"Basic RAG (Top-{self.top_k} similarity search)"
        }

    def get_chunk_stats(self):
        """Get statistics about the processed chunks"""
        if hasattr(self, 'chunks'):
            total_chunks = len(self.chunks)
            avg_length = sum(len(chunk.page_content) for chunk in self.chunks) / total_chunks if total_chunks > 0 else 0
            return {
                "total_chunks": total_chunks,
                "avg_chunk_length": int(avg_length),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        return None

    def run(self, query):
        return self.basic_rag(query)

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
st.set_page_config(page_title="Basic RAG", page_icon="📚", layout="wide")
st.title("Basic RAG")
st.markdown("""`Basic RAG is the standard, straightforward implementation of Retrieval-Augmented Generation. 
It involves retrieving relevant information from a knowledge base in response to a query, then using this 
information to generate an answer using a language model.`""")

# Add information about the process
with st.expander("📚 How Basic RAG Works"):
    st.markdown("""
    **Basic RAG follows these steps:**
    
    1. **Document Loading**: Load and parse PDF documents into text
    2. **Text Chunking**: Split documents into smaller, manageable chunks
    3. **Embedding Creation**: Convert text chunks into vector embeddings
    4. **Vector Storage**: Store embeddings in a vector database (Chroma)
    5. **Query Processing**: Convert user query into vector embedding
    6. **Similarity Search**: Find most relevant chunks based on vector similarity
    7. **Context Assembly**: Combine retrieved chunks into context
    8. **Response Generation**: Use LLM to generate answer based on context
    
    **Key Components:**
    - **Retriever**: Finds relevant documents using similarity search
    - **Generator**: Language model that creates the final answer
    - **Vector Store**: Database for efficient similarity search
    - **Embeddings**: Vector representations of text for semantic search
    """)

# Additional Information Section
with st.expander("❓ Why We Need RAG"):
    st.markdown("""
    **Benefits of RAG:**
    
    1. **Knowledge Integration**: Combines the broad knowledge of language models with specific, up-to-date information
    2. **Improved Accuracy**: Grounds responses in retrieved facts, reducing hallucinations
    3. **Reduced Hallucinations**: Provides factual basis for responses instead of relying solely on training data
    4. **Easy Updates**: Allows updating knowledge base without retraining the entire model
    5. **Domain Expertise**: Enables AI to answer questions about specific documents or domains
    6. **Transparency**: Shows which documents were used to generate the answer
    
    **Use Cases:**
    - Document Q&A systems
    - Customer support chatbots
    - Research assistants
    - Legal document analysis
    - Technical documentation queries
    - Educational content exploration
    """)

# Performance Tips
with st.expander("⚡ Performance Tips"):
    st.markdown("""
    **Optimizing Basic RAG Performance:**
    
    **Document Processing:**
    - Use appropriate chunk sizes (1000-2000 chars for most documents)
    - Set overlap to 10-20% of chunk size for better context preservation
    - Consider document structure when chunking (paragraphs, sections)
    
    **Retrieval Settings:**
    - Start with Top-K = 3-5 for most queries
    - Increase Top-K for complex questions requiring more context
    - Monitor context length to stay within LLM limits
    
    **Model Selection:**
    - Use gemini-2.5-flash for faster responses
    - Use gemini-2.5-pro for complex reasoning tasks
    - Adjust temperature based on task (0.0-0.3 for factual, 0.5-0.8 for creative)
    
    **Quality Improvement:**
    - Ensure documents are well-formatted and structured
    - Remove irrelevant content before processing
    - Use descriptive filenames and metadata
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
        step=0.1,
        help="Controls randomness in responses. Lower values = more focused, higher values = more creative"
    )
    
    st.markdown("---")
    st.subheader("📝 Chunking Settings")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=4000,
        value=1000,
        step=100,
        help="Size of text chunks for processing"
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50,
        help="Overlap between consecutive chunks to maintain context"
    )
    
    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider(
        "Top-K Documents",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Number of most relevant documents to retrieve"
    )

# PDF Upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Query
query = st.text_input("Ask a question about your PDFs")

# Search
if st.button("Ask"):
    if model_name and uploaded_files and query:
        try:
            with st.spinner("Processing PDFs and running Basic RAG..."):
                rag = BasicRAG(model_name, temperature, chunk_size, chunk_overlap, top_k)
                rag.load_pdfs(uploaded_files)
                result = rag.run(query)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Answer", 
                "📄 Retrieved Documents", 
                "📊 Document Statistics",
                "ℹ️ Process Info"
            ])
            
            with tab1:
                st.subheader("Basic RAG Answer")
                st.write(result["answer"])
                
                # Show query for reference
                with st.expander("📝 Query Details"):
                    st.write(f"**Question:** {result['query']}")
                    st.write(f"**Documents Retrieved:** {len(result['retrieved_docs'])}")
                    st.write(f"**Context Length:** {len(result['context'])} characters")
            
            with tab2:
                display_docs(result["retrieved_docs"], "📄 Retrieved Documents")
                
                # Show full context
                with st.expander("📋 Complete Context Used"):
                    st.text_area(
                        "Context sent to LLM:",
                        result["context"],
                        height=300,
                        disabled=True
                    )
            
            with tab3:
                st.subheader("Document Processing Statistics")
                
                chunk_stats = rag.get_chunk_stats()
                if chunk_stats:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Chunks Created", chunk_stats["total_chunks"])
                        st.metric("Chunk Size Setting", f"{chunk_stats['chunk_size']} chars")
                    
                    with col2:
                        st.metric("Average Chunk Length", f"{chunk_stats['avg_chunk_length']} chars")
                        st.metric("Chunk Overlap", f"{chunk_stats['chunk_overlap']} chars")
                
                # Show sample chunks
                st.subheader("Sample Document Chunks")
                if hasattr(rag, 'chunks') and len(rag.chunks) > 0:
                    sample_size = min(5, len(rag.chunks))
                    for i in range(sample_size):
                        with st.expander(f"Sample Chunk {i+1}"):
                            st.write(f"**Length:** {len(rag.chunks[i].page_content)} characters")
                            st.write(f"**Content:** {rag.chunks[i].page_content[:200]}...")
                            if rag.chunks[i].metadata:
                                st.write(f"**Metadata:** {rag.chunks[i].metadata}")
            
            with tab4:
                st.subheader("RAG Process Information")
                st.write(f"**Method:** {result['retrieval_method']}")
                st.write(f"**Embedding Model:** models/gemini-embedding-001")
                st.write(f"**LLM Model:** {model_name}")
                st.write(f"**Vector Store:** ChromaDB")
                st.write(f"**Documents Retrieved:** {len(result['retrieved_docs'])}")
                st.write(f"**Temperature:** {temperature}")
                
                st.markdown("""
                **Basic RAG Process:**
                1. **Document Chunking**: Split documents into overlapping chunks
                2. **Embedding Creation**: Convert chunks to vector embeddings using Gemini
                3. **Vector Storage**: Store embeddings in ChromaDB vector database
                4. **Query Embedding**: Convert user query to vector representation
                5. **Similarity Search**: Find most similar chunks using cosine similarity
                6. **Context Assembly**: Combine retrieved chunks into coherent context
                7. **Response Generation**: Use Gemini LLM to generate answer from context
                """)
                
                st.markdown("""
                **Configuration Impact:**
                - **Higher Chunk Size**: More context per chunk, but less granular retrieval
                - **Higher Overlap**: Better context continuity, but more storage needed
                - **Higher Top-K**: More comprehensive answers, but longer processing time
                - **Higher Temperature**: More creative responses, but potentially less factual
                """)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure you have set GOOGLE_API_KEY in your .env file")
    else:
        st.error("Please upload PDFs and enter a question")

