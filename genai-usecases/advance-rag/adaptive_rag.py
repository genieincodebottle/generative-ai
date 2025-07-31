import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import nest_asyncio
# Apply the patch to allow nested event loops
nest_asyncio.apply()

# Environment Variables
from dotenv import load_dotenv

load_dotenv()
# Ensure the GEMINI_API_KEY is set in the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it at .env file before running the app.")
os.environ["GOOGLE_API_KEY"] = api_key

class PDFRAG:
    def __init__(self, model_name, temperature, chunk_size, chunk_overlap):
        # Using a current and valid model name
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            # Using the recommended model for embeddings
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
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
st.markdown("""`Adaptive-RAG is a technique that improves upon traditional RAG systems by 
dynamically adapting the retrieval process based on the specific query and context. This 
approach allows for more relevant and context-aware responses, enhancing the overall user 
experience in applications such as question answering, document retrieval, and conversational agents.`""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model Selection
    model_name = st.selectbox("Model Name", ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro"])
    st.markdown("Free-tier API Key : https://aistudio.google.com/apikey")
    # Text Chunking Configuration
    temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
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
    if model_name and uploaded_files and query:
        try:
            with st.spinner("Processing PDFs..."):
                rag = PDFRAG(model_name, temperature, chunk_size, chunk_overlap)
                rag.load_pdfs(uploaded_files)
                answer = rag.run(query)
            
            st.write(answer)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please provide API keys, upload PDFs, and enter a question")