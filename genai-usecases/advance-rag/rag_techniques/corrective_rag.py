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

class CorrectiveRAG:
    def __init__(self, model_name, temperature, chunk_size, chunk_overlap):
        # Using Gemini models
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
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
st.markdown("""`Corrective RAG is a technique that introduces an additional step to verify and correct 
the information retrieved before generating the final response. This method aims to reduce errors and 
inconsistencies in the generated output by cross-checking the retrieved information against known facts 
or trusted sources through a self-critique mechanism.`""")

# Add information about the process
with st.expander("🔧 How Corrective RAG Works"):
    st.markdown("""
    **Corrective RAG follows these steps:**
    
    1. **Initial Retrieval**: Retrieve the top 3 most relevant documents for the query
    2. **Initial Response Generation**: Generate a response using the retrieved context
    3. **Critique Generation**: The model critiques its own response, identifying potential errors or missing information
    4. **Additional Retrieval**: Based on the critique, retrieve additional relevant documents
    5. **Final Response Generation**: Generate an improved response considering the initial response, critique, and additional context
    
    This approach helps reduce errors and inconsistencies by implementing a self-correction mechanism.
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
    st.markdown("Free-tier API Key : https://aistudio.google.com/apikey")
    
    # Temperature Configuration
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
            with st.spinner("Processing PDFs and running Corrective RAG..."):
                rag = CorrectiveRAG(model_name, temperature, chunk_size, chunk_overlap)
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
                st.write(result["critique"])
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload PDFs and enter a question")


