# Standard library imports
import base64
import io
import os
import tempfile
from datetime import datetime
from typing import Any

# Environment and API setup
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from google import genai

# Image processing
from PIL import Image

# RAG components
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# CrewAI components
from crewai import Agent, Task, Crew, Process

# UI
import streamlit as st

# Load environment variables from .env file
load_dotenv()

os.environ["OPIK_API_KEY"] =  os.getenv("OPIK_API_KEY")
os.environ["OPIK_WORKSPACE"] = "genieincodebottle"

# Get API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
if "documents" not in st.session_state:
    st.session_state.documents = []
    
if "agent_tasks" not in st.session_state:
    st.session_state.agent_tasks = []

# Set page configuration
st.set_page_config(
    page_title="Llama-4 Multi-Function App",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# For Chat task
def chat_with_groq(model_name, messages):
    """Chat with the Groq API using text-only inputs"""
    if not GROQ_API_KEY:
        return "Groq API key not set."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq API: {str(e)}"
    
# For OCR task    
def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    # Determine the format based on the original image format or default to PNG
    if hasattr(image, 'format') and image.format:
        img_format = image.format
    else:
        img_format = "PNG"
    
    buffered = io.BytesIO()
    image.save(buffered, format=img_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str, img_format.lower()

def process_with_groq(model_name, image, prompt):
    """Process the image with the Groq and Llama-4 Scout Model"""
    if not GROQ_API_KEY:
        return "Groq API key not set."
    # Encode image to base64
    base64_string, img_format = encode_image_to_base64(image)
    
    # Create Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Determine media type based on image format
    if img_format == 'jpg' or img_format == 'jpeg':
        media_type = "image/jpeg"
    elif img_format == 'png':
        media_type = "image/png"
    else:
        # Default to JPEG if format is unknown
        media_type = "image/jpeg"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_string}"
                            },
                        },
                    ],
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq API: {str(e)}"

# For RAG Task
def process_document(uploaded_file):
    """Process an uploaded document for RAG"""
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # Load documents based on file type
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith('.txt'):
        loader = TextLoader(tmp_path)
    elif uploaded_file.name.endswith('.csv'):
        loader = CSVLoader(tmp_path)
    else:
        # Delete the temporary file
        os.unlink(tmp_path)
        return None, "Unsupported file format"
    
    try:
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings (use a small model for simplicity)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create or update vector store
        if st.session_state.vector_store is None:
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.vector_store = vector_store
        else:
            st.session_state.vector_store.add_documents(chunks)
        
        # Add to documents list
        st.session_state.documents.append({
            "name": uploaded_file.name,
            "chunks": len(chunks),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Delete the temporary file
        os.unlink(tmp_path)
        
        return chunks, None
    except Exception as e:
        # Delete the temporary file
        os.unlink(tmp_path)
        return None, f"Error processing document: {str(e)}"

def query_documents(query, top_k=3):
    """Query the vector store for relevant documents"""
    if st.session_state.vector_store is None:
        return []
    
    results = st.session_state.vector_store.similarity_search(query, k=top_k)
    return results

# For CreWAI based Agentic AI task
def create_groq_llm(model_name):
    """Create a LangChain ChatGroq LLM instance for CrewAI"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024
    )

def fallback_to_gemini(task_description):
    """Fallback to Gemini model when Llama/Groq fails"""
    if not GOOGLE_API_KEY:
        return "Gemini API key not set in environment variables. Please add GEMINI_API_KEY to your .env file."
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        # Create a comprehensive prompt for the task
        prompt = f"""
        I need you to complete the following task as if you were a team of three expert collaborators:
        
        1. First, as a Researcher, find and organize all relevant information for this task.
        2. Then, as an Analyst, analyze this information and extract key insights.
        3. Finally, as a Writer, create a comprehensive final deliverable.
        
        The task is:
        {task_description}
        
        Please structure your response with clear sections for:
        - Research Findings
        - Analysis & Insights
        - Final Deliverable
        
        Make your response comprehensive, well-organized, and directly addressing the task.
        """
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt
            ]
        )
        # Return the text response
        return response.text
    except Exception as e:
        return f"Error using Gemini fallback: {str(e)}"
    
def execute_agent_task(task_description, model_name):
    """Execute a task using CrewAI framework with fallback to Gemini model if Llama fails"""
    try:
        # Create LLM instance for agents
        llm = create_groq_llm(model_name)
        
        # Create different specialized agents for the crew
        researcher = Agent(
            role="Researcher",
            goal="Find and gather all relevant information for the task",
            backstory="You are an expert researcher with keen attention to detail and a talent for finding relevant information.",
            verbose=True,
            allow_delegation=True,
            llm=llm
        )
        
        analyst = Agent(
            role="Analyst",
            goal="Analyze the information and extract insights",
            backstory="You are a skilled analyst with a talent for identifying patterns and deriving valuable insights from data.",
            verbose=True,
            allow_delegation=True,
            llm=llm
        )
        
        writer = Agent(
            role="Writer",
            goal="Create well-written, clear, and comprehensive content",
            backstory="You are an accomplished writer with a talent for creating engaging and informative content.",
            verbose=True,
            allow_delegation=True,
            llm=llm
        )
        
        # Create tasks for each agent
        research_task = Task(
            description=f"Research task: {task_description}\n\nFind and gather all relevant information needed to complete this task.",
            agent=researcher,
            expected_output="Comprehensive research findings"
        )
        
        analysis_task = Task(
            description=f"Analysis task: After reviewing the research, perform an in-depth analysis to extract key insights for: {task_description}",
            agent=analyst,
            expected_output="Analytical report with key insights",
            context=[research_task]
        )
        
        writing_task = Task(
            description=f"Writing task: Create a final deliverable for: {task_description}\n\nBased on the research and analysis, create a well-structured and comprehensive final output.",
            agent=writer,
            expected_output="Final deliverable",
            context=[research_task, analysis_task]
        )
        
        # Create the crew with the process
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            verbose=True,
            process=Process.sequential  # Execute tasks sequentially
        )
        
        # Execute the crew's work and get the result
        result = crew.kickoff()
        
        # Check if result is empty or None
        if not result or result.strip() == "":
            # If empty result, fallback to Gemini
            fallback_result = fallback_to_gemini(task_description)
            result = f"[FALLBACK TO GEMINI MODEL DUE TO EMPTY RESPONSE FROM PRIMARY MODEL]\n\n{fallback_result}"
        
        # Add to agent tasks history
        st.session_state.agent_tasks.append({
            "task": task_description,
            "response": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return result
    
    except Exception as e:
        error_message = f"Error executing agent task: {str(e)}"
        st.warning(f"Primary model error: {str(e)}. Attempting fallback to Gemini model...")
        
        # Fallback to Gemini
        try:
            fallback_result = fallback_to_gemini(task_description)
            response = f"[FALLBACK TO GEMINI MODEL DUE TO ERROR: {str(e)}]\n\n{fallback_result}"
            
            # Add to agent tasks history
            st.session_state.agent_tasks.append({
                "task": task_description,
                "response": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return response
        except Exception as gemini_error:
            # If both models fail, return combined error
            final_error = f"Both models failed.\nPrimary error: {str(e)}\nFallback error: {str(gemini_error)}"
            
            # Add error to history
            st.session_state.agent_tasks.append({
                "task": task_description,
                "response": final_error,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return final_error

# Sidebar for model selection and configuration
with st.sidebar:
    st.title("🦙 Llama-4 Scout App")
    
    # Model selection
    st.subheader("Model Configuration")
    
    available_models = ["meta-llama/llama-4-scout-17b-16e-instruct"]
    model = st.selectbox(
        "Select Model",
        options=available_models,
        index=0,
        help="Select the model to use"
    )
    
    # API Key Information
    st.markdown("---")
    st.markdown("""
    ### Setup Instructions
       - Use .env file available at root folder
       - Add your API keys as follows:
       ```
       GROQ_API_KEY=your_key_here
       GOOGLE_API_KEY=your_key_here 
       ```
    """)
    
    # Document information for RAG
    if st.session_state.documents:
        st.markdown("---")
        st.subheader("📚 Loaded Documents")
        for i, doc in enumerate(st.session_state.documents):
            st.markdown(f"**{i+1}. {doc['name']}**")
            st.caption(f"Chunks: {doc['chunks']} | Uploaded: {doc['uploaded_at']}")
        
        if st.button("Clear All Documents"):
            st.session_state.documents = []
            st.session_state.vector_store = None
            st.success("All documents cleared!")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 OCR", "📚 RAG", "🤖 Agent AI"])

# Tab 1: Normal Chat
with tab1:
    st.header("💬 Chat")
    
    # Chat history display area
    chat_container = st.container()
    with chat_container:
        for i, message_obj in enumerate(st.session_state.chat_history):
            if message_obj["role"] == "user":
                st.markdown(f"**You:** {message_obj['content']}")
            else:
                st.markdown(f"**Llama-4:** {message_obj['content']}")
            
            # Add a divider except after the last message
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
    
    # Chat input area
    user_input = st.text_area("Your message:", key="chat_input", height=100)
    
    if st.button("Send", key="send_chat"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Format messages for the API
            formatted_messages = [
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.chat_history
            ]
            
            # Get response from model
            with st.spinner("Llama-4 is thinking..."):
                response = chat_with_groq(model, formatted_messages)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update UI
            st.rerun()
    
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# Tab 2: OCR (from original code)
with tab2:
    st.header("📄 OCR")
    st.markdown("Upload an image and extract text using the Llama-4 Scout model")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="ocr_uploader")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.info(f"Image format: {image.format}, Size: {image.size}")
        
        custom_prompt = st.text_area(
            "Enter your prompt for OCR",
            """Transcribe all the text content, including both plain text and tables, from the provided document or image. 
            Maintain the original structure, including headers, paragraphs, and any content preceding or following the 
            table. Format the table in Markdown format, preserving numerical data and relationships. Ensure no text is excluded, 
            including any introductory or explanatory text before or after the table.""",
            key="ocr_prompt"
        )
        
        # Process image button
        if st.button("Extract Text", key="ocr_button"):
            with st.spinner(f"Processing image with {model}..."):
                result = process_with_groq(model, image, custom_prompt)
                
                # Display results
                with col2:
                    st.subheader("Extracted Text")
                    st.markdown(result)
                    
                    # Option to download results
                    result_bytes = result.encode()
                    st.download_button(
                        label="Download Text",
                        data=result_bytes,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )

# Tab 3: RAG
with tab3:
    st.header("📚 Retrieval-Augmented Generation (RAG)")
    st.markdown("Upload documents, then ask questions about their content")
    
    # Document upload section
    st.subheader("Upload Documents")
    uploaded_docs = st.file_uploader(
        "Upload documents (PDF, TXT, CSV)",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        key="rag_uploader"
    )
    
    if uploaded_docs:
        for doc in uploaded_docs:
            if any(d["name"] == doc.name for d in st.session_state.documents):
                st.warning(f"Document '{doc.name}' has already been uploaded.")
                continue
                
            with st.spinner(f"Processing {doc.name}..."):
                chunks, error = process_document(doc)
                if error:
                    st.error(error)
                else:
                    st.success(f"Processed {doc.name} into {len(chunks)} chunks")
    
    # Query section
    st.subheader("Ask Questions About Your Documents")
    
    if not st.session_state.documents:
        st.info("Please upload documents to use RAG functionality.")
    else:
        query = st.text_input("Enter your question:", key="rag_query")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Ask", key="rag_ask_button"):
                if query:
                    with st.spinner("Searching documents and generating answer..."):
                        # Retrieve relevant documents
                        relevant_docs = query_documents(query, top_k=3)
                        
                        # Format context from retrieved documents
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Create prompt with context
                        rag_prompt = f"""Answer the following question based on the provided context. 
                        If you cannot find the answer in the context, say so clearly.
                        
                        Context:
                        {context}
                        
                        Question: {query}"""
                        
                        # Get response from model
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": rag_prompt}
                        ]
                        response = chat_with_groq(model, messages)
                        
                        # Display response
                        st.subheader("Answer")
                        st.markdown(response)
        
        with col2:
            if st.session_state.documents:
                st.subheader("Document Details")
                for i, doc in enumerate(st.session_state.documents):
                    with st.expander(f"{doc['name']}"):
                        st.write(f"Chunks: {doc['chunks']}")
                        st.write(f"Uploaded: {doc['uploaded_at']}")

# Tab 4: Agent
with tab4:
    st.header("🤖 Agentic AI with CrewAI")
    st.markdown("Define tasks for the AI crew to perform collaboratively")
    
    # Introduction to CrewAI
    with st.expander("About CrewAI Framework", expanded=False):
        st.markdown("""
        ### How CrewAI Works
        
        This tab uses the **CrewAI** framework which enables multiple AI agents to collaborate on your task:
        
        1. **Researcher Agent**: Gathers all relevant information needed for the task
        2. **Analyst Agent**: Analyzes the information and extracts key insights
        3. **Writer Agent**: Creates the final deliverable based on research and analysis
        
        The agents work sequentially in a coordinated process, passing their results to each other.
        """)
    
    # Task definition
    st.subheader("Define a Task for the AI Crew")
    agent_task = st.text_area(
        "Describe the task in detail:", 
        height=100,
        key="agent_task",
        help="Describe what you want the AI crew to do. Be specific and provide any necessary information."
    )
    
    # Example tasks
    with st.expander("Example Tasks"):
        st.markdown("""
        - Research and produce a comprehensive report on the latest advancements in quantum computing.
        - Create a detailed marketing strategy for a new eco-friendly product launching next month.
        - Analyze the current market trends in renewable energy and provide investment recommendations.
        - Develop a comprehensive business plan for a startup in the health tech industry.
        - Write a research paper on the impact of artificial intelligence on the future of work.
        """)
    
    # Show advanced options
    with st.expander("Advanced Options"):
        process_type = st.radio(
            "Process Type",
            ["Sequential", "Hierarchical"],
            index=0,
            help="Sequential: Agents work one after another | Hierarchical: Manager agent delegates tasks"
        )
        
        custom_agents = st.multiselect(
            "Select Agents for your Crew",
            ["Researcher", "Analyst", "Writer", "Critic", "Manager"],
            default=["Researcher", "Analyst", "Writer"],
            help="Select which specialist agents you want in your AI crew"
        )
        
        st.caption("Note: The more agents you select, the longer the process may take.")
    
    # Execute task
    if st.button("Execute Task", key="execute_agent"):
        if agent_task:
            with st.spinner("AI Crew is working on your task... This may take a few minutes."):
                try:
                    model_name = "groq/"+model
                    response = execute_agent_task(agent_task, model_name)
                    
                    # Display response
                    st.subheader("Task Result")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error occurred: {str(e)}")
    
    # Task history
    if st.session_state.agent_tasks:
        st.subheader("Task History")
        for i, task in enumerate(reversed(st.session_state.agent_tasks)):
            with st.expander(f"Task {len(st.session_state.agent_tasks) - i}: {task['task'][:50]}..." if len(task['task']) > 50 else f"Task {len(st.session_state.agent_tasks) - i}: {task['task']}"):
                st.caption(f"Executed: {task['timestamp']}")
                st.markdown(task['response'])

# Footer
st.markdown("---")
st.caption("This app uses Llama-4 Scout via Groq API for Chat, OCR, RAG, and Agent tasks. Google's Gemini (free tier) acts as a fallback in CrewAI’s Multi-Agent setup if Llama 4 Scout fails.")