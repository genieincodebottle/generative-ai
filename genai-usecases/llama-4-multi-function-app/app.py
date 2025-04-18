import torch
# To override warning related torch path reload when using streamlit 
torch.classes.__path__ = []

# Standard library imports
import base64
import io
import os
import tempfile
from datetime import datetime

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
from langchain_huggingface import HuggingFaceEmbeddings

# CrewAI components
from crewai import Agent, Task, Crew, Process

# UI
import streamlit as st

# Import evaluation functions
from rag_evaluation import evaluate_with_gemini, evaluate_single_query, create_evaluation_chart

# Load environment variables from .env file
load_dotenv()

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
    img_format = image.format if hasattr(image, 'format') and image.format else "PNG"
    
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
    
    # Determine media type based on image format
    media_type = "image/jpeg" if img_format in ['jpg', 'jpeg'] else "image/png"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
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
        
        # Create embeddings
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
        return "Gemini API key not set in environment variables. Please add GOOGLE_API_KEY to your .env file."
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
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        return f"Error using Gemini fallback: {str(e)}"
    
def execute_agent_task(task_description, model_name, process_type="Sequential", custom_agents=None):
    """Execute a task using CrewAI framework with fallback to Gemini model if Llama fails
    
    Args:
        task_description (str): The task to be performed
        model_name (str): The LLM model to use
        process_type (str): "Sequential" or "Hierarchical" process type
        custom_agents (list): List of agent roles to include
    """
    try:
        # Set default agents if none provided
        if not custom_agents:
            custom_agents = ["Researcher", "Analyst", "Writer"]
        
        # Create LLM instance for agents
        llm = create_groq_llm(model_name)
        
        agents = []
        tasks = []
        
        # Create selected agents based on custom_agents list
        if "Researcher" in custom_agents:
            researcher = Agent(
                role="Researcher",
                goal="Find and gather all relevant information for the task",
                backstory="You are an expert researcher with keen attention to detail and a talent for finding relevant information.",
                verbose=True,
                allow_delegation=True,
                llm=llm
            )
            agents.append(researcher)
            
            research_task = Task(
                description=f"Research task: {task_description}\n\nFind and gather all relevant information needed to complete this task.",
                agent=researcher,
                expected_output="Comprehensive research findings"
            )
            tasks.append(research_task)
        
        if "Analyst" in custom_agents:
            analyst = Agent(
                role="Analyst",
                goal="Analyze the information and extract insights",
                backstory="You are a skilled analyst with a talent for identifying patterns and deriving valuable insights from data.",
                verbose=True,
                allow_delegation=True,
                llm=llm
            )
            agents.append(analyst)
            
            analysis_task = Task(
                description=f"Analysis task: After reviewing the research, perform an in-depth analysis to extract key insights for: {task_description}",
                agent=analyst,
                expected_output="Analytical report with key insights",
                context=[t for t in tasks]  # Use all previous tasks as context
            )
            tasks.append(analysis_task)
        
        if "Writer" in custom_agents:
            writer = Agent(
                role="Writer",
                goal="Create well-written, clear, and comprehensive content",
                backstory="You are an accomplished writer with a talent for creating engaging and informative content.",
                verbose=True,
                allow_delegation=True,
                llm=llm
            )
            agents.append(writer)
            
            writing_task = Task(
                description=f"Writing task: Create a final deliverable for: {task_description}\n\nBased on the research and analysis, create a well-structured and comprehensive final output.",
                agent=writer,
                expected_output="Final deliverable",
                context=[t for t in tasks]  # Use all previous tasks as context
            )
            tasks.append(writing_task)
        
        if "Critic" in custom_agents:
            critic = Agent(
                role="Critic",
                goal="Evaluate and improve the quality of the final deliverable",
                backstory="You are a meticulous critic with an eye for detail and a commitment to excellence.",
                verbose=True,
                allow_delegation=True,
                llm=llm
            )
            agents.append(critic)
            
            critique_task = Task(
                description=f"Critique task: Review the final deliverable for: {task_description}\n\nProvide constructive criticism and suggestions for improvement.",
                agent=critic,
                expected_output="Critique with specific improvements",
                context=[t for t in tasks]  # Use all previous tasks as context
            )
            tasks.append(critique_task)
        
        if "Manager" in custom_agents:
            manager = Agent(
                role="Manager",
                goal="Coordinate the work of the team and ensure high-quality output",
                backstory="You are an experienced project manager skilled at coordinating complex tasks and ensuring successful outcomes.",
                verbose=True,
                allow_delegation=True,
                llm=llm
            )
            agents.append(manager)
            
            # In hierarchical process, manager is typically the first agent
            if process_type == "Hierarchical":
                management_task = Task(
                    description=f"Management task: Coordinate the completion of: {task_description}\n\nDelegate subtasks as needed and compile final results.",
                    agent=manager,
                    expected_output="Complete project with coordinated results"
                )
                # For hierarchical, we can just use this one task
                tasks = [management_task]
        
        # Set process type based on selection
        process = Process.sequential if process_type == "Sequential" else Process.hierarchical
        
        # Create the crew with the process
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=process
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "process_type": process_type,
            "agents_used": custom_agents
        })
        
        return result
    
    except Exception as e:
        st.warning(f"Primary model error: {str(e)}. Attempting fallback to Gemini model...")
        
        # Fallback to Gemini
        try:
            fallback_result = fallback_to_gemini(task_description)
            response = f"[FALLBACK TO GEMINI MODEL DUE TO ERROR: {str(e)}]\n\n{fallback_result}"
            
            # Add to agent tasks history
            st.session_state.agent_tasks.append({
                "task": task_description,
                "response": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "process_type": process_type,
                "agents_used": custom_agents
            })
            
            return response
        except Exception as gemini_error:
            # If both models fail, return combined error
            final_error = f"Both models failed.\nPrimary error: {str(e)}\nFallback error: {str(gemini_error)}"
            
            # Add error to history
            st.session_state.agent_tasks.append({
                "task": task_description,
                "response": final_error,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "process_type": process_type,
                "agents_used": custom_agents
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
       Using free-tier API for dev with rate limits
       
       - For Groq API Key follow -> [Link](https://console.groq.com/keys)
                
       - List of Groq models -> [Link](https://console.groq.com/docs/models)
       
       - For Google API Key follow -> [Link](https://aistudio.google.com/app/apikey)
    
       - List of Gemini models -> [Link](https://ai.google.dev/gemini-api/docs/models)
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
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 OCR", "📚 RAG with Evaluation", "🤖 Agentic AI"])

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

# Tab 2: OCR
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

# Tab 3: RAG with Gemini & RAGAS evaluation
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
        
        # RAG configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("Number of chunks to retrieve", 1, 5, 3, 
                           help="Number of most relevant document chunks to use")
            
            use_ground_truth = st.checkbox("Add Ground Truth", value=False,
                                        help="Provide a ground truth answer for reference")
        
        with col2:
            enable_evaluation = st.checkbox("Enable Evaluation", value=True, 
                                         help="Evaluate the quality of RAG response")
            
            if enable_evaluation:
                eval_method = st.radio(
                    "Evaluation Method",
                    options=["Gemini as Evaluator", "RAGAs"],
                    index=0,
                    help="Choose between Gemini (lightweight) or RAGAs (comprehensive) evaluation"
                )
                
                if eval_method == "RAGAs":
                    reuse_same_model = st.checkbox(
                        "Use same model for evaluation", 
                        value=False,
                        help="Use the same model for both generation and evaluation; otherwise, RAGA defaults to OpenAI for evaluation, which requires setting the OPENAI_API_KEY."
                    )
        
        # Optional ground truth input
        ground_truth = st.text_area("Ground Truth Answer (Optional)", height=100) if use_ground_truth else None
        
        if st.button("Ask", key="rag_ask_button"):
            if query:
                with st.spinner("Searching documents and generating answer..."):
                    # Retrieve relevant documents
                    relevant_docs = query_documents(query, top_k=top_k)
                    
                    if not relevant_docs:
                        st.warning("No relevant documents found. Try modifying your query or upload more documents.")
                    else:
                        # Format contexts from retrieved documents
                        contexts = [doc.page_content for doc in relevant_docs]
                        context_text = "\n\n".join(contexts)
                        
                        # Create prompt with context
                        rag_prompt = f"""Answer the following question based on the provided context. 
                        If you cannot find the answer in the context, say so clearly.
                        
                        Context:
                        {context_text}
                        
                        Question: {query}"""
                        
                        # Get response from model
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": rag_prompt}
                        ]
                        response = chat_with_groq(model, messages)
                        
                        # Create tabs for different sections
                        result_tabs = st.tabs(["Answer", "Evaluation", "Retrieved Context"])
                        
                        # Answer Tab
                        with result_tabs[0]:
                            st.markdown("### Answer")
                            st.markdown(response)
                            
                            if use_ground_truth and ground_truth:
                                st.markdown("### Ground Truth")
                                st.info(ground_truth)
                        
                        # Evaluation Tab
                        with result_tabs[1]:
                            if enable_evaluation:
                                with st.spinner(f"Running {eval_method} evaluation..."):
                                    try:
                                        # Run the chosen evaluation method
                                        if eval_method == "Gemini as Evaluator":
                                            eval_results = evaluate_with_gemini(
                                                query=query,
                                                answer=response,
                                                contexts=contexts,
                                                ground_truth=ground_truth if use_ground_truth else None
                                            )
                                        else:  # RAGAs
                                            eval_results = evaluate_single_query(
                                                model_provider="groq",
                                                model_name=model,
                                                question=query,
                                                answer=response,
                                                contexts=contexts,
                                                resuse_same_model_for_eval=reuse_same_model,
                                                ground_truth=ground_truth if use_ground_truth else None,
                                            )
                                        
                                        # Display metrics
                                        st.markdown("### Evaluation Metrics")
                                        
                                        # Create metrics display
                                        num_metrics = len(eval_results)
                                        metric_cols = st.columns(min(num_metrics, 4))  # Max 4 columns
                                        
                                        # Display individual metrics
                                        metric_labels = {
                                            "answer_relevancy": "Answer Relevancy",
                                            "faithfulness": "Faithfulness",
                                            "context_precision": "Context Precision",
                                            "context_recall": "Context Recall"
                                        }
                                        
                                        for i, (metric, score) in enumerate(eval_results.items()):
                                            col_index = i % min(num_metrics, 4)
                                            with metric_cols[col_index]:
                                                st.metric(
                                                    metric_labels.get(metric, metric),
                                                    f"{score:.2f}"
                                                )
                                        
                                        # Create visualization
                                        chart = create_evaluation_chart(eval_results)
                                        st.plotly_chart(chart, use_container_width=True)
                                        
                                        # Explanation of metrics
                                        with st.expander(f"Understanding {eval_method} Evaluation Metrics"):
                                            if eval_method == "Gemini as Evaluator":
                                                st.markdown("""
                                                **Answer Relevancy (0-1)**: Measures how directly the answer addresses the question.
                                                
                                                **Faithfulness (0-1)**: Measures how factually accurate the answer is based only on the provided context.
                                                
                                                **Context Precision (0-1)**: When ground truth is provided, measures how well the answer aligns with the known correct answer.
                                                """)
                                            else:  # RAGAS
                                                st.markdown("""
                                                **Answer Relevancy (0-1)**: RAGAS metric for how well the answer addresses the specific question asked.
                                                
                                                **Faithfulness (0-1)**: RAGAS metric for factual consistency between the answer and context.
                                                
                                                **Context Precision (0-1)**: When ground truth is provided, RAGAS metric for precision of the retrieved context relative to the ground truth.
                                                """)
                                    except Exception as e:
                                        st.error(f"{eval_method} evaluation failed: {str(e)}")
                                        
                                        if eval_method == "RAGAs":
                                            st.error("Make sure you have RAGAS installed: pip install ragas datasets")
                                            st.info("Consider trying the Gemini evaluation method instead.")
                                        elif eval_method == "Gemini as Evaluator":
                                            st.error("Make sure you have the Google API library installed: pip install google-generativeai")
                                            st.info("Also check that your GOOGLE_API_KEY is set in the environment variables.")
                            else:
                                st.info("Enable the evaluation checkbox to see quality metrics for this response.")
                                
                        # Context Tab
                        with result_tabs[2]:
                            st.markdown("### Retrieved Document Chunks")
                            
                            for i, doc in enumerate(relevant_docs):
                                with st.expander(f"Context Chunk {i+1}"):
                                    st.markdown(doc.page_content)
                                    
                                    # Show metadata if available
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        st.markdown("**Source:**")
                                        for key, value in doc.metadata.items():
                                            if key != "page_content":  # Skip content itself
                                                st.caption(f"**{key}**: {value}")

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
                    
                    # Convert process_type to proper format for function
                    process_type_value = process_type if 'process_type' in locals() else "Sequential"
                    
                    # Get custom_agents if defined, otherwise use default
                    selected_agents = custom_agents if 'custom_agents' in locals() else ["Researcher", "Analyst", "Writer"]
                    
                    # Pass the advanced settings to the execute_agent_task function
                    response = execute_agent_task(
                        agent_task, 
                        model_name, 
                        process_type=process_type_value, 
                        custom_agents=selected_agents
                    )
                    
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
st.caption("This app uses Llama-4 Scout via Groq API for Chat, OCR, RAG, and Agent tasks. Google's Gemini (limited free tier) acts as a fallback in CrewAI's Multi-Agent setup if Llama 4 Scout fails.")