# Agentic RAG System


![Agentic RAG](https://img.shields.io/badge/Agentic-RAG-blue)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/downloads/)
![Streamlit](https://img.shields.io/badge/Streamlit-UI_Framework-ff4b4b)
![Gemini LLM](https://img.shields.io/badge/Gemini-LLM_&_Embedding_Model-00bfa5)
[![LangChain](https://img.shields.io/badge/LangChain-AI_Framework-0e76a8.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Framework-f39c12.svg)](https://langchain.com/langgraph)
![ChromaDB](https://img.shields.io/badge/Chroma-Vector_DB-9b59b6)


![Agentic RAG](./images/agentic-rag.png)

An **Agentic Retrieval-Augmented Generation (RAG)** system built with LangChain, LangGraph, and Google's Gemini LLM. This system implements advanced multi-agent workflows for intelligent question answering with adaptive reasoning strategies.

## ✨ Key Features

### 🧠 Multi-Agent Architecture
- **Planner Agent**: Analyzes queries and creates intelligent execution plans
- **Retriever Agent**: Performs semantic document retrieval from vector stores
- **Research Agent**: Conducts web searches for current information
- **Synthesizer Agent**: Combines information from multiple sources
- **Validator Agent**: Validates and refines final answers

### 🔮 Advanced Capabilities
- **Adaptive Query Planning**: Automatically detects query complexity and selects optimal strategies
- **Multi-Modal Processing**: Handles text, PDFs, and CSV document formats
- **Web-Augmented RAG**: Combines document knowledge with real-time web search
- **Confidence Scoring**: Provides transparency in answer reliability
- **Source Tracking**: Detailed attribution of information sources
- **Interactive Web Interface**: Professional tabbed interface with real-time updates

### 🔧 Tech Stack

- **LangGraph**: State-of-the-art agent workflow orchestration
- **Gemini LLM API**: Google's AI models (2.0-flash, 2.0-pro, 2.5-pro, 2.5-flash)
- **Tavily Search**: Advanced web search integration
- **ChromaDB**: High-performance vector database
- **Streamlit**: Interactive web interface with tabbed navigation

## ⚡ Quick Start

## 📋 Installation

   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\advance-rag\agentic-rag
      ```
   2. Open the Project in VS Code or any code editor.
   3. Create a virtual environment by running the following command in the terminal:
   
      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. Install dependencies:
      
      ```bash
      uv pip install -r requirements.txt
      ```
   5. Configure Environment
      * Rename .env.example → .env
      * Update with your keys:

        ```bash
        GOOGLE_API_KEY=your_key_here # Using the free-tier API Key
        TAVILY_API_KEY=your_key_here # Optional- For Web Search
        ```
      * Get **GOOGLE_API_KEY** here -> https://aistudio.google.com/app/apikey
      * Get **TAVILY_API_KEY** here (Optional- For Web Search)-> https://tavily.com/home

   6. Run the Application

      ```bash
      streamlit run streamlit_app.py
      ```

## 🎨 Streamlit Interface

The web interface is organized into **5 intuitive tabs**:

### 📁 Document Management
- Upload PDF, TXT, and CSV documents
- Process documents into vector store
- View vector store status
- Automatic existing vector store detection

### 💬 Ask Questions
- Natural language query input
- Real-time answer generation
- Confidence scoring with visual gauge
- Performance metrics display
- New conversation threading

### 🧠 Query Analysis
- Query complexity analysis (Simple, Moderate, Complex, Research)
- Sub-questions breakdown
- Execution steps and logs
- Strategy selection details

### 📋 Sources & History
- Document sources with previews
- Web search results and links
- Complete query history table
- Source citations and attributions

### 🔄 System Overview
- Agent workflow visualization
- Current system configuration
- Model and parameter settings
- Architecture overview

## 🏗️ System Architecture

![system architecture](./images/agentic-graph-system-architecture.png)

## 📊 Query Processing Flow

1. **Query Analysis**: The system analyzes query complexity and requirements
2. **Strategy Selection**: Chooses optimal retrieval and reasoning strategies
3. **Document Retrieval**: Searches vector store for relevant information
4. **Web Research**: Fetches current information if needed
5. **Information Synthesis**: Combines all sources into coherent answer
6. **Validation**: Ensures answer quality and confidence scoring

## 🎯 Query Complexity Levels

- **Simple**: Direct factual questions with straightforward answers
- **Moderate**: Questions requiring analysis or comparison
- **Complex**: Multi-faceted queries needing synthesis from multiple sources
- **Research**: Queries requiring current information and web search

## 📁 File Structure

```
agentic-rag/
├── agentic_rag_system.py     # Core system implementation
├── streamlit_app.py          # Web interface
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
└── README.md               # This file
```

## ⚙️ Configuration Options

```python
config = AgenticRAGConfig(
    # Model settings
    llm_model="gemini-2.0-flash",  # Available: gemini-2.0-flash, gemini-2.0-pro, gemini-2.5-pro, gemini-2.5-flash
    temperature=0.1,
    max_tokens=8192,
    
    # Retrieval settings
    k_retrieval=8,
    chunk_size=1000,
    chunk_overlap=200,
    
    # Agent settings
    max_iterations=10,
    confidence_threshold=0.7,
    
    # Features
    enable_web_search=True,
    max_web_results=5
)
```

## 🔍 Example Queries

Try these sample queries to explore the system's capabilities:

### Simple Queries
- "What are the main renewable energy sources mentioned?"
- "Define artificial intelligence."

### Moderate Queries
- "How do renewable energy solutions impact economic growth?"
- "Compare machine learning and deep learning approaches."

### Complex Queries
- "Analyze the relationship between AI development and climate change solutions."
- "What are the synergies between renewable energy and AI technologies?"

### Research Queries (requires web search)
- "What are the latest breakthroughs in AI safety research in 2025?"
- "Recent developments in quantum computing applications."

## 🛠️ Advanced Usage

### Retrieval Strategies
The system automatically selects optimal retrieval strategies:

- **Vector Similarity**: Standard semantic search
- **Web Augmented**: Document + web search combination
- **Multi-Hop**: Sequential reasoning across multiple sources

## 🔧 Troubleshooting

### Common Issues

**1. Google API Key Error**
```
Error: GOOGLE_API_KEY environment variable not set
```
Solution: Ensure your Google API key is properly set in the `.env` file

**2. Web Search Not Working**
```
Warning: Tavily API key not configured
```
Solution: Add your Tavily API key to enable web search capabilities

**3. Document Loading Fails**
```
Error: No documents could be loaded from files
```
Solution: Ensure document files exist and are in supported formats (PDF, TXT, CSV)

**4. Memory Issues with Large Documents**
```
ChromaDB memory error
```
Solution: Reduce `chunk_size` in configuration or process documents in batches

**5. Event Loop Error**
```
There is no current event loop in thread
```
Solution: The system automatically handles this with proper async initialization

### Performance Optimization

- **Reduce `k_retrieval`** for faster queries with less context
- **Disable web search** for document-only queries
- **Use smaller models** like `gemini-2.0-flash` for better speed
- **Adjust `chunk_size`** based on document complexity

## 📦 Dependencies

All dependencies are automatically handled through `requirements.txt`:

- **Core**: LangChain, LangGraph, Pydantic
- **Models**: Google Gemini AI integration
- **Vector Store**: ChromaDB with persistent storage
- **Web Search**: Tavily Python client
- **Interface**: Streamlit with Plotly visualizations
- **Document Processing**: PyPDF for PDF files
