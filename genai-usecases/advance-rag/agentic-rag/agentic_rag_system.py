"""
Agentic RAG Implementation with LangChain and Google's Gemini API

This module implements a state-of-the-art agentic RAG system that combines:
- Advanced multi-agent architecture using LangGraph 
- Google Gemini LLM capabilities
- Web search integration with Tavily
- Adaptive query planning and execution
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import asyncio

# Core LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool

# LangChain Google Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Document processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# LangGraph for agent workflows
from langgraph.graph import StateGraph, START, END

# Web search capabilities
from tavily import TavilyClient

# Pydantic for structured outputs
from pydantic import BaseModel, Field, PrivateAttr

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Enum for query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RESEARCH = "research"

class RetrievalStrategy(Enum):
    """Enhanced retrieval strategies"""
    VECTOR_SIMILARITY = "vector_similarity"
    WEB_AUGMENTED = "web_augmented"
    MULTI_HOP = "multi_hop"

@dataclass
class QueryPlan:
    """Structured query execution plan"""
    original_query: str
    complexity: QueryComplexity
    sub_queries: List[str] = field(default_factory=list)
    requires_web_search: bool = False
    estimated_steps: int = 1
    confidence: float = 0.8

@dataclass
class AgentState:
    """State for the agentic RAG workflow"""
    original_query: str
    query_plan: Optional[QueryPlan] = None
    retrieved_documents: List[Document] = field(default_factory=list)
    web_results: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    execution_log: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)

class QueryAnalysis(BaseModel):
    """Structured query analysis output"""
    complexity: str = Field(description="Query complexity level: simple, moderate, complex, or research")
    main_intent: str = Field(description="Primary intent of the query")
    sub_questions: List[str] = Field(description="List of sub-questions to answer")
    requires_current_info: bool = Field(description="Whether query needs current/recent information")
    domain_expertise: str = Field(description="Domain or field the query belongs to")
    confidence: float = Field(description="Confidence in analysis (0-1)")

class WebSearchTool(BaseTool):
    """Enhanced web search tool using Tavily"""
    name: str = "web_search"
    description: str = "Search the web for current information, recent developments, or facts not in the knowledge base"
    _tavily_client: Optional[TavilyClient] = PrivateAttr(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, tavily_client: Optional[TavilyClient] = None, **kwargs):
        super().__init__(
            name="web_search",
            description="Search the web for current information, recent developments, or facts not in the knowledge base",
            **kwargs
        )
        self._tavily_client = tavily_client or TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    @property
    def tavily_client(self) -> Optional[TavilyClient]:
        return self._tavily_client
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Execute web search and return structured results"""
        try:
            if not self.tavily_client:
                return [{"error": "Tavily API key not configured"}]
            
            results = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=True
            )
            
            processed_results = []
            for result in results.get("results", []):
                processed_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "source": "web_search"
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [{"error": f"Web search failed: {str(e)}"}]

@dataclass
class AgenticRAGConfig:
    """Configuration for the Agentic RAG system"""
    # Model configurations
    llm_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/text-embedding-004"
    temperature: float = 0.1
    max_tokens: int = 8192
    
    # Retrieval configurations
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k_retrieval: int = 8
    
    # Agent configurations
    max_iterations: int = 10
    confidence_threshold: float = 0.7
    
    # Vector store configuration
    persist_directory: str = "./chroma_db_agentic"
    collection_name: str = "agentic_rag_v1"
    
    # Web search configuration
    enable_web_search: bool = True
    max_web_results: int = 5

class AgenticRAGSystem:
    """
    Advanced Agentic RAG System with multi-agent architecture
    """
    
    @staticmethod
    def _ensure_event_loop():
        """Ensure there's an event loop for async operations"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def __init__(self, config: AgenticRAGConfig = None, google_api_key: str = None, tavily_api_key: str = None):
        """
        Initialize the Agentic RAG System
        
        Args:
            config: System configuration
            google_api_key: Google API key for Gemini
            tavily_api_key: Tavily API key for web search
        """
        # Ensure event loop exists
        self._ensure_event_loop()
        
        self.config = config or AgenticRAGConfig()
        
        # Set up API keys
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("Google API key is required")
        
        # Initialize core components
        self._initialize_models()
        self._initialize_tools()
        self._initialize_vector_store()
        self._initialize_workflow()
        
        logger.info("Agentic RAG System initialized successfully")
    
    def _initialize_models(self):
        """Initialize AI models"""
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.google_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model,
            google_api_key=self.google_api_key
        )
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _initialize_tools(self):
        """Initialize tools for the agents"""
        self.tools = []
        
        # Web search tool
        if self.config.enable_web_search and self.tavily_api_key:
            self.web_search_tool = WebSearchTool(
                tavily_client=TavilyClient(api_key=self.tavily_api_key)
            )
            self.tools.append(self.web_search_tool)
        
    
    def _initialize_vector_store(self):
        """Initialize vector store - check if existing store exists"""
        self.vector_store = None
        self.retriever = None
        
        # Try to load existing vector store if it exists
        try:
            from pathlib import Path
            persist_dir = Path(self.config.persist_directory)
            if persist_dir.exists() and any(persist_dir.iterdir()):
                # Existing vector store found, try to load it
                self.vector_store = Chroma(
                    persist_directory=self.config.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.config.collection_name
                )
                
                # Create retriever
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.k_retrieval}
                )
                
                logger.info("Loaded existing vector store")
        except Exception as e:
            logger.info(f"No existing vector store found or failed to load: {e}")
            self.vector_store = None
            self.retriever = None
    
    def _initialize_workflow(self):
        """Initialize the agent workflow using LangGraph"""
        
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("planner", self._planning_agent)
        workflow.add_node("retriever", self._retrieval_agent)
        workflow.add_node("researcher", self._research_agent)
        workflow.add_node("synthesizer", self._synthesis_agent)
        workflow.add_node("validator", self._validation_agent)
        
        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "researcher")
        workflow.add_edge("researcher", "synthesizer")
        workflow.add_edge("synthesizer", "validator")
        workflow.add_edge("validator", END)
        
        # Compile workflow without memory to avoid async issues
        # Memory can be added back if needed with proper async handling
        self.workflow = workflow.compile()
    
    def load_documents(self, file_paths: List[str]) -> bool:
        """
        Load and process documents into the vector store
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            bool: Success status
        """
        try:
            documents = []
            
            for file_path in file_paths:
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path)
                elif file_extension == '.csv':
                    loader = CSVLoader(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_extension}")
                    continue
                
                docs = loader.load()
                # Split documents
                split_docs = self.text_splitter.split_documents(docs)
                documents.extend(split_docs)
            
            if not documents:
                logger.error("No documents could be loaded")
                return False
            
            # Create or update vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.config.persist_directory,
                collection_name=self.config.collection_name
            )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.k_retrieval}
            )
            
            logger.info(f"Successfully loaded {len(documents)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False
    
    def _planning_agent(self, state: AgentState) -> AgentState:
        """Planning agent - analyzes query and creates execution plan"""
        
        logger.info("Planning Agent: Analyzing query and creating execution plan")
        
        # Query analysis prompt
        analysis_prompt = ChatPromptTemplate.from_template("""
        You are an expert query planning agent. Analyze the following query and create a detailed execution plan.
        
        Query: {query}
        
        Analyze the query considering:
        1. Complexity level (simple, moderate, complex, research)
        2. Whether it requires current/recent information
        3. If it needs multi-step reasoning
        4. What retrieval strategies would be most effective
        5. If web search is needed
        
        Provide your analysis in JSON format:
        {{
            "complexity": "simple|moderate|complex|research",
            "main_intent": "description of main intent",
            "sub_questions": ["list", "of", "sub_questions"],
            "requires_current_info": true/false,
            "domain_expertise": "relevant domain",
            "confidence": 0.8
        }}
        """)
        
        try:
            # Parse query using structured output
            parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
            
            prompt = analysis_prompt.format_prompt(query=state.original_query)
            response = self.llm.invoke(prompt.to_messages())
            
            # Try to parse structured output
            try:
                analysis = parser.parse(response.content)
            except:
                # Fallback to basic analysis
                analysis = QueryAnalysis(
                    complexity="moderate",
                    main_intent=state.original_query,
                    sub_questions=[state.original_query],
                    requires_current_info=False,
                    domain_expertise="general",
                    confidence=0.6
                )
            
            # Create query plan
            complexity_map = {
                "simple": QueryComplexity.SIMPLE,
                "moderate": QueryComplexity.MODERATE,
                "complex": QueryComplexity.COMPLEX,
                "research": QueryComplexity.RESEARCH
            }
            
            state.query_plan = QueryPlan(
                original_query=state.original_query,
                complexity=complexity_map.get(analysis.complexity, QueryComplexity.MODERATE),
                sub_queries=analysis.sub_questions,
                requires_web_search=analysis.requires_current_info,
                estimated_steps=len(analysis.sub_questions) + (1 if analysis.requires_current_info else 0),
                confidence=analysis.confidence
            )
            
            state.execution_log.append(f"Created execution plan with {len(analysis.sub_questions)} sub-queries")
            logger.info(f"Query plan created: {state.query_plan.complexity.value} complexity")
            
        except Exception as e:
            logger.error(f"Planning agent error: {e}")
            # Fallback plan
            state.query_plan = QueryPlan(
                original_query=state.original_query,
                complexity=QueryComplexity.MODERATE,
                sub_queries=[state.original_query],
                confidence=0.5
            )
        
        return state
    
    def _retrieval_agent(self, state: AgentState) -> AgentState:
        """Retrieval agent - retrieves relevant documents from vector store"""
        
        logger.info("Retrieval Agent: Retrieving relevant documents")
        
        if not self.retriever:
            state.execution_log.append("No vector store available for retrieval")
            return state
        
        try:
            # Retrieve for main query and sub-queries
            all_queries = [state.original_query] + (state.query_plan.sub_queries if state.query_plan else [])
            
            retrieved_docs = []
            for query in all_queries:
                docs = self.retriever.invoke(query)
                retrieved_docs.extend(docs)
            
            # Remove duplicates based on content
            unique_docs = []
            seen_content = set()
            
            for doc in retrieved_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
            
            state.retrieved_documents = unique_docs[:self.config.k_retrieval]
            state.execution_log.append(f"Retrieved {len(state.retrieved_documents)} unique documents")
            
            # Add sources
            for doc in state.retrieved_documents:
                source_info = {
                    "type": "document",
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                if source_info not in state.sources:
                    state.sources.append(source_info)
            
        except Exception as e:
            logger.error(f"Retrieval agent error: {e}")
            state.execution_log.append(f"Retrieval failed: {str(e)}")
        
        return state
    
    def _research_agent(self, state: AgentState) -> AgentState:
        """Research agent - performs web search if needed"""
        
        logger.info("Research Agent: Performing additional research")
        
        # Check if web search is needed
        if not (state.query_plan and state.query_plan.requires_web_search and self.config.enable_web_search):
            state.execution_log.append("Web search not required or not available")
            return state
        
        try:
            web_results = []
            queries_to_search = [state.original_query]
            
            if state.query_plan.sub_queries:
                queries_to_search.extend(state.query_plan.sub_queries[:3])  # Limit web searches
            
            for query in queries_to_search:
                results = self.web_search_tool._run(query, max_results=3)
                web_results.extend(results)
            
            state.web_results = web_results
            state.execution_log.append(f"Retrieved {len(web_results)} web results")
            
            # Add web sources
            for result in web_results:
                if "error" not in result:
                    source_info = {
                        "type": "web",
                        "source": result.get("url", "unknown"),
                        "title": result.get("title", ""),
                        "content_preview": result.get("content", "")[:200] + "..." if result.get("content", "") else ""
                    }
                    state.sources.append(source_info)
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            state.execution_log.append(f"Web research failed: {str(e)}")
        
        return state
    
    def _synthesis_agent(self, state: AgentState) -> AgentState:
        """Synthesis agent - combines all information to generate final answer"""
        
        logger.info("Synthesis Agent: Generating comprehensive answer")
        
        # Prepare context from all sources
        doc_context = "\n\n".join([
            f"Document: {doc.page_content}" for doc in state.retrieved_documents
        ])
        
        web_context = "\n\n".join([
            f"Web Source ({result.get('title', 'Unknown')}): {result.get('content', '')}"
            for result in state.web_results if "error" not in result
        ])
        
        # Synthesis prompt
        synthesis_prompt = ChatPromptTemplate.from_template("""
        You are an expert information synthesis agent. Your task is to provide a comprehensive, accurate, and well-structured answer to the user's query using all available context.

        Original Query: {query}

        Document Context:
        {doc_context}

        Web Context (if available):
        {web_context}

        Instructions:
        1. Provide a direct, comprehensive answer to the query
        2. Synthesize information from multiple sources
        3. Cite specific sources when making claims
        4. If there are conflicting information, note the discrepancies
        5. Be clear about what information comes from documents vs. web sources
        6. If insufficient information is available, state this clearly
        7. Structure your response logically with clear sections if appropriate

        Generate a well-structured, informative response:
        """)
        
        try:
            formatted_prompt = synthesis_prompt.invoke({
                "query": state.original_query,
                "doc_context": doc_context or "No document context available",
                "web_context": web_context or "No web context available"
            })
            
            response = self.llm.invoke(formatted_prompt)
            state.final_answer = response.content
            
            # Calculate confidence based on available information
            doc_score = min(len(state.retrieved_documents) / self.config.k_retrieval, 1.0) * 0.6
            web_score = min(len([r for r in state.web_results if "error" not in r]) / 3, 1.0) * 0.4
            state.confidence_score = doc_score + web_score
            
            state.execution_log.append("Generated comprehensive answer")
            
        except Exception as e:
            logger.error(f"Synthesis agent error: {e}")
            state.final_answer = f"I encountered an error while generating the response: {str(e)}"
            state.confidence_score = 0.1
        
        return state
    
    def _validation_agent(self, state: AgentState) -> AgentState:
        """Validation agent - validates and refines the final answer"""
        
        logger.info("Validation Agent: Validating answer quality")
        
        # Simple validation based on answer length and confidence
        if len(state.final_answer) < 50:
            state.execution_log.append("Warning: Answer seems too short")
            state.confidence_score *= 0.8
        
        if not state.retrieved_documents and not state.web_results:
            state.execution_log.append("Warning: No supporting evidence found")
            state.confidence_score *= 0.5
        
        # Validation passed
        state.execution_log.append(f"Validation complete. Final confidence: {state.confidence_score:.2f}")
        
        return state
    
    def query(self, question: str, thread_id: str = None) -> Dict[str, Any]:
        """
        Process a query through the agentic RAG system
        
        Args:
            question: User's question
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dictionary with comprehensive results
        """
        logger.info(f"Processing query: {question}")
        
        # Create initial state
        initial_state = AgentState(original_query=question)
        
        # Generate thread ID if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        try:
            # Ensure event loop exists before executing workflow
            self._ensure_event_loop()
            
            # Execute the workflow synchronously
            final_state = self.workflow.invoke(initial_state)
            
            # Prepare comprehensive response
            response = {
                "answer": final_state["final_answer"],
                "confidence": final_state["confidence_score"],
                "sources": final_state["sources"],
                "retrieved_documents": len(final_state["retrieved_documents"]),
                "web_results": len(final_state["web_results"]),
                "execution_log": final_state["execution_log"],
                "query_plan": {
                    "complexity": final_state["query_plan"].complexity.value if final_state["query_plan"] else "unknown",
                    "sub_queries": final_state["query_plan"].sub_queries if final_state["query_plan"] else [],
                    "estimated_steps": final_state["query_plan"].estimated_steps if final_state["query_plan"] else 1
                } if final_state["query_plan"] else None,
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Query processed successfully. Confidence: {response['confidence']:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "vector_store_initialized": self.vector_store is not None,
            "retriever_available": self.retriever is not None,
            "web_search_enabled": self.config.enable_web_search and self.tavily_api_key is not None,
            "model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "tools_available": len(self.tools),
            "configuration": {
                "max_iterations": self.config.max_iterations,
                "confidence_threshold": self.config.confidence_threshold,
                "k_retrieval": self.config.k_retrieval,
                "chunk_size": self.config.chunk_size
            }
        }