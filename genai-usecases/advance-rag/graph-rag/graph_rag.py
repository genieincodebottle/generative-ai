import os
from dotenv import load_dotenv

from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_graph_retriever import GraphRetriever
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from graph_retriever.strategies import Eager
from graph_rag_example_helpers.datasets.animals import fetch_documents
from agentic_router import create_agentic_router

load_dotenv()

if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file or environment.")

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

class RetrieverType(Enum):
    TRAVERSAL = "traversal"
    STANDARD = "standard"
    HYBRID = "hybrid"

@dataclass
class GraphRAGConfig:
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "gemini-2.0-flash"
    llm_provider: str = "google_genai"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k_retrieval: int = 5
    max_depth: int = 2

class DocumentProcessor:
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def load_documents_from_files(self, files: List[str]) -> List[Document]:
        documents = []
        for file_path in files:
            try:
                documents.extend(self._load_single_file(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return documents
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

class GraphRAGSystem:
    def __init__(self, config: GraphRAGConfig = None):
        self.config = config or GraphRAGConfig()
        self.embeddings = None
        self.vector_store = None
        self.traversal_retriever = None
        self.standard_retriever = None
        self.llm = None
        self.agentic_router = None
        self.document_processor = DocumentProcessor(self.config)
        self._initialize_components()
    
    def _initialize_components(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self.llm = init_chat_model(
            self.config.llm_model, 
            model_provider=self.config.llm_provider
        )
    
    def initialize_with_default_data(self):
        try:
            animals = fetch_documents()
            self._create_vector_store_and_retrievers(animals)
            return True
        except Exception as e:
            print(f"Error loading default data: {e}")
            return False
    
    def initialize_with_documents(self, documents: List[Document]):
        if not documents:
            raise ValueError("No documents provided")
        self._create_vector_store_and_retrievers(documents)
    
    def initialize_with_files(self, file_paths: List[str]):
        documents = self.document_processor.load_documents_from_files(file_paths)
        if not documents:
            raise ValueError("No documents could be loaded from files")
        self._create_vector_store_and_retrievers(documents)
    
    def initialize_with_text(self, text: str):
        doc = Document(page_content=text, metadata={"source": "text_input"})
        self._create_vector_store_and_retrievers([doc])
    
    def _create_vector_store_and_retrievers(self, documents: List[Document]):
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        
        # Auto-detect edges based on content and metadata
        edges = self._detect_edges(documents)
        print(f"Detected edges: {edges}")
        self.traversal_retriever = GraphRetriever(
            store=self.vector_store,
            edges=edges,
            strategy=Eager(k=self.config.k_retrieval, start_k=1, max_depth=self.config.max_depth),
        )
        
        self.standard_retriever = GraphRetriever(
            store=self.vector_store,
            edges=edges,
            strategy=Eager(k=self.config.k_retrieval, start_k=self.config.k_retrieval, max_depth=0),
        )
        
        # Initialize agentic router
        self.agentic_router = create_agentic_router(
            llm=self.llm,
            traversal_retriever=self.traversal_retriever,
            standard_retriever=self.standard_retriever
        )
        print("Agentic router initialized successfully")
    
    def _detect_edges(self, documents: List[Document]) -> List[tuple]:
        """Auto-detect graph edges based on document content and metadata"""
        edges = []
        
        # Debug: Print sample metadata to understand structure
        print(f"Sample document metadata: {documents[0].metadata if documents else 'None'}")
        
        # Check if this is the default animal dataset - look for specific animal dataset indicators
        sample_metadata = [str(doc.metadata).lower() for doc in documents[:5]]
        sample_content = [doc.page_content.lower() for doc in documents[:5]]
        
        # Look for animal-specific fields or content
        animal_indicators = ["habitat", "origin", "species", "mammal", "bird", "reptile", "capybara", "elephant"]
        
        if (any("habitat" in meta for meta in sample_metadata) or 
            any("origin" in meta for meta in sample_metadata) or
            any(indicator in " ".join(sample_content) for indicator in animal_indicators)):
            print("Detected animal dataset - using animal-specific edges")
            return [("habitat", "habitat"), ("origin", "origin"), ("category", "category")]
        
        # Extract common metadata keys
        metadata_keys = set()
        for doc in documents[:10]:
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata_keys.update(doc.metadata.keys())
        
        # Common relationship patterns
        relationship_patterns = {
            'source': 'source',
            'author': 'author', 
            'category': 'category',
            'topic': 'topic',
            'type': 'type',
            'section': 'section',
            'chapter': 'chapter',
            'department': 'department',
            'location': 'location',
            'date': 'date',
            'tag': 'tag',
            'subject': 'subject'
        }
        
        # Add edges for matching metadata keys
        for key in metadata_keys:
            key_lower = key.lower()
            for pattern, edge_name in relationship_patterns.items():
                if pattern in key_lower:
                    edges.append((key, key))
                    break
        
        # Content-based edges
        sample_text = " ".join([doc.page_content[:500] for doc in documents[:5]]).lower()
        
        content_patterns = {
            'person': ['person', 'people', 'individual', 'name', 'author'],
            'organization': ['company', 'organization', 'corp', 'university'],
            'location': ['city', 'country', 'state', 'region', 'place'],
            'technology': ['software', 'system', 'platform', 'tool'],
            'concept': ['concept', 'idea', 'theory', 'method']
        }
        
        for pattern_name, keywords in content_patterns.items():
            if any(keyword in sample_text for keyword in keywords):
                edges.append((pattern_name, pattern_name))
        
        # Default fallback
        if not edges:
            edges = [("source", "source"), ("content_type", "content_type")]
            
        return edges
    
    def get_detected_relationships(self) -> List[str]:
        """Get list of detected relationships for display"""
        if not self.vector_store or not self.traversal_retriever:
            return []
            
        if hasattr(self.traversal_retriever, 'edges'):
            return [f"{edge[0]} ↔ {edge[1]}" for edge in self.traversal_retriever.edges]
        return []
    
    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(f"Content: {doc.page_content}\nMetadata: {doc.metadata}" for doc in docs)
    
    def query(self, question: str, retriever_type: RetrieverType = RetrieverType.TRAVERSAL, return_details: bool = False) -> str | Dict[str, Any]:
        print(f"Starting query: {question}")
        
        if not self.vector_store:
            raise ValueError("System not initialized. Please load documents first.")
        
        try:
            routing_info = None  # Initialize routing_info
            
            # Select retriever
            if retriever_type == RetrieverType.TRAVERSAL:
                retriever = self.traversal_retriever
                print("Using traversal retriever")
            elif retriever_type == RetrieverType.STANDARD:
                retriever = self.standard_retriever
                print("Using standard retriever")
            else:  # HYBRID - agentic routing
                retriever, routing_info = self._get_agentic_retriever(question)
                print(f"Using agentic router: {routing_info['strategy']} (confidence: {routing_info['confidence']:.2f})")
                print(f"Reasoning: {routing_info['reasoning']}")
            
            if not retriever:
                raise ValueError("Retriever not initialized properly")
            
            print("Retrieving documents...")
            # First test retrieval directly
            docs = retriever.invoke(question)
            print(f"Retrieved {len(docs)} documents")
            
            if not docs:
                return "No relevant documents found for your question."
            
            # Format documents
            context = self.format_docs(docs)
            print(f"Context length: {len(context)} characters")
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the context provided. Be comprehensive and accurate.

                Context: {context}

                Question: {question}
                
                Answer:"""
            )
            
            print("Generating response with LLM...")
            # Generate response
            formatted_prompt = prompt.invoke({"context": context, "question": question})
            response = self.llm.invoke(formatted_prompt)
            
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            print("Query completed successfully")
            
            if return_details:
                # Return detailed query information
                query_details = {
                    "answer": result,
                    "retrieved_documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source": doc.metadata.get("source", "Unknown"),
                            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        } for doc in docs
                    ],
                    "retrieval_info": {
                        "retriever_type": retriever_type.value,
                        "num_documents": len(docs),
                        "context_length": len(context)
                    },
                    "question": question
                }
                
                # Add routing info if hybrid was used
                if retriever_type == RetrieverType.HYBRID:
                    query_details["routing_info"] = routing_info
                    
                return query_details
            else:
                return result
            
        except Exception as e:
            print(f"Error in query: {e}")
            return f"Error generating response: {str(e)}"
    
    def _get_agentic_retriever(self, question: str) -> tuple[BaseRetriever, Dict[str, Any]]:
        """Use agentic router for intelligent retrieval strategy selection"""
        if not self.agentic_router:
            raise ValueError("Agentic router not initialized. Please load documents first.")
        
        return self.agentic_router.route(question)
    
    def get_routing_explanation(self, question: str) -> Dict[str, Any]:
        """Get detailed explanation of routing decision for a question"""
        if not self.agentic_router:
            raise ValueError("Agentic router not initialized. Please load documents first.")
        
        return self.agentic_router.get_routing_explanation(question)
