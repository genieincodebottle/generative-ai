import os
import base64
from typing import List, Dict, Any
import uuid
import hashlib

# LangChain imports for 2025 updates
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

# Document parsing (using PyPDF and PyMuPDF for image extraction)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. PDF image extraction disabled. Install with: pip install PyMuPDF")

# Google AI imports
import google.generativeai as genai

class MultimodalRAGSystem:
  
    def __init__(self, google_api_key: str, persist_directory: str = "./chroma_db_v2", 
                 main_model: str = "gemini-1.5-pro", vision_model: str = "gemini-1.5-flash", 
                 embedding_model: str = "models/text-embedding-004", temperature: float = 0.1, 
                 max_tokens: int = 8192):
        """
        Initialize the Multimodal RAG system
        
        Args:
            google_api_key: Google API key for Gemini
            persist_directory: Directory to persist vector database
            main_model: Main LLM model for text generation
            vision_model: Vision model for image processing
            embedding_model: Embedding model for vector embeddings
            temperature: Temperature for model responses
            max_tokens: Maximum tokens for model responses
        """
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory
        
        # Configure Google AI
        genai.configure(api_key=google_api_key)
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize Gemini models with configurable parameters
        self.llm = ChatGoogleGenerativeAI(
            model=main_model,
            google_api_key=google_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Vision model for image processing
        self.vision_model = ChatGoogleGenerativeAI(
            model=vision_model,
            google_api_key=google_api_key,
            temperature=temperature
        )
        
        # Embedding model for vector embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key
        )
        
        # Enhanced text splitter with better chunking strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for better context
            chunk_overlap=300,  # Better overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Multi-vector retriever components
        self.vectorstore = None
        self.retriever = None
        self.doc_store = InMemoryByteStore()
        self.image_metadata = {}
        
    def setup_multi_vector_retriever(self):
        """Setup the multi-vector retriever with enhanced capabilities."""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="multimodal_rag_v2"
            )
        
        # Initialize multi-vector retriever with latest pattern
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.doc_store,
            id_key="doc_id",
            search_kwargs={"k": 6}  # Retrieve more candidates
        )
    
    def parse_documents(self, file_paths: List[str]) -> Dict[str, List]:
        """
        Parse documents using basic document loaders.
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            Dictionary containing separated elements (text, tables, images)
        """
        all_elements = {"texts": [], "tables": [], "images": []}
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.pdf'):
                    # Extract text using PyPDF
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        all_elements["texts"].append({
                            "content": doc.page_content,
                            "metadata": {"source": file_path, "type": "text", "page": doc.metadata.get("page", 0)}
                        })
                    
                    # Extract images from PDF using PyMuPDF (if available)
                    if PYMUPDF_AVAILABLE:
                        pdf_images = self._extract_images_from_pdf(file_path)
                        all_elements["images"].extend(pdf_images)
                    else:
                        print(f"Skipping image extraction from {file_path} - PyMuPDF not installed")
                    
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        all_elements["texts"].append({
                            "content": doc.page_content,
                            "metadata": {"source": file_path, "type": "text", "page": 0}
                        })
                else:
                    continue
                    
            except Exception as e:
                print(f"Error parsing {file_path}: {str(e)}")
        
        return all_elements
    
    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract images from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted image data
        """
        extracted_images = []
        
        if not PYMUPDF_AVAILABLE:
            print("PyMuPDF not available - cannot extract images from PDF")
            return extracted_images
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create unique identifier
                    image_hash = hashlib.md5(image_bytes).hexdigest()
                    
                    extracted_images.append({
                        "content": image_bytes,
                        "metadata": {
                            "source": pdf_path,
                            "type": "image",
                            "page": page_num + 1,
                            "image_index": img_index,
                            "extension": image_ext,
                            "hash": image_hash
                        }
                    })
                    
                    print(f"Extracted image {img_index + 1} from page {page_num + 1} of {pdf_path}")
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting images from PDF {pdf_path}: {str(e)}")
        
        return extracted_images
    
    def process_images_advanced(self, image_paths: List[str]) -> List[Dict]:
        """
        Image processing Gemini Vision Model.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processed image data
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                # Load and process image
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                # Create image hash for unique identification
                image_hash = hashlib.md5(image_data).hexdigest()
                
                # Image description with Gemini LLM
                description = self._generate_enhanced_image_description(image_data)
                
                # Extract any text from the image
                text_content = self._extract_text_from_image(image_data)
                
                processed_image = {
                    "path": image_path,
                    "hash": image_hash,
                    "description": description,
                    "text_content": text_content,
                    "type": "image",
                    "metadata": {
                        "source": image_path,
                        "type": "image",
                        "hash": image_hash
                    }
                }
                
                processed_images.append(processed_image)
                print(f"Processed image: {image_path}")
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
        
        return processed_images
    
    def _generate_enhanced_image_description(self, image_data: bytes) -> str:
        """Generate detailed description using latest Gemini vision model capabilities."""
        try:
            # Convert to base64 for API
            image_b64 = base64.b64encode(image_data).decode()
            
            prompt = """
            Analyze this image comprehensively and provide a detailed description that includes:
            
            1. **Main Content**: Objects, people, scenes, activities
            2. **Visual Elements**: Colors, composition, style, lighting
            3. **Text Content**: Any visible text, signs, labels, captions
            4. **Context & Setting**: Environment, location, time period if apparent
            5. **Technical Details**: Charts, graphs, diagrams, technical elements
            6. **Relationships**: How elements in the image relate to each other
            7. **Actionable Information**: Key insights or data points visible
            
            Format your response to be detailed yet concise, optimized for retrieval and understanding.
            Focus on information that would be valuable for question-answering scenarios.
            """
            
            # Use latest vision model with correct message format
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.vision_model.invoke([message])
            
            return response.content
            
        except Exception as e:
            print(f"Error generating enhanced image description: {str(e)}")
            return "Error generating image description"
    
    def _extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text content from images using Gemini OCR capabilities."""
        try:
            image_b64 = base64.b64encode(image_data).decode()
            
            prompt = """
            Extract all visible text from this image. Include:
            - Any readable text, numbers, labels
            - Text in charts, graphs, diagrams
            - Signs, captions, headers
            - Technical annotations
            
            Provide only the extracted text content, maintaining structure where possible.
            If no text is visible, respond with "No text detected".
            """
            
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.vision_model.invoke([message])
            
            return response.content
            
        except Exception as e:
            print(f"Error extracting text from image: {str(e)}")
            return "No text detected"
    
    def create_summaries_for_retrieval(self, elements: List[Dict]) -> List[str]:
        """
        Create optimized summaries for better retrieval using Gemini LLM.
        
        Args:
            elements: List of elements (text, table, image data)
            
        Returns:
            List of summaries optimized for retrieval
        """
        summaries = []
        
        for element in elements:
            try:
                content = element.get("content", "")
                element_type = element.get("metadata", {}).get("type", "text")
                
                if element_type == "table":
                    prompt = f"""
                    Summarize this table for optimal retrieval. Focus on:
                    - Key data points and metrics
                    - Column headers and categories
                    - Notable trends or patterns
                    - Actionable insights
                    
                    Table content:
                    {content[:2000]}  # Limit content length
                    
                    Provide a concise summary optimized for semantic search.
                    """
                elif element_type == "image":
                    # For images, use the description we already generated
                    summaries.append(content)
                    continue
                else:
                    prompt = f"""
                    Create a retrieval-optimized summary of this text. Include:
                    - Main topics and themes
                    - Key facts and figures
                    - Important concepts
                    - Actionable information
                    
                    Text content:
                    {content[:2000]}
                    
                    Provide a comprehensive yet concise summary.
                    """
                
                response = self.llm.invoke(prompt)
                summaries.append(response.content)
                
            except Exception as e:
                print(f"Error creating summary: {str(e)}")
                summaries.append(str(element.get("content", ""))[:500])
        
        return summaries
    
    def build_enhanced_vector_database(self, file_paths: List[str], image_paths: List[str]):
        """
        Build enhanced vector database with multi-vector retrieval capabilities.
        
        Args:
            file_paths: List of document file paths
            image_paths: List of image file paths
        """
        # Setup multi-vector retriever
        self.setup_multi_vector_retriever()
        
        # Parse documents using basic extraction
        parsed_elements = self.parse_documents(file_paths)
        
        # Process standalone image files
        processed_images = self.process_images_advanced(image_paths)
        
        # Process images extracted from PDFs
        pdf_extracted_images = []
        for img_data in parsed_elements["images"]:
            try:
                # Generate description for PDF-extracted image
                description = self._generate_enhanced_image_description(img_data["content"])
                text_content = self._extract_text_from_image(img_data["content"])
                
                processed_img = {
                    "description": description,
                    "text_content": text_content,
                    "metadata": img_data["metadata"]
                }
                pdf_extracted_images.append(processed_img)
                print(f"Processed extracted image from {img_data['metadata']['source']} (page {img_data['metadata']['page']})")
            except Exception as e:
                print(f"Error processing extracted image: {str(e)}")
        
        # Combine all elements
        all_elements = []
        
        # Add text elements
        for text_elem in parsed_elements["texts"]:
            all_elements.append(text_elem)
        
        # Add table elements
        for table_elem in parsed_elements["tables"]:
            all_elements.append(table_elem)
        
        # Add standalone image elements
        for img_elem in processed_images:
            all_elements.append({
                "content": img_elem["description"],
                "metadata": img_elem["metadata"],
                "original_data": img_elem
            })
            
        # Add PDF-extracted image elements
        for img_elem in pdf_extracted_images:
            all_elements.append({
                "content": img_elem["description"],
                "metadata": img_elem["metadata"]
            })
        
        if not all_elements:
            print("No elements to process")
            return
        
        # Create summaries optimized for retrieval
        summaries = self.create_summaries_for_retrieval(all_elements)
        
        # Generate unique IDs for parent documents
        doc_ids = [str(uuid.uuid4()) for _ in all_elements]
        
        # Create summary documents for vector store
        summary_docs = []
        for i, (summary, element) in enumerate(zip(summaries, all_elements)):
            doc = Document(
                page_content=summary,
                metadata={
                    "doc_id": doc_ids[i],
                    "source": element["metadata"].get("source", "unknown"),
                    "type": element["metadata"].get("type", "text")
                }
            )
            summary_docs.append(doc)
        
        # Add summaries to vector store
        self.retriever.vectorstore.add_documents(summary_docs)
        
        # Store original documents in doc store
        doc_pairs = []
        for i, element in enumerate(all_elements):
            original_doc = Document(
                page_content=element["content"],
                metadata=element["metadata"]
            )
            if "original_data" in element:
                original_doc.metadata["original_data"] = element["original_data"]
            
            doc_pairs.append((doc_ids[i], original_doc))
        
        self.retriever.docstore.mset(doc_pairs)
        
        # ChromaDB auto-persists, no need for manual persist() call
        
        print(f"Enhanced vector database built with {len(all_elements)} elements")
        print(f"- Text elements: {len(parsed_elements['texts'])}")
        print(f"- Table elements: {len(parsed_elements['tables'])}")
        print(f"- Image elements: {len(processed_images)}")
    
    def retrieve_multimodal_context(self, query: str, k: int = 6) -> List[Document]:
        """
        Retrieve relevant multimodal context using multi-vector retrieval.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            print("Multi-vector retriever not initialized")
            return []
        
        try:
            # Use multi-vector retriever
            relevant_docs = self.retriever.invoke(query)
            return relevant_docs[:k]
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return []
    
    def generate_advanced_response(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Generate advanced response using Gemini LLM with multimodal context.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Prepare multimodal context
            text_context = []
            image_references = []
            table_context = []
            
            for doc in context_docs:
                doc_type = doc.metadata.get("type", "text")
                source = doc.metadata.get("source", "unknown")
                
                if doc_type == "image":
                    image_references.append({
                        "source": source,
                        "description": doc.page_content
                    })
                elif doc_type == "table":
                    table_context.append({
                        "source": source,
                        "content": doc.page_content
                    })
                else:
                    text_context.append({
                        "source": source,
                        "content": doc.page_content
                    })
            
            # Create enhanced prompt template
            prompt_template = ChatPromptTemplate.from_template("""
            You are an advanced AI assistant with multimodal reasoning capabilities. Answer the user's question using the provided context from various sources including text documents, tables, and image descriptions.

            ## Text Context:
            {text_context}

            ## Table Context:
            {table_context}

            ## Image Context:
            {image_context}

            ## User Question:
            {question}

            ## Instructions:
            1. Provide a comprehensive, accurate answer based on the context
            2. Synthesize information across different modalities when relevant
            3. Cite specific sources when referencing information
            4. If information comes from images, mention that it's from visual analysis
            5. If referencing tables, mention specific data points
            6. Be precise and detailed in your response
            7. If the context doesn't contain sufficient information, state this clearly

            ## Response:
            """)
            
            # Format context for prompt
            text_ctx = "\n\n".join([f"Source: {item['source']}\nContent: {item['content']}" for item in text_context])
            table_ctx = "\n\n".join([f"Source: {item['source']}\nTable: {item['content']}" for item in table_context])
            image_ctx = "\n\n".join([f"Source: {item['source']}\nDescription: {item['description']}" for item in image_references])
            
            # Create chain
            chain = prompt_template | self.llm | StrOutputParser()
            
            # Generate response
            response = chain.invoke({
                "text_context": text_ctx or "No text context available",
                "table_context": table_ctx or "No table context available", 
                "image_context": image_ctx or "No image context available",
                "question": query
            })
            
            # Prepare metadata
            sources = []
            for doc in context_docs:
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("type", "text"),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            return {
                "response": response,
                "sources": sources,
                "retrieved_docs": len(context_docs),
                "multimodal_summary": {
                    "text_sources": len(text_context),
                    "table_sources": len(table_context),
                    "image_sources": len(image_references)
                }
            }
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                "response": "I encountered an error while generating the response.",
                "sources": [],
                "retrieved_docs": 0,
                "multimodal_summary": {"text_sources": 0, "table_sources": 0, "image_sources": 0}
            }
    
    def query(self, question: str, k: int = 6) -> Dict[str, Any]:
        """
        Main query method with enhanced multimodal capabilities.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing response and metadata
        """
        # Retrieve relevant multimodal context
        relevant_docs = self.retrieve_multimodal_context(question, k=k)
        
        if not relevant_docs:
            return {
                "response": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "retrieved_docs": 0,
                "multimodal_summary": {"text_sources": 0, "table_sources": 0, "image_sources": 0}
            }
        
        # Generate advanced response
        return self.generate_advanced_response(question, relevant_docs)


