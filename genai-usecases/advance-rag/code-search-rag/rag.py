"""
RAG System for Code Search
==========================
A system where developers can ask questions example "How do I authenticate 
with OAuth2?" and get relevant code examples from code repositories.

Tech Stack:
- LLM & Embeddings: Google Gemini
- Vector Database: ChromaDB
- Framework: LangChain
- Code Parsing: Tree-sitter

Author: Rajesh Srivastava
"""

# ======================================================================
# 1. IMPORTS AND SETUP
# ======================================================================

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# ChromaDB
import chromadb
from chromadb.config import Settings

# Tree-sitter for code parsing
try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available. Using regex-based parsing.")


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for the RAG Code Search System"""
    # API Keys (set via environment variables)
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Model settings
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.0-flash"
    
    # Chunking settings
    chunk_size: int = 500  # tokens
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    
    # Retrieval settings
    top_k_initial: int = 100  # Initial retrieval
    top_k_rerank: int = 3    # After filtering
    top_k_final: int = 2      # Final results
    
    # Database settings
    chroma_persist_dir: str = "./chroma_code_db"
    collection_name: str = "code_repository"
    
    # Supported languages
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"
    ])


config = Config()


# =============================================================================
# 3. CODE PARSER - Extract Functions, Classes, Docstrings
# =============================================================================

@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""
    content: str
    chunk_type: str  # function, class, method, module
    name: str
    docstring: Optional[str]
    signature: Optional[str]
    language: str
    file_path: str
    repo_name: str
    start_line: int
    end_line: int
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeParser:
    """Parse code files and extract functions, classes, and metadata"""
    
    def __init__(self):
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._init_tree_sitter()
    
    def _init_tree_sitter(self):
        """Initialize tree-sitter parsers for supported languages"""
        try:
            # Python parser
            py_parser = Parser(Language(tspython.language()))
            self.parsers['python'] = py_parser
            
            # JavaScript parser
            js_parser = Parser(Language(tsjavascript.language()))
            self.parsers['javascript'] = js_parser
            self.parsers['typescript'] = js_parser
        except Exception as e:
            print(f"Tree-sitter init error: {e}")
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'unknown')
    
    def parse_file(self, file_path: str, repo_name: str = "unknown") -> List[CodeChunk]:
        """Parse a code file and extract chunks"""
        language = self.detect_language(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        if language == 'python':
            return self._parse_python(content, file_path, repo_name)
        elif language in ['javascript', 'typescript']:
            return self._parse_javascript(content, file_path, repo_name, language)
        else:
            return self._parse_generic(content, file_path, repo_name, language)
    
    def _parse_python(self, content: str, file_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse Python code using regex (fallback) or tree-sitter"""
        chunks = []
        lines = content.split('\n')
        
        # Extract imports
        imports = re.findall(r'^(?:from\s+[\w.]+\s+)?import\s+[\w.,\s]+', content, re.MULTILINE)
        
        # Function pattern
        func_pattern = r'(?P<indent>[ \t]*)def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)(?:\s*->\s*[^:]+)?:'
        
        # Class pattern
        class_pattern = r'^class\s+(?P<name>\w+)(?:\([^)]*\))?:'
        
        # Find all functions
        for match in re.finditer(func_pattern, content):
            name = match.group('name')
            params = match.group('params')
            indent = len(match.group('indent'))
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            # Find function body
            func_content, end_line = self._extract_python_block(lines, start_line - 1, indent)
            
            # Extract docstring
            docstring = self._extract_python_docstring(func_content)
            
            # Get context (3 lines before)
            context_start = max(0, start_line - 4)
            context = '\n'.join(lines[context_start:start_line - 1])
            
            chunks.append(CodeChunk(
                content=func_content,
                chunk_type='method' if indent > 0 else 'function',
                name=name,
                docstring=docstring,
                signature=f"def {name}({params})",
                language='python',
                file_path=file_path,
                repo_name=repo_name,
                start_line=start_line,
                end_line=end_line,
                imports=imports,
                metadata={'context_before': context}
            ))
        
        # Find all classes
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            name = match.group('name')
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            class_content, end_line = self._extract_python_block(lines, start_line - 1, 0)
            docstring = self._extract_python_docstring(class_content)
            
            chunks.append(CodeChunk(
                content=class_content,
                chunk_type='class',
                name=name,
                docstring=docstring,
                signature=f"class {name}",
                language='python',
                file_path=file_path,
                repo_name=repo_name,
                start_line=start_line,
                end_line=end_line,
                imports=imports
            ))
        
        return chunks
    
    def _extract_python_block(self, lines: List[str], start_idx: int, base_indent: int) -> Tuple[str, int]:
        """Extract a Python code block based on indentation"""
        block_lines = [lines[start_idx]]
        end_idx = start_idx
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                block_lines.append(line)
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent > base_indent or (line.strip().startswith(('"""', "'''", '#'))):
                block_lines.append(line)
                end_idx = i
            else:
                break
        
        return '\n'.join(block_lines), end_idx + 1
    
    def _extract_python_docstring(self, content: str) -> Optional[str]:
        """Extract docstring from Python code"""
        patterns = [
            r'"""(.*?)"""',
            r"'''(.*?)'''"
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
    
    def _parse_javascript(self, content: str, file_path: str, repo_name: str, language: str) -> List[CodeChunk]:
        """Parse JavaScript/TypeScript code"""
        chunks = []
        lines = content.split('\n')
        
        # Extract imports
        imports = re.findall(r'^(?:import|const|let|var)\s+.*?(?:from|require)\s*[\'"]([^\'"]+)[\'"]', 
                            content, re.MULTILINE)
        
        # Function patterns
        patterns = [
            # Regular function
            r'(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)',
            # Arrow function assigned to variable
            r'(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
            # Class method
            r'(?:async\s+)?(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*{'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                name = match.group('name')
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find function body (matching braces)
                func_content, end_line = self._extract_js_block(content, match.end() - 1)
                
                # Extract JSDoc comment
                docstring = self._extract_jsdoc(content, start_pos)
                
                chunks.append(CodeChunk(
                    content=func_content,
                    chunk_type='function',
                    name=name,
                    docstring=docstring,
                    signature=match.group(0),
                    language=language,
                    file_path=file_path,
                    repo_name=repo_name,
                    start_line=start_line,
                    end_line=end_line,
                    imports=imports
                ))
        
        # Class pattern
        class_pattern = r'(?:export\s+)?class\s+(?P<name>\w+)(?:\s+extends\s+\w+)?'
        for match in re.finditer(class_pattern, content):
            name = match.group('name')
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            class_content, end_line = self._extract_js_block(content, content.find('{', match.end()))
            docstring = self._extract_jsdoc(content, start_pos)
            
            chunks.append(CodeChunk(
                content=class_content,
                chunk_type='class',
                name=name,
                docstring=docstring,
                signature=f"class {name}",
                language=language,
                file_path=file_path,
                repo_name=repo_name,
                start_line=start_line,
                end_line=end_line,
                imports=imports
            ))
        
        return chunks
    
    def _extract_js_block(self, content: str, start_brace: int) -> Tuple[str, int]:
        """Extract JavaScript block by matching braces"""
        if start_brace < 0 or start_brace >= len(content):
            return "", 0
        
        brace_count = 0
        end_pos = start_brace
        
        for i in range(start_brace, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        block = content[start_brace:end_pos]
        end_line = content[:end_pos].count('\n') + 1
        return block, end_line
    
    def _extract_jsdoc(self, content: str, func_start: int) -> Optional[str]:
        """Extract JSDoc comment before function"""
        before = content[:func_start]
        match = re.search(r'/\*\*(.*?)\*/\s*$', before, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _parse_generic(self, content: str, file_path: str, repo_name: str, language: str) -> List[CodeChunk]:
        """Generic parser for unsupported languages - chunk by size"""
        chunks = []
        lines = content.split('\n')
        
        # Simple chunking by lines
        chunk_size = 50  # lines
        for i in range(0, len(lines), chunk_size):
            chunk_content = '\n'.join(lines[i:i + chunk_size])
            chunks.append(CodeChunk(
                content=chunk_content,
                chunk_type='module',
                name=f"chunk_{i // chunk_size}",
                docstring=None,
                signature=None,
                language=language,
                file_path=file_path,
                repo_name=repo_name,
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines))
            ))
        
        return chunks


# =============================================================================
# 4. INDEXING PIPELINE
# =============================================================================

class IndexingPipeline:
    """Offline indexing pipeline for code repositories"""
    
    def __init__(self, config: Config):
        self.config = config
        self.parser = CodeParser()
        
        # Initialize Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=config.collection_name,
            embedding_function=self.embeddings
        )
    
    def create_embedding_text(self, chunk: CodeChunk) -> str:
        """Create text for embedding: function body + docstring + signature"""
        parts = []
        
        if chunk.signature:
            parts.append(f"Signature: {chunk.signature}")
        
        if chunk.docstring:
            parts.append(f"Description: {chunk.docstring}")
        
        parts.append(f"Code:\n{chunk.content}")
        
        return '\n\n'.join(parts)
    
    def create_metadata(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Create metadata for storage"""
        return {
            'chunk_type': chunk.chunk_type,
            'name': chunk.name,
            'language': chunk.language,
            'file_path': chunk.file_path,
            'repo_name': chunk.repo_name,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'has_docstring': chunk.docstring is not None,
            'imports': json.dumps(chunk.imports[:10]),  # Limit imports
            'indexed_at': datetime.now().isoformat()
        }
    
    def index_repository(self, repo_path: str, repo_name: str = None) -> int:
        """Index a single repository"""
        repo_path = Path(repo_path)
        if repo_name is None:
            repo_name = repo_path.name
        
        documents = []
        
        # Find all code files
        extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c']
        code_files = []
        for ext in extensions:
            code_files.extend(repo_path.rglob(f'*{ext}'))
        
        print(f"Found {len(code_files)} code files in {repo_name}")
        
        for file_path in code_files:
            # Skip node_modules, venv, etc.
            if any(skip in str(file_path) for skip in ['node_modules', 'venv', '.git', '__pycache__', 'dist']):
                continue
            
            chunks = self.parser.parse_file(str(file_path), repo_name)
            
            for chunk in chunks:
                # Create document for vector store
                embedding_text = self.create_embedding_text(chunk)
                metadata = self.create_metadata(chunk)
                
                # Generate unique ID
                doc_id = hashlib.md5(
                    f"{repo_name}:{chunk.file_path}:{chunk.name}:{chunk.start_line}".encode()
                ).hexdigest()
                
                documents.append(Document(
                    page_content=embedding_text,
                    metadata={**metadata, 'id': doc_id, 'raw_content': chunk.content}
                ))
        
        # Batch add to vector store
        if documents:
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vector_store.add_documents(batch)
            print(f"Indexed {len(documents)} code chunks from {repo_name}")
        
        return len(documents)
    
    def index_multiple_repositories(self, repo_paths: List[str]) -> Dict[str, int]:
        """Index multiple repositories"""
        results = {}
        for repo_path in repo_paths:
            try:
                count = self.index_repository(repo_path)
                results[repo_path] = count
            except Exception as e:
                print(f"Error indexing {repo_path}: {e}")
                results[repo_path] = 0
        return results
    
    def index_from_code_string(self, code: str, language: str,
                                repo_name: str = "inline", file_name: str = "example") -> int:
        """Index code directly from a string (useful for testing)"""
        print(f"[DEBUG] IndexingPipeline: Indexing code from string - {file_name} ({language})")

        # Create temporary file path
        ext_map = {'python': '.py', 'javascript': '.js', 'typescript': '.ts'}
        ext = ext_map.get(language, '.txt')
        file_path = f"{file_name}{ext}"

        # Parse based on language
        print(f"[DEBUG] IndexingPipeline: Parsing {language} code...")
        if language == 'python':
            chunks = self.parser._parse_python(code, file_path, repo_name)
        elif language in ['javascript', 'typescript']:
            chunks = self.parser._parse_javascript(code, file_path, repo_name, language)
        else:
            chunks = self.parser._parse_generic(code, file_path, repo_name, language)

        print(f"[DEBUG] IndexingPipeline: Found {len(chunks)} chunks")

        documents = []
        for chunk in chunks:
            embedding_text = self.create_embedding_text(chunk)
            metadata = self.create_metadata(chunk)
            doc_id = hashlib.md5(f"{repo_name}:{file_path}:{chunk.name}".encode()).hexdigest()

            documents.append(Document(
                page_content=embedding_text,
                metadata={**metadata, 'id': doc_id, 'raw_content': chunk.content}
            ))

        if documents:
            print(f"[DEBUG] IndexingPipeline: Adding {len(documents)} documents to vector store...")
            self.vector_store.add_documents(documents)
            print(f"[DEBUG] IndexingPipeline: Documents added successfully")

        return len(documents)


# =============================================================================
# 5. QUERY PIPELINE
# =============================================================================

class QueryUnderstanding:
    """Query understanding and expansion"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Analyze this code search query and extract:
            1. Intent: Is it asking "how to" (implementation), "what is" (explanation), or "show me" (examples)?
            2. Key entities: Programming concepts, libraries, patterns mentioned
            3. Expanded terms: Synonyms and related terms

            Query: {query}

            Respond in JSON format:
            {{
                "intent": "how_to|what_is|show_me",
                "entities": ["entity1", "entity2"],
                "expanded_terms": ["term1", "term2"],
                "language_hint": "python|javascript|any"
            }}"""
                    )
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze and expand the query"""
        print(f"[DEBUG] QueryUnderstanding: Starting query analysis for: '{query}'")
        try:
            print(f"[DEBUG] QueryUnderstanding: Calling LLM for intent analysis...")
            response = self.llm.invoke(self.intent_prompt.format(query=query))
            print(f"[DEBUG] QueryUnderstanding: LLM response received")
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"[DEBUG] QueryUnderstanding: Parsed result: {result}")
                return result
        except Exception as e:
            print(f"[ERROR] Query analysis error: {e}")

        # Default fallback
        print(f"[DEBUG] QueryUnderstanding: Using default fallback analysis")
        return {
            "intent": "how_to",
            "entities": query.split()[:5],
            "expanded_terms": [],
            "language_hint": "any"
        }
    
    def expand_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """Expand query with synonyms and related terms"""
        expanded = query
        if analysis.get('expanded_terms'):
            expanded += ' ' + ' '.join(analysis['expanded_terms'])
        return expanded


class CodeRetriever:
    """Multi-stage retrieval with filtering and reranking"""
    
    def __init__(self, vector_store, llm, config: Config):
        self.vector_store = vector_store
        self.llm = llm
        self.config = config
        self.query_understanding = QueryUnderstanding(llm)
    
    def retrieve(self, query: str, filters: Dict[str, Any] = None) -> List[Document]:
        """Multi-stage retrieval pipeline"""
        print(f"\n[DEBUG] CodeRetriever.retrieve() called with query: '{query}'")
        print(f"[DEBUG] Filters: {filters}")

        # Stage 0: Query understanding
        print(f"[DEBUG] Stage 0: Starting query understanding...")
        analysis = self.query_understanding.analyze_query(query)
        expanded_query = self.query_understanding.expand_query(query, analysis)
        print(f"[DEBUG] Stage 0: Expanded query: '{expanded_query}'")

        # Stage 1: Vector search - retrieve top-k candidates
        print(f"[DEBUG] Stage 1: Starting vector search (top_k={self.config.top_k_initial})...")
        initial_results = self.vector_store.similarity_search_with_score(
            expanded_query,
            k=self.config.top_k_initial
        )
        print(f"[DEBUG] Stage 1: Found {len(initial_results)} initial results")

        # Stage 2: Metadata filtering
        print(f"[DEBUG] Stage 2: Applying metadata filters...")
        filtered_results = self._apply_filters(initial_results, filters, analysis)
        print(f"[DEBUG] Stage 2: {len(filtered_results)} results after filtering")

        # Stage 3: Reranking
        print(f"[DEBUG] Stage 3: Reranking top {self.config.top_k_rerank} results...")
        reranked_results = self._rerank(query, filtered_results[:self.config.top_k_rerank])
        print(f"[DEBUG] Stage 3: Reranking complete. Returning top {self.config.top_k_final} results")

        return reranked_results[:self.config.top_k_final]
    
    def _apply_filters(self, results: List[Tuple[Document, float]], 
                       filters: Dict[str, Any], analysis: Dict[str, Any]) -> List[Document]:
        """Apply metadata filters and boost scores"""
        filtered = []
        
        for doc, score in results:
            meta = doc.metadata
            
            # Apply explicit filters
            if filters:
                if filters.get('language') and meta.get('language') != filters['language']:
                    continue
                if filters.get('repo_name') and meta.get('repo_name') != filters['repo_name']:
                    continue
            
            # Apply language hint from query analysis
            if analysis.get('language_hint') != 'any':
                if meta.get('language') == analysis['language_hint']:
                    score *= 0.9  # Boost (lower score = better in similarity)
            
            # Boost functions with docstrings
            if meta.get('has_docstring'):
                score *= 0.95
            
            filtered.append((doc, score))
        
        # Sort by score
        filtered.sort(key=lambda x: x[1])
        return [doc for doc, _ in filtered]
    
    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using LLM-based scoring"""
        if not documents:
            print(f"[DEBUG] _rerank: No documents to rerank")
            return []

        print(f"[DEBUG] _rerank: Reranking {len(documents)} documents...")

        rerank_prompt = PromptTemplate(
            input_variables=["query", "code"],
            template="""Rate how well this code answers the query on a scale of 1-10.
            Consider: semantic relevance, code quality, completeness, and clarity.

            Query: {query}
            Code: {code}

            Respond with only a number 1-10."""
                    )

        scored_docs = []
        for i, doc in enumerate(documents[:self.config.top_k_rerank], 1):
            try:
                print(f"[DEBUG] _rerank: Scoring document {i}/{len(documents[:self.config.top_k_rerank])}...")
                score_str = self.llm.invoke(
                    rerank_prompt.format(query=query, code=doc.page_content[:1000])
                )
                score = float(re.search(r'\d+', score_str).group())
                print(f"[DEBUG] _rerank: Document {i} scored: {score}")
            except Exception as e:
                print(f"[ERROR] _rerank: Error scoring document {i}: {e}")
                score = 5  # Default score

            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] _rerank: Reranking complete")
        return [doc for doc, _ in scored_docs]


# =============================================================================
# 6. RESPONSE GENERATION
# =============================================================================

class ResponseGenerator:
    """Generate helpful responses with code examples"""
    
    def __init__(self, llm, config: Config):
        self.llm = llm
        self.config = config
        
        self.response_prompt = PromptTemplate(
            input_variables=["query", "code_examples", "metadata"],
            template="""You are a helpful coding assistant. Based on the code examples provided, 
            answer the developer's question with clear explanations and relevant code.

            Developer Question: {query}

            Retrieved Code Examples:
            {code_examples}

            Metadata:
            {metadata}

            Instructions:
            1. First, briefly explain the pattern or approach
            2. Show the most relevant code example with syntax highlighting
            3. Explain key parts of the code
            4. Mention any important considerations or alternatives

            Response:"""
                    )
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """Generate a helpful response"""
        print(f"\n[DEBUG] ResponseGenerator.generate() called with {len(documents)} documents")

        if not documents:
            print(f"[DEBUG] ResponseGenerator: No documents found, returning default message")
            return "I couldn't find relevant code examples for your query. Please try rephrasing or provide more context."

        # Assemble context
        print(f"[DEBUG] ResponseGenerator: Assembling context from documents...")
        code_examples = []
        metadata_info = []

        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            code_examples.append(f"""
            Example {i} - {meta.get('name', 'Unknown')} ({meta.get('language', 'unknown')})
            Repository: {meta.get('repo_name', 'unknown')}
            File: {meta.get('file_path', 'unknown')}

            ```{meta.get('language', '')}
            {meta.get('raw_content', doc.page_content)[:800]}
            ```
            """)
            metadata_info.append(
                f"- {meta.get('name')}: {meta.get('chunk_type')} in {meta.get('repo_name')}"
            )

        # Generate response
        print(f"[DEBUG] ResponseGenerator: Calling LLM to generate final response...")
        response = self.llm.invoke(
            self.response_prompt.format(
                query=query,
                code_examples='\n'.join(code_examples),
                metadata='\n'.join(metadata_info)
            )
        )
        print(f"[DEBUG] ResponseGenerator: Response generated successfully")

        return response


# =============================================================================
# 7. RAG CODE SEARCH SYSTEM - Main Class
# =============================================================================

class RAGCodeSearchSystem:
    """Main RAG system for code search"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Validate API key
        if not self.config.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.config.google_api_key,
            temperature=0.3
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model,
            google_api_key=self.config.google_api_key
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings
        )
        
        # Initialize components
        self.indexing_pipeline = IndexingPipeline(self.config)
        self.retriever = CodeRetriever(self.vector_store, self.llm, self.config)
        self.generator = ResponseGenerator(self.llm, self.config)
    
    def index_repository(self, repo_path: str, repo_name: str = None) -> int:
        """Index a code repository"""
        return self.indexing_pipeline.index_repository(repo_path, repo_name)
    
    def index_code(self, code: str, language: str, 
                   repo_name: str = "inline", file_name: str = "example") -> int:
        """Index code from a string"""
        return self.indexing_pipeline.index_from_code_string(code, language, repo_name, file_name)
    
    def search(self, query: str, filters: Dict[str, Any] = None) -> str:
        """Search for code and generate a response"""
        print(f"\n{'='*80}")
        print(f"[DEBUG] RAGCodeSearchSystem.search() START")
        print(f"[DEBUG] Query: '{query}'")
        print(f"[DEBUG] Filters: {filters}")
        print(f"{'='*80}\n")

        # Retrieve relevant code
        print(f"[DEBUG] Step 1: Retrieving relevant code...")
        documents = self.retriever.retrieve(query, filters)
        print(f"[DEBUG] Step 1: Retrieved {len(documents)} documents")

        # Generate response
        print(f"[DEBUG] Step 2: Generating response...")
        response = self.generator.generate(query, documents)
        print(f"[DEBUG] Step 2: Response generated")

        print(f"\n{'='*80}")
        print(f"[DEBUG] RAGCodeSearchSystem.search() COMPLETE")
        print(f"{'='*80}\n")

        return response

    def search_with_details(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search with full details including retrieved documents"""
        print(f"\n{'='*80}")
        print(f"[DEBUG] RAGCodeSearchSystem.search_with_details() START")
        print(f"[DEBUG] Query: '{query}'")
        print(f"[DEBUG] Filters: {filters}")
        print(f"{'='*80}\n")

        print(f"[DEBUG] Step 1: Retrieving documents...")
        documents = self.retriever.retrieve(query, filters)
        print(f"[DEBUG] Step 1: Retrieved {len(documents)} documents")

        print(f"[DEBUG] Step 2: Generating response...")
        response = self.generator.generate(query, documents)
        print(f"[DEBUG] Step 2: Response generated")

        print(f"[DEBUG] Step 3: Preparing detailed results...")
        result = {
            'query': query,
            'response': response,
            'documents': [
                {
                    'name': doc.metadata.get('name'),
                    'language': doc.metadata.get('language'),
                    'repo': doc.metadata.get('repo_name'),
                    'file': doc.metadata.get('file_path'),
                    'content': doc.metadata.get('raw_content', doc.page_content)[:500]
                }
                for doc in documents
            ],
            'num_results': len(documents)
        }

        print(f"\n{'='*80}")
        print(f"[DEBUG] RAGCodeSearchSystem.search_with_details() COMPLETE")
        print(f"{'='*80}\n")

        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        collection = self.chroma_client.get_collection(self.config.collection_name)
        return {
            'total_documents': collection.count(),
            'collection_name': self.config.collection_name,
            'persist_directory': self.config.chroma_persist_dir
        }


# =============================================================================
# 8. EVALUATION METRICS
# =============================================================================

class Evaluator:
    """Evaluation metrics for the RAG system"""
    
    @staticmethod
    def mean_reciprocal_rank(relevant_docs: List[str], retrieved_docs: List[Document]) -> float:
        """Calculate MRR - Is correct answer in top-k?"""
        for i, doc in enumerate(retrieved_docs, 1):
            if doc.metadata.get('name') in relevant_docs:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def recall_at_k(relevant_docs: List[str], retrieved_docs: List[Document], k: int = 5) -> float:
        """Calculate Recall@k"""
        retrieved_names = {doc.metadata.get('name') for doc in retrieved_docs[:k]}
        relevant_set = set(relevant_docs)
        
        if not relevant_set:
            return 0.0
        
        return len(retrieved_names & relevant_set) / len(relevant_set)
    
    @staticmethod
    def precision_at_k(relevant_docs: List[str], retrieved_docs: List[Document], k: int = 5) -> float:
        """Calculate Precision@k"""
        retrieved_names = [doc.metadata.get('name') for doc in retrieved_docs[:k]]
        relevant_set = set(relevant_docs)
        
        if not retrieved_names:
            return 0.0
        
        relevant_retrieved = sum(1 for name in retrieved_names if name in relevant_set)
        return relevant_retrieved / len(retrieved_names)


# =============================================================================
# 9. EXAMPLE USAGE AND DEMO
# =============================================================================

def demo():
    """Demonstrate the RAG Code Search System"""
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print("\nPlease set your Google API key:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        print("=" * 60)
        return
    
    print("=" * 60)
    print("RAG CODE SEARCH SYSTEM - DEMO")
    print("=" * 60)
    
    # Initialize system
    print("\n1. Initializing RAG system...")
    system = RAGCodeSearchSystem()
    
    # Sample code for indexing
    sample_codes = [
        # OAuth2 Python example
        ("""
def authenticate_oauth2(client_id: str, client_secret: str, redirect_uri: str):
    '''
    Authenticate user using OAuth2 flow.
    
    Args:
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        redirect_uri: Callback URL after authentication
    
    Returns:
        Access token for API calls
    '''
    import requests
    from urllib.parse import urlencode
    
    # Authorization URL
    auth_params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': 'read write'
    }
    auth_url = f"https://oauth.example.com/authorize?{urlencode(auth_params)}"
    
    # Exchange code for token
    token_response = requests.post(
        'https://oauth.example.com/token',
        data={
            'grant_type': 'authorization_code',
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri,
            'code': 'AUTH_CODE_HERE'
        }
    )
    return token_response.json()['access_token']
""", "python", "auth_library", "oauth2_auth.py"),
        
        # JWT Token validation
        ("""
def validate_jwt_token(token: str, secret_key: str) -> dict:
    '''
    Validate and decode a JWT token.
    
    Args:
        token: JWT token string
        secret_key: Secret key for verification
    
    Returns:
        Decoded payload if valid
    
    Raises:
        ValueError: If token is invalid or expired
    '''
    import jwt
    from datetime import datetime
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        
        # Check expiration
        if payload.get('exp') and datetime.utcnow().timestamp() > payload['exp']:
            raise ValueError("Token has expired")
        
        return payload
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")
""", "python", "auth_library", "jwt_validation.py"),
        
        # API Key authentication
        ("""
def authenticate_api_key(api_key: str, valid_keys: list) -> bool:
    '''
    Simple API key authentication.
    
    Args:
        api_key: The API key to validate
        valid_keys: List of valid API keys
    
    Returns:
        True if valid, False otherwise
    '''
    import hashlib
    import hmac
    
    for valid_key in valid_keys:
        if hmac.compare_digest(api_key, valid_key):
            return True
    return False

class APIKeyMiddleware:
    '''Middleware for API key authentication in web frameworks'''
    
    def __init__(self, app, api_keys: list):
        self.app = app
        self.api_keys = api_keys
    
    def __call__(self, request):
        api_key = request.headers.get('X-API-Key')
        if not authenticate_api_key(api_key, self.api_keys):
            return {'error': 'Invalid API key'}, 401
        return self.app(request)
""", "python", "auth_library", "api_key_auth.py"),
        
        # JavaScript OAuth2
        ("""
async function authenticateWithOAuth2(clientId, clientSecret, code) {
    /**
     * Exchange authorization code for access token
     * @param {string} clientId - OAuth2 client ID
     * @param {string} clientSecret - OAuth2 client secret
     * @param {string} code - Authorization code from redirect
     * @returns {Promise<string>} Access token
     */
    const tokenEndpoint = 'https://oauth.example.com/token';
    
    const response = await fetch(tokenEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            grant_type: 'authorization_code',
            client_id: clientId,
            client_secret: clientSecret,
            code: code,
            redirect_uri: window.location.origin + '/callback'
        })
    });
    
    const data = await response.json();
    
    if (data.error) {
        throw new Error(data.error_description || 'OAuth2 authentication failed');
    }
    
    // Store token securely
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    
    return data.access_token;
}
""", "javascript", "frontend_auth", "oauth2.js"),
        
        # Database connection
        ("""
def create_database_connection(host: str, port: int, database: str, user: str, password: str):
    '''
    Create a secure database connection with connection pooling.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Username
        password: Password
    
    Returns:
        Connection pool object
    '''
    import psycopg2
    from psycopg2 import pool
    
    connection_pool = pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        sslmode='require'
    )
    
    return connection_pool
""", "python", "database_utils", "connection.py"),
    ]
    
    # Index sample code
    print("\n2. Indexing sample code repositories...")
    total_indexed = 0
    for code, language, repo, file in sample_codes:
        count = system.index_code(code, language, repo, file)
        total_indexed += count
        print(f"   - Indexed {count} chunks from {repo}/{file}")
    
    print(f"\n   Total indexed: {total_indexed} code chunks")
    
    # Example queries
    queries = [
        "How do I authenticate with OAuth2?",
        "Show me JWT token validation",
        "How to implement API key authentication?",
        "Database connection with pooling"
    ]
    
    print("\n3. Running example queries...")
    print("-" * 60)
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        try:
            result = system.search_with_details(query)
            print(f"\n🔍 Found {result['num_results']} relevant code examples")
            print(f"\n💬 Response:\n{result['response'][:1500]}...")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 60)
    
    # Show stats
    print("\n4. System Statistics:")
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")

