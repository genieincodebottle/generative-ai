# RAG-Powered Code Intelligence & Search System

End-to-end RAG pipeline for code search combining semantic embeddings, intelligent reranking, and Tree-sitter AST parsing for structural code understanding.

## 📁 File Overview

| File | Role |
|------|------|
| `app.py` | **Entry point** — Streamlit web application; run this file to start the UI |
| `rag.py` | Core backend — code parsing, chunking, vector indexing, semantic search, response generation |
| `test_oauth2_examples.py` | Sample code used as **demo data** (not a test file — do not run with pytest). Index it via the app to explore code search with realistic OAuth2 examples. |
| `requirements.txt` | All Python dependencies |
| `.env.example` | Template for environment variables — copy to `.env` and add your key |

## Features

### 📥 Index Code (Vector Storage)
- **Direct Code Input**: Paste code snippets directly
- **Upload Files**: Upload individual code files (.py, .js, .ts, etc.)
- **Index Directory**: Index entire local directories
- **Clone from GitHub**: Clone and index GitHub repositories (requires `git` CLI installed)
- **Import from JSON**: Bulk import from JSON files

### 🔎 Search Code (Retrieval)
- **Natural Language Search**: Use plain English to find code
- **Language Filtering**: Filter by programming language
- **Repository Filtering**: Filter by repository name
- **AI-Powered Responses**: Get explanations along with code examples
- **Detailed Results**: View metadata and context for each result

### 📈 Analytics
- View system statistics
- Track indexing operations
- Monitor search history
- Database information

## Setup

<strong>🛠️ Setup Instructions</strong>

<strong>✅ Prerequisites</strong>
   - Python 3.10 or higher
   - pip (Python package installer)
   - `git` CLI (only needed if you want to clone GitHub repositories from within the app)

<strong>📦 Installation & Running App</strong>
   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git

      # Windows
      cd genai-usecases\advance-rag\code-search-rag

      # Linux / macOS
      cd genai-usecases/advance-rag/code-search-rag
      ```
   2. Open the project in VS Code or any code editor.
   3. Create a virtual environment:

      ```bash
      pip install uv  # skip if uv is already installed
      uv venv

      # Windows
      .venv\Scripts\activate

      # Linux / macOS
      source .venv/bin/activate
      ```
   4. Install dependencies (a `requirements.txt` is already included in this folder):

      ```bash
      uv pip install -r requirements.txt
      ```
   5. Configure environment — **never commit the `.env` file to version control**:
      * Rename `.env.example` → `.env`
      * Add your API key:

         ```bash
         GOOGLE_API_KEY=your_key_here
         ```
      * Get a free **GOOGLE_API_KEY** at https://aistudio.google.com/app/apikey

   6. Start the Streamlit app:

      ```bash
      streamlit run app.py
      ```
      The application will open in your default browser at `http://localhost:8501`

## Usage

### Indexing Code

1. Navigate to the **"Index Code"** tab
2. Choose your preferred indexing method:
   - **Direct Code Input**: Paste code, select language, and click "Index Code"
   - **Upload File**: Upload code files and click "Index Files"
   - **Index Directory**: Enter directory path and click "Index Directory"
   - **Clone from GitHub**: Enter GitHub URL and click "Clone and Index"
   - **Import from JSON**: Upload JSON file and click "Import from JSON"

#### JSON Format Example

```json
[
  {
    "code": "def hello(): print('Hello')",
    "name": "hello",
    "language": "python",
    "repo": "my-repo",
    "file_path": "src/hello.py",
    "description": "A hello function"
  }
]
```

### Searching Code

1. Navigate to the **"Search Code"** tab
2. Enter your search query in natural language
3. (Optional) Apply filters:
   - Filter by programming language
   - Filter by repository name
4. Toggle "Show Details" to see retrieved documents
5. Click "Search"

#### Example Queries

- "How do I authenticate with OAuth2?"
- "Show me JWT token validation"
- "Database connection with pooling"
- "Async function examples in JavaScript"
- "Error handling best practices"

### Viewing Analytics

1. Navigate to the **"Analytics"** tab
2. View:
   - Total documents indexed
   - Total searches performed
   - Recent indexing operations
   - Recent search queries
   - Database information

## Architecture

### Components

1. **RAG System** ([rag.py](rag.py))
   - Code parsing and chunking
   - Vector indexing
   - Semantic search
   - Response generation

2. **Streamlit UI** ([app.py](app.py))
   - Web interface
   - Session management
   - Multi-method indexing
   - Search interface

3. **Vector Database** (ChromaDB)
   - Persistent storage at `./chroma_code_db`
   - Efficient similarity search
   - Metadata filtering

### Tech Stack

- **LLM & Embeddings**: Google Gemini
- **Vector Database**: ChromaDB
- **Framework**: LangChain
- **UI**: Streamlit
- **Code Parsing**: Tree-sitter (AST-based), Regex (fallback)

## File Structure

```
code-search-rag/
├── app.py                     # Streamlit UI application (entry point)
├── rag.py                     # Core RAG system
├── test_oauth2_examples.py    # Sample OAuth2 code for demo indexing (not a pytest file)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Your API keys (not committed)
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── chroma_code_db/            # Vector database (created on first run)
```

## Advanced Configuration

In the sidebar, expand "Advanced Settings" to configure:
- Chunk Size (default: 500)
- Top K Results
- Embedding Model
- LLM Model

## Troubleshooting

### API Key Issues
- Ensure your Google API key is valid
- Check that you have API access enabled for Gemini

### Indexing Issues
- Verify file paths are correct and accessible
- Ensure files are in supported formats
- Check that the `git` CLI is installed for GitHub cloning

### Search Issues
- Ensure you have indexed some code first
- Try different search queries
- Check filter settings

## Examples

### Using the Included OAuth2 Demo Code

> **What is `test_oauth2_examples.py`?**
> This file contains realistic OAuth2 authentication examples (Authorization Code, PKCE, Client Credentials, token validation, middleware). Its name begins with `test_` only because it was originally written in a test-file style — **it is not a pytest test file**. Its purpose here is to serve as meaningful sample code that you can index into the RAG system to explore how code search works on real-world patterns.

The file demonstrates five OAuth2 patterns:

**1. Authorization Code Flow (Web Applications)**
```python
client = OAuth2Client(
    client_id="my-web-app",
    client_secret="super-secret-key",
    redirect_uri="https://myapp.com/callback",
    auth_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token"
)
auth_url = client.get_authorization_url(scope="profile email")
token_data = client.exchange_code_for_token(authorization_code)
response = client.make_authenticated_request("https://api.provider.com/user/profile")
```

**2. PKCE Flow (Mobile/Single-Page Apps)**
```python
client = OAuth2PKCEClient(
    client_id="my-mobile-app",
    redirect_uri="myapp://callback",
    auth_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token"
)
verifier, challenge = client.generate_pkce_pair()
auth_url = client.get_authorization_url(scope="profile")
```

**3. Client Credentials Flow (Service-to-Service)**
```python
token_data = authenticate_client_credentials(
    client_id="my-service",
    client_secret="secret-key",
    token_url="https://auth.example.com/oauth/token",
    scope="api.read api.write"
)
```

**4. Token Validation and Introspection**
```python
validator = OAuth2TokenValidator(
    introspection_url="https://auth.example.com/oauth/introspect",
    client_id="client-id",
    client_secret="client-secret"
)
result = validator.validate_token(access_token)
```

**5. OAuth2 Middleware for Web Frameworks**
```python
middleware = OAuth2Middleware(
    introspection_url="https://auth.example.com/oauth/introspect",
    client_id="client-id",
    client_secret="client-secret",
    required_scopes=["read", "write"]
)
token_info = middleware.validate_request(authorization_header)
```

To index and search these OAuth2 examples:
1. Navigate to "Index Code" tab
2. Select "Upload File"
3. Upload `test_oauth2_examples.py`
4. Search using queries like:
   - "How to implement OAuth2 with PKCE?"
   - "OAuth2 token validation"
   - "Client credentials authentication"

### Sample Searches

After indexing:
- "How to validate JWT tokens?"
- "OAuth2 authentication flow"
- "API key middleware implementation"

## Performance Tips

- **Batch Indexing**: Use directory or GitHub indexing for large codebases
- **Chunk Size**: Adjust based on your code structure (smaller for functions, larger for modules)
- **Top K Results**: Increase for more comprehensive results, decrease for faster responses
