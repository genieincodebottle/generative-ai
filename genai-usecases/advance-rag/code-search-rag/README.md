# RAG-Powered Code Intelligence & Search System

End-to-end RAG pipeline for code search combining semantic embeddings, intelligent reranking, and Tree-sitter AST parsing for structural code understanding.

## Features

### 📥 Index Code (Vector Storage)
- **Direct Code Input**: Paste code snippets directly
- **Upload Files**: Upload individual code files (.py, .js, .ts, etc.)
- **Index Directory**: Index entire local directories
- **Clone from GitHub**: Clone and index GitHub repositories
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

<strong>📦 Installation & Running App</strong>
   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\advance-rag\rag-with-code
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
         ```
      * Get **GOOGLE_API_KEY** here -> https://aistudio.google.com/app/apikey

   9. Run App
      
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
- **Code Parsing**: Tree-sitter, Regex

## File Structure

```
rag-code/
├── app.py                     # Streamlit UI application
├── rag.py                     # Core RAG system
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
- Check that git is installed for GitHub cloning

### Search Issues
- Ensure you have indexed some code first
- Try different search queries
- Check filter settings

## Examples

### OAuth2 Authentication Example Code

The repository includes comprehensive OAuth2 implementation examples in [test_oauth2_examples.py](test_oauth2_examples.py). This file demonstrates:

**1. Authorization Code Flow (Web Applications)**
```python
# Initialize OAuth2 client for web apps
client = OAuth2Client(
    client_id="my-web-app",
    client_secret="super-secret-key",
    redirect_uri="https://myapp.com/callback",
    auth_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token"
)

# Generate authorization URL
auth_url = client.get_authorization_url(scope="profile email")

# Exchange authorization code for token
token_data = client.exchange_code_for_token(authorization_code)

# Make authenticated requests
response = client.make_authenticated_request("https://api.provider.com/user/profile")
```

**2. PKCE Flow (Mobile/Single-Page Apps)**
```python
# Initialize PKCE client for mobile apps
client = OAuth2PKCEClient(
    client_id="my-mobile-app",
    redirect_uri="myapp://callback",
    auth_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token"
)

# Generate PKCE pair and authorization URL
verifier, challenge = client.generate_pkce_pair()
auth_url = client.get_authorization_url(scope="profile")
```

**3. Client Credentials Flow (Service-to-Service)**
```python
# Authenticate service-to-service
token_data = authenticate_client_credentials(
    client_id="my-service",
    client_secret="secret-key",
    token_url="https://auth.example.com/oauth/token",
    scope="api.read api.write"
)
```

**4. Token Validation and Introspection**
```python
# Validate OAuth2 tokens
validator = OAuth2TokenValidator(
    introspection_url="https://auth.example.com/oauth/introspect",
    client_id="client-id",
    client_secret="client-secret"
)
result = validator.validate_token(access_token)
```

**5. OAuth2 Middleware for Web Frameworks**
```python
# Protect routes with OAuth2
middleware = OAuth2Middleware(
    introspection_url="https://auth.example.com/oauth/introspect",
    client_id="client-id",
    client_secret="client-secret",
    required_scopes=["read", "write"]
)
token_info = middleware.validate_request(authorization_header)
```

To index these OAuth2 examples:
1. Navigate to "Index Code" tab
2. Select "Upload File" or "Index Directory"
3. Upload/index [test_oauth2_examples.py](test_oauth2_examples.py)
4. Search using queries like:
   - "How to implement OAuth2 with PKCE?"
   - "OAuth2 token validation"
   - "Client credentials authentication"

### Sample Searches

After indexing authentication-related code:
- "How to validate JWT tokens?"
- "OAuth2 authentication flow"
- "API key middleware implementation"

## Performance Tips

- **Batch Indexing**: Use directory or GitHub indexing for large codebases
- **Chunk Size**: Adjust based on your code structure (smaller for functions, larger for modules)
- **Top K Results**: Increase for more comprehensive results, decrease for faster responses

