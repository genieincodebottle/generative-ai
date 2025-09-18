# Customer Call Sentiment & Aggressiveness Analyzer

A beginner friendly GenAI application that analyzes customer call transcripts using Ollama based local open models to extract sentiment and aggressiveness scores with real time processing and SQLite database storage.

## Features

- 🤖 **Local Models** - Support for various Ollama based open models (Llama, Gemma, etc.)
- 🛡️ **Structured Output** - Pydantic models ensure consistent sentiment and aggressiveness scoring
- 🔐 **Security** - Parameterized SQL queries prevent injection attacks
- 🗄️ **Database Integration** - SQLite storage with automatic setup and sample data
- ❌ **Zero Setup** - No database installation required - SQLite auto-creates everything

## 🛠️ Setup Instructions

### Prerequisites

1. **Install Ollama**
   - Download from [ollama.com](https://ollama.com/download)
   - Follow installation instructions for your operating system (macOS, Linux, Windows)

   Check Ollama-based open models → https://ollama.com/search

2. **After Ollama installation, pull a local open model based on your choice and system capacity**
   ```bash
   ollama pull llama3.2:1b # Options: llama3.2:1b, llama3.2:3b, llama3.1:8b, gemma2:2b, gemma2:9b
   ```
   Reference guide for memory requirements:
   - **llama3.2:1b** (1B parameters) - ~0.7GB RAM
   - **llama3.2:3b** (3B parameters) - ~2GB RAM
   - **llama3.1:8b** (8B parameters) - ~4.5GB RAM
   - **gemma2:2b** (2B parameters) - ~1.5GB RAM
   - **gemma2:9b** (9B parameters) - ~5GB RAM

   **Note**: Ollama uses Q4_0 quantization (~0.5-0.7GB per billion parameters)

3. **Database Setup**
   - **No installation required** SQLite database is created automatically
   - Database file and sample data are initialized on first run

4. **Start Ollama Service** (if needed)
   ```bash
   ollama serve  # Only needed if Ollama isn't running automatically
   ```

   **Note**: Most desktop installations start Ollama automatically. Check if it's running by visiting `http://localhost:11434` in your browser.

## ⚙️ Installation & Running App

1. Clone the repository:
   ```bash
   git clone https://github.com/genieincodebottle/generative-ai.git
   cd genai-usecases\sentiment-analysis
   ```

2. Open the project in VS Code or any code editor.

3. Create a virtual environment:
   ```bash
   pip install uv # if uv not installed
   uv venv
   .venv\Scripts\activate # On Linux -> source venv/bin/activate
   ```

4. The `requirements.txt` file contains the following dependencies:
   ```bash
   langchain>=0.3.27
   langchain-community>=0.3.29
   langchain-ollama>=0.3.8
   streamlit>=1.49.1
   python-dotenv>=1.0.0
   requests>=2.32.5
   ```

5. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

6. Run the application (database is created automatically):
   ```bash
   streamlit run app.py
   ```

## 🗄️ Database Setup

### ✨ Manual Setup Required

**Setup steps when you first run the application:**

1. **Go to Setup Database tab** - Click the first tab in the interface
2. **Click "Initialize Database"** - Creates SQLite database with sample data
3. **Verify setup** - App shows "Database ready!" with 10 calls available
4. **Start analyzing** - Go to Process Calls tab and click "Fetch and Process All Calls"

The database includes 10 sample customer calls with varying sentiments - perfect for testing.

## ⚙️ Configuration

### Environment Variables (.env file)
Create a `.env` file in the project directory:
```env
# SQLite Database Configuration
DB_PATH=sentiment_analysis.db

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TEMPERATURE=0.0
```

### Sidebar Configuration
All settings can be configured through the **sidebar interface**:

#### 📊 Status Section
- **Ollama Status**: Real-time connection status with setup instructions
- **Setup Guide**: Expandable instructions when Ollama is not running

#### 🤖 Model Settings
- **Model Selection**: Choose from available models based on your system memory
- **Temperature**: Controls response consistency (0 = focused, 1 = creative)

#### 🔧 Advanced Settings (Expandable)
- **Ollama URL**: Server location (default: http://localhost:11434)

#### 🗄️ Database Settings (Expandable)
- **Host, Port, Database Name**: Connection details
- **Username, Password**: Database credentials
- **Connection Test**: Verify database connectivity and table existence

## 💡 Usage

### Main Interface Tabs

#### 🗄️ Setup Database Tab
1. **Initialize Database** - Create database with sample customer call data
2. **Database Status** - Shows database readiness and record counts
3. **Reset Database** - Clear all data and start fresh
4. **Auto-Recovery** - Handles missing tables gracefully

#### 📞 Process Calls Tab
1. **Prerequisites Check** - Ensures database is initialized before processing
2. **Batch Processing** - Analyze all customer calls with progress tracking
3. **Smart Error Handling** - Failed calls are logged separately without stopping batch
4. **Real-time Feedback** - Progress bar and status updates during analysis

#### 📊 View Results Tab
1. **Auto-Load Results** - Displays analysis results immediately (no button required)
2. **Interactive Table** - Browse all analyzed calls with sortable columns
3. **Comprehensive Statistics**:
   - Total calls analyzed
   - Sentiment distribution (Positive, Negative, Neutral)
   - Average aggressiveness score
   - High-aggression calls count (≥7)
4. **Manual Refresh** - Optional refresh button for latest data

#### ℹ️ Database Info Tab
1. **Schema Information** - View required table structures
2. **Connection Testing** - Test database connectivity
3. **Setup Instructions** - SQL commands for table creation

### Output Format

The analysis returns:
- **Sentiment**: `Positive`, `Negative`, or `Neutral`
- **Aggressiveness**: Integer scale from 1 (calm) to 10 (very aggressive)

### Configuration Tips
- **Smaller models** (1b-3b): Faster processing, less resource usage
- **Larger models** (8b+): Better accuracy, more resource intensive
- **Temperature = 0**: Consistent, deterministic scoring
- **Higher temperature**: More varied responses (not recommended for scoring)

## 🐛 Troubleshooting

### Common Issues

1. **"Ollama server is not running"**
   - Ensure Ollama is installed and running: `ollama serve`
   - Check if accessible at `http://localhost:11434`
   - Verify the Base URL in sidebar settings

2. **"Model not found"**
   - Pull the model first: `ollama pull [model-name]`
   - Wait for download to complete
   - Select correct model in sidebar dropdown

3. **"Database connection failed"**
   - Check if you have write permissions in the project directory
   - Verify database path in .env file or sidebar
   - Use "Initialize Database" button in Setup Database tab

4. **"Processing errors" or "No customer calls found"**
   - First ensure database is initialized in Setup Database tab
   - Check call content for invalid characters
   - Ensure call text is between 10-10,000 characters
   - Verify model is properly initialized

5. **"Missing tables" or sqlite3.OperationalError**
   - Go to Setup Database tab and click "Initialize Database"
   - If database file exists but tables missing, app will show warning with initialize option
   - Use "Reset Database" to clear and recreate everything
   - Check write permissions in project directory

6. **After database reset - no data showing**
   - This is normal - reset clears all data
   - Go to Setup Database tab and click "Initialize Database" to reload sample data
   - Then process calls again in Process Calls tab

7. **Memory Issues**
   - Choose smaller models (llama3.2:1b, gemma2:2b)
   - Process fewer calls at once
   - Close other resource-intensive applications

## 🔧 Technical Architecture

### Analysis Pipeline
1. **Input Validation**: Validates call content length and format
2. **LLM Processing**: Uses structured prompts with Pydantic output parsing
3. **Database Storage**: Saves results with conflict resolution (upsert)
4. **Error Handling**: Continues processing even if individual calls fail

### Technology Stack
- **Frontend**: Streamlit with interactive UI components
- **Backend**: LangChain 0.3+ with Ollama integration
- **Database**: SQLite with automatic initialization
- **Validation**: Pydantic models for structured output
- **Configuration**: python-dotenv for environment management

