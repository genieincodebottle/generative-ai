# Text-to-SQL Query Tool

A Streamlit web application that converts natural language questions into SQL queries using Google Gemini or Groq APIs. Query the Chinook database using plain English..

## Features

- 🗃️ **Natural Language to SQL**: Ask questions in plain English
- 📊 **Interactive Results**: View both formatted tables and raw SQL results
- 🔍 **Query Visualization**: See the generated SQL queries
- 💡 **Example Queries**: Pre-built examples to get started quickly
- 📋 **Database Explorer**: Browse available tables and sample data

## Installation & Running App

   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\advance-rag\text-to-sql
      ```
   2. Open the Project in VS Code or any code editor.
   3. Create a virtual environment by running the following command in the terminal:
   
      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. Create a `requirements.txt` file and add the following libraries:
      
      ```bash
        streamlit>=1.49.1
        langchain>=0.3.27
        google-generativeai>=0.8.5
        langchain-google-genai>=2.1.12
        langchain-groq>=0.3.8
        langchain-community>=0.3.29
        pandas>=2.3.2
      ```
   5. Install dependencies:
      
      ```bash
      uv pip install -r requirements.txt
      ```
   6. Configure Environment
      * Rename .env.example → .env
      * Update with your keys:

         ```bash
         GOOGLE_API_KEY=your_key_here # Using the free-tier API Key
         GROQ_API_KEY=your_key_here # Using the free-tier API Key
         ```
      * Get **GOOGLE_API_KEY** here -> https://aistudio.google.com/app/apikey
      * Get **GROQ_API_KEY** here -> https://console.groq.com/

   7. Run the Application:

      ```bash
      streamlit run app.py
      ```

## Usage

1. **Select API Provider**: Choose between Google Gemini or Groq in the sidebar
2. **Enter API Key**: Provide your API key for the selected provider
3. **Choose Model**: Select from available models for your provider
4. **Ask Questions**: Type natural language questions about the database
5. **View Results**: See the AI-generated SQL query and formatted results

## Example Questions

- "How many employees are there?"
- "Which country's customers spent the most?"
- "What are the top 5 best-selling tracks?"
- "How many customers are from Germany?"
- "What is the total revenue for each genre?"
- "Which artist has the most albums?"

## Database Schema

The application uses the Chinook database, which represents a digital media store with:

- **artists**: Music artists
- **albums**: Album information
- **tracks**: Individual songs with genre and media type
- **customers**: Customer information
- **employees**: Store employee data
- **invoices**: Purchase transactions
- **invoice_items**: Individual items in purchases
- **genres**: Music genres
- **media_types**: Format types (MP3, AAC, etc.)
- **playlists**: User-created playlists

## Technical Details

- **Framework**: Streamlit for web interface
- **AI Integration**: LangChain for LLM orchestration
- **Database**: SQLite (Chinook sample database)
- **Query Processing**: Chain-based approach with SQL generation and execution
- **Error Handling**: Comprehensive error handling and user feedback