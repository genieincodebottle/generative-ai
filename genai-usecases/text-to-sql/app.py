"""
Text-to-SQL Query Application
============================
A Streamlit application that converts natural language to SQL queries using AI providers.

Structure:
- Business Logic & Backend: Core functionality and data processing
- UI Components: Streamlit interface and user interactions
"""

import streamlit as st
import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# Load environment variables
load_dotenv()

# ================================
# BUSINESS LOGIC & BACKEND
# ================================

class DatabaseManager:
    """Handles database connections and operations"""

    @staticmethod
    @st.cache_resource
    def get_database_connection():
        """Initialize and return database connection"""
        db_path = os.path.join(os.path.dirname(__file__), "chinook.db")
        if not os.path.exists(db_path):
            st.error(f"Database file not found at: {db_path}")
            return None

        try:
            db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            return db
        except Exception as e:
            st.error(f"Error connecting to database: {str(e)}")
            return None

    @staticmethod
    def execute_sql(clean_sql, db_path):
        """Execute SQL and return pandas DataFrame"""
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(clean_sql, conn)
        conn.close()
        return df


class APIManager:
    """Handles API key management and provider detection"""

    @staticmethod
    def load_api_keys():
        """Load API keys from environment variables"""
        return {
            'google': os.getenv("GOOGLE_API_KEY"),
            'groq': os.getenv("GROQ_API_KEY")
        }

    @staticmethod
    def get_available_providers(api_keys):
        """Detect which API providers are available"""
        providers = []
        if api_keys['google']:
            providers.append("Google Gemini")
        if api_keys['groq']:
            providers.append("Groq")
        return providers

    @staticmethod
    def get_model_options(provider):
        """Get available models for each provider"""
        models = {
            "Google Gemini": ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.5-flash", "gemini-2.5-pro"],
            "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
        }
        return models.get(provider, [])


class LLMManager:
    """Handles LLM initialization and operations"""

    @staticmethod
    def initialize_llm(provider, api_key, model, temperature):
        """Initialize the appropriate LLM based on provider"""
        if not api_key:
            return None

        try:
            if provider == "Google Gemini":
                return ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=temperature,
                    convert_system_message_to_human=True
                )
            else:  # Groq
                return ChatGroq(
                    groq_api_key=api_key,
                    model_name=model,
                    temperature=temperature
                )
        except Exception as e:
            st.error(f"Error initializing {provider}: {str(e)}")
            return None


class SQLProcessor:
    """Handles SQL query generation, cleaning, and execution"""

    @staticmethod
    def clean_sql_query(query):
        """Remove all possible markdown formatting from SQL query"""
        if not query:
            return ""

        # Convert to string and strip
        clean = str(query).strip()

        # Remove markdown code blocks - handle all variations
        clean = re.sub(r'^```(?:sql|sqlite|SQL|SQLITE)?\s*\n?', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\n?```\s*$', '', clean)

        # Pattern 2: ` ... ` (single backticks)
        if clean.startswith('`') and clean.endswith('`') and clean.count('`') == 2:
            clean = clean[1:-1]

        # Remove any remaining backticks at start/end
        while clean.startswith('`'):
            clean = clean[1:]
        while clean.endswith('`'):
            clean = clean[:-1]

        # Remove common prefixes that might be left
        prefixes_to_remove = ['sql', 'sqlite', 'SQL', 'SQLITE', 'ql', 'lite', 'ite']
        for prefix in prefixes_to_remove:
            if clean.lower().startswith(prefix.lower() + '\n'):
                clean = clean[len(prefix):].lstrip('\n ')
            elif clean.lower().startswith(prefix.lower() + ' '):
                clean = clean[len(prefix):].lstrip(' ')

        # Clean up whitespace
        clean = clean.strip()

        # Remove any remaining non-SQL characters at the beginning
        while clean and not clean[0].isalnum() and clean[0] not in ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE', 'DROP', 'ALTER'):
            if clean.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE', 'DROP', 'ALTER')):
                break
            clean = clean[1:].strip()

        return clean

    @staticmethod
    def generate_sql_query(llm, db, question):
        """Generate SQL query from natural language"""
        write_query = create_sql_query_chain(llm, db)
        return write_query.invoke({"question": question})

    @staticmethod
    def execute_query(db, clean_sql):
        """Execute SQL query and return results"""
        try:
            return db.run(clean_sql)
        except Exception as e:
            raise e

    @staticmethod
    def generate_answer(llm, question, query, result):
        """Generate natural language answer from SQL results"""
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
        )

        answer_chain = answer_prompt | llm | StrOutputParser()
        return answer_chain.invoke({
            "question": question,
            "query": query,
            "result": result
        })


class QueryEngine:
    """Main query processing engine that orchestrates all components"""

    def __init__(self, llm, db):
        self.llm = llm
        self.db = db
        self.sql_processor = SQLProcessor()

    def process_query(self, question):
        """Process a natural language question and return results"""
        try:
            # Generate SQL query
            sql_query = self.sql_processor.generate_sql_query(self.llm, self.db, question)

            # Clean SQL query
            clean_sql = self.sql_processor.clean_sql_query(sql_query)

            # Execute query
            raw_result = self.sql_processor.execute_query(self.db, clean_sql)

            # Generate answer
            answer = self.sql_processor.generate_answer(self.llm, question, clean_sql, raw_result)

            return {
                'success': True,
                'answer': answer,
                'sql_query': clean_sql,
                'raw_result': raw_result,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'answer': None,
                'sql_query': clean_sql if 'clean_sql' in locals() else None,
                'raw_result': None,
                'error': str(e)
            }


# ================================
# UI COMPONENTS & STREAMLIT CODE
# ================================

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Text-to-SQL Query App",
        page_icon="🗃️",
        layout="wide"
    )


def render_header():
    """Render the main header and description"""
    st.title("🗃️ Text-to-SQL Query App")
    st.markdown("Convert natural language questions into SQL queries and get answers from your database.")


def render_sidebar(api_manager):
    """Render the sidebar with API configuration"""
    st.sidebar.header("🛠️ API Configuration")

    # Load API keys and get available providers
    api_keys = api_manager.load_api_keys()
    available_providers = api_manager.get_available_providers(api_keys)

    if not available_providers:
        st.error("❌ No API keys found in environment variables. Please set GOOGLE_API_KEY and/or GROQ_API_KEY in your .env file.")
        st.stop()

    # API Provider Selection
    api_provider = st.sidebar.selectbox(
        "Select API Provider",
        available_providers,
        index=0
    )

    # Set API key and get model options
    if api_provider == "Google Gemini":
        api_key = api_keys['google']
        st.sidebar.success("✅ Google Gemini API key loaded from environment")
    else:  # Groq
        api_key = api_keys['groq']
        st.sidebar.success("✅ Groq API key loaded from environment")

    # Model selection
    model_options = api_manager.get_model_options(api_provider)
    model_name = st.sidebar.selectbox(
        f"Select {api_provider} Model",
        model_options,
        index=0
    )

    # Temperature setting
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Controls randomness in responses. Lower values are more deterministic."
    )

    return api_provider, api_key, model_name, temperature


def render_database_info(db):
    """Render database information in an expandable section"""
    with st.expander("📊 Database Information"):
        st.write(f"**Database Type:** {db.dialect}")
        st.write(f"**Available Tables:** {', '.join(db.get_usable_table_names())}")

        # Show sample data for each table
        tables = db.get_usable_table_names()
        selected_table = st.selectbox("Preview table data:", tables)

        if selected_table:
            try:
                sample_data = db.run(f"SELECT * FROM {selected_table} LIMIT 5;")
                st.code(sample_data, language="sql")
            except Exception as e:
                st.error(f"Error fetching sample data: {str(e)}")


def render_query_input():
    """Render the query input section"""
    question = st.text_area(
        "Enter your question about the database:",
        placeholder="e.g., How many employees are there? or Which country's customers spent the most?",
        height=100
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        process_query = st.button("⚙️ Process Query", type="primary")

    with col2:
        if st.button("🧹 Clear"):
            st.rerun()

    return question, process_query


def render_results_tabs(result_data, clean_sql, db_path):
    """Render results in organized tabs"""
    st.subheader("📝 Results")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Answer", "🔍 Generated SQL", "📊 Raw Results", "📋 Formatted Results"])

    with tab1:
        if result_data['success']:
            st.info(result_data['answer'])
        else:
            st.error(f"Error: {result_data['error']}")

    with tab2:
        if clean_sql:
            st.code(clean_sql, language="sql")
            if st.button("📋 Copy SQL", key="copy_sql"):
                st.code(clean_sql)
                st.success("SQL query displayed above for copying!")
        else:
            st.warning("No SQL query generated.")

    with tab3:
        if result_data['raw_result'] is not None:
            st.code(result_data['raw_result'])
        else:
            st.warning("No raw results to display due to SQL execution error.")

    with tab4:
        if result_data['raw_result'] is not None and result_data['success']:
            try:
                if result_data['raw_result'] and result_data['raw_result'] != "[]":
                    # Execute query directly to get pandas DataFrame
                    df = DatabaseManager.execute_sql(clean_sql, db_path)

                    if not df.empty:
                        st.dataframe(df, width="stretch")

                        # Add download button for CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Query executed successfully but returned no data.")
                else:
                    st.info("Query executed successfully but returned empty results.")
            except Exception as e:
                st.error(f"Could not format results as table: {str(e)}")
        else:
            st.warning("No formatted results to display due to SQL execution error.")


def render_example_queries():
    """Render example queries section"""
    with st.expander("💡 Example Queries"):
        examples = [
            "How many employees are there?",
            "List the total sales per country. Which country's customers spent the most?",
            "What are the top 5 best-selling tracks?",
            "How many customers are from Germany?",
            "What is the total revenue for each genre?",
            "Which artist has the most albums?",
            "Show me the invoices from 2009",
            "What are the different media types available?"
        ]

        for example in examples:
            if st.button(f"📌 {example}", key=f"example_{example}"):
                st.session_state.example_query = example
                st.rerun()

        # Handle example query selection
        if hasattr(st.session_state, 'example_query'):
            st.code(st.session_state.example_query)
            if st.button("Use this example"):
                return st.session_state.example_query
    return None


def render_error_instructions():
    """Render instructions when LLM initialization fails"""
    st.error("❌ Failed to initialize the language model. Please check your API key configuration.")

    st.markdown("""
    ## Environment Setup Required

    To use this application, you need to set up environment variables for API keys (as per provider you choose):

    1. **Create a `.env` file** in the project directory
    2. **Add your API keys** to the `.env` file:
       ```
       GOOGLE_API_KEY=your_google_gemini_api_key_here
       GROQ_API_KEY=your_groq_api_key_here
       ```
    3. **Restart the application** after setting up the environment variables

    ## 📊 About the Database

    This tool uses the **Chinook database**, which represents a digital media store including:
    - **Artists** and their albums
    - **Tracks** with genre and media type information
    - **Customers** and their purchase history
    - **Employees** and store information
    - **Invoices** and sales data

    ## 🔑 Get API Keys

    - **Google Gemini**: Get your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
    - **Groq**: Get your free API key at [Groq Console](https://console.groq.com/)
    """)


# ================================
# MAIN APPLICATION LOGIC
# ================================

def main():
    """Main application function"""
    # Setup page configuration
    setup_page_config()

    # Render header
    render_header()

    # Initialize managers
    api_manager = APIManager()
    db_manager = DatabaseManager()

    # Render sidebar and get configuration
    api_provider, api_key, model_name, temperature = render_sidebar(api_manager)

    # Initialize database
    db = db_manager.get_database_connection()
    if db is None:
        st.stop()

    # Render database information
    render_database_info(db)

    # Initialize LLM
    llm_manager = LLMManager()
    llm = llm_manager.initialize_llm(api_provider, api_key, model_name, temperature)

    if llm:
        st.success(f"✅ {api_provider} initialized successfully!")

        # Initialize query engine
        query_engine = QueryEngine(llm, db)

        # Render query input
        question, process_query = render_query_input()

        # Check for example query selection
        example_question = render_example_queries()
        if example_question:
            question = example_question

        # Process query if button clicked and question provided
        if process_query and question:
            with st.spinner("Processing your question..."):
                # Process the query using the query engine
                result_data = query_engine.process_query(question)

                # Render results in tabs
                db_path = os.path.join(os.path.dirname(__file__), "chinook.db")
                render_results_tabs(result_data, result_data['sql_query'], db_path)

    else:
        render_error_instructions()


# Run the application
if __name__ == "__main__":
    main()