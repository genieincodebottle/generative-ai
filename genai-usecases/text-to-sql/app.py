import streamlit as st
import sqlite3
import pandas as pd
import os
import warnings
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import re

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Text-to-SQL Query Tool",
    page_icon="🗃️",
    layout="wide"
)

st.title("🗃️ Text-to-SQL Query Tool")
st.markdown("Convert natural language questions into SQL queries and get answers from your database.")

# Load API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Sidebar for API configuration
st.sidebar.header("🛠️ API Configuration")

# Check which APIs are available based on environment variables
available_providers = []
if GOOGLE_API_KEY:
    available_providers.append("Google Gemini")
if GROQ_API_KEY:
    available_providers.append("Groq")

if not available_providers:
    st.error("❌ No API keys found in environment variables. Please set GOOGLE_API_KEY and/or GROQ_API_KEY in your .env file.")
    st.stop()

# API Provider Selection (only show available providers)
api_provider = st.sidebar.selectbox(
    "Select API Provider",
    available_providers,
    index=0
)

# Set API key and model options based on provider
if api_provider == "Google Gemini":
    api_key = GOOGLE_API_KEY
    model_name = st.sidebar.selectbox(
        "Select Gemini Model",
        ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )
    st.sidebar.success("✅ Google Gemini API key loaded from environment")
else:  # Groq
    api_key = GROQ_API_KEY
    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        index=0
    )
    st.sidebar.success("✅ Groq API key loaded from environment")

# Temperature setting
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Controls randomness in responses. Lower values are more deterministic."
)

# Database connection
@st.cache_resource
def get_database_connection():
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

# Initialize database
db = get_database_connection()

if db is None:
    st.stop()

# Display database info
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

# Initialize LLM based on provider
def get_llm(provider, api_key, model, temp):
    if not api_key:
        return None

    try:
        if provider == "Google Gemini":
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temp,
                convert_system_message_to_human=True
            )
        else:  # Groq
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model,
                temperature=temp
            )
    except Exception as e:
        st.error(f"Error initializing {provider}: {str(e)}")
        return None

# Main application
llm = get_llm(api_provider, api_key, model_name, temperature)

if llm:
    st.success(f"✅ {api_provider} initialized successfully!")

    # Query input
    question = st.text_area(
        "Enter your question about the database:",
        placeholder="e.g., How many employees are there? or Which country's customers spent the most?",
        height=100
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        process_query = st.button("🚀 Process Query", type="primary")

    with col2:
        if st.button("🧹 Clear"):
            st.rerun()

    if process_query and question:
        with st.spinner("Processing your question..."):
            try:
                # Create the SQL query chain
                template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the {top_k} answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

Provide SQL query as a simple string without any markdown formatting.

Question: {input}'''

                # Create a simple SQL query chain without the complex chaining
                write_query = create_sql_query_chain(llm, db)

                # Get the SQL query
                sql_query = write_query.invoke({"question": question})

                # Clean SQL query (remove markdown formatting - foolproof method)
                def clean_sql_query(query):
                    """Remove all possible markdown formatting from SQL query"""
                    if not query:
                        return ""

                    # Convert to string and strip
                    clean = str(query).strip()

                    # Remove markdown code blocks - handle all variations
                    # Pattern 1: ```sql ... ```
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

                clean_sql = clean_sql_query(sql_query)

                # Execute query to get raw results
                try:
                    raw_result = db.run(clean_sql)
                except Exception as e:
                    st.error(f"SQL execution error: {str(e)}")
                    raw_result = None

                # Generate answer using the cleaned results
                if raw_result is not None:
                    answer_prompt = PromptTemplate.from_template(
                        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
                    )

                    answer_chain = answer_prompt | llm | StrOutputParser()
                    result = answer_chain.invoke({
                        "question": question,
                        "query": clean_sql,
                        "result": raw_result
                    })
                else:
                    result = "Unable to execute the SQL query due to an error."

                # Display results in tabs
                st.subheader("📝 Results")

                # Create tabs
                tab1, tab2, tab3, tab4 = st.tabs(["💬 Answer", "🔍 Generated SQL", "📊 Raw Results", "📋 Formatted Results"])

                with tab1:
                    st.info(result)

                with tab2:
                    st.code(clean_sql, language="sql")
                    if st.button("📋 Copy SQL", key="copy_sql"):
                        st.code(clean_sql)
                        st.success("SQL query displayed above for copying!")

                with tab3:
                    if raw_result is not None:
                        st.code(raw_result)
                    else:
                        st.warning("No raw results to display due to SQL execution error.")

                with tab4:
                    if raw_result is not None:
                        try:
                            if raw_result and raw_result != "[]":
                                # Execute query directly to get pandas DataFrame
                                conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "chinook.db"))
                                df = pd.read_sql_query(clean_sql, conn)
                                conn.close()

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

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

        # Example queries section
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
                    question = st.session_state.example_query
else:
    st.error("❌ Failed to initialize the language model. Please check your API key configuration.")

    # Instructions for environment setup
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

    - **Google Gemini**: Get your free API key at [Google AI Studio](https://aistudio.google.com/apikey)
    - **Groq**: Get your free API key at [Groq Console](https://console.groq.com/)
    """)