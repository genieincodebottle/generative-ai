"""
Customer Call Sentiment & Aggressiveness Tagging with Ollama
===========================================================
A Streamlit application for analyzing customer calls using local Ollama LLM.
Extracts sentiment and aggressiveness scores from customer call data stored in PostgreSQL.

"""

import streamlit as st
import os
import requests
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Database imports
import sqlite3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================================
# DEFAULT CONFIGURATION VALUES
# ================================
DEFAULT_CONFIG = {
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    "ollama_temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
    "db_path": os.getenv("DB_PATH", "sentiment_analysis.db")
}


# ================================
# PYDANTIC MODELS
# ================================

class Classification(BaseModel):
    """Schema for structured sentiment and aggressiveness analysis response"""
    sentiment: str = Field(
        description="The sentiment of the text (Positive, Negative, or Neutral). "
                   "If no clear sentiment is detected, the value should be 'Neutral'"
    )
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10. "
                   "1 means not aggressive at all, 10 means extremely aggressive. "
                   "If no aggressiveness is detected, the value should be 1",
        ge=1, le=10
    )


# ================================
# BACKEND CODE - BUSINESS LOGIC
# ================================

def check_ollama_status(base_url: str) -> bool:
    """Check if Ollama server is running and accessible"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def initialize_llm(base_url: str, model: str, temperature: float) -> Optional[OllamaLLM]:
    """Initialize Ollama LLM with given configuration"""
    try:
        llm = OllamaLLM(
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=30  # Add timeout
        )
        # Test the connection with a simple prompt
        test_response = llm.invoke("Hello")
        if not test_response:
            raise Exception("LLM returned empty response")
        return llm
    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to Ollama at {base_url}. Make sure Ollama is running.")
        return None
    except Exception as e:
        st.error(f"Failed to initialize LLM '{model}': {str(e)}")
        return None


def initialize_database(config: Dict[str, Any]) -> bool:
    """Initialize SQLite database with required tables"""
    try:
        connection = sqlite3.connect(config["db_path"])
        cursor = connection.cursor()

        # Create customer_calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                call_details TEXT NOT NULL,
                call_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create customer_tagging table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_tagging (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id INTEGER UNIQUE,
                call_details TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                aggressiveness INTEGER NOT NULL,
                tagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES customer_calls (id)
            )
        """)

        # Insert sample data if customer_calls table is empty
        cursor.execute("SELECT COUNT(*) FROM customer_calls")
        if cursor.fetchone()[0] == 0:
            sample_data = [
                (101, 'Customer expressed happiness about the quick resolution of their billing issue.'),
                (102, 'Customer was delighted with the advanced features of our new product.'),
                (103, 'Customer was upset about the delayed shipment and requested an expedited delivery.'),
                (104, 'Customer felt neutral and inquired about the specifications of various models.'),
                (105, 'Customer expressed dissatisfaction with the recent service, noting multiple unresolved issues.'),
                (106, 'Customer joyfully reported that our product exceeded their expectations.'),
                (107, 'Customer was frustrated due to a misunderstanding about warranty coverage.'),
                (108, 'Customer was indifferent when discussing the upcoming software update details.'),
                (109, 'Customer was angry about receiving the wrong order and demanded a prompt resolution.'),
                (110, 'Customer was thrilled to hear about our loyalty program upgrades.')
            ]
            cursor.executemany(
                "INSERT INTO customer_calls (customer_id, call_details) VALUES (?, ?)",
                sample_data
            )

        connection.commit()
        connection.close()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")
        return False


def get_database_connection(config: Dict[str, Any]):
    """Create SQLite database connection with error handling"""
    try:
        db_path = config["db_path"]
        connection = sqlite3.connect(db_path, timeout=10)
        connection.row_factory = sqlite3.Row  # Enable column access by name
        return connection
    except sqlite3.Error as e:
        st.error(f"Database connection failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None


def fetch_customer_calls(config: Dict[str, Any]) -> Optional[List[Dict]]:
    """Fetch customer calls from database"""
    try:
        connection = get_database_connection(config)
        if not connection:
            return None

        cursor = connection.cursor()
        cursor.execute("SELECT * FROM customer_calls ORDER BY id")
        records = cursor.fetchall()
        return [dict(record) for record in records]
    except sqlite3.OperationalError:
        # Tables don't exist yet
        return None
    except Exception as e:
        st.error(f"Error fetching customer calls: {str(e)}")
        return None
    finally:
        if connection:
            connection.close()


def save_tagging_result(call_id: int, call_details: str, sentiment: str, aggressiveness: int, config: Dict[str, Any]) -> bool:
    """Save tagging results to database with proper error handling"""
    try:
        connection = get_database_connection(config)
        if not connection:
            return False

        cursor = connection.cursor()
        # Use parameterized queries to prevent SQL injection
        # SQLite uses INSERT OR REPLACE for upsert functionality
        insert_query = """
            INSERT OR REPLACE INTO customer_tagging (call_id, call_details, sentiment, aggressiveness, tagged_at)
            VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, (call_id, call_details, sentiment, aggressiveness, datetime.now().isoformat()))
        connection.commit()
        return True
    except Exception as e:
        st.error(f"Error saving tagging result: {str(e)}")
        return False
    finally:
        if connection:
            connection.close()


def get_tagging_results(config: Dict[str, Any]) -> Optional[List[Dict]]:
    """Fetch existing tagging results"""
    try:
        connection = get_database_connection(config)
        if not connection:
            return None

        cursor = connection.cursor()
        cursor.execute("""
            SELECT ct.*, cc.customer_id, cc.call_time
            FROM customer_tagging ct
            JOIN customer_calls cc ON ct.call_id = cc.id
            ORDER BY ct.tagged_at DESC
        """)
        records = cursor.fetchall()
        return [dict(record) for record in records]
    except sqlite3.OperationalError:
        # Tables don't exist yet
        return None
    except Exception as e:
        st.error(f"Error fetching tagging results: {str(e)}")
        return None
    finally:
        if connection:
            connection.close()


def validate_call_content(text: str) -> bool:
    """Validate call content before processing"""
    if not text or not text.strip():
        return False
    if len(text.strip()) < 10:
        return False
    if len(text) > 10000:  # Limit text length
        return False
    return True


def analyze_sentiment_and_aggressiveness(text: str, llm: OllamaLLM) -> Optional[Dict[str, Any]]:
    """Analyze text for sentiment and aggressiveness using LLM"""
    try:
        # Validate input
        if not validate_call_content(text):
            st.warning(f"Invalid call content: text too short, too long, or empty")
            return None

        # Create prompt template
        tagging_prompt = PromptTemplate.from_template("""
        You are a customer service analyst. Analyze the following customer call transcript for sentiment and aggressiveness.

        Customer call transcript:
        "{input}"

        Provide your analysis in the following JSON format:
        {{
            "sentiment": "<Positive, Negative, or Neutral>",
            "aggressiveness": <number from 1 to 10>
        }}

        Analysis Guidelines:

        SENTIMENT:
        - Positive: Customer is happy, satisfied, appreciative, or expressing gratitude
        - Negative: Customer is upset, frustrated, disappointed, or complaining
        - Neutral: Customer is calm, matter-of-fact, or simply requesting information

        AGGRESSIVENESS (1-10 scale):
        - 1-2: Very polite, calm, respectful tone
        - 3-4: Slightly concerned but still polite
        - 5-6: Moderately frustrated, some impatience evident
        - 7-8: Clearly angry, demanding, using strong language
        - 9-10: Very hostile, threatening, extremely rude or abusive

        Key indicators of aggressiveness:
        - Demanding immediate action
        - Using words like "angry", "furious", "unacceptable"
        - Making threats or ultimatums
        - Excessive complaints or blame
        - Interrupting or being disrespectful

        Analyze the tone, word choice, and emotional intensity to determine the appropriate aggressiveness score.
        """)

        # Create parser
        parser = JsonOutputParser(pydantic_object=Classification)

        # Create chain
        chain = tagging_prompt | llm | parser

        # Process the text
        result = chain.invoke({"input": text})
        return result
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


# ================================
# UI CODE - USER INTERFACE
# ================================

def setup_page():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Customer Call Sentiment Analyzer",
        page_icon="📞",
        layout="wide"
    )

    # Hide Streamlit style
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def render_header():
    """Render page header"""
    st.subheader("📞 Customer Call Sentiment Analyzer")
    st.markdown("Analyze customer call sentiment and aggressiveness using local Ollama LLM")


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and defaults"""
    config = DEFAULT_CONFIG.copy()

    # Override with any environment variables that are set
    env_mappings = {
        "OLLAMA_BASE_URL": "ollama_base_url",
        "OLLAMA_MODEL": "ollama_model",
        "OLLAMA_TEMPERATURE": "ollama_temperature",
        "DB_PATH": "db_path"
    }

    for env_var, config_key in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            if config_key == "ollama_temperature":
                config[config_key] = float(env_value)
            else:
                config[config_key] = env_value

    return config


def render_sidebar():
    """Render sidebar with configuration"""
    st.sidebar.title("⚙️ Settings")

    # Initialize session state for config
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    # Status section
    st.sidebar.header("📊 Status")

    # Check Ollama status
    ollama_url = st.session_state.config.get("ollama_base_url", DEFAULT_CONFIG["ollama_base_url"])
    if check_ollama_status(ollama_url):
        st.sidebar.success("✅ Ollama running")
    else:
        st.sidebar.error("❌ Ollama not running")
        with st.sidebar.expander("🚀 Setup Instructions"):
            st.markdown("""
            **Quick setup:**
            1. Install from [ollama.com](https://ollama.com)
            2. Run: `ollama serve`
            3. Pull model: `ollama pull llama3.2:1b`
            4. Refresh this page
            """)

    # Model settings
    st.sidebar.header("🤖 Model Settings")
    st.session_state.config["ollama_model"] = st.sidebar.selectbox(
        "Model",
        options=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "gemma2:2b", "gemma2:9b"],
        index=0,
        help="Choose model based on your RAM"
    )

    st.session_state.config["ollama_temperature"] = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.config["ollama_temperature"],
        step=0.1,
        help="0 = focused, 1 = creative"
    )

    # Advanced settings in expander
    with st.sidebar.expander("🔧 Advanced Settings"):
        st.session_state.config["ollama_base_url"] = st.text_input(
            "Ollama URL",
            value=st.session_state.config["ollama_base_url"]
        )

    # Database settings
    st.sidebar.header("🗄️ Database Settings")
    with st.sidebar.expander("Database Configuration"):
        st.session_state.config["db_path"] = st.text_input(
            "Database Path", value=st.session_state.config.get("db_path", "sentiment_analysis.db"),
            help="SQLite database file path (from DB_PATH env var if set)"
        )

        # Show database file info
        db_path = st.session_state.config["db_path"]
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            st.sidebar.info(f"📁 Database exists ({file_size} bytes)")
        else:
            st.sidebar.info("📁 Database will be created on first use")

    return st.session_state.config


def main():
    """Main application function"""
    # Setup page
    setup_page()
    render_header()

    # Initialize session state
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    # Render sidebar and get configuration
    config = render_sidebar()

    # Check Ollama status
    if not check_ollama_status(config["ollama_base_url"]):
        st.error("❌ Ollama not running. Check sidebar for setup instructions.")
        st.stop()

    # Initialize LLM if needed
    current_llm_config = f"{config['ollama_base_url']}_{config['ollama_model']}_{config['ollama_temperature']}"
    if (st.session_state.llm is None or
        st.session_state.get('llm_config') != current_llm_config):
        with st.spinner(f"⚡ Initializing {config['ollama_model']}..."):
            st.session_state.llm = initialize_llm(
                config["ollama_base_url"],
                config["ollama_model"],
                config["ollama_temperature"]
            )
            if st.session_state.llm is None:
                st.stop()
            st.session_state.llm_config = current_llm_config
        # st.toast("✅ Model initialized!", icon="🤖")  # Remove overwhelming messages

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🗄️ Setup Database", "📞 Process Calls", "📊 View Results", "ℹ️ Database Info"])

    with tab1:
        st.header("Database Setup")

        # Check if database exists
        db_path = config["db_path"]
        db_exists = os.path.exists(db_path)

        if db_exists:
            connection = get_database_connection(config)
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM customer_calls")
                    call_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM customer_tagging")
                    tagging_count = cursor.fetchone()[0]
                    connection.close()

                    st.success(f"✅ Database ready! {call_count} calls available, {tagging_count} analyzed.")

                    if st.button("🔄 Reset Database (Clear All Data)"):
                        if os.path.exists(db_path):
                            os.remove(db_path)
                            st.rerun()
                except sqlite3.OperationalError:
                    connection.close()
                    # Database file exists but tables are missing - show as needs initialization
                    st.warning("⚠️ Database file exists but tables are missing. Please initialize below.")
                    if st.button("🚀 Initialize Database", type="primary"):
                        with st.spinner("Creating database and loading sample data..."):
                            if initialize_database(config):
                                st.success("✅ Database created successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to create database")
            else:
                st.error("❌ Database file exists but cannot connect")
        else:
            st.info("📋 No database found. Click below to create it with sample data.")

            if st.button("🚀 Initialize Database", type="primary"):
                with st.spinner("Creating database and loading sample data..."):
                    if initialize_database(config):
                        st.success("✅ Database created successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to create database")

    with tab2:
        st.subheader("Process Customer Calls")

        # Check if database exists before processing
        if not os.path.exists(config["db_path"]):
            st.warning("⚠️ Please initialize the database first in the 'Setup Database' tab.")
        else:
            if st.button("🔄 Fetch and Process All Calls", type="primary"):
                with st.spinner("Fetching customer calls..."):
                    calls = fetch_customer_calls(config)

                if calls:
                    st.info(f"Found {len(calls)} customer calls - starting analysis...")

                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    processed = 0
                    for i, call in enumerate(calls):
                        status_text.text(f"Processing call {i+1}/{len(calls)} (Customer ID: {call['customer_id']})")

                        # Skip if call_details is empty or invalid
                        if not call.get('call_details') or not call['call_details'].strip():
                            st.session_state.processing_status[call['id']] = {
                                'status': 'failed',
                                'error': 'Empty or invalid call details'
                            }
                            progress_bar.progress((i + 1) / len(calls))
                            continue

                        # Analyze the call
                        result = analyze_sentiment_and_aggressiveness(call['call_details'], st.session_state.llm)

                        if result:
                            # Save to database
                            if save_tagging_result(
                                call['id'],
                                call['call_details'],
                                result['sentiment'],
                                result['aggressiveness'],
                                config
                            ):
                                processed += 1
                                st.session_state.processing_status[call['id']] = {
                                    'status': 'success',
                                    'result': result
                                }
                            else:
                                st.session_state.processing_status[call['id']] = {
                                    'status': 'failed',
                                    'error': 'Failed to save to database'
                                }
                        else:
                            st.session_state.processing_status[call['id']] = {
                                'status': 'failed',
                                'error': 'Failed to analyze call - invalid content or LLM error'
                            }

                        progress_bar.progress((i + 1) / len(calls))

                    status_text.text(f"Processing complete! Successfully processed {processed}/{len(calls)} calls")
                    if processed > 0:
                        st.success(f"✅ Analysis complete! {processed}/{len(calls)} calls processed successfully")
                    if processed < len(calls):
                        failed = len(calls) - processed
                        st.warning(f"⚠️ {failed} calls failed processing - check details below")
                else:
                    st.error("No customer calls found. Please check database setup.")

        # Show individual call processing status if any (only failures)
        if st.session_state.processing_status:
            failed_calls = {k: v for k, v in st.session_state.processing_status.items() if v['status'] == 'failed'}
            if failed_calls:
                with st.expander(f"Failed Calls ({len(failed_calls)})", expanded=False):
                    for call_id, status_info in failed_calls.items():
                        st.error(f"Call {call_id}: {status_info.get('error', 'Unknown error')}")

    with tab3:
        st.header("Tagging Results")

        # Auto-load results on tab open
        if st.button("🔄 Refresh Results"):
            st.rerun()

        # Always try to load and display results
        results = get_tagging_results(config)
        if results:
            st.dataframe(
                    results,
                    column_config={
                        "call_details": st.column_config.TextColumn("Call Details", width="large"),
                        "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                        "aggressiveness": st.column_config.NumberColumn("Aggressiveness", width="small"),
                        "tagged_at": st.column_config.DatetimeColumn("Tagged At", width="medium")
                    }
                )

            # Show some statistics
            st.subheader("📈 Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                sentiment_counts = {}
                for r in results:
                    sentiment_counts[r['sentiment']] = sentiment_counts.get(r['sentiment'], 0) + 1
                st.metric("Total Calls Analyzed", len(results))
                for sentiment, count in sentiment_counts.items():
                    st.metric(f"{sentiment} Calls", count)

            with col2:
                aggression_scores = [r['aggressiveness'] for r in results]
                if aggression_scores:
                    avg_aggression = sum(aggression_scores) / len(aggression_scores)
                    st.metric("Average Aggressiveness", f"{avg_aggression:.1f}")
                    st.metric("Max Aggressiveness", max(aggression_scores))

            with col3:
                high_aggression = len([r for r in results if r['aggressiveness'] >= 7])
                st.metric("High Aggression Calls (≥7)", high_aggression)
        else:
            st.info("📊 No analysis results found. Process some calls first in the 'Process Calls' tab.")

    with tab4:
        st.header("Database Information")
        st.markdown("""
        ### SQLite Database Schema:

        **customer_calls** - Source data table
        ```sql
        CREATE TABLE customer_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            call_details TEXT NOT NULL,
            call_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ```

        **customer_tagging** - Results table
        ```sql
        CREATE TABLE customer_tagging (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id INTEGER UNIQUE,
            call_details TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            aggressiveness INTEGER NOT NULL,
            tagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (call_id) REFERENCES customer_calls (id)
        );
        ```

        **Note**: Database and sample data are created automatically on first use.
        """)

        # Test database connection and initialize
        if st.button("🔗 Test & Initialize Database"):
            if initialize_database(config):
                connection = get_database_connection(config)
                if connection:
                    try:
                        cursor = connection.cursor()
                        # Test if required tables exist and have data
                        cursor.execute("""
                            SELECT name FROM sqlite_master
                            WHERE type='table' AND name IN ('customer_calls', 'customer_tagging')
                        """)
                        tables = [row[0] for row in cursor.fetchall()]

                        cursor.execute("SELECT COUNT(*) FROM customer_calls")
                        call_count = cursor.fetchone()[0]

                        cursor.execute("SELECT COUNT(*) FROM customer_tagging")
                        tagging_count = cursor.fetchone()[0]

                        if len(tables) == 2:
                            st.success(f"✅ Database ready! Found {call_count} calls, {tagging_count} analyzed.")
                        else:
                            missing = [t for t in ['customer_calls', 'customer_tagging'] if t not in tables]
                            st.warning(f"⚠️ Missing tables: {', '.join(missing)}")
                    except Exception as e:
                        st.error(f"❌ Database query failed: {str(e)}")
                    finally:
                        connection.close()
                else:
                    st.error("❌ Database connection failed!")
            else:
                st.error("❌ Database initialization failed!")


if __name__ == "__main__":
    main()