"""
Customer Call Sentiment & Aggressiveness Tagging
=================================================
A Streamlit application for analyzing customer calls using Cloud LLMs (Groq / Gemini).
Extracts sentiment and aggressiveness scores from customer call data stored in SQLite.
Supports dual LLM providers: Groq (primary, free tier) and Google Gemini (secondary).
"""

import streamlit as st
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Provider imports (graceful)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Database imports
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# ================================
# CONSTANTS & CONFIGURATION
# ================================

PROVIDER_CONFIG = {
    "groq": {
        "label": "Groq (Free - Recommended)",
        "env_key": "GROQ_API_KEY",
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"],
        "config_model_key": "groq_model",
        "available": GROQ_AVAILABLE,
        "key_url": "[console.groq.com](https://console.groq.com/keys)",
        "pkg": "langchain-groq",
    },
    "gemini": {
        "label": "Google Gemini",
        "env_key": "GOOGLE_API_KEY",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "config_model_key": "gemini_model",
        "available": GEMINI_AVAILABLE,
        "key_url": "[aistudio.google.com](https://aistudio.google.com/apikey)",
        "pkg": "langchain-google-genai",
    },
}

DEFAULT_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "groq"),
    "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
    "db_path": os.getenv("DB_PATH", "sentiment_analysis.db"),
}

SAMPLE_CALLS = [
    (101, "Customer expressed happiness about the quick resolution of their billing issue."),
    (102, "Customer was delighted with the advanced features of our new product."),
    (103, "Customer was upset about the delayed shipment and requested an expedited delivery."),
    (104, "Customer felt neutral and inquired about the specifications of various models."),
    (105, "Customer expressed dissatisfaction with the recent service, noting multiple unresolved issues."),
    (106, "Customer joyfully reported that our product exceeded their expectations."),
    (107, "Customer was frustrated due to a misunderstanding about warranty coverage."),
    (108, "Customer was indifferent when discussing the upcoming software update details."),
    (109, "Customer was angry about receiving the wrong order and demanded a prompt resolution."),
    (110, "Customer was thrilled to hear about our loyalty program upgrades."),
]

TAGGING_PROMPT = PromptTemplate.from_template("""
You are a customer service analyst. Analyze the following customer call transcript for sentiment and aggressiveness.

Customer call transcript:
"{input}"

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

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no reasoning, no extra text.

{format_instructions}
""")

MIN_CALL_LENGTH = 10
MAX_CALL_LENGTH = 10_000


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
# DATABASE LAYER
# ================================

@contextmanager
def db_connection(db_path: str):
    """Context manager for safe SQLite connections — auto-closes on exit."""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def initialize_database(db_path: str) -> bool:
    """Create tables and seed sample data if empty."""
    try:
        with db_connection(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    call_details TEXT NOT NULL,
                    call_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

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

            cursor.execute("SELECT COUNT(*) FROM customer_calls")
            if cursor.fetchone()[0] == 0:
                cursor.executemany(
                    "INSERT INTO customer_calls (customer_id, call_details) VALUES (?, ?)",
                    SAMPLE_CALLS,
                )

            conn.commit()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False


def fetch_customer_calls(db_path: str) -> Optional[List[Dict]]:
    """Return all customer calls, or None if tables don't exist yet."""
    try:
        with db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customer_calls ORDER BY id")
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return None
    except Exception as e:
        st.error(f"Error fetching customer calls: {e}")
        return None


def save_tagging_result(
    call_id: int, call_details: str, sentiment: str, aggressiveness: int, db_path: str
) -> bool:
    """Upsert a tagging result for a given call."""
    try:
        with db_connection(db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO customer_tagging
                   (call_id, call_details, sentiment, aggressiveness, tagged_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (call_id, call_details, sentiment, aggressiveness, datetime.now().isoformat()),
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving tagging result: {e}")
        return False


def get_tagging_results(db_path: str) -> Optional[List[Dict]]:
    """Return all tagging results joined with call metadata."""
    try:
        with db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ct.*, cc.customer_id, cc.call_time
                FROM customer_tagging ct
                JOIN customer_calls cc ON ct.call_id = cc.id
                ORDER BY ct.tagged_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return None
    except Exception as e:
        st.error(f"Error fetching tagging results: {e}")
        return None


def get_table_counts(db_path: str) -> Optional[tuple]:
    """Return (call_count, tagging_count) or None if tables are missing."""
    try:
        with db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customer_calls")
            call_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM customer_tagging")
            tagging_count = cursor.fetchone()[0]
            return call_count, tagging_count
    except sqlite3.OperationalError:
        return None


# ================================
# LLM LAYER
# ================================

def get_api_key(provider: str) -> Optional[str]:
    """Return the API key for the provider, or None if unset."""
    env_key = PROVIDER_CONFIG[provider]["env_key"]
    return os.getenv(env_key) or None


def initialize_llm(provider: str, model: str, temperature: float):
    """Create and test an LLM instance for the given provider."""
    prov = PROVIDER_CONFIG[provider]

    if not prov["available"]:
        st.error(f"{prov['pkg']} is not installed. Run: `pip install {prov['pkg']}`")
        return None

    api_key = get_api_key(provider)
    if not api_key:
        st.error(f"{prov['env_key']} not found. Please set it in your `.env` file.")
        return None

    try:
        if provider == "groq":
            llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        else:
            llm = ChatGoogleGenerativeAI(
                model=model, temperature=temperature, google_api_key=api_key
            )

        # Quick connectivity test
        llm.invoke("Say hello in one word")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize {provider} / {model}: {e}")
        return None


def analyze_sentiment(text: str, llm) -> Optional[Dict[str, Any]]:
    """Run the sentiment + aggressiveness chain on a single text."""
    stripped = text.strip() if text else ""
    if not stripped or len(stripped) < MIN_CALL_LENGTH or len(text) > MAX_CALL_LENGTH:
        st.warning("Invalid call content: text too short, too long, or empty")
        return None

    try:
        parser = JsonOutputParser(pydantic_object=Classification)
        chain = TAGGING_PROMPT | llm | parser
        return chain.invoke({"input": text, "format_instructions": parser.get_format_instructions()})
    except Exception as e:
        st.error(f"Error analyzing text: {e}")
        return None


# ================================
# UI HELPERS
# ================================

def _init_database_button(db_path: str):
    """Reusable 'Initialize Database' button + spinner."""
    if st.button("🚀 Initialize Database", type="primary"):
        with st.spinner("Creating database and loading sample data..."):
            if initialize_database(db_path):
                st.success("✅ Database created successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to create database")


# ================================
# UI TABS
# ================================

def render_tab_setup(config: Dict[str, Any]):
    """Tab 1 — Setup Database"""
    st.header("Database Setup")
    db_path = config["db_path"]

    if not os.path.exists(db_path):
        st.info("📋 No database found. Click below to create it with sample data.")
        _init_database_button(db_path)
        return

    counts = get_table_counts(db_path)
    if counts is None:
        st.warning("⚠️ Database file exists but tables are missing. Please initialize below.")
        _init_database_button(db_path)
        return

    call_count, tagging_count = counts
    st.success(f"✅ Database ready! {call_count} calls available, {tagging_count} analyzed.")
    if st.button("🔄 Reset Database (Clear All Data)"):
        os.remove(db_path)
        st.rerun()


def render_tab_process(config: Dict[str, Any]):
    """Tab 2 — Process Calls"""
    st.subheader("Process Customer Calls")
    db_path = config["db_path"]

    if not os.path.exists(db_path):
        st.warning("⚠️ Please initialize the database first in the 'Setup Database' tab.")
        return

    if not st.button("🔄 Fetch and Process All Calls", type="primary"):
        _show_failed_calls()
        return

    with st.spinner("Fetching customer calls..."):
        calls = fetch_customer_calls(db_path)

    if not calls:
        st.error("No customer calls found. Please check database setup.")
        return

    st.info(f"Found {len(calls)} customer calls - starting analysis...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed = 0

    for i, call in enumerate(calls):
        status_text.text(f"Processing call {i+1}/{len(calls)} (Customer ID: {call['customer_id']})")

        if not call.get("call_details", "").strip():
            _record_failure(call["id"], "Empty or invalid call details")
            progress_bar.progress((i + 1) / len(calls))
            continue

        result = analyze_sentiment(call["call_details"], st.session_state.llm)

        if result and save_tagging_result(
            call["id"], call["call_details"], result["sentiment"], result["aggressiveness"], db_path
        ):
            processed += 1
            st.session_state.processing_status[call["id"]] = {"status": "success", "result": result}
        else:
            error = "Failed to save to database" if result else "LLM analysis failed"
            _record_failure(call["id"], error)

        progress_bar.progress((i + 1) / len(calls))

    status_text.text(f"Processing complete! Successfully processed {processed}/{len(calls)} calls")
    if processed > 0:
        st.success(f"✅ Analysis complete! {processed}/{len(calls)} calls processed successfully")
    if processed < len(calls):
        st.warning(f"⚠️ {len(calls) - processed} calls failed processing - check details below")

    _show_failed_calls()


def _record_failure(call_id: int, error: str):
    """Helper to record a processing failure in session state."""
    st.session_state.processing_status[call_id] = {"status": "failed", "error": error}


def _show_failed_calls():
    """Display failed calls in an expander if any exist."""
    failed = {k: v for k, v in st.session_state.processing_status.items() if v["status"] == "failed"}
    if failed:
        with st.expander(f"Failed Calls ({len(failed)})", expanded=False):
            for call_id, info in failed.items():
                st.error(f"Call {call_id}: {info.get('error', 'Unknown error')}")


def render_tab_results(config: Dict[str, Any]):
    """Tab 3 — View Results"""
    st.header("Tagging Results")

    if st.button("🔄 Refresh Results"):
        st.rerun()

    results = get_tagging_results(config["db_path"])
    if not results:
        st.info("📊 No analysis results found. Process some calls first in the 'Process Calls' tab.")
        return

    st.dataframe(
        results,
        column_config={
            "call_details": st.column_config.TextColumn("Call Details", width="large"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "aggressiveness": st.column_config.NumberColumn("Aggressiveness", width="small"),
            "tagged_at": st.column_config.DatetimeColumn("Tagged At", width="medium"),
        },
    )

    st.subheader("📈 Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        sentiment_counts = {}
        for r in results:
            sentiment_counts[r["sentiment"]] = sentiment_counts.get(r["sentiment"], 0) + 1
        st.metric("Total Calls Analyzed", len(results))
        for sentiment, count in sentiment_counts.items():
            st.metric(f"{sentiment} Calls", count)

    with col2:
        scores = [r["aggressiveness"] for r in results]
        st.metric("Average Aggressiveness", f"{sum(scores) / len(scores):.1f}")
        st.metric("Max Aggressiveness", max(scores))

    with col3:
        st.metric("High Aggression Calls (>=7)", len([s for s in scores if s >= 7]))


def render_tab_db_info(config: Dict[str, Any]):
    """Tab 4 — Database Info"""
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

    if st.button("🔗 Test & Initialize Database"):
        db_path = config["db_path"]
        if not initialize_database(db_path):
            st.error("❌ Database initialization failed!")
            return

        counts = get_table_counts(db_path)
        if counts:
            st.success(f"✅ Database ready! Found {counts[0]} calls, {counts[1]} analyzed.")
        else:
            st.error("❌ Tables could not be verified after initialization.")


# ================================
# SIDEBAR
# ================================

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar with provider, model, and DB settings. Returns config dict."""
    st.sidebar.title("⚙️ Settings")

    if "config" not in st.session_state:
        st.session_state.config = DEFAULT_CONFIG.copy()

    # --- Provider Selection ---
    st.sidebar.header("🔌 LLM Provider")

    available = {k: v for k, v in PROVIDER_CONFIG.items() if v["available"]}
    if not available:
        st.sidebar.error("No LLM provider installed! Install langchain-groq or langchain-google-genai.")
        st.stop()

    provider_ids = list(available.keys())
    provider_labels = [available[p]["label"] for p in provider_ids]

    current = st.session_state.config.get("provider", provider_ids[0])
    default_idx = provider_ids.index(current) if current in provider_ids else 0

    selected_label = st.sidebar.selectbox(
        "Provider", options=provider_labels, index=default_idx,
        help="Groq offers a generous free tier — great for getting started",
    )
    selected_provider = provider_ids[provider_labels.index(selected_label)]
    st.session_state.config["provider"] = selected_provider
    prov = PROVIDER_CONFIG[selected_provider]

    # --- API Key Status ---
    st.sidebar.header("🔑 API Key Status")
    if get_api_key(selected_provider):
        st.sidebar.success(f"✅ {prov['env_key']} is set")
    else:
        st.sidebar.error(f"❌ {prov['env_key']} not set")
        st.sidebar.markdown(f"Get your key → {prov['key_url']}")

    # --- Model Settings ---
    st.sidebar.header("🤖 Model Settings")
    model_key = prov["config_model_key"]
    models = prov["models"]
    current_model = st.session_state.config.get(model_key, models[0])
    default_model_idx = models.index(current_model) if current_model in models else 0

    st.session_state.config[model_key] = st.sidebar.selectbox(
        "Model", options=models, index=default_model_idx, help="Choose a model based on your needs",
    )

    st.session_state.config["temperature"] = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.0,
        value=st.session_state.config["temperature"], step=0.1,
        help="0 = focused & deterministic, 1 = creative",
    )

    # --- Database Settings ---
    with st.sidebar.expander("🗄️ Database Settings"):
        st.session_state.config["db_path"] = st.text_input(
            "Database Path",
            value=st.session_state.config.get("db_path", "sentiment_analysis.db"),
            help="SQLite database file path",
        )
        db_path = st.session_state.config["db_path"]
        if os.path.exists(db_path):
            st.info(f"📁 Database exists ({os.path.getsize(db_path)} bytes)")
        else:
            st.info("📁 Database will be created on first use")

    return st.session_state.config


# ================================
# MAIN
# ================================

def main():
    """Application entry point."""
    st.set_page_config(page_title="Customer Call Sentiment Analyzer", page_icon="📞", layout="wide")
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )
    st.subheader("📞 Customer Call Sentiment Analyzer")
    st.markdown("Analyze customer call sentiment and aggressiveness using Cloud LLMs (Groq / Gemini)")

    # Session state defaults
    st.session_state.setdefault("processing_status", {})
    st.session_state.setdefault("llm", None)

    config = render_sidebar()

    # Resolve current provider + model
    provider = config["provider"]
    prov = PROVIDER_CONFIG[provider]
    model = config[prov["config_model_key"]]

    # Gate on API key
    if not get_api_key(provider):
        st.error(f"❌ API key not configured for {provider}. Set it in your `.env` file. See sidebar for details.")
        st.stop()

    # (Re-)initialize LLM when config changes
    llm_fingerprint = f"{provider}_{model}_{config['temperature']}"
    if st.session_state.llm is None or st.session_state.get("llm_config") != llm_fingerprint:
        with st.spinner(f"⚡ Initializing {provider} / {model}..."):
            st.session_state.llm = initialize_llm(provider, model, config["temperature"])
            if st.session_state.llm is None:
                st.stop()
            st.session_state.llm_config = llm_fingerprint

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗄️ Setup Database", "📞 Process Calls", "📊 View Results", "ℹ️ Database Info"]
    )
    with tab1:
        render_tab_setup(config)
    with tab2:
        render_tab_process(config)
    with tab3:
        render_tab_results(config)
    with tab4:
        render_tab_db_info(config)


if __name__ == "__main__":
    main()
