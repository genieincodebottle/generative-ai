"""Streamlit UI: widgets and HTTP calls, nothing else.

Every decision this file makes is a presentation decision. SQL generation,
validation, and execution all live behind the FastAPI service.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 120

st.set_page_config(page_title="Text-to-SQL Query App", layout="wide")


# --------------------------------------------------------------------------
# API client - the only place this file talks to the outside world
# --------------------------------------------------------------------------

def api_get(path: str, **params):
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_providers() -> dict:
    return api_get("/providers")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_database_info() -> dict:
    return api_get("/database")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("Text-to-SQL Query App")
st.markdown(
    "Ask a question in plain English. The API turns it into SQL, runs it "
    "against the bundled Chinook database, and explains the result."
)

try:
    provider_data = fetch_providers()
except requests.RequestException:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}.\n\n"
        "Start both services with `python run.py` from the project root, "
        "or start the API on its own with "
        "`uvicorn api.main:app --reload --port 8000`."
    )
    st.stop()

providers = provider_data["providers"]
if not providers:
    st.error(
        "No API keys found. Copy `.env.example` to `.env` and set "
        "`GOOGLE_API_KEY` and/or `GROQ_API_KEY`, then restart the API."
    )
    st.stop()

with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("API provider", providers, index=0)
    st.success(f"{provider} key loaded")
    model = st.selectbox("Model", provider_data["models"][provider], index=0)
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.0, 0.1,
        help="Lower is more deterministic. 0.0 is a good default for SQL.",
    )

with st.expander("Database information"):
    try:
        info = fetch_database_info()
        st.write(f"**Database type:** {info['dialect']}")
        st.write(f"**Available tables:** {', '.join(info['tables'])}")
        table = st.selectbox("Preview table data:", info["tables"])
        if table:
            preview = api_get(f"/database/tables/{table}", limit=5)
            st.dataframe(
                pd.DataFrame(preview["rows"], columns=preview["columns"]),
                width="stretch",
            )
    except requests.RequestException as exc:
        st.error(f"Could not load database information: {exc}")

question = st.text_area(
    "Enter your question about the database:",
    placeholder="e.g., How many employees are there? or Which country's customers spent the most?",
    height=100,
)

col_run, col_clear = st.columns([1, 4])
run = col_run.button("Process query", type="primary")
if col_clear.button("Clear"):
    st.rerun()

if run:
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Generating SQL and running it..."):
        try:
            result = api_post("/query", {
                "question": question,
                "provider": provider,
                "model": model,
                "temperature": temperature,
            })
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = exc.response.text[:300]
            st.error(f"API error ({exc.response.status_code}): {detail}")
            st.stop()
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            st.stop()

    st.subheader("Results")
    tab_answer, tab_sql, tab_raw, tab_table = st.tabs(
        ["Answer", "Generated SQL", "Raw results", "Formatted results"]
    )

    with tab_answer:
        if result["success"]:
            st.info(result["answer"])
        else:
            st.error(result["error"])

    with tab_sql:
        if result.get("sql_query"):
            st.code(result["sql_query"], language="sql")
        else:
            st.warning("No SQL query was generated.")

    with tab_raw:
        if result.get("raw_result"):
            st.code(result["raw_result"])
        else:
            st.warning("No raw results to display.")

    with tab_table:
        if result.get("rows"):
            frame = pd.DataFrame(result["rows"], columns=result["columns"])
            st.dataframe(frame, width="stretch")
            st.download_button(
                "Download as CSV",
                frame.to_csv(index=False),
                file_name="query_results.csv",
                mime="text/csv",
            )
        elif result["success"]:
            st.info("The query ran successfully but returned no rows.")
        else:
            st.warning("No formatted results to display.")

st.divider()
st.caption("Example questions")
for example in [
    "How many customers are there?",
    "Which country's customers spent the most?",
    "List the top 5 best-selling tracks by number of purchases.",
    "What is the average invoice total per country?",
]:
    st.markdown(f"- {example}")
