"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 600  # the agent pipeline runs several LLM steps per query

st.set_page_config(page_title="Agentic RAG", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:300])
    except ValueError:
        return exc.response.text[:300]


st.title("Agentic RAG")
st.markdown(
    "A multi-agent pipeline: **plan** the query, **retrieve** from your "
    "documents, **research** the web when needed, **synthesise**, then "
    "**validate** the answer."
)

try:
    meta = api("GET", "/models")
    status = api("GET", "/status")
except requests.RequestException:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}.\n\n"
        "Start both services with `python run.py` from the project root."
    )
    st.stop()

if not status.get("google_key"):
    st.error(
        "GOOGLE_API_KEY is not set. Copy `.env.example` to `.env`, add your "
        "key, then restart the API."
    )
    st.stop()

with st.sidebar:
    st.header("Configuration")

    llm_model = st.selectbox("LLM model", meta["llm_models"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    if meta["web_search_available"]:
        enable_web_search = st.checkbox("Enable web search (Tavily)", value=True)
    else:
        enable_web_search = False
        st.info(
            "Web search is off: no `TAVILY_API_KEY` found. The system answers "
            "from your documents only. Add a key to `.env` to enable it."
        )

    with st.expander("Advanced"):
        chunk_size = st.number_input("Chunk size", 200, 8000, 1000, 100)
        chunk_overlap = st.number_input("Chunk overlap", 0, 2000, 200, 50)
        k_retrieval = st.number_input("Chunks retrieved (k)", 1, 50, 8)
        max_iterations = st.number_input("Max agent iterations", 1, 50, 10)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05)
        max_web_results = st.number_input("Max web results", 1, 20, 5)

    if st.button("Apply configuration", type="primary"):
        with st.spinner("Rebuilding the system..."):
            try:
                status = api("POST", "/configure", json={
                    "llm_model": llm_model, "temperature": temperature,
                    "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                    "k_retrieval": k_retrieval, "max_iterations": max_iterations,
                    "confidence_threshold": confidence_threshold,
                    "enable_web_search": enable_web_search,
                    "max_web_results": max_web_results,
                })
                st.success("Configuration applied.")
            except requests.HTTPError as exc:
                st.error(detail(exc))

    st.header("Documents")
    pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index documents", disabled=not pdf_files):
        with st.spinner("Indexing..."):
            try:
                result = api("POST", "/documents", files=[
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in pdf_files
                ])
                st.success(f"Indexed {result['loaded']} document(s).")
                status = api("GET", "/status")
            except requests.HTTPError as exc:
                st.error(detail(exc))

    if status.get("documents"):
        st.caption("Indexed: " + ", ".join(status["documents"]))

tab_ask, tab_status = st.tabs(["Ask", "System status"])

with tab_ask:
    if not status.get("retriever_available"):
        st.info("Upload and index at least one PDF to begin.")
    question = st.text_area(
        "Your question",
        placeholder="e.g., What are the main risks described in these documents?",
        height=100,
    )
    if st.button("Run the pipeline", type="primary", disabled=not question.strip()):
        with st.spinner("Planning, retrieving, researching, synthesising..."):
            try:
                result = api("POST", "/query", json={"question": question})
            except requests.HTTPError as exc:
                st.error(detail(exc))
                result = None
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                result = None

        if result:
            st.subheader("Answer")
            st.markdown(result["answer"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{result.get('confidence', 0):.2f}")
            col2.metric("Documents used", result.get("retrieved_documents", 0))
            col3.metric("Web results", result.get("web_results", 0))

            plan = result.get("query_plan")
            if plan:
                with st.expander("Query plan"):
                    st.write(f"**Complexity:** {plan['complexity']}")
                    st.write(f"**Estimated steps:** {plan['estimated_steps']}")
                    for sub in plan.get("sub_queries", []):
                        st.write(f"- {sub}")

            if result.get("sources"):
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.write(f"- {source}")

            if result.get("execution_log"):
                with st.expander("Execution log"):
                    for line in result["execution_log"]:
                        st.text(line)

with tab_status:
    st.json(status)
    config_block = status.get("configuration")
    if config_block:
        st.dataframe(
            pd.DataFrame(
                sorted(config_block.items()), columns=["Setting", "Value"]
            ).astype(str),
            width="stretch",
        )
