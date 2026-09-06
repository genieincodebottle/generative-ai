"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 600

st.set_page_config(page_title="Graph RAG", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:300])
    except ValueError:
        return exc.response.text[:300]


st.title("Graph RAG")
st.markdown(
    "Vector search finds documents that **look like** your question. Graph "
    "traversal then follows the edges between documents to reach what they "
    "**connect to** - which is often where the answer actually lives."
)

try:
    cat = api("GET", "/catalogue")
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
    llm_model = st.selectbox("LLM model", cat["llm_models"])
    with st.expander("Retrieval"):
        chunk_size = st.number_input("Chunk size", 200, 8000, cat["defaults"]["chunk_size"], 100)
        chunk_overlap = st.number_input("Chunk overlap", 0, 2000, cat["defaults"]["chunk_overlap"], 50)
        k_retrieval = st.number_input("Documents retrieved (k)", 1, 50, cat["defaults"]["k_retrieval"])
        max_depth = st.number_input("Max traversal depth", 1, 5, cat["defaults"]["max_depth"])

    if st.button("Apply configuration"):
        with st.spinner("Rebuilding (this clears loaded documents)..."):
            try:
                status = api("POST", "/configure", json={
                    "llm_model": llm_model, "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap, "k_retrieval": k_retrieval,
                    "max_depth": max_depth,
                })
                st.success("Configuration applied.")
            except requests.HTTPError as exc:
                st.error(detail(exc))

    st.header("Documents")
    files = st.file_uploader(
        "Upload PDF, TXT or CSV", type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )

    def load(use_sample: bool):
        with st.spinner("Building the graph (first run downloads an embedding model)..."):
            try:
                payload = [] if use_sample else [
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in files
                ]
                result = api("POST", "/documents",
                             data={"use_sample": use_sample},
                             files=payload or None)
                st.success(f"Loaded: {result['source']}")
                st.session_state.loaded = result
            except requests.HTTPError as exc:
                st.error(detail(exc))
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

    col_a, col_b = st.columns(2)
    if col_a.button("Load", type="primary", disabled=not files):
        load(False)
    if col_b.button("Use sample"):
        load(True)

    try:
        status = api("GET", "/status")
    except requests.RequestException:
        pass

    if status.get("ready"):
        st.success(f"Ready: {status['source']}")
        if status.get("relationships"):
            with st.expander(f"Detected edges ({len(status['relationships'])})"):
                for rel in status["relationships"][:40]:
                    st.markdown(f"- {rel}")

if not status.get("ready"):
    st.info(
        "Upload documents and press **Load**, or press **Use sample** to load "
        "the bundled animals dataset."
    )
    st.stop()

labels = {r["label"]: r for r in cat["retrievers"]}
chosen = st.radio("Retriever", list(labels), horizontal=True)
retriever = labels[chosen]
st.caption(retriever["blurb"])

question = st.text_area(
    "Your question",
    placeholder="e.g., Which animals live in similar habitats?",
    height=90,
)

col_run, col_explain = st.columns([1, 1])
run = col_run.button("Ask", type="primary", disabled=not question.strip())
explain = col_explain.button(
    "Explain routing only", disabled=retriever["id"] != "hybrid" or not question.strip()
)

if explain:
    with st.spinner("Asking the router..."):
        try:
            info = api("POST", "/routing-explanation", json={
                "question": question, "retriever": retriever["id"]})
            st.json(info)
        except requests.HTTPError as exc:
            st.error(detail(exc))

if run:
    with st.spinner(f"Retrieving with {chosen}..."):
        try:
            result = api("POST", "/query", json={
                "question": question, "retriever": retriever["id"]})
        except requests.HTTPError as exc:
            st.error(detail(exc))
            result = None
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            result = None

    if result:
        st.subheader("Answer")
        st.markdown(result.get("answer", ""))

        routing = result.get("routing_info")
        if routing:
            with st.expander("Why the router chose this", expanded=True):
                st.write(f"**Strategy:** {routing.get('strategy')}")
                st.write(f"**Confidence:** {routing.get('confidence')}")
                st.write(routing.get("reasoning", ""))

        docs = result.get("documents") or result.get("retrieved_documents") or []
        if docs:
            with st.expander(f"Retrieved documents ({len(docs)})"):
                for i, doc in enumerate(docs, 1):
                    content = doc if isinstance(doc, str) else doc.get("content", str(doc))
                    st.markdown(f"**Document {i}**")
                    st.text(content[:800])
