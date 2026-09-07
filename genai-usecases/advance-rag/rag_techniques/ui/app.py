"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 600

st.set_page_config(page_title="RAG Techniques", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:300])
    except ValueError:
        return exc.response.text[:300]


st.title("RAG techniques, compared")
st.markdown(
    "Five retrieval strategies over **one shared index**. Because they all "
    "query the same chunks, the difference you see is the technique, not the "
    "indexing."
)

try:
    cat = api("GET", "/catalogue")
except requests.RequestException:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}.\n\n"
        "Start both services with `python run.py` from the project root."
    )
    st.stop()

if not cat["providers"]:
    st.error(
        "No API keys found. Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` "
        "or `GROQ_API_KEY`, then restart the API."
    )
    for name, url in cat["key_urls"].items():
        st.markdown(f"- {name}: [get a key]({url})")
    st.stop()

defaults = cat["defaults"]
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("indexed", None)

with st.sidebar:
    st.header("Provider")
    provider = st.selectbox("Provider", cat["providers"])
    model = st.selectbox("Model", cat["models"][provider])
    temperature = st.slider("Temperature", 0.0, 1.0, defaults["temperature"], 0.05)
    st.caption(f"Embeddings: {cat['embeddings'][provider]}")

    st.header("Documents")
    files = st.file_uploader(
        "Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True
    )
    with st.expander("Chunking"):
        chunk_size = st.number_input("Chunk size", 200, 8000, defaults["chunk_size"], 100)
        chunk_overlap = st.number_input("Chunk overlap", 0, 2000, defaults["chunk_overlap"], 50)

    col_a, col_b = st.columns(2)

    def index_documents(use_samples: bool):
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
            return
        with st.spinner("Indexing..."):
            try:
                payload = {
                    "provider": provider, "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap, "use_samples": use_samples,
                }
                upload_files = [] if use_samples else [
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in files
                ]
                result = api("POST", "/sessions", data=payload, files=upload_files or None)
                st.session_state.session_id = result["session_id"]
                st.session_state.indexed = result
            except requests.HTTPError as exc:
                st.error(detail(exc))
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

    if col_a.button("Index", type="primary", disabled=not files):
        index_documents(False)
    if col_b.button("Use samples"):
        index_documents(True)

    if st.session_state.indexed:
        info = st.session_state.indexed
        st.success(
            f"{info['total_chunks']} chunks from "
            f"{len(info['filenames'])} file(s), "
            f"average {info['avg_chunk_length']} characters"
        )

if not st.session_state.session_id:
    st.info(
        "Upload a PDF or TXT and press **Index**, or press **Use samples** to "
        "index the two documents bundled with this project."
    )
    st.stop()

labels = {t["label"]: t for t in cat["techniques"]}
chosen = st.radio("Technique", list(labels), horizontal=True)
technique = labels[chosen]
st.caption(technique["blurb"])

options = {}
if "top_k" in technique["options"]:
    options["top_k"] = st.slider("Chunks retrieved (k)", 1, 20, defaults["top_k"])
if "bm25_weight" in technique["options"]:
    bm25 = st.slider("BM25 weight", 0.0, 1.0, defaults["bm25_weight"], 0.05)
    options["bm25_weight"] = bm25
    options["vector_weight"] = round(1.0 - bm25, 2)
    st.caption(f"Vector weight: {options['vector_weight']}")
if "reranker" in technique["options"]:
    options["reranker"] = st.selectbox("Re-ranker", cat["rerankers"])

query = st.text_area(
    "Your question",
    placeholder="e.g., What are the main benefits described in these documents?",
    height=90,
)

if st.button("Run", type="primary", disabled=not query.strip()):
    with st.spinner(f"Running {chosen}..."):
        try:
            result = api(
                "POST", f"/sessions/{st.session_state.session_id}/query",
                json={"query": query, "technique": technique["id"],
                      "model": model, "temperature": temperature, **options},
            )
        except requests.HTTPError as exc:
            if exc.response.status_code == 404:
                st.session_state.session_id = None
                st.error("That session expired. Index your documents again.")
            else:
                st.error(detail(exc))
            result = None
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            result = None

    if result:
        st.subheader("Answer")
        st.markdown(result["answer"])
        st.caption(result["retrieval_method"])

        with st.expander("What it did", expanded=True):
            for step in result.get("steps", []):
                st.markdown(f"- {step}")

        if result.get("critique"):
            with st.expander("Initial answer and critique"):
                st.markdown("**Initial answer**")
                st.info(result["initial_response"])
                st.markdown("**Critique**")
                st.warning(result["critique"])

        if result.get("bm25_documents"):
            left, right = st.columns(2)
            with left:
                st.markdown("**BM25 (keyword)**")
                for doc in result["bm25_documents"][:3]:
                    st.text(doc["content"][:280])
            with right:
                st.markdown("**Vector (semantic)**")
                for doc in result["vector_documents"][:3]:
                    st.text(doc["content"][:280])

        with st.expander(f"Retrieved chunks ({len(result.get('documents', []))})"):
            for i, doc in enumerate(result.get("documents", []), 1):
                st.markdown(f"**Chunk {i}** - {doc['metadata'].get('source', '')}")
                st.text(doc["content"][:800])
