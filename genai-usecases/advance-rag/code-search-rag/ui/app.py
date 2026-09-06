"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 600

st.set_page_config(page_title="Code Search RAG", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:300])
    except ValueError:
        return exc.response.text[:300]


st.title("Code Search RAG")
st.markdown(
    "Search a codebase in plain English. Code is chunked by **function and "
    "class**, not by character count, so a retrieved chunk is a thing that "
    "compiles rather than an arbitrary window."
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
        top_k_initial = st.number_input("Initial candidates", 1, 500, cat["defaults"]["top_k_initial"])
        top_k_rerank = st.number_input("After filtering", 1, 50, cat["defaults"]["top_k_rerank"])
        top_k_final = st.number_input("Final results", 1, 20, cat["defaults"]["top_k_final"])

    if st.button("Apply configuration"):
        with st.spinner("Rebuilding (this clears the index)..."):
            try:
                status = api("POST", "/configure", json={
                    "llm_model": llm_model, "top_k_initial": top_k_initial,
                    "top_k_rerank": top_k_rerank, "top_k_final": top_k_final,
                })
                st.success("Configuration applied.")
            except requests.HTTPError as exc:
                st.error(detail(exc))

    st.header("Code")
    st.caption("Supported: " + ", ".join(cat["languages"]))
    files = st.file_uploader(
        "Upload source files", accept_multiple_files=True,
        type=["py", "js", "jsx", "ts", "tsx", "java", "go", "rs", "cpp", "cc", "c", "h", "hpp"],
    )

    def index(use_samples: bool):
        with st.spinner("Parsing and embedding..."):
            try:
                payload = [] if use_samples else [
                    ("files", (f.name, f.getvalue(), "text/plain")) for f in files
                ]
                result = api("POST", "/index",
                             data={"use_samples": use_samples},
                             files=payload or None)
                st.success(f"Indexed {result['chunks']} chunks from {len(result['files'])} file(s).")
                if result.get("skipped"):
                    st.warning("Skipped: " + ", ".join(result["skipped"]))
            except requests.HTTPError as exc:
                st.error(detail(exc))
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

    col_a, col_b = st.columns(2)
    if col_a.button("Index", type="primary", disabled=not files):
        index(False)
    if col_b.button("Use samples"):
        index(True)

    try:
        status = api("GET", "/status")
    except requests.RequestException:
        pass
    if status.get("ready"):
        st.success(f"{status['chunks']} chunks indexed")

if not status.get("ready"):
    st.info(
        "Upload source files and press **Index**, or press **Use samples** to "
        "index the OAuth2 sample code bundled with this project."
    )
    st.stop()

query = st.text_area(
    "What are you looking for?",
    placeholder="e.g., How is the OAuth2 token refreshed when it expires?",
    height=90,
)
language = st.selectbox("Filter by language", ["(any)"] + cat["languages"])

if st.button("Search", type="primary", disabled=not query.strip()):
    with st.spinner("Searching..."):
        try:
            result = api("POST", "/search", json={
                "query": query,
                "language": None if language == "(any)" else language,
            })
            st.subheader("Answer")
            st.markdown(result["answer"])
        except requests.HTTPError as exc:
            st.error(detail(exc))
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
