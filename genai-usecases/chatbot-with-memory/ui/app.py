"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 300  # indexing a large PDF is slow the first time

st.set_page_config(page_title="PDF Chat Bot", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def error_detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:300])
    except ValueError:
        return exc.response.text[:300]


st.title("PDF chat bot with memory")
st.markdown("Upload PDFs and ask questions. The bot remembers your conversation.")

try:
    provider_data = api("GET", "/providers")
except requests.RequestException:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}.\n\n"
        "Start both services with `python run.py` from the project root."
    )
    st.stop()

providers = provider_data["providers"]
if not providers:
    st.error(
        "No API keys found. Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` "
        "or `GROQ_API_KEY`, then restart the API."
    )
    st.stop()

defaults = provider_data["defaults"]
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("history", [])

with st.sidebar:
    st.title("Settings")

    by_id = {p["id"]: p for p in providers}
    provider_id = st.radio("LLM provider", list(by_id), horizontal=True)
    provider = by_id[provider_id]
    st.success(f"{provider_id} key loaded")

    model = st.selectbox("Model", provider["models"], help=provider["model_help"])
    temperature = st.slider("Temperature", 0.0, 1.0, defaults["temperature"], 0.1)

    if provider["embeddings"] == "huggingface":
        st.caption(
            "Groq serves no embedding model, so retrieval runs locally on CPU. "
            "The first upload downloads about 90 MB of weights."
        )

    with st.expander("Advanced"):
        chunk_size = st.number_input("Chunk size", 500, 8000, defaults["chunk_size"], 100)
        chunk_overlap = st.number_input("Chunk overlap", 0, 2000, defaults["chunk_overlap"], 50)
        retriever_k = st.number_input("Chunks retrieved (k)", 1, 20, defaults["retriever_k"])

    st.header("Upload PDFs")
    pdf_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Index documents", type="primary", disabled=not pdf_files):
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            with st.spinner("Reading and indexing..."):
                try:
                    result = api(
                        "POST", "/sessions",
                        files=[("files", (f.name, f.getvalue(), "application/pdf"))
                               for f in pdf_files],
                        data={
                            "provider": provider_id, "model": model,
                            "temperature": temperature, "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap, "retriever_k": retriever_k,
                        },
                    )
                    st.session_state.session_id = result["session_id"]
                    st.session_state.history = []
                    st.success(
                        f"Indexed {result['pages']} pages into "
                        f"{result['chunks']} chunks."
                    )
                except requests.HTTPError as exc:
                    st.error(error_detail(exc))
                except requests.RequestException as exc:
                    st.error(f"Could not reach the API: {exc}")

    if st.session_state.session_id and st.button("Clear conversation"):
        try:
            api("DELETE", f"/sessions/{st.session_state.session_id}/history")
            st.session_state.history = []
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not clear the conversation: {exc}")

if not st.session_state.session_id:
    st.info("Upload one or more PDFs and click **Index documents** to begin.")
    st.stop()

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            result = api(
                "POST", f"/sessions/{st.session_state.session_id}/chat",
                json={"question": question, "temperature": temperature,
                      "retriever_k": retriever_k},
            )
            st.markdown(result["answer"])
            st.session_state.history = result["history"]
        except requests.HTTPError as exc:
            if exc.response.status_code == 404:
                st.session_state.session_id = None
                st.error("That session expired. Please index your PDFs again.")
            else:
                st.error(error_detail(exc))
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
