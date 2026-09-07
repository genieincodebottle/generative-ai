"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 900  # describing every image at index time is slow

st.set_page_config(page_title="Multimodal RAG", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:300])
    except ValueError:
        return exc.response.text[:300]


st.title("Multimodal RAG")
st.markdown(
    "Ask questions across text, tables **and images** in the same documents. "
    "Images are described by a vision model when they are indexed, which is "
    "what makes them searchable alongside the text."
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
    st.header("Models")
    main_model = st.selectbox("Answering model", cat["main_models"])
    vision_model = st.selectbox("Vision model (describes images)", cat["vision_models"])
    temperature = st.slider("Temperature", 0.0, 1.0, cat["defaults"]["temperature"], 0.05)

    if st.button("Apply configuration"):
        with st.spinner("Rebuilding (this clears the index)..."):
            try:
                status = api("POST", "/configure", json={
                    "main_model": main_model, "vision_model": vision_model,
                    "temperature": temperature,
                })
                st.success("Configuration applied.")
            except requests.HTTPError as exc:
                st.error(detail(exc))

    st.header("Documents and images")
    uploads = st.file_uploader(
        "Upload PDFs or images",
        type=["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp"],
        accept_multiple_files=True,
    )
    if st.button("Index", type="primary", disabled=not uploads):
        with st.spinner("Indexing. Each image is described by the vision model, so this takes a while..."):
            try:
                result = api("POST", "/documents", files=[
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in uploads
                ])
                st.success(
                    f"Indexed {len(result['documents'])} document(s) and "
                    f"{len(result['images'])} image(s)."
                )
                status = api("GET", "/status")
            except requests.HTTPError as exc:
                st.error(detail(exc))
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

    if status.get("ready"):
        indexed = status["documents"] + status["images"]
        st.success(f"Indexed: {', '.join(indexed) if indexed else 'ready'}")

if not status.get("ready"):
    st.info("Upload at least one PDF or image and press **Index** to begin.")
    st.stop()

question = st.text_area(
    "Your question",
    placeholder="e.g., What does the chart on page 3 show?",
    height=90,
)
k = st.slider("Sources retrieved (k)", 1, 30, cat["defaults"]["k"])

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving across text, tables and images..."):
        try:
            result = api("POST", "/query", json={"question": question, "k": k})
        except requests.HTTPError as exc:
            st.error(detail(exc))
            result = None
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            result = None

    if result:
        st.subheader("Answer")
        st.markdown(result.get("response", ""))

        summary = result.get("multimodal_summary") or {}
        if summary:
            col1, col2, col3 = st.columns(3)
            col1.metric("Text sources", summary.get("text_sources", 0))
            col2.metric("Table sources", summary.get("table_sources", 0))
            col3.metric("Image sources", summary.get("image_sources", 0))

        sources = result.get("sources") or []
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}**")
                    st.text(str(source)[:800])
