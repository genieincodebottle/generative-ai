"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 1800  # loading a model and answering a whole dataset is slow

st.set_page_config(page_title="Cache-Augmented Generation", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:500])
    except ValueError:
        return exc.response.text[:500]


st.title("Cache-Augmented Generation")
st.markdown(
    "Put the **whole document** into the model's context once, keep the "
    "resulting KV cache, and answer every question by reusing it. No "
    "retrieval, no chunking, no vector store - and no retrieval mistakes "
    "either, because there is no retrieval step to get wrong."
)

try:
    cat = api("GET", "/catalogue")
    status = api("GET", "/health")
except requests.RequestException:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}.\n\n"
        "Start both services with `python run.py` from the project root."
    )
    st.stop()

with st.sidebar:
    st.header("Model")
    labels = {m["label"]: m for m in cat["models"]}
    label = st.selectbox("Model", list(labels))
    model = labels[label]
    st.caption(model["note"])

    if model["gated"] and not status["hf_token"]:
        st.warning(
            "This model is gated and no HF_TOKEN is set. The ungated models "
            "need no token at all."
        )

    quantized = st.checkbox(
        "Quantized (needs a GPU)", value=False,
        help="4-bit quantization through bitsandbytes. CPU-only machines "
             "should leave this off.",
    )

    if st.button("Load model", type="primary"):
        with st.spinner("Downloading and loading (first time is slow)..."):
            try:
                status = api("POST", "/model", json={
                    "model_id": model["id"], "quantized": quantized})
                st.success(f"Loaded {status['model']}")
            except requests.HTTPError as exc:
                st.error(detail(exc))
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

    if status.get("loaded"):
        st.success(f"Loaded: {status['model']}")
        if st.button("Unload"):
            api("DELETE", "/model")
            st.rerun()

if not status.get("loaded"):
    st.info("Choose a model in the sidebar and press **Load model** to begin.")
    st.stop()

st.subheader("Document")
st.caption(
    f"Everything here goes into the context in one go. "
    f"Maximum {cat['max_document_chars']:,} characters."
)

st.session_state.setdefault("document", "")
try:
    bundled = api("GET", "/dataset")
except requests.RequestException:
    bundled = None

if bundled and st.button(
    f"Use the bundled corpus ({bundled['corpus_chars']:,} characters, "
    f"{len(bundled['rows'])} documents)"
):
    st.session_state.document = bundled["corpus"]

document = st.text_area("Document text", height=220,
                        max_chars=cat["max_document_chars"],
                        key="document")

st.subheader("Questions")
uploaded = st.file_uploader("Question/answer CSV (optional)", type=["csv"])
if not uploaded and bundled:
    st.caption(f"Using the bundled dataset ({len(bundled['rows'])} questions).")
    with st.expander("Preview"):
        st.dataframe(pd.DataFrame(bundled["rows"]), width="stretch")
elif not uploaded:
    st.warning("No bundled dataset found. Upload a CSV instead.")

use_cache = st.checkbox(
    "Use the KV cache", value=True,
    help="Off runs the same questions without the cache, so you can compare.",
)

if st.button("Run", type="primary", disabled=not document.strip()):
    with st.spinner("Answering every question..."):
        try:
            files = ([("dataset", (uploaded.name, uploaded.getvalue(), "text/csv"))]
                     if uploaded else None)
            result = api("POST", "/run",
                         params={"document": document, "use_cache": use_cache},
                         files=files)
        except requests.HTTPError as exc:
            st.error(detail(exc))
            result = None
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            result = None

    if result:
        col1, col2, col3 = st.columns(3)
        col1.metric("Average similarity", f"{result['avg_similarity']:.3f}")
        col2.metric("Cache build", f"{result['avg_cache_time']:.2f} s")
        col3.metric("Average generate", f"{result['avg_generate_time']:.2f} s")
        st.caption(
            f"{result['questions']} questions over "
            f"{result['document_chars']:,} characters, "
            f"cache {'on' if result['use_cache'] else 'off'}"
        )
        if result.get("details"):
            with st.expander("Per-question results"):
                st.dataframe(pd.DataFrame(result["details"]), width="stretch")
