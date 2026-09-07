"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 300  # tagging every call is a batch of LLM round trips

st.set_page_config(page_title="Customer Call Sentiment", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def show_api_error(exc: Exception) -> None:
    if isinstance(exc, requests.HTTPError):
        try:
            detail = exc.response.json().get("detail", exc.response.text[:300])
        except ValueError:
            detail = exc.response.text[:300]
        st.error(f"API error ({exc.response.status_code}): {detail}")
    else:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")


st.title("Customer call sentiment and aggressiveness tagging")
st.markdown(
    "Classify customer call transcripts for sentiment and an aggressiveness "
    "score from 1 to 10."
)

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

with st.sidebar:
    st.header("Configuration")
    labels = {p["label"]: p for p in providers}
    default_index = next(
        (i for i, p in enumerate(providers) if p["id"] == provider_data["default_provider"]), 0
    )
    chosen_label = st.selectbox("Provider", list(labels), index=default_index)
    provider = labels[chosen_label]
    model = st.selectbox(
        "Model", provider["models"],
        index=(provider["models"].index(provider["default_model"])
               if provider["default_model"] in provider["models"] else 0),
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    st.caption(f"[Get a key]({provider['key_url']})")

tab_setup, tab_process, tab_results, tab_try = st.tabs(
    ["Database", "Process calls", "Results", "Try one"]
)

with tab_setup:
    st.header("Database setup")
    try:
        status = api("GET", "/database")
    except requests.RequestException as exc:
        show_api_error(exc)
        status = {"ready": False}

    if not status["ready"]:
        st.info("No database yet. Create it with sample calls below.")
        if st.button("Initialize database", type="primary"):
            try:
                result = api("POST", "/database/init")
                st.success(f"Database created. Seeded {result['seeded']} sample calls.")
                st.rerun()
            except requests.RequestException as exc:
                show_api_error(exc)
    else:
        st.success(
            f"Database ready: {status['calls']} calls, {status['tagged']} analyzed."
        )
        if st.button("Reset database (clear all data)"):
            try:
                api("DELETE", "/database")
                st.rerun()
            except requests.RequestException as exc:
                show_api_error(exc)

with tab_process:
    st.header("Process customer calls")
    if st.button("Analyze all calls", type="primary"):
        with st.spinner("Classifying every call..."):
            try:
                result = api("POST", "/taggings/run", json={
                    "provider": provider["id"], "model": model,
                    "temperature": temperature,
                })
            except requests.RequestException as exc:
                show_api_error(exc)
                result = None

        if result:
            st.success(f"{result['succeeded']} of {result['total']} calls processed.")
            if result["failed"]:
                st.warning(f"{result['failed']} failed.")
                with st.expander(f"Failed calls ({result['failed']})"):
                    for outcome in result["outcomes"]:
                        if not outcome["success"]:
                            st.error(f"Call {outcome['call_id']}: {outcome['error']}")

with tab_results:
    st.header("Tagging results")
    if st.button("Refresh"):
        st.rerun()
    try:
        results = api("GET", "/taggings")
        stats = api("GET", "/stats")
    except requests.RequestException as exc:
        show_api_error(exc)
        results, stats = [], None

    if not results:
        st.info("No results yet. Run the analysis in the 'Process calls' tab.")
    else:
        st.dataframe(pd.DataFrame(results), width="stretch")
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total analyzed", stats["total"])
        for sentiment, count in stats["sentiment_counts"].items():
            col1.metric(f"{sentiment} calls", count)
        col2.metric("Average aggressiveness", f"{stats['average_aggressiveness']:.1f}")
        col2.metric("Max aggressiveness", stats["max_aggressiveness"])
        col3.metric("High aggression (>=7)", stats["high_aggression_calls"])

with tab_try:
    st.header("Try a single transcript")
    text = st.text_area(
        "Call transcript",
        placeholder="Customer was angry about receiving the wrong order and demanded a refund.",
        height=120,
    )
    if st.button("Analyze", type="primary"):
        try:
            result = api("POST", "/analyze", json={
                "text": text, "provider": provider["id"],
                "model": model, "temperature": temperature,
            })
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", result["sentiment"])
            col2.metric("Aggressiveness", f"{result['aggressiveness']} / 10")
        except requests.RequestException as exc:
            show_api_error(exc)
