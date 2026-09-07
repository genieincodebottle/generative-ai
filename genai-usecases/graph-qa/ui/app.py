"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 180

st.set_page_config(page_title="Graph QA Chatbot", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:400])
    except ValueError:
        return exc.response.text[:400]


st.title("Graph QA Chatbot")
st.markdown(
    "Ask the graph a question in English. The model writes Cypher, the Cypher "
    "is **checked for writes before it runs**, and the results are explained."
)

try:
    cat = api("GET", "/catalogue")
    health = api("GET", "/health")
except requests.RequestException:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}.\n\n"
        "Start both services with `python run.py` from the project root."
    )
    st.stop()

if not cat["providers"]:
    st.error("No LLM provider configured. Add a key to `.env` and restart the API.")
    for name, url in cat["key_urls"].items():
        st.markdown(f"- {name}: [get a key]({url})")
    st.stop()

if health["graph"] != "ok":
    st.error(f"Neo4j is not reachable.\n\n{health['graph']}")
    st.code(
        "docker run -d --name neo4j -p 7474:7474 -p 7687:7687 "
        "-e NEO4J_AUTH=neo4j/your-password neo4j:5-community",
        language="bash",
    )
    st.stop()

with st.sidebar:
    st.markdown("### Configuration")
    provider = st.radio("LLM provider", cat["providers"], index=0)
    model = st.selectbox("Model", cat["models"][provider])
    with st.expander("Graph schema"):
        try:
            st.code(api("GET", "/schema")["schema"], language="text")
        except requests.RequestException as exc:
            st.error(f"Could not load the schema: {exc}")

st.session_state.setdefault("messages", [])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question (e.g., Who acted in The Matrix?)"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Writing Cypher and running it..."):
        try:
            result = api("POST", "/ask", json={
                "question": question, "provider": provider, "model": model,
            })
        except requests.HTTPError as exc:
            result = {"success": False, "error": detail(exc)}
        except requests.RequestException as exc:
            result = {"success": False, "error": f"Could not reach the API: {exc}"}

        if result.get("success"):
            body = result["answer"]
            st.markdown(body)
            if result.get("cypher"):
                with st.expander("Cypher"):
                    st.code(result["cypher"], language="cypher")
            if result.get("rows"):
                with st.expander(f"Rows ({len(result['rows'])})"):
                    st.dataframe(pd.DataFrame(result["rows"]), width="stretch")
        else:
            body = result.get("error", "Something went wrong.")
            st.error(body)
            if result.get("cypher"):
                st.code(result["cypher"], language="cypher")

    st.session_state.messages.append({"role": "assistant", "content": body})
