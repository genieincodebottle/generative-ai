"""Streamlit UI: widgets and HTTP calls only.

The form for each app is built from the catalogue the API returns, so this
file knows nothing about any individual workflow.
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 900  # agent workflows make many model calls

st.set_page_config(page_title="Agentic AI Platform", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:400])
    except ValueError:
        return exc.response.text[:400]


st.title("Agentic AI Platform")
st.markdown(
    "Agentic workflow patterns, LangGraph pipelines and multi-agent "
    "orchestration - all behind one API, all sharing one provider layer."
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
    st.error(
        "No providers are available. Copy `.env.example` to `.env` and set at "
        "least one key, or install Ollama to run locally with no key."
    )
    st.stop()

by_id = {p["id"]: p for p in cat["providers"]}

with st.sidebar:
    st.header("Provider")
    provider_id = st.selectbox("Provider", list(by_id))
    provider = by_id[provider_id]
    model = st.selectbox("Model", provider["models"])
    if provider["note"]:
        st.caption(provider["note"])

    ollama_base_url = None
    if provider_id == "Ollama":
        ollama_base_url = st.text_input(
            "Ollama base URL", value="http://localhost:11434"
        )
        if health["ollama_reachable"]:
            st.success("Ollama server is reachable")
        else:
            st.warning(
                "No Ollama server answered. Start it with `ollama serve`, "
                "then pull a model with `ollama pull llama3.2:3b`."
            )
    elif provider["needs_key"]:
        st.success(f"{provider_id} key loaded")

families: dict[str, list[dict]] = {}
for app_spec in cat["apps"]:
    families.setdefault(app_spec["family"], []).append(app_spec)

family = st.radio("Family", list(families), horizontal=True)
labels = {a["label"]: a for a in families[family]}
chosen = st.selectbox("App", list(labels))
app_spec = labels[chosen]
st.caption(app_spec["blurb"])

inputs: dict = {}
for field in app_spec["fields"]:
    key = f"{app_spec['id']}:{field['name']}"
    if field["kind"] == "textarea":
        inputs[field["name"]] = st.text_area(
            field["label"], value=str(field["default"]), height=110,
            help=field["help"] or None, key=key,
        )
    elif field["kind"] == "select":
        options = field["options"]
        index = options.index(field["default"]) if field["default"] in options else 0
        inputs[field["name"]] = st.selectbox(
            field["label"], options, index=index, key=key
        )
    elif field["kind"] == "bool":
        inputs[field["name"]] = st.checkbox(
            field["label"], value=bool(field["default"]), key=key
        )
    elif field["kind"] == "number":
        inputs[field["name"]] = st.number_input(
            field["label"], value=float(field["default"] or 0), key=key
        )
    else:
        inputs[field["name"]] = st.text_input(
            field["label"], value=str(field["default"]), key=key
        )

if st.button("Run", type="primary"):
    with st.spinner(f"Running {chosen}..."):
        try:
            result = api("POST", f"/apps/{app_spec['id']}/run", json={
                "provider": provider_id, "model": model,
                "ollama_base_url": ollama_base_url, "inputs": inputs,
            })
        except requests.HTTPError as exc:
            st.error(detail(exc))
            result = None
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            result = None

    if result:
        payload = result.get("result")

        # Most workflows return a dict with an answer-shaped field somewhere.
        # Show that first, then the whole structure for the curious.
        if isinstance(payload, dict):
            for key in ("final_output", "final_answer", "answer", "response",
                        "synthesized_result", "output"):
                if payload.get(key):
                    st.subheader("Result")
                    st.markdown(str(payload[key]))
                    break
            st.subheader("Full response")
            st.json(payload)
        else:
            st.subheader("Result")
            st.write(payload)

        if result.get("stats"):
            with st.expander("Workflow stats"):
                st.json(result["stats"])

        st.download_button(
            "Download result as JSON",
            json.dumps(result, indent=2, default=str),
            file_name=f"{app_spec['id']}_result.json",
            mime="application/json",
        )
