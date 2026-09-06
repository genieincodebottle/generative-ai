"""Streamlit UI: widgets and HTTP calls only."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 300

st.set_page_config(page_title="Llama 4 Multi-Function App", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", exc.response.text[:400])
    except ValueError:
        return exc.response.text[:400]


st.title("Llama 4 Multi-Function App")
st.markdown(
    "One model, four jobs: chat, reading images, retrieval over your own "
    "documents, and agent tasks. Llama 4 Scout on Groq, with a Gemini "
    "fallback that is always reported rather than hidden."
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

if not health["keys"].get("groq"):
    st.error("GROQ_API_KEY is not set. This app needs it for every feature.")
    st.markdown(f"Get one at [{cat['key_urls']['groq']}]({cat['key_urls']['groq']})")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("Model", cat["models"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    if health["keys"].get("google"):
        st.success("Gemini fallback available")
        allow_fallback = st.checkbox("Use fallback on failure", value=True)
    else:
        allow_fallback = False
        st.info(
            "No GOOGLE_API_KEY, so there is no fallback. A failed Groq call "
            "will be reported as an error rather than silently answered."
        )

tab_chat, tab_vision, tab_rag = st.tabs(["Chat", "OCR / vision", "Documents"])

with tab_chat:
    st.session_state.setdefault("messages", [])
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                reply = api("POST", "/chat", json={
                    "messages": st.session_state.messages,
                    "model": model, "temperature": temperature,
                    "allow_fallback": allow_fallback,
                })
                st.markdown(reply["text"])
                if reply.get("fallback_used"):
                    st.warning(
                        f"Answered by {reply['provider']} ({reply['model']}) "
                        f"because the primary model failed: "
                        f"{reply.get('fallback_reason')}"
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply["text"]}
                )
            except requests.HTTPError as exc:
                st.error(detail(exc))
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

with tab_vision:
    st.markdown("Upload an image and ask about it. Llama 4 Scout reads text and describes scenes.")
    image = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    vision_model = st.selectbox("Vision model", cat["vision_models"])
    vision_prompt = st.text_area(
        "Prompt", value="Read all the text in this image.", height=80
    )
    if image:
        st.image(image, width=420)
    if st.button("Analyse image", type="primary", disabled=not image):
        with st.spinner("Reading the image..."):
            try:
                reply = api("POST", "/vision",
                            files=[("file", (image.name, image.getvalue(),
                                             image.type))],
                            data={"prompt": vision_prompt, "model": vision_model})
                st.markdown(reply["text"])
            except requests.HTTPError as exc:
                st.error(detail(exc))
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

with tab_rag:
    status = api("GET", "/documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or CSV", type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )
    col_add, col_clear = st.columns(2)
    if col_add.button("Index", type="primary", disabled=not uploads):
        with st.spinner("Indexing (the first run downloads an embedding model)..."):
            try:
                result = api("POST", "/documents", files=[
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in uploads
                ])
                st.success(f"Indexed {len(result['added'])} document(s).")
                for failure in result.get("failed", []):
                    st.warning(f"{failure['name']}: {failure['error']}")
                status = result
            except requests.HTTPError as exc:
                st.error(detail(exc))
    if col_clear.button("Clear index"):
        api("DELETE", "/documents")
        st.rerun()

    if status.get("indexed"):
        st.caption(
            f"{len(status['documents'])} document(s), "
            f"{status['total_chunks']} chunks indexed"
        )
        query = st.text_input("Search your documents")
        top_k = st.slider("Results", 1, 20, 3)
        if st.button("Search", disabled=not query.strip()):
            try:
                result = api("POST", "/search",
                             json={"query": query, "top_k": top_k})
                for i, hit in enumerate(result["results"], 1):
                    with st.expander(f"{i}. {hit['metadata'].get('source', '')}"):
                        st.text(hit["content"][:1200])
            except requests.HTTPError as exc:
                st.error(detail(exc))
    else:
        st.info("Upload a document and press Index to enable search.")
