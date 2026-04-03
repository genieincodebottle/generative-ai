import streamlit as st
from utils import get_available_providers, get_models, run_query

# ── Page config ────────────────────────────────────────────
st.set_page_config(page_title="Graph QA Chatbot", page_icon="🌐", layout="wide")
st.markdown(
    "<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────
def render_sidebar():
    """Draw sidebar widgets and return (provider, model_name)."""
    st.sidebar.image("img/globe.png")
    st.sidebar.markdown("### Configuration")

    providers = get_available_providers()
    if not providers:
        st.error(
            "No LLM provider configured. "
            "Add **GROQ_API_KEY** or **GOOGLE_API_KEY** to your `.env` file and restart."
        )
        st.stop()

    provider = st.sidebar.radio("LLM Provider", providers, index=0)
    model_name = st.sidebar.selectbox("Model", get_models(provider))

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Uses **Neo4j Movie Database** — ask about actors, movies, "
        "directors, and their relationships."
    )
    return provider, model_name


provider, model_name = render_sidebar()

# ── Main area ──────────────────────────────────────────────
st.header("Graph QA Chatbot")
st.info(
    "Ask natural-language questions about the Neo4j Movie Database. "
    "The app converts your question into a Cypher query, runs it, and returns an answer."
)

# ── Chat history ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle user input ─────────────────────────────────────
user_input = st.chat_input("Ask a question (e.g., Who acted in The Matrix?)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Querying graph database..."):
            try:
                result = run_query(user_input, provider, model_name)
            except Exception as e:
                result = f"Error: {e}"
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})
