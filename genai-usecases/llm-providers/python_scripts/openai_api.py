import streamlit as st
from langchain_openai import ChatOpenAI
import os
# Environment Variables
from dotenv import load_dotenv
from llm_text import message_text

load_dotenv()

# Set page config
st.set_page_config(
    page_title="OpenAI Assistant",
    layout="wide"
)

# OpenAI is paid. Groq exposes an OpenAI-COMPATIBLE endpoint, so the exact
# same client code runs against a free key by pointing base_url at it - which
# means you can exercise this app without an OpenAI balance.
#
# Set OPENAI_BASE_URL=https://api.groq.com/openai/v1 and put your Groq key in
# OPENAI_API_KEY, or just set GROQ_API_KEY and this picks it up.
BASE_URL = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key and os.getenv("GROQ_API_KEY"):
    api_key = os.getenv("GROQ_API_KEY")
    BASE_URL = BASE_URL or "https://api.groq.com/openai/v1"

if not api_key:
    st.error(
        "No key found. Copy .env.example to .env and set OPENAI_API_KEY "
        "(paid), or set GROQ_API_KEY to use Groq's free OpenAI-compatible "
        "endpoint with this same app.\n\n"
        "OpenAI: https://platform.openai.com/api-keys\n"
        "Groq (free): https://console.groq.com/keys"
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
USING_GROQ = bool(BASE_URL and "groq.com" in BASE_URL)

# Title and description
st.title("OpenAI Assistant")
st.markdown("A helpful assistant powered by OpenAI's GPT models that explains problems step-by-step.")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    # Model selection
    # gpt-4.5 was shut down on 2025-07-14 and gpt-5-2025-08-07 is a dated
    # snapshot with a retirement date; neither belongs in a default list.
    if USING_GROQ:
        st.caption("Using Groq's OpenAI-compatible endpoint (free tier)")
        model_options = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    else:
        model_options = [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-4.1"
        ]
    selected_model = st.selectbox("Select Model:", model_options)

    # Temperature setting
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Main interface
if api_key:

    # Initialize the LLM
    try:
        llm = ChatOpenAI(
            model=selected_model,
            temperature=temperature,
            # None means "use OpenAI". Anything else is an OpenAI-compatible
            # endpoint, and the rest of this file does not change.
            base_url=BASE_URL,
            # Reasoning tokens come out of this same budget. Leave it low and
            # the model can spend the whole allowance thinking, returning a
            # 200 with EMPTY content - which looks like success.
            max_tokens=2048,
        )

        # Problem input
        st.header("Ask GPT")
        problem = st.text_area(
            "Enter your problem or question:",
            placeholder="e.g., Provide me python code for sudoku, Explain quantum physics, Solve 2x + 5 = 15",
            height=100
        )

        # Submit button
        if st.button("Get Solution", type="primary"):
            if problem.strip():
                with st.spinner("GPT is thinking..."):
                    try:
                        # Create messages
                        messages = [
                            (
                                "system",
                                "You are a helpful assistant that explains problems step-by-step."
                            ),
                            ("human", f"Solve this problem step by step: {problem}")
                        ]

                        # Get response from OpenAI
                        ai_msg = llm.invoke(messages)

                        # Display response
                        st.header("GPT's Response:")
                        st.markdown(message_text(ai_msg))

                    except Exception as e:
                        st.error(f"Error getting response from OpenAI: {str(e)}")
            else:
                st.warning("Please enter a problem or question.")

    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {str(e)}")
        st.info("Please check your API key and try again.")

else:
    st.warning("Please enter your OpenAI API key in the .env file to get started.")
    st.info("""
    To use this app:
    1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
    2. Enter it in the .env file as OPENAI_API_KEY
    3. Ask GPT any question or problem
    4. Get step-by-step explanations!
    """)