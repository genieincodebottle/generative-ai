import streamlit as st
from langchain_groq import ChatGroq
import os
# Environment Variables
from dotenv import load_dotenv
from llm_text import message_text

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Groq Assistant",
    layout="wide"
)

# Ensure the GROQ_API_KEY is set in the environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error(
        "GROQ_API_KEY is not set. Copy .env.example to .env and put your "
        "key in it, then restart this app.\n\nGet a key at: https://console.groq.com/keys"
    )
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# Title and description
st.title("Groq Assistant")
st.markdown("A helpful assistant powered by Groq's fast AI inference for open-source LLMs that explains problems step-by-step.")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    # Model selection
    # Every Llama, Gemma and Qwen3-32B model this list used to offer was
    # shut down for the free and developer tiers during 2025-2026. These two
    # are Groq's own stated replacements.
    model_options = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    selected_model = st.selectbox("Select Groq Model:", model_options)

    # Temperature setting
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Main interface
if api_key:

    # Initialize the LLM
    try:
        llm = ChatGroq(
            model=selected_model,
            temperature=temperature
        )

        # Problem input
        st.header("Ask Groq")
        problem = st.text_area(
            "Enter your problem or question:",
            placeholder="e.g., Provide me python code for sudoku, Explain quantum physics, Solve 2x + 5 = 15",
            height=100
        )

        # Submit button
        if st.button("Get Solution", type="primary"):
            if problem.strip():
                with st.spinner("Groq is thinking..."):
                    try:
                        # Create messages
                        messages = [
                            (
                                "system",
                                "You are a helpful assistant that explains problems step-by-step."
                            ),
                            ("human", f"Solve this problem step by step: {problem}")
                        ]

                        # Get response from Groq
                        ai_msg = llm.invoke(messages)

                        # Display response
                        st.header("Groq's Response:")
                        st.markdown(message_text(ai_msg))

                    except Exception as e:
                        st.error(f"Error getting response from Groq: {str(e)}")
            else:
                st.warning("Please enter a problem or question.")

    except Exception as e:
        st.error(f"Failed to initialize Groq: {str(e)}")
        st.info("Please check your API key and try again.")

else:
    st.warning("Please enter your Groq API key in the .env file to get started.")
    st.info("""
    To use this app:
    1. Get your API key from [Groq Console](https://console.groq.com/keys)
    2. Enter it in the .env file as GROQ_API_KEY
    3. Ask Groq any question or problem
    4. Get step-by-step explanations with fast AI inference!
    """)