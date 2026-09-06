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

# Ensure the OPENAI_API_KEY is set in the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and put your "
        "key in it, then restart this app.\n\nGet a key at: https://platform.openai.com/api-keys"
    )
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# Title and description
st.title("OpenAI Assistant")
st.markdown("A helpful assistant powered by OpenAI's GPT models that explains problems step-by-step.")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    # Model selection
    # gpt-4.5 was shut down on 2025-07-14 and gpt-5-2025-08-07 is a dated
    # snapshot with a retirement date; neither belongs in a default list.
    model_options = [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-4.1"
    ]
    selected_model = st.selectbox("Select OpenAI Model:", model_options)

    # Temperature setting
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Main interface
if api_key:

    # Initialize the LLM
    try:
        llm = ChatOpenAI(
            model=selected_model,
            temperature=temperature
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