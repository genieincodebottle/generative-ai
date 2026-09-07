import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
# Environment Variables
from dotenv import load_dotenv
from llm_text import message_text

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Gemini Assistant",
    layout="wide"
)

# Ensure the GOOGLE_API_KEY is set in the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "GOOGLE_API_KEY is not set. Copy .env.example to .env and put your "
        "key in it, then restart this app.\n\nGet a key at: https://aistudio.google.com/apikey"
    )
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# Title and description
st.title("Gemini Assistant")
st.markdown("A helpful assistant powered by Google's Gemini that explains problems step-by-step.")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    # Model selection
    model_options = [
        "gemini-pro-latest",
        "gemini-2.5-flash",
        "gemini-pro-latest"
    ]
    selected_model = st.selectbox("Select Gemini Model:", model_options)

    # Temperature setting
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Main interface
if api_key:

    # Initialize the LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=temperature
        )

        # Problem input
        st.header("Ask Gemini")
        problem = st.text_area(
            "Enter your problem or question:",
            placeholder="e.g., Provide me python code for sudoku, Explain quantum physics, Solve 2x + 5 = 15",
            height=100
        )

        # Submit button
        if st.button("Get Answer", type="primary"):
            if problem.strip():
                with st.spinner("Gemini is thinking..."):
                    try:
                        # Create messages
                        messages = [
                            (
                                "system",
                                "You are a helpful assistant that explains problems step-by-step."
                            ),
                            ("human", f"Solve this problem step by step: {problem}")
                        ]

                        # Get response from Gemini
                        ai_msg = llm.invoke(messages)

                        # Display response
                        st.header("Gemini's Response:")
                        st.markdown(message_text(ai_msg))

                    except Exception as e:
                        st.error(f"Error getting response from Gemini: {str(e)}")
            else:
                st.warning("Please enter a problem or question.")

    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        st.info("Please check your API key and try again.")

else:
    st.warning("Please enter your Google API key in the .env file to get started.")
    st.info("""
    To use this app:
    1. Get your API key from [Google AI Studio](https://aistudio.google.com/apikeyy)
    2. Enter it in the .env file as GOOGLE_API_KEY
    3. Ask Gemini any question or problem
    4. Get step-by-step explanations!
    """)