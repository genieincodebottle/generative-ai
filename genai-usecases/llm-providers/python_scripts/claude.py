import streamlit as st
from langchain_anthropic import ChatAnthropic
import os
# Environment Variables
from dotenv import load_dotenv
from llm_text import message_text

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Claude Assistant",
    layout="wide"
)

# Ensure the ANTHROPIC_API_KEY is set in the environment
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error(
        "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and put your "
        "key in it, then restart this app.\n\nGet a key at: https://console.anthropic.com/settings/keys"
    )
    st.stop()
os.environ["ANTHROPIC_API_KEY"] = api_key


# Title and description
st.title("Claude Assistant")
st.markdown("A helpful assistant powered by Anthropic's Claude that explains problems step-by-step.")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    # Model selection
    # Aliases, not dated snapshots. A dated snapshot ID stops resolving when
    # that snapshot retires, and the app then fails on its first call.
    model_options = [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5"
    ]
    selected_model = st.selectbox("Select Claude Model:", model_options)

    # Temperature setting
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Main interface
if api_key:

    # Initialize the LLM
    try:
        llm = ChatAnthropic(
            model=selected_model,
            temperature=temperature
        )

        # Problem input
        st.header("Ask Claude")
        problem = st.text_area(
            "Enter your problem or question:",
            placeholder="e.g., Provide me python code for sudoku, Explain quantum physics, Solve 2x + 5 = 15",
            height=100
        )

        # Submit button
        if st.button("Get Answer", type="primary"):
            if problem.strip():
                with st.spinner("Claude is thinking..."):
                    try:
                        # Create messages
                        messages = [
                            (
                                "system",
                                "You are a helpful assistant that explains problems step-by-step."
                            ),
                            ("human", f"Solve this problem step by step: {problem}")
                        ]

                        # Get response from Claude
                        ai_msg = llm.invoke(messages)

                        # Display response
                        st.header("Claude's Response:")
                        st.markdown(message_text(ai_msg))

                    except Exception as e:
                        st.error(f"Error getting response from Claude: {str(e)}")
            else:
                st.warning("Please enter a problem or question.")

    except Exception as e:
        st.error(f"Failed to initialize Claude: {str(e)}")
        st.info("Please check your API key and try again.")

else:
    st.warning("Please enter your Anthropic API key in the .env file to get started.")
    st.info("""
    To use this app:
    1. Get your API key from [Anthropic Console](https://console.anthropic.com/settings/keys)
    2. Enter it in the sidebar
    3. Ask Claude any question or problem
    4. Get step-by-step explanations!
    """)
