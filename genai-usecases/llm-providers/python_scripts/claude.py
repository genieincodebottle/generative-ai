import streamlit as st
from langchain_anthropic import ChatAnthropic
import os
# Environment Variables
from dotenv import load_dotenv

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Claude Assistant",
    page_icon="🤖",
    layout="wide"
)

# Ensure the GOOGLE_API_KEY is set in the environment
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please set it at .env file before running the app.")
os.environ["ANTHROPIC_API_KEY"] = api_key


# Title and description
st.title("🤖 Claude Assistant")
st.markdown("A helpful assistant powered by Anthropic's Claude that explains problems step-by-step.")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    # Model selection
    model_options = [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022"
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
                        st.markdown(ai_msg.content)
                        
                    except Exception as e:
                        st.error(f"Error getting response from Claude: {str(e)}")
            else:
                st.warning("Please enter a problem or question.")
                
    except Exception as e:
        st.error(f"Failed to initialize Claude: {str(e)}")
        st.info("Please check your API key and try again.")

else:
    st.warning("👈 Please enter your Anthropic API key in the .env file to get started.")
    st.info("""
    To use this app:
    1. Get your API key from [Anthropic Console](https://console.anthropic.com/settings/keys)
    2. Enter it in the sidebar
    3. Ask Claude any question or problem
    4. Get step-by-step explanations!
    """)
