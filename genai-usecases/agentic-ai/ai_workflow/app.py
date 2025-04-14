import streamlit as st
from utils.llm import LLM_CONFIGS

# Importing all level of AI Workflow
from prompt_chaining import render_prompt_chain_medical_analysis
from parallelization import render_parallelization_medical_analysis
from query_routing import render_query_routing_medical_analysis
from evaluator_and_optimizer import render_eval_and_optimize_medical_analysis
from orchestrator import render_orchestrator_medical_analysis
from tool_calling import render_tool_calling_medical_analysis

# Set page config as the first command
st.set_page_config(
        page_title="Agentic-Workflow based Healthcare system",
        page_icon="🤖",
        layout="wide"
    )

def reset_session_state():
    """
    Ensure all required session state variables exist and 
    initialize them with default values if not present
    """
    defaults = {
        'temperature': 0.3,
        'processing': False,
        'selected_llm_provider': list(LLM_CONFIGS.keys())[0],
        'selected_llm_model': LLM_CONFIGS[list(LLM_CONFIGS.keys())[0]]["models"][0]
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_llm_configuration():
    """Render LLM configuration section in sidebar"""
    st.markdown("## LLM Configuration")
    
    # LLM Provider Selection
    provider = st.selectbox(
        "Select LLM Provider",
        options=list(LLM_CONFIGS.keys()),
        key='selected_llm_provider',
        help="Choose the AI model provider"
    )
    
    # Model Selection based on provider
    available_models = LLM_CONFIGS[provider]["models"]
    st.selectbox(
        "Select Model",
        options=available_models,
        key='selected_llm_model',
        help=f"Choose the specific {provider} model"
    )
    
    # Temperature configuration with proper spacing
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        key='temperature',
        help="Lower values for more focused responses, higher for more creative ones"
    )
    
    # Model information display with improved formatting
    st.markdown("""
    ### Current Configuration
    """)
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(f"""
        - Provider: {provider}
        - Model: {st.session_state.selected_llm_model}
        """)
    with info_col2:
        st.markdown(f"""
        - Temperature: {st.session_state.temperature}
        """)
    
    st.markdown("""
    ### Capabilities
    - Medical terminology processing
    - Clinical reasoning
    - Evidence-based analysis
    """)

def main():
    # Initialize session state
    reset_session_state()
    
    # Sidebar layout
    with st.sidebar:
        st.markdown("# Agentic Workflow Patterns")
        
        # Workflow Selection with improved spacing
        selected_workflow = st.selectbox(
            "Select Workflow for Medical Analysis",
            ["Prompt Chaining", 
             "Parallelization",
             "Routing",
             "Evaluator and Optimizers",
             "Orchestrator",
             "Tool Calling"],
            key='workflow_selector'
        )
        
        st.markdown("---")
        render_llm_configuration()
        
        st.markdown("---")
        st.markdown("""
        ## Documentation
        
        ### Workflow Levels
        1. **Prompt Chaining**: Sequential prompts for basic analysis
        2. **Parallelization**: Concurrent analysis tasks
        3. **Query Routing**: Dynamic task distribution
        4. **Evaluator/Optimizer**: Quality control and improvement
        5. **Orchestrator**: Complex workflow management
        6. **Tool Calling**: External tool integration
        
        ### Best Practices
        - Start with simpler workflows
        - Monitor performance metrics
        - Review agent decisions
        - Adjust temperature as needed
        """)
    
    # Main content area
    st.header("🏥 Healthcare System")
    
    try:
        # Route to appropriate Agentic Level Workflow
        workflow_mapping = {
            "Prompt Chaining": render_prompt_chain_medical_analysis,
            "Parallelization": render_parallelization_medical_analysis,
            "Routing": render_query_routing_medical_analysis,
            "Evaluator and Optimizers": render_eval_and_optimize_medical_analysis,
            "Orchestrator": render_orchestrator_medical_analysis,
            "Tool Calling": render_tool_calling_medical_analysis
        }
        
        if selected_workflow in workflow_mapping:
            workflow_mapping[selected_workflow]()
     
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.session_state.processing:
            st.session_state.processing = False

if __name__ == "__main__":
    main()