"""
Medical Query Routing System
This module implements a routing system for medical queries,
directing cases to appropriate specialized medical teams based on case analysis.
"""
from typing import Dict, Tuple
import streamlit as st
from utils.llm import llm_call, extract_xml

from PIL import Image
from pathlib import Path

SAMPLE_MEDICAL_QUERIES = [
    """Patient Case:
    58-year-old male experiencing severe chest pain radiating to left arm for past hour.
    History of hypertension. No prior cardiac events.
    Sweating and nauseous. BP 160/95.
    Requesting guidance on immediate steps.""",

    """Patient Case:
    42-year-old female with type 2 diabetes for 5 years.
    Recent A1C 8.2%, up from 7.1% three months ago.
    Reports difficulty following diet plan and checking glucose regularly.
    Seeking management strategy adjustment.""",

    """Patient Case:
    35-year-old female, no significant medical history.
    Due for annual check-up. Last mammogram 2 years ago.
    Family history of breast cancer (mother, age 45).
    Requesting preventive care planning."""
]

MEDICAL_ROUTES = {
    "urgent_care": """You are an urgent care triage specialist. Follow these guidelines:
    1. Start with "Urgent Care Assessment:"
    2. Evaluate symptoms for severity
    3. Determine appropriate level of care
    4. Provide immediate action steps
    5. Include red flag warnings if applicable""",

    "chronic_care": """You are a chronic care coordinator. Follow these guidelines:
    1. Start with "Chronic Care Management:"
    2. Review ongoing conditions
    3. Assess medication compliance
    4. Evaluate lifestyle factors
    5. Provide long-term management strategies""",

    "preventive_care": """You are a preventive care specialist. Follow these guidelines:
    1. Start with "Preventive Care Recommendations:"
    2. Review risk factors
    3. Suggest screening tests
    4. Provide lifestyle recommendations
    5. Set prevention goals""",

    "specialist_referral": """You are a medical referral coordinator. Follow these guidelines:
    1. Start with "Specialist Referral Assessment:"
    2. Identify specialty need
    3. Review case complexity
    4. Determine urgency
    5. Provide referral process steps"""
}

# Core Routing Functions
def analyze_and_route(input: str, routes: Dict[str, str], temperature: float, provider: str, model: str) -> Tuple[str, str, str]:
    """
    Analyze query and determine appropriate medical team routing.
    
    Args:
        input (str): The medical query to analyze
        routes (Dict[str, str]): Available routing options
        temperature (float): LLM temperature setting
        provider (str): LLM Provider
        model (str): LLM Model
        
    Returns:
        Tuple[str, str, str]: Tuple containing (reasoning, selected route, error message if any)
    """
    try:
        selector_prompt = f"""
        Analyze the input and select the most appropriate medical team from these options: {list(routes.keys())}
        First explain your reasoning, then provide your selection in this XML format:

        <reasoning>
        Brief explanation of why this case should be routed to a specific team.
        Consider symptoms, urgency, and care requirements.
        </reasoning>

        <selection>
        The chosen team name
        </selection>

        Input: {input}""".strip()

        route_response = llm_call(selector_prompt, temperature, provider, model)
        reasoning = extract_xml(route_response, 'reasoning')
        route_key = extract_xml(route_response, 'selection').strip().lower()

        if not reasoning or not route_key:
            return "", "", "Could not extract routing information"
            
        if route_key not in routes:
            return reasoning, route_key, f"Invalid route '{route_key}' selected"

        return reasoning, route_key, ""
        
    except Exception as e:
        return "", "", str(e)

def get_team_response(input: str, route_key: str, routes: Dict[str, str], temperature: float, provider: str, model: str) -> str:
    """
    Get response from selected medical team.
    
    Args:
        input (str): Original medical query
        route_key (str): Selected team identifier
        routes (Dict[str, str]): Available routing options
        temperature (float): LLM temperature setting
        provider (str): LLM Provider
        model (str): LLM Model
        
    Returns:
        str: Team's response to the query
    """
    try:
        selected_prompt = routes[route_key]
        response = llm_call(
            f"{selected_prompt}\nInput: {input}", 
            temperature,
            provider,
            model
        )
        return response
    except Exception as e:
        return f"Error getting team response: {str(e)}"

def process_routing_request(input: str, routes: Dict[str, str], show_reasoning: bool = True) -> str:
    """
    Process a medical query through the routing system.
    
    Args:
        input (str): The medical query to process
        routes (Dict[str, str]): Available routing options
        show_reasoning (bool): Whether to display routing analysis
        
    Returns:
        str: Processed response from selected team
    """
    if st.session_state.processing:
        return "Processing already in progress"
        
    st.session_state.processing = True
    
    if show_reasoning:
        st.markdown("### 🔄 Routing Analysis")
        st.markdown("---")
        st.text(f"Available routes: {list(routes.keys())}")
    
    try:
        # Initialize progress
        progress_bar = st.progress(0)
        
        # Step 1: Route Analysis
        progress_bar.progress(0.3, text="Determining optimal care team...")
        reasoning, route_key, error = analyze_and_route(
            input, 
            routes, 
            st.session_state.temperature,
            st.session_state.selected_llm_provider,
            st.session_state.selected_llm_model
        )
        
        if error:
            st.error(f"❌ Error: {error}")
            return f"Error: {error}"
            
        # Step 2: Display Routing Decision
        progress_bar.progress(0.6, text="Processing routing decision...")
        if show_reasoning:
            st.markdown("### 📋 Routing Decision")
            st.info(reasoning)
            st.success(f"Selected Team: {route_key.upper()}")
            
        # Step 3: Get Team Response
        progress_bar.progress(0.8, text="Getting team response...")
        response = get_team_response(
            input, 
            route_key, 
            routes, 
            st.session_state.temperature,
            st.session_state.selected_llm_provider,
            st.session_state.selected_llm_model
        )
        
        progress_bar.progress(1.0, text="Response complete!")
        return response
        
    except Exception as e:
        st.error(f"❌ Error in routing: {str(e)}")
        return f"Error: {str(e)}"
        
    finally:
        st.session_state.processing = False

# Display Functions
def render_workflow_diagram() -> None:
    """Render the query routing workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'routing.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'routing_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def render_usage_instructions() -> None:
    """Render usage instructions for the query routing system."""
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Input Query**
            - Choose a sample case or enter your own
            - Provide detailed patient information
        
        2. **Configuration**
            - Enable/disable analysis steps display
            - Adjust model temperature if needed
        
        3. **Review Results**
            - Check routing analysis
            - Read team response
            - View available routes
        
        ### Available Care Teams
        
        - 🚑 Emergency Team: Urgent/critical cases
        - 👨‍⚕️ Primary Care: General medical issues
        - 👩‍⚕️ Specialist Team: Specific conditions
        - 🧠 Psychiatric Team: Mental health cases
        
        ### Best Practices
        
        - Provide clear, detailed information
        - Include relevant medical history
        - Specify current symptoms
        - Note any urgent concerns
        """)

def render_patient_info_form() -> Tuple[bool, str, bool]:
    """
    Render the medical query input form.
    
    Returns:
        Tuple[bool, str, bool]: Tuple containing (submitted state, query text, show reasoning flag)
    """
    with st.form("query_info"):
        use_sample = st.checkbox("Use sample query", value=True)
        if use_sample:
            selected_query = st.selectbox(
                "Select sample query",
                range(len(SAMPLE_MEDICAL_QUERIES)),
                format_func=lambda x: f"Medical Case {x+1}"
            )
            query = SAMPLE_MEDICAL_QUERIES[selected_query]
            st.text_area(
                "Sample Query (read-only)", 
                query, 
                height=150, 
                disabled=True,
                help="This is a sample medical case for demonstration"
            )
        else:
            query = st.text_area(
                "Medical Query",
                """Patient Case:
                [Age, gender]
                [Presenting symptoms]
                [Relevant history]
                [Current status]""",
                height=150,
                help="Enter the patient's medical query details here"
            )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            show_reasoning = st.checkbox(
                "Show Analysis Steps",
                value=True,
                help="Display detailed routing analysis steps"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Start Analysis",
                use_container_width=True
            )
            
    return submitted, query, show_reasoning

def display_routing_results(response: str) -> None:
    """
    Display the routing results in a well-organized format.
    
    Args:
        response (str): Team response to display
    """
    st.success("✅ Query processing completed!")
    
    tab1, tab2 = st.tabs([
        "📋 Team Response",
        "ℹ️ Available Routes"
    ])
    
    with tab1:
        st.markdown("### 👨‍⚕️ Care Team Response")
        st.markdown("---")
        st.write(response)
    
    with tab2:
        st.markdown("### 🔀 Available Care Teams")
        st.markdown("---")
        for team, description in MEDICAL_ROUTES.items():
            with st.expander(f"📍 {team.upper()}", expanded=False):
                st.write(description.split('\n')[0])

def render_query_routing_medical_analysis() -> None:
    """Main function to render the Query Routing Medical Analysis interface."""
    st.subheader("Query Routing - Medical Analysis")

    # Render workflow diagram and instructions
    render_workflow_diagram()
    render_usage_instructions()
    
    # Get input from form
    submitted, query, show_reasoning = render_patient_info_form()
    
    if submitted:
        if query:
            try:
                with st.spinner("Processing medical query..."):
                    response = process_routing_request(query, MEDICAL_ROUTES, show_reasoning)
                    
                    if not response.startswith("Error"):
                        display_routing_results(response)
                    else:
                        st.error("❌ Query processing failed.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                st.exception(e)
        else:
            st.warning("⚠️ Please enter a medical query")