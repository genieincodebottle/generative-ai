"""
Medical Tool Calling System
This module implements a tool calling system for medical cases,
analyzing cases and executing appropriate medical tools based on analysis.
"""
from typing import Dict, Tuple
import json
import streamlit as st
from utils.llm import llm_call, extract_xml
from utils.llm_tools import execute_medical_tool
from utils.prompts import MEDICAL_TOOL_PROMPT

from PIL import Image
from pathlib import Path

SAMPLE_MEDICAL_REPORT = """
    Patient Visit Summary:
    73-year-old female presents with increasing shortness of breath over past 2 weeks.
    BP 138/82, HR 88, RR 20, T 37.2°C, SpO2 94% on room air.
    Reports fatigue and mild chest pain on exertion.
    History of hypertension and type 2 diabetes.
    Current medications include Metformin 1000mg BID and Lisinopril 10mg daily.
    Physical exam reveals mild bilateral ankle edema and decreased breath sounds at bases.
    ECG shows normal sinus rhythm with nonspecific ST-T wave changes.
    Basic labs ordered including CBC, CMP, and BNP.
    Patient scheduled for follow-up echocardiogram.
    """

# Core Tool Functions
def validate_tool_data(tool_data: Dict) -> None:
    """
    Validate tool selection data structure.
    
    Args:
        tool_data (Dict): Tool data to validate
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(tool_data, dict):
        raise ValueError(f"Expected dict, got {type(tool_data)}")
        
    if "function" not in tool_data:
        raise ValueError("Missing 'function' in tool data")
        
    if "args" not in tool_data:
        raise ValueError("Missing 'args' in tool data")
        
    if not isinstance(tool_data["args"], dict):
        raise ValueError(f"Expected dict for args, got {type(tool_data['args'])}")

def clean_json_string(json_str: str) -> str:
    """
    Clean and prepare JSON string for parsing.
    
    Args:
        json_str (str): Raw JSON string
        
    Returns:
        str: Cleaned JSON string
    """
    return (json_str
        .strip()
        .replace('\n', '')
        .replace('\r', '')
        .replace('\\n', '')
        .replace('\\r', '')
        .replace('\\', '')
    )

def process_tool_selection(response: str, show_debug: bool = False) -> Tuple[str, Dict]:
    """
    Process LLM tool selection response.
    
    Args:
        response (str): Raw LLM response
        show_debug (bool): Whether to show debug information
        
    Returns:
        Tuple[str, Dict]: Tuple of (tool name, tool arguments)
    """
    try:
        tool_selection = extract_xml(response, "tool_selection")
        
        if show_debug:
            st.write("Raw tool selection:", tool_selection)
        
        cleaned_json = clean_json_string(tool_selection)
        
        if show_debug:
            st.write("Cleaned JSON:", cleaned_json)
        
        tool_data = json.loads(cleaned_json)
        validate_tool_data(tool_data)
        
        return tool_data["function"], tool_data["args"]
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error. Raw content: {tool_selection}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        st.error(f"Processing error. Content: {tool_selection}")
        raise ValueError(f"Error processing tool selection: {str(e)}")

def analyze_and_execute(
    medical_case: str, 
    patient_id: str, 
    show_debug: bool = False
) -> Dict:
    """
    Analyze medical case and execute appropriate tool.
    
    Args:
        medical_case (str): Medical case description
        patient_id (str): Patient identifier
        show_debug (bool): Whether to show debug information
        
    Returns:
        Dict: Analysis and execution results
    """
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Step 1: Format case and prompt
        progress_text.text("Preparing analysis...")
        progress_bar.progress(0.2)
        
        formatted_case = f"Patient ID: {patient_id}\n{medical_case}"
        prompt = MEDICAL_TOOL_PROMPT.format(
            case=formatted_case,
            patient_id=patient_id
        )
        
        # Step 2: Get LLM's analysis
        progress_text.text("Analyzing case...")
        progress_bar.progress(0.4)
        
        response = llm_call(prompt, st.session_state.temperature, 
                            st.session_state.selected_llm_provider, 
                            st.session_state.selected_llm_model)
        analysis = extract_xml(response, "analysis")
        
        # Step 3: Process tool selection
        progress_text.text("Selecting appropriate tool...")
        progress_bar.progress(0.6)
        
        tool_name, tool_args = process_tool_selection(response, show_debug)
        
        # Ensure patient_id is in arguments
        tool_args["patient_id"] = tool_args.get("patient_id", patient_id)
        
        # Step 4: Execute tool
        progress_text.text("Executing selected tool...")
        progress_bar.progress(0.8)
        
        result = execute_medical_tool(tool_name, tool_args)
        
        progress_text.text("Process complete!")
        progress_bar.progress(1.0)
        
        return {
            "analysis": analysis,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result
        }
        
    except Exception as e:
        st.error(f"Error in analysis and execution: {str(e)}")
        return {"error": str(e)}

# Display Functions
def render_workflow_diagram() -> None:
    """Render the tool calling workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
                # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'tool_calling.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'tool_calling_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def render_usage_instruction() -> None:
    """Render usage instructions for the tool calling system."""
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Input Case**
            - Enter patient ID
            - Use sample case or enter your own
            - Provide detailed medical information
        
        2. **Configuration**
            - Enable debug information if needed
            - Adjust model temperature if required
        
        3. **Review Results**
            - Check case analysis
            - Verify tool selection
            - Review execution results
        
        ### Available Tools
        
        - 📅 Schedule Appointments
        - 🔬 Order Lab Tests
        - 👨‍⚕️ Make Specialist Referrals
        - 📝 Update Patient Records
        
        ### Best Practices
        
        - Provide clear, detailed case information
        - Review tool selection before execution
        - Save important results using download option
        - Enable debug mode for troubleshooting
        """)

def render_patient_info_form() -> Tuple[bool, str, str, bool]:
    """
    Render the medical case input form.
    
    Returns:
        Tuple[bool, str, str, bool]: Tuple containing (submitted state, patient_id, 
                                    medical case, show debug flag)
    """
    with st.form("case_info"):
        col1, col2 = st.columns([1, 3])
        with col1:
            patient_id = st.text_input("Patient ID", "P12345")
            
        use_sample = st.checkbox("Use sample medical case", value=True)
        if use_sample:
            medical_case = SAMPLE_MEDICAL_REPORT
            st.text_area(
                "Sample Case (read-only)", 
                medical_case, 
                height=150, 
                disabled=True,
                help="This is a sample medical case for demonstration"
            )
        else:
            medical_case = st.text_area(
                "Medical Case",
                """Patient Case:
                [Detailed patient information]
                [Clinical findings]
                [Relevant history]
                [Current status]""",
                height=150,
                help="Enter the patient's medical case details here"
            )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            show_debug = st.checkbox(
                "Show Debug Info",
                value=False,
                help="Display detailed processing information"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Start Analysis",
                use_container_width=True
            )
            
    return submitted, patient_id, medical_case, show_debug

def display_execution_results(result: Dict) -> None:
    """
    Display the execution results in a well-organized format.
    
    Args:
        result (Dict): Execution results to display
    """
    st.success("✅ Case processing completed successfully!")
    
    tab1, tab2, tab3 = st.tabs([
        "📋 Analysis",
        "🔧 Tool Execution",
        "📊 Results"
    ])
    
    with tab1:
        st.markdown("### 🔍 Case Analysis")
        st.markdown("---")
        st.markdown(result["analysis"])
    
    with tab2:
        st.markdown("### ⚙️ Selected Tool")
        st.markdown("---")
        st.info(f"**Selected Tool:** {result['tool_name']}")
        
        with st.expander("Tool Arguments", expanded=True):
            st.json(result["tool_args"])
    
    with tab3:
        st.markdown("### 📊 Execution Results")
        st.markdown("---")
        st.success(result["result"])
        
        result_json = json.dumps({
            "analysis": result["analysis"],
            "tool_execution": {
                "name": result["tool_name"],
                "arguments": result["tool_args"]
            },
            "result": result["result"]
        }, indent=2)
        
        st.download_button(
            label="📥 Download Results",
            data=result_json,
            file_name="tool_execution_results.json",
            mime="application/json"
        )

def render_tool_calling_medical_analysis() -> None:
    """Main function to render the tool orchestration interface."""
    st.subheader("Tool Calling - Medical Analysis")
    
    # Render workflow diagram and instructions
    render_workflow_diagram()
    render_usage_instruction()
    
    # Get input from form
    submitted, patient_id, medical_case, show_debug = render_patient_info_form()
    
    if submitted:
        if medical_case:
            try:
                with st.spinner("Processing medical case..."):
                    result = analyze_and_execute(
                        medical_case, 
                        patient_id, 
                        show_debug
                    )
                    
                    if "error" not in result:
                        display_execution_results(result)
                    else:
                        st.error("❌ Case processing failed.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
        else:
            st.warning("⚠️ Please enter a medical case")