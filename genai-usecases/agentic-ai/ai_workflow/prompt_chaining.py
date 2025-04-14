"""
Medical Report Analysis Using Prompt Chaining
This module implements a prompt chaining pattern for medical report analysis,
processing reports through sequential specialized prompts.
"""
from typing import Dict, List, Tuple
import streamlit as st
from utils.llm import llm_call
from utils.prompts import MEDICAL_REPORT_STEPS

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

def validate_report(report: str, validation_criteria: Dict):
    """
    Validates if the medical report has required components.
    
    Args:
        report (str): Medical report text
        validation_criteria (Dict): Criteria for validating report components
        
    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: Lists of present and missing components
    """
    report_lower = report.lower()
    present_components = []
    missing_components = []
    
    for key, criteria in validation_criteria.items():
        if any(keyword in report_lower for keyword in criteria["keywords"]):
            present_components.append((criteria["icon"], criteria["message"]))
        else:
            missing_components.append((criteria["icon"], criteria["message"]))
            
    return present_components, missing_components

def get_validation_criteria():
    """
    Returns the validation criteria for medical report components.
    
    Returns:
        Dict: Validation criteria for different report components
    """
    return {
        "patient": {
            "keywords": ["patient", "demographics", "year-old", "y.o.", "yo"],
            "message": "patient information",
            "icon": "👤"
        },
        "symptoms": {
            "keywords": [
                "symptoms", "chief complaint", "presents with", "reports", 
                "complains of", "experiencing", "pain", "shortness of breath",
                "fatigue", "discomfort"
            ],
            "message": "symptoms or chief complaint",
            "icon": "🤒"
        },
        "history": {
            "keywords": ["history", "previous", "prior", "past medical"],
            "message": "medical history",
            "icon": "📚"
        },
        "examination": {
            "keywords": [
                "exam", "examination", "physical", "assessment",
                "reveals", "noted", "observed", "auscultation",
                "palpation", "inspection"
            ],
            "message": "physical examination",
            "icon": "🔍"
        },
        "vitals": {
            "keywords": [
                "vitals", "vital signs", "bp", "hr", "rr", "temp", "temperature",
                "blood pressure", "heart rate", "respiratory rate", "spo2", "pulse"
            ],
            "message": "vital signs",
            "icon": "📊"
        }
    }

def process_prompt_chain(input_text: str, prompts: List[str], show_steps: bool = True) -> str:
    """
    Process medical report through sequential prompt chain.
    
    Args:
        input_text (str): Initial medical report text
        prompts (List[str]): List of prompts for sequential processing
        show_steps (bool): Whether to display intermediate steps
        
    Returns:
        str: Processed result string
    """
    if st.session_state.get('processing', False):
        return "Processing already in progress"
    
    st.session_state.processing = True
    result = input_text
    
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, prompt in enumerate(prompts, 1):
            progress = i / len(prompts)
            step_name = prompt.split('\n')[0]
            
            progress_text.text(f"Processing: {step_name}")
            progress_bar.progress(progress)
            
            result = llm_call(
                f"{prompt}\nInput: {result}", 
                st.session_state.temperature,
                st.session_state.selected_llm_provider,
                st.session_state.selected_llm_model
            )
            
            if show_steps:
                with st.expander(f"Step {i}: {step_name}", expanded=False):
                    st.code(result)
        
        progress_text.text("✨ Processing complete!")
        progress_bar.progress(1.0)
        return result
        
    except Exception as e:
        st.error(f"❌ Error in chain processing: {str(e)}")
        return f"Error: {str(e)}"
    
    finally:
        st.session_state.processing = False

# Display Functions
def format_report_result(result: str) -> str:
    """
    Format the analysis result with proper markdown structure.
    
    Args:
        result (str): Raw analysis result string
        
    Returns:
        str: Formatted markdown string
    """
    formatted_result = "# 📋 Medical Report Analysis\n\n"
    section_icons = {
        "clinical summary": "🏥",
        "risk assessment": "⚠️",
        "recommendations": "💡",
        "follow-up plan": "📅"
    }
    
    for section in result.split('**'):
        section = section.strip()
        if not section or section.lower() in ["medical report analysis", "standardized report format"]:
            continue
        
        is_heading = any(heading in section.lower() for heading in section_icons)
        if is_heading:
            current_section = section.lower()
            icon = section_icons.get(current_section, "📝")
            formatted_result += f"## {icon} {section}\n\n"
        else:
            formatted_result += f"{section}\n\n"
    
    return formatted_result

def display_validation_results(present: List[Tuple[str, str]], missing: List[Tuple[str, str]]) -> None:
    """
    Display validation results in the Streamlit UI.
    
    Args:
        present (List[Tuple[str, str]]): List of present components
        missing (List[Tuple[str, str]]): List of missing components
    """
    if present:
        st.markdown("### ✅ Present Components")
        for icon, component in present:
            st.success(f"{icon} {component.title()}")
    
    if missing:
        st.markdown("### ❌ Missing Components")
        for icon, component in missing:
            st.warning(f"{icon} {component.title()}")

def display_analysis_results(result: str) -> None:
    """
    Display the analysis results in a well-organized format.
    
    Args:
        result (str): Analysis result string to display
    """
    st.success("✅ Analysis completed successfully!")
    
    tab1, tab2 = st.tabs(["📋 Report Analysis", "⚙️ Processing Steps"])
    
    with tab1:
        formatted_result = format_report_result(result)
        st.markdown(formatted_result)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download as Markdown",
                data=formatted_result,
                file_name="medical_report_analysis.md",
                mime="text/markdown",
                key='markdown_download'
            )
        with col2:
            st.download_button(
                label="📥 Download as Text",
                data=result,
                file_name="medical_report_analysis.txt",
                mime="text/plain",
                key='text_download'
            )
    
    with tab2:
        st.markdown("### Analysis Pipeline")
        for i, step in enumerate(MEDICAL_REPORT_STEPS, 1):
            with st.expander(f"Step {i}: {step.split('/n')[0]}", expanded=False):
                st.markdown(step)

def render_workflow_diagram() -> None:
    """Render the medical report processing workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent / 'images'
        
        prompt_chain_diagram = Image.open(image_path/ 'prompt_chaining.png')
        st.image(prompt_chain_diagram, caption='High Level Architecture')
        
        sequence_diagram = Image.open(image_path/ 'prompt_chain_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def render_usage_instruction() -> None:
    """Render usage instructions for the medical report analysis system."""
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Input Report**
            - Use sample report or enter your own
            - Ensure all required components are present
        
        2. **Configuration**
            - Enable/disable analysis steps display
        
        3. **Review Results**
            - Check report validation
            - Review analysis steps
            - Download results
        
        ### Required Components
        
        - 👤 Patient Information
        - 🤒 Symptoms/Chief Complaint
        - 📚 Medical History
        - 🔍 Physical Examination
        - 📊 Vital Signs
        """)

def render_patient_info_form() -> Tuple[bool, str, bool]:
    """
    Render the medical report input form.
    
    Returns:
        Tuple[bool, str, bool]: Tuple containing (submitted state, input text, show steps flag)
    """
    with st.form("report_info"):
        use_sample = st.checkbox("Use sample medical report", value=True)
        input_text = SAMPLE_MEDICAL_REPORT if use_sample else st.text_area(
            "Medical Report",
            """Patient Information:
            [Demographics]
            Chief Complaint:
            [Primary symptoms]
            History:
            [Relevant medical history]
            Examination:
            [Physical findings]
            Vitals:
            [Current vital signs]""",
            height=150,
            help="Enter the patient's medical report here"
        )
        
        if use_sample:
            st.text_area(
                "Sample Report (read-only)", 
                input_text, 
                height=150, 
                disabled=True,
                help="This is a sample medical report for demonstration"
            )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            show_steps = st.checkbox(
                "Show Analysis Steps",
                value=True,
                help="Display intermediate analysis steps"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Start Analysis",
                use_container_width=True
            )
            
    return submitted, input_text, show_steps

def render_prompt_chain_medical_analysis() -> None:
    """Main function to render the medical report analysis using Prompt Chaining."""
    st.subheader("Prompt Chaining - Medical Analysis")
    
    # Render workflow diagram and instructions
    render_workflow_diagram()
    render_usage_instruction()
    
    # Get input from form
    submitted, input_text, show_steps = render_patient_info_form()
    
    if submitted:
        if not input_text:
            st.warning("⚠️ Please enter a medical report to analyze")
            return
        
        try:
            validation_criteria = get_validation_criteria()
            present, missing = validate_report(input_text, validation_criteria)
            display_validation_results(present, missing)
            
            if not missing:
                with st.spinner("Analyzing medical report..."):
                    result = process_prompt_chain(input_text, MEDICAL_REPORT_STEPS, show_steps)
                    
                    if not result.startswith("Error"):
                        display_analysis_results(result)
                    else:
                        st.error("❌ Analysis failed")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)