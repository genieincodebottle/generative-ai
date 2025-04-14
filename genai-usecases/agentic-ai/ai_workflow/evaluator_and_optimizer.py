"""
Medical Case Analysis Evaluator and Optimizer
This module implements evaluation and optimization patterns for medical case analysis,
providing iterative improvement based on clinical feedback.
"""
from typing import List, Dict
import streamlit as st
import json

from utils.llm import llm_call, extract_xml
from utils.prompts import (
    MEDICAL_EVALUATOR_PROMPT,
    MEDICAL_GENERATOR_PROMPT
)

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

def generate_medical_response(
    prompt: str, 
    task: str, 
    context: str = "", 
    show_details: bool = True
) -> tuple[str, str]:
    """
    Generate and improve a medical analysis based on feedback.
    
    Args:
        prompt (str): Base prompt for generation
        task (str): Medical task description
        context (str, optional): Additional context from previous iterations
        show_details (bool): Whether to display generation details
        
    Returns:
        tuple[str, str]: Tuple containing thoughts and generated result
    """
    try:
        full_prompt = f"{prompt}\n{context}\nMedical Task: {task}" if context else f"{prompt}\nMedical Task: {task}"
        response = llm_call(full_prompt, 
                            st.session_state.temperature,
                            st.session_state.selected_llm_provider,
                            st.session_state.selected_llm_model)
        thoughts = extract_xml(response, "thoughts")
        result = extract_xml(response, "response")
        
        if show_details:
            with st.expander("🔍 Generation Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 💭 Analysis Thoughts")
                    st.markdown(thoughts)
                with col2:
                    st.markdown("### 📋 Generated Response")
                    st.markdown(result)
        
        return thoughts, result
    except Exception as e:
        st.error(f"❌ Error generating medical response: {str(e)}")
        return "", ""

def evaluate_medical_response(
    prompt: str, 
    content: str, 
    task: str, 
    show_details: bool = True
) -> tuple[str, str]:
    """
    Evaluate if a medical analysis meets clinical requirements.
    
    Args:
        prompt (str): Evaluation prompt
        content (str): Content to evaluate
        task (str): Original medical task
        show_details (bool): Whether to display evaluation details
        
    Returns:
        tuple[str, str]: Tuple containing evaluation status and feedback
    """
    try:
        full_prompt = f"{prompt}\nOriginal task: {task}\nContent to evaluate: {content}"
        response = llm_call(full_prompt, 
                            st.session_state.temperature,
                            st.session_state.selected_llm_provider,
                            st.session_state.selected_llm_model)
        evaluation = extract_xml(response, "evaluation")
        feedback = extract_xml(response, "feedback")
        
        if show_details:
            with st.expander("📊 Evaluation Details", expanded=True):
                st.markdown("### ⚖️ Evaluation Status")
                status_colors = {
                    "PASS": "green",
                    "NEEDS_IMPROVEMENT": "orange",
                    "FAIL": "red"
                }
                status_color = status_colors.get(evaluation, "gray")
                st.markdown(f"""
                <div style='padding:10px;border-radius:5px;background-color:{status_color};
                color:white;text-align:center;font-weight:bold'>
                {evaluation}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💡 Feedback")
                st.info(feedback)
        
        return evaluation, feedback
    except Exception as e:
        st.error(f"❌ Error evaluating medical response: {str(e)}")
        return "ERROR", str(e)

def medical_improvement_loop(
    task: str, 
    evaluator_prompt: str, 
    generator_prompt: str,
    show_steps: bool = True
) -> tuple[str, list[dict]]:
    """
    Iteratively improve medical analysis until requirements are met.
    
    Args:
        task (str): Medical task description
        evaluator_prompt (str): Prompt for evaluation
        generator_prompt (str): Prompt for generation
        show_steps (bool): Whether to display iteration steps
        
    Returns:
        tuple[str, list[dict]]: Final result and improvement chain
    """
    if st.session_state.processing:
        return "Processing already in progress", []
        
    st.session_state.processing = True
    memory = []
    chain_of_thought = []
    
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Initial Analysis
        progress_text.text("🔄 Generating initial analysis...")
        progress_bar.progress(0.2)
        thoughts, result = generate_medical_response(generator_prompt, task, show_details=show_steps)
        memory.append(result)
        chain_of_thought.append({"thoughts": thoughts, "result": result})
        
        iteration = 1
        while True:
            if show_steps:
                st.markdown(f"### 📍 Iteration {iteration}")
            
            # Evaluation
            progress_text.text("🔍 Evaluating analysis...")
            progress_bar.progress(0.4 + (iteration * 0.1))
            evaluation, feedback = evaluate_medical_response(
                evaluator_prompt, 
                result, 
                task,
                show_details=show_steps
            )
            
            if evaluation == "PASS":
                progress_text.text("✅ Analysis complete!")
                progress_bar.progress(1.0)
                if show_steps:
                    st.success("Medical analysis meets all requirements!")
                return result, chain_of_thought
            
            # Improvement
            context = "\n".join([
                "Previous attempts:",
                *[f"- {m}" for m in memory],
                f"\nClinical Feedback: {feedback}"
            ])
            
            progress_text.text("💡 Improving analysis based on feedback...")
            thoughts, result = generate_medical_response(
                generator_prompt, 
                task, 
                context,
                show_details=show_steps
            )
            memory.append(result)
            chain_of_thought.append({"thoughts": thoughts, "result": result})
            
            iteration += 1
            if iteration > 5:
                progress_text.text("⚠️ Maximum iterations reached")
                progress_bar.progress(1.0)
                if show_steps:
                    st.warning("Maximum iterations reached. Please review the final analysis.")
                return result, chain_of_thought
                
    except Exception as e:
        st.error(f"❌ Error in improvement loop: {str(e)}")
        return str(e), chain_of_thought
        
    finally:
        st.session_state.processing = False

# UI Functions
def render_workflow_diagram():
    """Render the evaluator and optimizer workflow diagram with mermaid."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'eval.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'eval_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')


def render_usage_instructions():
    """Render comprehensive usage instructions for the evaluation analysis."""
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Input Medical Case**
            - Use sample case or enter your own
            - Ensure comprehensive information
            - Include relevant clinical details
        
        2. **Configure Analysis**
            - Choose to show/hide analysis steps
            - Optionally customize prompts
            - Adjust model temperature if needed
        
        3. **Review Results**
            - Check final analysis
            - Review improvement history
            - Download results if needed
        
        ### Analysis Process
        
        - 🔄 Initial Analysis Generation
        - 🔍 Quality Evaluation
        - 💡 Feedback Generation
        - ✨ Iterative Improvement
        
        ### Best Practices
        
        - Provide detailed medical information
        - Review each iteration's changes
        - Check evaluation feedback
        - Save important analyses
        """)

def render_patient_info_form() -> tuple:
    """
    Render the medical case input form with configuration options.
    
    Returns:
        tuple: Form submission details including inputs and configuration
    """
    with st.form("case_info"):
        use_sample = st.checkbox("Use sample medical case", value=True)
        if use_sample:
            medical_task = SAMPLE_MEDICAL_REPORT
            st.text_area(
                "Sample Case (read-only)", 
                medical_task, 
                height=150, 
                disabled=True,
                help="This is a sample medical case for demonstration"
            )
        else:
            medical_task = st.text_area(
                "Medical Case",
                """Patient Case:
                [Detailed patient information]
                [Clinical findings]
                [Relevant history]
                [Current status]""",
                height=150,
                help="Enter the patient's medical case details here"
            )
        
        use_custom_prompts = st.checkbox(
            "Customize Evaluation Prompts", 
            value=False,
            help="Use custom prompts for evaluation and generation"
        )
        
        if use_custom_prompts:
            col1, col2 = st.columns(2)
            with col1:
                custom_evaluator_prompt = st.text_area(
                    "Evaluator Prompt",
                    MEDICAL_EVALUATOR_PROMPT,
                    height=150
                )
            with col2:
                custom_generator_prompt = st.text_area(
                    "Generator Prompt",
                    MEDICAL_GENERATOR_PROMPT,
                    height=150
                )
        else:
            custom_evaluator_prompt = MEDICAL_EVALUATOR_PROMPT
            custom_generator_prompt = MEDICAL_GENERATOR_PROMPT
        
        col1, col2 = st.columns([2, 1])
        with col1:
            show_steps = st.checkbox(
                "Show Analysis Steps",
                value=True,
                help="Display detailed analysis steps"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Start Analysis",
                use_container_width=True
            )
            
        return (
            submitted, 
            medical_task, 
            custom_evaluator_prompt, 
            custom_generator_prompt,
            show_steps
        )

def display_analysis_results(final_result: str, improvement_chain: List[Dict]):
    """
    Display the analysis results with tabs and interactive elements.
    
    Args:
        final_result (str): Final analysis result
        improvement_chain (List[Dict]): History of improvements
    """
    st.success("✅ Analysis completed successfully!")
    
    tab1, tab2, tab3 = st.tabs([
        "📋 Final Analysis",
        "📊 Improvement History",
        "📈 Statistics"
    ])
    
    with tab1:
        st.markdown("### 🎯 Final Medical Analysis")
        st.markdown("---")
        st.markdown(final_result)
        
        st.download_button(
            "📥 Download Final Analysis",
            final_result,
            file_name="final_medical_analysis.md",
            mime="text/markdown"
        )
    
    with tab2:
        st.markdown("### 🔄 Analysis Evolution")
        st.markdown("---")
        
        for i, step in enumerate(improvement_chain, 1):
            with st.expander(f"📍 Iteration {i}", expanded=i == len(improvement_chain)):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 💭 Clinical Reasoning")
                    st.markdown(step["thoughts"])
                with col2:
                    st.markdown("#### 📋 Analysis")
                    st.markdown(step["result"])
    
    with tab3:
        st.markdown("### 📊 Analysis Statistics")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Iterations", len(improvement_chain))
        with col2:
            st.metric(
                "Improvement Rate",
                f"{(len(improvement_chain) - 1) / len(improvement_chain):.0%}"
            )
        with col3:
            final_length = len(final_result.split())
            initial_length = len(improvement_chain[0]["result"].split())
            st.metric(
                "Content Growth",
                f"{final_length} words",
                delta=f"{final_length - initial_length} words",
                delta_color="normal"
            )
        
        st.download_button(
            "📥 Download Complete Analysis History",
            json.dumps(improvement_chain, indent=2),
            file_name="analysis_history.json",
            mime="application/json"
        )

def render_eval_and_optimize_medical_analysis():
    """Render the main evaluator and optimizer medical analysis interface."""
    st.subheader("Evaluator & Optimizer - Medical Analysis")
    
    render_workflow_diagram()
    render_usage_instructions()

    submitted, medical_task, evaluator_prompt, generator_prompt, show_steps = render_patient_info_form()
    
    if submitted:
        if medical_task:
            try:
                with st.spinner("Processing medical analysis..."):
                    final_result, improvement_chain = medical_improvement_loop(
                        medical_task,
                        evaluator_prompt,
                        generator_prompt,
                        show_steps
                    )
                    
                    if not final_result.startswith("Error"):
                        display_analysis_results(final_result, improvement_chain)
                    else:
                        st.error("❌ Analysis failed.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
        else:
            st.warning("⚠️ Please enter a medical case to analyze")