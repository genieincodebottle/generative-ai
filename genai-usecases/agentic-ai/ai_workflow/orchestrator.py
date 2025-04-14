"""
Medical Case Analysis Orchestrator
This module implements an orchestrator pattern for medical case analysis,
breaking down complex cases into specialized tasks processed by worker LLMs.
"""
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import json
import streamlit as st
from utils.llm import llm_call, extract_xml
from utils.prompts import MEDICAL_ORCHESTRATOR_PROMPT

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

# Orchestrator Functions
def get_orchestrator_analysis(medical_case: str, context: Dict) -> Dict:
    """
    Dynamic orchestrator that analyzes the case and determines necessary subtasks.
    
    Args:
        medical_case (str): The medical case to analyze
        context (Dict): Additional context like clinical setting and urgency
        
    Returns:
        Dict: Analysis results and list of tasks
    """
    
    try:
        orchestrator_input = MEDICAL_ORCHESTRATOR_PROMPT.format(
            case=medical_case,
            context=str(context)
        )
        orchestrator_response = llm_call(orchestrator_input, 
                                        st.session_state.temperature,
                                        st.session_state.selected_llm_provider,
                                        st.session_state.selected_llm_model)
        
        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_dynamic_tasks(tasks_xml)
        
        return {
            "analysis": analysis,
            "tasks": tasks
        }
    except Exception as e:
        raise ValueError(f"Orchestrator analysis error: {str(e)}")

def parse_dynamic_tasks(tasks_xml: str) -> List[Dict]:
    """
    Parse XML tasks with flexible schema for dynamic task types.
    
    Args:
        tasks_xml (str): XML string containing task definitions
        
    Returns:
        List[Dict]: List of parsed task dictionaries
    """
    tasks = []
    if not tasks_xml or not tasks_xml.strip():
        return tasks
        
    task_blocks = tasks_xml.split('</task>')
    
    for block in task_blocks:
        if not block.strip():
            continue
            
        task = {}
        for tag in ['type', 'description', 'requirements', 'priority']:
            start_tag = f'<{tag}>'
            end_tag = f'</{tag}>'
            start_idx = block.find(start_tag)
            end_idx = block.find(end_tag)
            
            if start_idx > -1 and end_idx > -1:
                value = block[start_idx + len(start_tag):end_idx].strip()
                task[tag] = value
        
        if task:
            tasks.append(task)
    
    return tasks

# Worker Functions
def create_worker_prompt(task_info: Dict, original_case: str, context: Dict) -> str:
    """
    Create specialized prompt for each worker based on task requirements.
    
    Args:
        task_info (Dict): Information about the specific task
        original_case (str): The original medical case
        context (Dict): Additional clinical context
        
    Returns:
        str: Formatted prompt for the worker
    """
    return f"""
    You are a specialized medical analyzer focusing on: {task_info['type']}
    
    Original Case: {original_case}
    Task Description: {task_info['description']}
    Required Expertise: {task_info['requirements']}
    Priority Level: {task_info['priority']}
    Clinical Setting: {context.get('clinical_setting', 'Not specified')}
    Urgency Level: {context.get('urgency_level', 'Not specified')}
    
    Return your response in this EXACT format:

    <response>
    1. Key Clinical Findings:
    - [List major findings from the case]
    - [Include relevant vital signs and symptoms]
    - [Note any abnormal results]

    2. Risk Assessment:
    - [Identify immediate risks]
    - [List potential complications]
    - [Note contributing factors]

    3. Detailed Recommendations:
    - [Specify immediate actions]
    - [List follow-up tests]
    - [Include medication adjustments]

    4. Required Care Coordination:
    - [Specify which specialists to involve]
    - [Define follow-up timeline]
    - [Note special care requirements]
    </response>
    """

def process_worker_task(task_info: Dict, original_case: str, context: Dict, temperature: float, provider: str, model: str) -> Dict:
    """
    Process a single worker task with comprehensive output.
    
    Args:
        task_info (Dict): Information about the specific task
        original_case (str): The original medical case
        context (Dict): Additional clinical context
        temperature (float): LLM temperature setting
        provider (str): LLM Provider
        model (str): LLM Model
        
    Returns:
        Dict: Processed task results
    """
    try:
        worker_prompt = create_worker_prompt(task_info, original_case, context)
        response = llm_call(worker_prompt, temperature, provider, model)
        result = extract_xml(response, "response")
        
        if not result or result.isspace():
            result = generate_default_response(task_info)
            
        formatted_result = format_worker_response(result)
        
        return {
            "type": task_info["type"],
            "description": task_info["description"],
            "priority": task_info["priority"],
            "result": formatted_result
        }
    except Exception as e:
        return generate_error_response(task_info, str(e))

def generate_default_response(task_info: Dict) -> str:
    """Generate a default response when worker analysis is empty."""
    return f"""
    1. Key Clinical Findings:
    - Based on the case presentation showing {task_info['type']}
    - Current vital signs and symptoms need evaluation
    - Specific findings require detailed assessment

    2. Risk Assessment:
    - Potential risks identified for {task_info['type']}
    - Complications need to be monitored
    - Contributing factors should be evaluated

    3. Detailed Recommendations:
    - Immediate assessment of {task_info['type']} needed
    - Follow-up testing should be scheduled
    - Treatment plan needs to be established

    4. Required Care Coordination:
    - Specialist consultation recommended
    - Follow-up timeline to be determined
    - Care requirements to be specified based on findings
    """

def generate_error_response(task_info: Dict, error: str) -> Dict:
    """Generate an error response for failed worker tasks."""
    return {
        "type": task_info["type"],
        "description": task_info["description"],
        "priority": task_info.get("priority", "high"),
        "error": error,
        "result": generate_default_response(task_info)
    }

def format_worker_response(result: str) -> str:
    """Format the worker response for better readability."""
    return (result.replace("1.", "\n1.")
                  .replace("2.", "\n2.")
                  .replace("3.", "\n3.")
                  .replace("4.", "\n4.")
                  .replace("- ", "\n- "))

# Task Processing Functions
def process_specialized_tasks(case: str, tasks: List[Dict], context: Dict) -> List[Dict]:
    """
    Process specialized tasks in parallel with dynamic prioritization.
    
    Args:
        case (str): The medical case to analyze
        tasks (List[Dict]): List of tasks to process
        context (Dict): Additional clinical context
        
    Returns:
        List[Dict]: List of processed task results
    """
    worker_results = []
    temperature = st.session_state.temperature
    provider = st.session_state.selected_llm_provider,
    model = st.session_state.selected_llm_model
    
    try:
        if not tasks:
            st.warning("No tasks to process")
            return worker_results
            
        num_workers = max(1, min(len(tasks), 3))
        priority_map = {"high": 0, "medium": 1, "low": 2}
        sorted_tasks = sorted(tasks, key=lambda x: priority_map.get(x.get('priority', 'low'), 3))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_worker_task, task, case, context, temperature, provider, model)
                for task in sorted_tasks
            ]
            
            progress_bar = st.progress(0)
            for i, future in enumerate(futures):
                progress = (i + 1) / len(sorted_tasks)
                progress_bar.progress(progress)
                
                try:
                    result = future.result(timeout=60)
                    worker_results.append(result)
                except Exception as e:
                    st.error(f"Error in worker task: {str(e)}")
                    worker_results.append(generate_error_response(sorted_tasks[i], str(e)))
        
        return worker_results
        
    except Exception as e:
        st.error(f"Error in task processing: {str(e)}")
        return worker_results

# Display Functions
def display_analysis_results(results: Dict):
    """
    Display the analysis results with proper formatting.
    
    Args:
        results (Dict): Analysis results to display
    """
    if "error" in results:
        st.error(results["error"])
        return

    st.markdown("## Orchestrator Analysis")
    st.write(results["analysis"])
    
    st.markdown("## Specialized Analyses")
    for result in results["worker_results"]:
        with st.expander(
            f"{result['type'].replace('_', ' ').title()}", 
            expanded=True
        ):
            st.markdown("### Focus")
            st.write(result["description"])
            
            if "priority" in result:
                st.markdown(f"**Priority**: {result['priority'].upper()}")
            
            st.markdown("### Findings & Recommendations")
            if "error" in result:
                st.error(result["error"])
            else:
                # Format each section with proper markdown
                lines = result["result"].split('\n')
                formatted_lines = []
                
                for line in lines:
                    if line.strip():
                        if line.strip()[0].isdigit():
                            # Section header
                            formatted_lines.append(f"\n#### {line.strip()}")
                        elif line.strip().startswith('-'):
                            # Bullet point
                            formatted_lines.append(f"{line.strip()}")
                        else:
                            formatted_lines.append(line)
                            
                st.markdown('\n'.join(formatted_lines))

def format_task_display(task: Dict) -> str:
    """
    Format a single task for display with proper styling.
    
    Args:
        task (Dict): Task information to format
        
    Returns:
        str: Formatted markdown string
    """
    return f"""
    #### Analysis Type: {task['type'].replace('_', ' ').title()}
    - **Priority**: {task['priority'].upper()}
    - **Focus**: {task['description']}
    - **Requirements**: {task['requirements']}
    """

def orchestrate_analysis(medical_case: str, context: Optional[Dict] = None) -> Dict:
    """
    Main orchestration function with dynamic task management.
    
    Args:
        medical_case (str): The medical case to analyze
        context (Optional[Dict]): Additional clinical context
        
    Returns:
        Dict: Complete analysis results
    """
    if st.session_state.processing:
        return {"error": "Processing already in progress"}
    
    st.session_state.processing = True
    context = context or {}
    
    try:
        with st.spinner("Analyzing case and determining necessary specialized analyses..."):
            orchestrator_result = get_orchestrator_analysis(medical_case, context)
            
            st.markdown("### Initial Case Analysis")
            st.write(orchestrator_result["analysis"])
            
            st.markdown("### Identified Specialized Analyses")
            for task in orchestrator_result["tasks"]:
                st.markdown(format_task_display(task))

        with st.spinner("Processing specialized analyses in parallel..."):
            worker_results = process_specialized_tasks(
                medical_case,
                orchestrator_result["tasks"],
                context
            )

        return {
            "analysis": orchestrator_result["analysis"],
            "worker_results": worker_results,
        }
    
    except Exception as e:
        st.error(f"Orchestration error: {str(e)}")
        return {"error": str(e)}
    
    finally:
        st.session_state.processing = False

def render_workflow_diagram():
    """Render the orchestrator workflow diagram"""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'orchestrator.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'orchestrator_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def render_usage_instruction():
    # Add instructions
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Input Case**
            - Choose sample case or enter your own
            - Provide comprehensive medical information
            - Set clinical context and urgency
        
        2. **Configuration**
            - Select appropriate clinical setting
            - Define urgency level
            - Enable analysis steps if needed
        
        3. **Review Results**
            - Check orchestrator analysis
            - Review specialized assessments
            - Download complete report
        
        ### Analysis Components
        
        - 🎯 Initial Case Assessment
        - 👥 Specialized Team Reviews
        - ⚠️ Risk Evaluations
        - 💡 Treatment Recommendations
        
        ### Best Practices
        
        - Provide detailed case information
        - Set appropriate urgency level
        - Review all specialized analyses
        - Save important results
        """)

def render_patient_info_form():
    """Render the medical case input form"""
    with st.form("case_info"):
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
        
        # Context configuration
        st.markdown("### 🏥 Clinical Context")
        col1, col2 = st.columns(2)
        
        with col1:
            clinical_setting = st.selectbox(
                "Clinical Setting",
                ["Primary Care", "Emergency", "Specialist", "Inpatient"],
                help="Select the clinical environment"
            )
        
        with col2:
            urgency_level = st.selectbox(
                "Urgency Level",
                ["Routine", "Urgent", "Emergency"],
                help="Select the case urgency"
            )
        
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
            
    return submitted, medical_case, clinical_setting, urgency_level, show_steps

def display_analysis_results(results: Dict):
    """Display the analysis results in a well-organized format"""
    st.success("✅ Analysis completed successfully!")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "📋 Orchestrator Analysis",
        "🔍 Specialized Analyses",
        "📊 Summary"
    ])
    
    with tab1:
        st.markdown("### 🎯 Initial Case Analysis")
        st.markdown("---")
        st.write(results["analysis"])
    
    with tab2:
        st.markdown("### 👥 Specialized Team Analyses")
        st.markdown("---")
        
        for result in results["worker_results"]:
            with st.expander(f"📍 {result['type'].replace('_', ' ').title()}", expanded=True):
                # Priority badge
                priority_colors = {
                    "HIGH": "red",
                    "MEDIUM": "orange",
                    "LOW": "green"
                }
                priority = result.get("priority", "").upper()
                st.markdown(
                    f'<div style="padding:4px 8px;border-radius:4px;'
                    f'background-color:{priority_colors.get(priority, "gray")};'
                    f'display:inline-block;color:white;margin-bottom:10px">'
                    f'Priority: {priority}</div>',
                    unsafe_allow_html=True
                )
                
                # Description
                st.markdown("#### 🔍 Focus Area")
                st.markdown(result["description"])
                
                # Findings & Recommendations
                st.markdown("#### 📋 Analysis Results")
                if "error" in result:
                    st.error(result["error"])
                else:
                    sections = result["result"].split('\n\n')
                    for section in sections:
                        if section.strip():
                            if section.strip().startswith('1.'):
                                st.markdown("##### 🔬 Key Clinical Findings")
                            elif section.strip().startswith('2.'):
                                st.markdown("##### ⚠️ Risk Assessment")
                            elif section.strip().startswith('3.'):
                                st.markdown("##### 💡 Recommendations")
                            elif section.strip().startswith('4.'):
                                st.markdown("##### 👥 Care Coordination")
                            
                            # Format bullet points
                            formatted_section = section.replace('- ', '\n• ')
                            st.markdown(formatted_section)
    
    with tab3:
        st.markdown("### 📊 Analysis Overview")
        st.markdown("---")
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Analyses",
                len(results["worker_results"])
            )
        with col2:
            high_priority = sum(1 for r in results["worker_results"] 
                              if r.get("priority", "").upper() == "HIGH")
            st.metric(
                "High Priority Items",
                high_priority
            )
        with col3:
            error_count = sum(1 for r in results["worker_results"] 
                            if "error" in r)
            st.metric(
                "Processing Issues",
                error_count,
                delta="-" if error_count == 0 else None,
                delta_color="normal" if error_count == 0 else "inverse"
            )
        
        # Add download button
        st.markdown("### 📥 Download Results")
        result_json = json.dumps(results, indent=2)
        st.download_button(
            label="Download Analysis Report",
            data=result_json,
            file_name="specialized_analysis_results.json",
            mime="application/json"
        )

def render_orchestrator_medical_analysis():
    """Render the specialized analysis interface"""
    st.subheader("Orchestrator - Specialized Medical Analysis")
    
    # Render workflow diagram
    render_workflow_diagram()
    # Render usage instruction
    render_usage_instruction()
    
    # Get input from form
    submitted, medical_case, clinical_setting, urgency_level, show_steps = render_patient_info_form()
    
    if submitted:
        if medical_case:
            try:
                # Initialize progress
                progress_text = "Analysis in progress. Please wait..."
                my_bar = st.progress(0, text=progress_text)
                
                # Run the analysis
                with st.spinner("Processing specialized analyses..."):
                    context = {
                        "clinical_setting": clinical_setting,
                        "urgency_level": urgency_level
                    }
                    
                    results = orchestrate_analysis(
                        medical_case=medical_case,
                        context=context
                    )
                    
                    my_bar.progress(100, text="Analysis complete!")
                    
                    if "error" not in results:
                        display_analysis_results(results)
                    else:
                        st.error("❌ Analysis failed.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
        else:
            st.warning("⚠️ Please enter a medical case to analyze")
            
        