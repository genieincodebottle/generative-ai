"""
Medical Case Analysis Parallelization
This module implements parallel processing patterns for medical case analysis,
supporting both sectioning and voting approaches.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Union, Optional
from enum import Enum
import json
import streamlit as st

from utils.llm import llm_call
from utils.prompts import MEDICAL_REVIEW_PROMPTS

from PIL import Image
from pathlib import Path

HEALTHCARE_STAKEHOLDERS = [
    """Patient Care Team:
    - Primary care physician
    - Specialist consultants
    - Nursing staff
    - Allied health professionals""",

    """Patient and Family:
    - Patient's immediate needs
    - Family caregiver requirements
    - Home care considerations
    - Financial implications""",

    """Healthcare Facility:
    - Resource allocation
    - Staff scheduling
    - Equipment needs
    - Regulatory compliance""",

    """Support Services:
    - Laboratory
    - Radiology
    - Pharmacy
    - Physical therapy"""
]

class ParallelizationType(Enum):
    """Enum for different parallelization types."""
    SECTIONING = "sectioning"  # For stakeholder analysis
    VOTING = "voting"  # For content review

class ParallelProcessor:
    """
    Handles parallel processing with support for both sectioning and voting approaches
    using ThreadPoolExecutor for efficient task distribution.
    """
    def __init__(self, max_workers: int = 3, timeout: int = 60):
        """
        Initialize the ParallelProcessor.
        
        Args:
            max_workers (int): Maximum number of concurrent workers
            timeout (int): Maximum processing time for each task in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.results_cache: Dict[str, Any] = {}
    
    def _process_task(self, task_type: ParallelizationType, task_data: tuple) -> tuple[int, str]:
        """
        Process a single task based on its type with caching support.
        
        Args:
            task_type (ParallelizationType): Type of parallelization
            task_data (tuple): Task details including index, prompt, input text, and temperature
        
        Returns:
            tuple[int, str]: Task index and processing result
        """
        idx, prompt, input_text, temperature, provider, model = task_data
        try:
            cache_key = f"{task_type.value}_{prompt}_{input_text}_{temperature}_{provider}_{model}_{idx}"
            
            if cache_key in self.results_cache:
                return idx, self.results_cache[cache_key]
            
            result = llm_call(f"{prompt}\nInput: {input_text}", temperature, provider, model)
            self.results_cache[cache_key] = result
            return idx, result
        except Exception as e:
            error_msg = f"Error processing {task_type.value} task {idx}: {str(e)}"
            return idx, error_msg

    def _aggregate_review_results(self, results: List[str], flag_threshold: int = 2) -> Dict[str, Any]:
        """
        Aggregate and analyze results from different review aspects.
        
        Args:
            results (List[str]): List of review results
            flag_threshold (int): Number of flags needed to reject content
        
        Returns:
            Dict[str, Any]: Aggregated review results with flags and status
        """
        flags = {
            "safety": [], "accuracy": [], "compliance": [], 
            "risk": [], "ethics": []
        }
        
        for result in results:
            try:
                lines = result.strip().split('\n')
                result_line = lines[0].strip().upper()
                explanation = ' '.join(line.strip() for line in lines[1:] if line.strip())
                
                if not explanation:
                    continue
                    
                flag_mapping = {
                    'FLAG_SAFETY': 'safety',
                    'FLAG_ACCURACY': 'accuracy',
                    'FLAG_COMPLIANCE': 'compliance',
                    'FLAG_RISK': 'risk',
                    'FLAG_ETHICS': 'ethics'
                }
                
                for flag_key, category in flag_mapping.items():
                    if flag_key in result_line:
                        flags[category].append(explanation)
                        break
            except Exception as e:
                st.error(f"Error processing result: {str(e)}")
                continue
        
        flags = {k: v for k, v in flags.items() if v}
        total_flags = len(flags)
        
        return {
            'flags': flags,
            'total_flags': total_flags,
            'aspects_reviewed': len(results),
            'aspects_flagged': total_flags,
            'overall_status': ('REJECTED' if total_flags >= flag_threshold else 
                             'WARNING' if total_flags >= 1 else 
                             'APPROVED')
        }

    def run_parallel(
        self,
        parallel_type: ParallelizationType,
        prompt: str,
        inputs: List[str],
        temperature: float,
        provider: str,
        model: str,
        flag_threshold: int = 2
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Run parallel processing based on specified type with progress tracking.
        
        Args:
            parallel_type (ParallelizationType): Type of parallelization
            prompt (str): Processing prompt
            inputs (List[str]): List of input texts
            temperature (float): LLM temperature setting
            provider (str): Selected LLM Provider
            model (str): Selected LLM Model
            flag_threshold (int): Threshold for flagging in voting type
        
        Returns:
            Union[List[str], Dict[str, Any]]: Processed results or aggregated review
        """
        if st.session_state.get('processing', False):
            return (["Processing already in progress"] if parallel_type == ParallelizationType.SECTIONING 
                    else {"error": "Processing already in progress"})

        progress_text = st.empty()
        progress_bar = st.progress(0)
        error_placeholder = st.empty()

        try:
            st.session_state.processing = True

            if parallel_type == ParallelizationType.SECTIONING:
                input_tuples = [
                    (i, prompt, input_text, temperature, provider, model) 
                    for i, input_text in enumerate(inputs)
                ]
            else:
                input_tuples = [
                    (i, MEDICAL_REVIEW_PROMPTS[i], inputs[0], temperature, provider, model)
                    for i in range(len(MEDICAL_REVIEW_PROMPTS))
                ]

            results = []
            completed = 0
            total_tasks = len(input_tuples)

            with ThreadPoolExecutor(max_workers=min(self.max_workers, total_tasks)) as executor:
                future_to_input = {
                    executor.submit(self._process_task, parallel_type, args): args
                    for args in input_tuples
                }

                for future in as_completed(future_to_input):
                    try:
                        idx, result = future.result(timeout=self.timeout)
                        results.append((idx, result))
                        completed += 1
                        
                        progress = completed / total_tasks
                        progress_text.text(f"Processing: {completed}/{total_tasks}")
                        progress_bar.progress(progress)
                        
                    except Exception as e:
                        error_placeholder.error(f"Error processing task: {str(e)}")
                        if parallel_type == ParallelizationType.SECTIONING:
                            results.append((len(results), f"Error: {str(e)}"))
                        else:
                            results.append((len(results), "FLAG_SAFETY Error in processing"))

            sorted_results = [r[1] for r in sorted(results, key=lambda x: x[0])]

            return (self._aggregate_review_results(sorted_results, flag_threshold) 
                    if parallel_type == ParallelizationType.VOTING 
                    else sorted_results)

        except Exception as e:
            error_placeholder.error(f"Error in parallel processing: {str(e)}")
            return ([f"Error: {str(e)}"] * len(inputs) 
                    if parallel_type == ParallelizationType.SECTIONING 
                    else {"error": str(e)})
        finally:
            st.session_state.processing = False
            progress_text.text("Processing complete!")
            progress_bar.progress(1.0)

# UI Functions
def render_workflow_diagram():
    """Render the parallelization workflow diagram with mermaid."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent / 'images'
                
        parallel_diagram = Image.open(image_path/ 'parallelization.png')
        st.image(parallel_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'parallelization_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def render_usage_instructions():
    """Render comprehensive usage instructions for the parallelization analysis."""
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Select Analysis Type**
           - Sectioning: Analyze impact on different stakeholder groups
           - Voting: Review medical content for multiple aspects
        
        2. **Configure Analysis**
           - Set number of parallel workers
           - Define processing timeout
           - Configure type-specific settings
        
        3. **Review Results**
           - Check detailed analysis
           - Review summary view
           - Download results if needed
        
        ### Analysis Types
        
        **🔄 Sectioning Analysis**
        - Multiple stakeholder perspectives
        - Parallel analysis of different groups
        - Comprehensive impact assessment
        
        **🔍 Voting Analysis**
        - Multi-aspect content review
        - Safety and compliance checks
        - Aggregated decision making
        
        ### Best Practices
        
        - Provide clear, detailed input
        - Configure appropriate timeouts
        - Review all analysis aspects
        - Save important results
        """)

def render_patient_info_form(parallel_type: str) -> tuple:
    """
    Render the configuration form for parallelization analysis.
    
    Args:
        parallel_type (str): Type of parallelization (Sectioning or Voting)
    
    Returns:
        tuple: Form submission details including inputs and configuration
    """
    with st.form("config_form"):
        st.markdown("### ⚙️ Analysis Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            max_workers = st.slider(
                "Max Parallel Workers", 
                1, 10, 3, 
                help="Number of concurrent analysis processes"
            )
        with col2:
            timeout = st.slider(
                "Processing Timeout", 
                30, 180, 60,
                help="Maximum time (in seconds) for each analysis"
            )
        
        flag_threshold: Optional[int] = None
        if parallel_type == "Sectioning (Multiple Stakeholders)":
            use_sample = st.checkbox(
                "Use sample stakeholders", 
                value=True,
                help="Use predefined stakeholder groups"
            )
            
            if use_sample:
                inputs = HEALTHCARE_STAKEHOLDERS
                for i, input_text in enumerate(inputs):
                    st.text_area(
                        f"📍 Stakeholder Group {i+1}",
                        input_text,
                        height=100,
                        disabled=True
                    )
            else:
                num_inputs = st.number_input(
                    "Number of stakeholder groups",
                    min_value=1,
                    max_value=10,
                    value=4
                )
                inputs = []
                for i in range(num_inputs):
                    input_text = st.text_area(
                        f"📍 Stakeholder Group {i+1}",
                        f"Stakeholder group {i+1} considerations..."
                    )
                    inputs.append(input_text)
            
            prompt = st.text_area(
                "🔍 Analysis Prompt",
                """Analyze the impact of proposed care plan changes on this stakeholder group.
                Consider resource requirements, workflow adjustments, and potential challenges.
                Provide specific recommendations for successful implementation."""
            )
        else:
            inputs = [st.text_area(
                "📄 Medical Content to Review",
                """Example Medical Protocol:
                For patients presenting with acute respiratory symptoms:
                1. Initial assessment within 10 minutes of arrival
                2. Pulse oximetry monitoring every 15 minutes
                3. If SpO2 < 92%, initiate supplemental oxygen
                4. Consider nebulizer treatment for wheezing
                5. Reassess after 30 minutes
                6. If no improvement, escalate to emergency department
                
                Note: Protocol can be implemented by nursing staff without physician presence.""",
                height=200
            )]
            
            flag_threshold = st.slider(
                "Flag Threshold", 
                1, 5, 2,
                help="Number of flags needed to reject the content"
            )
            
            st.info("""
            ### 🔍 Review Categories:
            - 🚨 Patient Safety
            - 📋 Clinical Accuracy
            - ⚖️ Regulatory Compliance
            - ⚠️ Implementation Risks
            - 🤝 Ethical Considerations
            """)
            
            prompt = ""
        
        submitted = st.form_submit_button(
            "Start Analysis",
            use_container_width=True
        )
        
        return submitted, inputs, prompt, max_workers, timeout, flag_threshold

def display_sectioning_results(results: List[str]):
    """
    Display results from sectioning analysis with tabs and expandable sections.
    
    Args:
        results (List[str]): List of results from stakeholder analysis
    """
    st.success("✅ Analysis completed successfully!")
    
    tab1, tab2 = st.tabs([
        "📊 Detailed Analysis",
        "📑 Summary View"
    ])
    
    with tab1:
        for i, result in enumerate(results, 1):
            with st.expander(f"📍 Stakeholder Group {i}", expanded=True):
                st.markdown(result)
    
    with tab2:
        st.markdown("### 📊 Analysis Overview")
        st.markdown(f"**Total Stakeholder Groups Analyzed:** {len(results)}")
        
        st.download_button(
            "📥 Download Complete Analysis",
            "\n\n---\n\n".join(results),
            file_name="stakeholder_analysis.txt",
            mime="text/plain"
        )

def display_voting_results(results: Dict[str, Any]):
    """
    Display results from voting analysis with status indicators and metrics.
    
    Args:
        results (Dict[str, Any]): Aggregated review results dictionary
    """
    if 'error' not in results:
        st.success("✅ Review completed successfully!")
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Overall Status",
            "🔍 Detailed Findings",
            "📑 Summary"
        ])
        
        with tab1:
            status_color = {
                'APPROVED': 'green',
                'WARNING': 'orange',
                'REJECTED': 'red'
            }.get(results['overall_status'], 'white')
            
            st.markdown(f"""
            <div style='padding:20px;border-radius:10px;background-color:{status_color};
            color:white;text-align:center;font-size:24px;font-weight:bold'>
            {results['overall_status']}
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Aspects Reviewed", results['aspects_reviewed'])
            col2.metric("Aspects Flagged", results['aspects_flagged'])
            col3.metric(
                "Pass Rate",
                f"{((results['aspects_reviewed'] - results['aspects_flagged']) / results['aspects_reviewed'] * 100):.1f}%"
            )
        
        with tab2:
            if results['flags']:
                st.markdown("### ⚠️ Areas of Concern")
                for category, flags in results['flags'].items():
                    with st.expander(f"🔍 {category.title()}", expanded=True):
                        for flag in flags:
                            st.warning(flag)
            
            all_categories = {'safety', 'accuracy', 'compliance', 'risk', 'ethics'}
            passing_categories = all_categories - set(results['flags'].keys())
            if passing_categories:
                st.markdown("### ✅ Passing Categories")
                for category in sorted(passing_categories):
                    st.success(f"**{category.title()}:** No concerns identified")
        
        with tab3:
            st.markdown("### 📑 Review Summary")
            st.json(results)
            
            st.download_button(
                "📥 Download Complete Review",
                json.dumps(results, indent=2),
                file_name="content_review.json",
                mime="application/json"
            )
    else:
        st.error(f"❌ Error during review: {results['error']}")

def render_parallelization_medical_analysis():
    """
    Render the main parallelization medical analysis interface with all components.
    """
    st.subheader("Parallelization - Medical Analysis")
    
    render_workflow_diagram()
    render_usage_instructions()
    
    parallel_type = st.radio(
        "Select Analysis Type",
        ["Sectioning (Multiple Stakeholders)", "Voting (Content Review)"],
        help="Choose the type of parallel analysis to perform"
    )
    
    if 'parallel_processor' not in st.session_state:
        st.session_state.parallel_processor = ParallelProcessor()
    
    submitted, inputs, prompt, max_workers, timeout, flag_threshold = render_patient_info_form(parallel_type)
    
    if submitted:
        if all(inputs) and (prompt or parallel_type == "Voting (Content Review)"):
            try:
                st.session_state.parallel_processor.max_workers = max_workers
                st.session_state.parallel_processor.timeout = timeout
                
                progress_text = "Analysis in progress. Please wait..."
                my_bar = st.progress(0, text=progress_text)
                
                with st.spinner("Processing parallel analysis..."):
                    parallel_type_enum = (
                        ParallelizationType.SECTIONING 
                        if parallel_type == "Sectioning (Multiple Stakeholders)"
                        else ParallelizationType.VOTING
                    )
                    
                    results = st.session_state.parallel_processor.run_parallel(
                        parallel_type=parallel_type_enum,
                        prompt=prompt,
                        inputs=inputs,
                        temperature=st.session_state.temperature,
                        provider=st.session_state.selected_llm_provider,
                        model=st.session_state.selected_llm_model,
                        flag_threshold=flag_threshold if flag_threshold else 2
                    )
                    
                    my_bar.progress(100, text="Analysis complete!")
                    
                    if parallel_type == "Sectioning (Multiple Stakeholders)":
                        display_sectioning_results(results)
                    else:
                        display_voting_results(results)
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
        else:
            st.warning("⚠️ Please fill in all required information")