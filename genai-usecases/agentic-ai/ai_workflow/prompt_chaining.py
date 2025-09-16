"""
Business Document Analysis Using Prompt Chaining
This module implements a prompt chaining pattern for comprehensive business document analysis,
processing documents through sequential specialized prompts for insights and recommendations.
"""
from typing import Dict, List, Tuple
import streamlit as st
from utils.llm import llm_call
from utils.prompts import BUSINESS_DOCUMENT_ANALYSIS_STEPS

from PIL import Image
from pathlib import Path

SAMPLE_BUSINESS_DOCUMENT = """
    QUARTERLY BUSINESS REVIEW - Q3 2024
    
    Executive Summary:
    TechFlow Solutions experienced mixed performance in Q3 2024. Revenue increased 12% YoY to $4.2M, 
    driven primarily by our enterprise software division. However, customer acquisition costs rose 
    18% due to increased competition in the market.
    
    Key Metrics:
    - Total Revenue: $4.2M (12% YoY growth)
    - Gross Margin: 68% (down from 72% in Q2)
    - Customer Acquisition Cost: $1,850 (up 18% from Q2)
    - Customer Lifetime Value: $12,400
    - Monthly Recurring Revenue: $950K
    - Churn Rate: 3.2% (industry average: 5.1%)
    
    Department Performance:
    Sales: Exceeded targets by 8%, closed 3 enterprise deals worth $800K total
    Engineering: Delivered 2 major product features, reduced technical debt by 25%
    Marketing: Generated 340 qualified leads, improved conversion rate to 4.2%
    Customer Success: Maintained 97% customer satisfaction, reduced support tickets by 15%
    
    Challenges:
    - Increased competition leading to pricing pressure
    - Talent acquisition difficulties in engineering roles
    - Supply chain disruptions affecting hardware delivery
    
    Opportunities:
    - Expansion into European markets showing 25% demand growth
    - AI integration features requested by 60% of enterprise clients
    - Partnership opportunities with complementary service providers
    
    Strategic Initiatives for Q4:
    - Launch AI-powered analytics module
    - Implement customer referral program
    - Expand sales team by 3 additional reps
    - Optimize operational efficiency to improve margins
    """

def validate_document(document: str, validation_criteria: Dict):
    """
    Validates if the business document has required components.
    
    Args:
        document (str): Business document text
        validation_criteria (Dict): Criteria for validating document components
        
    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: Lists of present and missing components
    """
    document_lower = document.lower()
    present_components = []
    missing_components = []
    
    for key, criteria in validation_criteria.items():
        if any(keyword in document_lower for keyword in criteria["keywords"]):
            present_components.append((criteria["icon"], criteria["message"]))
        else:
            missing_components.append((criteria["icon"], criteria["message"]))
            
    return present_components, missing_components

def get_validation_criteria():
    """
    Returns the validation criteria for business document components.
    
    Returns:
        Dict: Validation criteria for different document components
    """
    return {
        "financial_metrics": {
            "keywords": ["revenue", "profit", "margin", "cost", "budget", "financial", "earnings", "sales"],
            "message": "financial metrics and performance",
            "icon": "💰"
        },
        "kpis": {
            "keywords": [
                "kpi", "metrics", "performance", "targets", "goals", 
                "conversion", "growth", "roi", "customer acquisition", "retention"
            ],
            "message": "key performance indicators",
            "icon": "📈"
        },
        "strategic_insights": {
            "keywords": ["strategy", "strategic", "opportunities", "challenges", "market", "competitive"],
            "message": "strategic insights and market analysis",
            "icon": "🎯"
        },
        "operational_data": {
            "keywords": [
                "operations", "efficiency", "processes", "workflow", 
                "productivity", "capacity", "utilization", "performance"
            ],
            "message": "operational performance data",
            "icon": "⚙️"
        },
        "recommendations": {
            "keywords": [
                "recommendations", "action items", "next steps", "initiatives", 
                "plans", "priorities", "objectives", "goals"
            ],
            "message": "actionable recommendations",
            "icon": "💡"
        }
    }

def process_prompt_chain(input_text: str, prompts: List[str], show_steps: bool = True) -> str:
    """
    Process business document through sequential prompt chain.
    
    Args:
        input_text (str): Initial business document text
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
def format_analysis_result(result: str) -> str:
    """
    Format the analysis result with proper markdown structure.
    
    Args:
        result (str): Raw analysis result string
        
    Returns:
        str: Formatted markdown string
    """
    formatted_result = "# 📊 Business Document Analysis\n\n"
    section_icons = {
        "executive summary": "📋",
        "key findings": "🔍", 
        "performance analysis": "📈",
        "strategic recommendations": "🎯",
        "risk assessment": "⚠️",
        "action items": "✅",
        "next steps": "➡️"
    }
    
    for section in result.split('**'):
        section = section.strip()
        if not section or section.lower() in ["business document analysis", "business analysis report"]:
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
    
    tab1, tab2 = st.tabs(["📊 Document Analysis", "⚙️ Processing Steps"])
    
    with tab1:
        formatted_result = format_analysis_result(result)
        st.markdown(formatted_result)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download as Markdown",
                data=formatted_result,
                file_name="business_document_analysis.md",
                mime="text/markdown",
                key='markdown_download'
            )
        with col2:
            st.download_button(
                label="📥 Download as Text",
                data=result,
                file_name="business_document_analysis.txt",
                mime="text/plain",
                key='text_download'
            )
    
    with tab2:
        st.markdown("### Analysis Pipeline")
        for i, step in enumerate(BUSINESS_DOCUMENT_ANALYSIS_STEPS, 1):
            with st.expander(f"Step {i}: {step.split('/n')[0]}", expanded=False):
                st.markdown(step)

def render_workflow_diagram() -> None:
    """Render the business document analysis workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent / 'images'
        
        try:
            prompt_chain_diagram = Image.open(image_path/ 'prompt_chaining.png')
            st.image(prompt_chain_diagram, caption='High Level Architecture')
            
            sequence_diagram = Image.open(image_path/ 'prompt_chain_sequence_diagram.png')
            st.image(sequence_diagram, caption='Sequence Diagram')
        except FileNotFoundError:
            st.info("📊 Workflow diagrams would be displayed here when available.")

def render_usage_instruction() -> None:
    """Render usage instructions for the business document analysis system."""
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Input Document**
            - Use sample business document or enter your own
            - Supports: quarterly reports, business plans, market analysis, financial documents
            - Ensure key business components are present
        
        2. **Configuration**
            - Enable/disable analysis steps display
            - Review document validation before processing
        
        3. **Review Results**
            - Check document validation
            - Review comprehensive analysis
            - Download formatted reports
        
        ### Expected Document Components
        
        - 💰 Financial Metrics & Performance
        - 📈 Key Performance Indicators 
        - 🎯 Strategic Insights & Market Analysis
        - ⚙️ Operational Performance Data
        - 💡 Actionable Recommendations
        
        ### Supported Document Types
        
        - Quarterly/Annual Business Reviews
        - Strategic Planning Documents
        - Market Analysis Reports  
        - Financial Performance Reports
        - Operational Assessment Documents
        """)

def render_document_input_form() -> Tuple[bool, str, bool]:
    """
    Render the business document input form.
    
    Returns:
        Tuple[bool, str, bool]: Tuple containing (submitted state, input text, show steps flag)
    """
    with st.form("document_info"):
        use_sample = st.checkbox("Use sample business document", value=True)
        input_text = SAMPLE_BUSINESS_DOCUMENT if use_sample else st.text_area(
            "Business Document",
            """Executive Summary:
            [Brief overview of document purpose and key findings]
            
            Key Metrics:
            [Financial performance, KPIs, and measurements]
            
            Performance Analysis:
            [Detailed performance breakdown by department/area]
            
            Challenges & Opportunities:
            [Current challenges and growth opportunities]
            
            Strategic Recommendations:
            [Actionable recommendations and next steps]""",
            height=200,
            help="Enter your business document content here"
        )
        
        if use_sample:
            st.text_area(
                "Sample Document (read-only)", 
                input_text, 
                height=200, 
                disabled=True,
                help="This is a sample quarterly business review for demonstration"
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
                "🚀 Start Analysis",
                use_container_width=True
            )
            
    return submitted, input_text, show_steps

def render_prompt_chain_business_analysis() -> None:
    """Main function to render the business document analysis using Prompt Chaining."""
    st.subheader("Prompt Chaining - Business Document Analysis")
    
    # Render workflow diagram and instructions
    render_workflow_diagram()
    render_usage_instruction()
    
    # Get input from form
    submitted, input_text, show_steps = render_document_input_form()
    
    if submitted:
        if not input_text:
            st.warning("⚠️ Please enter a business document to analyze")
            return
        
        try:
            validation_criteria = get_validation_criteria()
            present, missing = validate_document(input_text, validation_criteria)
            display_validation_results(present, missing)
            
            # Proceed with analysis even if some components are missing
            with st.spinner("🔍 Analyzing business document..."):
                result = process_prompt_chain(input_text, BUSINESS_DOCUMENT_ANALYSIS_STEPS, show_steps)
                
                if not result.startswith("Error"):
                    display_analysis_results(result)
                else:
                    st.error("❌ Analysis failed")
                    
            if missing:
                st.info("💡 **Tip:** Including the missing components above will provide more comprehensive analysis results.")
                    
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)