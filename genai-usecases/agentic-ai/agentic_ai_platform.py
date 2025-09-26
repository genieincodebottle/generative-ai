"""
Agentic AI Platform
Centralized interface for all agentic AI implementations
- CrewAI with YAML configuration
- LangGraph with type-safe state management
- Advanced workflow patterns with features
- Multi-agent orchestration with cross framework integration
"""

import streamlit as st
import sys
import asyncio
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "agentic_frameworks" / "crewai" / "data_analysis_crew"))
sys.path.append(str(current_dir / "agentic_frameworks" / "crewai" / "content_creation_crew"))
sys.path.append(str(current_dir / "agentic_frameworks" / "crewai" / "research_assistant_crew"))
sys.path.append(str(current_dir / "agentic_frameworks" / "crewai" / "code_review_crew"))
sys.path.append(str(current_dir / "agentic_frameworks" / "langgraph"))
sys.path.append(str(current_dir / "agentic_workflows"))
sys.path.append(str(current_dir / "multi_agent_orchestration"))

# Import platform implementations
try:
    from data_analysis_crew import render_crew_interface
    from content_creation_crew import render_content_crew_interface
    from research_assistant_crew import render_research_crew_interface
    from code_review_crew import render_code_review_interface
    crewai_available = True
except ImportError as e:
    crewai_available = False
    crewai_error = str(e)

try:
    from customer_support_agent import render_customer_support_interface
    from document_processing_pipeline import render_document_processing_interface
    from task_planning_system import render_task_planning_interface
    langgraph_available = True
except ImportError as e:
    langgraph_available = False
    langgraph_error = str(e)

# Individual workflow pattern imports
try:
    from prompt_chaining import render_prompt_chaining_interface
    from query_routing import render_query_routing_interface
    from parallel_execution import render_parallel_execution_interface
    from event_driven import render_event_driven_interface
    from tool_orchestration import render_tool_orchestration_interface
    individual_workflows_available = True
except ImportError as e:
    individual_workflows_available = False
    individual_workflows_error = str(e)

try:
    from multi_agent_orchestration import render_orchestration_interface
    orchestration_available = True
except ImportError as e:
    orchestration_available = False
    orchestration_error = str(e)

def render_welcome_page():
    """Render the welcome page with feature overview"""
    st.title("Agentic AI Learning")

    st.markdown("""
    Learn Agentic AI through hands-on implementations with leading frameworks and advanced orchestration techniques
    """)

    # Platform status overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "✅" if crewai_available else "❌"
        st.metric("CrewAI", status, "Framework")

    with col2:
        status = "✅" if langgraph_available else "❌"
        st.metric("LangGraph", status, "Framework")

    with col3:
        status = "✅" if individual_workflows_available else "❌"
        st.metric("Advanced Workflows", status, "Patterns")

    with col4:
        status = "✅" if orchestration_available else "❌"
        st.metric("Orchestration", status, "Multi-Agent")

    # Feature overview
    with st.expander("🎯 Capabilities", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🤖 CrewAI Use Cases

            📊 **Data Analysis**: Statistical analysis and visualization

            🎨 **Content Creation**: Multi-format content generation

            🔬 **Research Assistant**: Comprehensive research workflows

            👨‍💻 **Code Review**: Automated code analysis and security

            ✅ **YAML Configuration**: Maintainable setup

            ✅ **Multi-Agent Collaboration**: Specialized agent teams
            """)

        with col2:
            st.markdown("""
            ### 🌐 LangGraph Use Cases

            📊 **Data Analysis**: Multi-agent statistical analysis

            🎧 **Customer Support**: Intelligent conversation flows

            📄 **Document Processing**: Multi-format document pipeline

            📋 **Task Planning**: Dynamic workflow planning

            ✅ **Type-Safe State**: Annotated types with reducers

            ✅ **Advanced Workflows**: Graph-based execution
            """)
   
    # Additional workflow patterns info
    with st.expander("🌟 Advanced Patterns", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🌊 Workflow Patterns

            🎯 **Query Routing**: Intelligent traffic control

            ⚡ **Parallel Execution**: Concurrent processing

            🔗 **Prompt Chaining**: Multi-step reasoning

            📡 **Event-Driven**: Real-time reactive systems

            🛠️ **Tool Orchestration**: API coordination

            ✅ **Multi-Provider Support**: Ollama, OpenAI, Anthropic, Gemini, Groq
            """)

        with col2:
            st.markdown("""
            ### 🎭 Multi-Agent Orchestration

            🏗️ **Hierarchical CrewAI**: Structured coordination

            🕸️ **Graph LangGraph**: State-driven workflows

            🌐 **Hybrid Integration**: Cross-framework collaboration

            📡 **Event-Driven**: Real-time coordination

            """)

    # Getting started
    st.markdown("### 🎮 Getting Started")

    st.code("""
    # Ensure you are in the following directory:
    cd genai-usecases\\agentic-ai
    # Then run one of the following commands:
    streamlit run agentic_ai_platform.py                                                # This platform

    # CrewAI Examples:
    streamlit run agentic_frameworks\\crewai\\data_analysis_crew\\data_analysis_crew.py      # CrewAI Data Analysis
    streamlit run agentic_frameworks\\crewai\\content_creation_crew\\content_creation_crew.py # CrewAI Content Creation
    streamlit run agentic_frameworks\\crewai\\research_assistant_crew\\research_assistant_crew.py # CrewAI Research
    streamlit run agentic_frameworks\\crewai\\code_review_crew\\code_review_crew.py          # CrewAI Code Review

    # LangGraph Examples:
    streamlit run agentic_frameworks\\langgraph\\customer_support_agent.py                 # LangGraph Customer Support
    streamlit run agentic_frameworks\\langgraph\\document_processing_pipeline.py           # LangGraph Document Processing
    streamlit run agentic_frameworks\\langgraph\\task_planning_system.py                   # LangGraph Task Planning

    # Workflow Patterns:
    streamlit run agentic_workflows\\query_routing.py                                       # Query Routing
    streamlit run agentic_workflows\\parallel_execution.py                                  # Parallel Execution
    streamlit run agentic_workflows\\prompt_chaining.py                                     # Prompt Chaining
    streamlit run agentic_workflows\\event_driven.py                                        # Event-Driven
    streamlit run agentic_workflows\\tool_orchestration.py                                  # Tool Orchestration

    # Multi-Agent Orchestration:
    streamlit run multi_agent_orchestration\\multi_agent_orchestration.py                  # Advanced Orchestration
    """, language="bash")

def render_error_page(component_name: str, error_message: str):
    """Render error page for unavailable components"""
    st.title(f"❌ {component_name}")

    st.error(f"**{component_name} is currently unavailable**")

    with st.expander("🔍 Error Details", expanded=False):
        st.code(error_message, language="python")

    st.markdown("### 🛠️ Troubleshooting")

    # Check for common issues
    if "No module named 'streamlit'" in error_message:
        st.error("**Missing Dependencies Detected!**")
        st.markdown("""
        **🔧 Quick Fix:**
        ```bash
        pip install -r requirements.txt
        ```
        """)

    elif "No module named" in error_message:
        module_name = error_message.split("'")[1] if "'" in error_message else "unknown"
        st.warning(f"**Missing Module: {module_name}**")
        st.markdown(f"""
        **🔧 Quick Fix:**
        ```bash
        pip install {module_name}
        ```
        """)

    st.markdown("""
    **Common Solutions:**
    1. **Install Dependencies**: Ensure all required packages are installed
       ```bash
       pip install -r requirements.txt
       ```

    2. **Check File Paths**: Verify all files are in the correct locations

    3. **Environment Variables**: Ensure your `.env` file has the required API keys
       ```bash
       cp .env.example .env
       # Edit .env with your API keys
       ```

    4. **Restart Streamlit**: Sometimes a restart helps
       ```bash
       streamlit run agentic_ai_platform.py
       ```

    5. **Individual Component Testing**: Test components individually
       ```bash
       # Test CrewAI components:
       streamlit run agentic_frameworks\\crewai\\data_analysis_crew\\data_analysis_crew.py

       # Test LangGraph components:
       streamlit run agentic_frameworks\\langgraph\\customer_support_agent.py

       # Test Workflow patterns:
       streamlit run agentic_workflows\\query_routing.py

       # Test Orchestration:
       streamlit run multi_agent_orchestration\\multi_agent_orchestration.py
       ```
    """)

    st.info("💡 **Tip**: Try running the individual component files directly to get more specific error messages.")

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Agentic AI Learning",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar navigation
    with st.sidebar:
        st.subheader("Multi-Agent AI")

        page = st.selectbox(
            "Choose Component",
            [
                "🏠 Platform Overview",
                "📊 CrewAI - Data Analysis",
                "✍️ CrewAI - Content Creation",
                "🔬 CrewAI - Research Assistant",
                "👨‍💻 CrewAI - Code Review",
                "🎧 LangGraph - Customer Support",
                "📄 LangGraph - Document Processing",
                "🗂️ LangGraph - Task Planning",
                "🔗 Workflow - Prompt Chaining",
                "🎯 Workflow - Query Routing",
                "⚡ Workflow - Parallel Execution",
                "📡 Workflow - Event-Driven",
                "🛠️ Workflow - Tool Orchestration",
                "🌟 Multi-Agent Orchestration"
            ],
            key='page_selection'
        )

        st.markdown("---")

    # Main content based on selection
    if page == "🏠 Platform Overview":
        render_welcome_page()

    elif page == "📊 CrewAI - Data Analysis":
        if crewai_available:
            render_crew_interface()
        else:
            render_error_page("CrewAI - Data Analysis", crewai_error)

    elif page == "✍️ CrewAI - Content Creation":
        if crewai_available:
            render_content_crew_interface()
        else:
            render_error_page("CrewAI - Content Creation", crewai_error)

    elif page == "🔬 CrewAI - Research Assistant":
        if crewai_available:
            render_research_crew_interface()
        else:
            render_error_page("CrewAI - Research Assistant", crewai_error)

    elif page == "👨‍💻 CrewAI - Code Review":
        if crewai_available:
            render_code_review_interface()
        else:
            render_error_page("CrewAI - Code Review", crewai_error)

    elif page == "🎧 LangGraph - Customer Support":
        if langgraph_available:
            render_customer_support_interface()
        else:
            render_error_page("LangGraph - Customer Support", langgraph_error)

    elif page == "📄 LangGraph - Document Processing":
        if langgraph_available:
            render_document_processing_interface()
        else:
            render_error_page("LangGraph - Document Processing", langgraph_error)

    elif page == "🗂️ LangGraph - Task Planning":
        if langgraph_available:
            render_task_planning_interface()
        else:
            render_error_page("LangGraph - Task Planning", langgraph_error)

    elif page == "🔗 Workflow - Prompt Chaining":
        if individual_workflows_available:
            render_prompt_chaining_interface()
        else:
            render_error_page("Workflow - Prompt Chaining", individual_workflows_error)

    elif page == "🎯 Workflow - Query Routing":
        if individual_workflows_available:
            render_query_routing_interface()
        else:
            render_error_page("Workflow - Query Routing", individual_workflows_error)

    elif page == "⚡ Workflow - Parallel Execution":
        if individual_workflows_available:
            render_parallel_execution_interface()
        else:
            render_error_page("Workflow - Parallel Execution", individual_workflows_error)

    elif page == "📡 Workflow - Event-Driven":
        if individual_workflows_available:
            render_event_driven_interface()
        else:
            render_error_page("Workflow - Event-Driven", individual_workflows_error)

    elif page == "🛠️ Workflow - Tool Orchestration":
        if individual_workflows_available:
            render_tool_orchestration_interface()
        else:
            render_error_page("Workflow - Tool Orchestration", individual_workflows_error)

    elif page == "🌟 Multi-Agent Orchestration":
        if orchestration_available:
            asyncio.run(render_orchestration_interface())
        else:
            render_error_page("Multi-Agent Orchestration", orchestration_error)

if __name__ == "__main__":
    main()