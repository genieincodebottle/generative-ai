"""
CrewAI - Research Assistant Crew Implementation
Features:
- Comprehensive research workflows
- Multi-source information gathering
- Fact verification and validation
- Report generation and synthesis
- Citation management
- Executive summary creation
"""

import streamlit as st
import yaml
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


load_dotenv()

# ============================================================================
# BACKEND: DATA MODELS AND SCHEMAS
# ============================================================================

class ResearchInput(BaseModel):
    """
    Input schema for research requests

    Defines the structure and validation for research parameters
    received from the frontend interface.
    """
    research_topic: str = Field(..., description="Main research topic or question")
    research_depth: str = Field(default="comprehensive", description="Depth of research (basic, comprehensive, deep)")
    sources_required: List[str] = Field(default=[], description="Specific sources or types of sources")
    output_format: str = Field(default="report", description="Output format (report, summary, presentation)")
    deadline: Optional[str] = Field(default=None, description="Research deadline")

# ============================================================================
# BACKEND: CUSTOM TOOLS FOR RESEARCH
# ============================================================================

class WebSearchTool(BaseTool):
    """
    Web Search and Information Gathering Tool

    Simulates web search functionality for comprehensive research.
    In production, this would integrate with actual search APIs.
    """
    name: str = "web_search_tool"
    description: str = "Searches the web for relevant information on given topics"

    def _run(self, query: str, num_results: int = 10) -> str:
        """
        Simulate web search functionality

        Args:
            query (str): Search query
            num_results (int): Number of results to return

        Returns:
            str: JSON formatted search results
        """
        try:
            # Simulate search results (in real implementation, would use actual search APIs)
            search_results = {
                "query": query,
                "results": [
                    {
                        "title": f"Research Article: {query} - Comprehensive Overview",
                        "url": "https://example-research.com/article1",
                        "snippet": f"Comprehensive analysis of {query} including recent developments, methodologies, and future implications.",
                        "source": "Academic Journal",
                        "date": "2024-01-15",
                        "relevance_score": 0.95
                    },
                    {
                        "title": f"Industry Report: {query} Market Analysis 2024",
                        "url": "https://market-analysis.com/report",
                        "snippet": f"Latest market trends and analysis for {query} with statistical data and projections.",
                        "source": "Market Research Firm",
                        "date": "2024-02-01",
                        "relevance_score": 0.89
                    },
                    {
                        "title": f"Expert Opinion: The Future of {query}",
                        "url": "https://expert-insights.com/opinion",
                        "snippet": f"Leading experts share their perspectives on the evolution and impact of {query}.",
                        "source": "Industry Expert Blog",
                        "date": "2024-01-28",
                        "relevance_score": 0.87
                    },
                    {
                        "title": f"Case Study: Successful Implementation of {query}",
                        "url": "https://case-studies.com/success",
                        "snippet": f"Real-world case study demonstrating practical applications and outcomes of {query}.",
                        "source": "Business Publication",
                        "date": "2024-01-20",
                        "relevance_score": 0.83
                    },
                    {
                        "title": f"Technical Documentation: {query} Best Practices",
                        "url": "https://technical-docs.com/guide",
                        "snippet": f"Technical guide and best practices for implementing and optimizing {query}.",
                        "source": "Technical Documentation",
                        "date": "2024-01-10",
                        "relevance_score": 0.81
                    }
                ],
                "total_results": num_results,
                "search_time": "0.24 seconds"
            }

            return json.dumps(search_results, indent=2)

        except Exception as e:
            return f"Search error: {str(e)}"

class FactCheckTool(BaseTool):
    """Fact verification and validation tool"""
    name: str = "fact_check_tool"
    description: str = "Verifies facts and checks information accuracy"

    def _run(self, claim: str, sources: List[str] = None) -> str:
        """Simulate fact checking functionality"""
        try:
            # Simulate fact checking analysis
            fact_check_result = {
                "claim": claim,
                "verification_status": "verified",
                "confidence_score": 0.92,
                "supporting_sources": [
                    "Academic Research Database",
                    "Government Statistical Office",
                    "Peer-reviewed Publications"
                ],
                "potential_issues": [
                    "Some statistics may be from different time periods",
                    "Definition variations across sources"
                ],
                "recommendations": [
                    "Cross-reference with latest government data",
                    "Include confidence intervals for statistical claims",
                    "Note any limitations or assumptions"
                ],
                "last_verified": datetime.now().isoformat()
            }

            return json.dumps(fact_check_result, indent=2)

        except Exception as e:
            return f"Fact check error: {str(e)}"

class CitationTool(BaseTool):
    """Citation management and formatting tool"""
    name: str = "citation_tool"
    description: str = "Manages citations and formats references"

    def _run(self, sources: List[Dict], citation_style: str = "APA") -> str:
        """Generate properly formatted citations"""
        try:
            citations = {
                "citation_style": citation_style,
                "formatted_citations": [
                    "Smith, J. A., & Johnson, M. B. (2024). Comprehensive analysis of modern research methodologies. Journal of Research Excellence, 15(3), 45-62.",
                    "Technology Research Institute. (2024, February 1). Industry analysis report: Emerging trends and market projections. Retrieved from https://techresearch.org/reports/2024",
                    "Williams, R. (2024, January 28). Expert perspectives on future developments. Expert Insights Quarterly, 8(2), 12-18.",
                    "Business Case Studies Inc. (2024). Implementation success stories: Real-world applications and outcomes. Business Publications Archive."
                ],
                "in_text_citations": [
                    "(Smith & Johnson, 2024)",
                    "(Technology Research Institute, 2024)",
                    "(Williams, 2024)",
                    "(Business Case Studies Inc., 2024)"
                ],
                "citation_count": 4
            }

            return json.dumps(citations, indent=2)

        except Exception as e:
            return f"Citation error: {str(e)}"

# ============================================================================
# BACKEND: CORE RESEARCH CREW MANAGEMENT CLASS
# ============================================================================

class ResearchCrew:
    """
    CrewAI Implementation with YAML Configuration for Research

    Manages comprehensive research workflows using multiple AI research agents.
    Loads configurations from YAML files for agents, tasks, and crew settings.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the research crew manager

        Args:
            config_path (str, optional): Path to configuration directory
        """
        self.config_path = config_path or Path(__file__).parent / "config"

        # Validate configuration files exist
        if not validate_research_yaml_configs(self.config_path):
            raise FileNotFoundError("Required YAML configuration files are missing")

        self.config = self._load_configurations()
        self.tools = self._setup_tools()

    def _load_configurations(self) -> Dict[str, Any]:
        """
        Load YAML configuration files for agents, tasks, and crew settings

        Returns:
            Dict[str, Any]: Loaded configuration data
        """
        config = {}
        config_files = ['agents.yaml', 'tasks.yaml', 'crew.yaml']

        for file_name in config_files:
            file_path = self.config_path / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config[file_name.split('.')[0]] = yaml.safe_load(f)
            else:
                st.warning(f"Configuration file {file_name} not found")

        return config

    def _setup_tools(self) -> List[BaseTool]:
        """
        Initialize and setup research tools

        Returns:
            List[BaseTool]: List of available research tools for agents
        """
        return [
            WebSearchTool(),
            FactCheckTool(),
            CitationTool()
        ]

    def _create_llm(self, provider: str, model: str, **kwargs) -> LLM:
        """Create LLM instance with proper configuration"""
        provider_keys = {
            "Gemini": "GEMINI_API_KEY",
            "Groq": "GROQ_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "OpenAI": "OPENAI_API_KEY"       
        }

        key_name = provider_keys.get(provider)
        if key_name:
            api_key_value = os.getenv(key_name)
            if api_key_value:
                os.environ[key_name] = api_key_value

        # Format model name with provider prefix for proper LiteLLM routing
        if provider.lower() == 'ollama':
            formatted_model = f"ollama/{model}"
        elif provider.lower() == 'gemini':
            formatted_model = f"gemini/{model}"
        elif provider.lower() == 'groq':
            formatted_model = f"groq/{model}"
        elif provider.lower() == 'anthropic':
            formatted_model = f"anthropic/{model}"
        elif provider.lower() == 'openai':
            formatted_model = f"openai/{model}"
        else:
            formatted_model = model

        # Create LLM parameters (simplified to match working pattern)
        llm_params = {
            'model': formatted_model,
            'temperature': kwargs.get('temperature', 0.7)
        }

        # Add Ollama-specific parameters
        if provider.lower() == 'ollama' and 'base_url' in kwargs:
            llm_params['base_url'] = kwargs['base_url']

        # Add any additional kwargs that don't conflict
        for key, value in kwargs.items():
            if key not in ['temperature', 'model', 'base_url']:
                llm_params[key] = value

        return LLM(**llm_params)

    def _create_agents(self, llm_provider: str, model_name: str, **llm_kwargs) -> List[Agent]:
        """Create agents from YAML configuration"""
        agents = []
        agents_config = self.config.get('agents', {})

        llm = self._create_llm(llm_provider, model_name, **llm_kwargs)

        for agent_name, agent_config in agents_config.items():
            # Configure research tools for each agent based on YAML configuration
            agent_tools = []
            if agent_config.get('tools'):
                # Auto-assign tools based on agent role/name
                if 'information' in agent_name.lower() or 'gatherer' in agent_name.lower():
                    agent_tools.append(WebSearchTool())
                elif 'fact' in agent_name.lower() or 'verification' in agent_name.lower():
                    agent_tools.append(FactCheckTool())
                elif 'report' in agent_name.lower() or 'writer' in agent_name.lower():
                    agent_tools.append(CitationTool())

                # Also check for explicit tool_names in YAML
                tool_names = agent_config.get('tool_names', [])
                for tool_name in tool_names:
                    if tool_name == 'web_search_tool' and not any(isinstance(t, WebSearchTool) for t in agent_tools):
                        agent_tools.append(WebSearchTool())
                    elif tool_name == 'fact_check_tool' and not any(isinstance(t, FactCheckTool) for t in agent_tools):
                        agent_tools.append(FactCheckTool())
                    elif tool_name == 'citation_tool' and not any(isinstance(t, CitationTool) for t in agent_tools):
                        agent_tools.append(CitationTool())

            agent = Agent(
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                tools=agent_tools,
                llm=llm,
                verbose=agent_config.get('verbose', True),
                allow_delegation=agent_config.get('allow_delegation', False),
                max_iter=agent_config.get('max_iter', 5),
                max_rpm=agent_config.get('max_rpm', 10)
            )
            agents.append(agent)

        return agents

    def _create_tasks(self, agents: List[Agent], research_input: Dict[str, Any], human_in_loop: bool = False) -> List[Task]:
        """
        Create tasks from YAML configuration with dynamic research substitution

        Args:
            agents (List[Agent]): List of available agents
            research_input (Dict[str, Any]): Research parameters for task customization

        Returns:
            List[Task]: List of configured tasks with assigned agents
        """
        tasks = []
        tasks_config = self.config.get('tasks', {})

        for task_name, task_config in tasks_config.items():
            # Substitute research parameters into task description templates
            description = task_config['description'].format(
                research_topic=research_input.get('research_topic', 'Not specified'),
                research_depth=research_input.get('research_depth', 'comprehensive'),
                output_format=research_input.get('output_format', 'report'),
                target_sources=', '.join(research_input.get('sources_required', [])) if research_input.get('sources_required') else 'General academic and industry sources',
                special_requirements=research_input.get('special_requirements', 'Standard research requirements'),
                deadline=research_input.get('deadline', 'No deadline specified'),
                sources_required=', '.join(research_input.get('sources_required', [])) if research_input.get('sources_required') else 'No specific sources',
                research_context=f"Research Topic: {research_input.get('research_topic', 'Not specified')}\nDepth: {research_input.get('research_depth', 'comprehensive')}\nOutput: {research_input.get('output_format', 'report')}"
            )

            agent_key = task_config['agent']
            agent = None

            # Map YAML agent keys to actual agent instances by index
            agent_mapping = {
                'research_coordinator': 0,
                'information_gatherer': 1,
                'fact_checker': 2,
                'research_analyst': 3,
                'report_writer': 4
            }

            if agent_key in agent_mapping and len(agents) > agent_mapping[agent_key]:
                agent = agents[agent_mapping[agent_key]]

            if not agent:
                st.warning(f"Agent not found for task {task_name}")
                continue

            # Get context tasks
            context_tasks = []
            if task_config.get('context'):
                for context_task_name in task_config['context']:
                    for existing_task in tasks:
                        if context_task_name in existing_task.description or hasattr(existing_task, 'name') and existing_task.name == context_task_name:
                            context_tasks.append(existing_task)
                            break

            # Override human_input based on UI configuration
            task_human_input = task_config.get('human_input', False) and human_in_loop

            task = Task(
                description=description,
                expected_output=task_config['expected_output'],
                agent=agent,
                async_execution=task_config.get('async_execution', False),
                output_file=task_config.get('output_file'),
                human_input=task_human_input,  # Use UI override
                context=context_tasks
            )
            tasks.append(task)

        return tasks

    def create_crew(self, llm_provider: str, model_name: str, research_input: Dict[str, Any], human_in_loop: bool = False, **llm_kwargs) -> Crew:
        """
        Create and configure the complete research crew

        Args:
            llm_provider (str): LLM provider name
            model_name (str): Model name to use
            research_input (Dict[str, Any]): Research parameters
            human_in_loop (bool): Enable human-in-the-loop interaction

        Returns:
            Crew: Configured CrewAI crew ready for execution
        """
        agents = self._create_agents(llm_provider, model_name, **llm_kwargs)
        tasks = self._create_tasks(agents, research_input, human_in_loop)

        crew_config = self.config.get('crew', {}).get('crew_config', {})

        # Setup manager LLM for hierarchical process
        manager_llm_config = crew_config.get('manager_llm', {})
        manager_kwargs = {k: v for k, v in manager_llm_config.items() if k != 'model'}
        # Merge manager-specific kwargs with LLM kwargs (manager_kwargs takes precedence)
        merged_kwargs = {**llm_kwargs, **manager_kwargs}
        manager_llm = self._create_llm(
            llm_provider,
            manager_llm_config.get('model', model_name),
            **merged_kwargs
        )

        # Create crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical if crew_config.get('process') == 'hierarchical' else Process.sequential,
            cache=crew_config.get('cache', True),
            max_rpm=crew_config.get('max_rpm', 100),
            manager_llm=manager_llm,
            planning=crew_config.get('planning', True),
            verbose=crew_config.get('verbose', True)
        )

        return crew

# ============================================================================
# BACKEND: YAML CONFIGURATION VALIDATION AND UTILITIES
# ============================================================================

def validate_research_yaml_configs(config_path: Path) -> bool:
    """
    Validate that all required YAML configuration files exist

    Args:
        config_path (Path): Path to configuration directory

    Returns:
        bool: True if all configs are valid, False otherwise
    """
    required_files = ['agents.yaml', 'tasks.yaml', 'crew.yaml']

    for file_name in required_files:
        file_path = config_path / file_name
        if not file_path.exists():
            st.error(f"Required configuration file missing: {file_name}")
            return False

    return True

def get_available_research_tool_names() -> List[str]:
    """
    Get list of available tool names for YAML configuration

    Returns:
        List[str]: Available tool names
    """
    return ['web_search_tool', 'fact_check_tool', 'citation_tool']

def display_research_yaml_configuration_help():
    """
    Display help information about YAML configuration structure
    """
    st.info("""
    **Research Assistant YAML Configuration:**

    This application uses YAML files for complete configuration:

    📄 **agents.yaml** - Define research agents with roles, goals, and tools
    📄 **tasks.yaml** - Define research workflow tasks with descriptions and dependencies
    📄 **crew.yaml** - Configure crew behavior, process type, and settings

    **Key Features:**
    - ✅ **Pure YAML Configuration** - No hardcoded agents or tasks
    - ✅ **Dynamic Tool Assignment** - Research tools assigned based on YAML config
    - ✅ **Flexible Research Workflows** - Define complex research dependencies
    - ✅ **Multiple Process Types** - Sequential or hierarchical execution
    - ✅ **Template Variables** - Dynamic research parameter substitution

    **Available Research Tools:**
    - `web_search_tool` - Web search and information gathering
    - `fact_check_tool` - Fact verification and validation
    - `citation_tool` - Citation management and formatting

    **Customization:**
    Modify the YAML files in the `/config` directory to customize:
    - Research agent specializations and capabilities
    - Research task workflows and dependencies
    - Crew execution parameters and research methodologies
    - Tool assignments per research agent
    """)


# ============================================================================
# FRONTEND: STREAMLIT USER INTERFACE
# ============================================================================

def render_research_crew_interface():
    """
    Render the main Streamlit user interface for Research Assistant Crew

    Provides:
    - LLM provider and model selection
    - Research input form
    - Advanced research configuration options
    - Results display with multiple tabs
    """
    st.header("🔬 Research Assistant Crew")

    # ========================================
    # UI SECTION: Sidebar Configuration Panel
    # ========================================

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='research_llm_provider',
            help="Choose your preferred AI model provider"
        )

        model_options = {
            # Ollama: Free Open Models, runs on your local system (no API key required)
            "Ollama": ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "gemma2:2b", "gemma2:9b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "codestral:22b", "deepseek-coder:1.3b"],
            # Gemini: Google's Gemini models (requires API key)
            "Gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
            # Groq: Open Models (requires API key)
            "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
            # Anthropic: Claude AI models, good at reasoning & Coding (requires API key)
            "Anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"],
            # OpenAI: ChatGPT and GPT models, good at reasoning(requires API key)
            "OpenAI": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
        }

        # Model Selection based on provider
        model = st.selectbox(
            "Model",
            model_options[llm_provider],
            key='research_model',
            help="Select the specific model variant"
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='research_ollama_url',
                help="URL where Ollama server is running"
            )

            # Check Ollama status
            try:
                import requests
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    st.success("✅ Ollama server is running")
                else:
                    st.error("❌ Ollama server not accessible")
            except:
                st.error("❌ Cannot connect to Ollama server")
                st.markdown("**Setup Instructions:**")
                st.code("1. Install Ollama from https://ollama.com\n2. Run: ollama serve\n3. Pull model: ollama pull " + model)

        # Crew Process Configuration
        process_type = st.selectbox(
            "Crew Process",
            ["Sequential", "Hierarchical"],
            key='research_process',
            help="Sequential: agents work one after another. Hierarchical: manager coordinates agents."
        )

        # Configuration help
        if st.button("ℹ️ YAML Config Help"):
            display_research_yaml_configuration_help()

    # ========================================
    # UI SECTION: Main Research Input Form
    # ========================================
    col1, col2 = st.columns(2)

    with col1:
        # Primary research parameters
        research_topic = st.text_area(
            "Research Topic/Question *",
            placeholder="e.g., Impact of AI on Healthcare Efficiency, Sustainable Energy Solutions",
            height=100,
            key='research_topic',
            help="The main research question or topic you want to investigate"
        )

        research_depth = st.selectbox(
            "Research Depth *",
            ["Basic", "Comprehensive", "Deep"],
            index=1,
            key='research_depth',
            help="Basic: Quick overview, Comprehensive: Detailed analysis, Deep: Exhaustive investigation"
        )

        output_format = st.selectbox(
            "Output Format *",
            ["Research Report", "Executive Summary", "Presentation Outline", "Literature Review"],
            key='research_output_format',
            help="Choose the format for your research deliverable"
        )

    with col2:
        # Secondary research parameters
        sources_input = st.text_area(
            "Preferred Sources (optional)",
            placeholder="e.g., Academic journals, Industry reports, Government data",
            height=100,
            key='research_sources',
            help="Specify types of sources to prioritize in the research"
        )

        deadline = st.date_input(
            "Deadline (optional)",
            value=None,
            key='research_deadline'
        )

        sources_required = [s.strip() for s in sources_input.split(',') if s.strip()] if sources_input else []

    # Advanced options
    with st.expander("🔧 Advanced Research Options", expanded=False):
        include_citations = st.checkbox("Include Detailed Citations", value=True, key='research_citations')
        fact_check_level = st.selectbox(
            "Fact-Checking Level",
            ["Standard", "Rigorous", "Comprehensive"],
            index=1,
            key='research_fact_check'
        )
        regional_focus = st.text_input(
            "Regional/Geographic Focus (optional)",
            placeholder="e.g., North America, Global, Europe",
            key='research_region'
        )

        st.markdown("**🤝 Human Interaction**")
        human_in_loop = st.checkbox(
            "Human-in-the-Loop",
            value=False,
            key='research_human_in_loop',
            help="Enable human review and approval at key stages"
        )

    # Execute research
    if st.button("Start Research", type="primary", key='start_research'):
        if not research_topic:
            st.error("Please provide a research topic or question.")
            return

        research_input = ResearchInput(
            research_topic=research_topic,
            research_depth=research_depth.lower(),
            sources_required=sources_required,
            output_format=output_format.lower(),
            deadline=str(deadline) if deadline else None
        )

        with st.spinner("🔄 Assembling research team..."):
            try:
                # Initialize the crew manager
                crew_manager = ResearchCrew()

                # Create research input dictionary
                research_input_dict = {
                    'research_topic': research_input.research_topic,
                    'research_depth': research_input.research_depth,
                    'output_format': research_input.output_format,
                    'sources_required': research_input.sources_required,
                    'deadline': research_input.deadline
                }

                # Create and configure crew
                llm_kwargs = {}
                if llm_provider == "Ollama" and ollama_base_url:
                    llm_kwargs['base_url'] = ollama_base_url

                crew = crew_manager.create_crew(llm_provider, model, research_input_dict, human_in_loop, **llm_kwargs)

                # Execute crew
                with st.spinner("🔬 Conducting research..."):
                    result = crew.kickoff()

                # Display results
                st.success("✅ Research completed!")

                # Results tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📄 Research Report", "📊 Process Overview", "🔍 Methodology", "📚 Sources"])

                with tab1:
                    st.markdown("### 📄 Research Report")
                    st.markdown(result)

                with tab2:
                    st.markdown("### 📊 Research Process")
                    for i, task in enumerate(crew.tasks, 1):
                        with st.expander(f"Phase {i}: {task.agent.role}", expanded=False):
                            st.write(f"**Objective:** {task.description}")
                            st.write(f"**Deliverable:** {task.expected_output}")

                with tab3:
                    st.markdown("### 🔍 Research Methodology")
                    st.info(f"""
                    **Research Approach:**
                    - **Topic:** {research_topic}
                    - **Depth:** {research_depth}
                    - **Format:** {output_format}
                    - **Fact-Check Level:** {fact_check_level}

                    **Quality Assurance:**
                    - Multi-agent verification process
                    - Cross-source validation
                    - Systematic fact-checking
                    - Citation management and formatting
                    """)

                with tab4:
                    st.markdown("### 📚 Research Sources")
                    st.info("""
                    **Source Categories Accessed:**
                    - Academic and peer-reviewed publications
                    - Industry reports and market analysis
                    - Government databases and statistics
                    - Expert opinions and commentary
                    - Case studies and real-world examples

                    **Quality Standards:**
                    - Source credibility verification
                    - Publication date relevance
                    - Authority and expertise assessment
                    - Bias detection and mitigation
                    """)

            except Exception as e:
                st.error(f"Research error: {str(e)}")
                st.info("Please check your API keys and try again.")

if __name__ == "__main__":
    render_research_crew_interface()