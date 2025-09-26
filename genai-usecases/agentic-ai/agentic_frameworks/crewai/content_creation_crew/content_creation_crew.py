"""
CrewAI - Content Creation Crew Implementation
Features:
- Multi-agent content generation
- SEO optimization
- Brand voice consistency
- Content strategy planning
- Multi-format output (blog, social, email)
- Quality assurance workflow
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

class ContentInput(BaseModel):
    """
    Input schema for content creation requests

    Defines the structure and validation for content creation parameters
    received from the frontend interface.
    """
    topic: str = Field(..., description="Content topic or theme")
    content_type: str = Field(..., description="Type of content (blog, social, email, etc.)")
    target_audience: str = Field(..., description="Target audience description")
    tone: str = Field(default="professional", description="Content tone")
    keywords: List[str] = Field(default=[], description="SEO keywords to include")

# ============================================================================
# BACKEND: CUSTOM TOOLS FOR CONTENT CREATION
# ============================================================================

class SEOAnalysisTool(BaseTool):
    """
    SEO Analysis and Optimization Tool

    Provides comprehensive SEO analysis including keyword density,
    readability scores, and optimization recommendations.
    """
    name: str = "seo_analysis_tool"
    description: str = "Analyzes content for SEO optimization and provides recommendations"

    def _run(self, content: str, keywords: List[str] = None) -> str:
        """
        Analyze content for SEO metrics and provide recommendations

        Args:
            content (str): The content text to analyze
            keywords (List[str], optional): Target keywords for analysis

        Returns:
            str: JSON formatted SEO analysis report
        """
        try:
            # Calculate basic SEO metrics
            word_count = len(content.split())
            keyword_density = {}

            # Analyze keyword density if keywords provided
            if keywords:
                for keyword in keywords:
                    count = content.lower().count(keyword.lower())
                    density = (count / word_count) * 100 if word_count > 0 else 0
                    keyword_density[keyword] = round(density, 2)

            # Simple readability score calculation
            readability_score = min(100, max(0, 100 - (word_count / 10)))

            # Compile SEO analysis report
            analysis = {
                "word_count": word_count,
                "keyword_density": keyword_density,
                "readability_score": round(readability_score, 1),
                "recommendations": [
                    f"Optimal word count range: 800-1200 words (current: {word_count})",
                    "Add more subheadings for better structure",
                    "Include meta description and title tags",
                    "Optimize keyword density (aim for 1-2%)"
                ]
            }

            return json.dumps(analysis, indent=2)

        except Exception as e:
            return f"SEO analysis error: {str(e)}"

class ContentPlannerTool(BaseTool):
    """
    Content Strategy and Planning Tool

    Generates comprehensive content strategies including content pillars,
    posting schedules, and engagement tactics.
    """
    name: str = "content_planner_tool"
    description: str = "Creates content strategy and planning recommendations"

    def _run(self, topic: str, audience: str, content_type: str) -> str:
        """
        Generate comprehensive content strategy recommendations

        Args:
            topic (str): Content topic or theme
            audience (str): Target audience description
            content_type (str): Type of content being planned

        Returns:
            str: JSON formatted content strategy plan
        """
        try:
            # Generate content strategy framework
            strategy = {
                "content_pillars": [
                    "Educational content",
                    "Industry insights",
                    "Product/service highlights",
                    "Community engagement"
                ],
                "posting_schedule": {
                    "frequency": "3-4 posts per week",
                    "best_times": ["9 AM", "1 PM", "6 PM"],
                    "optimal_days": ["Tuesday", "Wednesday", "Thursday"]
                },
                "content_mix": {
                    "educational": "40%",
                    "promotional": "20%",
                    "entertaining": "25%",
                    "user_generated": "15%"
                },
                "engagement_tactics": [
                    "Ask questions to encourage comments",
                    "Use polls and interactive elements",
                    "Share behind-the-scenes content",
                    "Respond promptly to audience interactions"
                ]
            }

            return json.dumps(strategy, indent=2)

        except Exception as e:
            return f"Content planning error: {str(e)}"

# ============================================================================
# BACKEND: CORE CREW MANAGEMENT CLASS
# ============================================================================

class ContentCreationCrew:
    """
    CrewAI Implementation with YAML Configuration

    Manages the complete content creation workflow using multiple AI agents.
    Loads configurations from YAML files for agents, tasks, and crew settings.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the content creation crew manager

        Args:
            config_path (str, optional): Path to configuration directory
        """
        self.config_path = config_path or Path(__file__).parent / "config"

        # Validate configuration files exist
        if not validate_yaml_configs(self.config_path):
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
        Initialize and setup content creation tools

        Returns:
            List[BaseTool]: List of available tools for agents
        """
        return [
            SEOAnalysisTool(),
            ContentPlannerTool()
        ]

    def _create_llm(self, provider: str, model: str, **kwargs) -> LLM:
        """
        Create LLM instance with provider-specific configuration

        Args:
            provider (str): LLM provider (Ollama, Gemini, Groq, Anthropic, OpenAI)
            model (str): Model name
            **kwargs: Additional model parameters

        Returns:
            LLM: Configured LLM instance
        """
        # Extract specific parameters to avoid conflicts
        llm_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'timeout': kwargs.get('timeout', 120),
            'max_tokens': kwargs.get('max_tokens', 4000)
        }

        # Configure model string and API keys based on provider
        if provider == "Ollama":
            # Use CrewAI's built-in Ollama support via LiteLLM
            llm_params['model'] = f"ollama/{model}"
            if 'base_url' in kwargs:
                llm_params['base_url'] = kwargs['base_url']
        elif provider == "Gemini":
            # Gemini models don't need provider prefix in CrewAI
            llm_params['model'] = f"gemini/{model}"
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                llm_params['api_key'] = api_key
        elif provider == "Groq":
            llm_params['model'] = f"groq/{model}"
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                llm_params['api_key'] = api_key
        elif provider == "Anthropic":
            llm_params['model'] = f"anthropic/{model}"
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                llm_params['api_key'] = api_key
        elif provider == "OpenAI":
            llm_params['model'] = f"openai/{model}"
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                llm_params['api_key'] = api_key
        else:
            llm_params['model'] = model

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ['temperature', 'timeout', 'max_tokens', 'model', 'api_key']:
                llm_params[key] = value

        return LLM(**llm_params)

    def _create_agents(self, llm_provider: str, model_name: str, **llm_kwargs) -> List[Agent]:
        """
        Create AI agents based on YAML configuration

        Args:
            llm_provider (str): LLM provider name
            model_name (str): Model name to use

        Returns:
            List[Agent]: List of configured CrewAI agents
        """
        agents = []
        agents_config = self.config.get('agents', {})

        llm = self._create_llm(llm_provider, model_name, **llm_kwargs)

        for agent_name, agent_config in agents_config.items():
            # Configure tools for each agent based on YAML configuration
            agent_tools = []
            if agent_config.get('tools'):
                # If tools is True, assign tools based on agent role automatically
                if agent_name == 'content_strategist':
                    agent_tools.append(ContentPlannerTool())
                elif agent_name == 'seo_specialist':
                    agent_tools.append(SEOAnalysisTool())

                # Also check for explicit tool_names in YAML
                tool_names = agent_config.get('tool_names', [])
                for tool_name in tool_names:
                    if tool_name == 'seo_analysis_tool' and not any(isinstance(t, SEOAnalysisTool) for t in agent_tools):
                        agent_tools.append(SEOAnalysisTool())
                    elif tool_name == 'content_planning_tool' and not any(isinstance(t, ContentPlannerTool) for t in agent_tools):
                        agent_tools.append(ContentPlannerTool())

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

    def _create_tasks(self, agents: List[Agent], content_input: Dict[str, Any], human_in_loop: bool = False) -> List[Task]:
        """
        Create tasks from YAML configuration with dynamic content substitution

        Args:
            agents (List[Agent]): List of available agents
            content_input (Dict[str, Any]): Content parameters for task customization

        Returns:
            List[Task]: List of configured tasks with assigned agents
        """
        tasks = []
        tasks_config = self.config.get('tasks', {})

        for task_name, task_config in tasks_config.items():
            # Substitute content parameters into task description templates
            description = task_config['description'].format(
                content_topic=content_input.get('topic', 'Not specified'),
                target_audience=content_input.get('target_audience', 'Not specified'),
                content_type=content_input.get('content_type', 'Not specified'),
                tone_style=content_input.get('tone', 'professional'),
                word_count=content_input.get('word_count', '800-1000'),
                brand_guidelines=content_input.get('brand_guidelines', 'Standard brand guidelines'),
                business_objectives=content_input.get('business_objectives', 'Increase engagement and brand awareness'),
                target_keywords=', '.join(content_input.get('keywords', [])) if content_input.get('keywords') else 'No specific keywords',
                seo_requirements=content_input.get('seo_requirements', 'Standard SEO optimization')
            )

            agent_key = task_config['agent']
            agent = None

            # Map YAML agent keys to actual agent instances by index
            agent_mapping = {
                'content_strategist': 0,
                'content_writer': 1,
                'seo_specialist': 2,
                'quality_reviewer': 3
            }

            if agent_key in agent_mapping and len(agents) > agent_mapping[agent_key]:
                agent = agents[agent_mapping[agent_key]]

            if not agent:
                st.warning(f"Agent not found for task {task_name}")
                continue

            # Build task dependencies from context configuration
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

    def create_crew(self, llm_provider: str, model_name: str, content_input: Dict[str, Any], human_in_loop: bool = False, **llm_kwargs) -> Crew:
        """
        Create and configure the complete content creation crew

        Args:
            llm_provider (str): LLM provider name
            model_name (str): Model name to use
            content_input (Dict[str, Any]): Content creation parameters
            human_in_loop (bool): Enable human-in-the-loop interaction

        Returns:
            Crew: Configured CrewAI crew ready for execution
        """
        agents = self._create_agents(llm_provider, model_name, **llm_kwargs)
        tasks = self._create_tasks(agents, content_input, human_in_loop)

        crew_config = self.config.get('crew', {}).get('crew_config', {})

        # Configure manager LLM for hierarchical process if specified
        manager_llm_config = crew_config.get('manager_llm', {})
        manager_kwargs = {k: v for k, v in manager_llm_config.items() if k != 'model'}
        # Merge manager-specific kwargs with LLM kwargs (manager_kwargs takes precedence)
        merged_kwargs = {**llm_kwargs, **manager_kwargs}
        manager_llm = self._create_llm(
            llm_provider,
            manager_llm_config.get('model', model_name),
            **merged_kwargs
        )

        # Assemble the final crew with speed optimizations
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical if crew_config.get('process') == 'hierarchical' else Process.sequential,
            cache=crew_config.get('cache', True),
            max_rpm=crew_config.get('max_rpm', 150),
            manager_llm=manager_llm,
            planning=crew_config.get('planning', False),
            verbose=crew_config.get('verbose', False)
        )

        return crew

# ============================================================================
# BACKEND: YAML CONFIGURATION VALIDATION AND UTILITIES
# ============================================================================

def validate_yaml_configs(config_path: Path) -> bool:
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

def get_available_tool_names() -> List[str]:
    """
    Get list of available tool names for YAML configuration

    Returns:
        List[str]: Available tool names
    """
    return ['seo_analysis_tool', 'content_planning_tool']

def display_yaml_configuration_help():
    """
    Display help information about YAML configuration structure
    """
    st.info("""
    **YAML Configuration Structure:**

    This application uses YAML files for complete configuration:

    📄 **agents.yaml** - Define AI agents with roles, goals, and tools
    📄 **tasks.yaml** - Define workflow tasks with descriptions and dependencies
    📄 **crew.yaml** - Configure crew behavior, process type, and settings

    **Key Features:**
    - ✅ **Pure YAML Configuration** - No hardcoded agents or tasks
    - ✅ **Dynamic Tool Assignment** - Tools assigned based on YAML config
    - ✅ **Flexible Task Dependencies** - Define complex workflows
    - ✅ **Multiple Process Types** - Sequential or hierarchical execution
    - ✅ **Template Variables** - Dynamic content substitution in tasks

    **Available Tools:**
    - `seo_analysis_tool` - SEO optimization and analysis
    - `content_planning_tool` - Content strategy and planning

    **Customization:**
    Modify the YAML files in the `/config` directory to customize:
    - Agent personalities and capabilities
    - Task workflows and dependencies
    - Crew execution parameters
    - Tool assignments per agent
    """)

# ============================================================================
# FRONTEND: STREAMLIT USER INTERFACE
# ============================================================================

def render_content_crew_interface():
    """
    Render the main Streamlit user interface for Content Creation Crew

    Provides:
    - LLM provider and model selection
    - Content input form
    - Advanced configuration options
    - Results display with multiple tabs
    """
    st.header("🎨 Content Creation Crew")

    # ========================================
    # UI SECTION: Sidebar Configuration Panel
    # ========================================
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='content_llm_provider',
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
            key='content_model',
            help="Select the specific model variant"
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='content_ollama_url',
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
            key='content_process',
            help="Sequential: agents work one after another. Hierarchical: manager coordinates agents."
        )

        # Configuration help
        if st.button("ℹ️ YAML Config Help"):
            display_yaml_configuration_help()

    # ========================================
    # UI SECTION: Main Content Input Form
    # ========================================
    col1, col2 = st.columns(2)

    with col1:
        # Primary content parameters
        topic = st.text_input(
            "Content Topic *",
            placeholder="e.g., AI in Healthcare, Sustainable Technology",
            key='content_topic',
            help="The main subject or theme for your content"
        )

        content_type = st.selectbox(
            "Content Type *",
            ["Blog Post", "Social Media Post", "Email Newsletter", "Website Copy", "Product Description", "Press Release"],
            key='content_type',
            help="Choose the format for your content"
        )

        target_audience = st.text_input(
            "Target Audience *",
            placeholder="e.g., Healthcare professionals, Tech enthusiasts",
            key='content_audience',
            help="Describe who will be reading this content"
        )

    with col2:
        # Secondary content parameters
        tone = st.selectbox(
            "Content Tone *",
            ["Professional", "Casual", "Friendly", "Authoritative", "Conversational", "Technical"],
            key='content_tone',
            help="The voice and style for your content"
        )

        keywords_input = st.text_input(
            "SEO Keywords (comma-separated)",
            placeholder="e.g., artificial intelligence, healthcare technology",
            key='content_keywords',
            help="Keywords to optimize for search engines"
        )

        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()] if keywords_input else []

    # ========================================
    # UI SECTION: Advanced Configuration Options
    # ========================================
    with st.expander("🔧 Advanced Options", expanded=False):
        word_count_target = st.slider(
            "Target Word Count", 300, 2000, 800,
            key='content_word_count',
            help="Approximate length for the content"
        )
        include_cta = st.checkbox(
            "Include Call-to-Action", value=True,
            key='content_cta',
            help="Add action-oriented conclusion"
        )
        seo_focus = st.checkbox(
            "Heavy SEO Focus", value=True,
            key='content_seo_focus',
            help="Prioritize search engine optimization"
        )

        st.markdown("**🤝 Human Interaction**")
        human_in_loop = st.checkbox(
            "Human-in-the-Loop",
            value=False,
            key='content_human_in_loop',
            help="Enable human review and approval at key stages"
        )

    # ========================================
    # UI SECTION: Content Generation Execution
    # ========================================
    if st.button("Create Content", type="primary", key='create_content'):
        # Validate required fields
        if not topic or not target_audience:
            st.error("Please provide topic and target audience.")
            return

        # Prepare content input data
        content_input = ContentInput(
            topic=topic,
            content_type=content_type,
            target_audience=target_audience,
            tone=tone.lower(),
            keywords=keywords
        )

        # Backend processing with progress indicators
        with st.spinner("🔄 Assembling content creation crew..."):
            try:
                # Initialize the crew management system
                crew_manager = ContentCreationCrew()

                # Prepare backend data structure
                content_input_dict = {
                    'topic': content_input.topic,
                    'content_type': content_input.content_type,
                    'target_audience': content_input.target_audience,
                    'tone': content_input.tone,
                    'keywords': content_input.keywords,
                    'word_count': str(word_count_target)
                }

                # Create and configure the AI crew
                llm_kwargs = {}
                if llm_provider == "Ollama" and ollama_base_url:
                    llm_kwargs['base_url'] = ollama_base_url

                crew = crew_manager.create_crew(llm_provider, model, content_input_dict, human_in_loop, **llm_kwargs)

                # Execute the content creation workflow
                with st.spinner("🎨 Creating content..."):
                    result = crew.kickoff()

                # ========================================
                # UI SECTION: Results Display
                # ========================================
                st.success("✅ Content creation completed!")

                # Multi-tab results interface
                tab1, tab2, tab3 = st.tabs(["📄 Final Content", "📊 Process Details", "💡 Recommendations"])

                # Tab 1: Final Content Display
                with tab1:
                    st.markdown("### 📄 Created Content")
                    st.markdown(result)

                # Tab 2: Process Details and Workflow
                with tab2:
                    st.markdown("### 📊 Content Creation Process")
                    if hasattr(crew, 'tasks') and crew.tasks:
                        for i, task in enumerate(crew.tasks, 1):
                            with st.expander(f"Step {i}: {task.agent.role}", expanded=False):
                                st.write(f"**Description:** {task.description}")
                                st.write(f"**Expected Output:** {task.expected_output}")
                    else:
                        st.write("**Process Steps:**")
                        st.write("1. Content Strategy Development")
                        st.write("2. Content Creation")
                        st.write("3. SEO Optimization")
                        st.write("4. Quality Review")

                # Tab 3: Recommendations and Best Practices
                with tab3:
                    st.markdown("### 💡 Content Strategy Recommendations")
                    st.info("""
                    **Best Practices Applied:**
                    - Multi-agent collaboration for comprehensive content creation
                    - SEO optimization integrated into the workflow
                    - Quality assurance to ensure high standards
                    - Strategic planning for long-term content success

                    **Next Steps:**
                    - Review and approve the content
                    - Schedule publication according to strategy
                    - Monitor performance and engagement
                    - Iterate based on audience feedback
                    """)

            # ========================================
            # UI SECTION: Error Handling and User Feedback
            # ========================================
            except Exception as e:
                error_msg = str(e)
                st.error(f"Content creation error: {error_msg}")

                
# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    render_content_crew_interface()