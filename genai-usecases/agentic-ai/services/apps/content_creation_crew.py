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

import yaml
import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        self.config_path = (
            config_path
            or Path(__file__).parent / "config" / "content_creation_crew"
        )

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
                logger.warning(f"Configuration file {file_name} not found")

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
                logger.warning(f"Agent not found for task {task_name}")
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
            logger.error(f"Required configuration file missing: {file_name}")
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
    logger.info("""
    **YAML Configuration Structure:**

    This application uses YAML files for complete configuration:

   **agents.yaml** - Define AI agents with roles, goals, and tools
   **tasks.yaml** - Define workflow tasks with descriptions and dependencies
   **crew.yaml** - Configure crew behavior, process type, and settings

    **Key Features:**
    - **Pure YAML Configuration** - No hardcoded agents or tasks
    - **Dynamic Tool Assignment** - Tools assigned based on YAML config
    - **Flexible Task Dependencies** - Define complex workflows
    - **Multiple Process Types** - Sequential or hierarchical execution
    - **Template Variables** - Dynamic content substitution in tasks

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
