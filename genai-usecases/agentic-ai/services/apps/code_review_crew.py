"""
CrewAI - Code Review Crew Implementation
Features:
- Automated code analysis and review
- Security vulnerability detection
- Performance optimization suggestions
- Code quality assessment
- Best practices enforcement
- Documentation review
"""

import yaml
import json
import logging
import os
import re
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

class CodeReviewInput(BaseModel):
    """
    Input schema for code review requests

    Defines the structure and validation for code review parameters
    received from the frontend interface.
    """
    code_content: str = Field(..., description="Code content to review")
    programming_language: str = Field(..., description="Programming language")
    review_type: str = Field(default="comprehensive", description="Type of review")
    focus_areas: List[str] = Field(default=[], description="Specific areas to focus on")
    project_context: Optional[str] = Field(default=None, description="Project context and requirements")

# ============================================================================
# BACKEND: CUSTOM TOOLS FOR CODE REVIEW
# ============================================================================

class SecurityAnalyzerTool(BaseTool):
    """
    Security Vulnerability Analysis Tool

    Analyzes code for common security vulnerabilities and provides
    detailed security assessment with recommendations.
    """
    name: str = "security_analyzer_tool"
    description: str = "Analyzes code for security vulnerabilities and issues"

    def _run(self, code: str, language: str) -> str:
        """
        Analyze code for security vulnerabilities

        Args:
            code (str): Source code to analyze
            language (str): Programming language

        Returns:
            str: JSON formatted security analysis report
        """
        try:
            # Simulate security analysis
            security_issues = []

            # Common security patterns to check
            security_patterns = {
                "SQL Injection": r"(SELECT|INSERT|UPDATE|DELETE).*(\+|\|\|)",
                "XSS Vulnerability": r"innerHTML|outerHTML|document\.write",
                "Hardcoded Secrets": r"(password|secret|key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
                "Command Injection": r"exec|system|shell_exec|eval",
                "Insecure Random": r"Math\.random|Random\(\)",
                "Weak Cryptography": r"MD5|SHA1(?!SHA256|SHA512)"
            }

            for issue_type, pattern in security_patterns.items():
                if re.search(pattern, code, re.IGNORECASE):
                    security_issues.append({
                        "type": issue_type,
                        "severity": "Medium" if "Injection" in issue_type else "Low",
                        "description": f"Potential {issue_type.lower()} vulnerability detected",
                        "recommendation": f"Review and secure {issue_type.lower()} implementation"
                    })

            analysis = {
                "language": language,
                "security_score": max(0, 100 - (len(security_issues) * 15)),
                "vulnerabilities_found": len(security_issues),
                "issues": security_issues,
                "recommendations": [
                    "Use parameterized queries to prevent SQL injection",
                    "Sanitize all user inputs",
                    "Use secure random number generators",
                    "Implement proper authentication and authorization",
                    "Keep dependencies updated"
                ]
            }

            return json.dumps(analysis, indent=2)

        except Exception as e:
            return f"Security analysis error: {str(e)}"

class PerformanceAnalyzerTool(BaseTool):
    """
    Performance Analysis and Optimization Tool

    Analyzes code for performance bottlenecks, inefficient patterns,
    and provides optimization recommendations.
    """
    name: str = "performance_analyzer_tool"
    description: str = "Analyzes code for performance issues and optimization opportunities"

    def _run(self, code: str, language: str) -> str:
        """
        Analyze code for performance issues and optimization opportunities

        Args:
            code (str): Source code to analyze
            language (str): Programming language

        Returns:
            str: JSON formatted performance analysis report
        """
        try:
            performance_issues = []

            # Performance patterns to check
            performance_patterns = {
                "Nested Loops": r"for\s*\([^}]*\{[^}]*for\s*\(",
                "Inefficient String Concatenation": r"\+\=.*['\"]",
                "Redundant Database Calls": r"(SELECT|INSERT|UPDATE|DELETE).*for\s*\(",
                "Memory Leaks": r"new\s+\w+\s*\([^}]*(?!delete|free)",
                "Synchronous I/O": r"(readFileSync|writeFileSync)",
                "Large Object Creation": r"new\s+(Array|Object)\s*\(\s*\d{4,}"
            }

            for issue_type, pattern in performance_patterns.items():
                if re.search(pattern, code, re.IGNORECASE):
                    performance_issues.append({
                        "type": issue_type,
                        "impact": "High" if "Loop" in issue_type or "Database" in issue_type else "Medium",
                        "description": f"Potential {issue_type.lower()} performance issue",
                        "suggestion": f"Consider optimizing {issue_type.lower()}"
                    })

            # Calculate complexity score
            complexity_indicators = len(re.findall(r'\bif\b|\bfor\b|\bwhile\b|\bcatch\b', code, re.IGNORECASE))

            analysis = {
                "language": language,
                "performance_score": max(0, 100 - (len(performance_issues) * 12)),
                "complexity_score": min(100, complexity_indicators * 5),
                "issues_found": len(performance_issues),
                "issues": performance_issues,
                "optimizations": [
                    "Consider using more efficient algorithms",
                    "Implement caching where appropriate",
                    "Use asynchronous operations for I/O",
                    "Optimize database queries",
                    "Consider lazy loading for large datasets"
                ]
            }

            return json.dumps(analysis, indent=2)

        except Exception as e:
            return f"Performance analysis error: {str(e)}"

class CodeQualityTool(BaseTool):
    """
    Code Quality and Best Practices Analysis Tool

    Analyzes code quality metrics, style adherence, and
    compliance with programming best practices.
    """
    name: str = "code_quality_tool"
    description: str = "Analyzes code quality, style, and adherence to best practices"

    def _run(self, code: str, language: str) -> str:
        """
        Analyze code quality and best practices compliance

        Args:
            code (str): Source code to analyze
            language (str): Programming language

        Returns:
            str: JSON formatted code quality analysis report
        """
        try:
            quality_issues = []

            # Code quality checks
            lines = code.split('\n')
            total_lines = len(lines)
            comment_lines = len([line for line in lines if line.strip().startswith(('//','#','/*','*','"""',"'''"))])
            empty_lines = len([line for line in lines if not line.strip()])

            # Calculate metrics
            comment_ratio = (comment_lines / total_lines * 100) if total_lines > 0 else 0
            function_count = len(re.findall(r'\b(def|function|func)\s+\w+', code, re.IGNORECASE))
            avg_function_length = (total_lines - comment_lines - empty_lines) / max(1, function_count)

            # Quality assessments
            if comment_ratio < 10:
                quality_issues.append({
                    "type": "Insufficient Documentation",
                    "severity": "Medium",
                    "description": f"Only {comment_ratio:.1f}% of code is commented",
                    "recommendation": "Add more comments and documentation"
                })

            if avg_function_length > 50:
                quality_issues.append({
                    "type": "Large Functions",
                    "severity": "Medium",
                    "description": f"Average function length is {avg_function_length:.1f} lines",
                    "recommendation": "Break down large functions into smaller ones"
                })

            # Check for naming conventions
            if not re.search(r'[a-z][a-zA-Z0-9_]*', code):
                quality_issues.append({
                    "type": "Naming Convention",
                    "severity": "Low",
                    "description": "Inconsistent naming conventions detected",
                    "recommendation": "Follow consistent naming conventions"
                })

            analysis = {
                "language": language,
                "quality_score": max(0, 100 - (len(quality_issues) * 10)),
                "metrics": {
                    "total_lines": total_lines,
                    "comment_ratio": round(comment_ratio, 1),
                    "function_count": function_count,
                    "avg_function_length": round(avg_function_length, 1)
                },
                "issues": quality_issues,
                "best_practices": [
                    "Follow consistent naming conventions",
                    "Add comprehensive documentation",
                    "Keep functions small and focused",
                    "Use meaningful variable names",
                    "Implement proper error handling"
                ]
            }

            return json.dumps(analysis, indent=2)

        except Exception as e:
            return f"Code quality analysis error: {str(e)}"

# ============================================================================
# BACKEND: CORE CODE REVIEW CREW MANAGEMENT CLASS
# ============================================================================

class CodeReviewCrew:
    """
    CrewAI Implementation with YAML Configuration for Code Review

    Manages comprehensive code review workflows using multiple AI analysis agents.
    Loads configurations from YAML files for agents, tasks, and crew settings.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the code review crew manager

        Args:
            config_path (str, optional): Path to configuration directory
        """
        self.config_path = (
            config_path
            or Path(__file__).parent / "config" / "code_review_crew"
        )

        # Validate configuration files exist
        if not validate_code_review_yaml_configs(self.config_path):
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
        Initialize and setup code review tools

        Returns:
            List[BaseTool]: List of available code review tools for agents
        """
        return [
            SecurityAnalyzerTool(),
            PerformanceAnalyzerTool(),
            CodeQualityTool()
        ]

    def _create_llm(self, provider: str, model: str, **kwargs) -> LLM:
        """
        Create an AI Language Model (LLM) for code review

        This function takes a provider (like 'Gemini' or 'OpenAI') and a model name,
        then creates a configured AI model that can understand and review code.

        Think of this like choosing which AI assistant to use and setting it up.
        """

        # Step 1: Set up API keys for different AI providers
        # Each AI company requires a different API key to access their models
        provider_api_keys = {
            "Gemini": "GEMINI_API_KEY",      # Google's Gemini models
            "Groq": "GROQ_API_KEY",          # Groq's fast AI models
            "OpenAI": "OPENAI_API_KEY",      # ChatGPT and GPT models
            "Anthropic": "ANTHROPIC_API_KEY" # Claude AI models
        }

        # Get the correct API key name for this provider
        api_key_name = provider_api_keys.get(provider)
        if api_key_name:
            # Check if the API key exists in environment variables
            api_key_value = os.getenv(api_key_name)
            if api_key_value:
                # Set the API key in environment variables so the AI can connect
                os.environ[api_key_name] = api_key_value
            else:
                # If using a provider that needs an API key but it's not set, show a helpful error
                if provider.lower() != 'ollama':  # Ollama doesn't need API keys
                    raise ValueError(f"Missing API key: {api_key_name} is not set in environment variables. Please set it to use {provider} models.")

        # Step 2: Format the model name correctly for each provider
        # Each AI provider expects their model names in a specific format
        # Example: "gpt-4" becomes "openai/gpt-4" for proper routing

        if provider.lower() == 'ollama':
            # Ollama runs locally on your computer
            formatted_model = f"ollama/{model}"
        elif provider.lower() == 'gemini':
            # Google's Gemini models
            formatted_model = f"gemini/{model}"
        elif provider.lower() == 'groq':
            # Groq's API based Open Models
            formatted_model = f"groq/{model}"
        elif provider.lower() == 'anthropic':
            # Anthropic's Claude models
            formatted_model = f"anthropic/{model}"
        elif provider.lower() == 'openai':
            # OpenAI's GPT models
            formatted_model = f"openai/{model}"
        else:
            # If provider is unknown, use model name as-is
            formatted_model = model

        # Step 3: Set up the AI model with configuration parameters
        # These settings control how the AI behaves during code review
        ai_model_settings = {
            'model': formatted_model,                           # Which AI model to use
            'temperature': kwargs.get('temperature', 0.7),     # Creativity level (0.0 = very focused, 1.0 = more creative)
            'timeout': kwargs.get('timeout', 120),             # How long to wait for AI response (seconds)
            'max_tokens': kwargs.get('max_tokens', 4000)       # Maximum length of AI response
        }

        # Step 4: Add special settings for Ollama (local AI models)
        # Ollama runs on your computer, so we need to tell it where to connect
        if provider.lower() == 'ollama' and 'base_url' in kwargs:
            ai_model_settings['base_url'] = kwargs['base_url']  # Usually http://localhost:11434

        # Step 5: Add any extra settings that were provided
        # This allows for custom configurations without breaking existing settings
        for setting_name, setting_value in kwargs.items():
            # Only add settings that don't conflict with our main settings
            if setting_name not in ['temperature', 'timeout', 'max_tokens', 'model', 'base_url']:
                ai_model_settings[setting_name] = setting_value

        # Step 6: Create and return the final AI model
        # The ** means "unpack all the settings" - it's like passing each setting individually
        return LLM(**ai_model_settings)

    def _create_agents(self, llm_provider: str, model_name: str, **llm_kwargs) -> tuple[List[Agent], dict]:
        """
        Create AI Agents for Code Review

        Think of agents as specialized AI assistants, each with a specific job:
        - Security Agent: Finds security vulnerabilities
        - Performance Agent: Checks for speed issues
        - Quality Agent: Reviews code style and best practices

        This function reads their job descriptions from YAML files and creates them.
        """

        # Step 1: Set up empty lists to store our agents
        code_review_agents = []  # List to hold all our AI agents
        agent_name_to_object = {}  # Dictionary to easily find agents by name

        # Step 2: Load agent configurations from YAML files
        # YAML files contain job descriptions, roles, and capabilities for each agent
        agent_configurations = self.config.get('agents', {})

        # Step 3: Create the AI brain that all agents will use
        # This is the language model (like GPT or Gemini) that powers the agents
        shared_ai_brain = self._create_llm(llm_provider, model_name, **llm_kwargs)

        # Step 4: Create each agent based on their configuration
        for agent_name, agent_config in agent_configurations.items():

            # Step 4a: Give each agent the right tools for their job
            # Tools are like special abilities (security scanner, performance checker, etc.)
            agent_tools = []

            # Check if this agent should have tools (some agents just think, others use tools)
            if agent_config.get('tools'):

                # Automatically give tools based on the agent's specialty
                # If agent name or role mentions "security", give them security tools
                if 'security' in agent_name.lower() or 'security' in agent_config.get('role', '').lower():
                    agent_tools.append(SecurityAnalyzerTool())  # Tool to find security bugs
                elif 'performance' in agent_name.lower() or 'performance' in agent_config.get('role', '').lower():
                    agent_tools.append(PerformanceAnalyzerTool())  # Tool to check code speed
                elif 'quality' in agent_name.lower() or 'quality' in agent_config.get('role', '').lower():
                    agent_tools.append(CodeQualityTool())  # Tool to check code style

                # Also check if specific tools are requested in the YAML configuration
                requested_tools = agent_config.get('tool_names', [])
                for tool_name in requested_tools:
                    # Only add tools that the agent doesn't already have
                    if tool_name == 'security_analyzer_tool' and not any(isinstance(t, SecurityAnalyzerTool) for t in agent_tools):
                        agent_tools.append(SecurityAnalyzerTool())
                    elif tool_name == 'performance_analyzer_tool' and not any(isinstance(t, PerformanceAnalyzerTool) for t in agent_tools):
                        agent_tools.append(PerformanceAnalyzerTool())
                    elif tool_name == 'code_quality_tool' and not any(isinstance(t, CodeQualityTool) for t in agent_tools):
                        agent_tools.append(CodeQualityTool())

            # Step 4b: Create the actual AI agent with all its settings
            # This is like hiring a specialized AI assistant and giving them their job description
            new_agent = Agent(
                role=agent_config['role'],                                    # What job title does this agent have?
                goal=agent_config['goal'],                                    # What is their main objective?
                backstory=agent_config['backstory'],                          # What experience/background do they have?
                tools=agent_tools,                                           # What tools can they use?
                llm=shared_ai_brain,                                         # Which AI brain should they use?
                verbose=agent_config.get('verbose', True),                   # Should they explain their thinking?
                allow_delegation=agent_config.get('allow_delegation', False), # Can they ask other agents for help?
                max_iter=agent_config.get('max_iter', 5),                    # How many times can they try to solve a problem?
                max_rpm=agent_config.get('max_rpm', 10)                      # How many requests per minute are allowed?
            )

            # Step 4c: Add the new agent to our collection
            code_review_agents.append(new_agent)              # Add to the list of all agents
            agent_name_to_object[agent_name] = new_agent       # Also add to name-lookup dictionary

        # Step 5: Return both the list of agents and the name-lookup dictionary
        # The list is for processing in order, the dictionary is for finding specific agents
        return code_review_agents, agent_name_to_object

    def _create_tasks(self, agent_mapping: dict, code_review_input: dict, human_in_loop: bool = False) -> List[Task]:
        """
        Create Tasks for the Code Review Process

        Tasks are like work assignments that we give to our AI agents.
        Each task tells an agent exactly what to do with the code.

        Example tasks:
        - "Check this code for security vulnerabilities"
        - "Find performance problems in this code"
        - "Review code style and suggest improvements"
        """

        # Step 1: Set up an empty list to store all our tasks
        code_review_tasks = []

        # Step 2: Load task definitions from YAML configuration files
        # These files contain detailed instructions for each type of code review task
        task_definitions = self.config.get('tasks', {})

        # Step 3: Create each task based on its definition
        for task_name, task_details in task_definitions.items():

            # Step 3a: Customize the task description with actual code details
            # The YAML files have placeholders like {code_content} that we fill in with real values
            customized_description = task_details['description'].format(**code_review_input)

            # Step 3b: Find which agent should do this task
            # Each task is assigned to a specific agent (security expert, performance expert, etc.)
            assigned_agent_name = task_details['agent']
            assigned_agent = agent_mapping.get(assigned_agent_name)

            # If we can't find the agent, skip this task and warn the user
            if not assigned_agent:
                logger.warning(f"Could not find agent '{assigned_agent_name}' for task '{task_name}'. Skipping this task.")
                continue

            # Step 3c: Find any prerequisite tasks that must complete first
            # Some tasks depend on others (e.g., "write final report" depends on "security analysis")
            prerequisite_tasks = []
            if task_details.get('context'):  # Check if this task has dependencies
                for prerequisite_task_name in task_details['context']:
                    # Look through already created tasks to find the prerequisites
                    for already_created_task in code_review_tasks:
                        # Match by name or description content
                        if (prerequisite_task_name in already_created_task.description or
                            hasattr(already_created_task, 'name') and already_created_task.name == prerequisite_task_name):
                            prerequisite_tasks.append(already_created_task)
                            break

            # Step 3d: Override human_input based on UI configuration
            task_human_input = task_details.get('human_input', False) and human_in_loop

            # Step 3e: Create the actual task with all its settings
            new_task = Task(
                description=customized_description,                                    # What exactly should the agent do?
                expected_output=task_details['expected_output'],                       # What kind of result do we expect?
                agent=assigned_agent,                                                 # Which agent will do this work?
                async_execution=task_details.get('async_execution', False),          # Should this run at the same time as other tasks?
                output_file=task_details.get('output_file'),                          # Where should results be saved?
                human_input=task_human_input,                                         # Use UI override for human input control
                context=prerequisite_tasks                                            # Which tasks must complete before this one?
            )

            # Step 3e: Add the completed task to our list
            code_review_tasks.append(new_task)

        # Step 4: Return the complete list of tasks, ready to be executed
        return code_review_tasks

    def create_crew(self, llm_provider: str, model_name: str, code_review_input: dict, human_in_loop: bool = False, **llm_kwargs) -> Crew:
        """Create the complete code review crew with output folder configuration"""
        agents, agent_mapping = self._create_agents(llm_provider, model_name, **llm_kwargs)
        tasks = self._create_tasks(agent_mapping, code_review_input, human_in_loop)

        crew_config = self.config.get('crew', {}).get('crew_config', {})

        # Setup output directory in crew folder
        output_dir = self.config_path.parent / "output"
        output_dir.mkdir(exist_ok=True)

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

        # Create crew with optimized settings
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical if crew_config.get('process') == 'hierarchical' else Process.sequential,
            cache=crew_config.get('cache', True),
            max_rpm=crew_config.get('max_rpm', 200),
            manager_llm=manager_llm,
            planning=crew_config.get('planning', False),
            verbose=crew_config.get('verbose', False),
            output_log_file=str(output_dir / "crew_log.txt"),
            step_callback=self._step_callback
        )

        return crew

    def _step_callback(self, step):
        """Enhanced callback to track crew progress with detailed monitoring"""

def validate_code_review_yaml_configs(config_path: Path) -> bool:
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

def get_available_code_review_tool_names() -> List[str]:
    """
    Get list of available tool names for YAML configuration

    Returns:
        List[str]: Available tool names
    """
    return ['security_analyzer_tool', 'performance_analyzer_tool', 'code_quality_tool']

def display_code_review_yaml_configuration_help():
    """
    Display help information about YAML configuration structure
    """
    logger.info("""
    **Code Review Crew YAML Configuration:**

    This application uses YAML files for complete configuration:

   **agents.yaml** - Define code review agents with roles, goals, and tools

   **tasks.yaml** - Define code review workflow tasks with descriptions and dependencies

   **crew.yaml** - Configure crew behavior, process type, and settings

    **Key Features:**
    - **Pure YAML Configuration** - No hardcoded agents or tasks
    - **Dynamic Tool Assignment** - Code analysis tools assigned based on YAML config
    - **Flexible Review Workflows** - Define complex code review dependencies
    - **Multiple Process Types** - Sequential or hierarchical execution
    - **Template Variables** - Dynamic code review parameter substitution

    **Available Code Review Tools:**
    - `security_analyzer_tool` - Security vulnerability analysis
    - `performance_analyzer_tool` - Performance analysis and optimization
    - `code_quality_tool` - Code quality and best practices assessment

    **Customization:**
    Modify the YAML files in the `/config` directory to customize:
    - Code review agent specializations and capabilities
    - Review task workflows and dependencies
    - Crew execution parameters and review methodologies
    - Tool assignments per review agent type
    """)

# ============================================================================
# FRONTEND: STREAMLIT USER INTERFACE
# ============================================================================
