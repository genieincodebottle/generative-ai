"""
Tool Orchestration Workflow Pattern
====================================

A sophisticated tool management and coordination system for multi-agent workflows,
featuring dynamic tool selection, intelligent routing, and automated orchestration.

Key Features:
- Dynamic tool registry with performance-based selection
- LLM-powered workflow planning and optimization
- Dependency-aware execution with context preservation
- Real-time performance monitoring and analytics
- Type-safe tool integration with standardized interfaces

Architecture:
- Backend: Tool registry, orchestration engine, and execution management
- Frontend: Interactive Streamlit interface for workflow configuration
- Integration: Multi-provider LLM support with intelligent planning
- Patterns: Sequential, parallel, and dependency-based execution flows

Use Cases:
- Complex data processing pipelines with multiple tools
- Multi-step analysis workflows with context flow
- Content generation systems with tool composition
- API orchestration with intelligent routing
- DevOps automation with performance monitoring
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dotenv import load_dotenv
import os
from dataclasses import dataclass
from enum import Enum

# LangChain core components for LLM integration
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from services.llm_text import message_text
from services.providers import create_llm

# Multi-provider LLM support with Ollama integration
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

# Load environment configuration
load_dotenv()

# =============================================================================
# BACKEND: DATA MODELS AND ENUMERATIONS
# =============================================================================

class ToolType(Enum):
    """
    Enumeration of available tool types for categorization and filtering.

    These types enable intelligent tool selection and routing based on
    workflow requirements and execution patterns.
    """
    ANALYSIS = "analysis"
    GENERATION = "generation"
    SEARCH = "search"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    STORAGE = "storage"
    COMPUTATION = "computation"

@dataclass
class Tool:
    """
    Comprehensive tool representation for the orchestration system.

    Encapsulates all tool metadata including execution function, dependencies,
    performance metrics, and configuration parameters for intelligent routing
    and optimization.

    Attributes:
        tool_id: Unique identifier for the tool
        name: Human-readable tool name
        description: Detailed tool description and capabilities
        tool_type: Category of tool (ToolType enum)
        function: Callable function implementing tool logic
        dependencies: List of required tool dependencies
        parameters: Expected input parameters with types
        performance_metrics: Runtime performance data
        active: Whether tool is available for execution
    """
    tool_id: str
    name: str
    description: str
    tool_type: ToolType
    function: Callable
    dependencies: List[str] = None
    parameters: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    active: bool = True

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}
        if self.performance_metrics is None:
            self.performance_metrics = {"avg_execution_time": 0.0, "success_rate": 100.0}

@dataclass
class ToolExecution:
    """
    Detailed execution record for tool invocations.

    Captures comprehensive execution metadata including timing, status,
    input/output data, and error information for analytics and debugging.

    Attributes:
        execution_id: Unique identifier for this execution
        tool_id: ID of the executed tool
        input_data: Parameters passed to the tool
        output_data: Results returned by the tool
        execution_time: Time taken for execution in seconds
        status: Execution status (pending, running, completed, failed)
        error_message: Error details if execution failed
        timestamp: ISO timestamp of execution start
    """
    execution_id: str
    tool_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = None
    execution_time: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    error_message: str = None
    timestamp: str = None

# =============================================================================
# BACKEND: TOOL REGISTRY AND MANAGEMENT
# =============================================================================

class ToolRegistry:
    """
    Comprehensive tool registry with dynamic management capabilities.

    Manages tool registration, discovery, performance tracking, and provides
    intelligent tool selection based on type, dependencies, and performance
    metrics.

    Features:
    - Dynamic tool registration and discovery
    - Performance-based tool selection
    - Dependency resolution and validation
    - Search and filtering capabilities
    - Execution history tracking
    """

    def __init__(self):
        """Initialize tool registry with default tools."""
        self.tools = {}
        self.execution_history = []
        self._register_default_tools()

    def register_tool(self, tool: Tool):
        """
        Register a new tool in the registry.

        Args:
            tool: Tool instance to register with complete metadata
        """
        self.tools[tool.tool_id] = tool

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """
        Retrieve a tool by its unique identifier.

        Args:
            tool_id: Unique identifier of the tool

        Returns:
            Tool instance if found, None otherwise
        """
        return self.tools.get(tool_id)

    def get_tools_by_type(self, tool_type: ToolType) -> List[Tool]:
        """
        Get all active tools of a specific type.

        Args:
            tool_type: Type of tools to retrieve (ToolType enum)

        Returns:
            List of active tools matching the specified type
        """
        return [tool for tool in self.tools.values() if tool.tool_type == tool_type and tool.active]

    def search_tools(self, query: str) -> List[Tool]:
        """
        Search tools by name or description using text matching.

        Args:
            query: Search query string

        Returns:
            List of tools matching the search criteria
        """
        query_lower = query.lower()
        return [
            tool for tool in self.tools.values()
            if query_lower in tool.name.lower() or query_lower in tool.description.lower()
        ]

    def _register_default_tools(self):
        """
        Register comprehensive set of default tools for demonstration.

        Creates a diverse toolkit covering analysis, generation, transformation,
        and validation capabilities with realistic mock implementations.
        """
        # -------------------------------------------------------------------------
        # Analysis Tools
        # -------------------------------------------------------------------------

        self.register_tool(Tool(
            tool_id="text_analyzer",
            name="Text Analyzer",
            description="Analyzes text for sentiment, topics, and key insights",
            tool_type=ToolType.ANALYSIS,
            function=self._text_analysis_tool,
            parameters={"text": "string", "analysis_type": "string"}
        ))

        self.register_tool(Tool(
            tool_id="data_summarizer",
            name="Data Summarizer",
            description="Summarizes structured data and generates insights",
            tool_type=ToolType.ANALYSIS,
            function=self._data_summarization_tool,
            parameters={"data": "dict", "summary_type": "string"}
        ))

        # -------------------------------------------------------------------------
        # Generation Tools
        # -------------------------------------------------------------------------

        self.register_tool(Tool(
            tool_id="content_generator",
            name="Content Generator",
            description="Generates various types of content based on prompts",
            tool_type=ToolType.GENERATION,
            function=self._content_generation_tool,
            parameters={"prompt": "string", "content_type": "string"}
        ))

        self.register_tool(Tool(
            tool_id="code_generator",
            name="Code Generator",
            description="Generates code in various programming languages",
            tool_type=ToolType.GENERATION,
            function=self._code_generation_tool,
            parameters={"requirements": "string", "language": "string"}
        ))

        # -------------------------------------------------------------------------
        # Transformation Tools
        # -------------------------------------------------------------------------

        self.register_tool(Tool(
            tool_id="format_converter",
            name="Format Converter",
            description="Converts data between different formats",
            tool_type=ToolType.TRANSFORMATION,
            function=self._format_conversion_tool,
            parameters={"data": "any", "source_format": "string", "target_format": "string"}
        ))

        # -------------------------------------------------------------------------
        # Validation Tools
        # -------------------------------------------------------------------------

        self.register_tool(Tool(
            tool_id="quality_checker",
            name="Quality Checker",
            description="Validates content quality and compliance",
            tool_type=ToolType.VALIDATION,
            function=self._quality_validation_tool,
            parameters={"content": "string", "criteria": "list"}
        ))

    # -------------------------------------------------------------------------
    # Mock Tool Implementations for Demonstration
    # -------------------------------------------------------------------------

    async def _text_analysis_tool(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Mock text analysis tool with simulated AI capabilities.

        Args:
            text: Input text to analyze
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results with sentiment, topics, and insights
        """
        await asyncio.sleep(0.5)  # Simulate realistic processing time
        return {
            "sentiment": "positive",
            "topics": ["technology", "innovation", "future"],
            "key_phrases": ["artificial intelligence", "machine learning", "automation"],
            "word_count": len(text.split()),
            "analysis_type": analysis_type,
            "confidence_score": 0.95
        }

    async def _data_summarization_tool(self, data: Dict[str, Any], summary_type: str = "executive") -> Dict[str, Any]:
        """Mock data summarization tool"""
        await asyncio.sleep(0.8)
        return {
            "summary": f"Data contains {len(data)} key elements with {summary_type} level insights",
            "key_metrics": {"total_items": len(data), "summary_type": summary_type},
            "insights": ["Data shows consistent patterns", "Notable trends identified"]
        }

    async def _content_generation_tool(self, prompt: str, content_type: str = "article") -> Dict[str, Any]:
        """Mock content generation tool"""
        await asyncio.sleep(1.0)
        return {
            "content": f"Generated {content_type} based on: {prompt[:50]}...",
            "content_type": content_type,
            "word_count": 250,
            "quality_score": 85
        }

    async def _code_generation_tool(self, requirements: str, language: str = "python") -> Dict[str, Any]:
        """Mock code generation tool"""
        await asyncio.sleep(1.2)
        return {
            "code": f"# Generated {language} code for: {requirements}\nprint('Hello, World!')",
            "language": language,
            "lines_of_code": 10,
            "complexity_score": "low"
        }

    async def _format_conversion_tool(self, data: Any, source_format: str, target_format: str) -> Dict[str, Any]:
        """Mock format conversion tool"""
        await asyncio.sleep(0.3)
        return {
            "converted_data": f"Data converted from {source_format} to {target_format}",
            "source_format": source_format,
            "target_format": target_format,
            "conversion_success": True
        }

    async def _quality_validation_tool(self, content: str, criteria: List[str]) -> Dict[str, Any]:
        """Mock quality validation tool"""
        await asyncio.sleep(0.6)
        return {
            "validation_passed": True,
            "quality_score": 92,
            "criteria_met": criteria,
            "recommendations": ["Minor improvements suggested", "Overall excellent quality"]
        }

# =============================================================================
# BACKEND: TOOL ORCHESTRATION ENGINE
# =============================================================================

class ToolOrchestrator:
    """
    Advanced tool orchestration engine for intelligent workflow management.

    Coordinates tool execution with LLM-powered planning, dependency resolution,
    context preservation, and comprehensive performance monitoring.

    Key Features:
    - LLM-driven workflow planning and optimization
    - Dependency-aware execution ordering
    - Context variable resolution between steps
    - Real-time performance tracking and analytics
    - Error handling with graceful degradation
    - Parallel and sequential execution patterns

    Args:
        llm: Language model for intelligent planning
        tool_registry: Registry containing available tools
    """

    def __init__(self, llm, tool_registry: ToolRegistry):
        """Initialize orchestrator with LLM and tool registry."""
        self.llm = llm
        self.tool_registry = tool_registry
        self.execution_queue = []
        self.active_executions = {}
        self.workflow_state = {}

    async def execute_tool(self, tool_id: str, input_data: Dict[str, Any]) -> ToolExecution:
        """Execute a single tool"""
        tool = self.tool_registry.get_tool(tool_id)
        if not tool:
            raise ValueError(f"Tool {tool_id} not found")

        execution = ToolExecution(
            execution_id=f"exec_{int(time.time())}_{tool_id}",
            tool_id=tool_id,
            input_data=input_data,
            timestamp=datetime.now().isoformat()
        )

        try:
            execution.status = "running"
            start_time = time.time()

            # Execute tool function
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**input_data)
            else:
                result = tool.function(**input_data)

            execution_time = time.time() - start_time
            execution.execution_time = execution_time
            execution.output_data = result
            execution.status = "completed"

            # Update tool performance metrics
            tool.performance_metrics["avg_execution_time"] = (
                tool.performance_metrics["avg_execution_time"] + execution_time
            ) / 2

        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            # Update tool success rate
            current_rate = tool.performance_metrics["success_rate"]
            tool.performance_metrics["success_rate"] = max(0, current_rate - 5)

        self.tool_registry.execution_history.append(execution)
        return execution

    async def execute_workflow(self, workflow_plan: List[Dict[str, Any]]) -> List[ToolExecution]:
        """Execute a planned workflow of tools"""
        results = []
        workflow_context = {}

        for step in workflow_plan:
            tool_id = step["tool_id"]
            input_data = step["input_data"]

            # Resolve context variables in input data
            resolved_input = self._resolve_context_variables(input_data, workflow_context)

            # Execute tool
            execution = await self.execute_tool(tool_id, resolved_input)
            results.append(execution)

            # Update workflow context with results
            if execution.status == "completed":
                workflow_context[f"step_{len(results)}_output"] = execution.output_data
                workflow_context[f"{tool_id}_result"] = execution.output_data

        return results

    def _resolve_context_variables(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve context variables in input data"""
        resolved = {}

        for key, value in input_data.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Context variable reference
                var_name = value[2:-1]
                resolved[key] = context.get(var_name, value)
            else:
                resolved[key] = value

        return resolved

    async def plan_workflow(self, objective: str, available_tools: List[str] = None) -> List[Dict[str, Any]]:
        """Use LLM to plan a workflow for achieving an objective"""
        if available_tools is None:
            available_tools = list(self.tool_registry.tools.keys())

        # Get tool descriptions
        tool_descriptions = []
        for tool_id in available_tools:
            tool = self.tool_registry.get_tool(tool_id)
            if tool:
                tool_descriptions.append({
                    "id": tool_id,
                    "name": tool.name,
                    "description": tool.description,
                    "type": tool.tool_type.value,
                    "parameters": tool.parameters
                })

        planning_prompt = f"""
        Objective: {objective}

        Available Tools:
        {json.dumps(tool_descriptions, indent=2)}

        Plan a workflow to achieve the objective using the available tools.
        Return a JSON array where each element has:
        - tool_id: The ID of the tool to use
        - input_data: The input parameters for the tool
        - description: Brief description of this step

        Use context variables like ${{step_1_output}} to reference previous step outputs.
        """

        messages = [
            SystemMessage(content="You are an expert workflow planner. Create efficient tool execution plans."),
            HumanMessage(content=planning_prompt)
        ]

        response = await self.llm.ainvoke(messages)

        try:
            # Extract JSON from response
            response_text = message_text(response)
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx >= 0 and end_idx > start_idx:
                workflow_plan = json.loads(response_text[start_idx:end_idx])
                return workflow_plan
            else:
                # Fallback: create simple single-tool workflow
                return [{"tool_id": available_tools[0], "input_data": {"objective": objective}, "description": "Single tool execution"}]

        except json.JSONDecodeError:
            # Fallback plan
            return [{"tool_id": available_tools[0], "input_data": {"objective": objective}, "description": "Fallback execution"}]

    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        total_executions = len(self.tool_registry.execution_history)
        successful_executions = len([e for e in self.tool_registry.execution_history if e.status == "completed"])

        return {
            "total_tools": len(self.tool_registry.tools),
            "active_tools": len([t for t in self.tool_registry.tools.values() if t.active]),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "recent_executions": self.tool_registry.execution_history[-5:] if self.tool_registry.execution_history else []
        }

# =============================================================================
# BACKEND: LLM PROVIDER CONFIGURATION
# =============================================================================



# =============================================================================
# FRONTEND: STREAMLIT INTERFACE AND USER INTERACTIONS
# =============================================================================
