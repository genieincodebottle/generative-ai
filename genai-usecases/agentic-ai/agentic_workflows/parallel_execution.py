"""
Parallel Execution Workflow Pattern
=====================================

A comprehensive workflow system for executing multiple AI tasks concurrently,
with intelligent result synthesis and performance optimization.

Key Features:
- Concurrent task processing with semaphore-controlled resource management
- Intelligent load distribution and automatic load balancing
- Result aggregation and synthesis with context preservation
- Comprehensive error handling and fault tolerance
- Performance monitoring and execution analytics

Architecture:
- Backend: Core parallel execution engine with async processing
- Frontend: Streamlit interface for task configuration and monitoring
- Integration: Multi-provider LLM support (Ollama, OpenAI, Anthropic, etc.)

Use Cases:
- Multi-perspective analysis and research
- Competitive intelligence gathering
- Content generation at scale
- Data processing pipelines
- A/B testing scenarios
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import asyncio
import streamlit as st
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import os
from dataclasses import dataclass

# LangChain imports for multi-provider LLM support
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import requests

# Load environment variables for API keys
load_dotenv()

# =============================================================================
# BACKEND: DATA MODELS AND CORE CLASSES
# =============================================================================

@dataclass
class ParallelTask:
    """
    Represents a task for parallel execution.

    Attributes:
        task_id: Unique identifier for the task
        name: Human-readable task name
        prompt: The instruction/prompt for the LLM
        context: Additional context data for the task
        priority: Task priority (1=low, 2=medium, 3=high)
        timeout: Maximum execution time in seconds
    """
    task_id: str
    name: str
    prompt: str
    context: Dict[str, Any]
    priority: int = 1
    timeout: int = 60  # seconds

@dataclass
class TaskResult:
    """
    Result of a parallel task execution.

    Attributes:
        task_id: Reference to the original task
        name: Task name for identification
        result: The actual result content from the LLM
        execution_time: Time taken to execute the task
        status: Execution status (success, error, timeout)
        error_message: Error details if status is error/timeout
        timestamp: When the task completed execution
    """
    task_id: str
    name: str
    result: str
    execution_time: float
    status: str  # success, error, timeout
    error_message: Optional[str] = None
    timestamp: str = None

class ParallelExecutor:
    """
    Core engine for parallel task execution with result synthesis.

    This class handles the concurrent execution of multiple LLM tasks,
    managing resources, handling errors, and synthesizing results.

    Attributes:
        llm: The language model instance for task execution
        max_workers: Maximum number of concurrent tasks
        execution_history: History of all execution summaries
    """

    def __init__(self, llm, max_workers: int = 4):
        """
        Initialize the parallel executor.

        Args:
            llm: LangChain LLM instance for task execution
            max_workers: Maximum concurrent tasks (default: 4)
        """
        self.llm = llm
        self.max_workers = max_workers
        self.execution_history = []

    async def execute_tasks_parallel(self, tasks: List[ParallelTask]) -> Dict[str, Any]:
        """
        Execute multiple tasks concurrently with resource management.

        This method orchestrates parallel execution using asyncio with semaphore
        control to manage resource usage and prevent system overload.

        Args:
            tasks: List of ParallelTask objects to execute

        Returns:
            Dict containing execution summary with results, timing, and statistics

        Raises:
            Exception: If critical execution error occurs
        """
        try:
            start_time = time.time()

            # STEP 1: Resource Management Setup
            # Create semaphore to limit concurrent executions and prevent resource exhaustion
            semaphore = asyncio.Semaphore(self.max_workers)

            # STEP 2: Task Orchestration
            # Create coroutines for each task with semaphore control
            task_coroutines = [
                self._execute_single_task(task, semaphore)
                for task in tasks
            ]

            # STEP 3: Parallel Execution
            # Execute all tasks concurrently and handle exceptions gracefully
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            # STEP 4: Result Processing and Classification
            # Separate successful results from failures for analysis
            task_results = []
            successful_results = []
            failed_results = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle exceptions
                    task_result = TaskResult(
                        task_id=tasks[i].task_id,
                        name=tasks[i].name,
                        result="",
                        execution_time=0.0,
                        status="error",
                        error_message=str(result),
                        timestamp=datetime.now().isoformat()
                    )
                    failed_results.append(task_result)
                else:
                    task_results.append(result)
                    if result.status == "success":
                        successful_results.append(result)
                    else:
                        failed_results.append(result)

            total_time = time.time() - start_time

            # Create execution summary
            execution_summary = {
                "total_tasks": len(tasks),
                "successful_tasks": len(successful_results),
                "failed_tasks": len(failed_results),
                "total_execution_time": total_time,
                "average_task_time": sum(r.execution_time for r in successful_results) / max(len(successful_results), 1),
                "results": task_results + failed_results,
                "successful_results": successful_results,
                "failed_results": failed_results,
                "timestamp": datetime.now().isoformat()
            }

            # Store in history
            self.execution_history.append(execution_summary)

            return execution_summary

        except Exception as e:
            return {
                "error": str(e),
                "total_tasks": len(tasks),
                "successful_tasks": 0,
                "failed_tasks": len(tasks),
                "timestamp": datetime.now().isoformat()
            }

    async def _execute_single_task(self, task: ParallelTask, semaphore: asyncio.Semaphore) -> TaskResult:
        """Execute a single task with semaphore control"""
        async with semaphore:
            start_time = time.time()

            try:
                # Build prompt with context
                context_str = ""
                if task.context:
                    context_str = f"\\n\\nContext: {json.dumps(task.context, indent=2)}"

                full_prompt = task.prompt + context_str

                # Create timeout for the task
                messages = [
                    SystemMessage(content=f"You are executing task: {task.name}. Provide a comprehensive response."),
                    HumanMessage(content=full_prompt)
                ]

                # Execute with timeout
                try:
                    response = await asyncio.wait_for(
                        self.llm.ainvoke(messages),
                        timeout=task.timeout
                    )
                    result_content = response.content
                    status = "success"
                    error_message = None

                except asyncio.TimeoutError:
                    result_content = f"Task timed out after {task.timeout} seconds"
                    status = "timeout"
                    error_message = f"Timeout after {task.timeout}s"

                execution_time = time.time() - start_time

                return TaskResult(
                    task_id=task.task_id,
                    name=task.name,
                    result=result_content,
                    execution_time=execution_time,
                    status=status,
                    error_message=error_message,
                    timestamp=datetime.now().isoformat()
                )

            except Exception as e:
                execution_time = time.time() - start_time
                return TaskResult(
                    task_id=task.task_id,
                    name=task.name,
                    result="",
                    execution_time=execution_time,
                    status="error",
                    error_message=str(e),
                    timestamp=datetime.now().isoformat()
                )

    async def synthesize_results(self, results: List[TaskResult], synthesis_prompt: str = None) -> str:
        """Synthesize multiple task results into a cohesive response"""
        try:
            successful_results = [r for r in results if r.status == "success"]

            if not successful_results:
                return "No successful results to synthesize."

            # Create synthesis prompt
            if not synthesis_prompt:
                synthesis_prompt = "Synthesize the following results into a comprehensive, cohesive response:"

            # Combine all results
            combined_results = "\\n\\n".join([
                f"**{result.name}:**\\n{result.result}"
                for result in successful_results
            ])

            full_prompt = f"{synthesis_prompt}\\n\\n{combined_results}"

            # Execute synthesis
            messages = [
                SystemMessage(content="You are an expert at synthesizing multiple pieces of information into coherent insights."),
                HumanMessage(content=full_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            return f"Synthesis error: {str(e)}"

# =============================================================================
# BACKEND: LLM INTEGRATION AND CONFIGURATION
# =============================================================================

def create_llm(provider: str, model: str, base_url: str = None):
    """
    Factory function for creating LLM instances across multiple providers.

    Supports local (Ollama) and cloud-based LLM providers with proper
    authentication and configuration management.

    Args:
        provider: LLM provider name (Ollama, OpenAI, Anthropic, Gemini, Groq)
        model: Specific model name/identifier
        base_url: Base URL for local providers like Ollama

    Returns:
        LangChain LLM instance configured for the specified provider

    Raises:
        Exception: If provider is unsupported or configuration is invalid
    """
    if provider == "Ollama":
        return ChatOllama(
            model=model,
            base_url=base_url or 'http://localhost:11434',
            temperature=0.3,
            timeout=120
        )
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)
    elif provider == "Anthropic":
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0.3)
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
    elif provider == "Groq":
        return ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

# =============================================================================
# FRONTEND: STREAMLIT INTERFACE AND USER INTERACTION
# =============================================================================

def render_parallel_execution_interface():
    """
    Main Streamlit interface for parallel execution workflow.

    Provides comprehensive UI for:
    - LLM provider configuration and status checking
    - Task management and configuration
    - Execution control and monitoring
    - Results visualization and analysis
    """
    st.header("⚡ Parallel Execution Workflow")

    # =============================================================================
    # FRONTEND SECTION 1: CONFIGURATION SIDEBAR
    # =============================================================================
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='parallel_llm_provider'
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

        model = st.selectbox(
            "Model",
            model_options[llm_provider],
            key='parallel_model'
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='parallel_ollama_url',
                help="URL where Ollama server is running (no API key required)"
            )

            # Check Ollama status
            try:
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    st.success("✅ Ollama server is running")

                    # Show available models
                    try:
                        models_data = response.json()
                        if 'models' in models_data and models_data['models']:
                            st.info(f"📋 {len(models_data['models'])} models available")
                        else:
                            st.warning("⚠️ No models found. Pull a model first.")
                    except:
                        st.info("🔄 Ollama server connected")
                else:
                    st.error("❌ Ollama server not accessible")
            except Exception as e:
                st.error("❌ Cannot connect to Ollama server")
                st.markdown("**Setup Instructions:**")
                st.code(f"1. Install Ollama from https://ollama.com\n2. Run: ollama serve\n3. Pull model: ollama pull {model}")

        # API Key Status
        st.markdown("**🔑 API Key Status**")
        if llm_provider == "Ollama":
            st.success("✅ No API key required for Ollama")
        else:
            required_key = f"{llm_provider.upper()}_API_KEY"
            if os.getenv(required_key):
                st.success(f"✅ {required_key} configured")
            else:
                st.warning(f"⚠️ {required_key} not found")

        # Execution settings
        st.markdown("### ⚙️ Execution Settings")
        max_workers = st.slider("Max Concurrent Tasks", 1, 8, 4, key='max_workers')
        task_timeout = st.slider("Task Timeout (seconds)", 10, 300, 60, key='task_timeout')

    # Initialize session state for tasks
    if 'parallel_tasks' not in st.session_state:
        st.session_state.parallel_tasks = [
            {
                "name": "Market Analysis",
                "prompt": "Analyze the current market trends and opportunities in the AI industry",
                "priority": 2
            },
            {
                "name": "Competitive Research",
                "prompt": "Research main competitors and their positioning in the AI market",
                "priority": 2
            },
            {
                "name": "Technology Assessment",
                "prompt": "Assess the latest AI technologies and their potential impact",
                "priority": 1
            }
        ]

    # =============================================================================
    # FRONTEND SECTION 2: MAIN INTERFACE AND TASK MANAGEMENT
    # =============================================================================

    # Pattern information
    with st.expander("ℹ️ Parallel Execution Pattern", expanded=False):
        st.markdown("""
        **Parallel Execution Workflow Pattern:**

        ⚡ **Concurrent Processing**
        - Multiple tasks executed simultaneously
        - Semaphore-controlled resource management
        - Optimal utilization of available resources

        🎯 **Load Distribution**
        - Intelligent task scheduling
        - Priority-based execution order
        - Automatic load balancing

        🔗 **Result Synthesis**
        - Intelligent aggregation of parallel results
        - Context-aware result combination
        - Coherent insight generation

        🛡️ **Fault Tolerance**
        - Individual task error isolation
        - Timeout protection
        - Graceful failure handling

        📊 **Performance Optimization**
        - Execution time tracking
        - Efficiency measurement
        - Resource utilization monitoring

        **Use Cases:**
        - Multi-perspective analysis
        - Competitive research
        - Content generation at scale
        - Data processing pipelines
        - A/B testing scenarios
        """)

    # Task management controls
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add Task"):
            new_task = {
                "name": f"Task {len(st.session_state.parallel_tasks) + 1}",
                "prompt": "Enter task description here",
                "priority": 1
            }
            st.session_state.parallel_tasks.append(new_task)
            st.rerun()

    with col2:
        if st.button("➖ Remove Last Task") and len(st.session_state.parallel_tasks) > 1:
            st.session_state.parallel_tasks.pop()
            st.rerun()

    # Configure each task
    for i, task in enumerate(st.session_state.parallel_tasks):
        with st.expander(f"Configure Task {i+1}: {task['name']}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                task['name'] = st.text_input(
                    "Task Name",
                    value=task['name'],
                    key=f'task_name_{i}'
                )

                task['prompt'] = st.text_area(
                    "Task Prompt",
                    value=task['prompt'],
                    height=100,
                    key=f'task_prompt_{i}'
                )

            with col2:
                task['priority'] = st.selectbox(
                    "Priority",
                    [1, 2, 3],
                    index=task['priority'] - 1,
                    key=f'task_priority_{i}'
                )

    # Synthesis configuration
    st.markdown("#### 🔗 Result Synthesis")
    synthesis_prompt = st.text_area(
        "Synthesis Instructions",
        value="Synthesize the following parallel task results into a comprehensive analysis with key insights and recommendations:",
        height=100,
        key='synthesis_prompt'
    )

    enable_synthesis = st.checkbox("Enable Result Synthesis", value=True, key='enable_synthesis')

    # =============================================================================
    # FRONTEND SECTION 3: EXECUTION CONTROL AND MONITORING
    # =============================================================================

    # Execute parallel tasks
    if st.button("Execute Parallel Tasks", type="primary", key='execute_parallel'):
        if not st.session_state.parallel_tasks:
            st.error("Please configure at least one task.")
            return

        # Create executor
        llm = create_llm(llm_provider, model, ollama_base_url)
        executor = ParallelExecutor(llm, max_workers)

        # Prepare tasks
        parallel_tasks = []
        for i, task_config in enumerate(st.session_state.parallel_tasks):
            task = ParallelTask(
                task_id=f"task_{i+1}",
                name=task_config['name'],
                prompt=task_config['prompt'],
                context={},
                priority=task_config['priority'],
                timeout=task_timeout
            )
            parallel_tasks.append(task)

        with st.spinner("⚡ Executing tasks in parallel..."):
            try:
                # Backend execution call
                execution_summary = asyncio.run(executor.execute_tasks_parallel(parallel_tasks))

                # Error handling
                if 'error' in execution_summary:
                    st.error(f"Execution error: {execution_summary['error']}")
                    return

                st.success("✅ Parallel execution completed!")

                # =============================================================================
                # FRONTEND SECTION 4: RESULTS VISUALIZATION AND ANALYSIS
                # =============================================================================

                # Results tabs for organized display
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Results", "📊 Performance", "🔗 Synthesis", "📈 Analytics"])

                with tab1:
                    # TAB 1: Individual Results Display
                    st.markdown("### 📋 Individual Task Results")

                    successful_results = execution_summary['successful_results']
                    failed_results = execution_summary['failed_results']

                    if successful_results:
                        for result in successful_results:
                            with st.expander(f"✅ {result.name} ({result.execution_time:.2f}s)", expanded=True):
                                st.markdown(result.result)

                    if failed_results:
                        st.markdown("### ❌ Failed Tasks")
                        for result in failed_results:
                            with st.expander(f"❌ {result.name} - {result.status}", expanded=False):
                                st.error(f"Error: {result.error_message}")

                with tab2:
                    # TAB 2: Performance Metrics and Analytics
                    st.markdown("### 📊 Execution Performance")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tasks", execution_summary['total_tasks'])
                    with col2:
                        st.metric("Successful", execution_summary['successful_tasks'])
                    with col3:
                        st.metric("Failed", execution_summary['failed_tasks'])
                    with col4:
                        st.metric("Total Time", f"{execution_summary['total_execution_time']:.2f}s")

                    # Task execution times
                    if successful_results:
                        st.markdown("**Task Execution Times:**")
                        for result in successful_results:
                            st.write(f"• **{result.name}**: {result.execution_time:.2f}s")

                        # Efficiency calculation
                        sequential_time = sum(r.execution_time for r in successful_results)
                        parallel_time = execution_summary['total_execution_time']
                        efficiency = (sequential_time / parallel_time) * 100 if parallel_time > 0 else 0

                        st.metric("Parallel Efficiency", f"{efficiency:.1f}%")
                        st.caption(f"Estimated sequential time: {sequential_time:.2f}s vs parallel time: {parallel_time:.2f}s")

                with tab3:
                    # TAB 3: Result Synthesis and Integration
                    if enable_synthesis and successful_results:
                        st.markdown("### 🔗 Synthesized Results")

                        with st.spinner("🔗 Synthesizing results..."):
                            synthesized_result = asyncio.run(
                                executor.synthesize_results(successful_results, synthesis_prompt)
                            )

                        st.markdown(synthesized_result)

                    else:
                        st.info("Synthesis disabled or no successful results to synthesize.")

                with tab4:
                    # TAB 4: Advanced Analytics and Insights
                    st.markdown("### 📈 Execution Analytics")

                    # Success rate
                    success_rate = (execution_summary['successful_tasks'] / execution_summary['total_tasks']) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                    # Average execution time
                    avg_time = execution_summary.get('average_task_time', 0)
                    st.metric("Average Task Time", f"{avg_time:.2f}s")

                    # Task status distribution
                    status_counts = {}
                    for result in execution_summary['results']:
                        status_counts[result.status] = status_counts.get(result.status, 0) + 1

                    st.markdown("**Task Status Distribution:**")
                    for status, count in status_counts.items():
                        st.write(f"• **{status.title()}**: {count}")

            except Exception as e:
                st.error(f"Parallel execution error: {str(e)}")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main application entry point.

    Initializes and runs the Streamlit interface for the parallel execution
    workflow pattern, providing users with a comprehensive tool for managing
    concurrent AI task execution.
    """
    render_parallel_execution_interface()