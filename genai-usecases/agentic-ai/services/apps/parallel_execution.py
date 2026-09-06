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
from services.llm_text import message_text
from services.providers import create_llm

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
                    result_content = message_text(response)
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
            return message_text(response)

        except Exception as e:
            return f"Synthesis error: {str(e)}"

# =============================================================================
# BACKEND: LLM INTEGRATION AND CONFIGURATION
# =============================================================================



# =============================================================================
# FRONTEND: STREAMLIT INTERFACE AND USER INTERACTION
# =============================================================================
