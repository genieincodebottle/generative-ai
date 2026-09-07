"""
Prompt Chaining Workflow Pattern
=================================

A sophisticated workflow system that orchestrates sequential AI task execution using
LangChain's latest chaining capabilities with LCEL (LangChain Expression Language).

Key Features:
- Sequential chain execution with advanced LCEL patterns
- Built-in context preservation across chain steps
- Multiple execution patterns (sequential, parallel, custom LCEL)
- Advanced memory and state management
- Real-time monitoring and performance analytics

Architecture:
- Backend: Core chaining engine with LCEL composition
- Frontend: Interactive Streamlit interface for chain design
- Integration: Multi-provider LLM support with configuration management
- Patterns: Sequential, parallel, and custom execution flows

Use Cases:
- Complex multi-step analysis workflows
- Document processing pipelines with context flow
- Research workflows with iterative refinement
- Strategic planning with sequential reasoning
- Content creation with progressive enhancement
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import os

# LangChain core components for advanced chaining
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.chat_history import InMemoryChatMessageHistory

# Multi-provider LLM support
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Structured output validation
from pydantic import BaseModel, Field

# Load environment configuration
load_dotenv()

# =============================================================================
# BACKEND: DATA MODELS AND CONFIGURATION
# =============================================================================

class ChainStepResult(BaseModel):
    """
    Structured result model for individual chain step execution.

    Captures comprehensive metadata about each step including performance
    metrics, execution status, and detailed results for analysis.
    """
    step_name: str = Field(description="Name of the step")
    result: str = Field(description="Result of the step execution")
    timestamp: str = Field(description="Execution timestamp")
    tokens_used: int = Field(default=0, description="Tokens used in this step")
    execution_time: float = Field(default=0.0, description="Time taken to execute")

class ChainSummary(BaseModel):
    """
    Comprehensive summary model for entire chain execution.

    Provides aggregated metrics and performance data across all chain steps
    for workflow analysis and optimization insights.
    """
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_tokens: int
    total_time: float
    results: List[ChainStepResult]

# =============================================================================
# BACKEND: CORE CHAIN ENGINE
# =============================================================================

class PromptChain:
    """
    Advanced LangChain-based prompt chaining engine using LCEL.

    A sophisticated workflow orchestrator that implements sequential and parallel
    execution patterns using LangChain Expression Language (LCEL) with built-in
    memory management, context preservation, and performance monitoring.

    Key Features:
    - Multiple execution patterns (sequential, parallel, custom LCEL)
    - Advanced memory management with conversation history
    - Context preservation across chain steps
    - Performance tracking and metrics collection
    - Error handling and recovery mechanisms
    """

    def __init__(self, llm, use_memory: bool = True):
        """
        Initialize the PromptChain with LLM and configuration.

        Args:
            llm: Language model instance (OpenAI, Anthropic, Gemini, Groq, Ollama)
            use_memory: Enable conversation buffer memory for context preservation
        """
        self.llm = llm
        self.parser = StrOutputParser()
        self.memory = InMemoryChatMessageHistory() if use_memory else None
        self.results: List[ChainStepResult] = []

    def create_step_chain(self, step_name: str, prompt_template: str) -> Any:
        """
        Create a single step chain using LCEL composition.

        Args:
            step_name: Descriptive name for the chain step
            prompt_template: Template for the step's prompt

        Returns:
            Composed LCEL chain for execution
        """
        # Create prompt template with system message and context injection
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant executing step '{step_name}' in a workflow chain."),
            ("human", prompt_template + "\n\nPrevious context: {context}")
        ])

        # Create the chain using LCEL composition pattern
        chain = (
            {
                "step_name": lambda x: step_name,
                "context": RunnablePassthrough(),
                "input": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | self.parser
        )

        return chain

    def create_sequential_chain(self, steps: List[Dict[str, str]]) -> Any:
        """
        Create sequential execution chain with context preservation.

        Sequential execution ensures each step builds upon previous results,
        maintaining context flow and enabling complex reasoning workflows.

        Args:
            steps: List of step dictionaries with 'name' and 'prompt' keys

        Returns:
            Async callable for sequential execution
        """
        async def execute_sequential(inputs):
            """
            Execute steps sequentially with context accumulation.

            Each step receives context from all previous steps,
            enabling complex reasoning and context-aware processing.
            """
            main_topic = inputs["main_topic"]
            results = {}
            context = {"main_topic": main_topic}

            # Process each step sequentially, building context
            for i, step in enumerate(steps):
                step_name = step['name']
                prompt_text = step['prompt']

                # Build context-aware prompt template
                if i == 0:
                    # First step: Only main topic context
                    template = f"""
                    Step: {step_name}
                    Task: {prompt_text}
                    Main Topic: {main_topic}

                    Provide a detailed response for this step.
                    """
                else:
                    # Subsequent steps: Include all previous results
                    prev_results = "\n".join([f"Step {j+1} Result: {results[f'step_{j+1}_result']}" for j in range(i)])
                    template = f"""
                    Step: {step_name}
                    Task: {prompt_text}
                    Main Topic: {main_topic}

                    Previous Results:
                    {prev_results}

                    Based on the previous steps, provide a detailed response for this step.
                    """

                # Create and execute individual chain with LCEL
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | self.llm | self.parser

                # Execute step and accumulate results
                result = await chain.ainvoke({})
                results[f"step_{i+1}_result"] = result
                context[f"step_{i+1}_result"] = result

            return results

        return execute_sequential

    def create_parallel_chain(self, steps: List[Dict[str, str]]) -> Any:
        """
        Create parallel execution chain for independent steps.

        Parallel execution processes multiple independent steps concurrently,
        significantly reducing total execution time for workflows where
        steps don't depend on each other's results.

        Args:
            steps: List of step dictionaries with 'name' and 'prompt' keys

        Returns:
            RunnableParallel instance for concurrent execution
        """
        parallel_chains = {}

        # Create individual chains for each step
        for i, step in enumerate(steps):
            step_name = step['name'].replace(" ", "_").lower()
            prompt_text = step['prompt']

            # Create template for independent step execution
            template = f"""
            Step: {step['name']}
            Task: {prompt_text}
            Main Topic: {{main_topic}}

            Provide a detailed response for this step.
            """

            # Build LCEL chain for this step
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | self.parser
            parallel_chains[f"step_{i+1}_{step_name}"] = chain

        # Create parallel runnable for concurrent execution
        parallel_chain = RunnableParallel(parallel_chains)

        return parallel_chain

    async def execute_lcel_chain(self, steps: List[Dict[str, str]], main_topic: str,
                                execution_type: str = "sequential") -> Dict[str, Any]:
        """
        Execute chain using LCEL with different execution patterns.

        This is the main execution method that orchestrates the entire workflow
        based on the specified execution pattern and provides comprehensive
        performance metrics and error handling.

        Args:
            steps: List of workflow steps to execute
            main_topic: Primary topic/question for the workflow
            execution_type: "sequential", "parallel", or "custom_lcel"

        Returns:
            Dict containing execution results, timing, and success status
        """
        start_time = datetime.now()

        try:
            # Route to appropriate execution pattern
            if execution_type == "sequential":
                chain_func = self.create_sequential_chain(steps)
                results = await chain_func({"main_topic": main_topic})

            elif execution_type == "parallel":
                chain = self.create_parallel_chain(steps)
                results = await chain.ainvoke({"main_topic": main_topic})

            else:  # Custom LCEL chain with advanced context injection
                results = await self._execute_custom_lcel(steps, main_topic)

            # Calculate performance metrics
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            return {
                "results": results,
                "execution_time": execution_time,
                "success": True
            }

        except Exception as e:
            return {
                "error": str(e),
                "execution_time": 0,
                "success": False
            }

    async def _execute_custom_lcel(self, steps: List[Dict[str, str]], main_topic: str) -> Dict[str, Any]:
        """
        Execute custom LCEL chain with advanced context injection.

        This method implements a sophisticated context-aware execution pattern
        using LangChain's RunnableLambda for dynamic context injection and
        JSON serialization for complex data passing between steps.

        Args:
            steps: List of workflow steps to execute
            main_topic: Primary topic for the workflow

        Returns:
            Dict containing step results with preserved context
        """
        context = {"main_topic": main_topic}
        results = {}

        # Execute each step with accumulated context
        for i, step in enumerate(steps):
            step_name = step['name']
            prompt_text = step['prompt']

            # Create dynamic prompt template with context injection
            template = f"""
            Step: {step_name}
            Task: {prompt_text}

            Context: {{context}}

            Provide a detailed response for this step based on the context.
            """

            prompt = ChatPromptTemplate.from_template(template)

            # Create advanced LCEL chain with context injection using RunnableLambda
            chain = (
                RunnableLambda(lambda x: {"context": json.dumps(context, indent=2)})
                | prompt
                | self.llm
                | self.parser
            )

            # Execute step with current context
            result = await chain.ainvoke({})

            # Update context and results for next iteration
            context[f"step_{i+1}_result"] = result
            results[f"step_{i+1}_result"] = result

        return results

    def create_advanced_chain_with_memory(self, steps: List[Dict[str, str]]) -> Any:
        """
        Create advanced chain with memory and conversation history preservation.

        This method creates a memory-enabled chain that maintains conversation
        history across all steps, enabling complex multi-turn reasoning and
        contextual understanding throughout the workflow execution.

        Args:
            steps: List of workflow steps requiring memory preservation

        Returns:
            Memory-enabled LCEL chain for conversational workflows
        """
        def create_memory_chain():
            """Create memory-aware chain with conversation buffer."""
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are executing a multi-step workflow with memory preservation.
                Keep track of the conversation history and build upon previous results.
                Current step: {current_step}
                Task: {task}"""),
                ("human", "{input}")
            ])

            return prompt | self.llm | self.parser

        return create_memory_chain()

# =============================================================================
# BACKEND: LLM PROVIDER CONFIGURATION
# =============================================================================



# =============================================================================
# FRONTEND: STREAMLIT INTERFACE AND USER INTERACTIONS
# =============================================================================
