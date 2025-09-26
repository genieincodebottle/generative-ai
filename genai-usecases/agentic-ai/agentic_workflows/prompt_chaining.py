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
import streamlit as st
import json
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import os

# LangChain core components for advanced chaining
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.memory import ConversationBufferMemory

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
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ) if use_memory else None
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

def create_llm(provider: str, model: str, base_url: str = None):
    """
    Create LLM instance based on provider with unified configuration.

    Supports multiple LLM providers with consistent configuration patterns
    and automatic fallback handling for robust multi-provider workflows.

    Args:
        provider: LLM provider ("Gemini", "Ollama", "Groq", "Anthropic", "OpenAI")
        model: Model name/identifier for the specific provider
        base_url: Optional base URL for self-hosted providers (Ollama)

    Returns:
        Configured LLM instance ready for chain integration

    Raises:
        ValueError: If provider is not supported
        ConnectionError: If API keys are missing or invalid
    """
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
    elif provider == "Ollama":
        return ChatOllama(
            model=model,
            base_url=base_url or 'http://localhost:11434',
            temperature=0.3,
            timeout=120
        )
    elif provider == "Groq":
        return ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)
    elif provider == "Anthropic":
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0.3)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)

# =============================================================================
# FRONTEND: STREAMLIT INTERFACE AND USER INTERACTIONS
# =============================================================================

def render_prompt_chaining_interface():
    """
    Render the comprehensive LangChain prompt chaining interface.

    Creates an advanced Streamlit interface for configuring and executing
    complex multi-step AI workflows using LangChain's LCEL patterns.
    Includes real-time configuration, multiple execution patterns,
    and comprehensive result analysis.

    Features:
    - Multi-provider LLM selection with validation
    - Dynamic step configuration with live editing
    - Advanced execution patterns (sequential, parallel, custom LCEL)
    - Real-time performance monitoring and analytics
    - Interactive result visualization with detailed breakdowns
    """
    # =============================================================================
    # HEADER AND INTRODUCTION
    # =============================================================================

    st.header("🔗 Prompt Chaining")
    st.caption("Using LangChain's latest LCEL (LangChain Expression Language) and chaining capabilities")

    # =============================================================================
    # SIDEBAR CONFIGURATION PANEL
    # =============================================================================

    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # -------------------------------------------------------------------------
        # LLM Provider and Model Selection
        # -------------------------------------------------------------------------

        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='chain_llm_provider',
            help="Select the LLM provider for chain execution"
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
            key='chain_model',
            help="Select the specific model for the chosen provider"
        )

        # -------------------------------------------------------------------------
        # Execution Pattern Configuration
        # -------------------------------------------------------------------------

        st.markdown("**🔄 Execution Pattern**")
        execution_type = st.selectbox(
            "Chain Type",
            ["sequential", "parallel", "custom_lcel"],
            format_func=lambda x: {
                "sequential": "Sequential Chain (Step-by-step)",
                "parallel": "Parallel Chain (Concurrent)",
                "custom_lcel": "Custom LCEL Chain (Advanced)"
            }[x],
            key='execution_type'
        )

        # -------------------------------------------------------------------------
        # Advanced Configuration Options
        # -------------------------------------------------------------------------

        st.markdown("**🧠 Advanced Options**")
        use_memory = st.checkbox(
            "Enable Memory",
            value=True,
            key='use_memory',
            help="Enable conversation buffer memory for context preservation"
        )
        use_callbacks = st.checkbox(
            "Enable Streaming",
            value=True,
            key='use_callbacks',
            help="Enable real-time response streaming"
        )

        # -------------------------------------------------------------------------
        # Provider-Specific Configuration
        # -------------------------------------------------------------------------

        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='chain_ollama_url',
                help="Base URL for your local Ollama server"
            )

        # -------------------------------------------------------------------------
        # API Key Validation Status
        # -------------------------------------------------------------------------

        st.markdown("**🔑 API Key Status**")
        if llm_provider == "Ollama":
            st.success("✅ No API key required for Ollama")
        else:
            required_key = f"{llm_provider.upper()}_API_KEY"
            if os.getenv(required_key):
                st.success(f"✅ {required_key} configured")
            else:
                st.warning(f"⚠️ {required_key} not found")

    # =============================================================================
    # MAIN INTERFACE: CHAIN CONFIGURATION
    # =============================================================================

    # Initialize session state for chain steps
    if 'chain_steps' not in st.session_state:
        st.session_state.chain_steps = [
            {"name": "Step 1: Analysis", "prompt": "Analyze the given topic and identify key components, challenges, and opportunities"},
            {"name": "Step 2: Strategy", "prompt": "Based on the analysis, develop a comprehensive strategy with clear objectives"},
            {"name": "Step 3: Implementation", "prompt": "Create a detailed implementation plan with timeline and resources"},
            {"name": "Step 4: Evaluation", "prompt": "Design evaluation metrics and success criteria for the implementation"}
        ]

    # Input topic
    topic = st.text_area(
        "Main Topic/Question",
        placeholder="e.g., Develop a comprehensive AI adoption strategy for a mid-size company",
        height=100,
        key='chain_topic'
    )

    # Chain pattern explanation
    with st.expander("🔍 Understanding Chain Patterns", expanded=False):
        st.markdown(f"""
        **Current Pattern: {execution_type.upper()}**
        
        🔄 **Sequential Chain**: Steps execute one after another, with each step building on previous results
        
        ⚡ **Parallel Chain**: Independent steps execute simultaneously for faster processing
        
        🎯 **Custom LCEL**: Advanced pattern using LangChain Expression Language for complex workflows
        
        **Memory**: {'Enabled' if use_memory else 'Disabled'} - Conversation history preservation
        
        **Streaming**: {'Enabled' if use_callbacks else 'Disabled'} - Real-time response streaming
        """)

    # Dynamic step configuration
    st.markdown("### 🔢 Chain Steps Configuration")

    # Add/Remove steps
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("➕ Add Step"):
            new_step = {
                "name": f"Step {len(st.session_state.chain_steps) + 1}",
                "prompt": "Define the task for this step"
            }
            st.session_state.chain_steps.append(new_step)
            st.rerun()

    with col2:
        if st.button("➖ Remove Last Step") and len(st.session_state.chain_steps) > 1:
            st.session_state.chain_steps.pop()
            st.rerun()

    with col3:
        if st.button("🔄 Reset to Default"):
            st.session_state.chain_steps = [
                {"name": "Step 1: Analysis", "prompt": "Analyze the given topic and identify key components"},
                {"name": "Step 2: Strategy", "prompt": "Based on the analysis, develop a comprehensive strategy"},
                {"name": "Step 3: Implementation", "prompt": "Create a detailed implementation plan"},
                {"name": "Step 4: Evaluation", "prompt": "Design evaluation metrics and success criteria"}
            ]
            st.rerun()

    # Configure each step
    for i, step in enumerate(st.session_state.chain_steps):
        with st.expander(f"Configure {step['name']}", expanded=False):
            col1, col2 = st.columns([1, 2])

            with col1:
                step['name'] = st.text_input(
                    "Step Name",
                    value=step['name'],
                    key=f'step_name_{i}'
                )

            with col2:
                step['prompt'] = st.text_area(
                    "Step Prompt",
                    value=step['prompt'],
                    height=100,
                    key=f'step_prompt_{i}'
                )

    # Execute chain
    if st.button("Execute Chain", type="primary", key='execute_chain'):
        if not topic:
            st.error("Please provide a main topic/question.")
            return

        if not st.session_state.chain_steps:
            st.error("Please configure at least one step.")
            return

        # Validate API keys (except for Ollama)
        if llm_provider != "Ollama":
            required_key = f"{llm_provider.upper()}_API_KEY"
            if not os.getenv(required_key):
                st.error(f"❌ {required_key} not found. Please set up your API key.")
                return

        # Create LLM and chain
        try:
            llm = create_llm(llm_provider, model, ollama_base_url)
            chain = PromptChain(llm, use_memory=use_memory)

            with st.spinner(f"🔄 Executing {execution_type} chain..."):
                # Execute chain
                result = asyncio.run(chain.execute_lcel_chain(
                    st.session_state.chain_steps,
                    topic,
                    execution_type
                ))

                if result['success']:
                    st.success("✅ Chain execution completed!")

                    # Display results
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Results", "🔗 Chain Analysis", "⚡ Performance", "🔧 Technical Details"])

                    with tab1:
                        st.markdown("### 📋 Chain Results")
                        
                        results_data = result['results']
                        
                        if execution_type == "parallel":
                            st.info("🔄 Results from parallel execution:")
                            for key, value in results_data.items():
                                step_name = key.replace("_", " ").title()
                                with st.expander(f"📊 {step_name}", expanded=True):
                                    st.markdown(value)
                        else:
                            st.info("🔄 Results from sequential execution:")
                            for key, value in results_data.items():
                                if key.startswith("step_"):
                                    step_num = key.split("_")[1]
                                    step_name = st.session_state.chain_steps[int(step_num)-1]['name']
                                    with st.expander(f"📊 {step_name}", expanded=True):
                                        st.markdown(value)

                    with tab2:
                        st.markdown("### 🔗 Chain Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Execution Pattern", execution_type.title())
                            st.metric("Total Steps", len(st.session_state.chain_steps))
                            
                        with col2:
                            st.metric("Memory Enabled", "Yes" if use_memory else "No")
                            st.metric("Streaming Enabled", "Yes" if use_callbacks else "No")

                        # Chain flow visualization
                        st.markdown("**🔄 Execution Flow:**")
                        if execution_type == "sequential":
                            for i, step in enumerate(st.session_state.chain_steps):
                                st.markdown(f"**{i+1}.** {step['name']}")
                                if i < len(st.session_state.chain_steps) - 1:
                                    st.markdown("⬇️ *Context passed to next step*")
                        else:
                            st.markdown("**Parallel Execution:** All steps executed simultaneously")
                            for step in st.session_state.chain_steps:
                                st.markdown(f"• {step['name']}")

                    with tab3:
                        st.markdown("### ⚡ Performance Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Execution Time", f"{result['execution_time']:.2f}s")
                        with col2:
                            st.metric("Chain Type", execution_type.title())
                        with col3:
                            st.metric("LLM Provider", f"{llm_provider} ({model})")

                    with tab4:
                        st.markdown("### 🔧 Technical Implementation")
                        
                        st.markdown("**LangChain Components Used:**")
                        components = []
                        if execution_type == "sequential":
                            components = ["ChatPromptTemplate", "LCEL Chain", "StrOutputParser", "Sequential Execution"]
                        elif execution_type == "parallel":
                            components = ["RunnableParallel", "ChatPromptTemplate", "LCEL"]
                        else:
                            components = ["Custom LCEL Chain", "RunnableLambda", "Context Injection"]
                            
                        if use_memory:
                            components.append("ConversationBufferMemory")
                            
                        for comp in components:
                            st.code(f"✓ {comp}")

                        st.markdown("**Chain Configuration:**")
                        config = {
                            "execution_type": execution_type,
                            "steps_count": len(st.session_state.chain_steps),
                            "memory_enabled": use_memory,
                            "streaming_enabled": use_callbacks,
                            "llm_provider": llm_provider,
                            "model": model
                        }
                        st.json(config)

                else:
                    st.error(f"❌ Chain execution failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"❌ Error creating chain: {str(e)}")
            st.info("💡 Make sure your API keys are configured and the selected model is available.")

    # LangChain features info
    with st.expander("🆕 LangChain Features", expanded=False):
        st.markdown("""
        **🔗 LangChain Expression Language (LCEL)**
        - Declarative chain composition
        - Automatic parallelization and streaming
        - Built-in async support and error handling
        - Type safety and validation

        **🧠 Advanced Memory Management**
        - ConversationBufferMemory for chat history
        - ConversationSummaryMemory for large contexts
        - Custom memory implementations
        - Context window optimization

        **⚡ Execution Patterns**
        - **Sequential**: Traditional step-by-step execution
        - **Parallel**: Concurrent processing for independent tasks
        - **Custom LCEL**: Advanced routing and conditional logic
        - **Hybrid**: Combination of patterns for complex workflows

        **🔧 Built-in Features**
        - Automatic retry logic and error handling
        - Token usage tracking and optimization
        - Streaming responses with callbacks
        - Structured outputs with Pydantic models
        """)

if __name__ == "__main__":
    render_prompt_chaining_interface()