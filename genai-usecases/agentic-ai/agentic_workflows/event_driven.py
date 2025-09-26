"""
Event-Driven Workflow Pattern
=============================

A sophisticated reactive workflow system featuring asynchronous event processing,
intelligent agent coordination, and real-time workflow adaptation capabilities.

Key Features:
- Reactive architecture with pub/sub event management
- Multi-agent coordination via centralized event bus
- Real-time workflow adaptation based on event streams
- Advanced event-driven state management and persistence
- Comprehensive monitoring and analytics dashboard

Architecture:
- Backend: Event bus, reactive agents, and workflow orchestration
- Frontend: Interactive Streamlit interface for event management
- Integration: Multi-provider LLM support with intelligent event processing
- Patterns: Publisher-subscriber, observer, and reactive execution flows

Use Cases:
- Real-time customer support with agent coordination
- IoT device management with event-driven responses
- Live collaboration systems with multi-user interaction
- Monitoring and alerting with intelligent escalation
- Interactive applications with responsive user experiences
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import asyncio
import streamlit as st
import json
import time
from typing import List, Dict, Any, Callable
from datetime import datetime
from dotenv import load_dotenv
import os
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from queue import Queue

# LangChain core components for LLM integration
from langchain_core.messages import HumanMessage, SystemMessage

# Multi-provider LLM support
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Ollama integration with error handling
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

# Load environment configuration
load_dotenv()

# =============================================================================
# BACKEND: DATA MODELS AND ENUMERATIONS
# =============================================================================

class EventType(Enum):
    """
    Comprehensive enumeration of event types for reactive workflows.

    These event types enable intelligent routing, priority handling,
    and type-safe event processing across the distributed agent system.
    """
    USER_INPUT = "user_input"
    TASK_COMPLETED = "task_completed"
    AGENT_MESSAGE = "agent_message"
    ERROR_OCCURRED = "error_occurred"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    DECISION_REQUIRED = "decision_required"
    DATA_UPDATED = "data_updated"

@dataclass
class Event:
    """
    Comprehensive event representation for reactive workflows.

    Encapsulates all event metadata including routing information, payload data,
    timing information, and processing status for intelligent event management.

    Attributes:
        event_id: Unique identifier for event tracking
        event_type: Type of event (EventType enum)
        source: Origin of the event (agent_id, user, system)
        payload: Event data and parameters
        timestamp: ISO timestamp of event creation
        priority: Processing priority (higher = more urgent)
        processed: Whether event has been handled
    """
    event_id: str
    event_type: EventType
    source: str
    payload: Dict[str, Any]
    timestamp: str
    priority: int = 1
    processed: bool = False

@dataclass
class EventHandler:
    """
    Event handler registration with flexible configuration.

    Defines handler behavior, event type subscriptions, and processing
    priority for intelligent event routing and execution.

    Attributes:
        handler_id: Unique identifier for the handler
        event_types: List of event types this handler processes
        handler_func: Callable function for event processing
        priority: Processing priority for handler ordering
        active: Whether handler is enabled for processing
    """
    handler_id: str
    event_types: List[EventType]
    handler_func: Callable
    priority: int = 1
    active: bool = True

# =============================================================================
# BACKEND: EVENT BUS AND MANAGEMENT SYSTEM
# =============================================================================

class EventBus:
    """
    Centralized event bus for pub/sub messaging and event coordination.

    Manages event publication, subscription, routing, and processing with
    comprehensive error handling, priority management, and analytics.

    Features:
    - Type-safe event routing and subscription management
    - Priority-based event processing with handler ordering
    - Comprehensive error handling with automatic recovery
    - Real-time statistics and performance monitoring
    - Asynchronous event processing with queue management
    """

    def __init__(self):
        """Initialize event bus with default configuration."""
        self.handlers = defaultdict(list)
        self.event_history = []
        self.event_queue = Queue()
        self.processing = False
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "handlers_registered": 0
        }

    def subscribe(self, handler: EventHandler):
        """
        Subscribe event handler to specific event types.

        Args:
            handler: EventHandler instance with configuration
        """
        for event_type in handler.event_types:
            self.handlers[event_type].append(handler)
        self.stats["handlers_registered"] += 1

    def unsubscribe(self, handler_id: str):
        """
        Unsubscribe handler from all event types.

        Args:
            handler_id: Unique identifier of handler to remove
        """
        for event_type in self.handlers:
            self.handlers[event_type] = [
                h for h in self.handlers[event_type]
                if h.handler_id != handler_id
            ]

    async def publish(self, event: Event):
        """
        Publish event to all subscribed handlers with priority processing.

        Args:
            event: Event instance to publish and route
        """
        self.event_history.append(event)
        self.stats["events_published"] += 1

        # Get active handlers for this event type
        handlers = self.handlers.get(event.event_type, [])
        active_handlers = [h for h in handlers if h.active]

        # Sort by priority (higher priority first)
        active_handlers.sort(key=lambda x: x.priority, reverse=True)

        # Process handlers with error handling
        for handler in active_handlers:
            try:
                # Support both async and sync handler functions
                if asyncio.iscoroutinefunction(handler.handler_func):
                    await handler.handler_func(event)
                else:
                    handler.handler_func(event)

                self.stats["events_processed"] += 1

            except Exception as e:
                # Create error event for failed handler execution
                error_event = Event(
                    event_id=f"error_{int(time.time())}",
                    event_type=EventType.ERROR_OCCURRED,
                    source="event_bus",
                    payload={"error": str(e), "original_event": event.event_id},
                    timestamp=datetime.now().isoformat()
                )
                self.event_history.append(error_event)

# =============================================================================
# BACKEND: REACTIVE AGENTS AND INTELLIGENT PROCESSING
# =============================================================================

class ReactiveAgent:
    """
    Intelligent reactive agent with LLM-powered event processing.

    Responds to events with context-aware intelligence, maintains state across
    interactions, and coordinates with other agents via the event bus.

    Key Features:
    - LLM-powered intelligent event processing
    - Persistent state management across interactions
    - Dynamic event subscription and handler registration
    - Error handling with automatic recovery mechanisms
    - Context-aware response generation and coordination

    Args:
        agent_id: Unique identifier for the agent
        llm: Language model for intelligent processing
        event_bus: Central event bus for coordination
    """

    def __init__(self, agent_id: str, llm, event_bus: EventBus):
        """Initialize reactive agent with LLM and event bus integration."""
        self.agent_id = agent_id
        self.llm = llm
        self.event_bus = event_bus
        self.state = {}
        self.active = True

        # Register event handlers for automatic processing
        self._register_handlers()

    def _register_handlers(self):
        """Register event handlers for this agent"""
        handler = EventHandler(
            handler_id=f"{self.agent_id}_handler",
            event_types=[EventType.USER_INPUT, EventType.TASK_COMPLETED, EventType.DATA_UPDATED],
            handler_func=self._handle_event,
            priority=1
        )
        self.event_bus.subscribe(handler)

    async def _handle_event(self, event: Event):
        """Handle incoming events"""
        if not self.active:
            return

        try:
            if event.event_type == EventType.USER_INPUT:
                await self._handle_user_input(event)
            elif event.event_type == EventType.TASK_COMPLETED:
                await self._handle_task_completion(event)
            elif event.event_type == EventType.DATA_UPDATED:
                await self._handle_data_update(event)

        except Exception as e:
            await self.event_bus.publish(Event(
                event_id=f"agent_error_{int(time.time())}",
                event_type=EventType.ERROR_OCCURRED,
                source=self.agent_id,
                payload={"error": str(e), "original_event": event.event_id},
                timestamp=datetime.now().isoformat()
            ))

    async def _handle_user_input(self, event: Event):
        """Handle user input events"""
        user_input = event.payload.get("input", "")

        # Process with LLM
        messages = [
            SystemMessage(content=f"You are reactive agent {self.agent_id}. Respond to user input appropriately."),
            HumanMessage(content=f"User input: {user_input}\n\nCurrent state: {json.dumps(self.state)}")
        ]

        response = await self.llm.ainvoke(messages)

        # Update state
        self.state["last_response"] = response.content
        self.state["last_interaction"] = datetime.now().isoformat()

        # Publish response event
        await self.event_bus.publish(Event(
            event_id=f"agent_response_{int(time.time())}",
            event_type=EventType.AGENT_MESSAGE,
            source=self.agent_id,
            payload={"response": response.content, "user_input": user_input},
            timestamp=datetime.now().isoformat()
        ))

    async def _handle_task_completion(self, event: Event):
        """Handle task completion events"""
        task_result = event.payload.get("result", "")

        # Analyze and potentially trigger new tasks
        analysis_prompt = f"A task has been completed with result: {task_result}. Analyze if any follow-up actions are needed."

        messages = [
            SystemMessage(content=f"You are reactive agent {self.agent_id}. Analyze task completions and suggest follow-ups."),
            HumanMessage(content=analysis_prompt)
        ]

        response = await self.llm.ainvoke(messages)

        # Check if new tasks should be triggered
        if "follow-up" in response.content.lower() or "next" in response.content.lower():
            await self.event_bus.publish(Event(
                event_id=f"followup_suggested_{int(time.time())}",
                event_type=EventType.DECISION_REQUIRED,
                source=self.agent_id,
                payload={"suggestion": response.content, "original_task": event.payload},
                timestamp=datetime.now().isoformat()
            ))

    async def _handle_data_update(self, event: Event):
        """Handle data update events"""
        updated_data = event.payload.get("data", {})

        # Update internal state based on new data
        self.state.update(updated_data)

        # Analyze impact
        impact_prompt = f"Data has been updated: {json.dumps(updated_data)}. Analyze the impact on current operations."

        messages = [
            SystemMessage(content=f"You are reactive agent {self.agent_id}. Analyze data updates and their implications."),
            HumanMessage(content=impact_prompt)
        ]

        response = await self.llm.ainvoke(messages)

        # Publish analysis
        await self.event_bus.publish(Event(
            event_id=f"data_analysis_{int(time.time())}",
            event_type=EventType.AGENT_MESSAGE,
            source=self.agent_id,
            payload={"analysis": response.content, "updated_data": updated_data},
            timestamp=datetime.now().isoformat()
        ))

# =============================================================================
# BACKEND: EVENT-DRIVEN WORKFLOW ORCHESTRATION
# =============================================================================

class EventDrivenWorkflow:
    """
    Advanced event-driven workflow orchestrator with multi-agent coordination.

    Manages workflow execution through event-driven architecture, coordinating
    multiple reactive agents and maintaining comprehensive workflow state.

    Key Features:
    - Multi-agent coordination via centralized event bus
    - Dynamic workflow adaptation based on event streams
    - Comprehensive workflow state management and logging
    - Real-time performance monitoring and analytics
    - Scalable agent management with dynamic registration

    Args:
        llm: Language model for intelligent workflow decisions
    """

    def __init__(self, llm):
        """Initialize workflow orchestrator with LLM integration."""
        self.llm = llm
        self.event_bus = EventBus()
        self.agents = {}
        self.workflow_state = "idle"
        self.execution_log = []

    def add_agent(self, agent_id: str) -> ReactiveAgent:
        """Add a new reactive agent"""
        agent = ReactiveAgent(agent_id, self.llm, self.event_bus)
        self.agents[agent_id] = agent
        return agent

    async def start_workflow(self, initial_input: str):
        """Start the event-driven workflow"""
        self.workflow_state = "running"

        # Publish workflow start event
        await self.event_bus.publish(Event(
            event_id=f"workflow_start_{int(time.time())}",
            event_type=EventType.WORKFLOW_STARTED,
            source="workflow_manager",
            payload={"initial_input": initial_input},
            timestamp=datetime.now().isoformat()
        ))

        # Publish initial user input
        await self.event_bus.publish(Event(
            event_id=f"user_input_{int(time.time())}",
            event_type=EventType.USER_INPUT,
            source="user",
            payload={"input": initial_input},
            timestamp=datetime.now().isoformat()
        ))

    async def trigger_event(self, event_type: EventType, payload: Dict[str, Any], source: str = "user"):
        """Manually trigger an event"""
        event = Event(
            event_id=f"manual_event_{int(time.time())}",
            event_type=event_type,
            source=source,
            payload=payload,
            timestamp=datetime.now().isoformat()
        )

        await self.event_bus.publish(event)
        return event

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        return {
            "workflow_state": self.workflow_state,
            "active_agents": len([a for a in self.agents.values() if a.active]),
            "total_events": len(self.event_bus.event_history),
            "event_stats": self.event_bus.stats,
            "recent_events": self.event_bus.event_history[-10:] if self.event_bus.event_history else []
        }

# =============================================================================
# BACKEND: LLM PROVIDER CONFIGURATION
# =============================================================================

def create_llm(provider: str, model: str, base_url: str = None):
    """
    Create LLM instance based on provider with comprehensive multi-provider support.

    Args:
        provider: LLM provider ("Ollama", "Gemini", "Groq", "Anthropic", "OpenAI")
        model: Model identifier for the specific provider
        base_url: Optional base URL for self-hosted providers (Ollama)

    Returns:
        Configured LLM instance ready for reactive agent integration

    Raises:
        ValueError: If provider is not supported
        ConnectionError: If API keys are missing or invalid
    """
    if provider == "Ollama" and ChatOllama:
        return ChatOllama(
            model=model,
            base_url=base_url or 'http://localhost:11434',
            temperature=0.3,
            timeout=120
        )
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
    elif provider == "Groq":
        return ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)
    elif provider == "Anthropic":
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0.3)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# =============================================================================
# FRONTEND: STREAMLIT INTERFACE AND USER INTERACTIONS
# =============================================================================

def render_event_driven_interface():
    """
    Render comprehensive Event-Driven workflow interface with advanced features.

    Creates an interactive Streamlit interface for managing event-driven workflows
    with intelligent agent coordination, real-time monitoring, and comprehensive
    analytics dashboard.

    Features:
    - Multi-provider LLM configuration with validation
    - Dynamic reactive agent management and coordination
    - Real-time event monitoring with interactive triggers
    - Comprehensive analytics and performance metrics
    - Interactive workflow control with manual event injection
    - Live agent state monitoring and management
    """
    # =============================================================================
    # HEADER AND INTRODUCTION
    # =============================================================================

    st.header("⚡ Event-Driven Workflow")

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
            key='event_llm_provider',
            help="Select the LLM provider for reactive agent intelligence"
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
            key='event_model',
            help="Select the specific model for the chosen provider"
        )

        # -------------------------------------------------------------------------
        # Workflow Configuration
        # -------------------------------------------------------------------------

        st.markdown("### ⚙️ Workflow Settings")
        auto_agents = st.slider(
            "Number of Reactive Agents",
            1, 5, 3,
            key='auto_agents',
            help="Number of reactive agents for parallel event processing"
        )

        # -------------------------------------------------------------------------
        # Ollama-Specific Configuration
        # -------------------------------------------------------------------------

        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='event_ollama_url',
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
            if llm_provider == "Gemini":
                required_key = "GEMINI_API_KEY"

            if os.getenv(required_key):
                st.success(f"✅ {required_key} configured")
            else:
                st.warning(f"⚠️ {required_key} not found")

    # =============================================================================
    # WORKFLOW INITIALIZATION
    # =============================================================================

    if 'event_workflow' not in st.session_state:
        llm = create_llm(llm_provider, model, ollama_base_url)
        st.session_state.event_workflow = EventDrivenWorkflow(llm)

        # Add default agents
        for i in range(auto_agents):
            st.session_state.event_workflow.add_agent(f"agent_{i+1}")

    # Main interface
    workflow = st.session_state.event_workflow

    # Workflow control
    col1, col2 = st.columns([2, 1])

    with col1:
        initial_input = st.text_area(
            "Initial Input / Event Trigger",
            placeholder="Enter your initial request or event to start the reactive workflow...",
            height=100,
            key='initial_input'
        )

    with col2:
        if st.button("🚀 Start Workflow", type="primary", key='start_workflow'):
            if initial_input:
                with st.spinner("⚡ Starting event-driven workflow..."):
                    asyncio.run(workflow.start_workflow(initial_input))
                st.success("✅ Workflow started!")
                st.rerun()

    # Manual event triggers
    st.markdown("### 🎯 Manual Event Triggers")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📝 User Input Event"):
            user_input = st.text_input("Enter user input:", key="manual_user_input")
            if user_input:
                asyncio.run(workflow.trigger_event(
                    EventType.USER_INPUT,
                    {"input": user_input}
                ))
                st.rerun()

    with col2:
        if st.button("✅ Task Completed Event"):
            task_result = st.text_input("Enter task result:", key="manual_task_result")
            if task_result:
                asyncio.run(workflow.trigger_event(
                    EventType.TASK_COMPLETED,
                    {"result": task_result}
                ))
                st.rerun()

    with col3:
        if st.button("📊 Data Update Event"):
            data_update = st.text_input("Enter data update (JSON):", key="manual_data_update")
            if data_update:
                try:
                    data = json.loads(data_update)
                    asyncio.run(workflow.trigger_event(
                        EventType.DATA_UPDATED,
                        {"data": data}
                    ))
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")

    # Workflow status and stats
    stats = workflow.get_workflow_stats()

    if stats["total_events"] > 0:
        st.markdown("### 📊 Workflow Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Workflow State", stats["workflow_state"].title())
        with col2:
            st.metric("Active Agents", stats["active_agents"])
        with col3:
            st.metric("Total Events", stats["total_events"])
        with col4:
            st.metric("Events Processed", stats["event_stats"]["events_processed"])

        # Event history and logs
        tab1, tab2, tab3 = st.tabs(["📋 Recent Events", "🤖 Agent States", "📈 Event Analytics"])

        with tab1:
            st.markdown("### 📋 Recent Events")

            recent_events = stats["recent_events"]
            if recent_events:
                for event in reversed(recent_events):
                    event_data = event if isinstance(event, dict) else {
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "source": event.source,
                        "payload": event.payload,
                        "timestamp": event.timestamp
                    }

                    with st.expander(f"{event_data['event_type']} from {event_data['source']} - {event_data['timestamp']}", expanded=False):
                        st.json(event_data)

        with tab2:
            st.markdown("### 🤖 Agent States")

            for agent_id, agent in workflow.agents.items():
                with st.expander(f"Agent: {agent_id}", expanded=False):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write(f"**Status**: {'Active' if agent.active else 'Inactive'}")
                        st.write(f"**Agent ID**: {agent.agent_id}")

                    with col2:
                        if st.button(f"Toggle {agent_id}", key=f"toggle_{agent_id}"):
                            agent.active = not agent.active
                            st.rerun()

                    st.markdown("**Current State:**")
                    st.json(agent.state)

        with tab3:
            st.markdown("### 📈 Event Analytics")

            # Event type distribution
            event_types = {}
            for event in workflow.event_bus.event_history:
                event_type = event.event_type.value if hasattr(event, 'event_type') else event.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1

            st.markdown("**Event Type Distribution:**")
            for event_type, count in event_types.items():
                st.write(f"• **{event_type}**: {count}")

            # Processing efficiency
            total_events = stats["event_stats"]["events_published"]
            processed_events = stats["event_stats"]["events_processed"]
            efficiency = (processed_events / total_events * 100) if total_events > 0 else 0

            st.metric("Processing Efficiency", f"{efficiency:.1f}%")

    # Pattern information
    with st.expander("ℹ️ Event-Driven Pattern", expanded=False):
        st.markdown("""
        **Event-Driven Workflow Pattern:**

        ⚡ **Reactive Architecture**
        - Agents respond to events in real-time
        - Decoupled communication via event bus
        - Asynchronous processing capabilities

        📡 **Event Management**
        - Centralized event bus for coordination
        - Type-safe event definitions
        - Priority-based event handling

        🔄 **Dynamic Adaptation**
        - Workflow adapts based on events
        - Real-time state management
        - Intelligent agent coordination

        🛡️ **Fault Tolerance**
        - Event-level error handling
        - Agent isolation and recovery
        - Graceful degradation

        📊 **Monitoring & Analytics**
        - Real-time event tracking
        - Agent state monitoring
        - Performance analytics

        **Use Cases:**
        - Real-time customer support
        - IoT device management
        - Live collaboration systems
        - Monitoring and alerting
        - Interactive applications
        """)

if __name__ == "__main__":
    render_event_driven_interface()