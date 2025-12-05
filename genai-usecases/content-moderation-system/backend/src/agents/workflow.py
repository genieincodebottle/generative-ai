"""
LangGraph workflow for content moderation multi-agent system.

This module constructs the StateGraph that orchestrates the agents with:
- Sequential analysis agents (Content, Toxicity, Policy)
- ReAct Decision Loop for synthesizing agent decisions
- Human-in-the-Loop (HITL) interrupt points
- User Reputation and Action Enforcement agents

Flow:
1. Content Analysis → 2. Toxicity Detection → 3. Policy Check
   ↓
4. ReAct Decision Loop (Think-Act-Observe synthesis)
   ↓
5. HITL Checkpoint (if triggered) ←→ Human Review Queue
   ↓
6. User Reputation Scoring → 7. Action Enforcement → END
"""

import os
import logging
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.models import ContentState, ContentStatus
from ..agents.agents import ContentModerationAgents
from ..database.moderation_db import ModerationDatabase
from ..ml.guardrails import GuardrailManager, GuardrailConfig
from ..memory.learning_tracker import LearningTracker

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def should_use_fast_mode(state: ContentState) -> bool:
    """
    Determine if content is eligible for fast mode processing.

    Fast mode criteria:
    - Content type is in eligible list (e.g., story_comment)
    - Content length is below threshold (default 200 characters)
    - Not explicitly marked for full pipeline

    Returns:
        True if fast mode should be used, False otherwise
    """
    # Check environment variables
    fast_mode_enabled = os.getenv("ENABLE_FAST_MODE", "true").lower() == "true"
    if not fast_mode_enabled:
        return False

    # Check if explicitly disabled for this content
    if state.get("force_full_pipeline", False):
        return False

    # Check content type eligibility
    content_type = state.get("content_type", "")
    eligible_types = os.getenv("FAST_MODE_CONTENT_TYPES", "story_comment").split(",")
    eligible_types = [t.strip() for t in eligible_types]

    if content_type not in eligible_types:
        return False

    # Check content length
    content_text = state.get("content_text", "")
    max_length = int(os.getenv("FAST_MODE_MAX_LENGTH", "200"))

    if len(content_text) > max_length:
        return False

    logger.info(f"✅ Fast mode eligible: type={content_type}, length={len(content_text)} chars")
    return True


def should_continue_from_content_analysis(state: ContentState) -> Literal["toxicity_detection", "END"]:
    """
    Routing function after Content Analysis Agent.

    Routes to:
    - END if flagged for review
    - toxicity_detection if approved
    """
    status = state.get("status")
    requires_review = state.get("requires_human_review", False)

    if status == ContentStatus.FLAGGED.value or requires_review:
        return "END"
    else:
        return "toxicity_detection"


def should_continue_from_toxicity(state: ContentState) -> Literal["policy_check", "END"]:
    """
    Routing function after Toxicity Detection Agent.

    Routes to:
    - END if content should be removed
    - policy_check if approved or flagged
    """
    status = state.get("status")
    requires_review = state.get("requires_human_review", False)

    if status == ContentStatus.REMOVED.value:
        return "END"
    elif status == ContentStatus.FLAGGED.value and requires_review:
        return "END"
    else:
        return "policy_check"


def should_continue_from_policy(state: ContentState) -> Literal["react_loop", "END"]:
    """
    Routing function after Policy Violation Agent.

    Routes to:
    - END if content should be removed immediately
    - react_loop for decision synthesis (Think-Act-Observe)
    """
    status = state.get("status")

    if status == ContentStatus.REMOVED.value:
        return "END"
    else:
        return "react_loop"


def should_continue_from_react(state: ContentState) -> Literal["hitl_review", "reputation_scoring", "action_enforcement", "END"]:
    """
    Routing function after ReAct Decision Loop.

    The ReAct loop synthesizes all agent decisions and determines if HITL is needed.

    Routes to:
    - hitl_review if human review is required
    - reputation_scoring if user-level action needed
    - action_enforcement if content decision only
    - END if approved with high confidence
    """
    status = state.get("status")
    hitl_required = state.get("hitl_required", False)
    react_decision = state.get("react_act_decision", "")

    logger.info(f"\nRouting from ReAct Loop: status={status}, hitl_required={hitl_required}, decision={react_decision}")

    # Check if HITL is required
    if hitl_required and status == ContentStatus.PENDING_HUMAN_REVIEW.value:
        return "hitl_review"

    # Check if we need user-level actions
    if react_decision in ["suspend_user", "ban_user", "warn", "remove"]:
        return "reputation_scoring"

    # High-confidence approve - skip to action enforcement
    if react_decision == "approve" and state.get("react_confidence", 0) >= 0.85:
        return "action_enforcement"

    # Default: go through reputation scoring
    return "reputation_scoring"


def should_continue_from_hitl(state: ContentState) -> Literal["reputation_scoring", "action_enforcement", "END"]:
    """
    Routing function after HITL Checkpoint.

    Routes based on human decision:
    - reputation_scoring for user actions (suspend, ban)
    - action_enforcement for content actions (warn, remove)
    - END for approvals or escalations
    """
    status = state.get("status")
    human_decision = state.get("hitl_human_decision", "")

    # Still waiting for human
    if status == ContentStatus.PENDING_HUMAN_REVIEW.value and not human_decision:
        return "END"  # Workflow pauses here until human provides input

    # Human approved - end workflow
    if human_decision == "approve":
        return "action_enforcement"

    # Human escalated - end workflow (handled externally)
    if human_decision == "escalate" or status == ContentStatus.ESCALATED.value:
        return "END"

    # User-level actions need reputation scoring
    if human_decision in ["suspend_user", "ban_user"]:
        return "reputation_scoring"

    # Content actions go to enforcement
    return "action_enforcement"


def should_continue_from_reputation(state: ContentState) -> Literal["action_enforcement", "END"]:
    """
    Routing function after User Reputation Scoring Agent.

    Routes to:
    - action_enforcement if action needs to be taken
    - END if approved
    """
    status = state.get("status")
    requires_review = state.get("requires_human_review", False)

    if status in [ContentStatus.ACTION_ENFORCEMENT.value, ContentStatus.WARNED.value]:
        return "action_enforcement"
    elif status == ContentStatus.APPROVED.value:
        return "END"
    elif requires_review:
        return "END"
    else:
        return "action_enforcement"


def create_moderation_workflow(db: ModerationDatabase, use_checkpointer: bool = True, enable_guardrails: bool = True, enable_learning: bool = True, enable_fast_mode: bool = True) -> StateGraph:
    """
    Create the LangGraph StateGraph for content moderation.

    The graph connects agents with conditional routing and HITL interrupts:

    Main Flow:
    1. Content Analysis Agent - Analyzes content structure and sentiment
    2. Toxicity Detection Agent - Detects toxic language patterns
    3. Policy Violation Agent - Checks against community guidelines
    4. ReAct Decision Loop - Synthesizes decisions (Think-Act-Observe)
    5. HITL Checkpoint - Human review if triggered (workflow pauses)
    6. User Reputation Scoring Agent - Evaluates user risk
    7. Action Enforcement Agent - Executes final actions

    Fast Mode Flow (for short comments):
    Fast Mode Agent → END (single LLM call, 1-2s processing)

    Appeal Flow:
    Appeal Review → Action Enforcement → END

    Args:
        db: ModerationDatabase instance
        use_checkpointer: If True, use MemorySaver for state persistence (enables HITL resume)
        enable_guardrails: If True, enable safety guardrails (loop limits, hallucination detection)
        enable_learning: If True, enable learning from decisions (episodic & semantic memory)
        enable_fast_mode: If True, enable fast mode for eligible content (short comments)

    Returns:
        Compiled StateGraph
    """
    logger.info("\nBuilding Content Moderation Workflow with HITL Support...")
    logger.info("=" * 40)

    # Initialize learning tracker
    learning_tracker = None
    if enable_learning:
        try:
            learning_tracker = LearningTracker()
        except Exception as learn_error:
            learning_tracker = None

    # Initialize guardrails
    guardrail_manager = None
    if enable_guardrails:
        logger.info("Initializing Guardrails...")
        try:
            guardrail_config = GuardrailConfig(
                max_reasoning_iterations=10,
                max_agent_calls=20,
                max_cost_usd=1.0,
                max_execution_time_seconds=300,
                hallucination_check_enabled=True,
                consistency_check_enabled=True
            )
            guardrail_manager = GuardrailManager(config=guardrail_config)
        except Exception as gr_error:
            logger.error(f"Failed to initialize guardrails: {gr_error}")
            logger.error("Continuing without guardrails...")
            guardrail_manager = None

    # Initialize agents
    logger.info("Initializing ContentModerationAgents...")
    try:
        agents = ContentModerationAgents()
    except Exception as agent_error:
        logger.error(f"Failed to initialize agents: {agent_error}")
        import traceback
        traceback.print_exc()
        raise agent_error

    # Create the graph
    logger.info("Creating StateGraph...")
    workflow = StateGraph(ContentState)
    logger.info("StateGraph created")

    # Create guardrail and learning-wrapped agent functions
    def create_agent_wrapper(agent_func, agent_name: str):
        """
        Wrap an agent function with guardrails and learning.

        Args:
            agent_func: The original agent function
            agent_name: Name of the agent for tracking

        Returns:
            Wrapped agent function with guardrails and learning
        """
        def wrapped_agent(state: ContentState) -> ContentState:
            # Check guardrails before agent execution
            if guardrail_manager:
                # Track iteration count (handle None case)
                iteration = state.get("_guardrail_iteration") or 0
                state["_guardrail_iteration"] = iteration + 1

                # Check all guardrails
                guardrail_result = guardrail_manager.check_all_guardrails(
                    state,
                    current_iteration=iteration,
                    operation_cost=0.001  # Estimate per agent call
                )

                # Record guardrail check in state
                if not state.get("_guardrail_checks"):
                    state["_guardrail_checks"] = []
                state["_guardrail_checks"].append({
                    "agent": agent_name,
                    "iteration": iteration,
                    "result": guardrail_result
                })

                # If guardrails failed, add warning to state
                if not guardrail_result["passed"]:
                    violations = state.get("guardrail_violations") or []
                    state["guardrail_violations"] = violations + guardrail_result["violations"]

                # Log warnings
                if guardrail_result["warnings"]:
                    warnings = state.get("guardrail_warnings") or []
                    state["guardrail_warnings"] = warnings + guardrail_result["warnings"]

            # Execute the agent
            result_state = agent_func(state)

            # Check for hallucinations in agent decisions (post-execution)
            if guardrail_manager and guardrail_manager.config.hallucination_check_enabled:
                decisions = result_state.get("agent_decisions", [])
                if decisions:
                    last_decision = decisions[-1]
                    hallucination_result = guardrail_manager.hallucination_detector.check_for_hallucination(
                        last_decision, result_state
                    )
                    if hallucination_result["hallucination_detected"]:
                        # Adjust confidence if hallucination detected
                        confidence_adj = hallucination_result.get("confidence_adjustment", 0)
                        if confidence_adj != 0:
                            last_decision.confidence = max(0.1, last_decision.confidence + confidence_adj)

            # Record decision for learning (if enabled)
            if learning_tracker and agent_name in ["toxicity_detection", "policy_check", "react_loop"]:
                decisions = result_state.get("agent_decisions", [])
                if decisions:
                    last_decision = decisions[-1]
                    # Record the decision (outcome will be updated if appealed)
                    try:
                        learning_tracker.record_decision(
                            agent_name=last_decision.agent_name,
                            content_text=result_state.get("content_text", ""),
                            toxicity_score=result_state.get("toxicity_score", 0.0),
                            policy_violations=result_state.get("policy_violations", []),
                            decision=last_decision.decision.value,
                            confidence=last_decision.confidence,
                            outcome=result_state.get("status", "unknown"),
                            context=f"toxicity_{result_state.get('toxicity_score', 0):.1f}",
                            metadata={"agent": agent_name}
                        )
                    except Exception as learn_err:
                        logger.error(f"Learning error: {learn_err}")

            return result_state

        return wrapped_agent

    # Add agent nodes with optional guardrails and learning
    logger.info("Adding agent nodes...")
    try:
        if guardrail_manager or learning_tracker:
            features = []
            if guardrail_manager:
                features.append("guardrails")
            if learning_tracker:
                features.append("learning")

            workflow.add_node("content_analysis", create_agent_wrapper(agents.content_analysis_agent, "content_analysis"))
            workflow.add_node("toxicity_detection", create_agent_wrapper(agents.toxicity_detection_agent, "toxicity_detection"))
            workflow.add_node("policy_check", create_agent_wrapper(agents.policy_violation_agent, "policy_check"))
            workflow.add_node("react_loop", create_agent_wrapper(agents.react_decision_loop_agent, "react_loop"))
            workflow.add_node("hitl_review", create_agent_wrapper(agents.hitl_checkpoint_agent, "hitl_review"))
            workflow.add_node("reputation_scoring", create_agent_wrapper(agents.user_reputation_agent, "reputation_scoring"))
            workflow.add_node("appeal_review", create_agent_wrapper(agents.appeal_review_agent, "appeal_review"))
            workflow.add_node("action_enforcement", create_agent_wrapper(agents.action_enforcement_agent, "action_enforcement"))

            # Add fast mode agent if enabled
            if enable_fast_mode:
                workflow.add_node("fast_mode", create_agent_wrapper(agents.fast_mode_agent, "fast_mode"))
                logger.info("   ✅ Fast mode agent added")
        else:
            # No guardrails - add agents directly
            workflow.add_node("content_analysis", agents.content_analysis_agent)
            workflow.add_node("toxicity_detection", agents.toxicity_detection_agent)
            workflow.add_node("policy_check", agents.policy_violation_agent)
            workflow.add_node("react_loop", agents.react_decision_loop_agent)
            workflow.add_node("hitl_review", agents.hitl_checkpoint_agent)
            workflow.add_node("reputation_scoring", agents.user_reputation_agent)
            workflow.add_node("appeal_review", agents.appeal_review_agent)
            workflow.add_node("action_enforcement", agents.action_enforcement_agent)

            # Add fast mode agent if enabled
            if enable_fast_mode:
                workflow.add_node("fast_mode", agents.fast_mode_agent)
                logger.info("   ✅ Fast mode agent added")
    except Exception as node_error:
        logger.error(f"   ❌ Failed to add node: {node_error}")
        import traceback
        traceback.print_exc()
        raise node_error

    # Entry router node function - routes to the appropriate starting point
    def entry_router(state: ContentState) -> ContentState:
        """
        Entry router that sets up the initial routing.
        The actual routing is handled by conditional edges from this node.
        """
        return state

    # Entry routing function
    def route_entry(state: ContentState) -> Literal["appeal_review", "hitl_review", "content_analysis", "fast_mode"]:
        """
        Route based on content type:
        - Appeals go to appeal_review
        - HITL resumes go to hitl_review (with human decision)
        - Fast mode eligible content goes to fast_mode (short comments)
        - New content starts with content_analysis
        """
        if state.get("is_appeal", False):
            return "appeal_review"
        elif state.get("hitl_human_decision") and state.get("status") == ContentStatus.PENDING_HUMAN_REVIEW.value:
            return "hitl_review"
        elif enable_fast_mode and should_use_fast_mode(state):
            return "fast_mode"
        else:
            return "content_analysis"

    # Add entry router node
    workflow.add_node("entry_router", entry_router)

    # Set entry point to the router
    workflow.set_entry_point("entry_router")

    # Add conditional edges from entry router
    edge_targets = {
        "appeal_review": "appeal_review",
        "hitl_review": "hitl_review",
        "content_analysis": "content_analysis"
    }

    # Add fast mode edge if enabled
    if enable_fast_mode:
        edge_targets["fast_mode"] = "fast_mode"

    workflow.add_conditional_edges(
        "entry_router",
        route_entry,
        edge_targets
    )

    # Add conditional edges for main workflow
    logger.info("Adding conditional edges...")

    # Content Analysis → Toxicity Detection or END
    workflow.add_conditional_edges(
        "content_analysis",
        should_continue_from_content_analysis,
        {
            "toxicity_detection": "toxicity_detection",
            "END": END
        }
    )

    # Toxicity Detection → Policy Check or END
    workflow.add_conditional_edges(
        "toxicity_detection",
        should_continue_from_toxicity,
        {
            "policy_check": "policy_check",
            "END": END
        }
    )

    # Policy Check → ReAct Loop or END
    workflow.add_conditional_edges(
        "policy_check",
        should_continue_from_policy,
        {
            "react_loop": "react_loop",
            "END": END
        }
    )

    # ReAct Loop → HITL Review OR Reputation Scoring OR Action Enforcement OR END
    workflow.add_conditional_edges(
        "react_loop",
        should_continue_from_react,
        {
            "hitl_review": "hitl_review",
            "reputation_scoring": "reputation_scoring",
            "action_enforcement": "action_enforcement",
            "END": END
        }
    )

    # HITL Review → Reputation Scoring OR Action Enforcement OR END (pauses here)
    workflow.add_conditional_edges(
        "hitl_review",
        should_continue_from_hitl,
        {
            "reputation_scoring": "reputation_scoring",
            "action_enforcement": "action_enforcement",
            "END": END
        }
    )

    # Reputation Scoring → Action Enforcement or END
    workflow.add_conditional_edges(
        "reputation_scoring",
        should_continue_from_reputation,
        {
            "action_enforcement": "action_enforcement",
            "END": END
        }
    )

    # Appeal review goes directly to action enforcement
    workflow.add_edge("appeal_review", "action_enforcement")

    # Action enforcement always ends
    workflow.add_edge("action_enforcement", END)

    # Fast mode always ends (single-pass decision)
    if enable_fast_mode:
        workflow.add_edge("fast_mode", END)

    # Compile the graph with optional checkpointer for HITL state persistence
    logger.info("Compiling workflow...")

    try:
        if use_checkpointer:
            checkpointer = MemorySaver()
            compiled_graph = workflow.compile(checkpointer=checkpointer)
        else:
            compiled_graph = workflow.compile()
    except Exception as compile_error:
        logger.error(f"Failed to compile workflow: {compile_error}")
        import traceback
        traceback.print_exc()
        raise compile_error

    return compiled_graph


def process_content(
    graph: StateGraph,
    initial_state: ContentState,
    config: Dict[str, Any] = None
) -> ContentState:
    """
    Process content through the multi-agent moderation workflow.

    This function handles:
    - Initial content processing
    - HITL workflow pauses (returns state with pending_human_review)
    - Workflow resumption after human input

    Args:
        graph: Compiled LangGraph
        initial_state: Initial content state
        config: Optional configuration (thread_id is used for HITL state persistence)

    Returns:
        Final content state after processing (may be pending if HITL triggered)
    """
    logger.info("\n" + "=" * 40)
    logger.info("STARTING CONTENT MODERATION")
    logger.info("=" * 40)

    if config is None:
        config = {"configurable": {"thread_id": initial_state.get("content_id", "default")}}

    # Run the workflow
    final_state = None
    step_count = 0
    try:
        logger.info("\nStarting workflow stream...")
        for state in graph.stream(initial_state, config):
            step_count += 1
            node_name = list(state.keys())[0] if state else "unknown"
            final_state = state
    except Exception as stream_error:
        logger.error(f"\nWORKFLOW STREAM ERROR at step {step_count}:")
        logger.error(f"Error Type: {type(stream_error).__name__}")
        logger.error(f"Error Message: {str(stream_error)}")
        import traceback
        traceback.print_exc()
        raise stream_error

    # Extract the actual state from the last node's output
    if final_state:
        final_state = list(final_state.values())[0]
    else:
        logger.warning("WARNING: final_state is None!")

    logger.info("\n" + "=" * 40)

    # Check if workflow is paused for HITL
    if final_state.get("status") == ContentStatus.PENDING_HUMAN_REVIEW.value:
        return final_state

    logger.info("CONTENT MODERATION COMPLETE")
    logger.info("=" * 40)

    # Display ReAct Decision Loop results
    if final_state.get("react_act_decision"):
        logger.info(f"\nReAct Loop Decision: {final_state.get('react_act_decision')}")
        logger.info(f"   Confidence: {final_state.get('react_confidence', 0):.2%}")

    if final_state.get("status") == ContentStatus.APPROVED.value:
        logger.info(f"\nCONTENT APPROVED")
    elif final_state.get("status") == ContentStatus.REMOVED.value:
        logger.info(f"\nCONTENT REMOVED")
        logger.info(f"Reason: {final_state.get('action_reason', 'Policy violation')}")
    elif final_state.get("status") == ContentStatus.WARNED.value:
        logger.info(f"\nUSER WARNED")
    elif final_state.get("status") == ContentStatus.ESCALATED.value:
        logger.info(f"\nESCALATED TO SENIOR MODERATOR")
    else:
        logger.info(f"\nUNDER REVIEW")

    # Display user action if any
    if final_state.get("user_suspended"):
        duration = final_state.get("suspension_duration_days", "permanent")
        logger.info(f"USER ACTION: Suspended for {duration} days")
    elif final_state.get("content_removed"):
        logger.info(f"CONTENT ACTION: Content removed from platform")

    # Display if HITL was involved
    if final_state.get("hitl_resolution_timestamp"):
        logger.info(f"\nHITL Review: Resolved at {final_state.get('hitl_resolution_timestamp')}")
        logger.info(f"Human Decision: {final_state.get('hitl_human_decision')}")

    # Display guardrail statistics if available
    if final_state.get("_guardrail_checks"):
        guardrail_checks = final_state.get("_guardrail_checks") or []
        violations = final_state.get("guardrail_violations") or []
        warnings = final_state.get("guardrail_warnings") or []

        if violations:
            logger.warning(f"Violation Details: {violations}")
        if warnings:
            logger.warning(f"Warning Details: {warnings[:3]}...")  # Show first 3 warnings

    return final_state


def resume_from_hitl(
    graph: StateGraph,
    content_id: str,
    human_decision: str,
    human_notes: str = "",
    reviewer_name: str = "Anonymous",
    confidence_override: float = None,
    existing_state: ContentState = None
) -> ContentState:
    """
    Resume a paused workflow with human decision.

    This is called when a human moderator makes a decision on content
    that was paused at an HITL checkpoint.

    Args:
        graph: Compiled LangGraph
        content_id: ID of the content being reviewed
        human_decision: Human's decision (approve, warn, remove, suspend_user, ban_user, escalate)
        human_notes: Optional notes from the reviewer
        reviewer_name: Name of the human reviewer
        confidence_override: Optional confidence override (default: 1.0)
        existing_state: The existing state to resume (if available)

    Returns:
        Final content state after workflow completion
    """
    logger.info("\n" + "=" * 40)
    logger.info("RESUMING WORKFLOW WITH HUMAN DECISION")
    logger.info("=" * 40)

    if existing_state is None:
        raise ValueError(f"No existing state found for content_id: {content_id}")

    # Update state with human decision
    existing_state["hitl_human_decision"] = human_decision
    existing_state["hitl_human_notes"] = human_notes
    existing_state["reviewer_name"] = reviewer_name

    if confidence_override is not None:
        existing_state["hitl_human_confidence_override"] = confidence_override

    # Resume workflow
    config = {"configurable": {"thread_id": content_id}}
    final_state = process_content(graph, existing_state, config)

    return final_state


def create_appeal_workflow(db: ModerationDatabase) -> StateGraph:
    """
    Create a simplified workflow for appeals.

    Appeals skip the initial analysis and go straight to appeal review.
    """
    logger.info("\nBuilding Appeal Review Workflow...")

    agents = ContentModerationAgents()
    workflow = StateGraph(ContentState)

    # Add only appeal-related nodes
    workflow.add_node("appeal_review", agents.appeal_review_agent)
    workflow.add_node("action_enforcement", agents.action_enforcement_agent)

    # Set entry point
    workflow.set_entry_point("appeal_review")

    # Appeal review goes to action enforcement
    workflow.add_edge("appeal_review", "action_enforcement")

    # Action enforcement ends
    workflow.add_edge("action_enforcement", END)

    compiled_graph = workflow.compile()

    logger.info("Appeal Review Workflow built successfully")

    return compiled_graph
