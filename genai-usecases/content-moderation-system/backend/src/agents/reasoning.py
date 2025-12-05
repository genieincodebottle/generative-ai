"""
ReAct, Plan-and-Execute, and Reflexion patterns for advanced agent reasoning.

This module implements:
1. ReAct (Reason + Act) - Agents reason before taking actions
2. Plan-and-Execute - Dynamic task decomposition and planning
3. Reflexion - Self-correction and critique mechanisms
"""

from typing import Dict, List, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from ..core.models import ContentState, AgentDecision, DecisionType


class ReasoningStepType(Enum):
    """Types of reasoning steps."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    PLAN = "plan"
    CRITIQUE = "critique"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_type: ReasoningStepType
    content: str
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass
class ExecutionPlan:
    """A plan for executing a moderation task."""
    plan_id: str
    steps: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    created_at: datetime
    updated_at: datetime


class ReActLoop:
    """
    Implements the ReAct (Reason + Act) pattern for agents.

    The agent follows this loop:
    1. Thought: Reason about the current state
    2. Action: Decide what action to take
    3. Observation: Observe the results
    4. Repeat until task is complete or max iterations reached
    """

    def __init__(self, llm: ChatGoogleGenerativeAI, max_iterations: int = 5):
        """
        Initialize ReAct loop.

        Args:
            llm: Language model for reasoning
            max_iterations: Maximum number of reasoning iterations
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.reasoning_history: List[ReasoningStep] = []

    def run(
        self,
        task_description: str,
        initial_state: Dict[str, Any],
        available_actions: List[Callable],
        completion_checker: Callable[[Dict[str, Any]], bool]
    ) -> Dict[str, Any]:
        """
        Run the ReAct loop.

        Args:
            task_description: Description of the task to accomplish
            initial_state: Initial state dictionary
            available_actions: List of callable actions
            completion_checker: Function to check if task is complete

        Returns:
            Final state after reasoning loop
        """
        state = initial_state.copy()
        iteration = 0

        while iteration < self.max_iterations:
            # THOUGHT: Reason about current state
            thought = self._generate_thought(task_description, state, iteration)
            self.reasoning_history.append(ReasoningStep(
                step_type=ReasoningStepType.THOUGHT,
                content=thought,
                timestamp=datetime.now()
            ))

            # Check if task is complete
            if completion_checker(state):
                break

            # ACTION: Decide and execute action
            action_result = self._select_and_execute_action(
                thought, state, available_actions
            )
            self.reasoning_history.append(ReasoningStep(
                step_type=ReasoningStepType.ACTION,
                content=json.dumps(action_result.get("action", {})),
                timestamp=datetime.now()
            ))

            # OBSERVATION: Record observation
            observation = action_result.get("observation", "No observation")
            self.reasoning_history.append(ReasoningStep(
                step_type=ReasoningStepType.OBSERVATION,
                content=observation,
                timestamp=datetime.now()
            ))

            # Update state with action results
            state.update(action_result.get("state_updates", {}))

            iteration += 1

        # Add final reasoning history to state
        state["reasoning_history"] = [
            {
                "type": step.step_type.value,
                "content": step.content,
                "timestamp": step.timestamp.isoformat()
            }
            for step in self.reasoning_history
        ]
        state["reasoning_iterations"] = iteration

        return state

    def _generate_thought(
        self,
        task: str,
        state: Dict[str, Any],
        iteration: int
    ) -> str:
        """Generate a thought about the current state."""
        prompt = f"""You are analyzing content for moderation. Think step by step.

        Task: {task}
        Current Iteration: {iteration + 1}/{self.max_iterations}

        Current State Summary:
        - Content: {state.get('content_text', 'N/A')[:200]}
        - Status: {state.get('status', 'unknown')}
        - Previous decisions: {len(state.get('agent_decisions', []))}

        Reasoning History:
        {self._format_reasoning_history()}

        Think about:
        1. What have we learned so far?
        2. What is the next logical step?
        3. What information is still needed?
        4. Are there any concerns or edge cases?

        Provide your thought in 2-3 sentences:"""

        response = self.llm.invoke(prompt)
        return response.content

    def _select_and_execute_action(
        self,
        thought: str,
        state: Dict[str, Any],
        available_actions: List[Callable]
    ) -> Dict[str, Any]:
        """Select and execute the best action based on reasoning."""
        # For now, return a placeholder
        # In production, this would intelligently select from available_actions
        return {
            "action": {"name": "continue", "params": {}},
            "observation": f"Completed reasoning step based on thought: {thought[:100]}",
            "state_updates": {"last_thought": thought}
        }

    def _format_reasoning_history(self) -> str:
        """Format reasoning history for prompts."""
        if not self.reasoning_history:
            return "No previous reasoning steps."

        recent_steps = self.reasoning_history[-6:]  # Last 6 steps
        formatted = []
        for step in recent_steps:
            formatted.append(f"- {step.step_type.value.upper()}: {step.content[:100]}")
        return "\n".join(formatted)


class PlanExecuteAgent:
    """
    Implements the Plan-and-Execute pattern.

    Instead of following a fixed pipeline, this agent:
    1. Analyzes the content
    2. Creates a dynamic plan
    3. Executes the plan steps
    4. Adapts the plan if needed
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize the Plan-Execute agent."""
        self.llm = llm
        self.execution_history: List[Dict[str, Any]] = []

    def create_plan(self, content_state: ContentState) -> ExecutionPlan:
        """
        Create a moderation plan based on content analysis.

        Args:
            content_state: Current content state

        Returns:
            ExecutionPlan with steps to execute
        """
        content_text = content_state.get("content_text", "")
        content_type = content_state.get("content_type", "text")
        user_profile = content_state.get("user_profile", {})

        prompt = f"""You are a content moderation planning agent. Analyze this content and create a moderation plan.

        Content Type: {content_type}
        Content: {content_text[:500]}
        User Reputation: {user_profile.get('reputation_score', 0.5)}
        User History: {user_profile.get('total_violations', 0)} violations

        Based on this content, create a step-by-step moderation plan. Consider:
        1. Does this need toxicity analysis?
        2. Does this need policy violation checking?
        3. Should we check user reputation?
        4. Does this need human review?
        5. What's the priority order?

        Provide a JSON plan with this structure:
        {{
            "reasoning": "Why this plan is appropriate",
            "confidence": 0.0-1.0,
            "steps": [
                {{
                    "agent": "agent_name",
                    "reason": "why this agent is needed",
                    "priority": 1-5,
                    "required": true/false
                }}
            ]
        }}

        Plan:"""

        response = self.llm.invoke(prompt)

        # Parse the LLM response
        try:
            plan_data = self._extract_json_from_response(response.content)

            return ExecutionPlan(
                plan_id=f"plan_{datetime.now().timestamp()}",
                steps=plan_data.get("steps", []),
                reasoning=plan_data.get("reasoning", ""),
                confidence=plan_data.get("confidence", 0.7),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        except Exception as e:
            # Fallback to default plan
            return self._create_default_plan()

    def execute_plan(
        self,
        plan: ExecutionPlan,
        state: ContentState,
        agent_registry: Dict[str, Callable]
    ) -> ContentState:
        """
        Execute the moderation plan.

        Args:
            plan: ExecutionPlan to execute
            state: Current content state
            agent_registry: Dictionary mapping agent names to callable agents

        Returns:
            Updated content state
        """
        state["execution_plan"] = {
            "plan_id": plan.plan_id,
            "reasoning": plan.reasoning,
            "total_steps": len(plan.steps)
        }

        # Sort steps by priority
        sorted_steps = sorted(plan.steps, key=lambda x: x.get("priority", 5))

        for i, step in enumerate(sorted_steps):
            agent_name = step.get("agent")
            required = step.get("required", True)

            # Execute agent if available
            if agent_name in agent_registry:
                try:
                    agent_func = agent_registry[agent_name]
                    state = agent_func(state)

                    self.execution_history.append({
                        "step": i + 1,
                        "agent": agent_name,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    if required:
                        state["requires_human_review"] = True
                        state["plan_execution_error"] = str(e)
                        break
                    else:
                        # Skip optional steps on error
                        self.execution_history.append({
                            "step": i + 1,
                            "agent": agent_name,
                            "status": "skipped",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })

        state["execution_history"] = self.execution_history
        return state

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx + 1]
            return json.loads(json_str)

        raise ValueError("No valid JSON found in response")

    def _create_default_plan(self) -> ExecutionPlan:
        """Create a default fallback plan."""
        return ExecutionPlan(
            plan_id=f"default_plan_{datetime.now().timestamp()}",
            steps=[
                {"agent": "content_analysis", "reason": "Analyze content", "priority": 1, "required": True},
                {"agent": "toxicity_detection", "reason": "Check toxicity", "priority": 2, "required": True},
                {"agent": "policy_check", "reason": "Check policies", "priority": 3, "required": True},
                {"agent": "reputation_scoring", "reason": "Evaluate user", "priority": 4, "required": False},
                {"agent": "action_enforcement", "reason": "Take action", "priority": 5, "required": True}
            ],
            reasoning="Default moderation pipeline",
            confidence=0.7,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )


class ReflexionCritic:
    """
    Implements the Reflexion pattern for self-correction.

    The agent:
    1. Makes a decision
    2. Critiques its own decision
    3. Identifies errors or improvements
    4. Corrects the decision if needed
    """

    def __init__(self, llm: ChatGoogleGenerativeAI, max_reflections: int = 2):
        """
        Initialize Reflexion critic.

        Args:
            llm: Language model for critique
            max_reflections: Maximum number of self-correction rounds
        """
        self.llm = llm
        self.max_reflections = max_reflections
        self.reflection_history: List[ReasoningStep] = []

    def critique_decision(
        self,
        decision: AgentDecision,
        content_state: ContentState,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Critique an agent's decision.

        Args:
            decision: The decision to critique
            content_state: Current content state
            context: Additional context for critique

        Returns:
            Dictionary with critique and improvement suggestions
        """
        prompt = f"""You are a critical evaluator of content moderation decisions. Analyze this decision.

        Content: {content_state.get('content_text', '')[:300]}
        Agent: {decision.agent_name}
        Decision: {decision.decision.value}
        Reasoning: {decision.reasoning}
        Confidence: {decision.confidence}

        Previous Decisions:
        {self._format_previous_decisions(content_state.get('agent_decisions', []))}

        Critique this decision by considering:
        1. Is the reasoning sound and logical?
        2. Does the decision match the evidence?
        3. Are there any contradictions with previous agent decisions?
        4. Is the confidence level appropriate?
        5. Are there any edge cases or concerns not addressed?
        6. Could this decision be biased or unfair?

        Provide critique in JSON format:
        {{
            "is_sound": true/false,
            "confidence_assessment": "appropriate/too_high/too_low",
            "concerns": ["list", "of", "concerns"],
            "improvements": ["suggested", "improvements"],
            "should_revise": true/false,
            "revised_decision": "decision_type if should_revise",
            "critique_reasoning": "detailed explanation"
        }}

        Critique:"""

        response = self.llm.invoke(prompt)

        try:
            critique = self._extract_json_from_response(response.content)

            self.reflection_history.append(ReasoningStep(
                step_type=ReasoningStepType.CRITIQUE,
                content=json.dumps(critique),
                timestamp=datetime.now(),
                confidence=critique.get("confidence_assessment", "appropriate") == "appropriate"
            ))

            return critique
        except Exception as e:
            # Fallback critique
            return {
                "is_sound": True,
                "confidence_assessment": "appropriate",
                "concerns": [],
                "improvements": [],
                "should_revise": False,
                "critique_reasoning": f"Error in critique: {str(e)}"
            }

    def self_correct(
        self,
        decision: AgentDecision,
        content_state: ContentState,
        original_agent_func: Callable
    ) -> AgentDecision:
        """
        Apply self-correction through reflection.

        Args:
            decision: Original decision
            content_state: Current content state
            original_agent_func: The agent function to re-run if needed

        Returns:
            Corrected decision (or original if no correction needed)
        """
        reflection_count = 0
        current_decision = decision

        while reflection_count < self.max_reflections:
            critique = self.critique_decision(current_decision, content_state)

            if not critique.get("should_revise", False):
                # Decision is good, no correction needed
                current_decision.metadata = current_decision.metadata or {}
                current_decision.metadata["reflection_count"] = reflection_count
                current_decision.metadata["critique"] = critique
                break

            # Generate improved decision
            improvement_prompt = f"""Based on this critique, provide an improved decision.

            Original Decision: {current_decision.decision.value}
            Original Reasoning: {current_decision.reasoning}

            Critique:
            {json.dumps(critique, indent=2)}

            Provide an improved decision with better reasoning. Format:
            {{
                "decision": "approve/remove/warn/flag/needs_review",
                "reasoning": "improved reasoning addressing the concerns",
                "confidence": 0.0-1.0
            }}

            Improved Decision:"""

            response = self.llm.invoke(improvement_prompt)

            try:
                improved = self._extract_json_from_response(response.content)

                # Update decision
                current_decision.decision = DecisionType(improved.get("decision", "needs_review"))
                current_decision.reasoning = improved.get("reasoning", current_decision.reasoning)
                current_decision.confidence = improved.get("confidence", current_decision.confidence)
                current_decision.metadata = current_decision.metadata or {}
                current_decision.metadata["self_corrected"] = True
                current_decision.metadata["correction_iteration"] = reflection_count + 1

            except Exception:
                # If improvement fails, keep original
                break

            reflection_count += 1

        # Add reflection history to decision
        current_decision.metadata = current_decision.metadata or {}
        current_decision.metadata["reflections"] = [
            {
                "type": step.step_type.value,
                "content": step.content,
                "timestamp": step.timestamp.isoformat()
            }
            for step in self.reflection_history
        ]

        return current_decision

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx + 1]
            return json.loads(json_str)

        raise ValueError("No valid JSON found in response")

    def _format_previous_decisions(self, decisions: List[AgentDecision]) -> str:
        """Format previous decisions for context."""
        if not decisions:
            return "No previous decisions"

        formatted = []
        for i, dec in enumerate(decisions[-3:]):  # Last 3 decisions
            formatted.append(
                f"{i+1}. {dec.agent_name}: {dec.decision.value} "
                f"(confidence: {dec.confidence:.2f})"
            )
        return "\n".join(formatted)