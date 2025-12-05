"""
Guardrails system for safe multi-agent operation.

This module implements:
1. Loop limits - Prevent infinite reasoning loops
2. Hallucination detection - Detect LLM hallucinations
3. Cost budgets - Enforce spending limits
4. Safety constraints - Ensure safe operation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime 
from dataclasses import dataclass, field
from enum import Enum
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from ..core.models import AgentDecision, ContentState


class GuardrailViolation(Enum):
    """Types of guardrail violations."""
    LOOP_LIMIT_EXCEEDED = "loop_limit_exceeded"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    HALLUCINATION_DETECTED = "hallucination_detected"
    INCONSISTENT_REASONING = "inconsistent_reasoning"
    LOW_CONFIDENCE_CASCADE = "low_confidence_cascade"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    RATE_LIMIT_VIOLATED = "rate_limit_violated"


@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""
    max_reasoning_iterations: int = 10
    max_agent_calls: int = 20
    max_cost_usd: float = 1.0
    max_execution_time_seconds: int = 300
    min_confidence_threshold: float = 0.3
    hallucination_check_enabled: bool = True
    consistency_check_enabled: bool = True
    require_justification: bool = True


@dataclass
class GuardrailViolationRecord:
    """Record of a guardrail violation."""
    violation_type: GuardrailViolation
    message: str
    timestamp: datetime
    agent_name: Optional[str] = None
    content_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"  # warning, error, critical


class LoopGuard:
    """
    Prevent infinite loops in reasoning systems.

    Monitors:
    - Number of reasoning iterations
    - Repeated states
    - Circular dependencies
    """

    def __init__(self, max_iterations: int = 10):
        """
        Initialize loop guard.

        Args:
            max_iterations: Maximum allowed iterations
        """
        self.max_iterations = max_iterations
        self.iteration_counts: Dict[str, int] = {}
        self.state_history: Dict[str, List[str]] = {}
        self.violations: List[GuardrailViolationRecord] = []

    def check_iteration_limit(self, task_id: str, current_iteration: int) -> bool:
        """
        Check if iteration limit is exceeded.

        Args:
            task_id: Unique task identifier
            current_iteration: Current iteration number

        Returns:
            True if within limit, False if exceeded
        """
        if current_iteration >= self.max_iterations:
            self.violations.append(GuardrailViolationRecord(
                violation_type=GuardrailViolation.LOOP_LIMIT_EXCEEDED,
                message=f"Iteration limit exceeded: {current_iteration}/{self.max_iterations}",
                timestamp=datetime.now(),
                metadata={"task_id": task_id, "iteration": current_iteration},
                severity="error"
            ))
            return False

        return True

    def check_state_repetition(self, task_id: str, state_signature: str) -> bool:
        """
        Check for repeated states (potential loop).

        Args:
            task_id: Unique task identifier
            state_signature: Hash or signature of current state

        Returns:
            True if state is new, False if repeated
        """
        if task_id not in self.state_history:
            self.state_history[task_id] = []

        # Check if we've seen this exact state before
        if state_signature in self.state_history[task_id]:
            self.violations.append(GuardrailViolationRecord(
                violation_type=GuardrailViolation.LOOP_LIMIT_EXCEEDED,
                message=f"State repetition detected - potential infinite loop",
                timestamp=datetime.now(),
                metadata={"task_id": task_id, "state": state_signature},
                severity="error"
            ))
            return False

        # Add state to history
        self.state_history[task_id].append(state_signature)

        # Limit history size
        if len(self.state_history[task_id]) > 50:
            self.state_history[task_id] = self.state_history[task_id][-50:]

        return True

    def reset(self, task_id: str):
        """Reset tracking for a task."""
        if task_id in self.iteration_counts:
            del self.iteration_counts[task_id]
        if task_id in self.state_history:
            del self.state_history[task_id]


class BudgetGuard:
    """
    Enforce cost budgets for LLM usage.

    Monitors:
    - Total cost
    - Cost per operation
    - Cost alerts and limits
    """

    def __init__(self, max_cost_usd: float = 1.0):
        """
        Initialize budget guard.

        Args:
            max_cost_usd: Maximum allowed cost in USD
        """
        self.max_cost_usd = max_cost_usd
        self.current_cost_usd = 0.0
        self.operation_costs: List[Dict[str, Any]] = []
        self.violations: List[GuardrailViolationRecord] = []
        self.warning_threshold = max_cost_usd * 0.8  # Warn at 80%

    def check_budget(self, operation_cost: float, operation_name: str = "") -> bool:
        """
        Check if operation would exceed budget.

        Args:
            operation_cost: Cost of the operation in USD
            operation_name: Name of the operation

        Returns:
            True if within budget, False if would exceed
        """
        projected_cost = self.current_cost_usd + operation_cost

        # Check if would exceed budget
        if projected_cost > self.max_cost_usd:
            self.violations.append(GuardrailViolationRecord(
                violation_type=GuardrailViolation.COST_BUDGET_EXCEEDED,
                message=f"Cost budget exceeded: ${projected_cost:.6f} > ${self.max_cost_usd:.6f}",
                timestamp=datetime.now(),
                metadata={
                    "operation": operation_name,
                    "operation_cost": operation_cost,
                    "current_cost": self.current_cost_usd,
                    "max_cost": self.max_cost_usd
                },
                severity="critical"
            ))
            return False

        # Warning if approaching limit
        if projected_cost > self.warning_threshold and self.current_cost_usd <= self.warning_threshold:
            self.violations.append(GuardrailViolationRecord(
                violation_type=GuardrailViolation.COST_BUDGET_EXCEEDED,
                message=f"Approaching cost budget: ${projected_cost:.6f} / ${self.max_cost_usd:.6f}",
                timestamp=datetime.now(),
                metadata={
                    "operation": operation_name,
                    "current_cost": projected_cost,
                    "max_cost": self.max_cost_usd
                },
                severity="warning"
            ))

        return True

    def record_cost(self, operation_cost: float, operation_name: str = ""):
        """
        Record an operation cost.

        Args:
            operation_cost: Cost in USD
            operation_name: Name of the operation
        """
        self.current_cost_usd += operation_cost
        self.operation_costs.append({
            "operation": operation_name,
            "cost": operation_cost,
            "timestamp": datetime.now().isoformat()
        })

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.max_cost_usd - self.current_cost_usd)

    def get_budget_summary(self) -> Dict[str, Any]:
        """Get budget summary."""
        return {
            "max_budget_usd": self.max_cost_usd,
            "current_cost_usd": round(self.current_cost_usd, 6),
            "remaining_usd": round(self.get_remaining_budget(), 6),
            "utilization_percent": round(
                (self.current_cost_usd / self.max_cost_usd) * 100, 2
            ),
            "total_operations": len(self.operation_costs),
            "violations": len([v for v in self.violations if v.severity in ["error", "critical"]])
        }


class HallucinationDetector:
    """
    Detect potential hallucinations in LLM outputs.

    Checks for:
    - Contradictions with input
    - Fabricated information
    - Inconsistent reasoning
    - Overconfident false statements
    """

    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """
        Initialize hallucination detector.

        Args:
            llm: Optional LLM for advanced detection
        """
        self.llm = llm
        self.violations: List[GuardrailViolationRecord] = []

    def check_for_hallucination(
        self,
        decision: AgentDecision,
        content_state: ContentState,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if decision contains hallucinations.

        Args:
            decision: Agent decision to check
            content_state: Content state for context
            context: Additional context

        Returns:
            Dictionary with detection results
        """
        issues = []

        # Check 1: Contradiction detection
        contradictions = self._check_contradictions(decision, content_state)
        if contradictions:
            issues.extend(contradictions)

        # Check 2: Unsupported claims
        unsupported = self._check_unsupported_claims(decision, content_state)
        if unsupported:
            issues.extend(unsupported)

        # Check 3: Confidence vs. evidence mismatch
        confidence_issues = self._check_confidence_mismatch(decision)
        if confidence_issues:
            issues.extend(confidence_issues)

        # Check 4: Fabricated entities or facts
        fabricated = self._check_fabricated_content(decision, content_state)
        if fabricated:
            issues.extend(fabricated)

        # Record violations
        if issues:
            self.violations.append(GuardrailViolationRecord(
                violation_type=GuardrailViolation.HALLUCINATION_DETECTED,
                message=f"Potential hallucination detected in {decision.agent_name}",
                timestamp=datetime.now(),
                agent_name=decision.agent_name,
                metadata={"issues": issues},
                severity="warning"
            ))

        return {
            "hallucination_detected": len(issues) > 0,
            "issue_count": len(issues),
            "issues": issues,
            "confidence_adjustment": -0.1 * len(issues)  # Reduce confidence
        }

    def _check_contradictions(
        self,
        decision: AgentDecision,
        content_state: ContentState
    ) -> List[str]:
        """Check for contradictions in reasoning."""
        issues = []

        reasoning = decision.reasoning.lower()
        content_text = content_state.get("content_text", "").lower()

        # Simple contradiction patterns
        if "no toxicity" in reasoning and "toxic" in content_text:
            issues.append("Claims no toxicity but content contains toxic language")

        if "no violations" in reasoning and decision.decision.value in ["remove", "warn"]:
            issues.append("Claims no violations but recommends removal/warning")

        return issues

    def _check_unsupported_claims(
        self,
        decision: AgentDecision,
        content_state: ContentState
    ) -> List[str]:
        """Check for claims not supported by evidence."""
        issues = []

        reasoning = decision.reasoning
        content_text = content_state.get("content_text", "")

        # Check if reasoning references things not in content
        # This is a simplified check - production would use more sophisticated NLP
        claimed_words = set(reasoning.lower().split())
        content_words = set(content_text.lower().split())

        # Look for specific claim patterns
        claim_patterns = [
            r"the user said ['\"]([^'\"]+)['\"]",
            r"contains the phrase ['\"]([^'\"]+)['\"]",
            r"explicitly mentions ['\"]([^'\"]+)['\"]"
        ]

        for pattern in claim_patterns:
            matches = re.findall(pattern, reasoning.lower())
            for match in matches:
                if match not in content_text.lower():
                    issues.append(f"Claims content contains '{match}' but it doesn't")

        return issues

    def _check_confidence_mismatch(self, decision: AgentDecision) -> List[str]:
        """Check if confidence matches the reasoning quality."""
        issues = []

        reasoning = decision.reasoning
        confidence = decision.confidence

        # High confidence with weak reasoning
        if confidence > 0.9 and len(reasoning) < 50:
            issues.append("Very high confidence with minimal reasoning")

        # High confidence with uncertainty words
        uncertainty_words = ["maybe", "possibly", "might", "could be", "perhaps", "unclear"]
        if confidence > 0.8 and any(word in reasoning.lower() for word in uncertainty_words):
            issues.append("High confidence with uncertain language")

        # Low confidence with definitive language
        definitive_words = ["definitely", "certainly", "clearly", "obviously", "without doubt"]
        if confidence < 0.5 and any(word in reasoning.lower() for word in definitive_words):
            issues.append("Low confidence with definitive language")

        return issues

    def _check_fabricated_content(
        self,
        decision: AgentDecision,
        content_state: ContentState
    ) -> List[str]:
        """Check for fabricated information."""
        issues = []

        reasoning = decision.reasoning
        content_text = content_state.get("content_text", "")

        # Check for specific numbers/statistics that aren't in content
        number_pattern = r'\b\d+(?:\.\d+)?%?\b'
        reasoning_numbers = set(re.findall(number_pattern, reasoning))
        content_numbers = set(re.findall(number_pattern, content_text))

        fabricated_numbers = reasoning_numbers - content_numbers
        if fabricated_numbers and len(fabricated_numbers) > 2:
            issues.append(f"Contains numbers not in source: {fabricated_numbers}")

        return issues


class ConsistencyChecker:
    """
    Check consistency across agent decisions.

    Ensures:
    - Decisions don't contradict each other
    - Reasoning is logically consistent
    - Escalation paths make sense
    """

    def __init__(self):
        """Initialize consistency checker."""
        self.violations: List[GuardrailViolationRecord] = []

    def check_decision_consistency(
        self,
        decisions: List[AgentDecision],
        content_state: ContentState
    ) -> Dict[str, Any]:
        """
        Check consistency across multiple agent decisions.

        Args:
            decisions: List of agent decisions
            content_state: Content state

        Returns:
            Dictionary with consistency check results
        """
        issues = []

        if len(decisions) < 2:
            return {"consistent": True, "issues": []}

        # Check 1: Conflicting decisions
        conflict_issues = self._check_conflicting_decisions(decisions)
        issues.extend(conflict_issues)

        # Check 2: Confidence progression
        confidence_issues = self._check_confidence_progression(decisions)
        issues.extend(confidence_issues)

        # Check 3: Reasoning consistency
        reasoning_issues = self._check_reasoning_consistency(decisions)
        issues.extend(reasoning_issues)

        # Record violations
        if issues:
            self.violations.append(GuardrailViolationRecord(
                violation_type=GuardrailViolation.INCONSISTENT_REASONING,
                message=f"Inconsistencies detected across {len(decisions)} decisions",
                timestamp=datetime.now(),
                metadata={"issues": issues, "decision_count": len(decisions)},
                severity="warning"
            ))

        return {
            "consistent": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues
        }

    def _check_conflicting_decisions(self, decisions: List[AgentDecision]) -> List[str]:
        """Check for conflicting decisions."""
        issues = []

        # Check if earlier agent says approve but later says remove
        decision_values = [d.decision.value for d in decisions]

        if "approve" in decision_values and "remove" in decision_values:
            approve_idx = decision_values.index("approve")
            remove_idx = decision_values.index("remove")
            issues.append(
                f"Conflicting decisions: {decisions[approve_idx].agent_name} approved "
                f"but {decisions[remove_idx].agent_name} removed"
            )

        return issues

    def _check_confidence_progression(self, decisions: List[AgentDecision]) -> List[str]:
        """Check if confidence levels make sense."""
        issues = []

        # Check for cascading low confidence
        low_confidence_count = sum(1 for d in decisions if d.confidence < 0.5)
        if low_confidence_count >= 3:
            issues.append(
                f"Cascading low confidence: {low_confidence_count} agents "
                "have confidence < 0.5"
            )

        return issues

    def _check_reasoning_consistency(self, decisions: List[AgentDecision]) -> List[str]:
        """Check if reasoning is consistent."""
        issues = []

        # Simple check: if one agent says "no toxicity" and another says "high toxicity"
        reasonings = [d.reasoning.lower() for d in decisions]

        toxicity_mentions = []
        for i, reasoning in enumerate(reasonings):
            if "no toxicity" in reasoning or "not toxic" in reasoning:
                toxicity_mentions.append((i, "low"))
            elif "high toxicity" in reasoning or "very toxic" in reasoning:
                toxicity_mentions.append((i, "high"))

        if len(toxicity_mentions) >= 2:
            levels = [level for _, level in toxicity_mentions]
            if "low" in levels and "high" in levels:
                issues.append("Inconsistent toxicity assessment across agents")

        return issues


class GuardrailManager:
    """
    Unified guardrail management system.

    Combines all guardrails:
    - Loop limits
    - Cost budgets
    - Hallucination detection
    - Consistency checking
    """

    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
        llm: Optional[ChatGoogleGenerativeAI] = None
    ):
        """
        Initialize guardrail manager.

        Args:
            config: Guardrail configuration
            llm: Optional LLM for advanced checks
        """
        self.config = config or GuardrailConfig()
        self.loop_guard = LoopGuard(max_iterations=self.config.max_reasoning_iterations)
        self.budget_guard = BudgetGuard(max_cost_usd=self.config.max_cost_usd)
        self.hallucination_detector = HallucinationDetector(llm=llm)
        self.consistency_checker = ConsistencyChecker()

        self.all_violations: List[GuardrailViolationRecord] = []

    def check_all_guardrails(
        self,
        content_state: ContentState,
        current_iteration: int = 0,
        operation_cost: float = 0.0
    ) -> Dict[str, Any]:
        """
        Check all guardrails.

        Args:
            content_state: Current content state
            current_iteration: Current iteration number
            operation_cost: Cost of current operation

        Returns:
            Dictionary with guardrail check results
        """
        results = {
            "passed": True,
            "violations": [],
            "warnings": []
        }

        content_id = content_state.get("content_id", "unknown")

        # Check loop guard
        if not self.loop_guard.check_iteration_limit(content_id, current_iteration):
            results["passed"] = False
            results["violations"].append("Loop limit exceeded")

        # Check budget
        if not self.budget_guard.check_budget(operation_cost, f"iteration_{current_iteration}"):
            results["passed"] = False
            results["violations"].append("Cost budget exceeded")

        # Check decisions for hallucinations and consistency
        decisions = content_state.get("agent_decisions", [])
        if decisions and self.config.hallucination_check_enabled:
            for decision in decisions[-1:]:  # Check most recent
                hallucination_result = self.hallucination_detector.check_for_hallucination(
                    decision, content_state
                )
                if hallucination_result["hallucination_detected"]:
                    results["warnings"].append(
                        f"Potential hallucination in {decision.agent_name}"
                    )

        if len(decisions) > 1 and self.config.consistency_check_enabled:
            consistency_result = self.consistency_checker.check_decision_consistency(
                decisions, content_state
            )
            if not consistency_result["consistent"]:
                results["warnings"].append("Inconsistent decisions detected")

        # Collect all violations
        all_violations = (
            self.loop_guard.violations +
            self.budget_guard.violations +
            self.hallucination_detector.violations +
            self.consistency_checker.violations
        )

        self.all_violations.extend(all_violations)
        results["violation_details"] = [
            {
                "type": v.violation_type.value,
                "message": v.message,
                "severity": v.severity,
                "timestamp": v.timestamp.isoformat()
            }
            for v in all_violations
        ]

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get guardrail summary."""
        return {
            "total_violations": len(self.all_violations),
            "budget_summary": self.budget_guard.get_budget_summary(),
            "violations_by_type": self._count_violations_by_type(),
            "violations_by_severity": self._count_violations_by_severity()
        }

    def _count_violations_by_type(self) -> Dict[str, int]:
        """Count violations by type."""
        counts = {}
        for violation in self.all_violations:
            vtype = violation.violation_type.value
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts

    def _count_violations_by_severity(self) -> Dict[str, int]:
        """Count violations by severity."""
        counts = {"warning": 0, "error": 0, "critical": 0}
        for violation in self.all_violations:
            counts[violation.severity] = counts.get(violation.severity, 0) + 1
        return counts
