"""
Evaluation system for multi-agent content moderation.

This module implements:
1. LLM-as-Judge - Evaluate decision quality using LLM
2. Cost and latency tracking - Monitor resource usage
3. A/B testing framework - Compare different moderation strategies
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import sqlite3
from contextlib import contextmanager

from langchain_google_genai import ChatGoogleGenerativeAI
from ..core.models import AgentDecision, ContentState


class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CONSISTENCY = "consistency"
    REASONING_QUALITY = "reasoning_quality"
    FAIRNESS = "fairness"
    LATENCY = "latency"
    COST = "cost"


@dataclass
class CostMetrics:
    """Track cost and resource usage for LLM calls."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    api_calls: int = 0
    agent_costs: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Pricing (Gemini Flash pricing as of 2024)
    INPUT_TOKEN_COST = 0.075 / 1_000_000  # $0.075 per 1M tokens
    OUTPUT_TOKEN_COST = 0.30 / 1_000_000  # $0.30 per 1M tokens


@dataclass
class LatencyMetrics:
    """Track latency for agent execution."""
    agent_name: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeEvaluation:
    """Result from LLM-as-Judge evaluation."""
    decision_id: str
    overall_score: float  # 0-10
    accuracy_score: float
    reasoning_score: float
    consistency_score: float
    fairness_score: float
    strengths: List[str]
    weaknesses: List[str]
    improvements: List[str]
    confidence: float
    judge_reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


class CostTracker:
    """
    Track API costs and token usage for LLM calls.

    Monitors:
    - Total tokens used (input + output)
    - Cost per agent
    - Cost per moderation decision
    - Budget alerts
    """

    def __init__(self, budget_limit_usd: Optional[float] = None):
        """
        Initialize cost tracker.

        Args:
            budget_limit_usd: Optional budget limit in USD
        """
        self.metrics = CostMetrics()
        self.budget_limit = budget_limit_usd
        self.session_costs: List[Dict[str, Any]] = []

    def track_llm_call(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini-flash"
    ) -> Dict[str, Any]:
        """
        Track a single LLM API call.

        Args:
            agent_name: Name of the agent making the call
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name

        Returns:
            Dictionary with call cost information
        """
        # Calculate cost
        input_cost = input_tokens * CostMetrics.INPUT_TOKEN_COST
        output_cost = output_tokens * CostMetrics.OUTPUT_TOKEN_COST
        total_call_cost = input_cost + output_cost

        # Update metrics
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_cost_usd += total_call_cost
        self.metrics.api_calls += 1

        # Track per-agent costs
        if agent_name not in self.metrics.agent_costs:
            self.metrics.agent_costs[agent_name] = 0.0
        self.metrics.agent_costs[agent_name] += total_call_cost

        # Record session
        call_record = {
            "agent": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": total_call_cost,
            "timestamp": datetime.now().isoformat()
        }
        self.session_costs.append(call_record)

        # Check budget
        if self.budget_limit and self.metrics.total_cost_usd > self.budget_limit:
            call_record["budget_exceeded"] = True

        return call_record

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "total_cost_usd": round(self.metrics.total_cost_usd, 6),
            "total_tokens": self.metrics.total_input_tokens + self.metrics.total_output_tokens,
            "input_tokens": self.metrics.total_input_tokens,
            "output_tokens": self.metrics.total_output_tokens,
            "api_calls": self.metrics.api_calls,
            "avg_cost_per_call": round(
                self.metrics.total_cost_usd / max(self.metrics.api_calls, 1), 6
            ),
            "agent_costs": {
                agent: round(cost, 6)
                for agent, cost in self.metrics.agent_costs.items()
            },
            "budget_limit": self.budget_limit,
            "budget_remaining": round(
                (self.budget_limit or 0) - self.metrics.total_cost_usd, 6
            ) if self.budget_limit else None,
            "budget_exceeded": (
                self.metrics.total_cost_usd > self.budget_limit
                if self.budget_limit else False
            )
        }

    def reset(self):
        """Reset cost tracking."""
        self.metrics = CostMetrics()
        self.session_costs = []


class LatencyTracker:
    """
    Track execution latency for agents and overall pipeline.
    """

    def __init__(self):
        """Initialize latency tracker."""
        self.measurements: List[LatencyMetrics] = []
        self.active_timers: Dict[str, float] = {}

    def start_timer(self, agent_name: str) -> float:
        """
        Start timing an agent execution.

        Args:
            agent_name: Name of the agent

        Returns:
            Start timestamp
        """
        start_time = time.time()
        self.active_timers[agent_name] = start_time
        return start_time

    def end_timer(
        self,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LatencyMetrics:
        """
        End timing and record metrics.

        Args:
            agent_name: Name of the agent
            metadata: Optional metadata

        Returns:
            LatencyMetrics object
        """
        end_time = time.time()
        start_time = self.active_timers.get(agent_name, end_time)

        duration_ms = (end_time - start_time) * 1000

        metric = LatencyMetrics(
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )

        self.measurements.append(metric)

        # Clean up timer
        if agent_name in self.active_timers:
            del self.active_timers[agent_name]

        return metric

    def get_summary(self) -> Dict[str, Any]:
        """Get latency summary."""
        if not self.measurements:
            return {
                "total_measurements": 0,
                "total_time_ms": 0,
                "avg_time_ms": 0,
                "agent_latencies": {}
            }

        total_time = sum(m.duration_ms for m in self.measurements)
        agent_times: Dict[str, List[float]] = {}

        for measurement in self.measurements:
            if measurement.agent_name not in agent_times:
                agent_times[measurement.agent_name] = []
            agent_times[measurement.agent_name].append(measurement.duration_ms)

        return {
            "total_measurements": len(self.measurements),
            "total_time_ms": round(total_time, 2),
            "avg_time_ms": round(total_time / len(self.measurements), 2),
            "agent_latencies": {
                agent: {
                    "calls": len(times),
                    "total_ms": round(sum(times), 2),
                    "avg_ms": round(sum(times) / len(times), 2),
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2)
                }
                for agent, times in agent_times.items()
            }
        }

    def reset(self):
        """Reset latency tracking."""
        self.measurements = []
        self.active_timers = {}


class LLMJudge:
    """
    Use LLM to evaluate moderation decision quality.

    Evaluates:
    - Accuracy: Is the decision correct?
    - Reasoning: Is the reasoning sound?
    - Consistency: Does it align with previous decisions?
    - Fairness: Is it fair and unbiased?
    """

    def __init__(self, llm: ChatGoogleGenerativeAI, cost_tracker: Optional[CostTracker] = None):
        """
        Initialize LLM Judge.

        Args:
            llm: Language model for evaluation
            cost_tracker: Optional cost tracker
        """
        self.llm = llm
        self.cost_tracker = cost_tracker

    def evaluate_decision(
        self,
        decision: AgentDecision,
        content_state: ContentState,
        ground_truth: Optional[str] = None,
        previous_decisions: Optional[List[AgentDecision]] = None
    ) -> JudgeEvaluation:
        """
        Evaluate a moderation decision.

        Args:
            decision: The decision to evaluate
            content_state: Content state context
            ground_truth: Optional ground truth decision
            previous_decisions: Previous decisions for consistency check

        Returns:
            JudgeEvaluation with scores and feedback
        """
        content_text = content_state.get("content_text", "")
        user_profile = content_state.get("user_profile", {})

        prompt = f"""You are an expert evaluator of content moderation decisions.
        Evaluate this moderation decision on a scale of 1-10.

        CONTENT:
        {content_text[:500]}

        USER CONTEXT:
        - Reputation Score: {user_profile.get('reputation_score', 0.5)}
        - Violations: {user_profile.get('total_violations', 0)}
        - Account Age: {user_profile.get('account_age_days', 0)} days

        DECISION TO EVALUATE:
        - Agent: {decision.agent_name}
        - Decision: {decision.decision.value}
        - Reasoning: {decision.reasoning}
        - Confidence: {decision.confidence}
        - Flags: {', '.join(decision.flags) if decision.flags else 'None'}

        {f'GROUND TRUTH: {ground_truth}' if ground_truth else ''}

        {self._format_previous_decisions(previous_decisions) if previous_decisions else ''}

        Evaluate on these criteria:
        1. ACCURACY (1-10): Is the decision appropriate for this content?
        2. REASONING (1-10): Is the reasoning logical and well-supported?
        3. CONSISTENCY (1-10): Is it consistent with similar cases?
        4. FAIRNESS (1-10): Is the decision fair and unbiased?

        Provide your evaluation in JSON format:
        {{
            "overall_score": 0-10,
            "accuracy_score": 0-10,
            "reasoning_score": 0-10,
            "consistency_score": 0-10,
            "fairness_score": 0-10,
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"],
            "improvements": ["suggested", "improvements"],
            "confidence": 0.0-1.0,
            "judge_reasoning": "detailed explanation of scores"
        }}

        Evaluation:"""

        # Track cost if tracker available
        if self.cost_tracker:
            # Estimate tokens (rough approximation)
            input_tokens = len(prompt.split()) * 1.3
            self.cost_tracker.track_llm_call(
                agent_name="LLM_Judge",
                input_tokens=int(input_tokens),
                output_tokens=300,  # Estimated
                model="gemini-flash"
            )

        response = self.llm.invoke(prompt)

        try:
            eval_data = self._extract_json_from_response(response.content)

            return JudgeEvaluation(
                decision_id=f"{decision.agent_name}_{datetime.now().timestamp()}",
                overall_score=eval_data.get("overall_score", 5.0),
                accuracy_score=eval_data.get("accuracy_score", 5.0),
                reasoning_score=eval_data.get("reasoning_score", 5.0),
                consistency_score=eval_data.get("consistency_score", 5.0),
                fairness_score=eval_data.get("fairness_score", 5.0),
                strengths=eval_data.get("strengths", []),
                weaknesses=eval_data.get("weaknesses", []),
                improvements=eval_data.get("improvements", []),
                confidence=eval_data.get("confidence", 0.7),
                judge_reasoning=eval_data.get("judge_reasoning", "")
            )
        except Exception as e:
            # Fallback evaluation
            return JudgeEvaluation(
                decision_id=f"{decision.agent_name}_{datetime.now().timestamp()}",
                overall_score=7.0,
                accuracy_score=7.0,
                reasoning_score=7.0,
                consistency_score=7.0,
                fairness_score=7.0,
                strengths=["Decision made"],
                weaknesses=[f"Evaluation error: {str(e)}"],
                improvements=["Fix evaluation system"],
                confidence=0.5,
                judge_reasoning="Automated fallback evaluation"
            )

    def batch_evaluate(
        self,
        decisions: List[Tuple[AgentDecision, ContentState]],
        ground_truths: Optional[List[str]] = None
    ) -> List[JudgeEvaluation]:
        """
        Evaluate multiple decisions in batch.

        Args:
            decisions: List of (decision, content_state) tuples
            ground_truths: Optional list of ground truth decisions

        Returns:
            List of JudgeEvaluation objects
        """
        evaluations = []

        for i, (decision, state) in enumerate(decisions):
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            evaluation = self.evaluate_decision(decision, state, ground_truth)
            evaluations.append(evaluation)

        return evaluations

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx + 1]
            return json.loads(json_str)

        raise ValueError("No valid JSON found in response")

    def _format_previous_decisions(self, decisions: List[AgentDecision]) -> str:
        """Format previous decisions for consistency checking."""
        formatted = ["PREVIOUS DECISIONS FOR CONSISTENCY CHECK:"]
        for i, dec in enumerate(decisions[-3:]):
            formatted.append(
                f"{i+1}. {dec.agent_name}: {dec.decision.value} - {dec.reasoning[:100]}"
            )
        return "\n".join(formatted)


class ABTestingFramework:
    """
    A/B testing framework for comparing moderation strategies.

    Allows testing:
    - Different agent configurations
    - Different prompts
    - Different thresholds
    - Different workflows
    """

    def __init__(self, db_path: str = "ab_testing.db"):
        """
        Initialize A/B testing framework.

        Args:
            db_path: Path to SQLite database for storing test results
        """
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Initialize A/B testing database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    variant_a_config TEXT,
                    variant_b_config TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    decision TEXT,
                    confidence REAL,
                    latency_ms REAL,
                    cost_usd REAL,
                    user_appealed INTEGER DEFAULT 0,
                    appeal_upheld INTEGER DEFAULT 0,
                    judge_score REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES ab_experiments(experiment_id)
                )
            """)

    def create_experiment(
        self,
        name: str,
        description: str,
        variant_a_config: Dict[str, Any],
        variant_b_config: Dict[str, Any]
    ) -> str:
        """
        Create a new A/B test experiment.

        Args:
            name: Experiment name
            description: Experiment description
            variant_a_config: Configuration for variant A
            variant_b_config: Configuration for variant B

        Returns:
            experiment_id
        """
        experiment_id = f"exp_{datetime.now().timestamp()}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ab_experiments
                (experiment_id, name, description, variant_a_config, variant_b_config, start_date, status)
                VALUES (?, ?, ?, ?, ?, ?, 'active')
            """, (
                experiment_id,
                name,
                description,
                json.dumps(variant_a_config),
                json.dumps(variant_b_config),
                datetime.now().isoformat()
            ))

        return experiment_id

    def record_result(
        self,
        experiment_id: str,
        content_id: str,
        variant: str,
        decision: str,
        confidence: float,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        judge_score: Optional[float] = None
    ):
        """Record an A/B test result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ab_results
                (experiment_id, content_id, variant, decision, confidence, latency_ms, cost_usd, judge_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                content_id,
                variant,
                decision,
                confidence,
                latency_ms,
                cost_usd,
                judge_score
            ))

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get results for an A/B test experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with statistical comparison
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get results for both variants
            cursor.execute("""
                SELECT
                    variant,
                    COUNT(*) as total_decisions,
                    AVG(confidence) as avg_confidence,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(cost_usd) as total_cost_usd,
                    AVG(judge_score) as avg_judge_score,
                    SUM(user_appealed) as total_appeals,
                    SUM(appeal_upheld) as total_overturned
                FROM ab_results
                WHERE experiment_id = ?
                GROUP BY variant
            """, (experiment_id,))

            results = cursor.fetchall()

            variant_stats = {}
            for row in results:
                variant_stats[row["variant"]] = {
                    "total_decisions": row["total_decisions"],
                    "avg_confidence": round(row["avg_confidence"] or 0, 3),
                    "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
                    "total_cost_usd": round(row["total_cost_usd"] or 0, 6),
                    "avg_judge_score": round(row["avg_judge_score"] or 0, 2),
                    "appeal_rate": round(
                        (row["total_appeals"] or 0) / max(row["total_decisions"], 1), 3
                    ),
                    "overturn_rate": round(
                        (row["total_overturned"] or 0) / max(row["total_appeals"] or 1, 1), 3
                    )
                }

            return {
                "experiment_id": experiment_id,
                "variants": variant_stats,
                "recommendation": self._analyze_winner(variant_stats)
            }

    def _analyze_winner(self, variant_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze which variant is better."""
        if "A" not in variant_stats or "B" not in variant_stats:
            return {"winner": None, "reason": "Insufficient data"}

        a_stats = variant_stats["A"]
        b_stats = variant_stats["B"]

        # Score based on multiple factors
        a_score = (
            a_stats["avg_judge_score"] * 2 +
            (1 - a_stats["appeal_rate"]) * 3 +
            (1 - a_stats["overturn_rate"]) * 3 +
            a_stats["avg_confidence"]
        )

        b_score = (
            b_stats["avg_judge_score"] * 2 +
            (1 - b_stats["appeal_rate"]) * 3 +
            (1 - b_stats["overturn_rate"]) * 3 +
            b_stats["avg_confidence"]
        )

        winner = "A" if a_score > b_score else "B"
        confidence = abs(a_score - b_score) / max(a_score, b_score)

        return {
            "winner": winner,
            "confidence": round(confidence, 3),
            "reason": f"Variant {winner} scored higher on quality metrics",
            "a_score": round(a_score, 3),
            "b_score": round(b_score, 3)
        }
