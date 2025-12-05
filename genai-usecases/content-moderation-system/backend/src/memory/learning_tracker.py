"""
Learning Tracker - Coordinates Learning Across All Agents

Manages episodic and semantic memory for all agents and tracks
system-wide improvement metrics.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from .agent_episodic_memory import AgentEpisodicMemory
from .agent_semantic_memory import AgentSemanticMemory
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class LearningTracker:
    """
    Central coordinator for multi-agent learning.

    Responsibilities:
    - Manage episodic memory for each agent
    - Manage semantic memory for each agent
    - Track cross-agent learning patterns
    - Provide system-wide analytics
    """

    def __init__(self):
        """Initialize learning tracker."""
        self.agent_episodic_memories: Dict[str, AgentEpisodicMemory] = {}
        self.agent_semantic_memories: Dict[str, AgentSemanticMemory] = {}

        self.session_count = 0
        self.total_decisions = 0
        self.total_appeals = 0
        self.total_overturns = 0

        # Track improvement over time
        self.session_metrics: List[Dict[str, Any]] = []

    def get_or_create_episodic_memory(self, agent_name: str) -> AgentEpisodicMemory:
        """Get episodic memory for an agent (create if doesn't exist)."""
        if agent_name not in self.agent_episodic_memories:
            self.agent_episodic_memories[agent_name] = AgentEpisodicMemory(
                agent_name=agent_name,
                capacity=100
            )
        return self.agent_episodic_memories[agent_name]

    def get_or_create_semantic_memory(self, agent_name: str) -> AgentSemanticMemory:
        """Get semantic memory for an agent (create if doesn't exist)."""
        if agent_name not in self.agent_semantic_memories:
            self.agent_semantic_memories[agent_name] = AgentSemanticMemory(
                agent_name=agent_name,
                decay_factor=0.9
            )
        return self.agent_semantic_memories[agent_name]

    def record_decision(
        self,
        agent_name: str,
        content_text: str,
        toxicity_score: float,
        policy_violations: List[str],
        decision: str,
        confidence: float,
        outcome: str,
        context: str,
        metadata: Optional[Dict] = None
    ):
        """
        Record a moderation decision for learning.

        Args:
            agent_name: Name of the agent that made the decision
            content_text: The moderated content
            toxicity_score: Toxicity score
            policy_violations: List of violations
            decision: Agent's decision
            confidence: Confidence in decision
            outcome: What happened
            context: Context for semantic learning
            metadata: Additional information
        """
        # Store in episodic memory (will learn from appeals later)
        episodic = self.get_or_create_episodic_memory(agent_name)
        episodic.store_decision(
            content_text=content_text,
            toxicity_score=toxicity_score,
            policy_violations=policy_violations,
            decision=decision,
            confidence=confidence,
            outcome=outcome,
            was_appealed=False,  # Will update if appealed
            metadata=metadata
        )

        # Record pattern in semantic memory (assume success unless appealed)
        semantic = self.get_or_create_semantic_memory(agent_name)
        semantic.record_decision_outcome(
            context=context,
            action=decision,
            success=True,  # Will update if overturned
            confidence=confidence,
            metadata=metadata
        )

        self.total_decisions += 1

    def record_appeal(
        self,
        content_id: str,
        agent_name: str,
        original_decision: str,
        appeal_outcome: str,
        content_text: str = "",
        toxicity_score: float = 0.0,
        policy_violations: List[str] = None
    ):
        """
        Record an appeal outcome and update learning.

        This is critical for learning! When a decision is overturned,
        the agent learns that its approach needs adjustment.

        Args:
            content_id: ID of the content
            agent_name: Agent whose decision was appealed
            original_decision: What the agent decided
            appeal_outcome: Result (upheld, overturned, partial)
            content_text: The content (for updating episodic memory)
            toxicity_score: Toxicity score
            policy_violations: Violations detected
        """
        episodic = self.get_or_create_episodic_memory(agent_name)
        semantic = self.get_or_create_semantic_memory(agent_name)

        # Update episodic memory with appeal result
        if episodic.episodes:
            # Find and update the relevant episode
            for episode in reversed(episodic.episodes):
                if episode['decision'] == original_decision:
                    episode['was_appealed'] = True
                    episode['appeal_outcome'] = appeal_outcome
                    episode['decision_correct'] = (appeal_outcome == "upheld")
                    break

        # Update semantic memory
        was_overturned = appeal_outcome in ["overturned", "partial"]

        # Re-record the pattern with correct success value
        context = self._get_context_from_scores(toxicity_score, policy_violations or [])
        semantic.record_decision_outcome(
            context=context,
            action=original_decision,
            success=not was_overturned,  # Overturned = failure
            metadata={'appeal': True, 'appeal_outcome': appeal_outcome}
        )

        self.total_appeals += 1
        if was_overturned:
            self.total_overturns += 1

        # Learn threshold adjustments if overturned
        if was_overturned and appeal_outcome == "overturned":
            self._learn_from_mistake(
                agent_name, original_decision, toxicity_score, policy_violations or []
            )

    def _learn_from_mistake(
        self,
        agent_name: str,
        decision: str,
        toxicity_score: float,
        policy_violations: List[str]
    ):
        """
        Learn from an overturned decision by adjusting thresholds.

        Args:
            agent_name: Agent that made the mistake
            decision: What was decided
            toxicity_score: Toxicity score
            policy_violations: Violations detected
        """
        semantic = self.get_or_create_semantic_memory(agent_name)

        # If removed content was overturned, raise removal threshold
        if decision == "remove":
            threshold_name = "toxicity_removal_threshold"
            # Suggest slightly higher threshold (be more cautious)
            suggested = min(toxicity_score + 0.05, 0.95)
            semantic.learn_threshold(threshold_name, suggested, weight=1.0)

        # If approval was overturned, lower approval threshold
        elif decision == "approve":
            threshold_name = "toxicity_approval_threshold"
            # Suggest lower threshold (be stricter)
            suggested = max(toxicity_score - 0.05, 0.05)
            semantic.learn_threshold(threshold_name, suggested, weight=1.0)

    def _get_context_from_scores(
        self,
        toxicity_score: float,
        policy_violations: List[str]
    ) -> str:
        """
        Generate context string from scores for semantic learning.

        Args:
            toxicity_score: Toxicity score
            policy_violations: List of violations

        Returns:
            Context string (e.g., "toxicity_0.6_to_0.7_hate_speech")
        """
        # Toxicity bucket
        if toxicity_score < 0.3:
            tox_bucket = "toxicity_low"
        elif toxicity_score < 0.6:
            tox_bucket = "toxicity_medium"
        elif toxicity_score < 0.8:
            tox_bucket = "toxicity_high"
        else:
            tox_bucket = "toxicity_severe"

        # Primary violation (if any)
        violation = policy_violations[0] if policy_violations else "no_violation"

        return f"{tox_bucket}_{violation}"

    def get_similar_cases(
        self,
        agent_name: str,
        content_text: str,
        toxicity_score: float,
        policy_violations: List[str],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get similar past cases for an agent to learn from.

        Args:
            agent_name: Agent requesting similar cases
            content_text: Current content
            toxicity_score: Current toxicity
            policy_violations: Current violations
            k: Number of cases to retrieve

        Returns:
            List of similar past cases
        """
        episodic = self.get_or_create_episodic_memory(agent_name)
        return episodic.retrieve_similar_decisions(
            content_text, toxicity_score, policy_violations, k
        )

    def get_recommended_action(
        self,
        agent_name: str,
        toxicity_score: float,
        policy_violations: List[str]
    ) -> Optional[Tuple[str, float]]:
        """
        Get recommended action based on learned patterns.

        Args:
            agent_name: Agent requesting recommendation
            toxicity_score: Current toxicity score
            policy_violations: Current violations

        Returns:
            (recommended_action, confidence) or None
        """
        semantic = self.get_or_create_semantic_memory(agent_name)
        context = self._get_context_from_scores(toxicity_score, policy_violations)
        return semantic.get_best_action(context, min_samples=3)

    def adjust_confidence(
        self,
        agent_name: str,
        base_confidence: float,
        toxicity_score: float,
        policy_violations: List[str],
        planned_action: str
    ) -> float:
        """
        Adjust confidence based on learned patterns.

        Args:
            agent_name: Agent name
            base_confidence: Original confidence
            toxicity_score: Toxicity score
            policy_violations: Violations
            planned_action: What the agent plans to do

        Returns:
            Adjusted confidence
        """
        semantic = self.get_or_create_semantic_memory(agent_name)
        context = self._get_context_from_scores(toxicity_score, policy_violations)
        return semantic.adjust_confidence(context, planned_action, base_confidence)

    def start_session(self):
        """Start a new learning session."""
        self.session_count += 1

    def end_session(self):
        """End current session and record metrics."""
        metrics = self.get_system_metrics()
        self.session_metrics.append({
            'session': self.session_count,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })


    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system-wide learning metrics.

        Returns:
            Dictionary with all metrics
        """
        agent_metrics = {}
        total_success = 0
        total_decisions = 0

        for agent_name in self.agent_episodic_memories.keys():
            episodic = self.agent_episodic_memories[agent_name]
            semantic = self.agent_semantic_memories.get(agent_name)

            ep_stats = episodic.get_statistics()
            sem_stats = semantic.get_all_statistics() if semantic else {}

            agent_metrics[agent_name] = {
                'episodic': ep_stats,
                'semantic': sem_stats
            }

            total_success += ep_stats.get('success_rate', 0) * ep_stats.get('total_decisions', 0)
            total_decisions += ep_stats.get('total_decisions', 0)

        overall_success_rate = total_success / total_decisions if total_decisions > 0 else 0.0

        return {
            'session_count': self.session_count,
            'total_decisions': self.total_decisions,
            'total_appeals': self.total_appeals,
            'total_overturns': self.total_overturns,
            'appeal_rate': self.total_appeals / self.total_decisions if self.total_decisions > 0 else 0.0,
            'overturn_rate': self.total_overturns / self.total_appeals if self.total_appeals > 0 else 0.0,
            'overall_success_rate': overall_success_rate,
            'agent_metrics': agent_metrics
        }

    def print_learning_report(self):
        """Print a comprehensive learning report."""
        logger.info("\n" + "=" * 40)
        logger.info("MULTI-AGENT LEARNING REPORT")
        logger.info("=" * 40)

        metrics = self.get_system_metrics()

        logger.info(f"\nSystem Overview:")
        logger.info(f"Sessions completed: {metrics['session_count']}")
        logger.info(f"Total decisions: {metrics['total_decisions']}")
        logger.info(f"Total appeals: {metrics['total_appeals']}")
        logger.info(f"Overturned: {metrics['total_overturns']}")
        logger.info(f"Appeal rate: {metrics['appeal_rate']:.1%}")
        logger.info(f"Overturn rate: {metrics['overturn_rate']:.1%}")
        logger.info(f"Overall success rate: {metrics['overall_success_rate']:.1%}")

        logger.info(f"\nAgent Performance:")
        for agent_name, agent_data in metrics['agent_metrics'].items():
            ep_stats = agent_data['episodic']
            logger.info(f"\n{agent_name}:")
            logger.info(f"Decisions: {ep_stats.get('total_decisions', 0)}")
            logger.info(f"Success rate: {ep_stats.get('success_rate', 0):.1%}")
            logger.info(f"Appeal rate: {ep_stats.get('appeal_rate', 0):.1%}")

        # Show improvement over sessions
        if len(self.session_metrics) > 1:
            logger.info(f"\nImprovement Over Time:")
            for i, session in enumerate(self.session_metrics[-5:], 1):
                logger.info(f"Session {session['session']}: {session['overall_success_rate']:.1%} success")

        logger.info("\n" + "=" * 40)
