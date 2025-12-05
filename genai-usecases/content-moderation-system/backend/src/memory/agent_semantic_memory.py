"""
Agent Semantic Memory for Content Moderation

Learns generalized patterns from moderation decisions:
- Optimal toxicity thresholds
- Success rates for different action types
- User reputation impact on decisions
- Policy violation severity patterns

This enables agents to learn: "In situations like X, action Y works best"
"""
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AgentSemanticMemory:
    """
    Stores generalized knowledge learned from moderation experiences.

    Unlike episodic memory (specific cases), this learns patterns:
    - "For toxicity 0.6-0.7, warnings work 85% of the time"
    - "Users with reputation < 0.3 need stricter moderation"
    - "Hate speech policy violations should always be removed"
    """

    def __init__(self, agent_name: str, decay_factor: float = 0.9):
        """
        Initialize agent semantic memory.

        Args:
            agent_name: Name of the agent using this memory
            decay_factor: How much to weigh old vs new learning (0.9 = 90% old, 10% new)
        """
        self.agent_name = agent_name
        self.decay_factor = decay_factor

        # Learned optimal thresholds and preferences
        self.thresholds: Dict[str, float] = defaultdict(float)

        # Action success patterns: (context, action) → {success, total}
        self.action_patterns: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'success': 0, 'total': 0}
        )

        # Context-specific patterns
        self.context_patterns: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)

        # Confidence adjustments based on learning
        self.confidence_adjustments: Dict[str, float] = defaultdict(float)

    def record_decision_outcome(
        self,
        context: str,
        action: str,
        success: bool,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """
        Record a decision outcome for pattern learning.

        Args:
            context: The situation (e.g., "toxicity_0.6_to_0.7")
            action: What was done (e.g., "warn", "remove")
            success: Whether the decision was correct (not appealed/overturned)
            confidence: Original confidence in the decision
            metadata: Additional context
        """
        # Update action patterns
        pattern_key = f"{context}_{action}"
        self.action_patterns[pattern_key]['total'] += 1
        if success:
            self.action_patterns[pattern_key]['success'] += 1

        # Store in context patterns
        self.context_patterns[context].append((action, success))

        # Update confidence adjustment if decision was wrong
        if not success and confidence > 0:
            # Lower confidence for this pattern in future
            adjustment = -0.05 if confidence > 0.7 else -0.02
            self.confidence_adjustments[pattern_key] += adjustment

        success_emoji = "✅" if success else "❌"
        rate = self.get_action_success_rate(context, action)
        logger.info(f"{success_emoji} [{self.agent_name}] {context} → {action}: {rate:.1%} success")

    def get_action_success_rate(self, context: str, action: str) -> float:
        """
        Get success rate for a specific context-action pair.

        Args:
            context: The situation
            action: The action taken

        Returns:
            Success rate (0.0 to 1.0)
        """
        pattern_key = f"{context}_{action}"
        stats = self.action_patterns[pattern_key]

        if stats['total'] == 0:
            return 0.0

        return stats['success'] / stats['total']

    def get_best_action(
        self,
        context: str,
        min_samples: int = 3
    ) -> Optional[Tuple[str, float]]:
        """
        Get the action that works best in a given context.

        Args:
            context: The current situation
            min_samples: Minimum attempts required

        Returns:
            (best_action, success_rate) or None if insufficient data
        """
        if context not in self.context_patterns:
            return None

        # Calculate success rate for each action in this context
        action_stats = defaultdict(lambda: {'success': 0, 'total': 0})

        for action, success in self.context_patterns[context]:
            action_stats[action]['total'] += 1
            if success:
                action_stats[action]['success'] += 1

        # Filter by minimum samples
        viable_actions = {
            action: stats for action, stats in action_stats.items()
            if stats['total'] >= min_samples
        }

        if not viable_actions:
            return None

        # Find action with highest success rate
        best = max(
            viable_actions.items(),
            key=lambda x: x[1]['success'] / max(x[1]['total'], 1)
        )

        action_name = best[0]
        success_rate = best[1]['success'] / best[1]['total']

        logger.info(f"[{self.agent_name}] Best for '{context}': {action_name} ({success_rate:.1%})")

        return (action_name, success_rate)

    def learn_threshold(
        self,
        threshold_name: str,
        suggested_value: float,
        weight: float = 1.0
    ):
        """
        Learn or adjust a threshold value using exponential decay.

        Example: Learn optimal toxicity threshold for removal

        Args:
            threshold_name: Name of threshold (e.g., "toxicity_removal_threshold")
            suggested_value: New suggested value
            weight: Importance of this update
        """
        old_value = self.thresholds[threshold_name]
        new_value = (
            self.decay_factor * old_value +
            (1 - self.decay_factor) * weight * suggested_value
        )

        self.thresholds[threshold_name] = new_value

        change = "📈" if new_value > old_value else "📉"
        logger.info(f"{change} [{self.agent_name}] Threshold '{threshold_name}': "
              f"{old_value:.3f} → {new_value:.3f}")

    def get_threshold(self, threshold_name: str, default: float = 0.5) -> float:
        """
        Get learned threshold value.

        Args:
            threshold_name: Name of threshold
            default: Default value if not learned yet

        Returns:
            Learned threshold value
        """
        return self.thresholds.get(threshold_name, default)

    def adjust_confidence(
        self,
        context: str,
        action: str,
        base_confidence: float
    ) -> float:
        """
        Adjust confidence based on historical performance.

        If this context-action pair has failed before, lower confidence.
        If it has high success rate, boost confidence.

        Args:
            context: The situation
            action: The planned action
            base_confidence: Original confidence

        Returns:
            Adjusted confidence
        """
        pattern_key = f"{context}_{action}"

        # Get success rate
        stats = self.action_patterns[pattern_key]
        if stats['total'] >= 3:  # Need some history
            success_rate = stats['success'] / stats['total']

            # Adjust based on success rate
            if success_rate > 0.8:
                adjustment = 0.05  # Boost confidence
            elif success_rate < 0.5:
                adjustment = -0.10  # Lower confidence
            else:
                adjustment = 0.0

            # Apply learned adjustments
            adjustment += self.confidence_adjustments.get(pattern_key, 0.0)

            adjusted = max(0.0, min(1.0, base_confidence + adjustment))

            if abs(adjustment) > 0.01:
                arrow = "↑" if adjustment > 0 else "↓"
                logger.info(f"{arrow} [{self.agent_name}] Confidence adjusted: "
                      f"{base_confidence:.2f} → {adjusted:.2f} (success rate: {success_rate:.1%})")

            return adjusted

        return base_confidence

    def get_context_statistics(self, context: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific context.

        Args:
            context: The context to analyze

        Returns:
            Dictionary with statistics
        """
        if context not in self.context_patterns:
            return {
                'context': context,
                'total_decisions': 0,
                'actions': []
            }

        patterns = self.context_patterns[context]
        action_stats = defaultdict(lambda: {'success': 0, 'total': 0})

        for action, success in patterns:
            action_stats[action]['total'] += 1
            if success:
                action_stats[action]['success'] += 1

        actions_list = []
        for action, stats in action_stats.items():
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
            actions_list.append({
                'action': action,
                'success_rate': rate,
                'success_count': stats['success'],
                'total_count': stats['total']
            })

        actions_list.sort(key=lambda x: x['success_rate'], reverse=True)

        return {
            'context': context,
            'total_decisions': len(patterns),
            'unique_actions': len(action_stats),
            'actions': actions_list
        }

    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all learned patterns.

        Returns:
            Dictionary with all statistics
        """
        total_patterns = sum(len(patterns) for patterns in self.context_patterns.values())
        total_successes = sum(stats['success'] for stats in self.action_patterns.values())
        total_attempts = sum(stats['total'] for stats in self.action_patterns.values())

        overall_success_rate = (
            total_successes / total_attempts if total_attempts > 0 else 0.0
        )

        # Top performing context-action pairs
        top_patterns = []
        for pattern_key, stats in self.action_patterns.items():
            if stats['total'] >= 3:
                rate = stats['success'] / stats['total']
                top_patterns.append((pattern_key, rate, stats))

        top_patterns.sort(key=lambda x: x[1], reverse=True)

        return {
            'agent_name': self.agent_name,
            'total_contexts': len(self.context_patterns),
            'total_patterns': total_patterns,
            'unique_action_patterns': len(self.action_patterns),
            'overall_success_rate': overall_success_rate,
            'total_successes': total_successes,
            'total_attempts': total_attempts,
            'learned_thresholds': dict(self.thresholds),
            'top_patterns': [
                {'pattern': p[0], 'success_rate': p[1], 'attempts': p[2]['total']}
                for p in top_patterns[:10]
            ]
        }

    def clear(self):
        """Clear all learned patterns."""
        self.thresholds.clear()
        self.action_patterns.clear()
        self.context_patterns.clear()
        self.confidence_adjustments.clear()
        logger.info(f"[{self.agent_name}] Semantic memory cleared")

    def __repr__(self):
        stats = self.get_all_statistics()
        return (f"AgentSemanticMemory(agent='{self.agent_name}', "
                f"patterns={stats['total_patterns']}, "
                f"success_rate={stats['overall_success_rate']:.1%})")
