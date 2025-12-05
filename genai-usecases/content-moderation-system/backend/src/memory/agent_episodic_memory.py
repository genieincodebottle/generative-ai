"""
Agent Episodic Memory for Content Moderation

Stores specific moderation decisions and their outcomes to enable
learning from past experiences. Each episode contains:
- Content details
- Agent decision
- Outcome (approved/removed/appealed)
- Whether the decision was correct

This allows agents to recall: "Last time I saw content like this,
what decision did I make and was it right?"
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class AgentEpisodicMemory:
    """
    Stores individual moderation decisions as episodes.

    Unlike the base ChromaDB memory (which stores all moderation results),
    this is agent-specific and tracks decision quality through appeals.
    """

    def __init__(self, agent_name: str, capacity: int = 100):
        """
        Initialize agent episodic memory.

        Args:
            agent_name: Name of the agent using this memory
            capacity: Maximum number of episodes to store
        """
        self.agent_name = agent_name
        self.capacity = capacity
        self.episodes: List[Dict[str, Any]] = []

    def store_decision(
        self,
        content_text: str,
        toxicity_score: float,
        policy_violations: List[str],
        decision: str,
        confidence: float,
        outcome: str,
        was_appealed: bool = False,
        appeal_outcome: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Store a moderation decision episode.

        Args:
            content_text: The content that was moderated
            toxicity_score: Toxicity score (0-1)
            policy_violations: List of policy violations detected
            decision: Agent's decision (approve, remove, warn, etc.)
            confidence: Confidence in decision (0-1)
            outcome: What happened (approved, removed, flagged, etc.)
            was_appealed: Whether user appealed the decision
            appeal_outcome: Result of appeal (upheld, overturned, partial)
            metadata: Additional context
        """
        # Determine if decision was correct
        if was_appealed:
            # If appealed and overturned = wrong decision
            decision_correct = appeal_outcome == "upheld"
        else:
            # No appeal = likely correct (user accepted it)
            decision_correct = True

        episode = {
            'agent_name': self.agent_name,
            'content_text': content_text[:200],  # Truncate for storage
            'toxicity_score': toxicity_score,
            'policy_violations': policy_violations,
            'decision': decision,
            'confidence': confidence,
            'outcome': outcome,
            'was_appealed': was_appealed,
            'appeal_outcome': appeal_outcome,
            'decision_correct': decision_correct,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'embedding': self._create_embedding(content_text, toxicity_score, policy_violations)
        }

        self.episodes.append(episode)

        # Remove oldest if capacity exceeded
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

        status = "Correct" if decision_correct else "Wrong"
        if was_appealed:
            status += f" (Appealed: {appeal_outcome})"

        logger.info(f"[{self.agent_name}] Stored decision: {decision} - {status}")

    def _create_embedding(
        self,
        content_text: str,
        toxicity_score: float,
        policy_violations: List[str]
    ) -> int:
        """
        Create embedding for similarity matching.

        In production, use actual embedding models.
        Here we use a simple hash for educational purposes.
        """
        text = f"{content_text} {toxicity_score} {' '.join(policy_violations)}".lower()
        hash_object = hashlib.md5(text.encode())
        return int(hash_object.hexdigest(), 16) % 100000

    def retrieve_similar_decisions(
        self,
        content_text: str,
        toxicity_score: float,
        policy_violations: List[str],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past moderation decisions.

        Args:
            content_text: Current content to match
            toxicity_score: Current toxicity score
            policy_violations: Current violations
            k: Number of similar episodes to retrieve

        Returns:
            List of similar past decisions
        """
        if not self.episodes:
            return []

        query_embedding = self._create_embedding(content_text, toxicity_score, policy_violations)

        # Calculate similarity scores
        scored_episodes = []
        for episode in self.episodes:
            distance = abs(episode['embedding'] - query_embedding)
            scored_episodes.append((distance, episode))

        # Sort by similarity
        scored_episodes.sort(key=lambda x: x[0])

        similar = [ep for _, ep in scored_episodes[:k]]

        if similar:
            correct_count = sum(1 for ep in similar if ep['decision_correct'])
            logger.info(f"[{self.agent_name}] Found {len(similar)} similar cases "
                  f"({correct_count}/{len(similar)} were correct)")

        return similar

    def get_success_rate(self) -> float:
        """
        Calculate overall success rate (decisions not overturned).

        Returns:
            Success rate (0.0 to 1.0)
        """
        if not self.episodes:
            return 0.0

        correct_decisions = sum(1 for ep in self.episodes if ep['decision_correct'])
        return correct_decisions / len(self.episodes)

    def get_appeal_rate(self) -> float:
        """
        Calculate what percentage of decisions were appealed.

        Returns:
            Appeal rate (0.0 to 1.0)
        """
        if not self.episodes:
            return 0.0

        appealed = sum(1 for ep in self.episodes if ep['was_appealed'])
        return appealed / len(self.episodes)

    def get_overturn_rate(self) -> float:
        """
        Calculate what percentage of appeals were successful.

        Returns:
            Overturn rate (0.0 to 1.0)
        """
        appealed_episodes = [ep for ep in self.episodes if ep['was_appealed']]

        if not appealed_episodes:
            return 0.0

        overturned = sum(
            1 for ep in appealed_episodes
            if ep['appeal_outcome'] in ['overturned', 'partial']
        )
        return overturned / len(appealed_episodes)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about agent performance.

        Returns:
            Dictionary with performance metrics
        """
        if not self.episodes:
            return {
                'agent_name': self.agent_name,
                'total_decisions': 0,
                'success_rate': 0.0,
                'appeal_rate': 0.0,
                'overturn_rate': 0.0
            }

        # Calculate metrics by decision type
        decision_stats = {}
        for episode in self.episodes:
            decision = episode['decision']
            if decision not in decision_stats:
                decision_stats[decision] = {'total': 0, 'correct': 0}

            decision_stats[decision]['total'] += 1
            if episode['decision_correct']:
                decision_stats[decision]['correct'] += 1

        # Calculate confidence distribution
        avg_confidence = sum(ep['confidence'] for ep in self.episodes) / len(self.episodes)

        return {
            'agent_name': self.agent_name,
            'total_decisions': len(self.episodes),
            'success_rate': self.get_success_rate(),
            'appeal_rate': self.get_appeal_rate(),
            'overturn_rate': self.get_overturn_rate(),
            'average_confidence': avg_confidence,
            'decisions_by_type': decision_stats,
            'oldest_episode': self.episodes[0]['timestamp'] if self.episodes else None,
            'newest_episode': self.episodes[-1]['timestamp'] if self.episodes else None
        }

    def get_recent_mistakes(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent incorrect decisions to learn from.

        Args:
            n: Number of mistakes to retrieve

        Returns:
            List of incorrect decision episodes
        """
        mistakes = [ep for ep in self.episodes if not ep['decision_correct']]
        return mistakes[-n:]

    def clear(self):
        """Clear all stored episodes."""
        self.episodes = []
        logger.info(f"[{self.agent_name}] Episodic memory cleared")

    def __len__(self):
        return len(self.episodes)

    def __repr__(self):
        success_rate = self.get_success_rate()
        return (f"AgentEpisodicMemory(agent='{self.agent_name}', "
                f"episodes={len(self.episodes)}, success_rate={success_rate:.1%})")
