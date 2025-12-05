"""
Memory management using ChromaDB for content moderation system.

Stores and retrieves:
- Historical moderation decisions
- Flagged content patterns
- User violation history
- Appeal outcomes
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Disable ChromaDB telemetry before importing to avoid telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ModerationMemoryManager:
    """Manages persistent memory for the moderation system using ChromaDB."""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the memory manager with ChromaDB.

        Args:
            persist_directory: Directory to persist ChromaDB data
                              If None, reads from CHROMA_DB_PATH env variable
                              Defaults to "./databases/chroma_moderation_db" if not set
        """
        if persist_directory is None:
            persist_directory = os.getenv("CHROMA_DB_PATH", "./databases/chroma_moderation_db")

        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create collections
        self._init_collections()

    def _init_collections(self):
        """Initialize ChromaDB collections."""
        # Collection for moderation decisions
        self.decisions_collection = self.client.get_or_create_collection(
            name="moderation_decisions",
            metadata={"description": "Historical moderation decisions and outcomes"}
        )

        # Collection for flagged content patterns
        self.patterns_collection = self.client.get_or_create_collection(
            name="flagged_patterns",
            metadata={"description": "Patterns of flagged/removed content"}
        )

        # Collection for user history
        self.user_history_collection = self.client.get_or_create_collection(
            name="user_violations",
            metadata={"description": "User violation history"}
        )

        logger.info(f"[OK] Memory collections initialized at {self.persist_directory}")

    def store_moderation_decision(
        self,
        content_id: str,
        content_text: str,
        user_id: str,
        action: str,
        violations: List[str],
        toxicity_score: float,
        agent_decisions: List[Any],
        primary_agent: Optional[str] = None,
        decision_context: Optional[str] = None,
        confidence: Optional[float] = None,
        was_appealed: bool = False,
        appeal_outcome: Optional[str] = None
    ):
        """
        Store a moderation decision in memory with enhanced metadata.

        Args:
            content_id: Unique content identifier
            content_text: The actual content text
            user_id: User ID who posted the content
            action: Moderation action taken
            violations: List of policy violations
            toxicity_score: Toxicity score
            agent_decisions: List of agent decisions
            primary_agent: Name of the agent that made the final decision
            decision_context: Context string for semantic learning (e.g., "toxicity_high_hate_speech")
            confidence: Confidence score of the decision (0.0 to 1.0)
            was_appealed: Whether the decision was appealed
            appeal_outcome: Result of appeal (upheld, overturned, partial)
        """
        try:
            # Determine primary agent if not provided
            if primary_agent is None and agent_decisions:
                # Use the last agent's decision as primary
                primary_agent = agent_decisions[-1].get("agent_name", "unknown") if isinstance(agent_decisions[-1], dict) else "unknown"

            # Calculate decision correctness based on appeal
            was_correct = True
            if was_appealed:
                was_correct = (appeal_outcome == "upheld")

            # Prepare enhanced metadata
            metadata = {
                "content_id": content_id,
                "user_id": user_id,
                "action": action,
                "violations": json.dumps(violations),
                "toxicity_score": toxicity_score,
                "timestamp": datetime.now().isoformat(),
                "was_removed": action in ["removed", "user_suspended", "user_banned"],
                "agent_count": len(agent_decisions),

                # Agent attribution (HIGH PRIORITY)
                "primary_agent": primary_agent or "unknown",
                "decision_context": decision_context or "general",
                "confidence": confidence if confidence is not None else 0.0,

                # Quality indicators (MEDIUM PRIORITY)
                "was_appealed": was_appealed,
                "appeal_outcome": appeal_outcome or "none",
                "was_correct": was_correct,

                # Store all agent contributions
                "agent_decisions": json.dumps([
                    {
                        "agent": str(d.get("agent_name", "unknown") if isinstance(d, dict) else "unknown"),
                        "decision": str(d.get("decision", "unknown") if isinstance(d, dict) else "unknown"),
                        "confidence": float(d.get("confidence", 0.0) if isinstance(d, dict) else 0.0)
                    }
                    for d in agent_decisions
                ]) if agent_decisions else "[]"
            }

            # Store in decisions collection
            self.decisions_collection.add(
                documents=[content_text],
                metadatas=[metadata],
                ids=[content_id]
            )

            # If content was flagged/removed, also store in patterns collection
            if metadata["was_removed"]:
                self.patterns_collection.add(
                    documents=[content_text],
                    metadatas=[metadata],
                    ids=[f"{content_id}_pattern"]
                )

            # Store user violation if applicable
            if violations:
                self._store_user_violation(user_id, content_id, violations, action)

        except Exception as e:
            logger.error(f"[WARNING] Error storing moderation decision: {e}")

    def _store_user_violation(
        self,
        user_id: str,
        content_id: str,
        violations: List[str],
        action: str
    ):
        """Store user violation in history."""
        try:
            metadata = {
                "user_id": user_id,
                "content_id": content_id,
                "violations": json.dumps(violations),
                "action": action,
                "timestamp": datetime.now().isoformat()
            }

            # Use content_id + user_id as unique ID for violation record
            violation_id = f"{user_id}_{content_id}"

            self.user_history_collection.add(
                documents=[f"User {user_id} violation: {', '.join(violations)}"],
                metadatas=[metadata],
                ids=[violation_id]
            )

        except Exception as e:
            logger.error(f"[WARNING] Error storing user violation: {e}")

    def retrieve_similar_content(
        self,
        content_text: str,
        content_type: str,
        user_id: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar content from historical decisions.

        Args:
            content_text: Content to find similar matches for
            content_type: Type of content
            user_id: User ID
            n_results: Number of results to return

        Returns:
            List of similar content with metadata
        """
        try:
            if not content_text or len(content_text.strip()) == 0:
                return []

            # Query the decisions collection
            results = self.decisions_collection.query(
                query_texts=[content_text],
                n_results=min(n_results, 10)
            )

            similar_content = []

            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    similar_content.append({
                        "content_id": metadata.get("content_id"),
                        "action": metadata.get("action"),
                        "violations": json.loads(metadata.get("violations", "[]")),
                        "toxicity_score": metadata.get("toxicity_score"),
                        "was_removed": metadata.get("was_removed"),
                        "timestamp": metadata.get("timestamp"),
                        "similarity": 1.0 - (i * 0.1),  # Approximate similarity score
                        # Enhanced metadata
                        "primary_agent": metadata.get("primary_agent", "unknown"),
                        "confidence": metadata.get("confidence", 0.0),
                        "was_appealed": metadata.get("was_appealed", False),
                        "was_correct": metadata.get("was_correct", True)
                    })

            return similar_content

        except Exception as e:
            logger.error(f"[WARNING] Error retrieving similar content: {e}")
            return []

    def retrieve_similar_content_for_agent(
        self,
        agent_name: str,
        content_text: str,
        content_type: str = "",
        user_id: str = "",
        n_results: int = 5,
        min_confidence: float = 0.0,
        only_correct: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar content filtered by specific agent (AGENT-SCOPED RETRIEVAL).

        This ensures agents learn from their own domain-specific decisions,
        preventing noise from other agents' expertise areas.

        Args:
            agent_name: Name of the agent to filter by (e.g., "Toxicity Detection Agent")
            content_text: Content to find similar matches for
            content_type: Type of content (optional filter)
            user_id: User ID (optional filter)
            n_results: Number of results to return
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            only_correct: If True, only return decisions that were not overturned

        Returns:
            List of similar content from this agent's past decisions
        """
        try:
            if not content_text or len(content_text.strip()) == 0:
                return []

            # Build where clause for agent filtering
            where_clause = {"primary_agent": agent_name}

            # Query with agent filter
            results = self.decisions_collection.query(
                query_texts=[content_text],
                where=where_clause,
                n_results=min(n_results * 2, 20)  # Get more for post-filtering
            )

            similar_content = []

            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]

                    # Apply confidence filter
                    confidence = metadata.get("confidence", 0.0)
                    if confidence < min_confidence:
                        continue

                    # Apply correctness filter
                    if only_correct and not metadata.get("was_correct", True):
                        continue

                    similar_content.append({
                        "content_id": metadata.get("content_id"),
                        "action": metadata.get("action"),
                        "violations": json.loads(metadata.get("violations", "[]")),
                        "toxicity_score": metadata.get("toxicity_score"),
                        "was_removed": metadata.get("was_removed"),
                        "timestamp": metadata.get("timestamp"),
                        "similarity": 1.0 - (i * 0.05),  # Finer-grained similarity
                        "primary_agent": metadata.get("primary_agent"),
                        "confidence": confidence,
                        "was_appealed": metadata.get("was_appealed", False),
                        "was_correct": metadata.get("was_correct", True),
                        "decision_context": metadata.get("decision_context", "general")
                    })

                    # Stop if we have enough results
                    if len(similar_content) >= n_results:
                        break

            return similar_content

        except Exception as e:
            logger.error(f"[WARNING] Error retrieving agent-specific content: {e}")
            return []

    def retrieve_with_filters(
        self,
        content_text: str,
        agent_name: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        min_confidence: Optional[float] = None,
        only_correct: bool = False,
        min_toxicity: Optional[float] = None,
        max_toxicity: Optional[float] = None,
        context: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Flexible multi-criteria retrieval with various filters.

        Args:
            content_text: Content to find similar matches for
            agent_name: Filter by specific agent
            user_id: Filter by specific user
            action: Filter by action type (approve, remove, warn, etc.)
            min_confidence: Minimum confidence threshold
            only_correct: Only return non-overturned decisions
            min_toxicity: Minimum toxicity score
            max_toxicity: Maximum toxicity score
            context: Filter by decision context
            n_results: Number of results to return

        Returns:
            List of filtered similar content
        """
        try:
            if not content_text or len(content_text.strip()) == 0:
                return []

            # Build where clause
            where_clause = {}

            if agent_name:
                where_clause["primary_agent"] = agent_name

            if user_id:
                where_clause["user_id"] = user_id

            if action:
                where_clause["action"] = action

            if context:
                where_clause["decision_context"] = context

            # Query with filters
            query_params = {
                "query_texts": [content_text],
                "n_results": min(n_results * 3, 30)  # Get more for post-filtering
            }

            if where_clause:
                query_params["where"] = where_clause

            results = self.decisions_collection.query(**query_params)

            filtered_content = []

            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]

                    # Post-filtering (ChromaDB has limited numeric range support)
                    confidence = metadata.get("confidence", 0.0)
                    toxicity = metadata.get("toxicity_score", 0.0)

                    if min_confidence and confidence < min_confidence:
                        continue

                    if only_correct and not metadata.get("was_correct", True):
                        continue

                    if min_toxicity and toxicity < min_toxicity:
                        continue

                    if max_toxicity and toxicity > max_toxicity:
                        continue

                    filtered_content.append({
                        "content_id": metadata.get("content_id"),
                        "action": metadata.get("action"),
                        "violations": json.loads(metadata.get("violations", "[]")),
                        "toxicity_score": toxicity,
                        "was_removed": metadata.get("was_removed"),
                        "timestamp": metadata.get("timestamp"),
                        "similarity": 1.0 - (i * 0.05),
                        "primary_agent": metadata.get("primary_agent"),
                        "confidence": confidence,
                        "was_appealed": metadata.get("was_appealed", False),
                        "was_correct": metadata.get("was_correct", True),
                        "decision_context": metadata.get("decision_context", "general")
                    })

                    if len(filtered_content) >= n_results:
                        break

            return filtered_content

        except Exception as e:
            logger.error(f"[WARNING] Error in filtered retrieval: {e}")
            return []

    def get_user_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get violation history for a specific user.

        Args:
            user_id: User ID to lookup
            limit: Maximum number of violations to return

        Returns:
            List of user violations
        """
        try:
            # Query user history collection
            results = self.user_history_collection.get(
                where={"user_id": user_id},
                limit=limit
            )

            violations = []

            if results and results['ids']:
                for i, violation_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    violations.append({
                        "content_id": metadata.get("content_id"),
                        "violations": json.loads(metadata.get("violations", "[]")),
                        "action": metadata.get("action"),
                        "timestamp": metadata.get("timestamp")
                    })

            # Sort by timestamp (newest first)
            violations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return violations

        except Exception as e:
            logger.error(f"[WARNING] Error retrieving user history: {e}")
            return []

    def check_flagged_patterns(
        self,
        content_text: str,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Check if content matches previously flagged patterns.

        Args:
            content_text: Content to check
            n_results: Number of pattern matches to return

        Returns:
            List of matching flagged patterns
        """
        try:
            if not content_text or len(content_text.strip()) == 0:
                return []

            # Query the patterns collection
            results = self.patterns_collection.query(
                query_texts=[content_text],
                n_results=min(n_results, 5)
            )

            patterns = []

            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i, pattern_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    patterns.append({
                        "content_id": metadata.get("content_id"),
                        "violations": json.loads(metadata.get("violations", "[]")),
                        "action": metadata.get("action"),
                        "toxicity_score": metadata.get("toxicity_score"),
                        "timestamp": metadata.get("timestamp")
                    })

            return patterns

        except Exception as e:
            logger.error(f"[WARNING] Error checking flagged patterns: {e}")
            return []

    def retrieve_with_temporal_decay(
        self,
        content_text: str,
        agent_name: Optional[str] = None,
        n_results: int = 10,
        decay_days: int = 30,
        decay_factor: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar content with temporal decay weighting.

        Recent decisions are weighted more heavily than older ones,
        as they better reflect current moderation standards.

        Args:
            content_text: Content to find similar matches for
            agent_name: Filter by specific agent
            n_results: Number of results to return
            decay_days: Number of days for decay calculation
            decay_factor: Decay multiplier (0.0 to 1.0, lower = more decay)

        Returns:
            List of similar content with time-weighted similarity scores
        """
        try:
            if not content_text or len(content_text.strip()) == 0:
                return []

            # Build where clause
            where_clause = {}
            if agent_name:
                where_clause["primary_agent"] = agent_name

            # Get more results for temporal filtering
            query_params = {
                "query_texts": [content_text],
                "n_results": min(n_results * 2, 30)
            }

            if where_clause:
                query_params["where"] = where_clause

            results = self.decisions_collection.query(**query_params)

            weighted_content = []

            if results and results['ids'] and len(results['ids'][0]) > 0:
                now = datetime.now()

                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]

                    # Calculate base similarity
                    base_similarity = 1.0 - (i * 0.05)

                    # Calculate temporal weight
                    timestamp_str = metadata.get("timestamp", now.isoformat())
                    try:
                        decision_time = datetime.fromisoformat(timestamp_str)
                        days_old = (now - decision_time).days

                        # Apply exponential decay
                        temporal_weight = decay_factor ** (days_old / decay_days)
                    except:
                        temporal_weight = 1.0  # Default if parsing fails

                    # Combined score
                    weighted_similarity = base_similarity * temporal_weight

                    weighted_content.append({
                        "content_id": metadata.get("content_id"),
                        "action": metadata.get("action"),
                        "violations": json.loads(metadata.get("violations", "[]")),
                        "toxicity_score": metadata.get("toxicity_score"),
                        "was_removed": metadata.get("was_removed"),
                        "timestamp": timestamp_str,
                        "similarity": base_similarity,
                        "temporal_weight": temporal_weight,
                        "weighted_similarity": weighted_similarity,
                        "primary_agent": metadata.get("primary_agent"),
                        "confidence": metadata.get("confidence", 0.0),
                        "was_appealed": metadata.get("was_appealed", False),
                        "was_correct": metadata.get("was_correct", True),
                        "decision_context": metadata.get("decision_context", "general")
                    })

                # Sort by weighted similarity
                weighted_content.sort(key=lambda x: x["weighted_similarity"], reverse=True)
                weighted_content = weighted_content[:n_results]

            return weighted_content

        except Exception as e:
            logger.error(f"[WARNING] Error in temporal decay retrieval: {e}")
            return []

    def update_decision_appeal_outcome(
        self,
        content_id: str,
        appeal_outcome: str
    ):
        """
        Update a decision's appeal outcome after appeal is processed.

        Note: ChromaDB doesn't support direct updates, so we need to
        retrieve, modify, and re-add the document.

        Args:
            content_id: Content ID to update
            appeal_outcome: Result of appeal (upheld, overturned, partial)
        """
        try:
            # Get the existing document
            results = self.decisions_collection.get(ids=[content_id])

            if not results or not results['ids']:
                logger.warning(f"[WARNING] Content {content_id} not found in memory")
                return

            # Update metadata
            metadata = results['metadatas'][0]
            metadata['was_appealed'] = True
            metadata['appeal_outcome'] = appeal_outcome
            metadata['was_correct'] = (appeal_outcome == "upheld")

            # Delete and re-add (ChromaDB update pattern)
            self.decisions_collection.delete(ids=[content_id])
            self.decisions_collection.add(
                documents=[results['documents'][0]],
                metadatas=[metadata],
                ids=[content_id]
            )

        except Exception as e:
            logger.error(f"[WARNING] Error updating appeal outcome: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data with enhanced metrics."""
        try:
            decisions_count = self.decisions_collection.count()
            patterns_count = self.patterns_collection.count()
            violations_count = self.user_history_collection.count()

            # Get all decisions for analysis
            all_decisions = self.decisions_collection.get()

            agent_stats = {}
            total_correct = 0
            total_appealed = 0
            total_overturned = 0

            if all_decisions and all_decisions['metadatas']:
                for metadata in all_decisions['metadatas']:
                    agent = metadata.get("primary_agent", "unknown")

                    if agent not in agent_stats:
                        agent_stats[agent] = {
                            "total": 0,
                            "correct": 0,
                            "appealed": 0,
                            "overturned": 0
                        }

                    agent_stats[agent]["total"] += 1

                    if metadata.get("was_correct", True):
                        agent_stats[agent]["correct"] += 1
                        total_correct += 1

                    if metadata.get("was_appealed", False):
                        agent_stats[agent]["appealed"] += 1
                        total_appealed += 1

                        if metadata.get("appeal_outcome") in ["overturned", "partial"]:
                            agent_stats[agent]["overturned"] += 1
                            total_overturned += 1

            return {
                "total_decisions": decisions_count,
                "flagged_patterns": patterns_count,
                "user_violations": violations_count,
                "total_correct": total_correct,
                "total_appealed": total_appealed,
                "total_overturned": total_overturned,
                "success_rate": total_correct / decisions_count if decisions_count > 0 else 0.0,
                "appeal_rate": total_appealed / decisions_count if decisions_count > 0 else 0.0,
                "overturn_rate": total_overturned / total_appealed if total_appealed > 0 else 0.0,
                "agent_statistics": agent_stats
            }

        except Exception as e:
            logger.error(f"[WARNING] Error getting statistics: {e}")
            return {
                "total_decisions": 0,
                "flagged_patterns": 0,
                "user_violations": 0,
                "agent_statistics": {}
            }

    def clear_all_data(self):
        """Clear all data from collections (use with caution!)."""
        try:
            self.client.reset()
            self._init_collections()
        except Exception as e:
            logger.error(f"[WARNING] Error clearing data: {e}")
