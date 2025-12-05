"""
Content Moderation & Community Safety Platform

A multi-agent AI system for automated content moderation.
"""

from .core.models import (
    ContentState,
    ContentType,
    ContentStatus,
    DecisionType,
    ToxicityLevel,
    PolicyCategory,
    ReputationTier,
    AgentDecision,
    UserProfile,
    ContentMetadata
)

from .agents.agents import ContentModerationAgents
from .agents.workflow import (
    create_moderation_workflow,
    create_appeal_workflow,
    process_content
)
from .database.moderation_db import ModerationDatabase
from .memory.memory import ModerationMemoryManager
from .utils.tools import (
    detect_toxicity,
    detect_hate_speech_patterns,
    check_policy_violations,
    calculate_user_reputation
)

__all__ = [
    "ContentState",
    "ContentType",
    "ContentStatus",
    "DecisionType",
    "ToxicityLevel",
    "PolicyCategory",
    "ReputationTier",
    "AgentDecision",
    "UserProfile",
    "ContentMetadata",
    "ContentModerationAgents",
    "create_moderation_workflow",
    "create_appeal_workflow",
    "process_content",
    "ModerationDatabase",
    "ModerationMemoryManager",
    "detect_toxicity",
    "detect_hate_speech_patterns",
    "check_policy_violations",
    "calculate_user_reputation"
]

__version__ = "1.0.0"
