"""
Data models for the Content Moderation & Community Safety Platform.

This module defines core data structures for content moderation processing.
"""

from enum import Enum
from dataclasses import dataclass
from typing import TypedDict, List, Optional, Dict, Any


class ContentType(Enum):
    """Types of content that can be moderated."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    COMMENT = "comment"
    POST = "post"
    MESSAGE = "message"
    STORY = "story"
    STORY_COMMENT = "story_comment"


class ContentStatus(Enum):
    """Content status throughout the moderation pipeline."""

    SUBMITTED = "submitted"
    ANALYZING = "analyzing"
    TOXICITY_CHECK = "toxicity_check"
    POLICY_CHECK = "policy_check"
    REACT_SYNTHESIS = "react_synthesis"  # ReAct loop decision synthesis
    REPUTATION_SCORING = "reputation_scoring"
    APPEAL_REVIEW = "appeal_review"
    ACTION_ENFORCEMENT = "action_enforcement"
    APPROVED = "approved"
    REMOVED = "removed"
    WARNED = "warned"
    FLAGGED = "flagged"
    UNDER_REVIEW = "under_review"
    # Human-in-the-Loop (HITL) statuses
    PENDING_HUMAN_REVIEW = "pending_human_review"  # Waiting for human decision
    HUMAN_REVIEW_COMPLETED = "human_review_completed"  # Human made decision
    ESCALATED = "escalated"  # Escalated to senior moderator


class DecisionType(Enum):
    """Types of decisions agents can make."""

    APPROVE = "approve"
    REMOVE = "remove"
    WARN = "warn"
    FLAG = "flag"
    SUSPEND_USER = "suspend_user"
    BAN_USER = "ban_user"
    NEEDS_REVIEW = "needs_review"
    # Human-in-the-Loop decisions
    AWAIT_HUMAN = "await_human"  # Pause for human input
    HUMAN_APPROVED = "human_approved"  # Human approved action
    HUMAN_ESCALATED = "human_escalated"  # Human escalated to higher authority


class HITLTriggerReason(Enum):
    """Reasons for triggering Human-in-the-Loop review."""

    LOW_CONFIDENCE = "low_confidence"  # Agent confidence below threshold
    HIGH_SEVERITY = "high_severity"  # Severe violation detected
    USER_APPEAL = "user_appeal"  # User requested appeal
    CONFLICTING_DECISIONS = "conflicting_decisions"  # Agents disagree
    EDGE_CASE = "edge_case"  # Content doesn't fit clear categories
    SENSITIVE_CONTENT = "sensitive_content"  # Politics, religion, etc.
    HIGH_PROFILE_USER = "high_profile_user"  # Verified/high-follower user
    POTENTIAL_FALSE_POSITIVE = "potential_false_positive"  # Possible mistake
    FIRST_OFFENSE_SEVERE = "first_offense_severe"  # First violation but severe
    LEGAL_CONCERN = "legal_concern"  # Potential legal implications


class ToxicityLevel(Enum):
    """Toxicity level classifications."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class PolicyCategory(Enum):
    """Policy violation categories."""

    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    BULLYING = "bullying"
    VIOLENCE = "violence"
    SPAM = "spam"
    SEXUAL_CONTENT = "sexual_content"
    MISINFORMATION = "misinformation"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    COPYRIGHT = "copyright"
    IMPERSONATION = "impersonation"
    PRIVACY_VIOLATION = "privacy_violation"
    NONE = "none"


class ReputationTier(Enum):
    """User reputation tiers."""

    NEW_USER = "new_user"
    TRUSTED = "trusted"
    VETERAN = "veteran"
    MODERATOR = "moderator"
    FLAGGED = "flagged"
    SUSPENDED = "suspended"
    BANNED = "banned"


@dataclass
class ContentMetadata:
    """Metadata about the content."""

    content_id: str
    content_type: str  # ContentType enum value
    platform: str  # twitter, reddit, discord, youtube, etc.
    created_at: str
    language: str
    parent_id: Optional[str] = None  # For comments/replies
    thread_id: Optional[str] = None
    media_urls: List[str] = None
    hashtags: List[str] = None
    mentions: List[str] = None


@dataclass
class UserProfile:
    """User profile information."""

    user_id: str
    username: str
    account_age_days: int
    total_posts: int
    total_violations: int
    previous_warnings: int
    previous_suspensions: int
    reputation_score: float  # 0.0 to 1.0
    reputation_tier: str  # ReputationTier enum value
    verified: bool = False
    follower_count: int = 0
    following_count: int = 0


@dataclass
class AgentDecision:
    """Represents a decision made by an agent."""

    agent_name: str
    decision: DecisionType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    flags: List[str]
    recommendations: List[str]
    extracted_data: Dict[str, Any]
    requires_human_review: bool = False
    processing_time: float = 0.0


class ContentState(TypedDict, total=False):
    """
    Central state object passed between all agents in the LangGraph workflow.

    This is the primary data structure that gets updated as the content
    moves through the moderation pipeline.
    """

    # Content identification
    content_id: str
    submission_id: str
    submission_timestamp: str

    # Content details
    content_text: Optional[str]
    content_type: str  # ContentType enum value
    content_metadata: ContentMetadata

    # Image/video analysis (for multimodal content)
    image_urls: List[str]
    video_urls: List[str]
    image_descriptions: List[str]
    detected_objects: List[str]
    detected_text_in_media: List[str]

    # User information
    user_profile: UserProfile
    user_id: str
    username: str

    # Content Analysis (populated by Content Analysis Agent)
    content_category: Optional[str]
    content_sentiment: Optional[str]
    content_topics: List[str]
    contains_sensitive_content: bool
    explicit_content_detected: bool

    # Toxicity Detection (populated by Toxicity Detection Agent)
    toxicity_score: Optional[float]  # 0.0 to 1.0
    toxicity_level: Optional[str]  # ToxicityLevel enum value
    toxicity_categories: List[str]  # profanity, insult, threat, etc.
    hate_speech_detected: bool
    harassment_detected: bool

    # Policy Violation (populated by Policy Violation Agent)
    policy_violations: List[str]  # PolicyCategory enum values
    violation_severity: Optional[str]  # low, medium, high, critical
    policy_flags: List[str]
    recommended_action: Optional[str]

    # Reputation Scoring (populated by Reputation Agent)
    user_reputation_score: Optional[float]
    user_reputation_tier: Optional[str]
    user_risk_score: Optional[float]  # 0.0 to 1.0
    user_history_flags: List[str]
    similar_violations_count: int

    # Appeal Information (for Appeal Review Agent)
    is_appeal: bool
    appeal_reason: Optional[str]
    original_decision: Optional[str]
    appeal_timestamp: Optional[str]

    # Action Enforcement (populated by Action Enforcement Agent)
    moderation_action: Optional[str]  # DecisionType enum value
    action_reason: str
    action_timestamp: Optional[str]
    user_notified: bool
    content_removed: bool
    user_suspended: bool
    suspension_duration_days: Optional[int]

    # Agent decisions tracking
    agent_decisions: List[AgentDecision]
    current_agent: Optional[str]

    # Workflow control
    status: str  # ContentStatus enum value
    requires_human_review: bool
    human_review_reason: Optional[str]
    overall_confidence: float

    # Manual review
    reviewer_name: Optional[str]
    review_notes: Optional[str]
    review_decision: Optional[str]
    review_timestamp: Optional[str]

    # Timestamps
    created_at: str
    processed_at: Optional[str]

    # Memory/learning
    similar_content: Optional[List[Dict[str, Any]]]
    historical_patterns: Optional[List[Dict[str, Any]]]

    # ReAct Loop (Think-Act-Observe synthesis)
    react_think_output: Optional[str]  # Analysis of all agent decisions
    react_act_decision: Optional[str]  # Synthesized decision
    react_observe_result: Optional[str]  # Observation after action
    react_confidence: Optional[float]  # Synthesized confidence score
    react_reasoning: Optional[str]  # Full reasoning chain

    # Human-in-the-Loop (HITL) fields
    hitl_required: bool  # Whether HITL is needed
    hitl_trigger_reasons: List[str]  # Why HITL was triggered (HITLTriggerReason values)
    hitl_checkpoint: Optional[str]  # Which checkpoint triggered HITL
    hitl_priority: Optional[str]  # Priority level: low, medium, high, critical
    hitl_assigned_to: Optional[str]  # Assigned human reviewer
    hitl_queue_position: Optional[int]  # Position in review queue
    hitl_waiting_since: Optional[str]  # Timestamp when entered HITL queue
    hitl_human_decision: Optional[str]  # Human's decision
    hitl_human_notes: Optional[str]  # Human's notes
    hitl_human_confidence_override: Optional[float]  # Human can override confidence
    hitl_resolution_timestamp: Optional[str]  # When human resolved

    # Guardrails tracking (internal use)
    _guardrail_iteration: Optional[int]  # Current iteration count
    _guardrail_checks: Optional[List[Dict[str, Any]]]  # Guardrail check results
    guardrail_violations: Optional[List[str]]  # List of guardrail violations
    guardrail_warnings: Optional[List[str]]  # List of guardrail warnings


@dataclass
class ToxicityAnalysis:
    """Toxicity detection analysis results."""

    toxicity_score: float  # 0.0 to 1.0
    toxicity_level: str  # ToxicityLevel enum value
    categories: List[str]  # profanity, insult, threat, hate, harassment
    hate_speech_score: float
    harassment_score: float
    threat_score: float
    profanity_count: int
    confidence: float


@dataclass
class PolicyAnalysis:
    """Policy violation analysis results."""

    violations: List[str]  # PolicyCategory enum values
    severity: str  # low, medium, high, critical
    confidence: float
    flags: List[str]
    recommended_action: str
    reasoning: str


@dataclass
class ReputationAnalysis:
    """User reputation analysis results."""

    reputation_score: float  # 0.0 to 1.0
    reputation_tier: str  # ReputationTier enum value
    risk_score: float  # 0.0 to 1.0
    trust_level: str
    history_flags: List[str]
    previous_violations: int
    account_health: str  # excellent, good, fair, poor, critical


@dataclass
class ModerationAction:
    """Moderation action details."""

    action: str  # DecisionType enum value
    reason: str
    confidence: float
    user_notified: bool
    content_removed: bool
    user_suspended: bool
    suspension_duration_days: Optional[int]
    appeal_allowed: bool
    timestamp: str


# Toxicity thresholds
TOXICITY_THRESHOLDS = {
    "none": 0.2,
    "low": 0.4,
    "medium": 0.6,
    "high": 0.8,
    "severe": 0.9
}

# Severity levels for policy violations
VIOLATION_SEVERITY = {
    "low": ["spam", "minor_harassment"],
    "medium": ["harassment", "bullying", "sexual_content"],
    "high": ["hate_speech", "violence", "self_harm"],
    "critical": ["illegal_activity", "severe_violence", "child_safety"]
}

# Reputation score ranges
REPUTATION_RANGES = {
    "new_user": (0.5, 0.7),
    "trusted": (0.7, 0.85),
    "veteran": (0.85, 1.0),
    "flagged": (0.3, 0.5),
    "suspended": (0.1, 0.3),
    "banned": (0.0, 0.1)
}

# Supported platforms
SUPPORTED_PLATFORMS = [
    "twitter",
    "reddit",
    "discord",
    "youtube",
    "facebook",
    "instagram",
    "tiktok",
    "twitch"
]

# Supported languages
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar", "hi"
]

# Human-in-the-Loop (HITL) Configuration
HITL_CONFIG = {
    # Confidence thresholds - below these trigger HITL
    "confidence_threshold": 0.70,  # Overall confidence below this triggers HITL
    "agent_agreement_threshold": 0.60,  # If agents agree less than 60%, trigger HITL

    # Severity thresholds - these always trigger HITL
    "always_review_severities": ["critical", "high"],

    # User profile triggers
    "high_profile_follower_threshold": 10000,  # Users with 10k+ followers
    "veteran_user_threshold_days": 365,  # 1+ year old accounts

    # Priority weights for queue ordering
    "priority_weights": {
        "critical": 100,  # Legal/safety concerns - immediate
        "high": 75,       # Severe violations
        "medium": 50,     # Moderate concerns
        "low": 25         # Edge cases, low-confidence
    },

    # HITL checkpoints in the workflow
    "checkpoints": {
        "post_analysis": True,      # After 3 analysis agents
        "post_react": True,         # After ReAct synthesis
        "pre_action": True,         # Before executing severe actions
        "post_reputation": True,    # After reputation scoring for suspensions/bans
    },

    # Auto-approve thresholds (skip HITL if all conditions met)
    "auto_approve_conditions": {
        "min_confidence": 0.90,
        "max_toxicity": 0.20,
        "no_violations": True,
        "user_reputation_min": 0.70
    }
}

# ReAct Loop Configuration
REACT_CONFIG = {
    # Weights for synthesizing agent decisions
    "agent_weights": {
        "Content Analysis Agent": 0.25,
        "Toxicity Detection Agent": 0.35,
        "Policy Violation Agent": 0.40
    },

    # Consensus thresholds
    "consensus_threshold": 0.67,  # 2 out of 3 agents must agree
    "strong_consensus_threshold": 1.0,  # All agents agree

    # Decision priority (higher wins in case of conflict)
    "decision_priority": {
        "ban_user": 6,
        "suspend_user": 5,
        "remove": 4,
        "warn": 3,
        "flag": 2,
        "approve": 1
    }
}
