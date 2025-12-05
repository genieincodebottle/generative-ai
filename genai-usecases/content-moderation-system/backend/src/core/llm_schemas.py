"""
Structured Output Schemas for LLM Responses.

This module defines Pydantic models for structured LLM outputs,
replacing string parsing with type-safe structured responses.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum

# ============================================================================
# Common Enums for LLM Responses
# ============================================================================

class ModerationDecision(str, Enum):
    """Possible moderation decisions."""
    APPROVE = "approve"
    FLAG = "flag"
    WARN = "warn"
    REMOVE = "remove"
    BAN_USER = "ban_user"
    SUSPEND_USER = "suspend_user"
    NEEDS_REVIEW = "needs_review"


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SeverityLevel(str, Enum):
    """Violation severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrendDirection(str, Enum):
    """Trend direction for metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    NEEDS_ATTENTION = "needs_attention"


# ============================================================================
# Content Analysis Agent Schemas
# ============================================================================

class TopicExtractionResponse(BaseModel):
    """Structured response for topic extraction."""
    topics: List[str] = Field(
        description="Main topics in the content (up to 5)",
        max_length=5
    )
    category: str = Field(
        description="Content category (news, opinion, meme, question, personal, commercial, etc.)"
    )
    entities: List[str] = Field(
        default=[],
        description="Detected entities (people, organizations, locations)"
    )
    sensitive_topics: List[str] = Field(
        default=[],
        description="Sensitive topics detected (politics, religion, health, finance)"
    )
    explicit_content: bool = Field(
        default=False,
        description="Whether explicit content is detected"
    )
    language: str = Field(
        default="en",
        description="Detected language code"
    )


class ContentAnalysisResponse(BaseModel):
    """Structured response for content analysis agent."""
    decision: ModerationDecision = Field(
        description="Recommended moderation action"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the decision (0.0 to 1.0)"
    )
    reasoning: str = Field(
        description="Explanation for the decision"
    )
    content_summary: str = Field(
        description="Brief summary of the content"
    )
    risk_factors: List[str] = Field(
        default=[],
        description="Identified risk factors"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Whether human review is recommended"
    )


# ============================================================================
# Toxicity Detection Agent Schemas
# ============================================================================

class ToxicityCategoryModel(BaseModel):
    """Individual toxicity category with score."""
    category: str = Field(description="Category name (profanity, threat, hate_speech, etc.)")
    score: float = Field(ge=0.0, le=1.0, description="Category score")
    evidence: List[str] = Field(default=[], description="Text evidence for this category")


class ToxicityAnalysisResponse(BaseModel):
    """Structured response for toxicity detection agent."""
    decision: ModerationDecision = Field(
        description="Recommended action: approve, flag, or remove"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the decision"
    )
    toxicity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall toxicity score"
    )
    toxicity_level: Literal["none", "low", "medium", "high", "severe"] = Field(
        description="Categorical toxicity level"
    )
    categories: List[ToxicityCategoryModel] = Field(
        default=[],
        description="Detected toxicity categories with scores"
    )
    is_satire: bool = Field(
        default=False,
        description="Whether content appears to be satire/humor"
    )
    is_quote: bool = Field(
        default=False,
        description="Whether toxic content is a quote/reference"
    )
    is_educational: bool = Field(
        default=False,
        description="Whether content is educational discussion"
    )
    context_notes: str = Field(
        default="",
        description="Notes about context that affects interpretation"
    )
    reasoning: str = Field(
        description="Detailed reasoning for the decision"
    )


# ============================================================================
# Policy Violation Agent Schemas
# ============================================================================

class PolicyViolation(BaseModel):
    """Individual policy violation."""
    policy: str = Field(description="Name of violated policy")
    severity: SeverityLevel = Field(description="Violation severity")
    evidence: str = Field(description="Evidence of violation")
    recommendation: str = Field(description="Recommended action for this violation")


class PolicyAnalysisResponse(BaseModel):
    """Structured response for policy violation agent."""
    decision: ModerationDecision = Field(
        description="Recommended moderation action"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the decision"
    )
    violations: List[PolicyViolation] = Field(
        default=[],
        description="List of detected policy violations"
    )
    overall_severity: SeverityLevel = Field(
        default=SeverityLevel.NONE,
        description="Overall severity of all violations"
    )
    is_repeat_offender: bool = Field(
        default=False,
        description="Whether user is a repeat offender"
    )
    escalation_recommended: bool = Field(
        default=False,
        description="Whether to escalate to senior moderator"
    )
    reasoning: str = Field(
        description="Detailed reasoning for the decision"
    )
    user_history_considered: bool = Field(
        default=True,
        description="Whether user history was considered"
    )


# ============================================================================
# User Reputation Agent Schemas
# ============================================================================

class ReputationAnalysisResponse(BaseModel):
    """Structured response for user reputation agent."""
    decision: ModerationDecision = Field(
        description="Recommended action regarding user"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the decision"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="User risk level"
    )
    trust_score: float = Field(
        ge=0.0,
        le=1.0,
        description="User trust score"
    )
    risk_factors: List[str] = Field(
        default=[],
        description="Identified risk factors"
    )
    positive_factors: List[str] = Field(
        default=[],
        description="Positive reputation factors"
    )
    recommended_tier: Literal["veteran", "trusted", "new_user", "flagged", "suspended", "banned"] = Field(
        description="Recommended user tier"
    )
    should_suspend: bool = Field(
        default=False,
        description="Whether user should be suspended"
    )
    should_ban: bool = Field(
        default=False,
        description="Whether user should be banned"
    )
    reasoning: str = Field(
        description="Detailed reasoning"
    )


# ============================================================================
# Appeal Review Agent Schemas
# ============================================================================

class AppealDecision(str, Enum):
    """Possible appeal decisions."""
    UPHOLD = "uphold"
    OVERTURN = "overturn"
    PARTIAL = "partial"
    NEEDS_MORE_INFO = "needs_more_info"


class AppealReviewResponse(BaseModel):
    """Structured response for appeal review agent."""
    appeal_decision: AppealDecision = Field(
        description="Decision on the appeal"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the decision"
    )
    original_decision_correct: bool = Field(
        description="Whether original moderation decision was correct"
    )
    new_evidence_found: bool = Field(
        default=False,
        description="Whether new evidence supports overturning"
    )
    false_positive_likely: bool = Field(
        default=False,
        description="Whether this was likely a false positive"
    )
    context_missing: bool = Field(
        default=False,
        description="Whether important context was missing"
    )
    recommended_action: ModerationDecision = Field(
        description="Recommended action after appeal"
    )
    user_notification: str = Field(
        description="Message to send to user about appeal result"
    )
    reasoning: str = Field(
        description="Detailed reasoning for appeal decision"
    )


# ============================================================================
# Action Enforcement Agent Schemas
# ============================================================================

class EnforcementAction(BaseModel):
    """Specific enforcement action to take."""
    action_type: str = Field(description="Type of action (remove_content, warn_user, etc.)")
    target: str = Field(description="Target of action (content_id, user_id)")
    severity: SeverityLevel = Field(description="Action severity")
    duration: Optional[str] = Field(default=None, description="Duration for temporary actions")


class ActionEnforcementResponse(BaseModel):
    """Structured response for action enforcement agent."""
    primary_action: ModerationDecision = Field(
        description="Primary action to take"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the action"
    )
    actions: List[EnforcementAction] = Field(
        default=[],
        description="List of specific actions to execute"
    )
    user_notification_required: bool = Field(
        default=True,
        description="Whether to notify the user"
    )
    user_notification_message: str = Field(
        default="",
        description="Message to send to user"
    )
    appeal_allowed: bool = Field(
        default=True,
        description="Whether user can appeal this decision"
    )
    escalate_to_legal: bool = Field(
        default=False,
        description="Whether to escalate to legal team"
    )
    reasoning: str = Field(
        description="Reasoning for enforcement actions"
    )


# ============================================================================
# ReAct Decision Loop Schemas
# ============================================================================

class AgentVote(BaseModel):
    """Individual agent's vote in synthesis."""
    agent_name: str = Field(description="Name of the agent")
    decision: ModerationDecision = Field(description="Agent's decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent's confidence")
    weight: float = Field(ge=0.0, le=1.0, description="Weight applied to this vote")


class ReActSynthesisResponse(BaseModel):
    """Structured response for ReAct decision synthesis."""
    final_decision: ModerationDecision = Field(
        description="Synthesized final decision"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in synthesized decision"
    )
    confidence_level: ConfidenceLevel = Field(
        description="Categorical confidence level"
    )
    consensus_level: Literal["strong", "moderate", "weak"] = Field(
        description="Level of agent agreement"
    )
    agent_votes: List[AgentVote] = Field(
        description="Individual agent votes"
    )
    conflicts: List[str] = Field(
        default=[],
        description="Identified conflicts between agents"
    )
    hitl_required: bool = Field(
        default=False,
        description="Whether human-in-the-loop review is required"
    )
    hitl_reasons: List[str] = Field(
        default=[],
        description="Reasons for requiring HITL"
    )
    hitl_priority: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Priority for HITL queue"
    )
    think_phase: str = Field(
        description="Reasoning from THINK phase"
    )
    act_phase: str = Field(
        description="Actions from ACT phase"
    )
    observe_phase: str = Field(
        description="Observations from OBSERVE phase"
    )


# ============================================================================
# HITL Checkpoint Schemas
# ============================================================================

class HITLCheckpointResponse(BaseModel):
    """Structured response for HITL checkpoint."""
    requires_review: bool = Field(
        description="Whether human review is required"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Review priority"
    )
    trigger_reasons: List[str] = Field(
        description="Reasons that triggered HITL"
    )
    content_summary: str = Field(
        description="Summary for human reviewer"
    )
    ai_recommendation: ModerationDecision = Field(
        description="AI's recommended decision"
    )
    ai_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="AI's confidence in recommendation"
    )
    key_concerns: List[str] = Field(
        default=[],
        description="Key concerns for reviewer"
    )
    relevant_policies: List[str] = Field(
        default=[],
        description="Relevant policies to consider"
    )
    similar_cases_count: int = Field(
        default=0,
        description="Number of similar past cases found"
    )


# ============================================================================
# Helper Functions for LLM Structured Output
# ============================================================================

def get_schema_prompt(schema_class: type[BaseModel]) -> str:
    """
    Generate a prompt-friendly schema description.

    Args:
        schema_class: Pydantic model class

    Returns:
        String description of expected JSON format
    """
    schema = schema_class.model_json_schema()
    return f"""
Respond with a JSON object matching this schema:
{schema}

Important:
- All fields must be present
- Use exact enum values where specified
- Confidence scores must be between 0.0 and 1.0
- Return ONLY valid JSON, no additional text
"""


def parse_llm_response(
    response: str,
    schema_class: type[BaseModel]
) -> Optional[BaseModel]:
    """
    Parse LLM response into structured Pydantic model.

    Args:
        response: Raw LLM response string
        schema_class: Target Pydantic model class

    Returns:
        Parsed model instance or None if parsing fails
    """
    import json

    try:
        # Try to find JSON in response
        text = response.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Find JSON object boundaries
        json_start = text.find("{")
        json_end = text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            return schema_class.model_validate(data)

        return None

    except (json.JSONDecodeError, Exception) as e:
        import logging
        logging.warning(f"Failed to parse LLM response: {e}")
        return None


def create_structured_prompt(
    base_prompt: str,
    schema_class: type[BaseModel],
    examples: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Create a complete prompt with schema instructions.

    Args:
        base_prompt: The main instruction prompt
        schema_class: Expected response schema
        examples: Optional example responses

    Returns:
        Complete prompt with schema instructions
    """
    schema_instruction = get_schema_prompt(schema_class)

    prompt_parts = [base_prompt, "\n", schema_instruction]

    if examples:
        prompt_parts.append("\nExamples:")
        for i, example in enumerate(examples, 1):
            import json
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"```json\n{json.dumps(example, indent=2)}\n```")

    return "\n".join(prompt_parts)
