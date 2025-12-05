"""
Content Moderation & Community Safety Multi-Agent System.

This module implements 6 specialized AI agents that work together to moderate content:
1. Content Analysis Agent - Analyzes content structure, sentiment, topics
2. Toxicity Detection Agent - Detects toxic language, hate speech, harassment
3. Policy Violation Agent - Checks against community guidelines and policies
4. User Reputation Scoring Agent - Evaluates user reputation and risk
5. Appeal Review Agent - Reviews appeals for content restoration
6. Action Enforcement Agent - Enforces moderation actions and notifies users

Enhanced with:
- ML-based toxicity detection (HateBERT)
- Structured LLM outputs using Pydantic schemas
"""

import json
import os
import logging
from typing import Dict, List, Any
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from ..core.models import (
    ContentState,
    AgentDecision,
    DecisionType,
    ContentStatus,
    ToxicityLevel,
    PolicyCategory,
    ReputationTier,
    HITLTriggerReason,
    TOXICITY_THRESHOLDS,
    HITL_CONFIG,
    REACT_CONFIG
)
from ..memory.memory import ModerationMemoryManager
from ..utils.tools import (
    analyze_text_sentiment,
    detect_toxicity,
    check_policy_violations,
    calculate_user_reputation,
    check_spam_indicators,
    detect_hate_speech_patterns
)
from ..core.llm_schemas import (
    TopicExtractionResponse,
    ToxicityAnalysisResponse,
    parse_llm_response,
    create_structured_prompt
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ContentModerationAgents:
    """
    Container for all 6 content moderation agents.

    Each agent is a function that takes the current ContentState and returns an updated ContentState.
    """

    def __init__(self):
        """Initialize the agents with LLM and memory manager."""
        logger.info("\nInitializing ContentModerationAgents...")

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("WARNING: GOOGLE_API_KEY is not set!")

        logger.info("Initializing LLM...")
        try:
            self.llm_flash = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                google_api_key=google_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize llm_flash: {e}")
            raise

        try:
            self.llm_pro = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                google_api_key=google_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize llm_pro: {e}")
            raise

        logger.info("Initializing ModerationMemoryManager...")
        try:
            self.memory_manager = ModerationMemoryManager()
        except Exception as e:
            logger.error(f"Failed to initialize memory_manager: {e}")
            raise

        logger.info("ContentModerationAgents initialization complete")

    def content_analysis_agent(self, state: ContentState) -> ContentState:
        """
        Agent 1: Content Analysis Agent

        Responsibilities:
        - Analyze content type and structure
        - Extract topics and entities
        - Detect sentiment
        - Identify sensitive content
        - Analyze multimodal content (images/videos)
        - Retrieve similar content from memory

        Returns updated state with:
        - Content category and topics
        - Sentiment analysis
        - Sensitive content flags
        - Extracted entities
        """
        logger.info("\nAGENT 1: Content Analysis Agent")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            content_id = state.get("content_id")
            content_text = state.get("content_text", "")
            content_type = state.get("content_type")
            user_profile = state.get("user_profile")

            # Analyze text sentiment
            sentiment_analysis = analyze_text_sentiment(content_text)
            extracted_data["sentiment"] = sentiment_analysis["sentiment"]
            extracted_data["sentiment_score"] = sentiment_analysis["score"]

            state["content_sentiment"] = sentiment_analysis["sentiment"]

            # Run toxicity detection early to capture the score
            toxicity_result = detect_toxicity(content_text)
            toxicity_score = toxicity_result["toxicity_score"]
            state["toxicity_score"] = toxicity_score
            state["toxicity_categories"] = toxicity_result["categories"]
            extracted_data["toxicity_score"] = toxicity_score
            extracted_data["toxicity_categories"] = toxicity_result["categories"]

            # Add toxicity flags
            if toxicity_score > 0.5:
                flags.append("high_toxicity")
            if "profanity" in toxicity_result["categories"]:
                flags.append("profanity")
            if "threat" in toxicity_result["categories"]:
                flags.append("threat")

            # Extract topics using LLM with structured output
            base_prompt = f"""
            Analyze this social media content and extract key information:

            Content: "{content_text}"
            Platform: {state.get("content_metadata").platform}
            Type: {content_type}

            Extract:
            1. Main topics (up to 5)
            2. Content category (news, opinion, meme, question, personal, commercial, etc.)
            3. Detected entities (people, organizations, locations)
            4. Sensitive topics (politics, religion, health, finance, etc.)
            5. Any explicit content indicators
            """

            topic_extraction_prompt = create_structured_prompt(
                base_prompt,
                TopicExtractionResponse
            )

            response = self.llm_flash.invoke(topic_extraction_prompt)
            analysis = response.content

            # Parse LLM response using structured parser
            parsed_response = parse_llm_response(analysis, TopicExtractionResponse)

            if parsed_response:
                analysis_data = parsed_response.model_dump()
            else:
                # Fallback: try manual JSON parsing
                try:
                    json_start = analysis.find("{")
                    json_end = analysis.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        analysis_data = json.loads(analysis[json_start:json_end])
                    else:
                        analysis_data = {
                            "topics": [],
                            "category": "general",
                            "entities": [],
                            "sensitive_topics": [],
                            "explicit_content": False
                        }
                except json.JSONDecodeError:
                    analysis_data = {
                        "topics": [],
                        "category": "general",
                        "entities": [],
                        "sensitive_topics": [],
                        "explicit_content": False
                    }

            # Update state
            state["content_topics"] = analysis_data.get("topics", [])
            state["content_category"] = analysis_data.get("category", "general")
            state["contains_sensitive_content"] = len(analysis_data.get("sensitive_topics", [])) > 0
            state["explicit_content_detected"] = analysis_data.get("explicit_content", False)

            extracted_data["topics"] = analysis_data.get("topics", [])
            extracted_data["category"] = analysis_data.get("category", "general")
            extracted_data["entities"] = analysis_data.get("entities", [])

            # Check for sensitive content
            if state["contains_sensitive_content"]:
                flags.append("sensitive_content")
                recommendations.append(f"Contains sensitive topics: {', '.join(analysis_data.get('sensitive_topics', []))}")

            if state["explicit_content_detected"]:
                flags.append("explicit_content")
                recommendations.append("Explicit content detected")

            # Retrieve similar content from memory using agent-scoped retrieval
            similar_content = self.memory_manager.retrieve_similar_content_for_agent(
                agent_name="Content Analysis Agent",
                content_text=content_text,
                content_type=content_type,
                user_id=user_profile.user_id,
                n_results=5,
                min_confidence=0.6,  # Only learn from confident decisions
                only_correct=True  # Only learn from correct decisions
            )

            state["similar_content"] = similar_content
            extracted_data["similar_content_found"] = len(similar_content)

            if similar_content:
                # Check if similar content was previously flagged
                flagged_similar = [c for c in similar_content if c.get("was_removed", False)]
                if flagged_similar:
                    flags.append("similar_to_removed_content")
                    recommendations.append(f"Similar to {len(flagged_similar)} previously removed content")

            # Determine confidence and decision
            confidence = 0.85

            if state["explicit_content_detected"]:
                confidence -= 0.20
            if len(flags) > 2:
                confidence -= 0.15

            # Use LLM for final analysis
            final_prompt = f"""
            Content Analysis Summary:

            Content: "{content_text}"
            Category: {state['content_category']}
            Topics: {', '.join(state['content_topics'])}
            Sentiment: {state['content_sentiment']}
            Explicit Content: {state['explicit_content_detected']}
            Sensitive Content: {state['contains_sensitive_content']}
            Similar Flagged Content: {len(flagged_similar) if similar_content else 0}

            User: {user_profile.username}
            Account Age: {user_profile.account_age_days} days
            Previous Violations: {user_profile.total_violations}

            Provide initial assessment:
            - Should this proceed to toxicity detection? (APPROVE or FLAG)
            - Any immediate concerns?
            - Confidence level (0.0 to 1.0)
            """

            response = self.llm_flash.invoke(final_prompt)
            reasoning = response.content

            # Determine decision
            if "FLAG" in reasoning or state["explicit_content_detected"]:
                decision_type = DecisionType.FLAG
                requires_review = True
            else:
                decision_type = DecisionType.APPROVE
                requires_review = confidence < 0.70

            decision = AgentDecision(
                agent_name="Content Analysis Agent",
                decision=decision_type,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=reasoning,
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=requires_review,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # Update state
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["current_agent"] = "Content Analysis Agent"
            state["requires_human_review"] = state.get("requires_human_review", False) or requires_review

            # Set next status
            if decision_type == DecisionType.APPROVE:
                state["status"] = ContentStatus.TOXICITY_CHECK.value
            else:
                state["status"] = ContentStatus.FLAGGED.value

        except Exception as e:
            logger.error(f"\nError in Content Analysis Agent: {e}")
            decision = AgentDecision(
                agent_name="Content Analysis Agent",
                decision=DecisionType.NEEDS_REVIEW,
                confidence=0.0,
                reasoning=f"Error analyzing content: {str(e)}",
                flags=["processing_error"],
                recommendations=["Manual review required"],
                extracted_data={},
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["status"] = ContentStatus.UNDER_REVIEW.value
            state["requires_human_review"] = True
            state["current_agent"] = "Content Analysis Agent"

        return state

    def toxicity_detection_agent(self, state: ContentState) -> ContentState:
        """
        Agent 2: Toxicity Detection Agent

        Responsibilities:
        - Detect toxic language and profanity
        - Identify hate speech patterns
        - Detect harassment and bullying
        - Calculate toxicity scores
        - Identify threatening language
        - Check against known toxic patterns

        Returns updated state with:
        - Toxicity score (0.0 to 1.0)
        - Toxicity categories
        - Hate speech detection results
        - Harassment indicators
        """
        logger.info("\nAGENT 2: Toxicity Detection Agent")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            content_id = state.get("content_id")
            content_text = state.get("content_text", "")
            user_profile = state.get("user_profile")

            # Detect toxicity
            toxicity_result = detect_toxicity(content_text)

            extracted_data["toxicity_score"] = toxicity_result["toxicity_score"]
            extracted_data["categories"] = toxicity_result["categories"]
            extracted_data["profanity_count"] = toxicity_result["profanity_count"]

            # Update state
            state["toxicity_score"] = toxicity_result["toxicity_score"]
            state["toxicity_categories"] = toxicity_result["categories"]

            # Determine toxicity level
            score = toxicity_result["toxicity_score"]
            if score >= TOXICITY_THRESHOLDS["severe"]:
                state["toxicity_level"] = ToxicityLevel.SEVERE.value
            elif score >= TOXICITY_THRESHOLDS["high"]:
                state["toxicity_level"] = ToxicityLevel.HIGH.value
            elif score >= TOXICITY_THRESHOLDS["medium"]:
                state["toxicity_level"] = ToxicityLevel.MEDIUM.value
            elif score >= TOXICITY_THRESHOLDS["none"]:
                state["toxicity_level"] = ToxicityLevel.LOW.value
            else:
                state["toxicity_level"] = ToxicityLevel.NONE.value

            # Detect hate speech
            hate_speech_result = detect_hate_speech_patterns(content_text)
            state["hate_speech_detected"] = hate_speech_result["detected"]
            extracted_data["hate_speech_score"] = hate_speech_result["score"]

            if hate_speech_result["detected"]:
                flags.append("hate_speech")
                recommendations.append(f"Hate speech detected: {', '.join(hate_speech_result['patterns'])}")

            # Check for harassment patterns
            harassment_keywords = ["kill yourself", "kys", "die", "harass", "stalk", "threaten"]
            harassment_detected = any(keyword in content_text.lower() for keyword in harassment_keywords)
            state["harassment_detected"] = harassment_detected

            if harassment_detected:
                flags.append("harassment")
                recommendations.append("Harassment language detected")

            # Check toxicity categories
            if "profanity" in toxicity_result["categories"]:
                flags.append("profanity")
            if "insult" in toxicity_result["categories"]:
                flags.append("insult")
            if "threat" in toxicity_result["categories"]:
                flags.append("threat")
                recommendations.append("Threatening language detected")

            # Use LLM for deeper analysis with structured output
            base_analysis_prompt = f"""
            Analyze this content for toxicity and harmful language:

            Content: "{content_text}"

            Toxicity Analysis (from ML classifier):
            - Score: {score:.2f} (0.0 = safe, 1.0 = extremely toxic)
            - Level: {state['toxicity_level']}
            - Categories: {', '.join(toxicity_result.get('categories', []))}
            - Detection Method: {toxicity_result.get('detection_method', 'unknown')}
            - ML Confidence: {toxicity_result.get('confidence', 0.7):.2f}
            - Hate Speech: {hate_speech_result['detected']}
            - Harassment: {harassment_detected}

            User Context:
            - Username: {user_profile.username}
            - Previous Violations: {user_profile.total_violations}
            - Previous Warnings: {user_profile.previous_warnings}

            Previous Agent Decisions:
            {self._format_previous_decisions(state)}

            Evaluate the content considering:
            1. Is this content toxic or harmful?
            2. What specific concerns exist?
            3. Context - satire, quotes, and educational content may contain toxic language without being harmful
            4. Is this a false positive? (e.g., discussing toxicity, quoting someone, artistic expression)
            """

            toxicity_prompt = create_structured_prompt(
                base_analysis_prompt,
                ToxicityAnalysisResponse
            )

            response = self.llm_pro.invoke(toxicity_prompt)
            analysis = response.content

            # Parse structured response
            parsed_toxicity = parse_llm_response(analysis, ToxicityAnalysisResponse)

            # Determine confidence and decision from structured response
            if parsed_toxicity:
                # Use LLM's structured assessment
                llm_decision = parsed_toxicity.decision.value
                confidence = parsed_toxicity.confidence

                # Adjust confidence based on context flags
                if parsed_toxicity.is_satire or parsed_toxicity.is_quote or parsed_toxicity.is_educational:
                    confidence = max(confidence, 0.75)  # Higher confidence if context understood

                # Map to DecisionType
                if llm_decision == "remove":
                    decision_type = DecisionType.REMOVE
                    requires_review = True
                elif llm_decision == "flag" or llm_decision == "warn":
                    decision_type = DecisionType.FLAG
                    requires_review = True
                else:
                    decision_type = DecisionType.APPROVE
                    requires_review = confidence < 0.75

                reasoning = parsed_toxicity.reasoning
                if parsed_toxicity.context_notes:
                    reasoning += f"\n\nContext: {parsed_toxicity.context_notes}"
            else:
                # Fallback to original string parsing
                confidence = 0.90
                if score > 0.7:
                    confidence -= 0.15
                if hate_speech_result["detected"]:
                    confidence -= 0.20
                if harassment_detected:
                    confidence -= 0.20

                if "REMOVE" in analysis or state["toxicity_level"] == ToxicityLevel.SEVERE.value:
                    decision_type = DecisionType.REMOVE
                    requires_review = True
                elif "FLAG" in analysis or score > 0.6:
                    decision_type = DecisionType.FLAG
                    requires_review = True
                else:
                    decision_type = DecisionType.APPROVE
                    requires_review = confidence < 0.75

                reasoning = analysis

            decision = AgentDecision(
                agent_name="Toxicity Detection Agent",
                decision=decision_type,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=reasoning,
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=requires_review,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # Update state
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["current_agent"] = "Toxicity Detection Agent"
            state["requires_human_review"] = state.get("requires_human_review", False) or requires_review

            # Set next status
            if decision_type == DecisionType.APPROVE:
                state["status"] = ContentStatus.POLICY_CHECK.value
            elif decision_type == DecisionType.REMOVE:
                state["status"] = ContentStatus.REMOVED.value
            else:
                state["status"] = ContentStatus.FLAGGED.value

        except Exception as e:
            logger.error(f"\nError in Toxicity Detection Agent: {e}")
            decision = AgentDecision(
                agent_name="Toxicity Detection Agent",
                decision=DecisionType.NEEDS_REVIEW,
                confidence=0.0,
                reasoning=f"Error detecting toxicity: {str(e)}",
                flags=["processing_error"],
                recommendations=["Manual review required"],
                extracted_data={},
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["status"] = ContentStatus.UNDER_REVIEW.value
            state["requires_human_review"] = True
            state["current_agent"] = "Toxicity Detection Agent"

        return state

    def policy_violation_agent(self, state: ContentState) -> ContentState:
        """
        Agent 3: Policy Violation Agent

        Responsibilities:
        - Check content against community guidelines
        - Identify policy violations
        - Assess violation severity
        - Check for spam indicators
        - Detect misinformation patterns
        - Recommend appropriate actions

        Returns updated state with:
        - Policy violations list
        - Violation severity
        - Recommended moderation action
        """
        logger.info("\nAGENT 3: Policy Violation Agent")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            content_id = state.get("content_id")
            content_text = state.get("content_text", "")
            user_profile = state.get("user_profile")
            toxicity_score = state.get("toxicity_score", 0.0)

            logger.info(f"\nChecking policy violations for content {content_id}")

            # Check policy violations
            policy_result = check_policy_violations(
                content_text=content_text,
                content_type=state.get("content_type"),
                user_reputation=user_profile.reputation_score,
                toxicity_score=toxicity_score
            )

            extracted_data["violations"] = policy_result["violations"]
            extracted_data["severity"] = policy_result["severity"]

            # Update state
            state["policy_violations"] = policy_result["violations"]
            state["violation_severity"] = policy_result["severity"]
            state["policy_flags"] = policy_result["flags"]

            # Check for spam
            spam_result = check_spam_indicators(content_text, user_profile)
            if spam_result["is_spam"]:
                flags.append("spam")
                state["policy_violations"].append(PolicyCategory.SPAM.value)
                recommendations.append(f"Spam indicators: {', '.join(spam_result['indicators'])}")

            # Add flags for specific violations
            for violation in policy_result["violations"]:
                if violation == PolicyCategory.HATE_SPEECH.value:
                    flags.append("hate_speech_policy")
                elif violation == PolicyCategory.HARASSMENT.value:
                    flags.append("harassment_policy")
                elif violation == PolicyCategory.VIOLENCE.value:
                    flags.append("violence_policy")
                elif violation == PolicyCategory.ILLEGAL_ACTIVITY.value:
                    flags.append("illegal_activity")

            # Use LLM for policy analysis
            analysis_prompt = f"""
            Review this content against community policies:

            Content: "{content_text}"

            Policy Check Results:
            - Violations: {', '.join(policy_result['violations']) if policy_result['violations'] else 'None'}
            - Severity: {policy_result['severity']}
            - Flags: {', '.join(policy_result['flags'])}

            Content Analysis:
            - Toxicity Score: {toxicity_score:.2f}
            - Hate Speech: {state.get('hate_speech_detected', False)}
            - Harassment: {state.get('harassment_detected', False)}

            User Profile:
            - Reputation: {user_profile.reputation_score:.2f}
            - Previous Violations: {user_profile.total_violations}
            - Previous Warnings: {user_profile.previous_warnings}

            Community Guidelines:
            1. No hate speech or discrimination
            2. No harassment or bullying
            3. No violence or threats
            4. No spam or manipulation
            5. No illegal content
            6. No misinformation
            7. Respect privacy

            Previous Agent Decisions:
            {self._format_previous_decisions(state)}

            Determine:
            1. Does this violate community guidelines?
            2. What is the severity?
            3. Recommended action: APPROVE, WARN, REMOVE, SUSPEND_USER, or BAN_USER
            4. Should this go to reputation scoring?
            """

            response = self.llm_pro.invoke(analysis_prompt)
            analysis = response.content

            # Store recommended action
            if "BAN_USER" in analysis:
                state["recommended_action"] = "ban_user"
            elif "SUSPEND_USER" in analysis:
                state["recommended_action"] = "suspend_user"
            elif "REMOVE" in analysis:
                state["recommended_action"] = "remove"
            elif "WARN" in analysis:
                state["recommended_action"] = "warn"
            else:
                state["recommended_action"] = "approve"

            # Determine confidence
            confidence = 0.85

            if policy_result["severity"] == "critical":
                confidence += 0.10
            elif policy_result["severity"] == "low":
                confidence -= 0.15

            # Determine decision
            if policy_result["severity"] in ["critical", "high"]:
                decision_type = DecisionType.REMOVE
                requires_review = True
            elif policy_result["severity"] == "medium":
                decision_type = DecisionType.FLAG
                requires_review = True
            elif len(policy_result["violations"]) > 0:
                decision_type = DecisionType.WARN
                requires_review = False
            else:
                decision_type = DecisionType.APPROVE
                requires_review = False

            decision = AgentDecision(
                agent_name="Policy Violation Agent",
                decision=decision_type,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=analysis,
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=requires_review,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # Update state
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["current_agent"] = "Policy Violation Agent"
            state["requires_human_review"] = state.get("requires_human_review", False) or requires_review

            # Set next status
            if decision_type == DecisionType.APPROVE:
                state["status"] = ContentStatus.REPUTATION_SCORING.value
            elif decision_type == DecisionType.REMOVE:
                state["status"] = ContentStatus.REMOVED.value
            else:
                state["status"] = ContentStatus.FLAGGED.value

        except Exception as e:
            logger.error(f"\nError in Policy Violation Agent: {e}")
            decision = AgentDecision(
                agent_name="Policy Violation Agent",
                decision=DecisionType.NEEDS_REVIEW,
                confidence=0.0,
                reasoning=f"Error checking policies: {str(e)}",
                flags=["processing_error"],
                recommendations=["Manual review required"],
                extracted_data={},
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["status"] = ContentStatus.UNDER_REVIEW.value
            state["requires_human_review"] = True
            state["current_agent"] = "Policy Violation Agent"

        return state

    def user_reputation_agent(self, state: ContentState) -> ContentState:
        """
        Agent 4: User Reputation Scoring Agent

        Responsibilities:
        - Calculate user reputation score
        - Analyze user history
        - Identify repeat offenders
        - Determine user risk level
        - Check for pattern violations
        - Recommend user-level actions

        Returns updated state with:
        - User reputation score
        - User risk assessment
        - Historical pattern analysis
        - User-level action recommendations
        """
        logger.info("\nAGENT 4: User Reputation Scoring Agent")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            user_profile = state.get("user_profile")

            # Calculate updated reputation
            reputation_result = calculate_user_reputation(
                current_score=user_profile.reputation_score,
                total_posts=user_profile.total_posts,
                total_violations=user_profile.total_violations,
                previous_warnings=user_profile.previous_warnings,
                previous_suspensions=user_profile.previous_suspensions,
                account_age_days=user_profile.account_age_days,
                current_violation_severity=state.get("violation_severity", "none")
            )

            extracted_data["new_reputation_score"] = reputation_result["reputation_score"]
            extracted_data["reputation_tier"] = reputation_result["tier"]
            extracted_data["risk_score"] = reputation_result["risk_score"]

            # Update state
            state["user_reputation_score"] = reputation_result["reputation_score"]
            state["user_reputation_tier"] = reputation_result["tier"]
            state["user_risk_score"] = reputation_result["risk_score"]

            # Check user history
            user_history = self.memory_manager.get_user_history(user_profile.user_id)
            state["user_history_flags"] = []

            # Analyze patterns
            if user_profile.total_violations >= 3:
                flags.append("repeat_offender")
                state["user_history_flags"].append("repeat_offender")
                recommendations.append("User has 3+ previous violations")

            if user_profile.previous_suspensions > 0:
                flags.append("previously_suspended")
                state["user_history_flags"].append("previously_suspended")
                recommendations.append(f"User was suspended {user_profile.previous_suspensions} time(s)")

            if reputation_result["risk_score"] > 0.7:
                flags.append("high_risk_user")
                state["user_history_flags"].append("high_risk_user")
                recommendations.append("High-risk user based on history")

            # Check for rapid violations (3+ in last 7 days)
            recent_violations = [v for v in user_history if
                                (datetime.now() - datetime.fromisoformat(v.get("timestamp", "2024-01-01"))).days <= 7]

            state["similar_violations_count"] = len(recent_violations)

            if len(recent_violations) >= 3:
                flags.append("rapid_violations")
                recommendations.append(f"{len(recent_violations)} violations in last 7 days")

            # Use LLM for reputation analysis
            analysis_prompt = f"""
            Analyze user reputation and risk:

            User Profile:
            - Username: {user_profile.username}
            - Account Age: {user_profile.account_age_days} days
            - Total Posts: {user_profile.total_posts}
            - Total Violations: {user_profile.total_violations}
            - Warnings: {user_profile.previous_warnings}
            - Suspensions: {user_profile.previous_suspensions}

            Current Scores:
            - Old Reputation: {user_profile.reputation_score:.2f}
            - New Reputation: {reputation_result['reputation_score']:.2f}
            - Risk Score: {reputation_result['risk_score']:.2f}
            - Tier: {reputation_result['tier']}

            Current Content Violation:
            - Severity: {state.get('violation_severity', 'none')}
            - Toxicity: {state.get('toxicity_score', 0.0):.2f}

            History:
            - Recent Violations (7 days): {len(recent_violations)}
            - Total Historical Violations: {len(user_history)}

            Current Content Decision: {state.get('recommended_action', 'unknown')}

            Previous Agent Decisions:
            {self._format_previous_decisions(state)}

            Determine:
            1. Is this user a repeat offender?
            2. Should we escalate to user-level action?
            3. Recommended action: APPROVE, WARN, SUSPEND_USER (1-7 days), or BAN_USER
            4. Justification for the action
            """

            response = self.llm_flash.invoke(analysis_prompt)
            analysis = response.content

            # Determine confidence
            confidence = 0.80

            if reputation_result["risk_score"] > 0.7:
                confidence += 0.10
            if len(flags) > 3:
                confidence -= 0.10

            # Determine decision based on risk
            if "BAN_USER" in analysis or reputation_result["tier"] == ReputationTier.BANNED.value:
                decision_type = DecisionType.BAN_USER
                requires_review = True
            elif "SUSPEND_USER" in analysis or reputation_result["risk_score"] > 0.7:
                decision_type = DecisionType.SUSPEND_USER
                requires_review = True
            elif state.get("recommended_action") == "remove":
                decision_type = DecisionType.REMOVE
                requires_review = True
            elif state.get("recommended_action") == "warn":
                decision_type = DecisionType.WARN
                requires_review = False
            else:
                decision_type = DecisionType.APPROVE
                requires_review = False

            decision = AgentDecision(
                agent_name="User Reputation Scoring Agent",
                decision=decision_type,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=analysis,
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=requires_review,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # Update state
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["current_agent"] = "User Reputation Scoring Agent"
            state["requires_human_review"] = state.get("requires_human_review", False) or requires_review

            # Set next status
            if decision_type in [DecisionType.BAN_USER, DecisionType.SUSPEND_USER, DecisionType.REMOVE]:
                state["status"] = ContentStatus.ACTION_ENFORCEMENT.value
            elif decision_type == DecisionType.WARN:
                state["status"] = ContentStatus.WARNED.value
            else:
                state["status"] = ContentStatus.APPROVED.value

        except Exception as e:
            logger.error(f"\nError in User Reputation Agent: {e}")
            decision = AgentDecision(
                agent_name="User Reputation Scoring Agent",
                decision=DecisionType.NEEDS_REVIEW,
                confidence=0.0,
                reasoning=f"Error evaluating reputation: {str(e)}",
                flags=["processing_error"],
                recommendations=["Manual review required"],
                extracted_data={},
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["status"] = ContentStatus.UNDER_REVIEW.value
            state["requires_human_review"] = True
            state["current_agent"] = "User Reputation Scoring Agent"

        return state

    def appeal_review_agent(self, state: ContentState) -> ContentState:
        """
        Agent 5: Appeal Review Agent

        Responsibilities:
        - Review user appeals
        - Re-evaluate original decisions
        - Check for false positives
        - Analyze context and intent
        - Make final appeal decision
        - Learn from overturned decisions

        Returns updated state with:
        - Appeal decision
        - Reasoning for appeal outcome
        - Updated moderation action
        """
        logger.info("\nAGENT 5: Appeal Review Agent")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            content_id = state.get("content_id")
            appeal_reason = state.get("appeal_reason", "")
            original_decision = state.get("original_decision", "")
            user_profile = state.get("user_profile")

            extracted_data["is_appeal"] = True
            extracted_data["original_decision"] = original_decision

            # Use LLM for appeal analysis
            analysis_prompt = f"""
            Review this appeal for a moderation decision:

            Original Content: "{state.get('content_text', '')}"

            Original Decision:
            - Action: {original_decision}
            - Violations: {', '.join(state.get('policy_violations', []))}
            - Toxicity Score: {state.get('toxicity_score', 0.0):.2f}

            User's Appeal:
            "{appeal_reason}"

            User Context:
            - Username: {user_profile.username}
            - Reputation: {user_profile.reputation_score:.2f}
            - Previous Violations: {user_profile.total_violations}

            Original Agent Decisions:
            {self._format_previous_decisions(state)}

            Appeal Review Guidelines:
            1. Was the original decision correct?
            2. Is there missing context that changes interpretation?
            3. Could this be satire, quote, or educational content?
            4. Is the user's appeal reasonable?
            5. Are there signs of false positive detection?

            Decide:
            - UPHOLD: Original decision was correct
            - OVERTURN: Decision was wrong, restore content
            - PARTIAL: Reduce severity (e.g., warn instead of remove)

            Provide detailed reasoning and confidence level.
            """

            response = self.llm_pro.invoke(analysis_prompt)
            analysis = response.content

            # Determine decision
            if "OVERTURN" in analysis:
                decision_type = DecisionType.APPROVE
                recommendations.append("Appeal approved - content will be restored")
                state["status"] = ContentStatus.APPROVED.value
            elif "PARTIAL" in analysis:
                decision_type = DecisionType.WARN
                recommendations.append("Appeal partially approved - action reduced to warning")
                state["status"] = ContentStatus.WARNED.value
            else:
                decision_type = DecisionType.REMOVE
                recommendations.append("Appeal denied - original decision upheld")
                state["status"] = ContentStatus.REMOVED.value

            confidence = 0.75

            decision = AgentDecision(
                agent_name="Appeal Review Agent",
                decision=decision_type,
                confidence=confidence,
                reasoning=analysis,
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=False,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # Update state
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["current_agent"] = "Appeal Review Agent"
            state["review_decision"] = decision_type.value
            state["review_timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"\nError in Appeal Review Agent: {e}")
            decision = AgentDecision(
                agent_name="Appeal Review Agent",
                decision=DecisionType.NEEDS_REVIEW,
                confidence=0.0,
                reasoning=f"Error reviewing appeal: {str(e)}",
                flags=["processing_error"],
                recommendations=["Manual review required"],
                extracted_data={},
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["status"] = ContentStatus.UNDER_REVIEW.value
            state["requires_human_review"] = True
            state["current_agent"] = "Appeal Review Agent"

        return state

    def action_enforcement_agent(self, state: ContentState) -> ContentState:
        """
        Agent 6: Action Enforcement Agent

        Responsibilities:
        - Execute moderation actions
        - Notify users of decisions
        - Remove/hide content if needed
        - Apply user suspensions/bans
        - Record actions in audit log
        - Store patterns in memory for learning

        Returns updated state with:
        - Executed action details
        - User notification status
        - Audit log entry
        """
        logger.info("\nAGENT 6: Action Enforcement Agent")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            content_id = state.get("content_id")
            user_profile = state.get("user_profile")

            # Determine final action based on all agent decisions
            final_action = self._determine_final_action(state)

            # Execute action
            if final_action == DecisionType.REMOVE.value:
                state["content_removed"] = True
                state["user_notified"] = True
                state["moderation_action"] = "removed"
                recommendations.append("Content has been removed")

            elif final_action == DecisionType.WARN.value:
                state["content_removed"] = False
                state["user_notified"] = True
                state["moderation_action"] = "warned"
                recommendations.append("User has been warned")

            elif final_action == DecisionType.SUSPEND_USER.value:
                state["content_removed"] = True
                state["user_suspended"] = True
                state["user_notified"] = True
                state["suspension_duration_days"] = self._calculate_suspension_duration(state)
                state["moderation_action"] = "user_suspended"
                recommendations.append(f"User suspended for {state['suspension_duration_days']} days")

            elif final_action == DecisionType.BAN_USER.value:
                state["content_removed"] = True
                state["user_suspended"] = True
                state["user_notified"] = True
                state["moderation_action"] = "user_banned"
                recommendations.append("User has been permanently banned")

            else:
                state["content_removed"] = False
                state["user_notified"] = False
                state["moderation_action"] = "approved"
                recommendations.append("Content approved")

            # Generate action reason
            action_reason_prompt = f"""
            Generate a user-friendly explanation for this moderation action:

            Action: {final_action}
            Content: "{state.get('content_text', '')[:200]}..."

            Violations: {', '.join(state.get('policy_violations', []))}
            Toxicity Score: {state.get('toxicity_score', 0.0):.2f}

            Agent Decisions:
            {self._format_previous_decisions(state)}

            Write a clear, professional message (2-3 sentences) explaining:
            1. What action was taken
            2. Why it was taken
            3. What the user can do next (if applicable)

            Be respectful but firm.
            """

            response = self.llm_flash.invoke(action_reason_prompt)
            action_reason = response.content

            state["action_reason"] = action_reason
            state["action_timestamp"] = datetime.now().isoformat()

            # Store in memory for learning with enhanced metadata
            # Determine primary agent and decision context
            agent_decisions = state.get("agent_decisions", [])
            primary_agent = "Action Enforcement Agent"
            confidence = 0.8  # Default confidence

            # Get the agent that made the most impactful decision
            for decision in reversed(agent_decisions):
                if hasattr(decision, 'decision') and decision.decision != DecisionType.APPROVE:
                    primary_agent = decision.agent_name
                    confidence = decision.confidence
                    break

            # Build decision context for semantic learning
            toxicity_score = state.get("toxicity_score", 0.0)
            violations = state.get("policy_violations", [])

            # Context string based on toxicity and violations
            if toxicity_score < 0.3:
                tox_context = "toxicity_low"
            elif toxicity_score < 0.6:
                tox_context = "toxicity_medium"
            elif toxicity_score < 0.8:
                tox_context = "toxicity_high"
            else:
                tox_context = "toxicity_severe"

            violation_context = violations[0] if violations else "no_violation"
            decision_context = f"{tox_context}_{violation_context}"

            self.memory_manager.store_moderation_decision(
                content_id=content_id,
                content_text=state.get("content_text", ""),
                user_id=user_profile.user_id,
                action=final_action,
                violations=violations,
                toxicity_score=toxicity_score,
                agent_decisions=agent_decisions,
                primary_agent=primary_agent,
                decision_context=decision_context,
                confidence=confidence,
                was_appealed=False,  # Will be updated if appealed later
                appeal_outcome=None
            )

            extracted_data["action_executed"] = final_action
            extracted_data["user_notified"] = state["user_notified"]
            extracted_data["content_removed"] = state.get("content_removed", False)

            decision = AgentDecision(
                agent_name="Action Enforcement Agent",
                decision=DecisionType.APPROVE,  # Always approve - action completed
                confidence=1.0,
                reasoning=action_reason,
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=False,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # Update final state
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["current_agent"] = "Action Enforcement Agent"
            state["processed_at"] = datetime.now().isoformat()

            # Set final status
            if final_action == DecisionType.REMOVE.value:
                state["status"] = ContentStatus.REMOVED.value
            elif final_action == DecisionType.WARN.value:
                state["status"] = ContentStatus.WARNED.value
            else:
                state["status"] = ContentStatus.APPROVED.value

        except Exception as e:
            logger.error(f"\nError in Action Enforcement Agent: {e}")
            decision = AgentDecision(
                agent_name="Action Enforcement Agent",
                decision=DecisionType.NEEDS_REVIEW,
                confidence=0.0,
                reasoning=f"Error enforcing action: {str(e)}",
                flags=["processing_error"],
                recommendations=["Manual review required"],
                extracted_data={},
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["status"] = ContentStatus.UNDER_REVIEW.value
            state["requires_human_review"] = True
            state["current_agent"] = "Action Enforcement Agent"

        return state

    def react_decision_loop_agent(self, state: ContentState) -> ContentState:
        """
        ReAct Decision Loop Agent - Synthesizes all agent decisions using Think-Act-Observe pattern.

        This agent runs AFTER the 3 analysis agents (Content, Toxicity, Policy) and:
        1. THINK: Analyzes and synthesizes all agent outputs
        2. ACT: Makes a consolidated decision
        3. OBSERVE: Records the decision and determines next steps

        Also implements Human-in-the-Loop (HITL) checkpoint evaluation.
        """
        logger.info("\nREACT DECISION LOOP: Think → Act → Observe")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            content_id = state.get("content_id")
            content_text = state.get("content_text", "")
            user_profile = state.get("user_profile")
            agent_decisions = state.get("agent_decisions", [])

            # ═══════════════════════════════════════════════════
            # THINK PHASE - Analyze all agent outputs
            # ═══════════════════════════════════════════════════
            logger.info("\nTHINK PHASE: Analyzing agent outputs...")

            think_analysis = self._analyze_agent_decisions(agent_decisions)
            state["react_think_output"] = json.dumps(think_analysis)
            extracted_data["think_analysis"] = think_analysis

            # Check for HITL triggers based on analysis
            hitl_triggers = self._evaluate_hitl_triggers(state, think_analysis)

            if hitl_triggers:
                state["hitl_trigger_reasons"] = hitl_triggers
                flags.append("hitl_triggered")

            # Use LLM for deeper synthesis
            synthesis_prompt = f"""
            You are a senior content moderation supervisor reviewing agent decisions.

            Content: "{content_text[:500]}..."

            === AGENT DECISIONS ===
            {self._format_detailed_decisions(agent_decisions)}

            === ANALYSIS SUMMARY ===
            - Total Agents Reporting: {think_analysis['total_agents']}
            - Average Confidence: {think_analysis['avg_confidence']:.2%}
            - Consensus Level: {think_analysis['consensus_level']}
            - Toxicity Score: {state.get('toxicity_score', 0.0):.2f}
            - Policy Violations: {state.get('policy_violations', [])}
            - Violation Severity: {state.get('violation_severity', 'none')}

            === USER CONTEXT ===
            - Username: {user_profile.username}
            - Reputation: {user_profile.reputation_score:.2f}
            - Account Age: {user_profile.account_age_days} days
            - Previous Violations: {user_profile.total_violations}
            - Verified: {user_profile.verified}

            THINK Phase - Analyze:
            1. Do the agents agree on the assessment?
            2. What is the overall risk level of this content?
            3. Are there any conflicting signals that need human review?
            4. What context might agents have missed?

            Provide your analysis in a structured format.
            """

            response = self.llm_pro.invoke(synthesis_prompt)
            think_output = response.content

            # ═══════════════════════════════════════════════════
            # ACT PHASE - Make consolidated decision
            # ═══════════════════════════════════════════════════
            act_decision = self._synthesize_final_decision(agent_decisions, think_analysis, state)
            state["react_act_decision"] = act_decision["decision"]
            state["react_confidence"] = act_decision["confidence"]
            extracted_data["act_decision"] = act_decision

            # ═══════════════════════════════════════════════════
            # OBSERVE PHASE - Record and plan next steps
            # ═══════════════════════════════════════════════════

            observe_result = {
                "decision_recorded": True,
                "next_step": "reputation_scoring" if act_decision["decision"] != "approve" else "action_enforcement",
                "hitl_required": len(hitl_triggers) > 0,
                "hitl_reasons": hitl_triggers,
                "confidence_level": "high" if act_decision["confidence"] > 0.85 else "medium" if act_decision["confidence"] > 0.70 else "low"
            }

            state["react_observe_result"] = json.dumps(observe_result)
            extracted_data["observe_result"] = observe_result

            # Full reasoning chain
            state["react_reasoning"] = f"""
            === REACT REASONING CHAIN ===

            THINK: {think_output[:500]}...

            ACT: Decision = {act_decision['decision']} (Confidence: {act_decision['confidence']:.2%})
                Reasoning: {act_decision['reasoning']}

            OBSERVE: Next Step = {observe_result['next_step']}
                    HITL Required = {observe_result['hitl_required']}
                    Confidence Level = {observe_result['confidence_level']}
            """

            # Determine if HITL should pause the workflow
            state["hitl_required"] = observe_result["hitl_required"]

            if state["hitl_required"]:
                # Calculate HITL priority
                hitl_priority = self._calculate_hitl_priority(state, hitl_triggers)
                state["hitl_priority"] = hitl_priority
                state["hitl_checkpoint"] = "post_react"
                state["hitl_waiting_since"] = datetime.now().isoformat()
                flags.append(f"hitl_priority_{hitl_priority}")
                recommendations.append(f"Human review required: {', '.join(hitl_triggers)}")

            # Determine decision type
            if state["hitl_required"] and HITL_CONFIG["checkpoints"].get("post_react", False):
                decision_type = DecisionType.AWAIT_HUMAN
                state["status"] = ContentStatus.PENDING_HUMAN_REVIEW.value
            else:
                # Map act_decision to DecisionType
                decision_map = {
                    "approve": DecisionType.APPROVE,
                    "warn": DecisionType.WARN,
                    "remove": DecisionType.REMOVE,
                    "flag": DecisionType.FLAG,
                    "suspend_user": DecisionType.SUSPEND_USER,
                    "ban_user": DecisionType.BAN_USER
                }
                decision_type = decision_map.get(act_decision["decision"], DecisionType.FLAG)
                state["status"] = ContentStatus.REPUTATION_SCORING.value

            confidence = act_decision["confidence"]

            decision = AgentDecision(
                agent_name="ReAct Decision Loop",
                decision=decision_type,
                confidence=confidence,
                reasoning=state["react_reasoning"],
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=state["hitl_required"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["current_agent"] = "ReAct Decision Loop"
            state["requires_human_review"] = state.get("requires_human_review", False) or state["hitl_required"]

        except Exception as e:
            logger.error(f"\nError in ReAct Decision Loop: {e}")
            import traceback
            traceback.print_exc()
            decision = AgentDecision(
                agent_name="ReAct Decision Loop",
                decision=DecisionType.NEEDS_REVIEW,
                confidence=0.0,
                reasoning=f"Error in ReAct synthesis: {str(e)}",
                flags=["processing_error"],
                recommendations=["Manual review required due to processing error"],
                extracted_data={},
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
            state["status"] = ContentStatus.PENDING_HUMAN_REVIEW.value
            state["hitl_required"] = True
            state["hitl_trigger_reasons"] = ["processing_error"]
            state["current_agent"] = "ReAct Decision Loop"

        return state

    def hitl_checkpoint_agent(self, state: ContentState) -> ContentState:
        """
        Human-in-the-Loop Checkpoint Agent - Manages HITL workflow interrupts.

        This agent:
        1. Evaluates if human review is needed at current checkpoint
        2. Prepares content for human review queue
        3. Handles human decisions when they come in
        4. Routes content back to workflow after human review

        Note: This agent PAUSES the workflow - it will resume when human provides input.
        """
        logger.info("\nHITL CHECKPOINT AGENT: Human-in-the-Loop Evaluation")
        logger.info("=" * 80)

        start_time = datetime.now()
        flags = []
        recommendations = []
        extracted_data = {}

        try:
            content_id = state.get("content_id")
            checkpoint = state.get("hitl_checkpoint", "unknown")
            hitl_reasons = state.get("hitl_trigger_reasons", [])

            # Check if human has already made a decision
            if state.get("hitl_human_decision"):
                return self._process_human_decision(state)

            # Prepare summary for human reviewer
            review_summary = self._prepare_hitl_summary(state)
            extracted_data["review_summary"] = review_summary

            # Calculate priority
            priority = self._calculate_hitl_priority(state, hitl_reasons)
            state["hitl_priority"] = priority

            # Generate review prompt for human
            review_prompt = f"""
            === CONTENT MODERATION REVIEW REQUEST ===

            Content ID: {content_id}
            Priority: {priority.upper()}
            Checkpoint: {checkpoint}

            === CONTENT ===
            "{state.get('content_text', '')[:1000]}"

            === AI ANALYSIS ===
            {review_summary}

            === RECOMMENDED ACTION ===
            AI Recommendation: {state.get('react_act_decision', 'No recommendation')}
            Confidence: {state.get('react_confidence', 0):.2%}

            === REASONS FOR HUMAN REVIEW ===
            {chr(10).join(f'• {reason}' for reason in hitl_reasons)}

            === YOUR OPTIONS ===
            1. APPROVE - Allow content (override AI if it recommended removal)
            2. WARN - Issue warning to user
            3. REMOVE - Remove content
            4. ESCALATE - Escalate to senior moderator
            5. SUSPEND_USER - Suspend user (if serious)

            Please provide your decision and reasoning.
            """

            state["hitl_review_prompt"] = review_prompt
            extracted_data["review_prompt"] = review_prompt

            # Set status to pending
            state["status"] = ContentStatus.PENDING_HUMAN_REVIEW.value
            state["hitl_waiting_since"] = datetime.now().isoformat()

            flags.append(f"awaiting_human_review_{priority}")
            recommendations.append(f"Content queued for {priority} priority human review")
            recommendations.append(f"Checkpoint: {checkpoint}")

            decision = AgentDecision(
                agent_name="HITL Checkpoint Agent",
                decision=DecisionType.AWAIT_HUMAN,
                confidence=1.0,  # Certain that human review is needed
                reasoning=f"Human review required at {checkpoint} checkpoint. Reasons: {', '.join(hitl_reasons)}",
                flags=flags,
                recommendations=recommendations,
                extracted_data=extracted_data,
                requires_human_review=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"\nError in HITL Checkpoint Agent: {e}")
            # On error, still require human review
            state["status"] = ContentStatus.PENDING_HUMAN_REVIEW.value
            state["hitl_required"] = True

        return state

    def _process_human_decision(self, state: ContentState) -> ContentState:
        """Process a human decision and route accordingly."""
        human_decision = state.get("hitl_human_decision", "").lower()
        human_notes = state.get("hitl_human_notes", "")

        # Map human decision to workflow action
        decision_map = {
            "approve": (DecisionType.HUMAN_APPROVED, ContentStatus.APPROVED),
            "warn": (DecisionType.WARN, ContentStatus.WARNED),
            "remove": (DecisionType.REMOVE, ContentStatus.REMOVED),
            "escalate": (DecisionType.HUMAN_ESCALATED, ContentStatus.ESCALATED),
            "suspend_user": (DecisionType.SUSPEND_USER, ContentStatus.ACTION_ENFORCEMENT),
            "ban_user": (DecisionType.BAN_USER, ContentStatus.ACTION_ENFORCEMENT)
        }

        decision_type, new_status = decision_map.get(
            human_decision,
            (DecisionType.NEEDS_REVIEW, ContentStatus.UNDER_REVIEW)
        )

        state["status"] = new_status.value
        state["hitl_resolution_timestamp"] = datetime.now().isoformat()

        decision = AgentDecision(
            agent_name="HITL Human Review",
            decision=decision_type,
            confidence=state.get("hitl_human_confidence_override", 1.0),
            reasoning=f"Human reviewer decision: {human_decision}. Notes: {human_notes}",
            flags=["human_reviewed"],
            recommendations=[f"Proceed with {human_decision} action"],
            extracted_data={
                "human_decision": human_decision,
                "human_notes": human_notes,
                "original_ai_recommendation": state.get("react_act_decision")
            },
            requires_human_review=False,
            processing_time=0.0
        )

        state["agent_decisions"] = state.get("agent_decisions", []) + [decision]
        state["current_agent"] = "HITL Human Review"

        return state

    def _analyze_agent_decisions(self, decisions: List[AgentDecision]) -> Dict[str, Any]:
        """Analyze all agent decisions for the THINK phase."""
        if not decisions:
            return {
                "total_agents": 0,
                "avg_confidence": 0.0,
                "consensus_level": "none",
                "decision_distribution": {},
                "flags_summary": [],
                "recommendations_summary": []
            }

        # Count decisions by type
        decision_counts = {}
        confidences = []
        all_flags = []
        all_recommendations = []

        for dec in decisions:
            decision_value = dec.decision.value if hasattr(dec.decision, 'value') else str(dec.decision)
            decision_counts[decision_value] = decision_counts.get(decision_value, 0) + 1
            confidences.append(dec.confidence)
            all_flags.extend(dec.flags)
            all_recommendations.extend(dec.recommendations)

        # Calculate consensus
        total = len(decisions)
        max_count = max(decision_counts.values()) if decision_counts else 0
        consensus_ratio = max_count / total if total > 0 else 0

        if consensus_ratio >= REACT_CONFIG["strong_consensus_threshold"]:
            consensus_level = "strong"
        elif consensus_ratio >= REACT_CONFIG["consensus_threshold"]:
            consensus_level = "moderate"
        else:
            consensus_level = "weak"

        return {
            "total_agents": total,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "consensus_level": consensus_level,
            "consensus_ratio": consensus_ratio,
            "decision_distribution": decision_counts,
            "flags_summary": list(set(all_flags)),
            "recommendations_summary": list(set(all_recommendations)),
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0
        }

    def _evaluate_hitl_triggers(self, state: ContentState, think_analysis: Dict) -> List[str]:
        """Evaluate if any HITL triggers are met."""
        triggers = []

        # Low confidence trigger
        if think_analysis["avg_confidence"] < HITL_CONFIG["confidence_threshold"]:
            triggers.append(HITLTriggerReason.LOW_CONFIDENCE.value)

        # Conflicting decisions trigger
        if think_analysis["consensus_level"] == "weak":
            triggers.append(HITLTriggerReason.CONFLICTING_DECISIONS.value)

        # High severity trigger
        severity = state.get("violation_severity", "none")
        if severity in HITL_CONFIG["always_review_severities"]:
            triggers.append(HITLTriggerReason.HIGH_SEVERITY.value)

        # High profile user trigger
        user_profile = state.get("user_profile")
        if user_profile:
            if user_profile.verified:
                triggers.append(HITLTriggerReason.HIGH_PROFILE_USER.value)
            if user_profile.follower_count >= HITL_CONFIG["high_profile_follower_threshold"]:
                triggers.append(HITLTriggerReason.HIGH_PROFILE_USER.value)

        # Sensitive content trigger
        if state.get("contains_sensitive_content", False):
            triggers.append(HITLTriggerReason.SENSITIVE_CONTENT.value)

        # First offense but severe trigger
        if user_profile and user_profile.total_violations == 0:
            toxicity = state.get("toxicity_score", 0)
            if toxicity > 0.7 or severity in ["high", "critical"]:
                triggers.append(HITLTriggerReason.FIRST_OFFENSE_SEVERE.value)

        # Check for potential false positive
        if self._check_potential_false_positive(state, think_analysis):
            triggers.append(HITLTriggerReason.POTENTIAL_FALSE_POSITIVE.value)

        return list(set(triggers))  # Remove duplicates

    def _check_potential_false_positive(self, state: ContentState, think_analysis: Dict) -> bool:
        """Check if this might be a false positive."""
        # Good user with sudden violation
        user_profile = state.get("user_profile")
        if user_profile:
            if user_profile.reputation_score > 0.8 and user_profile.total_violations == 0:
                if state.get("toxicity_score", 0) > 0.5:
                    return True

        # Conflicting signals
        if think_analysis["consensus_level"] == "weak":
            return True

        # Satire/quote indicators
        content = state.get("content_text", "").lower()
        satire_indicators = ["satire", "sarcasm", "/s", "jk", "kidding", "quoting", "quote:"]
        if any(ind in content for ind in satire_indicators):
            return True

        return False

    def _synthesize_final_decision(self, decisions: List[AgentDecision], analysis: Dict, state: ContentState) -> Dict:
        """Synthesize a final decision from all agent decisions."""
        # Get weighted scores for each decision type
        decision_scores = {}
        weights = REACT_CONFIG["agent_weights"]
        priorities = REACT_CONFIG["decision_priority"]

        for dec in decisions:
            agent_name = dec.agent_name
            decision_value = dec.decision.value if hasattr(dec.decision, 'value') else str(dec.decision)
            weight = weights.get(agent_name, 0.33)
            confidence = dec.confidence

            if decision_value not in decision_scores:
                decision_scores[decision_value] = 0

            # Score = weight * confidence * priority
            priority = priorities.get(decision_value, 1)
            decision_scores[decision_value] += weight * confidence * priority

        # Get the winning decision
        if decision_scores:
            winning_decision = max(decision_scores.items(), key=lambda x: x[1])
            final_decision = winning_decision[0]
            final_confidence = min(winning_decision[1] / max(priorities.values()), 1.0)
        else:
            final_decision = "flag"
            final_confidence = 0.5

        # Generate reasoning
        reasoning = f"Synthesized from {len(decisions)} agent decisions. "
        reasoning += f"Consensus: {analysis['consensus_level']}. "
        reasoning += f"Decision scores: {decision_scores}. "
        reasoning += f"Selected '{final_decision}' with adjusted confidence {final_confidence:.2%}."

        return {
            "decision": final_decision,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "decision_scores": decision_scores
        }

    def _calculate_hitl_priority(self, state: ContentState, triggers: List[str]) -> str:
        """Calculate HITL review priority."""
        priority_weights = HITL_CONFIG["priority_weights"]

        # Start with base priority
        score = 0

        # Add weight for each trigger
        trigger_weights = {
            HITLTriggerReason.HIGH_SEVERITY.value: 80,
            HITLTriggerReason.LEGAL_CONCERN.value: 100,
            HITLTriggerReason.HIGH_PROFILE_USER.value: 60,
            HITLTriggerReason.FIRST_OFFENSE_SEVERE.value: 70,
            HITLTriggerReason.SENSITIVE_CONTENT.value: 50,
            HITLTriggerReason.CONFLICTING_DECISIONS.value: 40,
            HITLTriggerReason.LOW_CONFIDENCE.value: 30,
            HITLTriggerReason.POTENTIAL_FALSE_POSITIVE.value: 45,
            HITLTriggerReason.EDGE_CASE.value: 35
        }

        for trigger in triggers:
            score += trigger_weights.get(trigger, 25)

        # Determine priority level
        if score >= priority_weights["critical"]:
            return "critical"
        elif score >= priority_weights["high"]:
            return "high"
        elif score >= priority_weights["medium"]:
            return "medium"
        else:
            return "low"

    def _prepare_hitl_summary(self, state: ContentState) -> str:
        """Prepare a summary for human reviewers."""
        summary = []

        # Content info
        summary.append(f"Content Type: {state.get('content_type', 'unknown')}")
        summary.append(f"Platform: {state.get('content_metadata', {}).platform if state.get('content_metadata') else 'unknown'}")

        # Analysis results
        summary.append(f"\nToxicity Score: {state.get('toxicity_score', 0):.2%}")
        summary.append(f"Toxicity Level: {state.get('toxicity_level', 'unknown')}")
        summary.append(f"Violations: {', '.join(state.get('policy_violations', [])) or 'None'}")
        summary.append(f"Severity: {state.get('violation_severity', 'none')}")

        # User info
        user = state.get("user_profile")
        if user:
            summary.append(f"\nUser: {user.username}")
            summary.append(f"Reputation: {user.reputation_score:.2%}")
            summary.append(f"Account Age: {user.account_age_days} days")
            summary.append(f"Previous Violations: {user.total_violations}")
            summary.append(f"Verified: {'Yes' if user.verified else 'No'}")

        # Agent decisions
        summary.append("\nAgent Decisions:")
        for dec in state.get("agent_decisions", []):
            dec_value = dec.decision.value if hasattr(dec.decision, 'value') else str(dec.decision)
            summary.append(f"  • {dec.agent_name}: {dec_value} ({dec.confidence:.0%})")

        return "\n".join(summary)

    def _format_detailed_decisions(self, decisions: List[AgentDecision]) -> str:
        """Format detailed agent decisions for LLM context."""
        formatted = []
        for i, dec in enumerate(decisions, 1):
            dec_value = dec.decision.value if hasattr(dec.decision, 'value') else str(dec.decision)
            formatted.append(f"""
            Agent {i}: {dec.agent_name}
            Decision: {dec_value}
            Confidence: {dec.confidence:.2%}
            Flags: {', '.join(dec.flags) if dec.flags else 'None'}
            Key Reasoning: {dec.reasoning[:200]}...
            """)
        return "\n".join(formatted)

    # Helper methods
    def _format_previous_decisions(self, state: ContentState) -> str:
        """Format previous agent decisions for LLM context."""
        decisions = state.get("agent_decisions", [])
        if not decisions:
            return "No previous decisions"

        formatted = []
        for dec in decisions:
            formatted.append(
                f"- {dec.agent_name}: {dec.decision.value} "
                f"(confidence: {dec.confidence:.2%}, flags: {len(dec.flags)})"
            )
        return "\n".join(formatted)

    def _determine_final_action(self, state: ContentState) -> str:
        """Determine final action based on all agent decisions."""
        decisions = state.get("agent_decisions", [])

        # Priority order: BAN > SUSPEND > REMOVE > WARN > FLAG > APPROVE
        for decision in reversed(decisions):
            if decision.decision == DecisionType.BAN_USER:
                return DecisionType.BAN_USER.value

        for decision in reversed(decisions):
            if decision.decision == DecisionType.SUSPEND_USER:
                return DecisionType.SUSPEND_USER.value

        for decision in reversed(decisions):
            if decision.decision == DecisionType.REMOVE:
                return DecisionType.REMOVE.value

        for decision in reversed(decisions):
            if decision.decision == DecisionType.WARN:
                return DecisionType.WARN.value

        return DecisionType.APPROVE.value

    def _calculate_suspension_duration(self, state: ContentState) -> int:
        """Calculate suspension duration based on violation severity and history."""
        user_profile = state.get("user_profile")
        severity = state.get("violation_severity", "low")

        base_days = {
            "low": 1,
            "medium": 3,
            "high": 7,
            "critical": 30
        }.get(severity, 1)

        # Increase for repeat offenders
        if user_profile.previous_suspensions > 0:
            base_days *= (user_profile.previous_suspensions + 1)

        return min(base_days, 90)  # Max 90 days

    def fast_mode_agent(self, state: ContentState) -> ContentState:
        """
        Fast Mode Agent - Simplified single-LLM pipeline for short comments.

        This agent provides rapid content moderation for short content (typically comments)
        using a single LLM call instead of the full multi-agent workflow.

        Responsibilities:
        - Analyze content toxicity in one pass
        - Check for policy violations
        - Make immediate approve/flag/remove decision
        - Significantly faster than full pipeline (~1-2s vs 6-12s)

        Use cases:
        - Short comments (<200 characters)
        - Low-risk content types
        - High-volume environments requiring fast response times

        Returns updated state with:
        - Final decision (approved/flagged/removed)
        - Toxicity score
        - Policy violations (if any)
        - Action reason
        """
        logger.info("\n🚀 FAST MODE AGENT - Single-Pass Moderation")
        logger.info("=" * 80)

        start_time = datetime.now()
        content_text = state.get("content_text", "")
        content_type = state.get("content_type", "unknown")
        user_profile = state.get("user_profile")

        logger.info(f"Content Type: {content_type}")
        logger.info(f"Content Length: {len(content_text)} chars")
        logger.info(f"User Reputation: {user_profile.reputation_score if user_profile else 'N/A'}")

        try:
            # Single comprehensive LLM call for fast moderation
            fast_mode_prompt = f"""
You are a content moderation AI. Analyze this content and make a quick moderation decision.

**Content to moderate:**
"{content_text}"

**Content Type:** {content_type}
**User Reputation Score:** {user_profile.reputation_score if user_profile else 0.5}
**User Previous Violations:** {user_profile.total_violations if user_profile else 0}

**Task:** Provide a rapid assessment with the following:

1. **Toxicity Score** (0.0 to 1.0):
   - 0.0-0.2: Clean content
   - 0.2-0.4: Mild toxicity
   - 0.4-0.6: Moderate toxicity
   - 0.6-0.8: High toxicity
   - 0.8-1.0: Severe toxicity

2. **Policy Violations** (list any):
   - hate_speech
   - harassment
   - bullying
   - profanity
   - spam
   - sexual_content
   - violence
   - misinformation
   - none

3. **Decision** (choose one):
   - approve: Safe content, no violations
   - flag: Minor issues, needs human review
   - warn: Borderline violation, warn user
   - remove: Clear policy violation, remove content

4. **Reason** (brief explanation in 1-2 sentences)

**Response format:**
{{
    "toxicity_score": <float>,
    "policy_violations": [<list of violations or empty>],
    "decision": "<approve|flag|warn|remove>",
    "reason": "<explanation>",
    "confidence": <float 0.0-1.0>
}}

Provide ONLY the JSON response, no additional text.
"""

            # Invoke LLM
            response = self.llm_flash.invoke(fast_mode_prompt)
            response_text = response.content.strip()

            # Parse JSON response
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            # Extract results
            toxicity_score = float(result.get("toxicity_score", 0.0))
            policy_violations = result.get("policy_violations", [])
            decision = result.get("decision", "approve")
            reason = result.get("reason", "Fast mode moderation completed")
            confidence = float(result.get("confidence", 0.75))

            # Update state
            state["toxicity_score"] = toxicity_score
            state["policy_violations"] = policy_violations
            state["action_reason"] = reason

            # Determine toxicity level
            if toxicity_score >= 0.8:
                state["toxicity_level"] = ToxicityLevel.SEVERE.value
            elif toxicity_score >= 0.6:
                state["toxicity_level"] = ToxicityLevel.HIGH.value
            elif toxicity_score >= 0.4:
                state["toxicity_level"] = ToxicityLevel.MEDIUM.value
            elif toxicity_score >= 0.2:
                state["toxicity_level"] = ToxicityLevel.LOW.value
            else:
                state["toxicity_level"] = ToxicityLevel.NONE.value

            # Map decision to action
            if decision == "remove":
                state["status"] = ContentStatus.REMOVED.value
                state["content_removed"] = True
                state["user_notified"] = True
                state["moderation_action"] = "removed"
            elif decision == "warn":
                state["status"] = ContentStatus.WARNED.value
                state["content_removed"] = False
                state["user_notified"] = True
                state["moderation_action"] = "warned"
            elif decision == "flag":
                state["status"] = ContentStatus.FLAGGED.value
                state["requires_human_review"] = True
                state["content_removed"] = False
                state["moderation_action"] = "flagged"
            else:  # approve
                state["status"] = ContentStatus.APPROVED.value
                state["content_removed"] = False
                state["user_notified"] = False
                state["moderation_action"] = "approved"

            # Create agent decision record
            processing_time = (datetime.now() - start_time).total_seconds()
            agent_decision = AgentDecision(
                agent_name="Fast Mode Agent",
                decision=DecisionType.APPROVE if decision == "approve" else DecisionType.REMOVE if decision == "remove" else DecisionType.WARN if decision == "warn" else DecisionType.FLAG,
                confidence=confidence,
                reasoning=reason,
                flags=policy_violations,
                recommendations=[f"Fast mode decision: {decision}"],
                extracted_data={
                    "toxicity_score": toxicity_score,
                    "policy_violations": policy_violations,
                    "processing_time_seconds": processing_time,
                    "fast_mode": True
                },
                processing_time=processing_time
            )

            # Add to agent decisions
            if "agent_decisions" not in state or state["agent_decisions"] is None:
                state["agent_decisions"] = []
            state["agent_decisions"].append(agent_decision)

            # Log results
            logger.info(f"\n✅ Fast Mode Decision: {decision.upper()}")
            logger.info(f"   Toxicity Score: {toxicity_score:.2f}")
            logger.info(f"   Confidence: {confidence:.2%}")
            logger.info(f"   Policy Violations: {', '.join(policy_violations) if policy_violations else 'None'}")
            logger.info(f"   Processing Time: {processing_time:.2f}s")
            logger.info(f"   Status: {state['status']}")

            # Store in memory for learning
            try:
                self.memory_manager.store_moderation_decision(
                    content_id=state.get("content_id", "unknown"),
                    content_text=content_text,
                    user_id=state.get("user_id", "unknown"),
                    action=decision,
                    violations=policy_violations,
                    toxicity_score=toxicity_score,
                    agent_decisions=[agent_decision],
                    primary_agent="Fast Mode Agent",
                    decision_context=f"Fast mode: {content_type}",
                    confidence=confidence
                )
            except Exception as mem_error:
                logger.warning(f"Failed to store in memory: {mem_error}")

        except json.JSONDecodeError as json_err:
            logger.error(f"❌ Failed to parse LLM response: {json_err}")
            logger.error(f"Response text: {response_text[:500]}")
            # Fallback to safe default
            state["status"] = ContentStatus.FLAGGED.value
            state["requires_human_review"] = True
            state["action_reason"] = "Fast mode parsing error - flagged for manual review"
            state["toxicity_score"] = 0.5

        except Exception as e:
            logger.error(f"❌ Fast Mode Agent Error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to safe default
            state["status"] = ContentStatus.FLAGGED.value
            state["requires_human_review"] = True
            state["action_reason"] = "Fast mode error - flagged for manual review"

        return state
