"""
Advanced tool management system for multi-agent content moderation.

This module implements:
1. Dynamic tool selection - Agents choose tools based on context
2. Tool sandboxing - Safe execution with error handling
3. Rate limiting - Prevent excessive tool usage
"""

from typing import Dict, List, Any, Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI
from ..utils.tools import (
    analyze_text_sentiment,
    detect_toxicity,
    check_policy_violations,
    calculate_user_reputation,
    check_spam_indicators,
    detect_hate_speech_patterns
)


class ToolCategory(Enum):
    """Categories of moderation tools."""
    TOXICITY_ANALYSIS = "toxicity_analysis"
    POLICY_CHECK = "policy_check"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SPAM_DETECTION = "spam_detection"
    REPUTATION_ANALYSIS = "reputation_analysis"
    HATE_SPEECH_DETECTION = "hate_speech_detection"


@dataclass
class ToolMetadata:
    """Metadata about a moderation tool."""
    name: str
    category: ToolCategory
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    cost_estimate: float  # Estimated cost in ms or API calls
    requires_api: bool = False
    safe_to_retry: bool = True


@dataclass
class ToolExecutionResult:
    """Result from executing a tool."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for tools."""
    max_calls_per_minute: int
    max_calls_per_hour: int
    max_concurrent_calls: int = 5
    cooldown_seconds: int = 1


class RateLimiter:
    """
    Rate limiter for tool calls.

    Prevents excessive API usage and protects against abuse.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.call_history: Dict[str, List[datetime]] = defaultdict(list)
        self.active_calls: Dict[str, int] = defaultdict(int)
        self.last_call_time: Dict[str, datetime] = {}

    def check_rate_limit(self, tool_name: str) -> bool:
        """
        Check if tool call is allowed under rate limits.

        Args:
            tool_name: Name of the tool

        Returns:
            True if allowed, False if rate limited
        """
        now = datetime.now()

        # Clean old history
        self._clean_history(tool_name, now)

        # Check concurrent calls
        if self.active_calls[tool_name] >= self.config.max_concurrent_calls:
            return False

        # Check calls per minute
        minute_ago = now - timedelta(minutes=1)
        recent_calls = [t for t in self.call_history[tool_name] if t > minute_ago]
        if len(recent_calls) >= self.config.max_calls_per_minute:
            return False

        # Check calls per hour
        hour_ago = now - timedelta(hours=1)
        hourly_calls = [t for t in self.call_history[tool_name] if t > hour_ago]
        if len(hourly_calls) >= self.config.max_calls_per_hour:
            return False

        # Check cooldown
        if tool_name in self.last_call_time:
            time_since_last = (now - self.last_call_time[tool_name]).total_seconds()
            if time_since_last < self.config.cooldown_seconds:
                return False

        return True

    def record_call(self, tool_name: str):
        """Record a tool call."""
        now = datetime.now()
        self.call_history[tool_name].append(now)
        self.active_calls[tool_name] += 1
        self.last_call_time[tool_name] = now

    def release_call(self, tool_name: str):
        """Release an active call."""
        if self.active_calls[tool_name] > 0:
            self.active_calls[tool_name] -= 1

    def _clean_history(self, tool_name: str, now: datetime):
        """Clean old call history."""
        hour_ago = now - timedelta(hours=1)
        self.call_history[tool_name] = [
            t for t in self.call_history[tool_name] if t > hour_ago
        ]

    def get_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get rate limit statistics for a tool."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        recent_calls = [t for t in self.call_history[tool_name] if t > minute_ago]
        hourly_calls = [t for t in self.call_history[tool_name] if t > hour_ago]

        return {
            "calls_last_minute": len(recent_calls),
            "calls_last_hour": len(hourly_calls),
            "active_calls": self.active_calls[tool_name],
            "limit_per_minute": self.config.max_calls_per_minute,
            "limit_per_hour": self.config.max_calls_per_hour,
            "remaining_minute": self.config.max_calls_per_minute - len(recent_calls),
            "remaining_hour": self.config.max_calls_per_hour - len(hourly_calls)
        }


class ToolSandbox:
    """
    Sandboxed tool execution environment.

    Provides:
    - Error handling and recovery
    - Execution timeouts
    - Retry logic
    - Logging and monitoring
    """

    def __init__(
        self,
        max_retries: int = 2,
        timeout_seconds: int = 30,
        enable_logging: bool = True
    ):
        """
        Initialize tool sandbox.

        Args:
            max_retries: Maximum retry attempts
            timeout_seconds: Execution timeout
            enable_logging: Enable execution logging
        """
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_logging = enable_logging
        self.execution_logs: List[Dict[str, Any]] = []

    def execute(
        self,
        tool_func: Callable,
        tool_name: str,
        *args,
        **kwargs
    ) -> ToolExecutionResult:
        """
        Execute a tool in a sandboxed environment.

        Args:
            tool_func: Tool function to execute
            tool_name: Name of the tool
            *args: Positional arguments for tool
            **kwargs: Keyword arguments for tool

        Returns:
            ToolExecutionResult
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            start_time = time.time()

            try:
                # Execute tool
                result = tool_func(*args, **kwargs)

                execution_time = (time.time() - start_time) * 1000

                # Log successful execution
                if self.enable_logging:
                    self._log_execution(
                        tool_name=tool_name,
                        success=True,
                        execution_time_ms=execution_time,
                        retries=retries
                    )

                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time,
                    retries=retries
                )

            except Exception as e:
                last_error = str(e)
                execution_time = (time.time() - start_time) * 1000

                if self.enable_logging:
                    self._log_execution(
                        tool_name=tool_name,
                        success=False,
                        execution_time_ms=execution_time,
                        retries=retries,
                        error=last_error
                    )

                retries += 1

                # Wait before retry
                if retries <= self.max_retries:
                    time.sleep(0.5 * retries)  # Exponential backoff

        # All retries failed
        return ToolExecutionResult(
            tool_name=tool_name,
            success=False,
            result=None,
            error=last_error,
            execution_time_ms=execution_time,
            retries=retries - 1
        )

    def _log_execution(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
        retries: int,
        error: Optional[str] = None
    ):
        """Log tool execution."""
        self.execution_logs.append({
            "tool_name": tool_name,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "retries": retries,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_logs(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get execution logs."""
        if tool_name:
            return [log for log in self.execution_logs if log["tool_name"] == tool_name]
        return self.execution_logs


class DynamicToolSelector:
    """
    Dynamically select appropriate tools based on content analysis.

    Uses LLM to intelligently choose which tools to apply.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize dynamic tool selector.

        Args:
            llm: Language model for tool selection
        """
        self.llm = llm
        self.tool_registry = self._build_tool_registry()

    def _build_tool_registry(self) -> Dict[str, ToolMetadata]:
        """Build registry of available tools."""
        return {
            "detect_toxicity": ToolMetadata(
                name="detect_toxicity",
                category=ToolCategory.TOXICITY_ANALYSIS,
                description="Detect toxic language, profanity, insults, and threats",
                input_schema={"text": "string"},
                output_schema={"toxicity_score": "float", "categories": "list"},
                cost_estimate=10.0,
                requires_api=False,
                safe_to_retry=True
            ),
            "check_policy_violations": ToolMetadata(
                name="check_policy_violations",
                category=ToolCategory.POLICY_CHECK,
                description="Check content against community policies",
                input_schema={"text": "string"},
                output_schema={"violations": "list", "severity": "string"},
                cost_estimate=15.0,
                requires_api=False,
                safe_to_retry=True
            ),
            "analyze_text_sentiment": ToolMetadata(
                name="analyze_text_sentiment",
                category=ToolCategory.SENTIMENT_ANALYSIS,
                description="Analyze sentiment and emotional tone",
                input_schema={"text": "string"},
                output_schema={"sentiment": "string", "score": "float"},
                cost_estimate=8.0,
                requires_api=False,
                safe_to_retry=True
            ),
            "check_spam_indicators": ToolMetadata(
                name="check_spam_indicators",
                category=ToolCategory.SPAM_DETECTION,
                description="Detect spam patterns and indicators",
                input_schema={"text": "string"},
                output_schema={"is_spam": "bool", "spam_score": "float"},
                cost_estimate=5.0,
                requires_api=False,
                safe_to_retry=True
            ),
            "detect_hate_speech_patterns": ToolMetadata(
                name="detect_hate_speech_patterns",
                category=ToolCategory.HATE_SPEECH_DETECTION,
                description="Detect hate speech and discriminatory language",
                input_schema={"text": "string"},
                output_schema={"hate_speech": "bool", "patterns": "list"},
                cost_estimate=12.0,
                requires_api=False,
                safe_to_retry=True
            ),
            "calculate_user_reputation": ToolMetadata(
                name="calculate_user_reputation",
                category=ToolCategory.REPUTATION_ANALYSIS,
                description="Calculate user reputation score and risk level",
                input_schema={"user_id": "string", "user_history": "dict"},
                output_schema={"reputation_score": "float", "risk_level": "string"},
                cost_estimate=10.0,
                requires_api=False,
                safe_to_retry=True
            )
        }

    def select_tools(
        self,
        content_text: str,
        content_type: str,
        user_context: Optional[Dict[str, Any]] = None,
        max_tools: int = 5
    ) -> List[str]:
        """
        Dynamically select appropriate tools for content analysis.

        Args:
            content_text: Content to analyze
            content_type: Type of content
            user_context: Optional user context
            max_tools: Maximum number of tools to select

        Returns:
            List of tool names to use
        """
        # Build tool descriptions
        tool_descriptions = []
        for name, metadata in self.tool_registry.items():
            tool_descriptions.append(
                f"- {name}: {metadata.description} (category: {metadata.category.value})"
            )

        tools_text = "\n".join(tool_descriptions)

        prompt = f"""You are a tool selection agent. Based on the content below, select the most appropriate moderation tools to use.

CONTENT TYPE: {content_type}
CONTENT: {content_text[:500]}

{f"USER CONTEXT: Reputation={user_context.get('reputation_score', 0.5)}, Violations={user_context.get('total_violations', 0)}" if user_context else ""}

AVAILABLE TOOLS:
{tools_text}

Select the most relevant tools (maximum {max_tools}) for analyzing this content. Consider:
1. What type of violations might this content have?
2. What tools would provide the most value?
3. What is the priority order?

Respond with a JSON list of tool names in priority order:
["tool1", "tool2", "tool3"]

Selected tools:"""

        try:
            response = self.llm.invoke(prompt)

            # Extract JSON from response
            import json
            start_idx = response.content.find('[')
            end_idx = response.content.rfind(']')

            if start_idx != -1 and end_idx != -1:
                json_str = response.content[start_idx:end_idx + 1]
                selected_tools = json.loads(json_str)

                # Validate tools exist
                valid_tools = [
                    tool for tool in selected_tools
                    if tool in self.tool_registry
                ]

                return valid_tools[:max_tools]

        except Exception:
            pass

        # Fallback to default tools
        return ["detect_toxicity", "check_policy_violations", "analyze_text_sentiment"]

    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool."""
        return self.tool_registry.get(tool_name)


class ToolManager:
    """
    Comprehensive tool management system.

    Combines:
    - Dynamic tool selection
    - Sandboxed execution
    - Rate limiting
    - Monitoring and logging
    """

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        rate_limit_config: Optional[RateLimitConfig] = None,
        sandbox_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tool manager.

        Args:
            llm: Language model for dynamic selection
            rate_limit_config: Rate limiting configuration
            sandbox_config: Sandbox configuration
        """
        self.selector = DynamicToolSelector(llm)
        self.sandbox = ToolSandbox(**(sandbox_config or {}))
        self.rate_limiter = RateLimiter(
            rate_limit_config or RateLimitConfig(
                max_calls_per_minute=60,
                max_calls_per_hour=1000,
                max_concurrent_calls=5
            )
        )

        # Map tool names to functions
        self.tool_functions = {
            "detect_toxicity": detect_toxicity,
            "check_policy_violations": check_policy_violations,
            "analyze_text_sentiment": analyze_text_sentiment,
            "check_spam_indicators": check_spam_indicators,
            "detect_hate_speech_patterns": detect_hate_speech_patterns,
            "calculate_user_reputation": calculate_user_reputation
        }

    def execute_with_selection(
        self,
        content_text: str,
        content_type: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dynamically select and execute tools.

        Args:
            content_text: Content to analyze
            content_type: Type of content
            user_context: Optional user context

        Returns:
            Dictionary with tool results
        """
        # Select tools
        selected_tools = self.selector.select_tools(
            content_text=content_text,
            content_type=content_type,
            user_context=user_context
        )

        results = {
            "selected_tools": selected_tools,
            "tool_results": {},
            "errors": [],
            "rate_limited": []
        }

        # Execute selected tools
        for tool_name in selected_tools:
            # Check rate limit
            if not self.rate_limiter.check_rate_limit(tool_name):
                results["rate_limited"].append(tool_name)
                continue

            # Record call
            self.rate_limiter.record_call(tool_name)

            try:
                # Get tool function
                tool_func = self.tool_functions.get(tool_name)
                if not tool_func:
                    results["errors"].append(f"Tool not found: {tool_name}")
                    continue

                # Execute in sandbox
                execution_result = self.sandbox.execute(
                    tool_func=tool_func,
                    tool_name=tool_name,
                    text=content_text
                )

                if execution_result.success:
                    results["tool_results"][tool_name] = execution_result.result
                else:
                    results["errors"].append(
                        f"{tool_name}: {execution_result.error}"
                    )

            finally:
                # Release call
                self.rate_limiter.release_call(tool_name)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "sandbox_logs": len(self.sandbox.execution_logs),
            "rate_limit_stats": {
                tool: self.rate_limiter.get_stats(tool)
                for tool in self.tool_functions.keys()
            }
        }
