"""
Observability system for multi-agent content moderation.

This module implements:
1. OpenTelemetry integration for distributed tracing
2. Structured logging with context
3. Performance monitoring and metrics
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import functools
import logging
import sys
from contextlib import contextmanager

# Note: OpenTelemetry will be optional dependency
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create mock classes for when OpenTelemetry is not installed
    class MockSpan:
        def set_attribute(self, key, value): pass
        def set_status(self, status): pass
        def add_event(self, name, attributes=None): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class MockTracer:
        def start_as_current_span(self, name, attributes=None):
            return MockSpan()


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class StructuredLog:
    """Structured log entry with context."""
    timestamp: datetime
    level: LogLevel
    message: str
    agent_name: Optional[str] = None
    content_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "agent_name": self.agent_name,
            "content_id": self.content_id,
            "user_id": self.user_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata
        }

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


class StructuredLogger:
    """
    Structured logging system with context awareness.

    Provides JSON-formatted logs with trace context.
    """

    def __init__(
        self,
        name: str = "content_moderation",
        min_level: LogLevel = LogLevel.INFO,
        output_file: Optional[str] = None
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            min_level: Minimum log level
            output_file: Optional file for log output
        """
        self.name = name
        self.min_level = min_level
        self.output_file = output_file
        self.logs: List[StructuredLog] = []

        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, min_level.value))

        # Add handlers
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "data": %(data)s}'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(
        self,
        level: LogLevel,
        message: str,
        agent_name: Optional[str] = None,
        content_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata
    ):
        """
        Log a structured message.

        Args:
            level: Log level
            message: Log message
            agent_name: Optional agent name
            content_id: Optional content ID
            user_id: Optional user ID
            **metadata: Additional metadata
        """
        # Check level
        if self._should_log(level):
            log_entry = StructuredLog(
                timestamp=datetime.now(),
                level=level,
                message=message,
                agent_name=agent_name,
                content_id=content_id,
                user_id=user_id,
                metadata=metadata
            )

            self.logs.append(log_entry)

            # Log to Python logger
            log_dict = log_entry.to_dict()
            log_func = getattr(self.logger, level.value.lower())
            log_func(message, extra={"data": json.dumps(metadata)})

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level."""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return levels.index(level) >= levels.index(self.min_level)

    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        agent_name: Optional[str] = None,
        content_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get filtered logs.

        Args:
            level: Filter by log level
            agent_name: Filter by agent name
            content_id: Filter by content ID
            limit: Maximum number of logs to return

        Returns:
            List of log dictionaries
        """
        filtered = self.logs

        if level:
            filtered = [log for log in filtered if log.level == level]
        if agent_name:
            filtered = [log for log in filtered if log.agent_name == agent_name]
        if content_id:
            filtered = [log for log in filtered if log.content_id == content_id]

        # Return most recent first
        filtered = sorted(filtered, key=lambda x: x.timestamp, reverse=True)
        return [log.to_dict() for log in filtered[:limit]]


class TelemetrySystem:
    """
    OpenTelemetry integration for distributed tracing.

    Provides:
    - Distributed tracing across agents
    - Span creation and management
    - Trace context propagation
    """

    def __init__(self, service_name: str = "content-moderation-system"):
        """
        Initialize telemetry system.

        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.enabled = OTEL_AVAILABLE

        if self.enabled:
            # Setup OpenTelemetry
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()

            # Add console exporter for development
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(span_processor)

            self.tracer = trace.get_tracer(service_name)
        else:
            self.tracer = MockTracer()

    @contextmanager
    def trace_agent(
        self,
        agent_name: str,
        content_id: Optional[str] = None,
        **attributes
    ):
        """
        Create a trace span for agent execution.

        Args:
            agent_name: Name of the agent
            content_id: Optional content ID
            **attributes: Additional span attributes

        Yields:
            Span object
        """
        span_attributes = {
            "agent.name": agent_name,
            "service.name": self.service_name,
            **attributes
        }

        if content_id:
            span_attributes["content.id"] = content_id

        with self.tracer.start_as_current_span(
            f"agent.{agent_name}",
            attributes=span_attributes
        ) as span:
            try:
                yield span
            except Exception as e:
                if hasattr(span, 'set_status'):
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """
        Create a trace span for a specific operation.

        Args:
            operation_name: Name of the operation
            **attributes: Span attributes

        Yields:
            Span object
        """
        with self.tracer.start_as_current_span(
            operation_name,
            attributes=attributes
        ) as span:
            try:
                yield span
            except Exception as e:
                if hasattr(span, 'set_status'):
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    def add_event(self, span, event_name: str, **attributes):
        """Add event to current span."""
        if hasattr(span, 'add_event'):
            span.add_event(event_name, attributes=attributes)


class PerformanceMonitor:
    """
    Performance monitoring system.

    Tracks:
    - Agent execution times
    - Throughput metrics
    - Resource utilization
    - Custom metrics
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: List[PerformanceMetric] = []
        self.timers: Dict[str, float] = {}

    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        **tags
    ):
        """
        Record a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric
            **tags: Metric tags
        """
        metric = PerformanceMetric(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags
        )
        self.metrics.append(metric)

    def start_timer(self, timer_name: str):
        """Start a named timer."""
        self.timers[timer_name] = time.time()

    def end_timer(self, timer_name: str, **tags) -> float:
        """
        End a named timer and record metric.

        Args:
            timer_name: Name of the timer
            **tags: Metric tags

        Returns:
            Elapsed time in milliseconds
        """
        if timer_name not in self.timers:
            return 0.0

        start_time = self.timers[timer_name]
        elapsed_ms = (time.time() - start_time) * 1000

        self.record_metric(
            metric_name=f"{timer_name}.duration_ms",
            value=elapsed_ms,
            metric_type=MetricType.TIMER,
            **tags
        )

        del self.timers[timer_name]
        return elapsed_ms

    @contextmanager
    def measure(self, operation_name: str, **tags):
        """
        Context manager for measuring operation duration.

        Args:
            operation_name: Name of the operation
            **tags: Metric tags

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self.record_metric(
                metric_name=f"{operation_name}.duration_ms",
                value=elapsed_ms,
                metric_type=MetricType.TIMER,
                **tags
            )

    def get_metrics_summary(
        self,
        metric_name: Optional[str] = None,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get metrics summary.

        Args:
            metric_name: Optional metric name filter
            time_window_minutes: Optional time window in minutes

        Returns:
            Dictionary with aggregated metrics
        """
        filtered_metrics = self.metrics

        # Filter by metric name
        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.metric_name == metric_name]

        # Filter by time window
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            filtered_metrics = [m for m in filtered_metrics if m.timestamp > cutoff_time]

        if not filtered_metrics:
            return {"count": 0}

        # Aggregate by metric name
        aggregated = {}
        for metric in filtered_metrics:
            name = metric.metric_name
            if name not in aggregated:
                aggregated[name] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "values": []
                }

            agg = aggregated[name]
            agg["count"] += 1
            agg["sum"] += metric.value
            agg["min"] = min(agg["min"], metric.value)
            agg["max"] = max(agg["max"], metric.value)
            agg["values"].append(metric.value)

        # Calculate averages and percentiles
        for name, agg in aggregated.items():
            agg["avg"] = agg["sum"] / agg["count"]

            # Calculate percentiles
            sorted_values = sorted(agg["values"])
            agg["p50"] = sorted_values[len(sorted_values) // 2]
            agg["p95"] = sorted_values[int(len(sorted_values) * 0.95)]
            agg["p99"] = sorted_values[int(len(sorted_values) * 0.99)]

            # Remove values list from output
            del agg["values"]

        return aggregated


class ObservabilityManager:
    """
    Unified observability manager.

    Combines:
    - Structured logging
    - Distributed tracing
    - Performance monitoring
    """

    def __init__(
        self,
        service_name: str = "content-moderation-system",
        log_level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None
    ):
        """
        Initialize observability manager.

        Args:
            service_name: Name of the service
            log_level: Minimum log level
            log_file: Optional log file path
        """
        self.logger = StructuredLogger(
            name=service_name,
            min_level=log_level,
            output_file=log_file
        )
        self.telemetry = TelemetrySystem(service_name=service_name)
        self.performance = PerformanceMonitor()

    @contextmanager
    def observe_agent(
        self,
        agent_name: str,
        content_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata
    ):
        """
        Comprehensive observation of agent execution.

        Args:
            agent_name: Name of the agent
            content_id: Optional content ID
            user_id: Optional user ID
            **metadata: Additional metadata

        Yields:
            Dictionary with logger, span, and performance context
        """
        # Start logging
        self.logger.info(
            f"Starting agent: {agent_name}",
            agent_name=agent_name,
            content_id=content_id,
            user_id=user_id,
            **metadata
        )

        # Start tracing and performance monitoring
        with self.telemetry.trace_agent(agent_name, content_id, **metadata) as span:
            with self.performance.measure(f"agent.{agent_name}", agent=agent_name):
                try:
                    yield {
                        "logger": self.logger,
                        "span": span,
                        "performance": self.performance
                    }

                    # Log success
                    self.logger.info(
                        f"Completed agent: {agent_name}",
                        agent_name=agent_name,
                        content_id=content_id,
                        status="success"
                    )

                except Exception as e:
                    # Log error
                    self.logger.error(
                        f"Agent failed: {agent_name}",
                        agent_name=agent_name,
                        content_id=content_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    raise

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get observability dashboard data.

        Returns:
            Dictionary with logs, metrics, and traces summary
        """
        return {
            "recent_logs": self.logger.get_logs(limit=50),
            "error_logs": self.logger.get_logs(level=LogLevel.ERROR, limit=20),
            "performance_metrics": self.performance.get_metrics_summary(),
            "telemetry_enabled": self.telemetry.enabled,
            "timestamp": datetime.now().isoformat()
        }


# Decorator for automatic tracing
def traced(operation_name: Optional[str] = None):
    """
    Decorator to automatically trace function execution.

    Args:
        operation_name: Optional operation name (defaults to function name)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get observability manager from args if available
            obs_manager = None
            for arg in args:
                if isinstance(arg, ObservabilityManager):
                    obs_manager = arg
                    break

            name = operation_name or func.__name__

            if obs_manager:
                with obs_manager.telemetry.trace_operation(name):
                    with obs_manager.performance.measure(name):
                        return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator
