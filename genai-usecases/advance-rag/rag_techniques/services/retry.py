"""Retry for embedding calls.

Google's embedding endpoint intermittently answers `500 INTERNAL` on the free
tier. It is transient - the identical request succeeds moments later - but
without a retry it surfaces as a hard failure mid-conversation, which reads
like a broken app rather than a blip.

Wrapping the Embeddings object rather than editing call sites means both the
indexing path and the retrieval path are covered by one change.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Transient server-side conditions. A 400 (bad request) is NOT here on
# purpose: retrying a malformed request just fails more slowly.
_RETRYABLE_MARKERS = (
    "500", "502", "503", "504",
    "INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED",
    "429", "RESOURCE_EXHAUSTED",
    "timeout", "Timeout", "connection",
)


def is_retryable(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def with_retry(fn: Callable[[], T], *, attempts: int = 4,
               base_delay: float = 1.0, what: str = "request") -> T:
    """Call ``fn``, retrying transient failures with exponential backoff.

    Jittered so that a burst of parallel calls hitting the same blip does not
    retry in lockstep.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            logger.warning(
                "Transient failure on %s (attempt %d/%d): %s - retrying in %.1fs",
                what, attempt + 1, attempts, exc, delay,
            )
            time.sleep(delay)
    raise last  # unreachable, kept for the type checker


class RetryingEmbeddings(Embeddings):
    """An Embeddings wrapper that retries transient provider failures."""

    def __init__(self, inner: Embeddings, attempts: int = 4) -> None:
        self._inner = inner
        self._attempts = attempts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return with_retry(
            lambda: self._inner.embed_documents(texts),
            attempts=self._attempts, what="embed_documents",
        )

    def embed_query(self, text: str) -> list[float]:
        return with_retry(
            lambda: self._inner.embed_query(text),
            attempts=self._attempts, what="embed_query",
        )

    def __getattr__(self, name):
        # Anything else (model name, client, ...) passes through to the wrapped
        # object, so this stays a drop-in replacement.
        return getattr(self._inner, name)
