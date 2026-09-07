"""Retry behaviour. No sleeping in tests: base_delay is set to zero."""

import pytest

from services.retry import RetryingEmbeddings, is_retryable, with_retry


class TestIsRetryable:
    @pytest.mark.parametrize(
        "message",
        [
            "Error embedding content: 500 INTERNAL.",
            "503 UNAVAILABLE",
            "429 RESOURCE_EXHAUSTED",
            "504 DEADLINE_EXCEEDED",
            "connection reset by peer",
        ],
    )
    def test_transient_failures_are_retryable(self, message):
        assert is_retryable(Exception(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "400 INVALID_ARGUMENT: content must not be empty",
            "401 UNAUTHENTICATED: API key not valid",
            "404 model not found",
        ],
    )
    def test_client_errors_are_not_retryable(self, message):
        # Retrying a malformed request only fails more slowly.
        assert is_retryable(Exception(message)) is False


class TestWithRetry:
    def test_returns_immediately_on_success(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        assert with_retry(fn, base_delay=0) == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("500 INTERNAL")
            return "ok"

        assert with_retry(fn, base_delay=0) == "ok"
        assert len(calls) == 3

    def test_gives_up_after_the_attempt_budget(self):
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError("500 INTERNAL")

        with pytest.raises(RuntimeError, match="500"):
            with_retry(fn, attempts=3, base_delay=0)
        assert len(calls) == 3

    def test_does_not_retry_a_client_error(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("400 INVALID_ARGUMENT")

        with pytest.raises(ValueError):
            with_retry(fn, attempts=4, base_delay=0)
        assert len(calls) == 1


class FlakyEmbeddings:
    """Fails the first ``fail_times`` calls, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.query_calls = 0
        self.document_calls = 0
        self.model = "stub-model"

    def embed_query(self, text):
        self.query_calls += 1
        if self.query_calls <= self.fail_times:
            raise RuntimeError("Error embedding content: 500 INTERNAL.")
        return [0.1, 0.2]

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.document_calls <= self.fail_times:
            raise RuntimeError("503 UNAVAILABLE")
        return [[0.1, 0.2] for _ in texts]


class TestRetryingEmbeddings:
    def test_query_survives_a_transient_500(self, monkeypatch):
        monkeypatch.setattr("services.retry.time.sleep", lambda _s: None)
        inner = FlakyEmbeddings(fail_times=2)
        assert RetryingEmbeddings(inner).embed_query("hello") == [0.1, 0.2]
        assert inner.query_calls == 3

    def test_documents_survive_a_transient_503(self, monkeypatch):
        monkeypatch.setattr("services.retry.time.sleep", lambda _s: None)
        inner = FlakyEmbeddings(fail_times=1)
        assert len(RetryingEmbeddings(inner).embed_documents(["a", "b"])) == 2

    def test_persistent_failure_still_raises(self, monkeypatch):
        monkeypatch.setattr("services.retry.time.sleep", lambda _s: None)
        with pytest.raises(RuntimeError):
            RetryingEmbeddings(FlakyEmbeddings(fail_times=99), attempts=3).embed_query("x")

    def test_unknown_attributes_pass_through_to_the_wrapped_object(self):
        # It must stay a drop-in replacement for the real Embeddings object.
        assert RetryingEmbeddings(FlakyEmbeddings(0)).model == "stub-model"
