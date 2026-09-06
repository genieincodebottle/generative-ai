"""Classifier logic. No API key, no model download, no network.

The important tests here are about what happens when the classifier does NOT
work. A guardrail that fails open is worse than no guardrail, because it is
trusted.
"""

import pytest

from services.guard_service import (
    BackendError,
    ClassifierError,
    InvalidText,
    _verdict,
    classify,
    parse_score,
    validate_text,
)


class TestParseScore:
    """The original did `except ValueError: score = 0.0`.

    A score of 0.0 is below every threshold, so an unparseable response, a
    truncated response, or a changed output format all became a confident
    "this text is benign".
    """

    @pytest.mark.parametrize("raw, expected",
                             [("0.9", 0.9), (0.5, 0.5), ("0", 0.0),
                              ("1", 1.0), (" 0.42 ", 0.42)])
    def test_parses_valid_scores(self, raw, expected):
        assert parse_score(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "   ", "benign", "MALICIOUS",
                                     "not-a-number", "0.5.1", []])
    def test_unparseable_raises_instead_of_defaulting_to_safe(self, raw):
        with pytest.raises(ClassifierError):
            parse_score(raw)

    @pytest.mark.parametrize("raw", ["-0.1", "1.5", "42"])
    def test_out_of_range_is_refused(self, raw):
        with pytest.raises(ClassifierError, match="outside"):
            parse_score(raw)

    def test_the_error_says_why_it_will_not_guess(self):
        with pytest.raises(ClassifierError, match="Refusing to report this text as safe"):
            parse_score("garbage")


class TestValidateText:
    def test_accepts_normal_text(self):
        assert validate_text("  hello  ") == "hello"

    @pytest.mark.parametrize("bad", [None, "", "   ", "\n\t"])
    def test_rejects_empty(self, bad):
        with pytest.raises(InvalidText, match="empty"):
            validate_text(bad)

    def test_rejects_overlong(self, monkeypatch):
        monkeypatch.setattr("services.guard_service.MAX_TEXT_LENGTH", 10)
        with pytest.raises(InvalidText, match="too long"):
            validate_text("x" * 11)


class TestVerdict:
    def test_score_above_threshold_is_malicious(self):
        v = _verdict("x", "groq", "m", 0.8, 0.7, 12.0)
        assert v.is_malicious is True and v.label == "MALICIOUS"

    def test_score_equal_to_threshold_is_benign(self):
        # Strictly greater than, so the threshold itself is not an attack.
        v = _verdict("x", "groq", "m", 0.7, 0.7, 12.0)
        assert v.is_malicious is False and v.label == "BENIGN"

    def test_lowering_the_threshold_catches_more(self):
        assert _verdict("x", "groq", "m", 0.6, 0.7, 1.0).is_malicious is False
        assert _verdict("x", "groq", "m", 0.6, 0.5, 1.0).is_malicious is True


class TestBackendSelection:
    def test_unknown_backend_is_refused(self):
        with pytest.raises(BackendError, match="Unknown backend"):
            classify("hello there", "nope")

    def test_missing_key_names_the_variable_and_the_url(self, monkeypatch):
        monkeypatch.setattr("services.guard_service.api_key_for", lambda b: None)
        with pytest.raises(BackendError) as exc:
            classify("hello there", "groq")
        assert "GROQ_API_KEY" in str(exc.value)
        assert "console.groq.com" in str(exc.value)

    def test_unknown_model_size_is_refused(self, monkeypatch):
        monkeypatch.setattr("services.guard_service.api_key_for", lambda b: "key")
        with pytest.raises(BackendError, match="Unknown model size"):
            classify("hello there", "groq", size="999M")

    def test_validation_happens_before_any_backend_work(self):
        # An empty string must not reach a network call.
        with pytest.raises(InvalidText):
            classify("   ", "groq")
