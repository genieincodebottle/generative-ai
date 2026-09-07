"""Business-logic tests. No API key, no network, no Streamlit."""

import pytest

from services.sentiment_service import (
    InvalidCallText,
    normalise_result,
    validate_call_text,
)


class TestValidateCallText:
    def test_accepts_normal_text(self):
        text = "Customer was angry about the wrong order."
        assert validate_call_text(text) == text

    def test_strips_surrounding_whitespace(self):
        assert validate_call_text("  a valid call transcript  ") == "a valid call transcript"

    @pytest.mark.parametrize("bad", [None, "", "   ", "\n\t "])
    def test_rejects_empty(self, bad):
        with pytest.raises(InvalidCallText, match="empty"):
            validate_call_text(bad)

    def test_rejects_too_short(self):
        with pytest.raises(InvalidCallText, match="too short"):
            validate_call_text("hi")

    def test_rejects_too_long(self):
        with pytest.raises(InvalidCallText, match="too long"):
            validate_call_text("x" * 10_001)

    def test_length_is_measured_after_stripping(self):
        # Whitespace padding must not smuggle a two-character call past the
        # minimum-length check.
        with pytest.raises(InvalidCallText, match="too short"):
            validate_call_text("  hi  " + " " * 50)


class TestNormaliseResult:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ({"sentiment": "positive"}, "Positive"),
            ({"sentiment": "NEGATIVE"}, "Negative"),
            ({"sentiment": "  neutral  "}, "Neutral"),
            ({"sentiment": "Positive"}, "Positive"),
        ],
    )
    def test_normalises_sentiment_casing(self, raw, expected):
        assert normalise_result(raw)["sentiment"] == expected

    @pytest.mark.parametrize("raw", [{"sentiment": "furious"}, {"sentiment": ""}, {}])
    def test_unknown_sentiment_falls_back_to_neutral(self, raw):
        assert normalise_result(raw)["sentiment"] == "Neutral"

    @pytest.mark.parametrize(
        "value, expected",
        [(7, 7), ("8", 8), (3.7, 3), (0, 1), (-5, 1), (99, 10), ("nonsense", 1), (None, 1)],
    )
    def test_clamps_aggressiveness_into_range(self, value, expected):
        assert normalise_result({"aggressiveness": value})["aggressiveness"] == expected

    def test_missing_aggressiveness_defaults_to_one(self):
        assert normalise_result({"sentiment": "Positive"})["aggressiveness"] == 1
