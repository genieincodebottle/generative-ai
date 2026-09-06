"""Routing rule scoring. No API key, no network.

These pin two bugs that a live run exposed: a billing question was routed to
the code-review processor with zero keyword matches and zero pattern matches.
"""

import pytest

from services.apps.query_routing import ROUTING_THRESHOLD, QueryRouter


@pytest.fixture
def router():
    # Scoring never touches the model, so the LLM can be None here.
    return QueryRouter(llm=None)


class TestScoring:
    def test_a_rule_that_matches_nothing_is_not_scored(self, router):
        """The priority bonus used to be applied unconditionally.

        Every rule therefore got `priority * 0.1` even with no keyword and no
        pattern match, so the highest-priority rule won any query that matched
        nothing at all.
        """
        scores = router._score_routing_rules("zzzz qqqq wwww")
        for entry in scores:
            assert entry["keyword_matches"] > 0 or entry["pattern_matches"] > 0, (
                f"{entry['name']} scored {entry['score']} with no matches"
            )

    def test_a_matching_query_still_scores(self, router):
        scores = router._score_routing_rules(
            "write a python function to reverse a linked list"
        )
        assert scores, "a clearly technical query should match some rule"
        assert scores[0]["score"] > 0

    def test_scores_are_sorted_best_first(self, router):
        scores = router._score_routing_rules("debug this python code error")
        assert scores == sorted(scores, key=lambda s: s["score"], reverse=True)


class TestThreshold:
    def test_the_float_that_defeated_the_guard(self):
        """`3 * 0.1` is 0.30000000000000004, which is > 0.3.

        The unmatched-rule score was exactly `priority(3) * 0.1`, so the
        "fall back to a general processor" guard - written as `score > 0.3` -
        did not fire for the one case it existed to catch.
        """
        assert 3 * 0.1 > 0.3          # the trap
        assert not round(3 * 0.1, 6) > ROUTING_THRESHOLD   # the fix

    def test_scores_are_rounded_so_comparisons_behave(self, router):
        for entry in router._score_routing_rules("write python code to sort a list"):
            assert entry["score"] == round(entry["score"], 6)
