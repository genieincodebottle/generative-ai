"""The read-only gate. No API key, no database, no network.

`GraphCypherQAChain` runs with `allow_dangerous_requests=True`, which means
whatever Cypher the model writes gets executed. For a question-answering app
the only legitimate output is a read.
"""

import pytest

from services.cypher_guard import (
    UnsafeCypherError,
    assert_read_only,
    is_read_only,
    normalise,
)


class TestAllowsReads:
    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (m:Movie) RETURN m.title LIMIT 10",
            "match (p:Person)-[:ACTED_IN]->(m) return p.name, m.title",
            "MATCH (m:Movie) WITH m ORDER BY m.released DESC RETURN m LIMIT 5",
            "UNWIND [1,2,3] AS n RETURN n",
            "MATCH (m:Movie) RETURN count(m) AS total",
            # OFFSET contains SET; word boundaries must not trip on it.
            "MATCH (m:Movie) RETURN m SKIP 10 LIMIT 5",
        ],
    )
    def test_read_queries_pass(self, cypher):
        assert_read_only(cypher)
        assert is_read_only(cypher)


class TestRejectsWrites:
    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (m:Movie {title: 'X'}) RETURN m",
            "MATCH (m:Movie) DELETE m",
            "MATCH (m:Movie) DETACH DELETE m",
            "MERGE (p:Person {name: 'X'}) RETURN p",
            "MATCH (m:Movie) SET m.title = 'X' RETURN m",
            "MATCH (m:Movie) REMOVE m.title RETURN m",
            "DROP INDEX movie_title",
            "LOAD CSV FROM 'file:///x.csv' AS row CREATE (:X)",
            "MATCH (m) FOREACH (x IN [1] | SET m.a = 1)",
        ],
    )
    def test_write_queries_are_refused(self, cypher):
        with pytest.raises(UnsafeCypherError):
            assert_read_only(cypher)
        assert not is_read_only(cypher)

    def test_a_write_hidden_after_a_read_is_still_caught(self):
        with pytest.raises(UnsafeCypherError):
            assert_read_only("MATCH (m:Movie) RETURN m; MATCH (n) DETACH DELETE n")


class TestStringLiterals:
    """A film called "Set It Off" must not trip the guard.

    Keyword scanning without stripping string literals produces false
    positives on perfectly ordinary data, which is worse than useless: it
    refuses correct queries.
    """

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (m:Movie {title: 'Set It Off'}) RETURN m",
            'MATCH (m:Movie) WHERE m.title = "The Merge" RETURN m',
            "MATCH (p:Person {name: 'Drop Doe'}) RETURN p",
            "MATCH (m) WHERE m.tagline = 'Create your destiny' RETURN m",
        ],
    )
    def test_write_words_inside_strings_are_allowed(self, cypher):
        assert_read_only(cypher)

    def test_normalise_blanks_string_contents(self):
        assert "SET IT OFF" not in normalise("MATCH (m {t:'Set It Off'}) RETURN m")


class TestComments:
    def test_a_write_in_a_comment_does_not_trip_the_guard(self):
        assert_read_only("// do not CREATE anything\nMATCH (m) RETURN m")

    def test_a_write_cannot_hide_behind_a_comment(self):
        with pytest.raises(UnsafeCypherError):
            assert_read_only("MATCH (m) // read\nDETACH DELETE m")


class TestDegenerateInput:
    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_is_refused(self, bad):
        with pytest.raises(UnsafeCypherError, match="did not return"):
            assert_read_only(bad)

    def test_prose_is_refused(self):
        with pytest.raises(UnsafeCypherError, match="does not look like a read"):
            assert_read_only("I am not able to answer that question.")
