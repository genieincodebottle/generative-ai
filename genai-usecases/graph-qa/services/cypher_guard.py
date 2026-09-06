"""Refuse Cypher that would change the graph.

`GraphCypherQAChain` is constructed with `allow_dangerous_requests=True`,
which is required for it to run at all. What that flag actually means is:
whatever Cypher the model writes gets executed against your database.

For a question-answering app the only legitimate output is a read. A prompt
asking for one is a request the model may decline; this is a gate the query
has to pass. It matters more here than in the original single-file version,
because the chain is now reachable over HTTP.
"""

from __future__ import annotations

import re

# Clauses that write, delete, or change the database.
WRITE_CLAUSES = (
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE",
    "DROP", "LOAD CSV", "FOREACH", "CALL DB.", "CALL DBMS.",
    "CALL APOC.CREATE", "CALL APOC.MERGE", "CALL APOC.PERIODIC",
)

# Strip string literals before scanning, so a film called "Set It Off" or a
# person named "Drop" cannot trip the guard.
_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


class UnsafeCypherError(RuntimeError):
    """The generated Cypher would modify the graph."""


def normalise(cypher: str) -> str:
    """Uppercase, with comments and string literals removed."""
    without_comments = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", cypher))
    without_strings = _STRING_LITERAL.sub("''", without_comments)
    return re.sub(r"\s+", " ", without_strings).upper().strip()


def is_read_only(cypher: str) -> bool:
    try:
        assert_read_only(cypher)
        return True
    except UnsafeCypherError:
        return False


def assert_read_only(cypher: str) -> None:
    """Raise :class:`UnsafeCypherError` unless ``cypher`` only reads."""
    if not cypher or not cypher.strip():
        raise UnsafeCypherError("The model did not return a Cypher query.")

    text = normalise(cypher)

    for clause in WRITE_CLAUSES:
        # Word boundaries so MERGE does not match inside a property name, and
        # so `SET` does not match the middle of `OFFSET`.
        if re.search(r"(?<![A-Z0-9_])" + re.escape(clause) + r"(?![A-Z0-9_])", text):
            raise UnsafeCypherError(
                f"The generated Cypher contains a write operation ({clause}). "
                f"This app answers questions; it does not modify the graph."
            )

    if not re.search(r"(?<![A-Z0-9_])(MATCH|RETURN|WITH|UNWIND|CALL|SHOW)(?![A-Z0-9_])", text):
        raise UnsafeCypherError("The generated Cypher does not look like a read query.")
