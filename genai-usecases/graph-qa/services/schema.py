"""Describe the graph without requiring the APOC plugin.

``Neo4jGraph.refresh_schema()`` calls ``apoc.meta.data()``. APOC is a plugin,
and the stock ``neo4j:5-community`` image does not ship it - so the setup
command every tutorial gives you produces an app that dies on connect with:

    Could not use APOC procedures. Please ensure the APOC plugin is installed

That is a confusing first-run failure for something that has nothing to do
with the app. These built-in procedures need no plugin:

    db.labels(), db.relationshipTypes(), db.propertyKeys()

Combined with a sampled scan for which properties each label actually carries,
they produce a schema good enough for a model to write Cypher against.
"""

from __future__ import annotations

SAMPLE_SIZE = 100


def build_schema(graph, sample_size: int = SAMPLE_SIZE) -> str:
    """Return a text schema built from built-in procedures only."""
    labels = [row["label"] for row in graph.query("CALL db.labels() YIELD label RETURN label")]
    rel_types = [
        row["relationshipType"]
        for row in graph.query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
    ]

    lines: list[str] = ["Node labels and their properties:"]
    for label in sorted(labels):
        # Backtick the label: a label with a space or a reserved word would
        # otherwise produce invalid Cypher here.
        rows = graph.query(
            f"MATCH (n:`{label}`) WITH n LIMIT {int(sample_size)} "
            f"UNWIND keys(n) AS key RETURN DISTINCT key ORDER BY key"
        )
        keys = [r["key"] for r in rows]
        count = graph.query(f"MATCH (n:`{label}`) RETURN count(n) AS c")[0]["c"]
        lines.append(
            f"  {label} ({count} nodes): "
            + (", ".join(keys) if keys else "no properties")
        )

    lines.append("")
    lines.append("Relationships:")
    for rel in sorted(rel_types):
        rows = graph.query(
            f"MATCH (a)-[r:`{rel}`]->(b) WITH a, r, b LIMIT {int(sample_size)} "
            f"RETURN DISTINCT labels(a) AS start, labels(b) AS end"
        )
        pairs = sorted({
            f"(:{'/'.join(r['start'])})-[:{rel}]->(:{'/'.join(r['end'])})"
            for r in rows if r.get("start") and r.get("end")
        })
        if pairs:
            lines.extend(f"  {pair}" for pair in pairs)
        else:
            lines.append(f"  [:{rel}]")

    return "\n".join(lines)


def schema_for(graph) -> str:
    """APOC-derived schema if available, otherwise the built-in fallback."""
    try:
        graph.refresh_schema()
        schema = (graph.schema or "").strip()
        if schema:
            return schema
    except Exception:
        # APOC missing, or not allowed by configuration. Fall through.
        pass
    return build_schema(graph)
