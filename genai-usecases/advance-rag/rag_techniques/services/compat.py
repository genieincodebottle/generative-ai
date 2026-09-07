"""Import shims for the LangChain 0.3 -> 1.x reorganisation.

LangChain 1.0 emptied the top-level `langchain.retrievers` namespace. The
retrievers and document compressors this project needs moved to
`langchain_classic`, so code written against 0.3 fails at import time on 1.x
with a bare `ModuleNotFoundError: No module named 'langchain.retrievers'`.

A fresh clone could reasonably resolve either version, so every import that
moved goes through this module: try the 1.x location first, fall back to the
0.3 one, and raise something that names the fix if neither is present.

| class | 0.3.x | 1.x |
|---|---|---|
| `ContextualCompressionRetriever` | `langchain.retrievers` | `langchain_classic.retrievers` |
| `EnsembleRetriever` | `langchain.retrievers` | `langchain_classic.retrievers` |
| document compressors | `langchain.retrievers.document_compressors` | `langchain_classic.retrievers.document_compressors` |
| `BM25Retriever` | `langchain_community.retrievers` | `langchain_community.retrievers` (unchanged) |
"""

from __future__ import annotations

import warnings


class MissingDependency(RuntimeError):
    """A LangChain class this project needs is not importable."""


def _first(paths: list[tuple[str, str]], what: str, install_hint: str):
    """Return the first importable ``name`` from ``(module, name)`` pairs."""
    errors = []
    for module_name, attr in paths:
        try:
            with warnings.catch_warnings():
                # langchain_classic re-exports emit deprecation warnings that
                # are not actionable here; the shim is the action.
                warnings.simplefilter("ignore")
                module = __import__(module_name, fromlist=[attr])
            return getattr(module, attr)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise MissingDependency(
        f"Could not import {what}. Tried:\n  " + "\n  ".join(errors) +
        f"\n{install_hint}"
    )


def contextual_compression_retriever():
    return _first(
        [("langchain_classic.retrievers", "ContextualCompressionRetriever"),
         ("langchain.retrievers", "ContextualCompressionRetriever")],
        "ContextualCompressionRetriever",
        "Run: pip install -r requirements.txt",
    )


def ensemble_retriever():
    return _first(
        [("langchain_classic.retrievers", "EnsembleRetriever"),
         ("langchain.retrievers", "EnsembleRetriever")],
        "EnsembleRetriever",
        "Run: pip install -r requirements.txt",
    )


def bm25_retriever():
    return _first(
        [("langchain_community.retrievers", "BM25Retriever"),
         ("langchain_classic.retrievers", "BM25Retriever")],
        "BM25Retriever",
        "Hybrid search needs rank_bm25. Run: pip install rank_bm25",
    )


def document_compressor(name: str, install_hint: str):
    """One of the document compressors, wherever it currently lives."""
    return _first(
        [("langchain_classic.retrievers.document_compressors", name),
         ("langchain.retrievers.document_compressors", name)],
        name,
        install_hint,
    )
