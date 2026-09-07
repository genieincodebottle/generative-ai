"""Technique behaviour, exercised with stub models. No API key, no network.

The stubs make the *strategy* testable: which chunks each technique asks for,
how many LLM calls it makes, and what it does when something returns nothing.
"""

import pytest
from langchain_core.documents import Document as LCDocument
from langchain_core.runnables import Runnable

from services.techniques import (
    Index,
    TechniqueError,
    adaptive,
    basic,
    classify_query,
    corrective,
    reranking,
    run,
)


class Reply:
    """Stands in for an AIMessage."""

    def __init__(self, text):
        self.content = text


class StubLLM(Runnable):
    """Returns queued replies and records every prompt it was given.

    A real Runnable, not a duck type: LangChain's `prompt | llm` validates the
    right-hand side and rejects anything that is not one.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, input, config=None, **kwargs):
        self.prompts.append(input)
        return Reply(self.replies.pop(0) if self.replies else "stub answer")


class StubStore:
    """A vector store that records every search and returns fixed chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.searches = []

    def similarity_search(self, query, k=4):
        self.searches.append((query, k))
        return self.chunks[:k]

    def as_retriever(self, search_kwargs=None):
        store = self
        k = (search_kwargs or {}).get("k", 4)

        class Retriever:
            def invoke(self, query, *a, **kw):
                return store.similarity_search(query, k=k)

        return Retriever()


def make_index(n=10):
    chunks = [LCDocument(page_content=f"chunk {i}", metadata={"id": i})
              for i in range(n)]
    return Index(chunks=chunks, vectorstore=StubStore(chunks), embeddings=object())


class TestBasic:
    def test_retrieves_top_k_and_answers(self):
        index = make_index()
        llm = StubLLM(["the answer"])
        result = basic(index, llm, "a question", top_k=3)
        assert result["answer"] == "the answer"
        assert index.vectorstore.searches == [("a question", 3)]
        assert len(result["documents"]) == 3

    def test_top_k_is_honoured(self):
        index = make_index()
        basic(index, StubLLM(["x"]), "q", top_k=7)
        assert index.vectorstore.searches[0][1] == 7


class TestClassifyQuery:
    @pytest.mark.parametrize(
        "reply, expected",
        [("simple", "simple"), ("COMPLEX", "complex"), ("  moderate  ", "moderate"),
         ("complex.", "complex"), ("simple - a direct lookup", "simple")],
    )
    def test_parses_the_label(self, reply, expected):
        assert classify_query(StubLLM([reply]), "q") == expected

    @pytest.mark.parametrize("reply", ["", "unclear", "banana", "   "])
    def test_falls_back_to_moderate(self, reply):
        # An unparseable classification must not fail the request.
        assert classify_query(StubLLM([reply]), "q") == "moderate"


class TestAdaptive:
    @pytest.mark.parametrize(
        "level, expected_k", [("simple", 2), ("moderate", 4), ("complex", 8)]
    )
    def test_k_scales_with_complexity(self, level, expected_k):
        index = make_index()
        result = adaptive(index, StubLLM([level, "answer"]), "q")
        assert result["k_used"] == expected_k
        # First call classifies, second retrieves.
        assert index.vectorstore.searches[0][1] == expected_k

    def test_reports_the_level_it_chose(self):
        result = adaptive(make_index(), StubLLM(["complex", "answer"]), "q")
        assert result["complexity_level"] == "complex"
        assert "complex" in result["retrieval_method"]


class TestCorrective:
    def test_makes_three_llm_calls_and_two_retrievals(self):
        index = make_index()
        llm = StubLLM(["draft", "the critique", "final answer"])
        result = corrective(index, llm, "q")

        assert result["initial_response"] == "draft"
        assert result["critique"] == "the critique"
        assert result["answer"] == "final answer"
        assert len(llm.prompts) == 3
        assert len(index.vectorstore.searches) == 2

    def test_second_retrieval_uses_the_critique_not_the_query(self):
        # This is the whole idea of corrective RAG: the follow-up search is
        # driven by what was missing, not by the original wording.
        index = make_index()
        corrective(index, StubLLM(["draft", "missing budget figures", "final"]), "q")
        second_search_query = index.vectorstore.searches[1][0]
        assert second_search_query == "missing budget figures"


class TestReranking:
    def test_falls_back_when_the_reranker_filters_everything(self, monkeypatch):
        """An aggressive filter can legitimately return nothing.

        Answering from an empty context would read as "the model knows nothing
        about your documents", so it falls back and says what happened.
        """
        index = make_index()

        class EmptyCompressionRetriever:
            def __init__(self, **kwargs):
                pass

            def invoke(self, query, *a, **k):
                return []

        import services.compat as compat
        import services.techniques as mod
        monkeypatch.setattr(mod, "build_reranker", lambda *a, **k: object())
        # Patch the shim, not a hard-coded module path: the real class moved
        # between LangChain 0.3 and 1.x, which is exactly what compat hides.
        monkeypatch.setattr(
            compat, "contextual_compression_retriever",
            lambda: EmptyCompressionRetriever,
        )

        result = reranking(index, StubLLM(["answer"]), "q")
        assert result["answer"] == "answer"
        assert result["documents"], "should fall back to the un-ranked chunks"
        assert any("filtered everything out" in s for s in result["steps"])


class TestRun:
    def test_rejects_unknown_technique(self):
        with pytest.raises(TechniqueError, match="Unknown technique"):
            run("nope", make_index(), StubLLM([]), "q")

    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    def test_rejects_blank_query(self, blank):
        with pytest.raises(TechniqueError, match="query is empty"):
            run("basic", make_index(), StubLLM([]), blank)

    def test_stamps_the_technique_and_query(self):
        result = run("basic", make_index(), StubLLM(["a"]), "  spaced  ")
        assert result["technique"] == "basic"
        assert result["query"] == "spaced"
