"""The five retrieval strategies, side by side.

They are in one file on purpose: comparing them is the point of this project,
and the differences are easier to see when they sit next to each other than
when they are five near-identical files.

Every technique takes the same inputs (an index, an LLM, a query) and returns
a dict with at least ``answer`` and ``retrieval_method``. What differs is the
retrieval strategy, and each class documents its own trade-off.

None of this imports Streamlit or FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from services import compat
from services.llm_text import message_text


class TechniqueError(RuntimeError):
    """The technique could not run - usually a missing optional dependency."""


def as_context(docs: list[LCDocument]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def as_payload(docs: list[LCDocument]) -> list[dict]:
    """Documents in a shape the UI can render without importing LangChain."""
    return [
        {"content": doc.page_content, "metadata": {k: str(v) for k, v in
                                                   (doc.metadata or {}).items()}}
        for doc in docs
    ]


@dataclass
class Index:
    """Everything a technique needs, built once per session."""

    chunks: list[LCDocument]
    vectorstore: object
    embeddings: object


# ---------------------------------------------------------------------------
# 1. Basic
# ---------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant tasked with answering questions based on the provided context. "
    "Please provide a comprehensive and accurate answer to the question using the context provided. "
    "If the context doesn't contain enough information to fully answer the question, "
    "please indicate what information is missing.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)


def basic(index: Index, llm, query: str, top_k: int = 4, **_) -> dict:
    """Embed the query, take the nearest ``top_k`` chunks, answer.

    The baseline. Everything below is a response to one of its failure modes.
    """
    docs = index.vectorstore.similarity_search(query, k=top_k)
    answer = message_text((ANSWER_PROMPT | llm).invoke(
        {"context": as_context(docs), "query": query}
    ))
    return {
        "answer": answer,
        "retrieval_method": f"Basic RAG (top-{top_k} similarity search)",
        "documents": as_payload(docs),
        "steps": [f"Retrieved {len(docs)} chunks by vector similarity"],
    }


# ---------------------------------------------------------------------------
# 2. Adaptive
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = (
    "Classify the following question into exactly one complexity category.\n\n"
    "Categories:\n"
    "- simple: a direct factual lookup; a short, specific answer is expected\n"
    "- moderate: requires explanation, context, or basic analysis\n"
    "- complex: requires synthesis from multiple sources, comparison of "
    "approaches, or multi-step reasoning\n\n"
    "Question: {query}\n\n"
    "Respond with a single word only (simple / moderate / complex):"
)

K_BY_COMPLEXITY = {"simple": 2, "moderate": 4, "complex": 8}

STYLE_BY_COMPLEXITY = {
    "simple": "Answer concisely and directly in one or two sentences.\n\n"
              "Context:\n{context}\n\nQuestion: {query}\nAnswer:",
    "moderate": "Explain the answer clearly, with the reasoning that supports "
                "it.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:",
    "complex": "Synthesise across the context. Compare where sources differ, "
               "and state what is not covered.\n\nContext:\n{context}\n\n"
               "Question: {query}\nAnswer:",
}


def classify_query(llm, query: str) -> str:
    """Ask the model how hard the question is. Fall back to 'moderate'.

    The fallback matters: an unparseable classification must not fail the
    request, it should just pick the middle setting.
    """
    raw = message_text(llm.invoke(CLASSIFY_PROMPT.format(query=query)))
    first = raw.strip().lower().split()
    level = first[0].strip(".,:;'\"") if first else ""
    return level if level in K_BY_COMPLEXITY else "moderate"


def adaptive(index: Index, llm, query: str, **_) -> dict:
    """Spend retrieval budget in proportion to how hard the question is.

    A lookup does not need eight chunks, and a comparison is not served by two.
    """
    level = classify_query(llm, query)
    k = K_BY_COMPLEXITY[level]
    docs = index.vectorstore.similarity_search(query, k=k)
    answer = message_text(llm.invoke(
        STYLE_BY_COMPLEXITY[level].format(context=as_context(docs), query=query)
    ))
    return {
        "answer": answer,
        "retrieval_method": f"Adaptive RAG (classified {level}, k={k})",
        "documents": as_payload(docs),
        "complexity_level": level,
        "k_used": k,
        "steps": [
            f"Classified the question as {level}",
            f"Retrieved {len(docs)} chunks (k chosen by complexity)",
            f"Answered with the {level} prompt style",
        ],
    }


# ---------------------------------------------------------------------------
# 3. Corrective
# ---------------------------------------------------------------------------

INITIAL_PROMPT = ChatPromptTemplate.from_template(
    "Based on the following context, please answer the query:\n"
    "Context: {context}\nQuery: {query}"
)
CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    "Please critique the following response to the query. Identify any "
    "potential errors or missing information:\nQuery: {query}\nResponse: {response}"
)
FINAL_PROMPT = ChatPromptTemplate.from_template(
    "Based on the initial response, critique, and additional context, please "
    "provide an improved answer to the query:\n"
    "Initial Response: {initial_response}\nCritique: {critique}\n"
    "Additional Context: {additional_context}\nQuery: {query}"
)


def corrective(index: Index, llm, query: str, **_) -> dict:
    """Answer, criticise that answer, retrieve again, answer better.

    The second retrieval is driven by the *critique*, not the question, which
    is how it finds context the first search missed. It costs three LLM calls.
    """
    initial_docs = index.vectorstore.similarity_search(query, k=3)
    initial = message_text((INITIAL_PROMPT | llm).invoke(
        {"context": as_context(initial_docs), "query": query}
    ))
    critique = message_text((CRITIQUE_PROMPT | llm).invoke(
        {"response": initial, "query": query}
    ))
    extra_docs = index.vectorstore.similarity_search(critique, k=2)
    final = message_text((FINAL_PROMPT | llm).invoke({
        "initial_response": initial,
        "critique": critique,
        "additional_context": as_context(extra_docs),
        "query": query,
    }))
    return {
        "answer": final,
        "retrieval_method": "Corrective RAG (answer, critique, re-retrieve, answer)",
        "documents": as_payload(initial_docs + extra_docs),
        "initial_response": initial,
        "critique": critique,
        "steps": [
            f"Retrieved {len(initial_docs)} chunks and drafted an answer",
            "Critiqued that draft",
            f"Retrieved {len(extra_docs)} more chunks using the critique",
            "Rewrote the answer",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Hybrid search
# ---------------------------------------------------------------------------

def hybrid(index: Index, llm, query: str, bm25_weight: float = 0.5,
           vector_weight: float = 0.5, top_k: int = 5, **_) -> dict:
    """Blend BM25 keyword search with vector search.

    Vector search understands paraphrase but misses exact tokens - product
    codes, error numbers, surnames. BM25 is the opposite. The ensemble covers
    both, which is why it is the usual first upgrade from basic RAG.
    """
    try:
        BM25Retriever = compat.bm25_retriever()
        EnsembleRetriever = compat.ensemble_retriever()
    except compat.MissingDependency as exc:
        raise TechniqueError(str(exc)) from exc

    bm25 = BM25Retriever.from_documents(index.chunks)
    bm25.k = top_k
    vector = index.vectorstore.as_retriever(search_kwargs={"k": top_k})

    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector], weights=[bm25_weight, vector_weight]
    )

    bm25_docs = bm25.invoke(query)
    vector_docs = vector.invoke(query)
    hybrid_docs = ensemble.invoke(query)

    answer = message_text((ANSWER_PROMPT | llm).invoke(
        {"context": as_context(hybrid_docs), "query": query}
    ))
    return {
        "answer": answer,
        "retrieval_method": (
            f"Hybrid search (BM25 {bm25_weight}, vector {vector_weight})"
        ),
        "documents": as_payload(hybrid_docs),
        "bm25_documents": as_payload(bm25_docs),
        "vector_documents": as_payload(vector_docs),
        "steps": [
            f"BM25 returned {len(bm25_docs)} chunks",
            f"Vector search returned {len(vector_docs)} chunks",
            f"Weighted ensemble kept {len(hybrid_docs)}",
        ],
    }


# ---------------------------------------------------------------------------
# 5. Re-ranking
# ---------------------------------------------------------------------------

def build_reranker(name: str, llm, embeddings):
    """Return a document compressor, or raise with an install hint.

    Every option here is optional-dependency territory, so the failure has to
    name the package rather than surfacing as an ImportError traceback.
    """
    try:
        if name == "Embeddings Filter":
            EmbeddingsFilter = compat.document_compressor(
                "EmbeddingsFilter", "Run: pip install -r requirements.txt")
            return EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.5)

        if name == "FlashRank":
            try:
                from langchain_community.document_compressors.flashrank_rerank import (
                    FlashrankRerank,
                )
            except ImportError as exc:
                raise TechniqueError(
                    "FlashRank is not installed. Run: pip install flashrank"
                ) from exc
            try:
                FlashrankRerank.model_rebuild()
            except Exception:
                pass
            return FlashrankRerank(top_n=5)

        if name == "Cross-Encoder (BGE)":
            CrossEncoderReranker = compat.document_compressor(
                "CrossEncoderReranker",
                "The cross-encoder needs sentence-transformers. "
                "Run: pip install sentence-transformers")
            try:
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            except ImportError as exc:
                raise TechniqueError(
                    "The cross-encoder needs sentence-transformers. "
                    "Run: pip install sentence-transformers"
                ) from exc
            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            return CrossEncoderReranker(model=model, top_n=5)

        if name == "LLM Listwise Rerank":
            LLMListwiseRerank = compat.document_compressor(
                "LLMListwiseRerank", "Run: pip install -r requirements.txt")
            return LLMListwiseRerank.from_llm(llm, top_n=5)

        if name == "LLM Chain Extractor":
            LLMChainExtractor = compat.document_compressor(
                "LLMChainExtractor", "Run: pip install -r requirements.txt")
            return LLMChainExtractor.from_llm(llm)
    except compat.MissingDependency as exc:
        raise TechniqueError(str(exc)) from exc

    raise TechniqueError(f"Unknown reranker: {name}")


def reranking(index: Index, llm, query: str,
              reranker: str = "Embeddings Filter", **_) -> dict:
    """Over-retrieve cheaply, then re-order with something more accurate.

    Embedding similarity is fast and approximate. Fetching 8 and re-ranking to
    5 usually beats fetching 5 directly, because the good chunk is often
    ranked fourth by the embedder and first by the re-ranker.
    """
    ContextualCompressionRetriever = compat.contextual_compression_retriever()

    base = index.vectorstore.as_retriever(search_kwargs={"k": 8})
    before = base.invoke(query)

    compressor = build_reranker(reranker, llm, index.embeddings)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )
    after = retriever.invoke(query)

    # An aggressive filter can legitimately return nothing. Answering from an
    # empty context would look like the model knowing nothing about your
    # documents, so fall back and say what happened.
    used = after or before
    answer = message_text((ANSWER_PROMPT | llm).invoke(
        {"context": as_context(used), "query": query}
    ))

    steps = [
        f"Retrieved {len(before)} chunks before re-ranking",
        f"{reranker} kept {len(after)}",
    ]
    if not after:
        steps.append(
            "The re-ranker filtered everything out; answered from the "
            "un-ranked chunks instead"
        )

    return {
        "answer": answer,
        "retrieval_method": f"Re-ranking RAG ({reranker})",
        "documents": as_payload(used),
        "documents_before_rerank": as_payload(before),
        "steps": steps,
    }


REGISTRY = {
    "basic": basic,
    "adaptive": adaptive,
    "corrective": corrective,
    "hybrid": hybrid,
    "reranking": reranking,
}


def run(technique: str, index: Index, llm, query: str, **options) -> dict:
    if technique not in REGISTRY:
        raise TechniqueError(f"Unknown technique: {technique}")
    if not query or not query.strip():
        raise TechniqueError("The query is empty.")
    result = REGISTRY[technique](index, llm, query.strip(), **options)
    result["technique"] = technique
    result["query"] = query.strip()
    return result
