"""Document ingestion and the conversational RAG chain.

Imports no web framework. PDFs arrive as ``(filename, bytes)`` pairs, not as
Streamlit upload objects, so this is callable from anywhere.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.config import MAX_FILES, MAX_UPLOAD_BYTES, PROVIDERS, api_key_for
from services.retry import RetryingEmbeddings

CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

QA_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the answer concise (3 sentences max).\n\n"
    "{context}"
)


class ProviderError(RuntimeError):
    """Unknown provider, or its API key is missing."""


class IngestError(RuntimeError):
    """The uploaded files could not be turned into a searchable index."""


@dataclass
class Document:
    """One uploaded file, already read into memory."""

    filename: str
    content: bytes


@dataclass
class IngestResult:
    filenames: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0


def get_llm(provider: str, model: str, temperature: float) -> BaseChatModel:
    key = api_key_for(provider)
    if provider not in PROVIDERS:
        raise ProviderError(f"Unknown provider: {provider}")
    if not key:
        raise ProviderError(
            f"{PROVIDERS[provider]['env_key']} is not set. "
            f"Add it to your .env file and restart the API."
        )

    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, api_key=key)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, google_api_key=key
    )


@lru_cache(maxsize=4)
def get_embeddings(provider: str):
    """Embeddings for ``provider``.

    Cached because the HuggingFace path loads a model into memory, and doing
    that once per upload is the difference between two seconds and thirty.
    """
    if provider not in PROVIDERS:
        raise ProviderError(f"Unknown provider: {provider}")

    if PROVIDERS[provider]["embeddings"] == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ProviderError(
                "langchain-huggingface is required for Groq embeddings. "
                "Run: pip install langchain-huggingface"
            ) from exc
        # Local model: no network call, so no retry needed.
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    key = api_key_for(provider)
    if not key:
        raise ProviderError(f"{PROVIDERS[provider]['env_key']} is not set.")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Wrapped: Google's embedding endpoint returns transient 500s on the free
    # tier, and an unretried blip mid-conversation looks like a broken app.
    return RetryingEmbeddings(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=key
        )
    )


def validate_uploads(documents: list[Document]) -> None:
    """Reject uploads before any of them is parsed."""
    if not documents:
        raise IngestError("No files were uploaded.")
    if len(documents) > MAX_FILES:
        raise IngestError(f"Too many files (maximum {MAX_FILES}).")

    total = sum(len(d.content) for d in documents)
    if total > MAX_UPLOAD_BYTES:
        raise IngestError(
            f"Upload is too large: {total / 1e6:.1f} MB, "
            f"maximum {MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )
    for doc in documents:
        if not doc.filename.lower().endswith(".pdf"):
            raise IngestError(f"{doc.filename} is not a PDF.")
        if not doc.content:
            raise IngestError(f"{doc.filename} is empty.")


def load_pdfs(documents: list[Document]) -> list:
    """Parse PDFs into LangChain documents via short-lived temp files."""
    loaded = []
    for doc in documents:
        handle, path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(handle, "wb") as tmp:
                tmp.write(doc.content)
            pages = PyPDFLoader(path).load()
            for page in pages:
                # PyPDFLoader records the temp path; replace it with the name
                # the user actually recognises.
                page.metadata["source"] = doc.filename
            loaded.extend(pages)
        except Exception as exc:
            raise IngestError(f"Could not read {doc.filename}: {exc}") from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    return loaded


def build_index(documents: list[Document], provider: str, chunk_size: int,
                chunk_overlap: int) -> tuple[FAISS, IngestResult]:
    """Turn uploaded PDFs into a FAISS index. Raises :class:`IngestError`."""
    validate_uploads(documents)
    pages = load_pdfs(documents)

    # A scanned PDF parses fine and yields no text. Saying "no content found"
    # here is far more useful than an empty index that answers "I don't know"
    # to everything.
    if not pages or not any(p.page_content.strip() for p in pages):
        raise IngestError(
            "No extractable text found. These may be scanned images rather "
            "than text PDFs; this app does not run OCR."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
    ).split_documents(pages)

    if not chunks:
        raise IngestError("The documents produced no chunks to index.")

    try:
        index = FAISS.from_documents(chunks, get_embeddings(provider))
    except ProviderError:
        raise
    except Exception as exc:
        raise IngestError(f"Could not build the search index: {exc}") from exc

    return index, IngestResult(
        filenames=[d.filename for d in documents],
        pages=len(pages),
        chunks=len(chunks),
    )


def build_chain(llm: BaseChatModel, index: FAISS, retriever_k: int,
                history_factory) -> RunnableWithMessageHistory:
    """Assemble the history-aware retrieval chain.

    ``history_factory`` is supplied by the caller so conversation memory lives
    in the session store, not in a dict this function creates. The original
    version built that dict here, which meant every rebuild of the chain
    silently discarded the conversation.
    """
    retriever = index.as_retriever(search_kwargs={"k": retriever_k})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # StrOutputParser returns a TextAccessor, not a str, in current LangChain.
    # FAISS passes whatever it is given straight to the embedding client, and
    # the Google SDK serialises a TextAccessor into a request the API rejects
    # with a *deterministic* 500 INTERNAL - which reads like an outage rather
    # than a type error. Coercing here is the whole fix.
    history_aware_retriever = (
        contextualize_prompt | llm | StrOutputParser() | (lambda q: str(q)) | retriever
    )

    def format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(context=history_aware_retriever | format_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        history_factory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
