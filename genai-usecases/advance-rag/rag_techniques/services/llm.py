"""Chat model and embedding factories, one per provider."""

from __future__ import annotations

from functools import lru_cache

from services.config import PROVIDERS, api_key_for


class ProviderError(RuntimeError):
    """Unknown provider, or its API key is missing."""


def _require_key(provider: str) -> str:
    if provider not in PROVIDERS:
        raise ProviderError(f"Unknown provider: {provider}")
    key = api_key_for(provider)
    if not key:
        raise ProviderError(
            f"{PROVIDERS[provider]['env_key']} is not set. "
            f"Add it to your .env file and restart the API."
        )
    return key


@lru_cache(maxsize=8)
def get_llm(provider: str, model: str, temperature: float):
    key = _require_key(provider)

    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, api_key=key)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, google_api_key=key
    )


@lru_cache(maxsize=4)
def get_embeddings(provider: str):
    """Embeddings for ``provider``.

    Cached: the Groq path loads a sentence-transformers model into memory, and
    doing that per upload is the difference between seconds and minutes.
    """
    key = _require_key(provider)

    if provider.startswith("Groq"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ProviderError(
                "langchain-huggingface is required for the Groq provider, "
                "because Groq serves no embedding model. "
                "Run: pip install langchain-huggingface"
            ) from exc
        return HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from services.retry import RetryingEmbeddings

    # Wrapped: Google's embedding endpoint returns transient 500s on the free
    # tier, and an unretried blip mid-run looks like a broken app.
    return RetryingEmbeddings(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=key
        )
    )
