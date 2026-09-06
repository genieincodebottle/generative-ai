"""LLM construction, kept behind one factory so providers stay swappable."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from services.config import api_key_for


class LLMError(RuntimeError):
    """Raised when a provider is unknown or its API key is missing."""


@lru_cache(maxsize=8)
def get_llm(provider: str, model: str, temperature: float = 0.0) -> BaseChatModel:
    """Build (and cache) a chat model for ``provider``.

    Cached on the argument triple so repeated requests reuse one client instead
    of opening a fresh connection pool per call.
    """
    key = api_key_for(provider)
    if not key:
        raise LLMError(
            f"No API key configured for {provider}. "
            f"Set it in your .env file, then restart the API."
        )

    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=key, temperature=temperature
        )

    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(api_key=key, model=model, temperature=temperature)

    raise LLMError(f"Unknown provider: {provider}")
