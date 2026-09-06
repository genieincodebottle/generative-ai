"""One provider layer for every app in this project.

`create_llm` and the model catalogue used to be copy-pasted into all thirteen
app files. That is why the Anthropic list went stale in thirteen places at
once, and why fixing it meant thirteen identical edits.

It also had a quiet bug worth keeping in mind: the original had no `else`
branch, so an unknown provider returned ``None`` and the caller crashed later
with ``AttributeError: 'NoneType' object has no attribute 'invoke'`` - a long
way from the actual mistake. This version raises where the mistake is.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

PROVIDERS: dict[str, dict] = {
    # Ollama runs locally and needs no key, so it is always offered.
    "Ollama": {
        "env_key": None,
        "key_url": "https://ollama.com/download",
        "models": ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "gemma2:2b",
                   "gemma2:9b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b",
                   "codestral:22b", "deepseek-coder:1.3b"],
        "note": "Runs on your machine. No API key, no cost, slower.",
    },
    "Gemini": {
        "env_key": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        # Rolling aliases first: they track Google's current generation, so
        # they do not 404 the way a retired pinned ID does.
        "models": ["gemini-flash-latest", "gemini-flash-lite-latest",
                   "gemini-pro-latest", "gemini-3.8-flash", "gemini-3.5-flash",
                   "gemini-3.1-flash-lite", "gemini-3.1-pro-preview",
                   "gemini-2.5-flash", "gemini-pro-latest"],
        "note": "Free tier, no card required.",
    },
    "Groq": {
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                   "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "note": "Free tier, very fast.",
    },
    "Anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/settings/keys",
        "models": ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
                   "claude-sonnet-4-6", "claude-haiku-4-5"],
        "note": "Strong at reasoning and code.",
    },
    "OpenAI": {
        "env_key": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "models": ["gpt-5", "gpt-5-mini", "gpt-5-nano",
                   "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o"],
        "note": "",
    },
}


class ProviderError(RuntimeError):
    """Unknown provider, missing key, or a missing provider package."""


def api_key_for(provider: str) -> str | None:
    spec = PROVIDERS.get(provider)
    if not spec:
        return None
    env_key = spec["env_key"]
    return "local" if env_key is None else os.getenv(env_key)


def available_providers() -> list[str]:
    """Providers that can actually be used right now.

    Ollama is always listed because it needs no key; whether the server is
    running is a separate question, answered by :func:`ollama_reachable`.
    """
    return [p for p in PROVIDERS if api_key_for(p)]


def models_for(provider: str) -> list[str]:
    spec = PROVIDERS.get(provider)
    return list(spec["models"]) if spec else []


_OLLAMA_PROBE: dict[str, tuple[float, bool]] = {}
_OLLAMA_PROBE_TTL = 15.0


def ollama_reachable(base_url: str | None = None) -> bool:
    """True if a local Ollama server answers. Never raises.

    Cached, and on a short timeout, because /health calls this on every
    request. An uncached 2-second probe made /health take just over two
    seconds, which is longer than the launcher's own poll timeout - so the
    launcher timed out on every attempt and reported that the API had never
    started, while the API was serving happily.
    """
    import time
    import urllib.error
    import urllib.request

    url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/api/tags"

    cached = _OLLAMA_PROBE.get(url)
    if cached and time.monotonic() - cached[0] < _OLLAMA_PROBE_TTL:
        return cached[1]

    try:
        with urllib.request.urlopen(url, timeout=0.7) as response:
            reachable = response.status == 200
    except (urllib.error.URLError, OSError):
        reachable = False

    _OLLAMA_PROBE[url] = (time.monotonic(), reachable)
    return reachable


@lru_cache(maxsize=16)
def create_llm(provider: str, model: str, base_url: str | None = None,
               temperature: float = DEFAULT_TEMPERATURE):
    """Build a chat model. Raises :class:`ProviderError` rather than returning None."""
    if provider not in PROVIDERS:
        raise ProviderError(
            f"Unknown provider: {provider}. "
            f"Choose one of: {', '.join(PROVIDERS)}."
        )

    key = api_key_for(provider)
    if not key:
        spec = PROVIDERS[provider]
        raise ProviderError(
            f"{spec['env_key']} is not set. Add it to your .env file "
            f"(get one at {spec['key_url']}), then restart the API."
        )

    try:
        if provider == "Ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model,
                base_url=base_url or OLLAMA_BASE_URL,
                temperature=temperature,
                timeout=120,
            )

        if provider == "Gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model, api_key=key, temperature=temperature
            )

        if provider == "Groq":
            from langchain_groq import ChatGroq

            return ChatGroq(model=model, api_key=key, temperature=temperature)

        if provider == "Anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model, api_key=key, temperature=temperature
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=key, temperature=temperature)

    except ImportError as exc:
        raise ProviderError(
            f"The package for {provider} is not installed: {exc}. "
            f"Run: pip install -r requirements.txt"
        ) from exc
