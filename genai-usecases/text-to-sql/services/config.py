"""Configuration: API keys, provider catalogue, and paths.

This module is the single place that reads the environment. Nothing in the UI
or the API layer touches ``os.environ`` directly, so there is exactly one file
to look at when a key is not picked up.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The project root, resolved from this file rather than the working directory,
# so the app behaves the same whether it is started from the repo root or from
# inside the project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("DB_PATH") or PROJECT_ROOT / "chinook.db")

# Where the Streamlit UI looks for the FastAPI service.
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# Models are listed newest first. The ``-latest`` aliases track Google's current
# generation, so this list does not go stale the way a pinned ID does; the
# pinned IDs below them are there when you need a reproducible run.
PROVIDER_MODELS: dict[str, list[str]] = {
    "Google Gemini": [
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-pro-latest",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-pro-latest",
    ],
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ],
}

_PROVIDER_ENV = {
    "Google Gemini": "GOOGLE_API_KEY",
    "Groq": "GROQ_API_KEY",
}


def api_key_for(provider: str) -> str | None:
    """Return the configured key for ``provider``, or None if it is unset."""
    env_name = _PROVIDER_ENV.get(provider)
    return os.getenv(env_name) if env_name else None


def available_providers() -> list[str]:
    """Providers that actually have a key set, in catalogue order."""
    return [p for p in PROVIDER_MODELS if api_key_for(p)]


def models_for(provider: str) -> list[str]:
    return PROVIDER_MODELS.get(provider, [])


def default_provider() -> str | None:
    providers = available_providers()
    return providers[0] if providers else None
