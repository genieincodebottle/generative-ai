"""Configuration: providers, models, defaults, and paths.

The single place that reads the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("DB_PATH") or PROJECT_ROOT / "sentiment_analysis.db")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# Call text outside this range is rejected before it reaches the model.
MIN_CALL_LENGTH = 10
MAX_CALL_LENGTH = 10_000

PROVIDERS: dict[str, dict] = {
    "groq": {
        "label": "Groq (free tier, fastest)",
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
    },
    "gemini": {
        "label": "Google Gemini (free tier)",
        "env_key": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        # Rolling aliases first: they track Google's current generation, so
        # they do not 404 the way a pinned ID does once it is retired.
        "models": [
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-pro-latest",
            "gemini-2.5-flash",
            "gemini-pro-latest",
        ],
    },
}

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

_DEFAULT_MODEL_ENV = {"groq": "GROQ_MODEL", "gemini": "GEMINI_MODEL"}


def api_key_for(provider: str) -> str | None:
    spec = PROVIDERS.get(provider)
    return os.getenv(spec["env_key"]) if spec else None


def available_providers() -> list[str]:
    return [p for p in PROVIDERS if api_key_for(p)]


def models_for(provider: str) -> list[str]:
    spec = PROVIDERS.get(provider)
    return list(spec["models"]) if spec else []


def default_model_for(provider: str) -> str:
    """Env override if set, otherwise the first model in the catalogue."""
    override = os.getenv(_DEFAULT_MODEL_ENV.get(provider, ""), "")
    models = models_for(provider)
    return override or (models[0] if models else "")
