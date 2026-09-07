"""Providers, models, and Neo4j connection settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

PROVIDERS: dict[str, dict] = {
    "Gemini": {
        "env_key": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        # Rolling aliases first: they track Google's current generation, so a
        # fresh clone does not 404 the way a retired pinned ID does.
        "models": ["gemini-flash-latest", "gemini-flash-lite-latest",
                   "gemini-pro-latest", "gemini-2.5-flash"],
    },
    "Groq": {
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                   "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    },
}


def api_key_for(provider: str) -> str | None:
    spec = PROVIDERS.get(provider)
    return os.getenv(spec["env_key"]) if spec else None


def available_providers() -> list[str]:
    return [p for p in PROVIDERS if api_key_for(p)]


def models_for(provider: str) -> list[str]:
    spec = PROVIDERS.get(provider)
    return list(spec["models"]) if spec else []
