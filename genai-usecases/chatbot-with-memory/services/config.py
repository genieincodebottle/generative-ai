"""Configuration: providers, models, defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# Uploads are held in memory; this caps what one request can cost.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_FILES = int(os.getenv("MAX_FILES", "10"))

# How many chat sessions (each holding a FAISS index) to keep before evicting
# the least recently used. Without this the server grows without bound.
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "20"))

PROVIDERS: dict[str, dict] = {
    "Gemini": {
        "env_key": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        "models": [
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-pro-latest",
            "gemini-2.5-flash",
            "gemini-pro-latest",
        ],
        "model_help": "flash is fast and free-tier friendly; pro is more capable",
        # Gemini embeddings are an API call, so no local model download.
        "embeddings": "gemini",
    },
    "Groq": {
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        "model_help": "8b-instant is fastest; 70b-versatile is more capable",
        # Groq serves no embedding model, so retrieval runs locally on CPU.
        # First use downloads ~90 MB of sentence-transformers weights.
        "embeddings": "huggingface",
    },
}

DEFAULTS = {
    "temperature": 0.3,
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "retriever_k": 3,
}


def api_key_for(provider: str) -> str | None:
    spec = PROVIDERS.get(provider)
    return os.getenv(spec["env_key"]) if spec else None


def available_providers() -> list[str]:
    return [p for p in PROVIDERS if api_key_for(p)]


def models_for(provider: str) -> list[str]:
    spec = PROVIDERS.get(provider)
    return list(spec["models"]) if spec else []
