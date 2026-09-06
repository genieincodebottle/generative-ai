"""Providers, models, and limits for the multi-function app."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

# Llama 4 Scout on Groq: one model that handles both text and images, which is
# what makes a single "multi-function" app possible.
GROQ_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
]
GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

# Rolling aliases first: they track Google's current generation, so they do
# not 404 the way a retired pinned ID does.
GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-pro-latest",
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FEATURES = {
    "chat": {"label": "Chat", "needs": ["groq"]},
    "vision": {"label": "OCR / vision", "needs": ["groq"]},
    "rag": {"label": "RAG with evaluation", "needs": ["groq"], "optional": ["google"]},
    "agents": {"label": "Agentic AI", "needs": ["groq"], "optional": ["google"]},
}

KEYS = {
    "groq": {
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "label": "Groq",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        "label": "Google Gemini",
    },
}


def api_key(name: str) -> str | None:
    spec = KEYS.get(name)
    return os.getenv(spec["env_key"]) if spec else None


def has(name: str) -> bool:
    return bool(api_key(name))


def available_features() -> list[str]:
    return [f for f, spec in FEATURES.items()
            if all(has(k) for k in spec["needs"])]
