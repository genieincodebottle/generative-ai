"""Environment and model catalogue for the Agentic RAG service."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_FILES = int(os.getenv("MAX_FILES", "20"))

# Rolling aliases first: they track Google's current generation, so they do not
# 404 the way a pinned ID does once it is retired.
LLM_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-pro-latest",
]

EMBEDDING_MODELS = ["models/gemini-embedding-001"]


def google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY")


def tavily_api_key() -> str | None:
    return os.getenv("TAVILY_API_KEY")


def has_google_key() -> bool:
    return bool(google_api_key())


def has_tavily_key() -> bool:
    """Web search is optional; the system degrades to documents-only without it."""
    return bool(tavily_api_key())
