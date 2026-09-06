"""Providers, models, and the technique catalogue."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = PROJECT_ROOT / "sample_docs"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))
MAX_FILES = int(os.getenv("MAX_FILES", "10"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "12"))

PROVIDERS: dict[str, dict] = {
    "Gemini (Google)": {
        "env_key": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        # Rolling aliases first - they track Google's current generation, so a
        # fresh clone does not 404 the way a retired pinned ID does.
        "models": [
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-pro-latest",
            "gemini-2.5-flash",
            "gemini-pro-latest",
        ],
        "embeddings": "gemini-embedding-001 (Google API)",
    },
    "Groq (Open Source)": {
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        # Groq serves no embedding model, so retrieval runs locally on CPU.
        "embeddings": "nomic-embed-text-v1.5 (local, ~250 MB first run)",
    },
}

# The five techniques this project compares, with the knobs each one exposes.
TECHNIQUES: dict[str, dict] = {
    "basic": {
        "label": "Basic RAG",
        "blurb": "Embed, retrieve top-k, answer. The baseline everything else "
                 "is measured against.",
        "options": ["top_k"],
    },
    "adaptive": {
        "label": "Adaptive RAG",
        "blurb": "Classify the question as simple / moderate / complex first, "
                 "then vary k and the prompt style to match.",
        "options": [],
    },
    "corrective": {
        "label": "Corrective RAG",
        "blurb": "Answer, critique that answer, retrieve again using the "
                 "critique, then answer a second time.",
        "options": [],
    },
    "hybrid": {
        "label": "Hybrid Search RAG",
        "blurb": "Combine BM25 keyword search with vector search in a weighted "
                 "ensemble, so exact terms and paraphrases both land.",
        "options": ["bm25_weight", "vector_weight"],
    },
    "reranking": {
        "label": "Re-ranking RAG",
        "blurb": "Over-retrieve, then re-order with a second, more accurate "
                 "model before the answer is written.",
        "options": ["reranker"],
    },
}

RERANKERS = [
    "Embeddings Filter",
    "FlashRank",
    "Cross-Encoder (BGE)",
    "LLM Listwise Rerank",
    "LLM Chain Extractor",
]

DEFAULTS = {
    "temperature": 0.2,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 4,
    "bm25_weight": 0.5,
    "vector_weight": 0.5,
    "reranker": "Embeddings Filter",
}


def api_key_for(provider: str) -> str | None:
    spec = PROVIDERS.get(provider)
    return os.getenv(spec["env_key"]) if spec else None


def available_providers() -> list[str]:
    return [p for p in PROVIDERS if api_key_for(p)]


def models_for(provider: str) -> list[str]:
    spec = PROVIDERS.get(provider)
    return list(spec["models"]) if spec else []
