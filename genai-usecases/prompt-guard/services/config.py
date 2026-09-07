"""Backends, models, and thresholds for the prompt-injection classifier."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))

# Meta's Prompt Guard 2 comes in two sizes. The 22M model is small enough to
# sit in front of every request; the 86M is more accurate and slower.
MODEL_SIZES = ["22M", "86M"]

BACKENDS: dict[str, dict] = {
    "groq": {
        "label": "Groq API",
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "note": "Hosted. Nothing to download, needs a key.",
        "models": {
            "22M": "meta-llama/llama-prompt-guard-2-22m",
            "86M": "meta-llama/llama-prompt-guard-2-86m",
        },
        # Groq returns a probability-like score for the malicious class.
        "default_threshold": 0.7,
    },
    "huggingface": {
        "label": "Local (HuggingFace)",
        "env_key": "HF_TOKEN",
        "key_url": "https://huggingface.co/settings/tokens",
        "note": ("Runs on your machine. The Prompt Guard repos are gated, so "
                 "you need a token and must accept the model licence first."),
        "models": {
            "22M": "meta-llama/Llama-Prompt-Guard-2-22M",
            "86M": "meta-llama/Llama-Prompt-Guard-2-86M",
        },
        "default_threshold": 0.5,
    },
}

DEFAULT_BACKEND = os.getenv("PROMPT_GUARD_BACKEND", "groq")


def api_key_for(backend: str) -> str | None:
    spec = BACKENDS.get(backend)
    return os.getenv(spec["env_key"]) if spec else None


def available_backends() -> list[str]:
    return [b for b in BACKENDS if api_key_for(b)]


def model_for(backend: str, size: str) -> str | None:
    spec = BACKENDS.get(backend)
    return spec["models"].get(size) if spec else None
