"""Models, dataset paths, and limits for cache-augmented generation."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "sample_qa_dataset.csv"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "40000"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "50"))

# Ungated models come first, deliberately.
#
# This project used to offer only `meta-llama/Llama-3.2-1B-Instruct`, which is
# a GATED repo: you need a HuggingFace token AND you must accept Meta's
# licence with that same account, or the download fails with a 401 that does
# not explain itself. That is a hard stop before you have seen the idea work.
#
# The ungated models below need no token at all and demonstrate KV caching
# just as well - the point of the project is the cache, not the model.
MODELS = [
    {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "label": "SmolLM2 360M (ungated, ~720 MB)",
        "gated": False,
        "note": "Smallest. Runs on CPU. No token needed.",
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B (ungated, ~1 GB)",
        "gated": False,
        "note": "Better answers than SmolLM2, still CPU-friendly. No token needed.",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B (ungated, ~3 GB)",
        "gated": False,
        "note": "Noticeably better. A GPU helps.",
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "label": "Llama 3.2 1B (GATED, ~2.5 GB)",
        "gated": True,
        "note": ("Needs HF_TOKEN and an accepted licence at "
                 "huggingface.co/meta-llama/Llama-3.2-1B-Instruct"),
    },
]

MODEL_IDS = [m["id"] for m in MODELS]
DEFAULT_MODEL = MODELS[0]["id"]


def model_spec(model_id: str) -> dict | None:
    return next((m for m in MODELS if m["id"] == model_id), None)


def is_gated(model_id: str) -> bool:
    spec = model_spec(model_id)
    return bool(spec and spec["gated"])


def hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or None
