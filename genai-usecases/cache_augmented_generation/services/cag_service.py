"""Owns the loaded model and runs cache-augmented generation experiments.

Cache-augmented generation puts the whole document into the model's context
once, keeps the resulting KV cache, and answers every question by reusing it.
No retrieval, no vector store, no chunking. The trade is obvious and worth
measuring: it only works if the document fits in the context window.

This module compares the two paths so the trade is a number rather than a
claim.
"""

from __future__ import annotations

import csv
import io
import threading
from dataclasses import dataclass, field

from services.cag_model import CAGModel, ModelLoadError
from services.config import (
    DEFAULT_DATASET,
    MAX_DOCUMENT_CHARS,
    MAX_QUESTIONS,
    MODEL_IDS,
    hf_token,
    is_gated,
    model_spec,
)


class NotLoaded(RuntimeError):
    """No model is loaded yet."""


class InvalidInput(ValueError):
    """The document or dataset is unusable."""


@dataclass
class LoadedModel:
    model_id: str
    quantized: bool
    model: CAGModel


@dataclass
class DatasetRow:
    question: str
    answer: str
    # Some datasets carry the source document per row. The bundled corpus
    # does, which is what lets this project run with nothing pasted in.
    document: str = ""


@dataclass
class State:
    loaded: LoadedModel | None = None


_state = State()
_lock = threading.Lock()


def validate_document(document: str) -> str:
    text = (document or "").strip()
    if not text:
        raise InvalidInput("The document is empty.")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise InvalidInput(
            f"The document is {len(text)} characters, over the "
            f"{MAX_DOCUMENT_CHARS} limit. Cache-augmented generation needs "
            f"the whole document to fit in the context window; past a point "
            f"that is what retrieval is for."
        )
    return text


def parse_dataset(content: bytes | str) -> list[DatasetRow]:
    """Read a two-column question/answer CSV."""
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    reader = csv.DictReader(io.StringIO(text))

    if not reader.fieldnames:
        raise InvalidInput("The dataset is empty.")
    lowered = {name.lower().strip(): name for name in reader.fieldnames}

    # Accept both shapes: a plain question/answer CSV, and the bundled corpus
    # whose columns are topic / text / sample_question / sample_ground_truth.
    question_key = lowered.get("question") or lowered.get("sample_question")
    answer_key = (lowered.get("answer") or lowered.get("ground_truth")
                  or lowered.get("sample_ground_truth"))
    document_key = lowered.get("text") or lowered.get("document")

    if not question_key or not answer_key:
        raise InvalidInput(
            f"The dataset needs question and answer columns "
            f"(question/answer, or sample_question/sample_ground_truth). "
            f"Found: {', '.join(reader.fieldnames)}"
        )

    rows = [
        DatasetRow(question=(r.get(question_key) or "").strip(),
                   answer=(r.get(answer_key) or "").strip(),
                   document=(r.get(document_key) or "").strip() if document_key else "")
        for r in reader
    ]
    rows = [r for r in rows if r.question]
    if not rows:
        raise InvalidInput("The dataset has no usable rows.")
    if len(rows) > MAX_QUESTIONS:
        raise InvalidInput(
            f"The dataset has {len(rows)} questions, over the "
            f"{MAX_QUESTIONS} limit for one run."
        )
    return rows


def default_dataset() -> list[DatasetRow]:
    if not DEFAULT_DATASET.exists():
        raise InvalidInput("No sample dataset is bundled with this project.")
    return parse_dataset(DEFAULT_DATASET.read_bytes())


def corpus_from(rows: list[DatasetRow]) -> str:
    """Join every row's document into the single context CAG needs.

    This is the whole premise made concrete: all ten documents go into the
    context at once, and all ten questions are answered from one cache.
    """
    parts = [r.document for r in rows if r.document]
    if not parts:
        raise InvalidInput(
            "This dataset carries no document text, so there is nothing to "
            "cache. Paste a document instead."
        )
    return "\n\n---\n\n".join(parts)


def default_corpus() -> str:
    return corpus_from(default_dataset())


def load(model_id: str, quantized: bool = False) -> LoadedModel:
    """Load a model, reusing it if the same one is already loaded."""
    if model_id not in MODEL_IDS:
        raise InvalidInput(
            f"Unknown model: {model_id}. Choose one of: {', '.join(MODEL_IDS)}"
        )
    token = hf_token()
    if is_gated(model_id) and not token:
        raise ModelLoadError(
            f"{model_id} is a gated repository and HF_TOKEN is not set. "
            f"Either set it and accept the licence at "
            f"https://huggingface.co/{model_id}, or choose one of the ungated "
            f"models, which need no token."
        )

    with _lock:
        if (_state.loaded and _state.loaded.model_id == model_id
                and _state.loaded.quantized == quantized):
            return _state.loaded

        model = CAGModel(hf_token=token or "")
        model.load_model(model_id, quantized)      # raises ModelLoadError
        _state.loaded = LoadedModel(model_id=model_id, quantized=quantized,
                                    model=model)
        return _state.loaded


def require_loaded() -> LoadedModel:
    if _state.loaded is None:
        raise NotLoaded("No model is loaded. Load one first.")
    return _state.loaded


def run_experiment(document: str, rows: list[DatasetRow],
                   use_cache: bool = True) -> dict:
    """Answer every question, with or without the KV cache, and time it."""
    text = validate_document(document)
    loaded = require_loaded()

    dataset = [(row.question, row.answer) for row in rows]
    results = loaded.model.process_questions(dataset, text, use_cache=use_cache)

    return {
        "model": loaded.model_id,
        "quantized": loaded.quantized,
        "use_cache": use_cache,
        "questions": len(dataset),
        "document_chars": len(text),
        "avg_similarity": round(results.avg_similarity, 4),
        "avg_cache_time": round(results.avg_cache_time, 4),
        "avg_generate_time": round(results.avg_generate_time, 4),
        "cache_build_time": round(getattr(results, "prepare_time", 0.0), 4),
        "details": getattr(results, "results", []),
    }


def status() -> dict:
    loaded = _state.loaded
    return {
        "loaded": loaded is not None,
        "model": loaded.model_id if loaded else None,
        "quantized": loaded.quantized if loaded else False,
        "hf_token": bool(hf_token()),
    }


def unload() -> None:
    with _lock:
        _state.loaded = None
