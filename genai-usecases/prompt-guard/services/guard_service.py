"""Classify text as a prompt-injection attempt, or not.

Two backends behind one interface: Meta's Prompt Guard 2 hosted on Groq, or
the same model running locally through transformers.

The one rule this module takes seriously: **a classifier that cannot classify
must not answer "safe".** The original Groq path did

    try:
        score = float(response_content)
    except ValueError:
        score = 0.0            # -> below threshold -> reported BENIGN

so an unparseable response, a truncated response, or a changed output format
all became a confident "this text is benign". A guardrail that fails open is
worse than no guardrail, because it is trusted. Here that raises
:class:`ClassifierError` and the API returns 502.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from services.config import (
    BACKENDS,
    MAX_TEXT_LENGTH,
    api_key_for,
    model_for,
)


class BackendError(RuntimeError):
    """Unknown backend, missing key, or a missing package."""


class ClassifierError(RuntimeError):
    """The classifier ran but its answer could not be trusted."""


class InvalidText(ValueError):
    """The text is empty or too long to classify."""


@dataclass
class Verdict:
    text_length: int
    backend: str
    model: str
    score: float
    threshold: float
    is_malicious: bool
    label: str
    inference_time_ms: float


def validate_text(text: str | None) -> str:
    stripped = (text or "").strip()
    if not stripped:
        raise InvalidText("The text is empty.")
    if len(stripped) > MAX_TEXT_LENGTH:
        raise InvalidText(
            f"The text is too long ({len(stripped)} characters, "
            f"maximum {MAX_TEXT_LENGTH})."
        )
    return stripped


def _require(backend: str, size: str) -> tuple[str, str]:
    if backend not in BACKENDS:
        raise BackendError(
            f"Unknown backend: {backend}. Choose one of: {', '.join(BACKENDS)}."
        )
    model = model_for(backend, size)
    if not model:
        raise BackendError(
            f"Unknown model size: {size}. Choose 22M or 86M."
        )
    key = api_key_for(backend)
    if not key:
        spec = BACKENDS[backend]
        raise BackendError(
            f"{spec['env_key']} is not set. Add it to your .env file "
            f"(get one at {spec['key_url']}), then restart the API."
        )
    return model, key


def parse_score(raw: object) -> float:
    """Turn the backend's answer into a probability, or refuse it.

    Returning a default here is what made the original fail open. If the
    answer is not a number in [0, 1], the honest outcome is "I do not know",
    not "benign".
    """
    if raw is None:
        raise ClassifierError(
            "The classifier returned no score. Refusing to report this text "
            "as safe on no evidence."
        )
    try:
        score = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ClassifierError(
            f"The classifier returned {raw!r}, which is not a score. "
            f"Refusing to report this text as safe on no evidence."
        ) from exc
    if not 0.0 <= score <= 1.0:
        raise ClassifierError(
            f"The classifier returned {score}, which is outside [0, 1]."
        )
    return score


def classify_with_groq(text: str, size: str, threshold: float) -> Verdict:
    model, key = _require("groq", size)
    try:
        from groq import Groq
    except ImportError as exc:
        raise BackendError("The groq package is not installed. "
                           "Run: pip install groq") from exc

    client = Groq(api_key=key)
    started = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=1, max_tokens=1, top_p=1, stream=False,
        )
    except Exception as exc:
        raise ClassifierError(f"The Groq request failed: {exc}") from exc
    elapsed = (time.time() - started) * 1000

    score = parse_score(completion.choices[0].message.content)
    return _verdict(text, "groq", model, score, threshold, elapsed)


@lru_cache(maxsize=2)
def _load_local(model: str, token: str):
    """Load the local classifier once. Downloads on first use."""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as exc:
        raise BackendError(
            "transformers and torch are required for the local backend. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        weights = AutoModelForSequenceClassification.from_pretrained(
            model, token=token
        )
    except Exception as exc:
        raise BackendError(
            f"Could not load {model}: {exc}\n"
            f"These repos are gated: accept the licence at "
            f"https://huggingface.co/{model} with the same account as your "
            f"HF_TOKEN."
        ) from exc

    device = -1
    try:
        import torch

        if torch.cuda.is_available():
            device = 0
    except ImportError:
        pass

    return pipeline("text-classification", model=weights,
                    tokenizer=tokenizer, device=device)


def classify_with_local(text: str, size: str, threshold: float) -> Verdict:
    model, token = _require("huggingface", size)
    classifier = _load_local(model, token)

    started = time.time()
    try:
        result = classifier(text)[0]
    except Exception as exc:
        raise ClassifierError(f"Local classification failed: {exc}") from exc
    elapsed = (time.time() - started) * 1000

    # The pipeline reports whichever class won, with its confidence. Convert
    # that to the probability of the malicious class either way, so the
    # threshold means one thing regardless of which label came back.
    label = str(result.get("label", "")).upper()
    confidence = parse_score(result.get("score"))
    score = confidence if "MALICIOUS" in label or label == "LABEL_1" \
        else 1.0 - confidence

    return _verdict(text, "huggingface", model, score, threshold, elapsed)


def _verdict(text: str, backend: str, model: str, score: float,
             threshold: float, elapsed_ms: float) -> Verdict:
    is_malicious = score > threshold
    return Verdict(
        text_length=len(text),
        backend=backend,
        model=model,
        score=round(score, 4),
        threshold=threshold,
        is_malicious=is_malicious,
        label="MALICIOUS" if is_malicious else "BENIGN",
        inference_time_ms=round(elapsed_ms, 2),
    )


def classify(text: str, backend: str, size: str = "22M",
             threshold: float | None = None) -> Verdict:
    """Classify one piece of text. Raises rather than guessing."""
    clean = validate_text(text)
    if backend not in BACKENDS:
        raise BackendError(
            f"Unknown backend: {backend}. Choose one of: {', '.join(BACKENDS)}."
        )
    if threshold is None:
        threshold = BACKENDS[backend]["default_threshold"]

    if backend == "groq":
        return classify_with_groq(clean, size, threshold)
    return classify_with_local(clean, size, threshold)
