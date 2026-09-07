"""Chat, vision, and the Gemini fallback.

The original returned its errors as ordinary strings::

    if not GROQ_API_KEY:
        return "Groq API key not set."
    except Exception as e:
        return f"Error connecting to Groq API: {str(e)}"

The caller could not tell those apart from an answer, so they were rendered in
the chat window as though the model had said them. Here they raise, and the
API turns them into status codes.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from services.config import (
    GROQ_MODELS,
    GROQ_VISION_MODELS,
    MAX_IMAGE_BYTES,
    api_key,
)

ALLOWED_IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpeg",
                       "image/jpg": "jpeg", "image/webp": "webp"}


class MissingKey(RuntimeError):
    """A required API key is not configured."""


class ModelError(RuntimeError):
    """The provider was reached but the request failed."""


class InvalidInput(ValueError):
    """The request was malformed before any provider was called."""


@dataclass
class Reply:
    text: str
    model: str
    provider: str
    fallback_used: bool = False
    fallback_reason: str | None = None


def _groq_client():
    key = api_key("groq")
    if not key:
        raise MissingKey(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "(get one at https://console.groq.com/keys), then restart the API."
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise ModelError("The groq package is not installed. "
                         "Run: pip install -r requirements.txt") from exc
    return Groq(api_key=key)


def chat(messages: list[dict], model: str, temperature: float = 0.7,
         max_tokens: int = 1024) -> Reply:
    """Text chat through Groq."""
    if not messages:
        raise InvalidInput("No messages were provided.")
    if model not in GROQ_MODELS:
        raise InvalidInput(f"Unknown model: {model}")

    client = _groq_client()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, top_p=1,
        )
    except Exception as exc:
        raise ModelError(f"The Groq request failed: {exc}") from exc

    text = response.choices[0].message.content
    if not text or not text.strip():
        # An empty completion is a failure, not an answer. Returning it would
        # render as the model having said nothing at all.
        raise ModelError("The model returned an empty response.")
    return Reply(text=text, model=model, provider="groq")


def encode_image(content: bytes, content_type: str) -> tuple[str, str]:
    """Validate an uploaded image and return (base64, media type)."""
    media_type = content_type.lower().split(";")[0].strip()
    if media_type not in ALLOWED_IMAGE_TYPES:
        raise InvalidInput(
            f"{media_type or 'unknown'} is not a supported image type "
            f"({', '.join(sorted(ALLOWED_IMAGE_TYPES))})."
        )
    if not content:
        raise InvalidInput("The image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise InvalidInput(
            f"The image is too large ({len(content) / 1e6:.1f} MB, "
            f"maximum {MAX_IMAGE_BYTES / 1e6:.0f} MB)."
        )

    # Round-trip through Pillow so a mislabelled or corrupt file is caught
    # here rather than by the provider.
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(content))
        image.verify()
    except Exception as exc:
        raise InvalidInput(f"That file is not a readable image: {exc}") from exc

    return base64.b64encode(content).decode("utf-8"), media_type


def describe_image(content: bytes, content_type: str, prompt: str,
                   model: str) -> Reply:
    """Vision: read or describe an image."""
    if model not in GROQ_VISION_MODELS:
        raise InvalidInput(
            f"{model} does not accept images. Choose one of: "
            f"{', '.join(GROQ_VISION_MODELS)}."
        )
    if not prompt or not prompt.strip():
        raise InvalidInput("The prompt is empty.")

    encoded, media_type = encode_image(content, content_type)
    client = _groq_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.strip()},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{encoded}"}},
                ],
            }],
            temperature=1, max_tokens=1024, top_p=1,
        )
    except Exception as exc:
        raise ModelError(f"The Groq vision request failed: {exc}") from exc

    text = response.choices[0].message.content
    if not text or not text.strip():
        raise ModelError("The model returned an empty response.")
    return Reply(text=text, model=model, provider="groq")


def gemini_fallback(prompt: str, model: str = "gemini-flash-latest") -> Reply:
    """Answer with Gemini when the primary model fails.

    The fallback is reported, never disguised: the caller always learns which
    provider actually produced the text.
    """
    key = api_key("google")
    if not key:
        raise MissingKey(
            "GOOGLE_API_KEY is not set, so there is no fallback available."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        from services.llm_text import message_text

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=key,
                                     temperature=0.7)
        text = message_text(llm.invoke(prompt))
    except Exception as exc:
        raise ModelError(f"The Gemini fallback also failed: {exc}") from exc

    if not text.strip():
        raise ModelError("The Gemini fallback returned an empty response.")
    return Reply(text=text, model=model, provider="google", fallback_used=True)
