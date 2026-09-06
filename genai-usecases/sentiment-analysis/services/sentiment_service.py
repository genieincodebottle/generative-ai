"""The business logic: classify a call, and tag calls in bulk.

Imports no web framework. Every function here is callable from a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from services import database
from services.config import (
    MAX_CALL_LENGTH,
    MIN_CALL_LENGTH,
    PROVIDERS,
    api_key_for,
)

VALID_SENTIMENTS = ("Positive", "Negative", "Neutral")


class LLMError(RuntimeError):
    """Provider unknown, or its API key is missing."""


class InvalidCallText(ValueError):
    """The call text is empty, too short, or too long."""


class Classification(BaseModel):
    """The structure the model is asked to return."""

    sentiment: str = Field(
        description="The sentiment of the text (Positive, Negative, or Neutral). "
                    "If no clear sentiment is detected, use 'Neutral'."
    )
    aggressiveness: int = Field(
        description="How aggressive the text is, 1 to 10. 1 is not aggressive "
                    "at all, 10 is extremely aggressive. Default to 1.",
        ge=1, le=10,
    )


TAGGING_PROMPT = PromptTemplate.from_template("""
You are a customer service analyst. Analyze the following customer call transcript for sentiment and aggressiveness.

Customer call transcript:
"{input}"

Analysis Guidelines:

SENTIMENT:
- Positive: Customer is happy, satisfied, appreciative, or expressing gratitude
- Negative: Customer is upset, frustrated, disappointed, or complaining
- Neutral: Customer is calm, matter-of-fact, or simply requesting information

AGGRESSIVENESS (1-10 scale):
- 1-2: Very polite, calm, respectful tone
- 3-4: Slightly concerned but still polite
- 5-6: Moderately frustrated, some impatience evident
- 7-8: Clearly angry, demanding, using strong language
- 9-10: Very hostile, threatening, extremely rude or abusive

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no reasoning, no extra text.

{format_instructions}
""")


@dataclass
class TagOutcome:
    """One call's result. ``error`` is set only when ``success`` is False."""

    call_id: int
    success: bool
    sentiment: str | None = None
    aggressiveness: int | None = None
    error: str | None = None


@lru_cache(maxsize=8)
def get_llm(provider: str, model: str, temperature: float = 0.0) -> BaseChatModel:
    key = api_key_for(provider)
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider: {provider}")
    if not key:
        raise LLMError(
            f"{PROVIDERS[provider]['env_key']} is not set. "
            f"Add it to your .env file and restart the API."
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, api_key=key)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, google_api_key=key
    )


def validate_call_text(text: str | None) -> str:
    """Return the stripped text, or raise :class:`InvalidCallText`."""
    stripped = (text or "").strip()
    if not stripped:
        raise InvalidCallText("The call text is empty.")
    if len(stripped) < MIN_CALL_LENGTH:
        raise InvalidCallText(
            f"The call text is too short (minimum {MIN_CALL_LENGTH} characters)."
        )
    if len(stripped) > MAX_CALL_LENGTH:
        raise InvalidCallText(
            f"The call text is too long (maximum {MAX_CALL_LENGTH} characters)."
        )
    return stripped


def normalise_result(raw: dict) -> dict:
    """Coerce the model's JSON into the shape the rest of the app expects.

    Models return ``"positive"``, ``"POSITIVE"``, and occasionally a string
    where an int belongs. The database column is not going to fix that, so it
    is fixed here rather than at every read site.
    """
    sentiment = str(raw.get("sentiment", "Neutral")).strip().capitalize()
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "Neutral"

    try:
        aggressiveness = int(float(raw.get("aggressiveness", 1)))
    except (TypeError, ValueError):
        aggressiveness = 1
    aggressiveness = max(1, min(10, aggressiveness))

    return {"sentiment": sentiment, "aggressiveness": aggressiveness}


def analyze(text: str, provider: str, model: str, temperature: float = 0.0) -> dict:
    """Classify one piece of call text. Raises on invalid input or LLM failure."""
    stripped = validate_call_text(text)
    llm = get_llm(provider, model, temperature)
    parser = JsonOutputParser(pydantic_object=Classification)
    raw = (TAGGING_PROMPT | llm | parser).invoke({
        "input": stripped,
        "format_instructions": parser.get_format_instructions(),
    })
    return normalise_result(raw)


def tag_all_calls(provider: str, model: str, temperature: float = 0.0,
                  db_path: str | None = None) -> list[TagOutcome]:
    """Classify and store every call. One failure does not stop the batch."""
    outcomes: list[TagOutcome] = []
    for call in database.fetch_calls(db_path):
        call_id = call["id"]
        try:
            result = analyze(call["call_details"], provider, model, temperature)
            database.save_tagging(
                call_id, call["call_details"],
                result["sentiment"], result["aggressiveness"], db_path,
            )
            outcomes.append(TagOutcome(call_id, True, **result))
        except Exception as exc:
            outcomes.append(TagOutcome(call_id, False, error=str(exc)))
    return outcomes


def statistics(db_path: str | None = None) -> dict:
    """Summary numbers over everything tagged so far."""
    results = database.fetch_taggings(db_path)
    if not results:
        return {
            "total": 0, "sentiment_counts": {},
            "average_aggressiveness": 0.0, "max_aggressiveness": 0,
            "high_aggression_calls": 0,
        }

    scores = [r["aggressiveness"] for r in results]
    sentiment_counts: dict[str, int] = {}
    for r in results:
        sentiment_counts[r["sentiment"]] = sentiment_counts.get(r["sentiment"], 0) + 1

    return {
        "total": len(results),
        "sentiment_counts": sentiment_counts,
        "average_aggressiveness": round(sum(scores) / len(scores), 2),
        "max_aggressiveness": max(scores),
        "high_aggression_calls": sum(1 for s in scores if s >= 7),
    }
