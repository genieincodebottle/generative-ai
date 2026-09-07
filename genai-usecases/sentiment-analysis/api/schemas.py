"""Request and response models: the contract between UI and service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    id: str
    label: str
    models: list[str]
    default_model: str
    key_url: str


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
    default_provider: str | None = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000,
                      examples=["Customer was angry about the wrong order."])
    provider: str = Field(..., examples=["gemini"])
    model: str = Field(..., examples=["gemini-flash-latest"])
    temperature: float = Field(0.0, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    sentiment: str
    aggressiveness: int


class TagAllRequest(BaseModel):
    provider: str
    model: str
    temperature: float = Field(0.0, ge=0.0, le=1.0)


class TagOutcomeModel(BaseModel):
    call_id: int
    success: bool
    sentiment: str | None = None
    aggressiveness: int | None = None
    error: str | None = None


class TagAllResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    outcomes: list[TagOutcomeModel]


class CallModel(BaseModel):
    id: int
    customer_id: int
    call_details: str
    call_time: str | None = None


class TaggingModel(BaseModel):
    call_id: int
    customer_id: int
    call_details: str
    sentiment: str
    aggressiveness: int
    tagged_at: str | None = None
    call_time: str | None = None


class StatsResponse(BaseModel):
    total: int
    sentiment_counts: dict[str, int]
    average_aggressiveness: float
    max_aggressiveness: int
    high_aggression_calls: int


class DatabaseStatusResponse(BaseModel):
    ready: bool
    calls: int = 0
    tagged: int = 0


class InitResponse(BaseModel):
    initialized: bool
    seeded: int


class HealthResponse(BaseModel):
    status: str
    database_ready: bool
    providers: list[str]
