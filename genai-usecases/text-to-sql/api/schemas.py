"""Request and response models. These are the contract between UI and service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          examples=["Which country's customers spent the most?"])
    provider: str = Field(..., examples=["Google Gemini"])
    model: str = Field(..., examples=["gemini-flash-latest"])
    temperature: float = Field(0.0, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    success: bool
    question: str
    sql_query: str | None = None
    raw_result: str | None = None
    answer: str | None = None
    error: str | None = None
    columns: list[str] = []
    rows: list[list] = []


class ProvidersResponse(BaseModel):
    """Which providers have a key configured, and the models each offers."""

    providers: list[str]
    models: dict[str, list[str]]


class DatabaseInfoResponse(BaseModel):
    dialect: str
    tables: list[str]


class TablePreviewResponse(BaseModel):
    table: str
    columns: list[str]
    rows: list[list]


class HealthResponse(BaseModel):
    status: str
    database: str
    providers: list[str]
