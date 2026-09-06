"""FastAPI routing layer.

Validates input, calls ``services``, maps exceptions to status codes. No
business logic lives here.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CallModel,
    DatabaseStatusResponse,
    HealthResponse,
    InitResponse,
    ProviderInfo,
    ProvidersResponse,
    StatsResponse,
    TagAllRequest,
    TagAllResponse,
    TaggingModel,
)
from services import config, database, sentiment_service
from services.sentiment_service import InvalidCallText, LLMError

app = FastAPI(
    title="Customer Call Sentiment API",
    description=(
        "Classifies customer call transcripts for sentiment and "
        "aggressiveness. The Streamlit UI in `ui/` is a client of this API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        database_ready=database.is_ready(),
        providers=config.available_providers(),
    )


@app.get("/providers", response_model=ProvidersResponse, tags=["meta"])
def providers() -> ProvidersResponse:
    """Only providers whose key is actually set, so no dead dropdown options."""
    available = config.available_providers()
    return ProvidersResponse(
        providers=[
            ProviderInfo(
                id=p,
                label=config.PROVIDERS[p]["label"],
                models=config.models_for(p),
                default_model=config.default_model_for(p),
                key_url=config.PROVIDERS[p]["key_url"],
            )
            for p in available
        ],
        default_provider=(
            config.DEFAULT_PROVIDER if config.DEFAULT_PROVIDER in available
            else (available[0] if available else None)
        ),
    )


@app.get("/database", response_model=DatabaseStatusResponse, tags=["database"])
def database_status() -> DatabaseStatusResponse:
    if not database.is_ready():
        return DatabaseStatusResponse(ready=False)
    return DatabaseStatusResponse(ready=True, **database.counts())


@app.post("/database/init", response_model=InitResponse, tags=["database"])
def database_init() -> InitResponse:
    try:
        return InitResponse(**database.initialize())
    except database.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/database", status_code=204, tags=["database"])
def database_reset() -> None:
    try:
        database.reset()
    except database.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/calls", response_model=list[CallModel], tags=["calls"])
def calls() -> list[CallModel]:
    try:
        return [CallModel(**c) for c in database.fetch_calls()]
    except database.DatabaseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/taggings", response_model=list[TaggingModel], tags=["results"])
def taggings() -> list[TaggingModel]:
    try:
        return [
            TaggingModel(**{k: v for k, v in row.items() if k != "user_id"})
            for row in database.fetch_taggings()
        ]
    except database.DatabaseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/stats", response_model=StatsResponse, tags=["results"])
def stats() -> StatsResponse:
    try:
        return StatsResponse(**sentiment_service.statistics())
    except database.DatabaseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Classify one transcript without touching the database."""
    try:
        return AnalyzeResponse(**sentiment_service.analyze(
            request.text, request.provider, request.model, request.temperature
        ))
    except InvalidCallText as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc


@app.post("/taggings/run", response_model=TagAllResponse, tags=["analysis"])
def tag_all(request: TagAllRequest) -> TagAllResponse:
    """Classify and store every call.

    One call failing does not abort the batch; each outcome is reported
    individually so a partial run is visible rather than silent.
    """
    if request.provider not in config.PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    if not config.api_key_for(request.provider):
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured for {request.provider}.",
        )
    try:
        outcomes = sentiment_service.tag_all_calls(
            request.provider, request.model, request.temperature
        )
    except database.DatabaseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    succeeded = sum(1 for o in outcomes if o.success)
    return TagAllResponse(
        total=len(outcomes),
        succeeded=succeeded,
        failed=len(outcomes) - succeeded,
        outcomes=[o.__dict__ for o in outcomes],
    )
