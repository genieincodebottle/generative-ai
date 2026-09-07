"""FastAPI routing layer.

Routes validate input, call into ``services``, and translate exceptions into
HTTP status codes. There is no business logic in this file by design: if you
find yourself writing an ``if`` about SQL here, it belongs in ``services``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    DatabaseInfoResponse,
    HealthResponse,
    ProvidersResponse,
    QueryRequest,
    QueryResponse,
    TablePreviewResponse,
)
from services import config, database, sql_service

app = FastAPI(
    title="Text-to-SQL API",
    description=(
        "Converts natural language questions into SQL, runs them against the "
        "bundled Chinook database, and explains the result. The Streamlit UI "
        "in `ui/` is a client of this API."
    ),
    version="1.0.0",
)

# The Streamlit UI is a separate origin, so it needs CORS. This is a local
# demo, hence the permissive default; narrow it before exposing the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness plus the two things that actually stop this app working."""
    try:
        database.get_database()
        db_status = "ok"
    except database.DatabaseError as exc:
        db_status = f"unavailable: {exc}"
    return HealthResponse(
        status="ok", database=db_status, providers=config.available_providers()
    )


@app.get("/providers", response_model=ProvidersResponse, tags=["meta"])
def providers() -> ProvidersResponse:
    """Providers with a key configured, so the UI never offers a dead option."""
    available = config.available_providers()
    return ProvidersResponse(
        providers=available,
        models={p: config.models_for(p) for p in available},
    )


@app.get("/database", response_model=DatabaseInfoResponse, tags=["database"])
def database_info() -> DatabaseInfoResponse:
    try:
        return DatabaseInfoResponse(**sql_service.database_info())
    except database.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/database/tables/{table}", response_model=TablePreviewResponse,
         tags=["database"])
def table_preview(table: str, limit: int = Query(5, ge=1, le=100)) -> TablePreviewResponse:
    try:
        frame = database.preview_table(table, limit)
    except database.DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TablePreviewResponse(
        table=table,
        columns=[str(c) for c in frame.columns],
        rows=frame.astype(object).where(frame.notna(), None).values.tolist(),
    )


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest) -> QueryResponse:
    """Ask a question in English; get SQL, rows, and an explanation back.

    A model or database failure is reported as ``success: false`` with 200,
    not as a 5xx: the request itself was valid, and the UI renders the error.
    """
    if request.provider not in config.PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    if not config.api_key_for(request.provider):
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured for {request.provider}.",
        )

    result = sql_service.answer_question(
        question=request.question,
        provider=request.provider,
        model=request.model,
        temperature=request.temperature,
    )
    return QueryResponse(**result.__dict__)
