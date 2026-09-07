"""FastAPI routing layer for the graph QA chatbot."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import config
from services.qa_service import GraphUnavailable, ProviderError, ask, graph_schema

app = FastAPI(
    title="Graph QA API",
    description=(
        "Ask a Neo4j graph questions in English. The model writes Cypher, the "
        "Cypher is checked for writes, and only then does it run. The "
        "Streamlit UI in `ui/` is a client of this API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    providers: list[str]
    graph: str


class CatalogueResponse(BaseModel):
    providers: list[str]
    models: dict[str, list[str]]
    key_urls: dict[str, str]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    provider: str = Field(..., examples=["Gemini"])
    model: str = Field(..., examples=["gemini-flash-latest"])


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        graph_schema()
        graph = "ok"
    except GraphUnavailable as exc:
        graph = f"unavailable: {str(exc).splitlines()[0]}"
    return HealthResponse(
        status="ok", providers=config.available_providers(), graph=graph
    )


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    available = config.available_providers()
    return CatalogueResponse(
        providers=available,
        models={p: config.models_for(p) for p in available},
        key_urls={p: config.PROVIDERS[p]["key_url"] for p in config.PROVIDERS},
    )


@app.get("/schema", tags=["graph"])
def schema() -> dict:
    """The graph schema the model is shown when it writes Cypher."""
    try:
        return {"schema": graph_schema()}
    except GraphUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask", tags=["query"])
def ask_question(request: AskRequest) -> dict:
    if request.provider not in config.PROVIDERS:
        raise HTTPException(
            status_code=400, detail=f"Unknown provider: {request.provider}"
        )
    try:
        answer = ask(request.question, request.provider, request.model)
    except GraphUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return answer.__dict__
