"""FastAPI routing layer for the RAG techniques comparison."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import config, documents
from services.documents import IngestError, Upload
from services.llm import ProviderError, get_llm
from services.store import SessionNotFound, store
from services.techniques import TechniqueError, run

app = FastAPI(
    title="RAG Techniques API",
    description=(
        "Five retrieval strategies over one index: basic, adaptive, "
        "corrective, hybrid, and re-ranking. Because they share the index, "
        "the comparison between them is fair. The Streamlit UI in `ui/` is a "
        "client of this API."
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
    sessions: int
    max_sessions: int


class TechniqueInfo(BaseModel):
    id: str
    label: str
    blurb: str
    options: list[str]


class CatalogueResponse(BaseModel):
    providers: list[str]
    models: dict[str, list[str]]
    embeddings: dict[str, str]
    key_urls: dict[str, str]
    techniques: list[TechniqueInfo]
    rerankers: list[str]
    defaults: dict


class SessionResponse(BaseModel):
    session_id: str
    provider: str
    filenames: list[str]
    total_chunks: int
    avg_chunk_length: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    technique: str = Field(..., examples=["basic"])
    model: str = Field(..., examples=["gemini-flash-latest"])
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    top_k: int = Field(4, ge=1, le=20)
    bm25_weight: float = Field(0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(0.5, ge=0.0, le=1.0)
    reranker: str = Field("Embeddings Filter")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", providers=config.available_providers(), **store.stats()
    )


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    """Everything the UI needs to build its controls, in one call."""
    available = config.available_providers()
    return CatalogueResponse(
        providers=available,
        models={p: config.models_for(p) for p in available},
        embeddings={p: config.PROVIDERS[p]["embeddings"] for p in available},
        key_urls={p: config.PROVIDERS[p]["key_url"] for p in config.PROVIDERS},
        techniques=[
            TechniqueInfo(id=k, label=v["label"], blurb=v["blurb"],
                          options=v["options"])
            for k, v in config.TECHNIQUES.items()
        ],
        rerankers=config.RERANKERS,
        defaults=config.DEFAULTS,
    )


@app.post("/sessions", response_model=SessionResponse, tags=["sessions"])
async def create_session(
    files: list[UploadFile] = File(default=[]),
    provider: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    use_samples: bool = Form(False),
) -> SessionResponse:
    """Index documents once; every technique then queries the same index."""
    if provider not in config.PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    try:
        if use_samples:
            uploads = documents.sample_uploads()
        else:
            uploads = [
                Upload(filename=f.filename or "upload.pdf", content=await f.read())
                for f in files
            ]
        session = store.create(uploads, provider, chunk_size, chunk_overlap)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SessionResponse(
        session_id=session.id, provider=session.provider,
        filenames=session.filenames,
        total_chunks=session.stats["total_chunks"],
        avg_chunk_length=session.stats["avg_chunk_length"],
    )


@app.delete("/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(session_id: str) -> None:
    try:
        store.delete(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


@app.post("/sessions/{session_id}/query", tags=["query"])
def query(session_id: str, request: QueryRequest) -> dict:
    """Run one technique against the session's index."""
    if request.technique not in config.TECHNIQUES:
        raise HTTPException(
            status_code=400, detail=f"Unknown technique: {request.technique}"
        )
    try:
        session = store.get(session_id)
    except SessionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found. It may have expired; index your documents again.",
        ) from exc

    try:
        llm = get_llm(session.provider, request.model, request.temperature)
        return run(
            request.technique, session.index, llm, request.query,
            top_k=request.top_k,
            bm25_weight=request.bm25_weight,
            vector_weight=request.vector_weight,
            reranker=request.reranker,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TechniqueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc
