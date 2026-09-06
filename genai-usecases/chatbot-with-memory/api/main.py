"""FastAPI routing layer for the PDF chatbot."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import config
from services.rag_service import Document, IngestError, ProviderError
from services.session_store import SessionNotFound, store

app = FastAPI(
    title="PDF Chat Bot API",
    description=(
        "Upload PDFs, then ask questions about them with conversation memory. "
        "The Streamlit UI in `ui/` is a client of this API."
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


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ProviderInfo(BaseModel):
    id: str
    models: list[str]
    model_help: str
    key_url: str
    embeddings: str


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
    defaults: dict


class HealthResponse(BaseModel):
    status: str
    providers: list[str]
    sessions: int
    max_sessions: int


class SessionResponse(BaseModel):
    session_id: str
    provider: str
    model: str
    filenames: list[str]
    pages: int
    chunks: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    retriever_k: int | None = Field(None, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    history: list[dict]


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", providers=config.available_providers(), **store.stats()
    )


@app.get("/providers", response_model=ProvidersResponse, tags=["meta"])
def providers() -> ProvidersResponse:
    return ProvidersResponse(
        providers=[
            ProviderInfo(
                id=p,
                models=config.models_for(p),
                model_help=config.PROVIDERS[p]["model_help"],
                key_url=config.PROVIDERS[p]["key_url"],
                embeddings=config.PROVIDERS[p]["embeddings"],
            )
            for p in config.available_providers()
        ],
        defaults=config.DEFAULTS,
    )


@app.post("/sessions", response_model=SessionResponse, tags=["sessions"])
async def create_session(
    files: list[UploadFile] = File(...),
    provider: str = Form(...),
    model: str = Form(...),
    temperature: float = Form(0.3),
    chunk_size: int = Form(2000),
    chunk_overlap: int = Form(200),
    retriever_k: int = Form(3),
) -> SessionResponse:
    """Index the uploaded PDFs and open a chat session over them."""
    if provider not in config.PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=422,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    documents = [
        Document(filename=f.filename or "upload.pdf", content=await f.read())
        for f in files
    ]

    try:
        session = store.create(
            documents, provider, model, temperature,
            chunk_size, chunk_overlap, retriever_k,
        )
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SessionResponse(
        session_id=session.id, provider=session.provider, model=session.model,
        filenames=session.ingest.filenames, pages=session.ingest.pages,
        chunks=session.ingest.chunks,
    )


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse, tags=["chat"])
def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    try:
        answer = store.ask(
            session_id, request.question, request.temperature, request.retriever_k
        )
    except SessionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found. It may have expired; upload your PDFs again.",
        ) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc

    return ChatResponse(answer=answer, history=store.get(session_id).messages())


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse, tags=["chat"])
def history(session_id: str) -> HistoryResponse:
    try:
        return HistoryResponse(
            session_id=session_id, history=store.get(session_id).messages()
        )
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


@app.delete("/sessions/{session_id}/history", status_code=204, tags=["chat"])
def clear_history(session_id: str) -> None:
    """Forget the conversation but keep the indexed documents."""
    try:
        store.clear_history(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


@app.delete("/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(session_id: str) -> None:
    try:
        store.delete(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
