"""FastAPI routing layer for the Agentic RAG system."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import config
from services.system_manager import IngestError, SystemNotReady, Upload, manager

app = FastAPI(
    title="Agentic RAG API",
    description=(
        "A multi-agent RAG pipeline: plan, retrieve, research the web, "
        "synthesise, validate. The Streamlit UI in `ui/` is a client of this API."
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


class HealthResponse(BaseModel):
    status: str
    google_key: bool
    tavily_key: bool
    configured: bool


class ModelsResponse(BaseModel):
    llm_models: list[str]
    embedding_models: list[str]
    web_search_available: bool


class ConfigureRequest(BaseModel):
    llm_model: str = "gemini-flash-latest"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(8192, ge=256, le=32768)
    chunk_size: int = Field(1000, ge=200, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=2000)
    k_retrieval: int = Field(8, ge=1, le=50)
    max_iterations: int = Field(10, ge=1, le=50)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    enable_web_search: bool = True
    max_web_results: int = Field(5, ge=1, le=20)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = None


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        google_key=config.has_google_key(),
        tavily_key=config.has_tavily_key(),
        configured=manager.is_configured,
    )


@app.get("/models", response_model=ModelsResponse, tags=["meta"])
def models() -> ModelsResponse:
    return ModelsResponse(
        llm_models=config.LLM_MODELS,
        embedding_models=config.EMBEDDING_MODELS,
        # Web search is optional. Saying so up front is better than a
        # research step that quietly returns nothing.
        web_search_available=config.has_tavily_key(),
    )


@app.get("/status", tags=["meta"])
def status() -> dict:
    return manager.status()


@app.post("/configure", tags=["system"])
def configure(request: ConfigureRequest) -> dict:
    """Build or rebuild the system. Indexed documents in Chroma survive this."""
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=422,
            detail="chunk_overlap must be smaller than chunk_size.",
        )
    try:
        manager.configure(**request.model_dump())
    except SystemNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not initialize the system: {exc}"
        ) from exc
    return manager.status()


@app.post("/documents", tags=["documents"])
async def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    uploads = [
        Upload(filename=f.filename or "upload.pdf", content=await f.read())
        for f in files
    ]
    try:
        result = manager.load_documents(uploads)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SystemNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"filenames": result.filenames, "loaded": result.loaded}


@app.post("/query", tags=["query"])
def query(request: QueryRequest) -> dict:
    """Run the full agent pipeline over the indexed documents."""
    try:
        return manager.query(request.question, request.thread_id)
    except SystemNotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc
