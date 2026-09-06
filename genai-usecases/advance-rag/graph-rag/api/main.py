"""FastAPI routing layer for Graph RAG."""

from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.graph_rag import MissingAPIKey
from services.manager import (
    LLM_MODELS,
    RETRIEVERS,
    IngestError,
    NotReady,
    Upload,
    manager,
)

app = FastAPI(
    title="Graph RAG API",
    description=(
        "Retrieval that walks the edges between documents, not just the "
        "distance between their embeddings. The Streamlit UI in `ui/` is a "
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
    google_key: bool
    ready: bool


class CatalogueResponse(BaseModel):
    llm_models: list[str]
    retrievers: list[dict]
    defaults: dict


class ConfigureRequest(BaseModel):
    llm_model: str = "gemini-flash-latest"
    chunk_size: int = Field(1000, ge=200, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=2000)
    k_retrieval: int = Field(5, ge=1, le=50)
    max_depth: int = Field(2, ge=1, le=5)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    retriever: str = Field("traversal")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        google_key=bool(os.getenv("GOOGLE_API_KEY")),
        ready=manager.is_ready,
    )


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    return CatalogueResponse(
        llm_models=LLM_MODELS,
        retrievers=RETRIEVERS,
        defaults={"chunk_size": 1000, "chunk_overlap": 200,
                  "k_retrieval": 5, "max_depth": 2},
    )


@app.get("/status", tags=["meta"])
def status() -> dict:
    return manager.status()


@app.post("/configure", tags=["system"])
def configure(request: ConfigureRequest) -> dict:
    """Rebuild the system. This clears any loaded documents."""
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=422, detail="chunk_overlap must be smaller than chunk_size."
        )
    try:
        manager.configure(**request.model_dump())
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NotReady as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not initialize the system: {exc}"
        ) from exc
    return manager.status()


@app.post("/documents", tags=["documents"])
async def load_documents(
    files: list[UploadFile] = File(default=[]),
    use_sample: bool = Form(False),
) -> dict:
    """Index uploaded files, or the bundled animals dataset."""
    try:
        if use_sample:
            result = manager.load_sample()
        else:
            uploads = [
                Upload(filename=f.filename or "upload.txt", content=await f.read())
                for f in files
            ]
            result = manager.load_uploads(uploads)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "source": result.source,
        "filenames": result.filenames,
        "relationships": result.relationships,
    }


@app.post("/query", tags=["query"])
def query(request: QueryRequest) -> dict:
    valid = {r["id"] for r in RETRIEVERS}
    if request.retriever not in valid:
        raise HTTPException(
            status_code=400, detail=f"Unknown retriever: {request.retriever}"
        )
    try:
        return manager.query(request.question, request.retriever)
    except NotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc


@app.post("/routing-explanation", tags=["query"])
def routing_explanation(request: QueryRequest) -> dict:
    """What the agentic router would choose, and why - without answering."""
    try:
        return manager.routing_explanation(request.question)
    except NotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Routing failed: {exc}") from exc
