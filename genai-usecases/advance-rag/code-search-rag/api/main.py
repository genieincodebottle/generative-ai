"""FastAPI routing layer for code search."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.manager import (
    LLM_MODELS,
    IngestError,
    MissingAPIKey,
    NotReady,
    Upload,
    manager,
)

app = FastAPI(
    title="Code Search RAG API",
    description=(
        "Retrieval over source code, chunked by function and class rather "
        "than by character count. The Streamlit UI in `ui/` is a client of "
        "this API."
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
    languages: list[str]
    defaults: dict


class ConfigureRequest(BaseModel):
    llm_model: str = "gemini-flash-latest"
    top_k_initial: int = Field(100, ge=1, le=500)
    top_k_rerank: int = Field(3, ge=1, le=50)
    top_k_final: int = Field(2, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    language: str | None = None


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    status = manager.status()
    return HealthResponse(
        status="ok", google_key=status["google_key"], ready=status["ready"]
    )


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    return CatalogueResponse(
        llm_models=LLM_MODELS,
        languages=manager.status()["languages"],
        defaults={"top_k_initial": 100, "top_k_rerank": 3, "top_k_final": 2},
    )


@app.get("/status", tags=["meta"])
def status() -> dict:
    return manager.status()


@app.post("/configure", tags=["system"])
def configure(request: ConfigureRequest) -> dict:
    """Rebuild the system. This clears the index."""
    if request.top_k_final > request.top_k_rerank:
        raise HTTPException(
            status_code=422,
            detail="top_k_final must not exceed top_k_rerank.",
        )
    try:
        manager.configure(**request.model_dump())
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not initialize the system: {exc}"
        ) from exc
    return manager.status()


@app.post("/index", tags=["index"])
async def index(
    files: list[UploadFile] = File(default=[]),
    use_samples: bool = Form(False),
    repo_name: str = Form(""),
) -> dict:
    """Index uploaded source files, or the bundled samples."""
    try:
        if use_samples:
            result = manager.index_samples()
        else:
            uploads = [
                Upload(filename=f.filename or "file.py", content=await f.read())
                for f in files
            ]
            if repo_name:
                result = manager.index_repository(uploads, repo_name)
            else:
                result = manager.index_uploads(uploads)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"chunks": result.chunks, "files": result.files,
            "skipped": result.skipped}


@app.post("/search", tags=["search"])
def search(request: SearchRequest) -> dict:
    filters = {"language": request.language} if request.language else None
    try:
        answer = manager.search(request.query, filters)
    except NotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc
    return {"query": request.query, "answer": answer,
            "language_filter": request.language}
