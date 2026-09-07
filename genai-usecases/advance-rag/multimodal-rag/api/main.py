"""FastAPI routing layer for multimodal RAG."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.manager import (
    MAIN_MODELS,
    VISION_MODELS,
    IngestError,
    MissingAPIKey,
    NotReady,
    Upload,
    manager,
)

app = FastAPI(
    title="Multimodal RAG API",
    description=(
        "Retrieval over text, tables, and images together. Images are "
        "described by a vision model at index time, so they become "
        "searchable text. The Streamlit UI in `ui/` is a client of this API."
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
    main_models: list[str]
    vision_models: list[str]
    defaults: dict


class ConfigureRequest(BaseModel):
    main_model: str = "gemini-pro-latest"
    vision_model: str = "gemini-flash-latest"
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(8192, ge=256, le=32768)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    k: int = Field(6, ge=1, le=30)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    status = manager.status()
    return HealthResponse(
        status="ok", google_key=status["google_key"], ready=status["ready"]
    )


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    return CatalogueResponse(
        main_models=MAIN_MODELS,
        vision_models=VISION_MODELS,
        defaults={"temperature": 0.1, "max_tokens": 8192, "k": 6},
    )


@app.get("/status", tags=["meta"])
def status() -> dict:
    return manager.status()


@app.post("/configure", tags=["system"])
def configure(request: ConfigureRequest) -> dict:
    """Rebuild the system. This clears anything already indexed."""
    try:
        manager.configure(**request.model_dump())
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not initialize the system: {exc}"
        ) from exc
    return manager.status()


@app.post("/documents", tags=["documents"])
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Index PDFs and images together.

    Indexing is slow for images: each one is described by the vision model
    before it can be retrieved as text.
    """
    uploads = [
        Upload(filename=f.filename or "upload.pdf", content=await f.read())
        for f in files
    ]
    try:
        result = manager.ingest(uploads)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"documents": result.documents, "images": result.images}


@app.post("/query", tags=["query"])
def query(request: QueryRequest) -> dict:
    try:
        return manager.query(request.question, k=request.k)
    except NotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc
