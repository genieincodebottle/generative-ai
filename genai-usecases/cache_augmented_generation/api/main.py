"""FastAPI routing layer for cache-augmented generation."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import cag_service, config
from services.cag_model import ModelLoadError
from services.cag_service import InvalidInput, NotLoaded

app = FastAPI(
    title="Cache-Augmented Generation API",
    description=(
        "Put the whole document in the context once, keep the KV cache, and "
        "answer every question from it. No retrieval. The Streamlit UI in "
        "`ui/` is a client of this API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    loaded: bool
    model: str | None
    hf_token: bool


class CatalogueResponse(BaseModel):
    models: list[dict]
    default_model: str
    max_document_chars: int
    max_questions: int


class LoadRequest(BaseModel):
    model_id: str = Field(config.DEFAULT_MODEL)
    quantized: bool = False


class RunRequest(BaseModel):
    document: str = Field(..., min_length=1)
    use_cache: bool = True


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", **cag_service.status())


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    """Models, with the gated ones clearly marked.

    Ungated models come first: they need no token, which means the project
    runs on a fresh clone with nothing but `pip install`.
    """
    return CatalogueResponse(
        models=config.MODELS,
        default_model=config.DEFAULT_MODEL,
        max_document_chars=config.MAX_DOCUMENT_CHARS,
        max_questions=config.MAX_QUESTIONS,
    )


@app.post("/model", tags=["model"])
def load_model(request: LoadRequest) -> dict:
    try:
        cag_service.load(request.model_id, request.quantized)
    except InvalidInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return cag_service.status()


@app.delete("/model", status_code=204, tags=["model"])
def unload_model() -> None:
    cag_service.unload()


@app.get("/dataset", tags=["dataset"])
def default_dataset() -> dict:
    """The bundled corpus: ten documents and one question about each.

    `corpus` is every document joined together - the single context that
    cache-augmented generation is built around. Returning it means the UI can
    run with nothing pasted in.
    """
    try:
        rows = cag_service.default_dataset()
        corpus = cag_service.corpus_from(rows)
    except InvalidInput as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "rows": [{"question": r.question, "answer": r.answer,
                  "document_chars": len(r.document)} for r in rows],
        "corpus": corpus,
        "corpus_chars": len(corpus),
    }


@app.post("/run", tags=["run"])
async def run(document: str = "", use_cache: bool = True,
              dataset: UploadFile | None = File(default=None)) -> dict:
    """Answer every question in the dataset, with or without the cache."""
    try:
        rows = (cag_service.parse_dataset(await dataset.read())
                if dataset else cag_service.default_dataset())
        # With no document supplied, fall back to the dataset's own corpus.
        text = document.strip() or cag_service.corpus_from(rows)
        return cag_service.run_experiment(text, rows, use_cache)
    except InvalidInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotLoaded as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"The run failed: {exc}") from exc
