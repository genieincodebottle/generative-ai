"""FastAPI routing layer for the Llama 4 multi-function app."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import config, documents
from services.documents import IngestError, NotIndexed, Upload
from services.llama_service import (
    InvalidInput,
    MissingKey,
    ModelError,
    chat,
    describe_image,
    gemini_fallback,
)

app = FastAPI(
    title="Llama 4 Multi-Function API",
    description=(
        "Chat, vision, retrieval and agents behind one API, all driven by "
        "Llama 4 on Groq with a Gemini fallback. The Streamlit UI in `ui/` "
        "is a client of it."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    keys: dict[str, bool]
    features: list[str]
    indexed: bool


class CatalogueResponse(BaseModel):
    models: list[str]
    vision_models: list[str]
    gemini_models: list[str]
    features: dict
    key_urls: dict[str, str]


class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    model: str = Field("meta-llama/llama-4-scout-17b-16e-instruct")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    allow_fallback: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=20)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        keys={name: config.has(name) for name in config.KEYS},
        features=config.available_features(),
        indexed=documents.status()["indexed"],
    )


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    return CatalogueResponse(
        models=config.GROQ_MODELS,
        vision_models=config.GROQ_VISION_MODELS,
        gemini_models=config.GEMINI_MODELS,
        features=config.FEATURES,
        key_urls={n: s["key_url"] for n, s in config.KEYS.items()},
    )


@app.post("/chat", tags=["chat"])
def chat_route(request: ChatRequest) -> dict:
    """Text chat, with an optional and always-reported Gemini fallback."""
    messages = [m.model_dump() for m in request.messages]
    try:
        reply = chat(messages, request.model, request.temperature)
    except InvalidInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MissingKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelError as exc:
        if not (request.allow_fallback and config.has("google")):
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        prompt = messages[-1]["content"]
        try:
            reply = gemini_fallback(prompt)
            reply.fallback_reason = str(exc)
        except (MissingKey, ModelError) as fallback_exc:
            raise HTTPException(
                status_code=502,
                detail=f"{exc}\nFallback also failed: {fallback_exc}",
            ) from fallback_exc
    return reply.__dict__


@app.post("/vision", tags=["vision"])
async def vision_route(
    file: UploadFile = File(...),
    prompt: str = Form("Read all the text in this image."),
    model: str = Form("meta-llama/llama-4-scout-17b-16e-instruct"),
) -> dict:
    """Read or describe an uploaded image."""
    content = await file.read()
    try:
        reply = describe_image(content, file.content_type or "", prompt, model)
    except InvalidInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MissingKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return reply.__dict__


@app.get("/documents", tags=["documents"])
def document_status() -> dict:
    return documents.status()


@app.post("/documents", tags=["documents"])
async def add_documents(files: list[UploadFile] = File(...)) -> dict:
    added, failed = [], []
    for upload in files:
        record = Upload(filename=upload.filename or "file.txt",
                        content=await upload.read())
        try:
            added.append(documents.add_document(record).__dict__)
        except IngestError as exc:
            # One bad file must not lose the good ones.
            failed.append({"name": record.filename, "error": str(exc)})
    if not added and failed:
        raise HTTPException(status_code=422, detail="; ".join(
            f"{f['name']}: {f['error']}" for f in failed))
    return {"added": added, "failed": failed, **documents.status()}


@app.delete("/documents", status_code=204, tags=["documents"])
def clear_documents() -> None:
    documents.reset()


@app.post("/search", tags=["documents"])
def search_route(request: SearchRequest) -> dict:
    try:
        return {"query": request.query,
                "results": documents.search(request.query, request.top_k)}
    except NotIndexed as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
