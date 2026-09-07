"""FastAPI routing layer for the agentic AI platform.

One route runs any registered app. The app declares its own inputs, so adding
one does not mean adding a route.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import providers, registry
from services.providers import ProviderError

app = FastAPI(
    title="Agentic AI Platform API",
    description=(
        "Agentic workflow patterns, LangGraph pipelines, CrewAI crews and "
        "multi-agent orchestration behind one API. The Streamlit UI in `ui/` "
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
    providers: list[str]
    ollama_reachable: bool


class ProviderInfo(BaseModel):
    id: str
    models: list[str]
    key_url: str
    note: str
    needs_key: bool


class CatalogueResponse(BaseModel):
    providers: list[ProviderInfo]
    apps: list[dict]


class RunRequest(BaseModel):
    provider: str = Field(..., examples=["Gemini"])
    model: str = Field(..., examples=["gemini-flash-latest"])
    ollama_base_url: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        providers=providers.available_providers(),
        # Ollama needs no key, so "configured" and "actually running" are
        # different questions. The UI needs both answered.
        ollama_reachable=providers.ollama_reachable(),
    )


@app.get("/catalogue", response_model=CatalogueResponse, tags=["meta"])
def catalogue() -> CatalogueResponse:
    return CatalogueResponse(
        providers=[
            ProviderInfo(
                id=name,
                models=providers.models_for(name),
                key_url=providers.PROVIDERS[name]["key_url"],
                note=providers.PROVIDERS[name]["note"],
                needs_key=providers.PROVIDERS[name]["env_key"] is not None,
            )
            for name in providers.available_providers()
        ],
        apps=registry.catalogue(),
    )


@app.post("/apps/{app_id}/run", tags=["apps"])
async def run_app(app_id: str, request: RunRequest) -> dict:
    """Run one app. Inputs are validated against the app's own declaration."""
    try:
        app_spec = registry.get(app_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown app: {app_id}. "
                   f"See GET /catalogue for the list.",
        ) from exc

    if request.provider not in providers.PROVIDERS:
        raise HTTPException(
            status_code=400, detail=f"Unknown provider: {request.provider}"
        )

    missing = [
        f.name for f in app_spec.fields
        if f.required and not str(request.inputs.get(f.name, "")).strip()
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required input(s): {', '.join(missing)}",
        )

    try:
        result = await app_spec.runner(
            request.provider, request.model, request.ollama_base_url,
            **request.inputs,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"{app_spec.label} failed: {exc}"
        ) from exc

    return {"app": app_id, "provider": request.provider,
            "model": request.model, **result}
