import os
import streamlit as st
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

load_dotenv()

# ── Provider registry ─────────────────────────────────────
# Each entry: key_env, factory, models.  Factories are filled
# only when the SDK package is installed (graceful fallback).

PROVIDERS = {}

try:
    from langchain_groq import ChatGroq

    PROVIDERS["Groq"] = {
        "key_env": "GROQ_API_KEY",
        "factory": lambda model, key: ChatGroq(model=model, api_key=key),
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
    }
except ImportError:
    pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    PROVIDERS["Gemini"] = {
        "key_env": "GOOGLE_API_KEY",
        "factory": lambda model, key: ChatGoogleGenerativeAI(
            model=model, google_api_key=key
        ),
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    }
except ImportError:
    pass


def get_available_providers():
    """Return provider names whose SDK is installed *and* API key is set."""
    return [name for name, cfg in PROVIDERS.items() if os.getenv(cfg["key_env"])]


def get_models(provider):
    """Return the model list for a provider."""
    return PROVIDERS[provider]["models"]


def _build_llm(provider, model_name):
    """Instantiate the LLM for the chosen provider and model."""
    cfg = PROVIDERS[provider]
    return cfg["factory"](model_name, os.getenv(cfg["key_env"]))


@st.cache_resource(show_spinner="Connecting to Neo4j...")
def _get_graph():
    """Create and cache the Neo4j graph connection."""
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )
    graph.refresh_schema()
    return graph


def run_query(query, provider, model_name):
    """Run a natural-language query against the Neo4j graph database."""
    graph = _get_graph()
    llm = _build_llm(provider, model_name)

    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        verbose=True,
        allow_dangerous_requests=True,
    )
    response = chain.invoke({"query": query})
    return response["result"]
