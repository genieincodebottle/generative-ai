# RAG Techniques — Streamlit Apps

Five RAG implementations as standalone Streamlit apps. Complete the [one-time setup in the parent README](../README.md#f-try-advanced-rag-techniques-in-streamlit-ui), then run any of the apps below.

---

## Quick Reference

| File | Technique | What it demonstrates | Level |
|------|-----------|---------------------|-------|
| `basic_rag.py` | **Basic RAG** | Core retrieve → generate loop. The baseline every other technique builds on. | ⭐ Beginner |
| `hybrid_search_rag.py` | **Hybrid Search RAG** | Combines keyword (BM25) + semantic (vector) search. Better recall for factual queries. | ⭐⭐ Intermediate |
| `re_ranking_rag.py` | **Re-ranking RAG** | Retrieves more docs first, then re-scores and reorders them before generating. | ⭐⭐ Intermediate |
| `corrective_rag.py` | **Corrective RAG** | Generates a draft answer, critiques it, retrieves extra context, then regenerates. | ⭐⭐ Intermediate |
| `adaptive_rag.py` | **Adaptive RAG** | Adjusts retrieval depth and strategy based on query complexity. | ⭐⭐⭐ Advanced |

**Recommended order:** Basic → Hybrid → Re-ranking → Corrective → Adaptive

---

## How to Run

From inside the `rag_techniques/` folder:

```bash
streamlit run basic_rag.py          # Basic RAG
streamlit run hybrid_search_rag.py  # Hybrid Search RAG
streamlit run re_ranking_rag.py     # Re-ranking RAG
streamlit run corrective_rag.py     # Corrective RAG
streamlit run adaptive_rag.py       # Adaptive RAG
```

The app opens at `http://localhost:8501`

---

## What API Keys Do You Need?

| Provider | Key | Purpose | Where to get it |
|----------|-----|---------|-----------------|
| **Gemini** (default) | `GOOGLE_API_KEY` | LLM + Embeddings | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — free |
| **Groq** (optional) | `GROQ_API_KEY` | Free open-source LLMs (Llama) | [console.groq.com/keys](https://console.groq.com/keys) — free |

> When using **Groq**, embeddings are handled locally via HuggingFace `nomic-embed-text-v1.5` (~270 MB, downloaded once on first run). No HuggingFace API key needed.

Copy `.env.example` → `.env` and add your key(s):

```bash
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here   # optional
```

---

## Key Concepts for Beginners

**What is "chunking"?**
Documents are split into smaller overlapping pieces before being embedded. Each piece is called a chunk.
- **Chunk Size** — how many characters per chunk (default: 1000). Larger = more context per chunk but less precise matching.
- **Chunk Overlap** — how many characters adjacent chunks share (default: 100). Prevents sentences from being cut off at chunk boundaries.

**What is an "embedding" / "vector"?**
A list of numbers that represents the *meaning* of a text. Similar texts produce similar vectors. The retriever finds relevant chunks by comparing vectors — not by exact keyword matching.

**What is ChromaDB?**
A lightweight local vector database. It stores your document embeddings on disk so you can search them by semantic similarity. No separate server or account needed — it just works.

**What is BM25?** *(used in Hybrid Search RAG)*
A classic keyword search algorithm. It finds documents that contain the exact words in your query. Complementary to vector search: BM25 is great for specific terms, vector search is great for concepts.

**What is re-ranking?** *(used in Re-ranking RAG)*
After retrieving a large set of candidate chunks, a second model scores them for relevance and reorders them. The most relevant chunks go to the LLM, improving answer quality.

---

## Troubleshooting

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `GOOGLE_API_KEY not set` | Missing `.env` file or key | Rename `.env.example` → `.env`, add your key |
| `GROQ_API_KEY not set` | Selected Groq but key missing | Add `GROQ_API_KEY` to `.env` or switch to Gemini |
| `No module named 'langchain_groq'` | Groq dependencies not installed | `pip install langchain-groq langchain-huggingface sentence-transformers` |
| PDF returns no results | PDF is scan-only (no text layer) | Use a PDF with selectable text, or try a different file |
| Very slow first run with Groq | HuggingFace model downloading | Wait for the ~270 MB model to download; subsequent runs are fast |
