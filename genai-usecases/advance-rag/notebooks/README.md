# RAG Technique Notebooks

This folder contains nine Google Colab notebooks, each demonstrating a different RAG technique. Every notebook has an **Open in Colab** button at the top — click it to run the notebook instantly without any local setup.

---

## ⚠️ Tech Stack (Different From the Streamlit Apps)

The notebooks use a **different stack** from the Streamlit applications in this repository:

| What | Notebooks | Streamlit apps |
|------|-----------|---------------|
| LLM | **Groq API** (free, open-source models) | **Gemini API** (Google) |
| Embeddings | **HuggingFace** sentence-transformers | **Gemini** embedding models |
| Vector store | **FAISS** | **ChromaDB** |
| Run environment | **Google Colab** (browser, no local install) | Local machine |

### API Keys You Will Need

| Key | Where to get it | Cost |
|-----|----------------|------|
| **Groq API key** | https://console.groq.com/keys | Free tier available |
| **HuggingFace token** | https://huggingface.co/settings/tokens | Free |

Each notebook prompts you to enter these keys at the top using `getpass` (keys are never stored in the notebook file).

---

## 📖 Recommended Reading Order

Work through the notebooks in this order to build your understanding progressively:

| # | Notebook | What it teaches |
|---|----------|----------------|
| 1 | [Basic RAG](basic-rag.ipynb) | The fundamental retrieve-then-generate loop: document loading, chunking, embedding, vector search, and LLM response generation |
| 2 | [Hybrid Search RAG](hybrid-search-rag.ipynb) | Combines keyword search (BM25) with semantic (vector) search using an ensemble retriever for better coverage |
| 3 | [Re-ranking RAG](re_ranking_rag.ipynb) | Adds a cross-encoder reranker after initial retrieval to reorder results by true relevance |
| 4 | [Corrective RAG](corrective-rag.ipynb) | Adds a self-evaluation step: if retrieved documents are judged irrelevant, the system corrects itself before answering |
| 5 | [Query Expansion RAG](query-expansion-rag.ipynb) | Rewrites or expands the user's query into multiple variants to improve recall from the vector store |
| 6 | [Hypothetical Document Embedding RAG](hypothetical-document-embedding-rag.ipynb) | Generates a hypothetical ideal answer first, embeds it, and uses that embedding to search — dramatically improving retrieval for abstract queries |
| 7 | [Multi-index RAG](multi-index-rag.ipynb) | Maintains separate vector indexes for different document types or sources and routes queries to the right index |
| 8 | [Adaptive RAG](adaptive-rag.ipynb) | Classifies the query at runtime and routes it to the most appropriate retrieval strategy dynamically |
| 9 | [Self Adaptive RAG](self-adaptive-rag.ipynb) | Extends Adaptive RAG with a self-grading loop: the system evaluates its own answer and retries with a different strategy if the score is too low |

---

## 🗺️ Technique Comparison at a Glance

| Technique | Best when… | Key trade-off |
|-----------|-----------|---------------|
| Basic RAG | Starting out, simple documents | Lowest accuracy on hard queries |
| Hybrid Search | Documents have both keyword and conceptual content | Slightly more complex setup |
| Re-ranking | You need high precision on top results | Extra latency from reranker |
| Corrective RAG | Retrieval quality is unpredictable | More LLM calls per query |
| Query Expansion | Short or ambiguous user queries | Higher token usage |
| Hypothetical Doc Embedding | Abstract or research-style questions | LLM call before retrieval |
| Multi-index RAG | Multiple heterogeneous document sources | Index management overhead |
| Adaptive RAG | Mixed query types in a single system | Routing adds complexity |
| Self Adaptive RAG | Highest accuracy required, latency tolerated | Most LLM calls, slowest |

---

## 🔧 How to Run a Notebook Locally (Optional)

If you prefer to run the notebooks locally instead of on Colab:

```bash
pip install jupyter notebook
jupyter notebook basic-rag.ipynb
```

Install each notebook's dependencies by running the `!pip install` cell at the top of the notebook before executing the rest.

---

> **Tip:** After understanding the concepts in a notebook, try the equivalent Streamlit app (where available) in the [`rag_techniques/`](../rag_techniques/) folder to see the same technique with a full interactive UI and the Gemini API.
