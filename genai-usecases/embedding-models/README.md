# Vector Embeddings

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)**. Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C)
![Key](https://img.shields.io/badge/key-free%20tier-brightgreen)

**One notebook that builds a working RAG pipeline on a Wikipedia page, then
scores its own answer with the same embeddings it retrieved with. Google,
OpenAI and HuggingFace embedding models, one free key needed.**

---

## 1. What an embedding is

An embedding turns text into a vector of numbers positioned so that texts
meaning similar things land near each other. That single property is what
makes semantic search, clustering and retrieval possible: you stop matching
words and start matching meaning.

`gemini-embedding-001` returns a 3072-dimension vector. Nothing in it is
interpretable on its own - the information lives entirely in the distances
between vectors.

## 2. What the notebook actually does

[embedding_models.ipynb](./embedding_models.ipynb), in order:

1. Loads the Wikipedia article on artificial intelligence.
2. Splits it into 500-character chunks with 50 characters of overlap.
3. Embeds every chunk with `gemini-embedding-001` and stores them in Chroma.
4. Retrieves the 5 nearest chunks for a question and answers from them.
5. **Scores the answer** by embedding the query, the answer and the retrieved
   context, then taking cosine similarities between them.

Step 5 is the part worth your attention. A real run:

```
Query-Response Similarity   : 0.7827
Response-Context Similarity : 0.7510
Overall Relevance Score     : 0.7668
```

Read those honestly: they say the answer is on-topic and grounded in what was
retrieved. They do **not** say it is correct. An answer confidently drawn from
the wrong chunk scores just as well. Section 6 says more about this.

The last third of the notebook shows the same interface with OpenAI's
`text-embedding-3-large` and a HuggingFace open-source model, so you can swap
providers with a one-line change.

## 3. Run it

### Colab, no setup

Open the notebook from its badge, paste a free Google key when prompted.

### Locally

```bash
git clone https://github.com/genieincodebottle/generative-ai.git
cd generative-ai/genai-usecases/embedding-models

pip install uv
uv venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell or cmd

uv pip install jupyter
jupyter notebook embedding_models.ipynb
```

The notebook's own first cell installs everything else. Python 3.10+.

### The key

Free: [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The
same key covers the Gemini LLM and the embedding model. It is read with
`getpass`, so it is never written to disk.

OpenAI and HuggingFace sections are optional and each ask for their own key
in their own cell. Skip them and the rest still runs.

## 4. Choosing a model

| Model | Provider | Cost | Notes |
|---|---|---|---|
| `gemini-embedding-001` | Google | free tier | 3072 dims; unifies the old `text-embedding-005` and `text-multilingual-embedding-002` |
| `text-embedding-3-large` | OpenAI | paid | 3072 dims; `-small` is cheaper and shorter |
| `nomic-embed-text-v1.5` | HuggingFace | free, local | needs `trust_remote_code=True`; runs on your machine |

`text-embedding-004` and `embedding-001` are **retired**. If an older copy of
this notebook 404s on its first embedding call, that is why.

Two rules that matter more than the choice itself:

- **Embed queries and documents with the same model.** Vectors from two
  different models are not comparable, and nothing raises an error if you mix
  them - you just get quietly meaningless distances.
- **Changing the model means re-embedding the whole store.** Dimensions
  differ, and so does the geometry even when they do not.

## 5. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: langchain.vectorstores` | LangChain 1.x moved it | `from langchain_chroma import Chroma`; this notebook is already updated |
| `ModuleNotFoundError: bs4` | `WebBaseLoader` needs beautifulsoup4 | it is in the install cell; re-run it |
| `404 ... models/text-embedding-004` | retired model ID | use `models/gemini-embedding-001` |
| `AttributeError: 'list' object has no attribute 'strip'` | a response `.content` is a list of blocks | read it through `message_text()`, as this notebook does |
| `USER_AGENT environment variable not set` | a warning, not an error | already set in the imports cell |
| `429 RESOURCE_EXHAUSTED` | free-tier rate limit | wait; embedding many chunks makes many calls |
| Chroma errors on Windows | native build | run it on Colab |
| Similarity scores near 1.0 for everything | you embedded query and docs with different models | use one model for both |

## 6. Honest limitations

- **Cosine similarity is not correctness.** The evaluation in this notebook
  measures whether the answer is *about* the query and *drawn from* the
  context. A fluent answer built on the wrong retrieved chunk scores highly.
  Treat it as a smoke test, not a grade.
- **One question, one document.** These numbers are a single sample on a
  single Wikipedia page. Comparing embedding models properly means a fixed
  query set with known-correct documents - see the
  [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
- **Chunking is not tuned.** 500/50 is a starting point, and it is often the
  single highest-leverage thing to change in a RAG pipeline.
- **The vector store is in memory.** It disappears when the kernel restarts,
  so every run re-embeds the whole article.
- **The Google and HuggingFace paths are verified end to end**; the local
  `nomic-embed-text-v1.5` returns 768 dimensions on CPU. The OpenAI section
  is correct against the current API but could not be executed - the key
  available had no credit.

## 7. Further reading

- [Vector embeddings guide (PDF)](./vector-embeddings-guide.pdf) - the visual
  explanation of the concepts above.
- [Gemini Embedding paper](https://arxiv.org/abs/2503.07891)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
