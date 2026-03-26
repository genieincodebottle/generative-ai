# Vector Embeddings Guide

Learn how vector embeddings work and how to generate them using Google, OpenAI, and HuggingFace embedding models.

## What are Embeddings?

Embeddings convert text into numerical vectors that capture meaning. Similar texts produce similar vectors, enabling semantic search, clustering, and RAG (Retrieval-Augmented Generation).

## Contents

| File | Description |
|------|-------------|
| `embedding_models.ipynb` | Hands-on notebook comparing embedding models from Google, OpenAI, and HuggingFace |
| `vector-embeddings-guide.pdf` | Visual guide explaining embedding concepts |

## Quick Start (Google Colab - Recommended)

1. Open the notebook in Google Colab (no local setup needed)
2. Get a free Google API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
3. Run the cells and experiment with different texts

## Run Locally

1. Create a virtual environment:
   ```bash
   pip install uv
   uv venv
   .venv\Scripts\activate    # On Linux/Mac: source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install jupyter langchain-google-genai python-dotenv numpy
   ```

3. Set up your API key:
   ```bash
   echo "GOOGLE_API_KEY=your_key_here" > .env
   ```

4. Launch Jupyter:
   ```bash
   jupyter notebook embedding_models.ipynb
   ```
