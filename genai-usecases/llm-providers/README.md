# Run & Experiment with LLMs

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)**. Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-FF4B4B)
![Providers](https://img.shields.io/badge/providers-6%20cloud%20%2B%202%20local-lightgrey)

**The same small chat app against six cloud providers and two local runtimes,
so you can see what actually differs between them: the model IDs, the
parameter names, and almost nothing else.**

---

## 1. What is here

| Folder | What it is | Needs |
|---|---|---|
| `notebooks/` | One Colab notebook per provider: OpenAI, Claude, Gemini, Groq, Cohere, DeepSeek | that provider's key |
| `python_scripts/` | Four Streamlit apps, one per provider, with a model picker and a temperature slider | that provider's key |
| `local_llms/ollama/` | The same thing on Ollama, running on your own machine | nothing |
| `local_llms/huggingface/` | Loading a model directly with transformers | a GPU, ideally |

Start with `notebooks/gemini.ipynb` or `notebooks/groq.ipynb` if you want to
be running in under two minutes on a free key.

## 2. Run the Streamlit apps

```bash
git clone https://github.com/genieincodebottle/generative-ai.git
cd generative-ai/genai-usecases/llm-providers/python_scripts

pip install uv
uv venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell or cmd

uv pip install -r requirements.txt

cp .env.example .env           # then put ONE key in it
```

Then run whichever provider you have a key for:

```bash
streamlit run gemini.py        # free tier
streamlit run groq_api.py      # free tier
streamlit run claude.py        # paid
streamlit run openai_api.py    # paid
```

**You only need one key.** Each app checks for its own and stops with a
message naming the key and where to get it, rather than a traceback.

### Running the OpenAI app without paying

Groq exposes an **OpenAI-compatible endpoint**, so `openai_api.py` runs
unchanged against a free Groq key - only `base_url` differs. Set `GROQ_API_KEY`
and leave `OPENAI_API_KEY` empty; the app detects that, points itself at Groq
and offers the `gpt-oss` models. Useful for exercising the OpenAI code path
without an OpenAI balance.

### Keys

| Provider | Where | Cost |
|---|---|---|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | free tier |
| Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | free tier |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/settings/keys) | paid |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | paid |
| Cohere | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) | free trial |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/) | paid |
| Ollama | none - local | free |

## 3. Run models locally

### Ollama

No key, no account, no data leaving the machine. Full setup guide:
[local_llms/ollama/](./local_llms/ollama/)

```bash
ollama pull llama3.2
ollama serve
python local_llms/ollama/ollama_example.py
```

### HuggingFace transformers

[local_llms/huggingface/huggingface_models.ipynb](./local_llms/huggingface/huggingface_models.ipynb)
loads a distilled DeepSeek-R1 and runs it directly. This is the slowest path
on a CPU by a wide margin - use Colab's free GPU for it.

## 4. Model IDs go stale, and that is the failure you will hit

Every app here picks its model from a short list at the top of the file. Those
lists are the part of this repo that rots fastest, and a retired ID fails on
the very first call:

- **Groq** shut down `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`,
  `gemma2-9b-it` and `qwen/qwen3-32b` for the free and developer tiers during
  2025-2026. The apps here now offer `openai/gpt-oss-20b` and
  `openai/gpt-oss-120b`, Groq's own stated replacements.
- **OpenAI** shut down `gpt-4.5-preview` in July 2025, and dated snapshot IDs
  like `gpt-5-2025-08-07` each carry their own retirement date.
- **Google** retired every `gemini-1.5-*` and `gemini-2.0-*` model. The Gemini
  app offers `gemini-pro-latest`, a **rolling alias** that does not rot.
- **Anthropic** dated IDs such as `claude-sonnet-4-20250514` are snapshots;
  the apps use family aliases like `claude-sonnet-5` instead.

Prefer an alias unless you need reproducibility. If you need a pinned ID,
write down when you pinned it.

## 5. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `model_not_found` / `model_decommissioned` | a retired model ID | pick another from the dropdown, or update the list at the top of the script |
| App stops with "KEY is not set" | no `.env`, or the wrong key in it | `cp .env.example .env` and add the key for the app you are running |
| `ImportError: cannot import name 'ChatOpenAI' ... circular import` | a file named `openai.py` next to your script shadows the `openai` package | this is why the script is `openai_api.py`; do not rename it back |
| `ModuleNotFoundError: langchain_openai` | partial install | `uv pip install -r requirements.txt` again |
| `401` / `invalid_api_key` | key copied with a newline or quotes | re-copy; no quotes in `.env` |
| `429` | free-tier rate limit | wait, or lower the temperature/length |
| `429 insufficient_quota` on OpenAI | the key is valid but has no credit | add credit, or set `GROQ_API_KEY` and use the compatible endpoint |
| A reply comes back empty | reasoning tokens share the `max_tokens` budget | raise `max_tokens`; the apps set 2048 |
| Ollama: `connection refused` | the daemon is not running | `ollama serve` in another terminal |
| HuggingFace notebook is extremely slow | CPU inference | use a Colab GPU runtime |
| `.env` ignored | you ran from the wrong directory | run from inside `python_scripts/` |

## 6. Layout

```
notebooks/                 one Colab notebook per cloud provider
  claude.ipynb  cohere.ipynb  deepseek.ipynb
  gemini.ipynb  groq.ipynb    openai.ipynb

python_scripts/            Streamlit apps
  gemini.py  groq_api.py  claude.py  openai_api.py
  llm_text.py              reads text out of a response safely
  .env.example  requirements.txt

local_llms/ollama/         local models, no key
local_llms/huggingface/    transformers directly
```

`llm_text.py` exists for one reason: on current models a response's `content`
can be a **list of content blocks** rather than a string, so `len(response)`
counts blocks and `.strip()` raises `AttributeError`. Every app reads text
through `message_text()` instead of touching `.content` directly.

## 7. Honest limitations

- **These are single-turn demos.** No conversation memory, no streaming, no
  tool use, no retries, no token accounting.
- **Cross-provider comparison here is qualitative.** The apps make it easy to
  ask the same question of four providers; they do not score the answers, and
  four samples is not a benchmark.
- **All four apps were booted end to end**, and the OpenAI app was verified
  making a real call through Groq's compatible endpoint (the OpenAI key
  available had no credit). Booting proves wiring, imports and the key guard;
  it does not prove every model in each dropdown answers.
- **Paid providers cost real money per call.** The apps have no spend guard.
- **Keys live in a plaintext `.env`.** Fine locally; not how you would deploy
  this.
