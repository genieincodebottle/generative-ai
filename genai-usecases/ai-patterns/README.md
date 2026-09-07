# AI Reasoning Patterns

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)**. Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Notebooks](https://img.shields.io/badge/notebooks-23-orange)
![SDK](https://img.shields.io/badge/google--genai-2.x-4285F4)

**23 self-contained notebooks, one reasoning pattern each. Every one runs
against a free Gemini key, prints its own reasoning, and ends by naming what
the pattern costs you.**

---

## 1. What these are for

A pattern is a decision about *how many calls to make and what each one sees*.
That is the whole subject. Chain-of-Thought spends one call and lets the model
reason in the open; Tree-of-Thought spends many and throws most away; ReAct
alternates between reasoning and acting; Skeleton-of-Thought splits one answer
into an outline plus N independent expansions so they can run in parallel.

Each notebook implements its pattern in plain Python against the Gemini API -
no framework, no abstraction to see through. Read it in five minutes, run it
in one.

## 2. Run one

### On Colab, no setup

Every notebook has an **Open in Colab** badge in its first cell. Click it,
run the cells, paste a key when prompted. Nothing to install.

### Locally

```bash
git clone https://github.com/genieincodebottle/generative-ai.git
cd generative-ai/genai-usecases/ai-patterns

pip install uv
uv venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell or cmd

uv pip install google-genai jupyter
jupyter notebook
```

Python 3.10+.

### The key

One free key covers every notebook:
[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Each notebook asks for it with `getpass`, so it is never written to disk and
never saved into the notebook's output.

One notebook needs an extra package, installed by its own first cell:
`RAG.ipynb` pulls in `chromadb`. Everything else needs only `google-genai`,
and the first half of `Model-Context-Protocol.ipynb` needs no key at all.

## 3. The patterns

Each notebook is standalone - there is no order you have to follow - but the
groups below run roughly from simplest to most involved.

### Reasoning in one call

| # | Pattern | Notebook | The idea |
|---|---------|----------|----------|
| 1 | Chain-of-Thought | `Chain-of-Thought.ipynb` | Reason step by step before answering |
| 2 | Skeleton-of-Thought | `Skeleton-of-Thought.ipynb` | Outline first, then expand each point independently |
| 3 | Meta-Prompting | `Meta-Prompting.ipynb` | Have the model write its own prompt first |

### Searching over several reasoning paths

| # | Pattern | Notebook | The idea |
|---|---------|----------|----------|
| 4 | Tree-of-Thought | `Tree-of-Thought.ipynb` | Branch, score the branches, keep the best |
| 5 | Graph-of-Thoughts | `Graph-of-Thoughts.ipynb` | Let thoughts merge, not just branch |
| 6 | Language Agent Tree Search | `Language-Agent-Tree-Search.ipynb` | Tree search over *actions*, with value estimates |
| 7 | Reasoning via Planning | `Reasoning-via-Planning.ipynb` | Treat reasoning as a planning problem |

### Decomposition

| # | Pattern | Notebook | The idea |
|---|---------|----------|----------|
| 8 | Least-to-Most | `Least-to-Most-Prompting.ipynb` | Solve easy sub-problems first, feed them forward |
| 9 | Decomposed Prompting | `Decomposed-Prompting.ipynb` | Route each sub-task to a specialised prompt |
| 10 | Plan-and-Solve | `Plan-and-Solve.ipynb` | Write the plan, then execute it |

### Critique and revision

| # | Pattern | Notebook | The idea |
|---|---------|----------|----------|
| 11 | Self-Refine | `Self-Refine.ipynb` | Draft, critique, rewrite, repeat |
| 12 | Reflexion | `Reflexion.ipynb` | Keep the lesson from a failure, retry with it |
| 13 | Chain-of-Verification | `Chain-of-Verification.ipynb` | Generate checking questions, answer them, revise |
| 14 | Recursive Criticism | `Recursive-Criticism-and-Improvement.ipynb` | Criticise and improve in a loop |

### Acting on the world

| # | Pattern | Notebook | The idea |
|---|---------|----------|----------|
| 15 | ReAct | `ReAct.ipynb` | Alternate Thought and Action until an answer |
| 16 | Toolformer | `Toolformer.ipynb` | Decide *when* a tool call is worth making |
| 17 | Automatic Reasoning + Tool Use | `Automatic-Reasoning-and-Tool-Use.ipynb` | Pick the tool as part of the reasoning |
| 18 | RAG | `RAG.ipynb` | Retrieve before answering (needs `chromadb`) |
| 19 | Model Context Protocol | `Model-Context-Protocol.ipynb` | A real MCP server and client, in-process |

### Many agents

| # | Pattern | Notebook | The idea |
|---|---------|----------|----------|
| 20 | Multi-Agent Debate | `Multi-Agent-Debate.ipynb` | Agents argue; disagreement surfaces errors |
| 21 | Orchestrator-Worker | `Orchestrator-Worker.ipynb` | One agent plans, others execute |
| 22 | Generative Agents | `Generative-Agents.ipynb` | Memory, retrieval and reflection over time |
| 23 | Self-Evolving Agent | `Self-Evolving-Agent.ipynb` | The agent rewrites its own rulebook between tasks |

`ai-patterns.pdf` is a one-page visual summary of all 23.

## 4. What the code looks like

Every notebook has the same five-cell opening, so once you have read one you
can skim any of them:

```python
!pip install -qU google-genai

from google import genai
import getpass

API_KEY = getpass.getpass("Enter your Google API key: ")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-flash-latest"
```

Then one class implementing the pattern, then a demo cell that runs it on two
or three examples and prints every intermediate step.

`gemini-flash-latest` is a **rolling alias**. Pinned IDs go out of service -
every `gemini-2.0-*` and `gemini-1.5-*` model these notebooks once used is now
retired, and a pinned notebook fails on its first call with a 404. The alias
does not rot. Pin a version only when you need reproducibility, and expect to
revisit it.

## 5. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `404 ... model not found` | a pinned model that has been retired | use `gemini-flash-latest` |
| `ImportError: cannot import name 'genai' from 'google'` | the old `google-generativeai` package is installed | `pip uninstall google-generativeai && pip install -U google-genai` |
| `429 RESOURCE_EXHAUSTED` | free-tier rate limit | wait a minute; the search patterns make many calls |
| `400 API key not valid` | key pasted with a space or newline | re-paste from AI Studio |
| A notebook makes far more calls than you expected | that is the pattern | Tree-of-Thought and LATS are branch-and-score by design |
| The model ignores the output format | a smaller model, or an unlucky sample | every notebook falls back rather than crashing; re-run the cell |
| `chromadb` errors on Windows | native build | use Colab for `RAG.ipynb`, or install a prebuilt wheel |

## 6. Honest limitations

- **These are teaching implementations.** They are written to be read in one
  sitting, not to be dropped into production. No retries, no rate limiting, no
  token budgets, no caching, no tracing.
- **No pattern here is verified.** Self-critique means the same model judges
  its own work, so a confident mistake survives every round. Where that
  matters most, the notebook says so in its final cell.
- **The demo tasks are chosen to make the pattern visible**, not to benchmark
  it. Do not read "Tree-of-Thought got this right" as evidence that
  Tree-of-Thought is better - measure on your own task, with your own eval.
- **Cost scales with the pattern, not the question.** Skeleton-of-Thought is
  1 + N calls, Multi-Agent Debate is agents x rounds, LATS is worse. Each
  notebook prints its own call count so the arithmetic is in front of you.
- **Output is non-deterministic.** Re-running a cell produces different text,
  and occasionally a different number of steps.
