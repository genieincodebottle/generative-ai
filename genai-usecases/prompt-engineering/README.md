# Prompt Engineering

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)**. Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Notebooks](https://img.shields.io/badge/notebooks-16-orange)
![Providers](https://img.shields.io/badge/Groq%20%7C%20Gemini%20%7C%20OpenAI%20%7C%20Claude%20%7C%20Ollama-lightgrey)

<img src="https://raw.githubusercontent.com/genieincodebottle/generative-ai/main/images/Prompt_engineering.png">

**Sixteen notebooks, one technique each, all running on a free Groq key by
default. Every notebook is standalone: open it, paste a key, run it.**

---

## 1. Run one

### On Colab, no setup

Each notebook opens in Colab from its own badge. Run the cells and paste a key
when prompted.

### Locally

```bash
git clone https://github.com/genieincodebottle/generative-ai.git
cd generative-ai/genai-usecases/prompt-engineering

pip install uv
uv venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell or cmd

uv pip install "langchain>=1.0" "langchain-groq>=0.2.4" jupyter
jupyter notebook
```

Python 3.10+. Each notebook's first cell installs whatever else it needs.

### Keys

| Provider | Where | Cost |
|---|---|---|
| **Groq** (default for 15 notebooks) | [console.groq.com/keys](https://console.groq.com/keys) | free tier |
| Google Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | free tier |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | paid |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/) | paid |
| **Ollama** | none - runs locally | free |

Keys are read with `getpass`, so nothing is written to disk or saved into a
notebook's output.

### Which model these run on

The Groq notebooks use **`openai/gpt-oss-20b`**, Groq's own recommended
replacement for `llama-3.1-8b-instant`, which was **shut down for the free and
developer tiers on 16 August 2026**. If you are reading an older copy of these
notebooks and every cell fails, that retired model ID is why.

Model IDs are a single string near the top of each notebook, so switching
provider or model is a one-line edit:

```python
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.5)
```

One exception: `automatic_reasoning_and_tool_use.ipynb` uses
**`qwen/qwen3.6-27b`**. ART teaches a *text* tool protocol - the model writes
JSON and the notebook parses it - and a tool-calling-native model emits a real
tool call instead, which the API then rejects because no tools were bound.

`prompt_engineering_main.ipynb` is the multi-provider notebook: it shows the
same techniques against Groq, Gemini, OpenAI and Claude, with each provider in
its own cell.

## 2. The techniques

| # | Technique | Notebook | The idea |
|---|---|---|---|
| 1 | Basic prompt engineering | [basic_prompt_engineering.ipynb](./basic_prompt_engineering.ipynb) | Clarity, context, constraints, examples, format |
| 2 | Chain-of-Thought | [chain_of_thought_CoT.ipynb](./chain_of_thought_CoT.ipynb) | Articulate the intermediate steps |
| 3 | Zero-Shot CoT | [zero_shot_chain_of_thought.ipynb](./zero_shot_chain_of_thought.ipynb) | Get reasoning with no examples at all |
| 4 | Few-Shot CoT | [few_shot_chain_of_thought.ipynb](./few_shot_chain_of_thought.ipynb) | Demonstrate the reasoning you want |
| 5 | Prompt chaining | [prompt_chaining.ipynb](./prompt_chaining.ipynb) | Feed one prompt's output into the next |
| 6 | ReAct | [reasoning_and_acting_ReAct.ipynb](./reasoning_and_acting_ReAct.ipynb) | Alternate reasoning and action |
| 7 | Tree of Thoughts | [tree_of_thoughts_ToT.ipynb](./tree_of_thoughts_ToT.ipynb) | Branch, evaluate, keep the best path |
| 8 | Self-Consistency | [self_consistency.ipynb](./self_consistency.ipynb) | Sample many answers, take the majority |
| 9 | HyDE | [hypothetical_document_embeddings_HyDE.ipynb](./hypothetical_document_embeddings_HyDE.ipynb) | Retrieve using an imagined answer, not the question |
| 10 | Least-to-Most | [least_to_most_prompting.ipynb](./least_to_most_prompting.ipynb) | Solve the easy sub-problems first |
| 11 | Graph prompting | [graph_prompting.ipynb](./graph_prompting.ipynb) | Represent relationships as a graph |
| 12 | Recursive prompting | [recursive_prompting.ipynb](./recursive_prompting.ipynb) | Refine the prompt itself, iteratively |
| 13 | ART | [automatic_reasoning_and_tool_use.ipynb](./automatic_reasoning_and_tool_use.ipynb) | Let the model choose the tool as it reasons |
| 14 | APE | [automatic_prompt_engineer_APE.ipynb](./automatic_prompt_engineer_APE.ipynb) | Have the model write and score its own prompts |
| 15 | Multi-provider comparison | [prompt_engineering_main.ipynb](./prompt_engineering_main.ipynb) | The same technique across four providers |
| 16 | Local models | [prompt_engineering_with_ollama_based_models.ipynb](./prompt_engineering_with_ollama_based_models.ipynb) | The same techniques on Ollama, no key, no cloud |

Four techniques are covered in the notes below but **have no notebook yet**:
Generated Knowledge, Reflexion, Prompt Ensembling, and Directional Stimulus
Prompting. Reflexion has a full implementation in the sibling
[ai-patterns](../ai-patterns/Reflexion.ipynb) folder.

## 3. Fundamentals

- **Clarity** - be specific and unambiguous.
- **Context** - supply the background the model cannot infer.
- **Constraints** - say what the answer may not do, not only what it should.
- **Examples** - show the shape you want; one good example beats a paragraph
  describing it.
- **Format** - state the output structure explicitly, and parse it defensively
  anyway.

### Optimising a prompt

1. **Iterative refinement** - start basic, change one thing at a time.
2. **A/B testing** - compare two versions on the same inputs, not on vibes.
3. **Prompt libraries** - keep what worked, with the task it worked on.
4. **Collaborative prompting** - other people find the failure cases you
   stopped seeing.

### Judging whether it worked

**Relevance** - does it answer the actual question. **Accuracy** - is it true.
**Coherence** - does it hold together. **Creativity** - for open tasks only.
**Efficiency** - how many turns did it take.

The one that matters most is the one nobody does: fix a set of inputs, write
down the expected outputs, and re-run them after every prompt change. Without
that you are comparing single samples of a non-deterministic system.

### Ethics

- **Bias** - prompts carry the assumptions of whoever wrote them.
- **Content safety** - guard the output, not just the input.
- **Privacy** - never put customer data in a prompt you would not put in a log.

## 4. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `model_not_found` / `model_decommissioned` on Groq | a retired model ID | use `openai/gpt-oss-20b`; see the note above |
| `401 Invalid API Key` | key pasted with a newline | re-copy from the provider console |
| `429 Too Many Requests` | free-tier rate limit | wait, or lower the sample count in Self-Consistency |
| `400 Tool choice is none, but model called a tool` | a tool-calling-native model against a text tool protocol | use `qwen/qwen3.6-27b`, as ART does |
| A cell returns success but prints nothing | some models put everything in a reasoning field | check `response.additional_kwargs`; try another model |
| `ImportError: langchain_groq` | first cell skipped | run the install cell |
| `ImportError` for openai / anthropic / google | that provider is optional | install only the provider you use |
| Ollama notebook cannot connect | Ollama not running | `ollama serve`, then `ollama pull llama3.2` |
| Output format ignored | small model, unlucky sample | re-run; the parsers fall back rather than crash |
| A `pip install` cell errors on a `#` | an old copy with a comment after a `\` continuation | fixed in this version; re-pull |

## 5. Honest limitations

- **These are demonstrations, not benchmarks.** Each notebook shows one
  technique on a handful of examples. A technique looking better here is one
  sample, not evidence.
- **Nothing is evaluated.** There is no scoring harness, so "this prompt is
  better" is a judgement you make by reading. Building the eval is the real
  next step, and it is the step most people skip.
- **Results are non-deterministic** at any temperature above 0, and several
  notebooks deliberately use a higher temperature.
- **All 14 Groq notebooks were run end to end against a live key.** Two
  needed real fixes to get there, and one uses `qwen/qwen3.6-27b` rather
  than `gpt-oss`: ART teaches a *text* tool protocol, and a tool-calling
  native model emits a real tool call that the API then rejects.
- **Cost is not shown.** Self-Consistency and Tree-of-Thoughts issue many
  calls per question. On a free tier that is a rate limit rather than a bill,
  but the arithmetic is the same.

## 6. Further reading

- [Prompt Engineering Guide](https://www.promptingguide.ai/introduction)
- [Coursera specialisation](https://www.coursera.org/specializations/prompt-engineering)
- [Kaggle whitepaper](https://www.kaggle.com/whitepaper-prompt-engineering)
