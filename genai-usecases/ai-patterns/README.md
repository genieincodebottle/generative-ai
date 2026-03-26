# AI Reasoning Patterns

23 hands-on Jupyter notebooks demonstrating advanced AI reasoning and agentic patterns. Each notebook is self-contained and runs on Google Colab (no local setup needed).

## Quick Start

Click any notebook below to open it in Google Colab, or run locally with `jupyter notebook`.

## Patterns

| # | Pattern | Notebook | Description |
|---|---------|----------|-------------|
| 1 | Chain-of-Thought | `Chain-of-Thought.ipynb` | Step-by-step reasoning before answering |
| 2 | Tree of Thought | `Tree-of-Thought.ipynb` | Explore multiple reasoning branches |
| 3 | Graph of Thoughts | `Graph-of-Thoughts.ipynb` | Non-linear reasoning with graph structure |
| 4 | Skeleton of Thought | `Skeleton-of-Thought.ipynb` | Generate outline first, then fill in details |
| 5 | Chain of Verification | `Chain-of-Verification.ipynb` | Verify answers with follow-up questions |
| 6 | ReAct | `ReAct.ipynb` | Reason + Act loop with tool use |
| 7 | Reflexion | `Reflexion.ipynb` | Self-reflection to improve responses |
| 8 | Self-Refine | `Self-Refine.ipynb` | Iterative self-improvement of outputs |
| 9 | Recursive Criticism | `Recursive-Criticism-and-Improvement.ipynb` | Critique and revise in a loop |
| 10 | Least-to-Most | `Least-to-Most-Prompting.ipynb` | Decompose into sub-problems, solve incrementally |
| 11 | Decomposed Prompting | `Decomposed-Prompting.ipynb` | Break complex tasks into simpler sub-tasks |
| 12 | Plan and Solve | `Plan-and-Solve.ipynb` | Create a plan, then execute step-by-step |
| 13 | Reasoning via Planning | `Reasoning-via-Planning.ipynb` | Use planning as a reasoning strategy |
| 14 | Meta-Prompting | `Meta-Prompting.ipynb` | LLM generates its own prompts |
| 15 | RAG | `RAG.ipynb` | Retrieval-Augmented Generation |
| 16 | Toolformer | `Toolformer.ipynb` | LLM learns when and how to use tools |
| 17 | Auto Reasoning + Tools | `Automatic-Reasoning-and-Tool-Use.ipynb` | Automatic tool selection and reasoning |
| 18 | Multi-Agent Debate | `Multi-Agent-Debate.ipynb` | Multiple agents argue to reach better answers |
| 19 | Orchestrator-Worker | `Orchestrator-Worker.ipynb` | One agent coordinates, others execute |
| 20 | Generative Agents | `Generative-Agents.ipynb` | Simulated agents with memory and reflection |
| 21 | Self-Evolving Agent | `Self-Evolving-Agent.ipynb` | Agent that improves itself over time |
| 22 | Language Agent Tree Search | `Language-Agent-Tree-Search.ipynb` | Tree search for optimal agent actions |
| 23 | Model Context Protocol | `Model-Context-Protocol.ipynb` | MCP standard for LLM tool interoperability |

## Reference

See `ai-patterns.pdf` for a visual guide covering all patterns.

## Run Locally (Optional)

```bash
pip install jupyter langchain-google-genai python-dotenv
jupyter notebook
```

You'll need a free Google Gemini API key: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
