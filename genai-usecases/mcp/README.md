# Model Context Protocol (MCP)

> **Learn how to build these projects step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)**. Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

**MCP is the answer to a question every tool-using LLM app eventually asks:
why does every model provider need its own bespoke integration for the same
tool?**

---

## What MCP actually solves

Before MCP, wiring a model to a tool meant writing that integration twice -
once against OpenAI's function-calling shape, once against Anthropic's, once
more for Gemini. `N` models times `M` tools is `N x M` integrations, and every
new model multiplied the work.

MCP inverts it. A tool is exposed **once**, by an MCP **server**, in a
standard shape. Any MCP **client** can then use it, whichever model sits behind
that client. `N + M` instead of `N x M`.

```
   Model / LLM app                 MCP client              MCP server
  (Gemini, Claude, GPT)  <----->  (speaks MCP)  <----->  (owns the tool)
                                                          web search, files,
                                                          a database, an API
```

The server owns the credentials and the actual work. The client owns the
conversation. Neither needs to know how the other is implemented, which is the
whole point.

Three things a server can offer:

| Primitive | What it is | In this project |
|---|---|---|
| **Tools** | Functions the model may call | `web_search` |
| **Resources** | Data the client can read | - |
| **Prompts** | Reusable prompt templates | - |

[Official MCP documentation](https://modelcontextprotocol.io/docs/getting-started/intro)

## Projects in this folder

### [`web_search_mcp/`](./web_search_mcp/) - Web Search MCP server + Gemini client

A complete, runnable pair: an MCP **server** that exposes live web search as a
tool, and a Gemini-powered MCP **client** with a Streamlit interface that
decides when to call it.

It is the smallest example that still shows the real shape of MCP: the model
does not search, and the server does not converse. The client passes the tool
definition to Gemini, Gemini decides a search is needed and returns a tool
call, the client routes it to the server, and the result comes back into the
conversation.

**What you need:** a `GEMINI_API_KEY` ([free tier](https://aistudio.google.com/app/apikey))
and a `SERPAPI_API_KEY` ([free tier](https://serpapi.com/manage-api-key)).

See [`web_search_mcp/README.md`](./web_search_mcp/README.md) for setup and how
it works.

## Where MCP fits alongside the rest of this repo

| If you want | Look at |
|---|---|
| A model calling tools you define in-process | [`../agentic-ai/`](../agentic-ai/) |
| A model calling tools over a standard protocol | **here** |
| Retrieval instead of tools | [`../advance-rag/`](../advance-rag/) |
| The reasoning patterns behind tool use | [`../ai-patterns/`](../ai-patterns/) (ReAct, Toolformer) |

MCP is not a replacement for function calling - it is a transport and a
contract around it. If your tool lives in the same process and only one app
will ever use it, plain function calling is less machinery. MCP earns its keep
when the tool is shared, remote, or owned by someone else.
