"""The catalogue of apps, and how to run each one.

Thirteen apps used to be thirteen Streamlit scripts, each carrying its own
copy of the provider list and its own `create_llm`. That is why the Anthropic
model list went stale in thirteen places at once. They are now service modules
behind one registry, so the API has one route and the UI has one form builder.

Each entry declares its inputs. The API validates against that declaration and
the UI renders from it, so adding an app is one entry here rather than an edit
in three layers.

Every runner is async, because every underlying workflow is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class Field:
    """One input an app needs, described well enough for a UI to draw it."""

    name: str
    label: str
    kind: str = "text"            # text | textarea | select | number | bool
    default: object = ""
    options: list[str] = field(default_factory=list)
    help: str = ""
    required: bool = True


@dataclass
class App:
    id: str
    label: str
    family: str
    blurb: str
    fields: list[Field]
    runner: Callable[..., Awaitable[dict]]


def _llm(provider: str, model: str, ollama_base_url: str | None = None,
         **kwargs):
    from services.providers import create_llm

    return create_llm(provider, model, ollama_base_url, **kwargs)


# --------------------------------------------------------------------------
# Runners: one dict-in, dict-out coroutine per app.
# --------------------------------------------------------------------------

async def run_query_routing(provider, model, ollama_base_url=None, *, query, **_):
    from services.apps.query_routing import QueryRouter

    router = QueryRouter(_llm(provider, model, ollama_base_url))
    return {"result": await router.route_and_process(query)}


async def run_prompt_chaining(provider, model, ollama_base_url=None, *,
                              topic, steps=None, execution_type="sequential", **_):
    from services.apps.prompt_chaining import PromptChain

    chain = PromptChain(_llm(provider, model, ollama_base_url), use_memory=False)
    # Step prompts are plain instructions with NO placeholders. The chain
    # supplies the topic and the previous step's output itself, so a literal
    # "{topic}" here collides with its template variables and the run fails
    # with "Input to ChatPromptTemplate is missing variables {'topic'}".
    chain_steps = steps or [
        {"name": "Step 1: Analysis",
         "prompt": "Analyse the given topic and identify the key components, "
                   "challenges and opportunities"},
        {"name": "Step 2: Strategy",
         "prompt": "Based on the analysis, develop a strategy with clear "
                   "objectives"},
        {"name": "Step 3: Implementation",
         "prompt": "Create an implementation plan with a timeline and the "
                   "resources it needs"},
        {"name": "Step 4: Evaluation",
         "prompt": "Define evaluation metrics and success criteria for that "
                   "plan"},
    ]
    return {"result": await chain.execute_lcel_chain(
        chain_steps, topic, execution_type
    )}


async def run_parallel_execution(provider, model, ollama_base_url=None, *,
                                 topic, **_):
    from services.apps.parallel_execution import ParallelExecutor, ParallelTask

    executor = ParallelExecutor(_llm(provider, model, ollama_base_url))
    prompts = [
        ("facts", f"Summarise the key facts about: {topic}"),
        ("risks", f"List the main risks or downsides of: {topic}"),
        ("benefits", f"List the main benefits or opportunities of: {topic}"),
    ]
    tasks = [
        # `context` has no default on the dataclass, so it must be supplied.
        ParallelTask(task_id=name, name=name.title(), prompt=prompt,
                     context={"topic": topic}, priority=1, timeout=120)
        for name, prompt in prompts
    ]
    summary = await executor.execute_tasks_parallel(tasks)
    return {"result": summary}


async def run_event_driven(provider, model, ollama_base_url=None, *,
                           initial_input, agent_count=2, **_):
    from services.apps.event_driven import EventDrivenWorkflow

    workflow = EventDrivenWorkflow(_llm(provider, model, ollama_base_url))

    # Reactive agents subscribe to the bus when they are created. Without at
    # least one, start_workflow publishes its events into a bus nobody is
    # listening to: the call succeeds, and nothing happens.
    for index in range(max(1, int(agent_count))):
        workflow.add_agent(f"agent_{index + 1}")

    # start_workflow publishes events and returns None by design; the output
    # accumulates in the workflow's own state. Returning its return value
    # gave the caller a null result and looked like a silent failure.
    await workflow.start_workflow(initial_input)
    stats = workflow.get_workflow_stats()
    return {
        "result": {
            "initial_input": initial_input,
            "workflow_state": stats["workflow_state"],
            "total_events": stats["total_events"],
            "recent_events": stats["recent_events"],
            "execution_log": workflow.execution_log,
            "last_response": workflow.state.get("last_response")
            if hasattr(workflow, "state") else None,
        },
        "stats": stats,
    }


async def run_tool_orchestration(provider, model, ollama_base_url=None, *,
                                 objective, execute=True, **_):
    from services.apps.tool_orchestration import ToolOrchestrator, ToolRegistry

    orchestrator = ToolOrchestrator(
        _llm(provider, model, ollama_base_url), ToolRegistry()
    )
    plan = await orchestrator.plan_workflow(objective, None)
    payload = {"plan": plan}
    if execute:
        payload["execution"] = await orchestrator.execute_workflow(plan)
    return {"result": payload}


async def run_document_processing(provider, model, ollama_base_url=None, *,
                                  content, filename="document.txt",
                                  output_fmt="markdown", **_):
    import asyncio

    from services.apps.document_processing import process_document

    filetype = filename.rsplit(".", 1)[-1] if "." in filename else "txt"
    # The pipeline is synchronous; keep it off the event loop.
    result = await asyncio.to_thread(
        process_document, content, filename, filetype, output_fmt,
        provider, model, 0.1, ollama_base_url,
    )
    return {"result": result}


REGISTRY: dict[str, App] = {
    "query_routing": App(
        id="query_routing", label="Query routing", family="Workflow patterns",
        blurb="Score a query against routing rules, then send it to the handler "
              "that fits. The cheapest way to stop one prompt trying to be five.",
        fields=[Field("query", "Your query", "textarea",
                      "My invoice is wrong and I would like a refund")],
        runner=run_query_routing,
    ),
    "prompt_chaining": App(
        id="prompt_chaining", label="Prompt chaining", family="Workflow patterns",
        blurb="Break one hard task into steps, where each step consumes the "
              "previous step's output.",
        fields=[
            Field("topic", "Topic", "textarea",
                  "Migrating a monolith to microservices"),
            Field("execution_type", "Execution type", "select", "sequential",
                  ["sequential", "parallel"], required=False),
        ],
        runner=run_prompt_chaining,
    ),
    "parallel_execution": App(
        id="parallel_execution", label="Parallel execution", family="Workflow patterns",
        blurb="Fan independent prompts out at once and collect them. Costs the "
              "latency of the slowest, not the sum of all of them.",
        fields=[Field("topic", "Topic", "textarea",
                      "Adopting Kubernetes at a 50-person company")],
        runner=run_parallel_execution,
    ),
    "event_driven": App(
        id="event_driven", label="Event driven", family="Workflow patterns",
        blurb="Emit events and let handlers react, instead of hard-coding what "
              "happens next.",
        fields=[
            Field("initial_input", "Input", "textarea",
                  "Analyse last quarter's support tickets"),
            Field("agent_count", "Reactive agents", "number", 2,
                  help="Agents subscribe to the event bus when created. "
                       "With none, events go nowhere.", required=False),
        ],
        runner=run_event_driven,
    ),
    "tool_orchestration": App(
        id="tool_orchestration", label="Tool orchestration", family="Workflow patterns",
        blurb="The model plans which registered tools to call and in what "
              "order, then the plan is executed.",
        fields=[
            Field("objective", "Objective", "textarea",
                  "Work out 17% of 2,400 and explain what it implies"),
            Field("execute", "Execute the plan, not just build it", "bool",
                  True, required=False),
        ],
        runner=run_tool_orchestration,
    ),
    "document_processing": App(
        id="document_processing", label="Document processing", family="LangGraph",
        blurb="A LangGraph pipeline: parse, analyse, validate, summarise, "
              "format - with an error path that routes rather than throws.",
        fields=[
            Field("content", "Document text", "textarea",
                  "Quarterly report. Revenue grew 12% to $4.2M. Headcount rose "
                  "from 34 to 41. Churn fell to 3.1%."),
            Field("filename", "File name", "text", "report.txt", required=False),
            Field("output_fmt", "Output format", "select", "markdown",
                  ["markdown", "json", "text"], required=False),
        ],
        runner=run_document_processing,
    ),
}


def get(app_id: str) -> App:
    if app_id not in REGISTRY:
        raise KeyError(app_id)
    return REGISTRY[app_id]


def catalogue() -> list[dict]:
    return [
        {
            "id": app.id, "label": app.label, "family": app.family,
            "blurb": app.blurb,
            "fields": [
                {"name": f.name, "label": f.label, "kind": f.kind,
                 "default": f.default, "options": f.options,
                 "help": f.help, "required": f.required}
                for f in app.fields
            ],
        }
        for app in REGISTRY.values()
    ]
