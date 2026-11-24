"""LangGraph agent: LLM node <-> tools node, with a human confirmation step for tickets.

Flow per question:  agent --(tool calls)--> tools --> agent ... --> END
- The tools node runs read-only tools concurrently with a timeout each, turns errors into
  tool messages the LLM can explain, and numbers every result as a source (S1, S2, ...).
- If the LLM asks for create_ticket, the tools node first pauses the graph with
  `interrupt()`. The API shows the ticket to the user; the graph resumes with
  {"approved": true/false}. Nothing is executed before that answer, so re-running the
  node on resume has no side effects.
- A step limit stops tool loops: at the limit the LLM must answer without tools.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import interrupt
from opentelemetry import trace
from pydantic import ValidationError

from copilot.agent.llm import usage_from_message
from copilot.agent.provenance import describe
from copilot.agent.tools import TOOLS, TOOLS_BY_NAME, ToolContext, ToolError
from copilot.db.querylog import QUERY_LOG

RESET = "reset"
_tracer = trace.get_tracer("copilot.tools")


def merge_sources(left: list[dict] | None, right: list[dict] | str | None) -> list[dict]:
    """Append new sources; the string RESET clears them (start of a new question)."""
    if right == RESET:
        return []
    return [*(left or []), *(right or [])]


def merge_usage(left: dict | None, right: dict | str | None) -> dict:
    """Sum token usage; RESET clears it (start of a new question)."""
    if right == RESET:
        return {}
    out = dict(left or {})
    for k, v in (right or {}).items():
        out[k] = out.get(k, 0) + v
    return out


class AgentState(TypedDict, total=False):
    """Graph state. `sources`, `usage` and `steps` are per question."""

    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[list[dict], merge_sources]
    usage: Annotated[dict, merge_usage]
    steps: int


def new_question(text: str) -> dict:
    """Graph input for a new user question (resets per-question fields)."""
    return {"messages": [("user", text)], "sources": RESET, "usage": RESET, "steps": 0}


def langchain_tools(names: list[str] | None = None) -> list[StructuredTool]:
    """Tool schemas for bind_tools (execution happens in the tools node, not here)."""

    def _never(**_: Any) -> str:  # pragma: no cover - schema only
        raise RuntimeError("executed by the tools node")

    return [
        StructuredTool.from_function(
            func=_never, name=t.name, description=t.description, args_schema=t.args
        )
        for t in TOOLS
        if names is None or t.name in names
    ]


def _source_label(name: str, args: dict) -> str:
    machine = f"machine {args['machine_id']}" if "machine_id" in args else "fleet"
    return {
        "get_machine_status": f"Status of {machine} (fleet DB)",
        "query_telemetry": f"Telemetry of {machine} (fleet DB)",
        "get_error_history": f"Error log of {machine} (fleet DB)",
        "get_maintenance_history": f"Maintenance log of {machine} (fleet DB)",
        "predict_failure_risk": f"Failure-risk model for {machine}",
        "fleet_overview": f"Fleet ranking by {args.get('metric', '?')} (fleet DB)",
        "create_ticket": f"Ticket for {machine}",
    }.get(name, name)


MAX_SOURCE_DATA_CHARS = 6000


def _source_data(result: dict) -> dict:
    """Tool output kept with the source so the UI can show what the agent saw."""
    if len(json.dumps(result, default=str)) <= MAX_SOURCE_DATA_CHARS:
        return result
    return {"note": "Output too large to store in full.",
            "keys": list(result)[:30]}


async def run_tool(
    ctx: ToolContext, name: str, raw_args: dict, config: RunnableConfig | None = None
) -> tuple[dict | None, str | None]:
    """Validate arguments and run one tool in a TOOL span. Returns (result, error message)."""
    with _tracer.start_as_current_span(f"tool {name}") as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("tool.name", name)
        span.set_attribute("input.value", json.dumps(raw_args, default=str))
        token = QUERY_LOG.set([])
        try:
            result, error = await _run_tool(ctx, name, raw_args, config)
            queries = QUERY_LOG.get() or []
        finally:
            QUERY_LOG.reset(token)
        if result is not None:
            result = {**result, "_queries": queries}  # removed before the LLM sees it
        span.set_attribute("output.value",
                           json.dumps(result if error is None else {"error": error},
                                      default=str)[:4000])
        if error is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, error[:200]))
        return result, error


async def _run_tool(
    ctx: ToolContext, name: str, raw_args: dict, config: RunnableConfig | None
) -> tuple[dict | None, str | None]:
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return None, f"Unknown tool {name}."
    try:
        args = spec.args.model_validate(raw_args)
    except ValidationError as exc:
        return None, f"Invalid arguments for {name}: {exc.errors(include_url=False)}"
    extra: dict[str, Any] = {}
    if spec.writes:
        conf = (config or {}).get("configurable", {})
        extra = {"user_id": conf.get("user_id"), "conversation_id": conf.get("thread_id")}
    try:
        result = await asyncio.wait_for(spec.fn(ctx, args, **extra), timeout=ctx.timeout)
        return result, None
    except ToolError as exc:
        return None, str(exc)
    except TimeoutError:
        return None, f"{name} timed out after {ctx.timeout:.0f} s."
    except Exception as exc:  # noqa: BLE001 - reported to the LLM, logged in traces
        return None, f"{name} failed: {type(exc).__name__}"


def build_graph(
    llm: BaseChatModel,
    ctx: ToolContext,
    system_prompt: str,
    *,
    max_steps: int = 8,
    max_history_tokens: int = 6000,
    checkpointer: Any = None,
    tool_names: list[str] | None = None,
    name: str | None = None,
):  # noqa: ANN201 - compiled graph type is internal to langgraph
    """Compile the agent graph (optionally with a subset of the tools)."""
    tools = langchain_tools(tool_names)
    llm_tools = llm.bind_tools(tools)

    async def agent(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        history = trim_messages(
            state["messages"],
            max_tokens=max_history_tokens,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            include_system=False,
        )
        messages = [SystemMessage(system_prompt), *history]
        if steps >= max_steps:
            messages.append(
                SystemMessage("Tool limit reached. Answer now with the information you have.")
            )
            response = await llm.ainvoke(messages)
            if getattr(response, "tool_calls", None):  # never leave unanswered tool calls
                response = AIMessage(
                    content=response.content or "I could not finish within the tool limit.",
                    usage_metadata=response.usage_metadata,
                )
        else:
            response = await llm_tools.ainvoke(messages)
        return {"messages": [response], "usage": usage_from_message(response), "steps": steps + 1}

    async def tools_node(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", []) or [])
        decision: dict | None = None
        ticket_calls = [c for c in calls if c["name"] == "create_ticket"]
        if ticket_calls:
            # Pause BEFORE executing anything; resumes with {"approved": bool}.
            decision = interrupt(
                {"type": "confirm_ticket", "tickets": [c["args"] for c in ticket_calls]}
            )
        n_sources = len(state.get("sources") or [])
        new_sources: list[dict] = []
        events: list[dict] = []

        async def execute(call: dict) -> tuple[dict, dict | None, str | None, float]:
            t0 = time.perf_counter()
            if call["name"] == "create_ticket" and not (decision or {}).get("approved"):
                return call, None, "The user declined to create this ticket.", 0.0
            result, error = await run_tool(ctx, call["name"], call["args"], config)
            return call, result, error, time.perf_counter() - t0

        outputs = await asyncio.gather(*(execute(c) for c in calls))
        messages = []
        for call, result, error, seconds in outputs:
            events.append({"tool": call["name"], "args": call["args"], "ok": error is None,
                           "seconds": round(seconds, 3)})
            if error is not None:
                messages.append(ToolMessage(
                    content=json.dumps({"error": error}), tool_call_id=call["id"],
                    name=call["name"], status="error",
                ))
                continue
            # UI-only provenance; the LLM never sees these keys.
            queries = result.pop("_queries", [])
            if call["name"] == "search_manuals":
                # Provenance for the UI only; the LLM sees doc, section and text.
                meta = result.pop("_meta", {})
                for passage in result.get("passages", []):
                    n_sources += 1
                    passage["source_id"] = f"S{n_sources}"
                    new_sources.append({
                        "id": passage["source_id"], "tool": "search_manuals",
                        "label": f"{passage['doc']} - {passage['section']}",
                        "chunk_id": passage["chunk_id"], "doc": passage["doc"],
                        "section": passage["section"], "text": passage["text"],
                        **meta.get(passage["chunk_id"], {}),
                    })
            else:
                n_sources += 1
                new_sources.append({
                    "id": f"S{n_sources}", "tool": call["name"],
                    "label": _source_label(call["name"], call["args"]), "args": call["args"],
                    "data": _source_data(result),
                    "origin": describe(call["name"], call["args"], result),
                    "queries": queries,
                })
                result = {"source_id": f"S{n_sources}", **result}
            messages.append(ToolMessage(
                content=json.dumps(result, default=str, separators=(",", ":")),
                tool_call_id=call["id"], name=call["name"],
            ))
        return {"messages": messages, "sources": new_sources,
                "usage": {"tool_calls": len(calls)}}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer, name=name)
