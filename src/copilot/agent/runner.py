"""Run the agent graph and turn its stream into simple events (used by API and eval).

Events (dicts with a "type" key):
  tool_start {tool, args}   tool_end {tool, ok}   token {text}
  confirm {tickets}         done {answer, sources, cited, usage, tools, latency_s}
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command

from copilot.agent.graph import new_question

CITATION = re.compile(r"\bS(\d+)\b")


def cited_ids(answer: str, sources: list[dict]) -> list[str]:
    """Source ids cited in square brackets in the answer that exist in `sources`."""
    known = {s["id"] for s in sources}
    found: list[str] = []
    for group in re.findall(r"\[([^\]]+)\]", answer):
        for num in CITATION.findall(group):
            sid = f"S{num}"
            if sid in known and sid not in found:
                found.append(sid)
    return found


async def stream_events(
    graph: Any,
    question: str | None,
    config: dict,
    *,
    resume: dict | None = None,
    durability: str | None = None,
) -> AsyncIterator[dict]:
    """Stream one question (or a resume after confirmation) as events."""
    t0 = time.perf_counter()
    payload: Any = Command(resume=resume) if resume is not None else new_question(question or "")
    tools_used: list[dict] = []
    first_token_s: float | None = None
    interrupted: dict | None = None
    extra = {"durability": durability} if durability else {}
    stream = graph.astream(payload, config, stream_mode=["messages", "updates"], subgraphs=True,
                           **extra)
    async for _namespace, mode, data in stream:
        if mode == "messages":
            chunk, meta = data
            if (
                meta.get("langgraph_node") == "agent"
                and isinstance(chunk, AIMessageChunk)
                and chunk.content
                and not chunk.tool_call_chunks
            ):
                if first_token_s is None:
                    first_token_s = time.perf_counter() - t0
                yield {"type": "token", "text": chunk.content}
            continue
        for node, update in data.items():
            if node == "__interrupt__":
                if interrupted is None:  # also reported once per enclosing graph level
                    interrupted = update[0].value
                    yield {**interrupted, "type": "confirm"}
            elif node == "agent" and isinstance(update, dict):
                for msg in update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        for call in msg.tool_calls:
                            yield {"type": "tool_start", "tool": call["name"],
                                   "args": call["args"]}
            elif node == "tools" and isinstance(update, dict):
                for msg in update.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        ok = msg.status != "error"
                        tools_used.append({"tool": msg.name, "ok": ok})
                        yield {"type": "tool_end", "tool": msg.name, "ok": ok}
    if interrupted is not None:
        return
    state = (await graph.aget_state(config)).values
    final = state["messages"][-1]
    answer = final.content if isinstance(final, AIMessage) else ""
    sources = state.get("sources", [])
    yield {
        "type": "done",
        "answer": answer,
        "sources": sources,
        "cited": cited_ids(answer, sources),
        "usage": state.get("usage", {}),
        "tools": tools_used,
        "latency_s": round(time.perf_counter() - t0, 3),
        "first_token_s": round(first_token_s, 3) if first_token_s is not None else None,
    }


async def run_question(graph: Any, question: str, config: dict) -> dict:
    """Run one question to completion (or until a confirmation is needed).

    Returns the 'done' event, or {"type": "confirm", ...} plus the tools started so far.
    """
    started: list[dict] = []
    last: dict = {"type": "error", "message": "no result"}
    t0 = time.perf_counter()
    async for event in stream_events(graph, question, config):
        if event["type"] == "tool_start":
            started.append({"tool": event["tool"], "args": event["args"]})
        elif event["type"] in ("done", "confirm"):
            event["tool_calls"] = started
            if event["type"] == "confirm":
                state = (await graph.aget_state(config)).values
                event["usage"] = state.get("usage", {})
                event["latency_s"] = round(time.perf_counter() - t0, 3)
            last = event
    return last
