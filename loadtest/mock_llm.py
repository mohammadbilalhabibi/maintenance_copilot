"""Mock OpenAI-compatible chat completions server (no API key, deterministic).

Used for the backend load test (measures our API, not the LLM provider) and for the
Docker Compose smoke test in CI. Behaviour: on a new question it asks for one tool call
(machine status if a machine number is mentioned, else document search); once tool results
are present it answers citing [S1]. MOCK_LATENCY_MS adds a delay per completion to mimic
a real model (default 600 ms); MOCK_TOKEN_DELAY_MS spaces streamed tokens.

Run: uvicorn loadtest.mock_llm:app --port 9000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock-llm")
LATENCY = float(os.getenv("MOCK_LATENCY_MS", "600")) / 1000
TOKEN_DELAY = float(os.getenv("MOCK_TOKEN_DELAY_MS", "10")) / 1000
USAGE = {"prompt_tokens": 1500, "completion_tokens": 40, "total_tokens": 1540,
         "prompt_tokens_details": {"cached_tokens": 1024}}


def decide(messages: list[dict], tools: list[dict] | None) -> tuple[str, list[dict]]:
    """Return (content, tool_calls) for the conversation so far."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
    tool_results = [m for m in messages[last_user:] if m.get("role") == "tool"]
    question = str(messages[last_user].get("content", "")) if messages else ""
    if tools and not tool_results:
        m = re.search(r"machine\s+(\d+)", question, re.I)
        if m:
            name, args = "get_machine_status", {"machine_id": int(m.group(1))}
        else:
            name, args = "search_manuals", {"query": question[:200] or "maintenance"}
        call = {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        return "", [call]
    return ("Here is what I found in the fleet data and documents [S1]. "
            "This answer comes from the mock LLM used for load and smoke tests."), []


async def stream(content: str, tool_calls: list[dict], model: str, usage: bool
                 ) -> AsyncIterator[str]:
    """OpenAI-style SSE chunks."""
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> str:
        body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(body)}\n\n"

    await asyncio.sleep(LATENCY)
    yield chunk({"role": "assistant", "content": ""})
    if tool_calls:
        for i, call in enumerate(tool_calls):
            yield chunk({"tool_calls": [{"index": i, **call}]})
        yield chunk({}, "tool_calls")
    else:
        for word in content.split(" "):
            yield chunk({"content": word + " "})
            await asyncio.sleep(TOKEN_DELAY)
        yield chunk({}, "stop")
    if usage:
        body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [], "usage": USAGE}
        yield f"data: {json.dumps(body)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def completions(request: Request):  # noqa: ANN201
    """Chat completions endpoint (streaming and non-streaming)."""
    body = await request.json()
    content, tool_calls = decide(body.get("messages", []), body.get("tools"))
    model = body.get("model", "mock")
    if body.get("stream"):
        usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(stream(content, tool_calls, model, usage),
                                 media_type="text/event-stream")
    await asyncio.sleep(LATENCY)
    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": USAGE,
    })


@app.get("/health")
async def health() -> dict:
    """Liveness."""
    return {"status": "ok"}
