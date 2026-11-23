"""Ask the agent one question from the command line (prints tool calls, answer, sources)."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import AsyncExitStack

from langgraph.checkpoint.memory import InMemorySaver

from copilot.agent.llm import cost_usd
from copilot.agent.runner import stream_events
from copilot.config import get_settings
from copilot.runtime import build_runtime


async def run(question: str, model: str | None, approve: bool) -> None:
    """Stream one question and print what happens."""
    settings = get_settings()
    if model:
        settings = settings.model_copy(update={"llm_model": model})
    async with AsyncExitStack() as stack:
        rt = await build_runtime(settings, stack, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "cli", "user_id": None}}
        resume = None
        while True:
            async for ev in stream_events(rt.graph, question, config, resume=resume):
                if ev["type"] == "tool_start":
                    print(f"-> {ev['tool']}({json.dumps(ev['args'])})")
                elif ev["type"] == "tool_end" and not ev["ok"]:
                    print(f"   {ev['tool']} returned an error")
                elif ev["type"] == "confirm":
                    print(f"?? confirm ticket(s): {ev['tickets']} -> {'yes' if approve else 'no'}")
                    resume = {"approved": approve}
                elif ev["type"] == "done":
                    print("\n" + ev["answer"] + "\n")
                    for s in ev["sources"]:
                        mark = "*" if s["id"] in ev["cited"] else " "
                        print(f" {mark} {s['id']}: {s['label']}")
                    u = ev["usage"]
                    print(f"\n{ev['latency_s']:.1f} s, {u.get('llm_calls', 0)} LLM calls, "
                          f"{u.get('input', 0)} in / {u.get('output', 0)} out tokens, "
                          f"${cost_usd(u, settings):.5f}")
                    return


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", default=None)
    parser.add_argument("--approve", action="store_true", help="approve ticket creation")
    args = parser.parse_args()
    asyncio.run(run(args.question, args.model, args.approve))


if __name__ == "__main__":
    main()
