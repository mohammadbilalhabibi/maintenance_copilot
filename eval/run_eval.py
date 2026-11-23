"""Run the agent on the reference set and write measured results to eval/results/<name>.json.

Questions run one at a time (honest latency). Ticket questions stop at the confirmation
step and are never approved, so the evaluation writes nothing to the database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
import uuid
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from copilot.agent.llm import cost_usd, make_llm
from copilot.agent.runner import run_question
from copilot.config import PROJECT_ROOT, get_settings
from copilot.evaluation import grade, make_judge, summarize
from copilot.runtime import build_runtime

EVAL_DIR = PROJECT_ROOT / "eval"


async def run(args: argparse.Namespace) -> Path:
    """Evaluate one configuration; returns the results path."""
    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})
    for key in ("llm_price_input_per_m", "llm_price_cached_input_per_m", "llm_price_output_per_m"):
        if getattr(args, key) is not None:
            settings = settings.model_copy(update={key: getattr(args, key)})
    ref = json.loads(args.reference.read_text())
    items = ref["items"]
    if args.only:
        items = [i for i in items if i["category"] in args.only.split(",")]
    if args.limit:
        items = items[: args.limit]
    judge = make_judge(make_llm(settings, args.judge_model))

    records = []
    async with AsyncExitStack() as stack:
        rt = await build_runtime(settings, stack, checkpointer=InMemorySaver())
        graph = rt.graph
        if args.variant != "single":
            from copilot.agent.variants import build_variant

            graph = build_variant(args.variant, rt, settings)
        t_start = time.perf_counter()
        for n, item in enumerate(items, 1):
            config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}", "user_id": None},
                      "run_name": f"eval-{item['id']}"}
            try:
                result = await run_question(graph, item["question"], config)
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded as a failed item
                result, error = {"type": "error", "answer": "", "usage": {}}, repr(exc)[:300]
            usage = result.get("usage", {}) or {}
            record = {
                "id": item["id"], "category": item["category"], "question": item["question"],
                "answer": result.get("answer", ""), "result_type": result.get("type"),
                "tool_calls": result.get("tool_calls", []), "tickets": result.get("tickets"),
                "sources": result.get("sources", []), "cited": result.get("cited", []),
                "usage": usage, "cost_usd": round(cost_usd(usage, settings), 7),
                "latency_s": result.get("latency_s"), "error": error,
            }
            record["grade"] = await grade(item, result, judge)
            records.append(record)
            mark = "ok " if record["grade"].get("answer_ok") else "BAD"
            tools = [c["tool"] for c in record["tool_calls"]]
            print(f"[{n:2d}/{len(items)}] {mark} {item['id']:22s} "
                  f"{record['latency_s'] or 0:5.1f}s tools={tools}")
        wall = time.perf_counter() - t_start

    payload = {
        "name": args.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "variant": args.variant,
        "model": settings.llm_model,
        "judge_model": args.judge_model or settings.llm_model,
        "prices_usd_per_m": {"input": settings.llm_price_input_per_m,
                             "cached_input": settings.llm_price_cached_input_per_m,
                             "output": settings.llm_price_output_per_m},
        "reference_set": {"path": str(args.reference.relative_to(PROJECT_ROOT)), "n": ref["n"],
                          "seed": ref["seed"], "evaluated": len(items)},
        "wall_seconds": round(wall, 1),
        "machine": platform.processor(),
        "summary": summarize(records),
        "records": records,
    }
    out = EVAL_DIR / "results" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    s = payload["summary"]["overall"]
    print(f"\n{args.name}: correct {s['answer_correct']}, tools {s['tool_selection']}, "
          f"p50 {s['latency_p50_s']} s, p95 {s['latency_p95_s']} s, "
          f"cost/question ${s['cost_usd_mean']}, total ${s['cost_usd_total']} -> {out}")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="results file name")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--variant", default="single", help="single | multi | routed")
    parser.add_argument("--reference", type=Path, default=EVAL_DIR / "reference_set.json")
    parser.add_argument("--only", default=None, help="comma-separated categories")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm-price-input-per-m", type=float, default=None)
    parser.add_argument("--llm-price-cached-input-per-m", type=float, default=None)
    parser.add_argument("--llm-price-output-per-m", type=float, default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
