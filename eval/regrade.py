"""Re-grade stored answers of a results file with the current grader (no agent rerun).

Used when the grader itself had a bug; the agent's answers are unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from copilot.agent.llm import make_llm
from copilot.config import get_settings
from copilot.evaluation import grade, make_judge, summarize


async def run(path: Path, reference: Path, judge_model: str | None) -> None:
    """Re-grade every record in `path` in place."""
    data = json.loads(path.read_text())
    ref = {i["id"]: i for i in json.loads(reference.read_text())["items"]}
    if judge_model:
        data["judge_model"] = judge_model
    judge = make_judge(make_llm(get_settings(), data.get("judge_model")))
    for r in data["records"]:
        result = {"type": r["result_type"], "answer": r["answer"], "tool_calls": r["tool_calls"],
                  "tickets": r.get("tickets"), "sources": r["sources"], "cited": r["cited"]}
        r["grade"] = await grade(ref[r["id"]], result, judge)
    data["summary"] = summarize(data["records"])
    data["regraded"] = True
    path.write_text(json.dumps(data, indent=2, default=str))
    s = data["summary"]["overall"]
    print(f"{path.name}: correct {s['answer_correct']}, tools {s['tool_selection']}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).parent / "reference_set.json")
    parser.add_argument("--judge-model", default=None, help="override the judge model")
    args = parser.parse_args()
    asyncio.run(run(args.results, args.reference, args.judge_model))


if __name__ == "__main__":
    main()
