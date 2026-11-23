"""Generate the README result tables from the results JSON files (no hand-typed numbers).

Replaces the text between <!-- TABLE:<name> --> and <!-- /TABLE:<name> --> in README.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN = "v2_prompt_gpt-4o-mini"
RUN_LABELS = {
    "v1_baseline_gpt-4o-mini": "v1 single agent, first prompt",
    "v2_prompt_gpt-4o-mini": "v2 single agent (main)",
    "v2_prompt_gpt-4.1": "v2 single agent, large model",
    "v2_multiagent_gpt-4o-mini": "v2 router + 4 specialists",
    "v2_routed_mini-to-4.1": "v2 router: simple -> mini, complex -> 4.1",
}
CATEGORY_LABELS = {
    "fact": "Machine facts (SQL ground truth)", "fleet": "Fleet rankings",
    "risk": "Failure-risk questions", "document": "Handbook / safety guides",
    "multi": "Multi-tool questions", "unanswerable": "Unanswerable / off-topic (refusal)",
    "ticket": "Ticket requests (correct ticket proposed)",
}


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def eval_runs() -> dict[str, dict]:
    runs = {}
    for p in sorted((ROOT / "eval" / "results").glob("*.json")):
        if p.stem != "retrieval":
            runs[p.stem] = load(p)
    return runs


def table_headline(runs: dict) -> str:
    s = runs[MAIN_RUN]["summary"]["overall"]
    n = runs[MAIN_RUN]["reference_set"]["evaluated"]
    rows = [
        ("Answer correct (79 reference questions)", pct(s["answer_correct"])),
        ("Right tools called", pct(s["tool_selection"])),
        ("Answers citing a source", pct(s["cited_any"])),
        ("Unrequested database writes", str(s["unrequested_writes"])),
        ("Citations to non-existent sources", str(s["invalid_citations"])),
        ("Latency p50 / p95", f"{s['latency_p50_s']} s / {s['latency_p95_s']} s"),
        ("Cost per question", f"${s['cost_usd_mean']:.5f}"),
    ]
    out = [f"| Metric ({runs[MAIN_RUN]['model']}, n={n}) | Value |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(out)


def table_categories(runs: dict) -> str:
    by = runs[MAIN_RUN]["summary"]["by_category"]
    base = runs.get("v1_baseline_gpt-4o-mini", {}).get("summary", {}).get("by_category", {})
    out = ["| Question type | n | v1 correct | v2 correct | v2 right tools |",
           "|---|---|---|---|---|"]
    for cat, label in CATEGORY_LABELS.items():
        if cat in by:
            b = base.get(cat, {}).get("answer_correct")
            out.append(f"| {label} | {by[cat]['n']} | {pct(b)} | {pct(by[cat]['answer_correct'])} "
                       f"| {pct(by[cat]['tool_selection'])} |")
    return "\n".join(out)


def table_experiments(runs: dict) -> str:
    out = ["| Setup | Model | Correct | Right tools | p50 | p95 | LLM calls | Tokens | "
           "Cached input | Cost / question |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, label in RUN_LABELS.items():
        if name not in runs:
            continue
        r = runs[name]
        s = r["summary"]["overall"]
        model = r["model"] if r["variant"] != "routed" else "mini + 4.1"
        out.append(
            f"| {label} | {model} | {pct(s['answer_correct'])} | {pct(s['tool_selection'])} | "
            f"{s['latency_p50_s']} s | {s['latency_p95_s']} s | {s['llm_calls_mean']} | "
            f"{s['tokens_mean']:.0f} | {pct(s['cached_input_share'])} | ${s['cost_usd_mean']:.5f} |"
        )
    return "\n".join(out)


def table_model() -> str:
    m = load(ROOT / "results" / "model_metrics.json")
    out = ["| Horizon | Model | PR-AUC | Precision | Recall | Recall >24 h before failure |",
           "|---|---|---|---|---|---|"]
    for h, hv in m["horizons"].items():
        base = hv["positive_rate"]["test"]
        for name in ("random", "rule", "logreg", "lightgbm"):
            t = hv["models"][name]["test"]
            label = {"random": "random", "rule": "error-count rule",
                     "logreg": "logistic regression", "lightgbm": "**LightGBM**"}[name]
            early = "-" if t["recall_early"] != t["recall_early"] else f"{t['recall_early']:.3f}"
            setup = f"{h} (base rate {base:.3f}, alerts on {hv['alert_rate']:.0%})"
            out.append(f"| {setup} | {label} | {t['pr_auc_any']:.3f} | "
                       f"{t['precision_at_rate']:.3f} | {t['recall_at_rate']:.3f} | {early} |")
    return "\n".join(out)


def table_retrieval() -> str:
    r = load(ROOT / "eval" / "results" / "retrieval.json")
    k = r["k"]
    out = [f"| Retriever (n={r['n_questions']} questions, {r['n_chunks']} chunks) | hit@1 | "
           f"hit@{k} | MRR | p50 latency |", "|---|---|---|---|---|"]
    for name, v in r["retrievers"].items():
        out.append(f"| {name} | {v['hit_at_1']:.2f} | {v[f'hit_at_{k}']:.2f} | "
                   f"{v['mrr_at_20']:.2f} | {v['latency_ms_p50']:.1f} ms |")
    return "\n".join(out)


def table_load() -> str:
    out = []
    for p in sorted((ROOT / "loadtest" / "results").glob("*.json")):
        r = load(p)
        out.append(f"**{r['name']}** - {r['note']}\n")
        out.append("| Users | Chat requests | Failures | Chat p50 | Chat p95 | First event p50 | "
                   "Requests/s (all) |")
        out.append("|---|---|---|---|---|---|---|")
        for step in r["steps"]:
            e = step["endpoints"]
            chat = e.get("chat (full answer)", {})
            first = e.get("chat first event", {})
            http = {k: v for k, v in e.items() if k not in ("chat first event",
                                                               "chat (full answer)")}
            rps = sum(v["rps"] or 0 for v in http.values())
            # A failed chat shows up in "chat (headers)" and "chat (full answer)": count it once.
            fails = chat.get("failures", 0) + sum(
                v["failures"] for k, v in http.items() if k != "chat (headers)")
            p50, p95 = (chat.get("median_ms") or 0) / 1000, (chat.get("p95_ms") or 0) / 1000
            first_p50 = (first.get("median_ms") or 0) / 1000
            out.append(f"| {step['users']} | {chat.get('requests', 0)} | {fails} | {p50:.2f} s | "
                       f"{p95:.2f} s | {first_p50:.2f} s | {rps:.1f} |")
        out.append("")
    return "\n".join(out).strip()


def main() -> None:
    """Rewrite all table blocks in README.md."""
    runs = eval_runs()
    tables = {
        "headline": lambda: table_headline(runs),
        "categories": lambda: table_categories(runs),
        "experiments": lambda: table_experiments(runs),
        "model": table_model,
        "retrieval": table_retrieval,
        "load": table_load,
    }
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for name, fn in tables.items():
        pattern = re.compile(rf"(<!-- TABLE:{name} -->\n)(?:.*?\n)?(<!-- /TABLE:{name} -->)",
                             re.S)
        if pattern.search(text):
            text = pattern.sub(lambda m, t=fn(): m.group(1) + t + "\n" + m.group(2), text)
            print(f"updated table {name}")
    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
