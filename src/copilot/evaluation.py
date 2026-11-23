"""Grading of agent answers against the reference set (code checks + a small LLM judge)."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

# Tokens that contain digits but are not answers: citations, ids, codes, dates, machine refs.
_NOISE = re.compile(
    r"\[S\d+(?:\s*,\s*S\d+)*\]|\bS\d+\b|\bcomp\d\b|\berror\d\b|\bmodel\d\b"
    r"|\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|\bmachines?\s*(?:#|no\.?\s*)?\d+",
    re.I,
)
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
# For machine-id answers ("Machine 32 had the most failures") keep machine references.
_NOISE_IDS = re.compile(
    r"\[S\d+(?:\s*,\s*S\d+)*\]|\bS\d+\b|\bcomp\d\b|\berror\d\b|\bmodel\d\b"
    r"|\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
    re.I,
)
JudgeFn = Callable[[str, str, str], Awaitable[dict]]


def numbers_in(text: str, keep_machine_ids: bool = False) -> list[float]:
    """Numbers mentioned in an answer, ignoring codes, dates, citations (and machine ids)."""
    clean = (_NOISE_IDS if keep_machine_ids else _NOISE).sub(" ", text)
    out = []
    for tok in _NUMBER.findall(clean):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def check_number(answer: str, value: float, tol: float) -> bool:
    """Any number in the answer within `tol` of value (rounding to 1 decimal allowed)."""
    tol = max(tol, 0.051 if value != int(value) else 0)
    return any(abs(n - value) <= tol for n in numbers_in(answer))


def check_probability(answer: str, value: float) -> bool:
    """Probability reported as percent (40.6%) or fraction (0.406), rounded to <= 1 decimal %."""
    nums = numbers_in(answer)
    return any(abs(n - value * 100) <= 0.6 or abs(n - value) <= 0.006 for n in nums)


def check_keywords(answer: str, groups: list[list[str]]) -> bool:
    """Every group has at least one keyword in the answer (case-insensitive)."""
    low = answer.lower()
    return all(any(k.lower() in low for k in g) for g in groups)


def tools_ok(required: list[list[str]], called: list[str]) -> bool:
    """Each any-of group is satisfied by at least one called tool."""
    return all(any(t in called for t in group) for group in required)


async def grade(item: dict, result: dict, judge: JudgeFn | None) -> dict:
    """Grade one result (the runner's 'done' or 'confirm' event) against its reference item."""
    called = [c["tool"] for c in result.get("tool_calls", [])]
    answer = result.get("answer", "") or ""
    is_ticket = item["category"] == "ticket"
    g: dict[str, Any] = {
        "tools_ok": tools_ok(item["required_tools"], called),
        "unrequested_write": "create_ticket" in called and not is_ticket,
    }
    if g["unrequested_write"]:
        g["tools_ok"] = False
    if is_ticket:
        expected = item["expected_ticket"]
        tickets = result.get("tickets") or []
        g["answer_ok"] = result.get("type") == "confirm" and any(
            all(t.get(k) == v for k, v in expected.items()) for t in tickets
        )
        return g

    checks: list[bool] = []
    for c in item["checks"]:
        kind = c["type"]
        if kind == "number":
            checks.append(check_number(answer, c["value"], c.get("tol", 0)))
        elif kind == "contains":
            checks.append(c["value"].lower() in answer.lower())
        elif kind == "probability":
            checks.append(check_probability(answer, c["value"]))
        elif kind == "probability_any":
            checks.append(any(check_probability(answer, v) for v in c["values"]))
        elif kind == "keywords":
            checks.append(check_keywords(answer, c["groups"]))
        elif kind in ("any_number", "all_numbers"):  # used for machine ids
            ids = set(numbers_in(answer, keep_machine_ids=True))
            hits = [float(v) in ids for v in c["values"]]
            checks.append(any(hits) if kind == "any_number" else all(hits))
        elif kind in ("refusal", "yes_no"):
            if judge is None:
                checks.append(False)
                continue
            criterion = (
                "The assistant does not answer with facts. It declines: it says the "
                "information is not available, or that the machine does not exist, or that it "
                "cannot predict this, or that the request is outside what it helps with. It "
                "must not state invented specifics (numbers, names, prices, dates). A short "
                "refusal is a pass."
                if kind == "refusal"
                else f"The answer clearly says {'YES' if c['value'] else 'NO'}."
            )
            verdict = await judge(item["question"], answer, criterion)
            g.setdefault("judge", []).append(verdict)
            checks.append(bool(verdict.get("pass")))
    g["answer_ok"] = all(checks) if checks else False

    sources = result.get("sources", [])
    cited = set(result.get("cited", []))
    mentioned = set(re.findall(r"\bS(\d+)\b", " ".join(re.findall(r"\[([^\]]+)\]", answer))))
    g["invalid_citations"] = len({f"S{n}" for n in mentioned} - {s["id"] for s in sources})
    if item["category"] != "unanswerable":
        g["cited"] = bool(cited)
        if item.get("expected_doc"):
            docs = {s.get("chunk_id", "").split("#")[0] for s in sources if s["id"] in cited}
            g["doc_cited"] = item["expected_doc"] in docs
    return g


def summarize(records: list[dict]) -> dict:
    """Aggregate metrics over graded records (overall and per category)."""

    def agg(rs: list[dict]) -> dict:
        def rate(key: str) -> float | None:
            vals = [r["grade"][key] for r in rs if key in r["grade"]]
            return round(float(np.mean(vals)), 4) if vals else None

        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        cost = [r["cost_usd"] for r in rs]
        tokens = [r["usage"].get("input", 0) + r["usage"].get("output", 0) for r in rs]
        return {
            "n": len(rs),
            "answer_correct": rate("answer_ok"),
            "tool_selection": rate("tools_ok"),
            "cited_any": rate("cited"),
            "doc_cited": rate("doc_cited"),
            "unrequested_writes": int(sum(r["grade"].get("unrequested_write", 0) for r in rs)),
            "invalid_citations": int(sum(r["grade"].get("invalid_citations", 0) for r in rs)),
            "errors": int(sum(1 for r in rs if r.get("error"))),
            "latency_p50_s": round(float(np.percentile(lat, 50)), 2) if lat else None,
            "latency_p95_s": round(float(np.percentile(lat, 95)), 2) if lat else None,
            "tokens_mean": round(float(np.mean(tokens)), 1) if tokens else None,
            "cached_input_share": round(
                sum(r["usage"].get("cached_input", 0) for r in rs)
                / max(sum(r["usage"].get("input", 0) for r in rs), 1), 4),
            "llm_calls_mean": round(float(np.mean([r["usage"].get("llm_calls", 0) for r in rs])),
                                    2) if rs else None,
            "cost_usd_mean": round(float(np.mean(cost)), 6) if cost else None,
            "cost_usd_total": round(float(np.sum(cost)), 4) if cost else None,
        }

    cats = sorted({r["category"] for r in records})
    return {"overall": agg(records),
            "by_category": {c: agg([r for r in records if r["category"] == c]) for c in cats}}


JUDGE_PROMPT = """You check whether an assistant's answer shows an EXPECTED behaviour.
Do not judge helpfulness or style. Only decide if the expected behaviour is present.

Question asked: {question}
Answer given: {answer}
Expected behaviour: {criterion}

Reply with JSON only: {{"pass": true or false, "reason": "<one short sentence>"}}"""


def make_judge(llm: Any) -> JudgeFn:
    """LLM judge (temperature 0) returning {"pass": bool, "reason": str}."""

    async def judge(question: str, answer: str, criterion: str) -> dict:
        msg = await llm.ainvoke(JUDGE_PROMPT.format(question=question, answer=answer,
                                                    criterion=criterion))
        text = msg.content.strip().strip("`").removeprefix("json").strip()
        try:
            out = json.loads(text)
            return {"pass": bool(out.get("pass")), "reason": str(out.get("reason", ""))[:200]}
        except json.JSONDecodeError:
            return {"pass": False, "reason": f"unparseable judge output: {text[:80]}"}

    return judge
