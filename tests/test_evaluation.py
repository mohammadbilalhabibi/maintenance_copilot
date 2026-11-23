"""Answer grading used by the evaluation."""

from __future__ import annotations

from copilot.evaluation import (
    check_keywords,
    check_number,
    check_probability,
    grade,
    numbers_in,
    summarize,
    tools_ok,
)


def test_numbers_ignore_ids_codes_dates_and_citations() -> None:
    text = "Machine 17 logged 3 error4 events on comp3 since 2015-11-08 06:00 [S1, S2]."
    assert numbers_in(text) == [3.0]


def test_number_and_probability_checks() -> None:
    assert check_number("Average vibration was 40.62 mm/s [S1].", 40.6234, 0.1)
    assert check_number("It was about 40.6.", 40.62, 0.0)  # one-decimal rounding allowed
    assert not check_number("Machine 7 had 2 errors.", 7, 0)  # machine id is not the answer
    assert check_probability("The 7-day risk is 40.6% [S2].", 0.4057)
    assert check_probability("probability 0.406", 0.4057)
    assert not check_probability("risk is 12%", 0.4057)


def test_keywords_and_tools() -> None:
    assert check_keywords("error4 relates to comp3 (hydraulic pressure)", [["comp3"], ["pressure"]])
    assert not check_keywords("error4 relates to comp3", [["comp3"], ["pressure"]])
    assert tools_ok([["a", "b"], ["c"]], ["b", "c"])
    assert not tools_ok([["a"], ["c"]], ["a"])


async def test_grade_ticket_and_unrequested_write() -> None:
    item = {"category": "ticket", "required_tools": [["create_ticket"]], "checks": [],
            "expected_ticket": {"machine_id": 5, "component": "comp2", "priority": "high"}}
    result = {"type": "confirm", "tool_calls": [{"tool": "create_ticket"}],
              "tickets": [{"machine_id": 5, "component": "comp2", "priority": "high",
                           "summary": "x"}]}
    g = await grade(item, result, None)
    assert g["answer_ok"] and g["tools_ok"] and not g["unrequested_write"]

    fact = {"category": "fact", "required_tools": [["get_machine_status"]],
            "checks": [{"type": "contains", "value": "model3"}]}
    bad = {"type": "done", "answer": "model3 [S1] [S9]",
           "tool_calls": [{"tool": "get_machine_status"}, {"tool": "create_ticket"}],
           "sources": [{"id": "S1"}], "cited": ["S1"]}
    g = await grade(fact, bad, None)
    assert g["unrequested_write"] and not g["tools_ok"]
    assert g["answer_ok"] and g["invalid_citations"] == 1


async def test_refusal_uses_judge_and_summary() -> None:
    async def judge(q: str, a: str, c: str) -> dict:
        return {"pass": "not available" in a, "reason": "test"}

    item = {"category": "unanswerable", "question": "price?", "required_tools": [],
            "checks": [{"type": "refusal"}]}
    g = await grade(item, {"type": "done", "answer": "That is not available."}, judge)
    assert g["answer_ok"]
    rec = {"category": "unanswerable", "grade": g, "usage": {"input": 10, "output": 5},
           "cost_usd": 0.001, "latency_s": 1.0}
    s = summarize([rec, rec])
    assert s["overall"]["answer_correct"] == 1.0 and s["overall"]["cost_usd_total"] == 0.002
