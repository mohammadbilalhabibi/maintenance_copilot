"""Generate the evaluation reference set (eval/reference_set.json).

Ground truth for data questions is computed here with pandas directly from the raw CSVs,
independently of the agent's SQL tools, so a tool bug cannot hide in the answer key.
Risk questions use the trained model's output (the agent must report it faithfully).
Document questions name the expected document and key facts from the text.
Machines are picked with a fixed seed. Do not edit the output by hand.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from copilot.config import PROJECT_ROOT, get_settings
from copilot.ml.predict import RiskModel

SIGNAL_UNITS = {"volt": "voltage", "rotate": "rotation speed", "pressure": "pressure",
                "vibration": "vibration"}


def load(raw: Path, fleet: int, now: datetime) -> dict[str, pd.DataFrame]:
    """Raw frames for machines 1..fleet, up to `now` (the agent cannot see later data)."""
    def read(name: str, **kw) -> pd.DataFrame:  # noqa: ANN003
        df = pd.read_csv(raw / name, **kw)
        df = df[df["machineID"] <= fleet]
        if "datetime" in df:
            df = df[df["datetime"] <= now]
        return df

    tel = []
    for chunk in pd.read_csv(raw / "telemetry.csv", parse_dates=["datetime"], chunksize=200_000):
        tel.append(chunk[(chunk["machineID"] <= fleet) & (chunk["datetime"] <= now)])
        if chunk["machineID"].min() > fleet:
            break
    return {
        "tel": pd.concat(tel),
        "err": read("errors.csv", parse_dates=["datetime"]),
        "maint": read("maint.csv", parse_dates=["datetime"]),
        "fail": read("failures.csv", parse_dates=["datetime"]),
        "machines": read("machines.csv"),
    }


def q(qid: str, category: str, question: str, *, tools: list[list[str]] | None = None,
      checks: list[dict] | None = None, **extra) -> dict:  # noqa: ANN003
    """One reference item. `tools`: any-of groups that must all be satisfied."""
    return {"id": qid, "category": category, "question": question,
            "required_tools": tools or [], "checks": checks or [], **extra}


def build(raw: Path, fleet: int, now: datetime, model: RiskModel, seed: int) -> list[dict]:
    """Build all reference items."""
    d = load(raw, fleet, now)
    rng = np.random.default_rng(seed)
    machines = d["machines"].set_index("machineID")
    items: list[dict] = []

    def pick(n: int, pool: list[int] | None = None) -> list[int]:
        pool = pool if pool is not None else list(machines.index)
        return [int(x) for x in rng.choice(pool, size=n, replace=False)]

    # --- machine facts -----------------------------------------------------------
    for i, m in enumerate(pick(5)):
        row = machines.loc[m]
        items.append(q(f"fact-model-{i}", "fact", f"What model is machine {m} and how old is it?",
                       tools=[["get_machine_status"]],
                       checks=[{"type": "contains", "value": row["model"]},
                               {"type": "number", "value": int(row["age"]), "tol": 0}]))

    week = now - timedelta(days=7)
    err7 = d["err"][d["err"]["datetime"] > week]
    with_errors = sorted(err7["machineID"].unique().tolist())
    for i, m in enumerate(pick(5, with_errors)):
        n = int((err7["machineID"] == m).sum())
        items.append(q(f"fact-errors7d-{i}", "fact",
                       f"How many errors did machine {m} log in the last 7 days?",
                       tools=[["get_error_history", "get_machine_status"]],
                       checks=[{"type": "number", "value": n, "tol": 0}]))

    month = now - timedelta(days=30)
    err30 = d["err"][d["err"]["datetime"] > month]
    for i, (m, code) in enumerate(
        err30.groupby(["machineID", "errorID"]).size().sample(4, random_state=seed).index
    ):
        n = int(((err30["machineID"] == m) & (err30["errorID"] == code)).sum())
        items.append(q(f"fact-errcode30d-{i}", "fact",
                       f"How many {code} events did machine {int(m)} have in the last 30 days?",
                       tools=[["get_error_history"]],
                       checks=[{"type": "number", "value": n, "tol": 0}]))

    for i, m in enumerate(pick(5)):
        comp = f"comp{rng.integers(1, 5)}"
        last = d["maint"][(d["maint"]["machineID"] == m) & (d["maint"]["comp"] == comp)]
        date = last["datetime"].max().strftime("%Y-%m-%d")
        items.append(q(f"fact-lastrepl-{i}", "fact",
                       f"When was {comp} of machine {m} last replaced?",
                       tools=[["get_maintenance_history", "get_machine_status"]],
                       checks=[{"type": "contains", "value": date}]))

    q90 = now - timedelta(days=90)
    fail90 = d["fail"][d["fail"]["datetime"] > q90]
    for i, m in enumerate(pick(4)):
        n = int((fail90["machineID"] == m).sum())
        items.append(q(f"fact-fail90-{i}", "fact",
                       f"How many component failures did machine {m} have in the last 90 days?",
                       tools=[["get_machine_status", "get_maintenance_history"]],
                       checks=[{"type": "number", "value": n, "tol": 0}]))

    tel = d["tel"]
    for i, m in enumerate(pick(6)):
        signal = ["volt", "rotate", "pressure", "vibration"][i % 4]
        hours = [24, 48, 12][i % 3]
        stat = ["average", "maximum"][i % 2]
        w = tel[(tel["machineID"] == m) & (tel["datetime"] > now - timedelta(hours=hours))]
        value = float(w[signal].mean() if stat == "average" else w[signal].max())
        items.append(q(f"fact-telemetry-{i}", "fact",
                       f"What was the {stat} {SIGNAL_UNITS[signal]} of machine {m} over the "
                       f"last {hours} hours?",
                       tools=[["query_telemetry"]],
                       checks=[{"type": "number", "value": round(value, 2), "tol": 0.1}]))

    # --- fleet questions ---------------------------------------------------------
    for i, days in enumerate((30, 60, 90)):
        f = d["fail"][d["fail"]["datetime"] > now - timedelta(days=days)]
        counts = f.groupby("machineID").size().sort_values(ascending=False)
        top = counts.iloc[0]
        leaders = sorted(int(x) for x in counts[counts == top].index)
        items.append(q(f"fleet-mostfail-{i}", "fleet",
                       f"Which machine had the most component failures in the last {days} days, "
                       "and how many?",
                       tools=[["fleet_overview"]],
                       checks=[{"type": "any_number", "values": leaders},
                               {"type": "number", "value": int(top), "tol": 0}]))
    for i, code in enumerate(("error1", "error4")):
        e = d["err"][(d["err"]["datetime"] > month) & (d["err"]["errorID"] == code)]
        counts = e.groupby("machineID").size().sort_values(ascending=False)
        leaders = sorted(int(x) for x in counts[counts == counts.iloc[0]].index)
        items.append(q(f"fleet-mosterr-{i}", "fleet",
                       f"Which machine logged the most {code} events in the last 30 days?",
                       tools=[["fleet_overview"]],
                       checks=[{"type": "any_number", "values": leaders}]))

    # --- risk (model output must be reported faithfully) -------------------------
    frames = {
        "telemetry": tel.rename(columns={"datetime": "ts", "machineID": "machine_id"})
        .loc[lambda x: x["ts"] > now - timedelta(hours=25)],
        "errors": d["err"].rename(columns={"datetime": "ts", "machineID": "machine_id",
                                           "errorID": "error_id"}),
        "maint": d["maint"].rename(columns={"datetime": "ts", "machineID": "machine_id",
                                            "comp": "component"}),
        "machines": d["machines"].rename(columns={"machineID": "machine_id"}),
    }
    risks = {r.machine_id: r for r in model.predict_many(frames, now)}
    order24 = sorted(risks, key=lambda m: -risks[m].horizons["24h"].failure_probability)
    order7 = sorted(risks, key=lambda m: -risks[m].horizons["7d"].failure_probability)
    risky = order7[:15]
    calm = order7[-40:]
    for i, m in enumerate(pick(3, risky) + pick(3, calm)):
        h = "7d" if i % 2 == 0 else "24h"
        r = risks[m].horizons[h]
        label = "7 days" if h == "7d" else "24 hours"
        items.append(q(f"risk-prob-{i}", "risk",
                       f"How likely is machine {m} to have a component failure in the next "
                       f"{label}?",
                       tools=[["predict_failure_risk"]],
                       checks=[{"type": "probability", "value": r.failure_probability}],
                       expected_alert=r.alert))
    for i, m in enumerate(pick(2, risky) + pick(2, calm)):
        r = risks[m].horizons["7d"]
        items.append(q(f"risk-alert-{i}", "risk",
                       f"Is machine {m} on a 7-day failure-risk alert right now?",
                       tools=[["predict_failure_risk"]],
                       checks=[{"type": "yes_no", "value": r.alert}]))
    items.append(q("fleet-risk-0", "fleet",
                   "Which 3 machines have the highest risk of failing within the next 24 hours?",
                   tools=[["fleet_overview"]],
                   checks=[{"type": "all_numbers", "values": [int(m) for m in order24[:3]]}]))
    items.append(q("fleet-risk-1", "fleet",
                   "Which machine is most likely to fail within the next 7 days?",
                   tools=[["fleet_overview"]],
                   checks=[{"type": "any_number", "values": [int(order7[0])]}]))

    # --- documents ---------------------------------------------------------------
    hb, lo, sg = "fleet_handbook", "osha3120_lockout_tagout", "osha3170_safeguarding_equipment"
    docs = [
        ("What does error4 mean and which component does it relate to?", hb,
         [["comp3"], ["pressure"]]),
        ("Which component is error5 related to?", hb, [["comp4"]]),
        ("What does error1 indicate?", hb, [["voltage"], ["comp1"]]),
        ("Which error codes point to the spindle and rotor assembly?", hb,
         [["error2"], ["error3"]]),
        ("What is the normal vibration range for these machines?", hb, [["32"], ["49"]]),
        ("Above what pressure should I raise an alarm?", hb, [["131"]]),
        ("What ticket priority should I use when a machine is on a 24-hour failure alert?", hb,
         [["high"]]),
        ("What counts as a critical ticket?", hb, [["safety", "stopped", "smoke", "fire"]]),
        ("How often is each machine visited for scheduled maintenance?", hb, [["15"]]),
        ("What should I do if a machine is on a 7-day alert but not on a 24-hour alert?", hb,
         [["next scheduled", "medium"]]),
        ("What must workers do before they remove their lockout device and re-energize a "
         "machine?", lo, [["tools", "remove", "workers", "employees", "clear", "safe"]]),
        ("When can tagout devices be used instead of lockout devices?", lo,
         [["lock", "cannot", "not capable", "equal"]]),
        ("What must an energy-control procedure include?", lo,
         [["steps", "procedure", "shut", "isolat", "verif"]]),
        ("What are point-of-operation hazards on machines?", sg,
         [["point of operation", "point-of-operation"]]),
    ]
    for i, (question, doc, groups) in enumerate(docs):
        items.append(q(f"doc-{i}", "document", question, tools=[["search_manuals"]],
                       checks=[{"type": "keywords", "groups": groups}], expected_doc=doc))

    # --- multi-tool --------------------------------------------------------------
    multi_pool = with_errors
    for i, m in enumerate(pick(4, multi_pool)):
        r = risks.get(m)
        items.append(q(f"multi-{i}", "multi",
                       f"Machine {m} has been throwing errors. What's wrong, how risky is it, "
                       "and what does the handbook say I should check?",
                       tools=[["get_machine_status", "get_error_history"],
                              ["predict_failure_risk"], ["search_manuals"]],
                       checks=[{"type": "probability_any",
                                "values": [h.failure_probability for h in r.horizons.values()]}],
                       expected_doc=hb))
    for i, m in enumerate(pick(2, risky)):
        items.append(q(f"multi-risk-doc-{i}", "multi",
                       f"Machine {m}: what is its 7-day failure risk and what ticket priority does "
                       "the handbook recommend for that situation?",
                       tools=[["predict_failure_risk"], ["search_manuals"]],
                       checks=[{"type": "probability",
                                "value": risks[m].horizons["7d"].failure_probability}],
                       expected_doc=hb))

    # --- unanswerable / out of scope ---------------------------------------------
    unans = [
        "How much does a replacement comp3 hydraulic pump cost?",
        "Who is the supplier of the bearings used in comp4?",
        "What is the name of the shift supervisor on duty today?",
        f"What is the status of machine {fleet + 150}?",
        "What will the vibration of machine 3 be next month?",
        "What's the weather forecast for tomorrow?",
        "What warranty period applies to the drive motors?",
        "Write me a poem about football.",
    ]
    for i, question in enumerate(unans):
        items.append(q(f"unanswerable-{i}", "unanswerable", question,
                       checks=[{"type": "refusal"}]))

    # --- ticket requests ---------------------------------------------------------
    for i, m in enumerate(pick(5)):
        comp = f"comp{rng.integers(1, 5)}"
        prio = ["high", "medium", "low", "critical", "high"][i]
        items.append(q(f"ticket-{i}", "ticket",
                       f"Please open a {prio} priority ticket for machine {m}, component {comp}: "
                       "inspect it at the next visit because of repeated warnings.",
                       tools=[["create_ticket"]],
                       expected_ticket={"machine_id": m, "component": comp, "priority": prio}))
    return items


def main() -> None:
    """CLI entry point."""
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "reference_set.json")
    args = parser.parse_args()
    items = build(args.raw, s.fleet_size, s.sim_now, RiskModel.load(s.model_path), args.seed)
    payload = {"sim_now": s.sim_now.isoformat(), "fleet_size": s.fleet_size, "seed": args.seed,
               "n": len(items), "items": items}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    cats = pd.Series([i["category"] for i in items]).value_counts().to_dict()
    print(f"wrote {len(items)} items to {args.out}: {cats}")


if __name__ == "__main__":
    main()
