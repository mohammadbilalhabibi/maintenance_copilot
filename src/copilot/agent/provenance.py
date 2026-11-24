"""Plain-language origin of each data source (where a number came from and how)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

DATASET = {
    "name": "Microsoft Predictive Maintenance Modelling Guide sample data "
            "(simulated fleet, CC BY 4.0)",
    "url": "https://github.com/microsoft/SQL-Server-R-Services-Samples/tree/master/"
           "PredictiveMaintenanceModelingGuide/Data",
}
MODEL_CAVEAT = (
    "This is a prediction, not a measured fact. On this simulated data the model is very "
    "sure about the next 24 hours, because the simulator logs a warning error shortly before "
    "almost every failure; 7-day predictions are less certain."
)
FMT = "%Y-%m-%d %H:%M"


def _parse(value: Any) -> datetime | None:
    try:
        return datetime.strptime(str(value), FMT)
    except ValueError:
        return None


def _window(start: Any, end: Any) -> dict | None:
    s, e = _parse(start), _parse(end)
    return {"from": s.strftime(FMT), "to": e.strftime(FMT)} if s and e else None


def describe(tool: str, args: dict, result: dict) -> dict:
    """Origin of one tool result: what it is, which tables, time window, machines."""
    m = args.get("machine_id")
    kind, tables, window, caveat = "database", [], None, None
    machines = [m] if m else []
    if tool == "get_machine_status":
        as_of = result.get("as_of")
        end = _parse(as_of)
        window = _window((end - timedelta(days=90)).strftime(FMT), as_of) if end else None
        summary = (f"Overview of machine {m} from the fleet database as of {as_of}: model and "
                   "age, the latest hourly sensor reading, errors in the last 24 hours and 7 "
                   "days, failures in the last 90 days, the last replacement of each component "
                   "and open tickets.")
        tables = ["machines", "telemetry", "errors", "failures", "maintenance", "tickets"]
    elif tool == "query_telemetry":
        window = _window(result.get("from"), result.get("to"))
        signals = ", ".join(result.get("signals", {}))
        summary = (f"Average, minimum and maximum of the hourly sensor readings ({signals}) of "
                   f"machine {m} from {result.get('from')} to {result.get('to')}.")
        tables = ["telemetry"]
    elif tool == "get_error_history":
        window = _window(result.get("from"), result.get("to"))
        summary = (f"Error log of machine {m} from {result.get('from')} to {result.get('to')}: "
                   "how often each error code occurred and the latest events.")
        tables = ["errors"]
    elif tool == "get_maintenance_history":
        window = _window(result.get("from"), result.get("to"))
        summary = (f"Maintenance log of machine {m} from {result.get('from')} to "
                   f"{result.get('to')}. A replacement counts as 'failure' when the failure "
                   "log has the same time and component, otherwise as 'scheduled'.")
        tables = ["maintenance", "failures"]
    elif tool == "predict_failure_risk":
        kind, caveat = "model", MODEL_CAVEAT
        at = result.get("at")
        end = _parse(at)
        window = _window((end - timedelta(hours=24)).strftime(FMT), at) if end else None
        summary = (f"Prediction of the trained LightGBM failure-risk model for machine {m} at "
                   f"{at}. Inputs: the last 24 hours of sensor readings and errors, the time "
                   "since each component was last replaced, the machine model and its age.")
        tables = ["telemetry", "errors", "maintenance", "machines"]
    elif tool == "fleet_overview":
        ranking = result.get("ranking", [])
        machines = [r["machine_id"] for r in ranking if "machine_id" in r]
        if result.get("metric") == "risk":
            kind, caveat = "model", MODEL_CAVEAT
            end = _parse(result.get("as_of"))
            window = (_window((end - timedelta(hours=24)).strftime(FMT), result.get("as_of"))
                      if end else None)
            summary = (f"The trained failure-risk model scored every machine of the fleet at "
                       f"{result.get('as_of')} ({result.get('horizon')} horizon); these are "
                       "the highest risks.")
            tables = ["telemetry", "errors", "maintenance", "machines"]
        else:
            metric = result.get("metric")
            window = _window(result.get("from"), result.get("to"))
            summary = (f"All machines ranked by number of {metric} from {result.get('from')} "
                       f"to {result.get('to')}, counted in the fleet database.")
            tables = ["failures" if metric == "failures" else "errors"]
    elif tool == "create_ticket":
        summary = f"Ticket written to the tickets table after you confirmed it (machine {m})."
        tables = ["tickets"]
    else:
        summary = f"Result of {tool}."
    return {"kind": kind, "summary": summary, "tables": tables, "window": window,
            "machine_ids": machines[:10], "dataset": DATASET, "caveat": caveat}
