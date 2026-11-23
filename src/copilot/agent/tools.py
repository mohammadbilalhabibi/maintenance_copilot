"""Agent tools: Pydantic argument schemas and async implementations.

Data tools use the read-only database connection. `create_ticket` is the only write
and is executed only after the user confirms it (see graph.py).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from copilot.agent import data_access as q
from copilot.db.models import Ticket
from copilot.db.session import Database
from copilot.ml.predict import RiskModel
from copilot.rag.retriever import Retriever

Signal = Literal["volt", "rotate", "pressure", "vibration"]
Component = Literal["comp1", "comp2", "comp3", "comp4"]
ErrorCode = Literal["error1", "error2", "error3", "error4", "error5"]


_H24 = timedelta(hours=24)
_D7 = timedelta(days=7)


class ToolError(Exception):
    """Expected, user-facing tool failure (bad machine id, missing data...)."""


# ------------------------------------------------------------------ schemas
class MachineArgs(BaseModel):
    machine_id: int = Field(..., ge=1, description="Machine number, e.g. 17")


class TelemetryArgs(MachineArgs):
    signals: list[Signal] = Field(
        default_factory=lambda: ["volt", "rotate", "pressure", "vibration"],
        description="Which sensor signals to summarise",
    )
    hours_back: int = Field(24, ge=1, le=720, description="Window length in hours (max 720)")


class ErrorArgs(MachineArgs):
    days_back: int = Field(7, ge=1, le=365, description="Window length in days")
    error_id: ErrorCode | None = Field(None, description="Only this error code (optional)")


class MaintenanceArgs(MachineArgs):
    days_back: int = Field(90, ge=1, le=730, description="Window length in days")
    component: Component | None = Field(None, description="Only this component (optional)")


class FleetArgs(BaseModel):
    metric: Literal["failures", "errors", "risk"] = Field(
        ..., description="Rank machines by number of failures, number of errors, or current "
        "predicted 24 h failure risk"
    )
    days_back: int = Field(30, ge=1, le=365, description="Window for failures/errors")
    top_n: int = Field(5, ge=1, le=20)
    horizon: Literal["24h", "7d"] = Field("24h", description="Risk horizon (metric=risk only)")
    filter_value: str | None = Field(
        None, description="Optional component (comp1-comp4) for failures or error code "
        "(error1-error5) for errors"
    )


class SearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, description="What to look up")


class TicketArgs(MachineArgs):
    component: Component | None = Field(None, description="Affected component, if known")
    priority: Literal["low", "medium", "high", "critical"]
    summary: str = Field(..., min_length=5, max_length=500, description="What and why")


# ------------------------------------------------------------------ context
@dataclass
class ToolContext:
    """Long-lived resources the tools need (created once at startup)."""

    db: Database
    risk_model: RiskModel | None
    retriever: Retriever | None
    now: datetime
    search_top_k: int = 4
    timeout: float = 15.0


def _iso(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, dict):
        return {k: _iso(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_iso(x) for x in v]
    return v


async def _require_machine(ctx: ToolContext, session, machine_id: int) -> Any:  # noqa: ANN001
    machine = await q.get_machine(session, machine_id)
    if machine is None:
        ids = await q.machine_ids(session)
        span = f"{min(ids)}-{max(ids)}" if ids else "none loaded"
        raise ToolError(f"Machine {machine_id} does not exist (fleet machines: {span}).")
    return machine


# ------------------------------------------------------------ implementations
async def get_machine_status(ctx: ToolContext, args: MachineArgs) -> dict:
    """Overview: model, age, latest readings, recent errors, last failures, open tickets."""
    now = ctx.now
    async with ctx.db.read() as s:
        m = await _require_machine(ctx, s, args.machine_id)
        return _iso({
            "machine_id": m.machine_id,
            "model": m.model,
            "age_years": m.age,
            "as_of": now,
            "latest_telemetry": await q.latest_telemetry(s, m.machine_id, now),
            "errors_last_24h": await q.error_counts(s, m.machine_id, now - _H24, now),
            "errors_last_7d": await q.error_counts(s, m.machine_id, now - _D7, now),
            "failures_last_90d": await q.failures_between(
                s, m.machine_id, now - timedelta(days=90), now
            ),
            "last_replacement": await q.last_replacements(s, m.machine_id, now),
            "open_tickets": await q.open_tickets(s, m.machine_id),
        })


async def query_telemetry(ctx: ToolContext, args: TelemetryArgs) -> dict:
    """Mean/min/max of sensor signals over the last N hours."""
    now = ctx.now
    async with ctx.db.read() as s:
        await _require_machine(ctx, s, args.machine_id)
        start = now - timedelta(hours=args.hours_back)
        stats = await q.telemetry_stats(s, args.machine_id, start, now, list(args.signals))
    if not stats:
        raise ToolError(f"No telemetry for machine {args.machine_id} in that window.")
    return _iso({"machine_id": args.machine_id, "from": start, "to": now, "signals": stats})


async def get_error_history(ctx: ToolContext, args: ErrorArgs) -> dict:
    """Error counts by code and the most recent error events."""
    now = ctx.now
    start = now - timedelta(days=args.days_back)
    async with ctx.db.read() as s:
        await _require_machine(ctx, s, args.machine_id)
        counts = await q.error_counts(s, args.machine_id, start, now)
        if args.error_id:
            counts = {args.error_id: counts.get(args.error_id, 0)}
        recent = await q.recent_errors(s, args.machine_id, start, now, args.error_id)
    return _iso({"machine_id": args.machine_id, "from": start, "to": now,
                 "counts": counts, "total": sum(counts.values()), "most_recent": recent})


async def get_maintenance_history(ctx: ToolContext, args: MaintenanceArgs) -> dict:
    """Component replacements (scheduled or after a failure) in the window."""
    now = ctx.now
    start = now - timedelta(days=args.days_back)
    async with ctx.db.read() as s:
        await _require_machine(ctx, s, args.machine_id)
        events = await q.maintenance_events(s, args.machine_id, start, now, args.component)
        last = await q.last_replacements(s, args.machine_id, now)
    return _iso({"machine_id": args.machine_id, "from": start, "to": now,
                 "replacements": events,
                 "failures_in_window": sum(e["reason"] == "failure" for e in events),
                 "last_replacement_per_component": last})


async def predict_failure_risk(ctx: ToolContext, args: MachineArgs) -> dict:
    """Failure probability within 24 h and 7 days, from the trained models."""
    if ctx.risk_model is None:
        raise ToolError("The failure-risk model is not loaded.")
    async with ctx.db.read() as s:
        await _require_machine(ctx, s, args.machine_id)
        frames = await q.history_frames(s, [args.machine_id], ctx.now)
    results = await asyncio.to_thread(ctx.risk_model.predict_many, frames, ctx.now)
    if not results:
        raise ToolError(f"Not enough recent telemetry for machine {args.machine_id}.")
    out = asdict(results[0])
    rates = ", ".join(
        f"{name}: top {h['alert_rate']:.0%} of machine-hours"
        for name, h in ctx.risk_model.horizons.items()
    )
    out["model_note"] = (
        "LightGBM models trained on Jan-Jul 2015 data. failure_probability = chance that any "
        f"component fails within the horizon. Alert thresholds flag ({rates})."
    )
    return _iso(out)


async def fleet_overview(ctx: ToolContext, args: FleetArgs) -> dict:
    """Rank machines by failures, errors or current predicted risk."""
    now = ctx.now
    start = now - timedelta(days=args.days_back)
    if args.metric == "risk":
        if ctx.risk_model is None:
            raise ToolError("The failure-risk model is not loaded.")
        async with ctx.db.read() as s:
            frames = await q.history_frames(s, None, now)
        results = await asyncio.to_thread(ctx.risk_model.predict_many, frames, now)
        h = args.horizon
        results.sort(key=lambda r: -r.horizons[h].failure_probability)
        ranking = [
            {"machine_id": r.machine_id,
             f"failure_probability_{h}": r.horizons[h].failure_probability,
             "alert": r.horizons[h].alert,
             "most_likely_component": r.horizons[h].most_likely_component}
            for r in results[: args.top_n]
        ]
        return _iso({"metric": "risk", "horizon": h, "as_of": now, "ranking": ranking,
                     "machines_on_alert": sum(r.horizons[h].alert for r in results),
                     "machines_scored": len(results)})
    async with ctx.db.read() as s:
        ranking = await q.fleet_ranking(s, args.metric, start, now, args.top_n, args.filter_value)
    return _iso({"metric": args.metric, "filter": args.filter_value, "from": start, "to": now,
                 "ranking": ranking})


async def search_manuals(ctx: ToolContext, args: SearchArgs) -> dict:
    """Search the maintenance handbook and safety guides."""
    if ctx.retriever is None:
        raise ToolError("Document search is not available.")
    hits = await asyncio.to_thread(ctx.retriever.search, args.query, ctx.search_top_k)
    if not hits:
        return {"query": args.query, "passages": [], "note": "No matching passages found."}
    return {
        "query": args.query,
        "passages": [
            {"doc": h.chunk.title, "section": h.chunk.section, "chunk_id": h.chunk.chunk_id,
             "text": h.chunk.text}
            for h in hits
        ],
        # Removed by the tools node before the LLM sees the result (UI provenance only).
        "_meta": {
            h.chunk.chunk_id: {"doc_id": h.chunk.doc_id, "source_url": h.chunk.source_url,
                               "page": h.chunk.page, "synthetic": h.chunk.synthetic}
            for h in hits
        },
    }


async def create_ticket(
    ctx: ToolContext,
    args: TicketArgs,
    user_id: int | None = None,
    conversation_id: str | None = None,
) -> dict:
    """Insert a maintenance ticket (only called after the user confirmed)."""
    async with ctx.db.read() as s:
        await _require_machine(ctx, s, args.machine_id)
    async with ctx.db.session() as s:
        ticket = Ticket(
            machine_id=args.machine_id, component=args.component, priority=args.priority,
            summary=args.summary, created_by=user_id, conversation_id=conversation_id,
        )
        s.add(ticket)
        await s.commit()
        return {"ticket_id": ticket.id, "status": "open", **args.model_dump()}


@dataclass(frozen=True)
class ToolSpec:
    """A tool's name, description, argument schema and implementation."""

    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[..., Awaitable[dict]]
    writes: bool = False


TOOLS: list[ToolSpec] = [
    ToolSpec("get_machine_status",
             "Overview of one machine: model, age, latest sensor readings, error counts "
             "(24 h, 7 days), failures in the last 90 days, last replacement per component, "
             "open tickets.", MachineArgs, get_machine_status),
    ToolSpec("query_telemetry",
             "Mean/min/max of sensor signals (volt, rotate, pressure, vibration) for one "
             "machine over the last N hours.", TelemetryArgs, query_telemetry),
    ToolSpec("get_error_history",
             "Error counts per code and the latest error events for one machine over the last "
             "N days.", ErrorArgs, get_error_history),
    ToolSpec("get_maintenance_history",
             "Component replacements (scheduled or after failure) for one machine over the last "
             "N days, and the last replacement per component.", MaintenanceArgs,
             get_maintenance_history),
    ToolSpec("predict_failure_risk",
             "Trained ML model: probability that a component of this machine fails within the "
             "next 24 hours and within 7 days, per-component probabilities, alert flags.",
             MachineArgs, predict_failure_risk),
    ToolSpec("fleet_overview",
             "Rank the machines of the fleet by number of failures, number of errors, or current "
             "predicted failure risk.", FleetArgs, fleet_overview),
    ToolSpec("search_manuals",
             "Search the maintenance handbook and safety guides (error code meanings, "
             "troubleshooting steps, procedures, safety rules, ticket priorities).",
             SearchArgs, search_manuals),
    ToolSpec("create_ticket",
             "Open a maintenance ticket. Only when the user asks for a ticket. The user is asked "
             "to confirm before it is created.", TicketArgs, create_ticket, writes=True),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
