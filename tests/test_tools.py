"""Agent tools against the synthetic SQLite fleet."""

from __future__ import annotations

import pytest
from sqlalchemy import select, text

from copilot.agent import tools as t
from copilot.agent.graph import run_tool
from copilot.db.models import Ticket


async def test_machine_status_hides_future(ctx: t.ToolContext) -> None:
    out = await t.get_machine_status(ctx, t.MachineArgs(machine_id=1))
    assert out["model"] == "model3" and out["age_years"] == 18
    assert out["errors_last_24h"] == {"error4": 2}  # error5 at NOW+1h is in the future
    assert out["latest_telemetry"]["ts"] == "2015-03-10 06:00"
    assert out["failures_last_90d"] == [{"ts": "2015-03-06 06:00", "component": "comp3"}]
    assert out["last_replacement"]["comp3"] == "2015-03-06 06:00"
    assert out["last_replacement"]["comp2"] is None


async def test_unknown_machine_is_a_tool_error(ctx: t.ToolContext) -> None:
    result, error = await run_tool(ctx, "get_machine_status", {"machine_id": 99})
    assert result is None and "does not exist" in error and "1-3" in error


async def test_invalid_arguments_are_reported(ctx: t.ToolContext) -> None:
    result, error = await run_tool(ctx, "query_telemetry", {"machine_id": 1, "hours_back": 9999})
    assert result is None and "Invalid arguments" in error


async def test_telemetry_window(ctx: t.ToolContext) -> None:
    out = await t.query_telemetry(ctx, t.TelemetryArgs(machine_id=2, signals=["volt"],
                                                         hours_back=24))
    assert set(out["signals"]) == {"volt"}
    assert out["signals"]["volt"]["n"] == 24


async def test_error_history_filter(ctx: t.ToolContext) -> None:
    out = await t.get_error_history(ctx, t.ErrorArgs(machine_id=1, days_back=7,
                                                     error_id="error4"))
    assert out["counts"] == {"error4": 2} and out["total"] == 2
    assert all(e["error"] == "error4" for e in out["most_recent"])


async def test_maintenance_marks_failures(ctx: t.ToolContext) -> None:
    out = await t.get_maintenance_history(ctx, t.MaintenanceArgs(machine_id=1, days_back=30))
    reasons = {e["component"]: e["reason"] for e in out["replacements"]}
    assert reasons == {"comp3": "failure", "comp1": "scheduled"}
    assert out["failures_in_window"] == 1


async def test_fleet_ranking_errors(ctx: t.ToolContext) -> None:
    out = await t.fleet_overview(ctx, t.FleetArgs(metric="errors", days_back=7, top_n=2))
    assert out["ranking"][0] == {"machine_id": 1, "count": 3}


async def test_predict_failure_risk(ctx: t.ToolContext) -> None:
    out = await t.predict_failure_risk(ctx, t.MachineArgs(machine_id=1))
    assert set(out["horizons"]) == {"24h", "7d"}
    p = out["horizons"]["24h"]["failure_probability"]
    assert 0.0 <= p <= 1.0
    assert "2x error4 in last 24 h" in out["notable_inputs"]


async def test_fleet_risk_scores_every_machine(ctx: t.ToolContext) -> None:
    out = await t.fleet_overview(ctx, t.FleetArgs(metric="risk", top_n=3, horizon="7d"))
    assert out["machines_scored"] == 3 and len(out["ranking"]) == 3


async def test_search_manuals(ctx: t.ToolContext) -> None:
    out = await t.search_manuals(ctx, t.SearchArgs(query="what does error4 mean"))
    assert out["passages"][0]["doc"] == "Test Handbook"


async def test_create_ticket_writes_row(ctx: t.ToolContext) -> None:
    args = t.TicketArgs(machine_id=1, component="comp3", priority="high", summary="error4 x2")
    out = await t.create_ticket(ctx, args, user_id=None, conversation_id="conv-1")
    async with ctx.db.session() as s:
        ticket = (await s.execute(select(Ticket))).scalar_one()
    assert ticket.id == out["ticket_id"] and ticket.conversation_id == "conv-1"


async def test_read_only_connection_rejects_writes(ctx: t.ToolContext) -> None:
    async with ctx.db.read() as s:
        with pytest.raises(Exception, match="readonly|read-only|query_only"):
            await s.execute(text("DELETE FROM machines"))
