"""Read-only queries behind the agent's data tools (work on SQLite and Postgres)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from copilot.db.models import ErrorEvent, Failure, Machine, Maintenance, Telemetry, Ticket

SIGNALS = ("volt", "rotate", "pressure", "vibration")


async def _frame(session: AsyncSession, stmt: Select, columns: list[str]) -> pd.DataFrame:
    rows = (await session.execute(stmt)).all()
    return pd.DataFrame(rows, columns=columns)


async def get_machine(session: AsyncSession, machine_id: int) -> Machine | None:
    """Machine row or None."""
    return await session.get(Machine, machine_id)


async def machine_ids(session: AsyncSession) -> list[int]:
    """All machine ids in the fleet."""
    return list((await session.execute(select(Machine.machine_id))).scalars())


async def latest_telemetry(session: AsyncSession, machine_id: int, now: datetime) -> dict | None:
    """Most recent telemetry reading at or before now."""
    stmt = (
        select(Telemetry)
        .where(Telemetry.machine_id == machine_id, Telemetry.ts <= now)
        .order_by(Telemetry.ts.desc())
        .limit(1)
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None
    return {"ts": row.ts, **{s: round(getattr(row, s), 2) for s in SIGNALS}}


async def telemetry_stats(
    session: AsyncSession, machine_id: int, start: datetime, end: datetime, signals: list[str]
) -> dict[str, dict[str, float]]:
    """mean/min/max per signal over (start, end] plus number of readings."""
    out: dict[str, dict[str, float]] = {}
    for s in signals:
        col = getattr(Telemetry, s)
        stmt = select(func.avg(col), func.min(col), func.max(col), func.count()).where(
            Telemetry.machine_id == machine_id, Telemetry.ts > start, Telemetry.ts <= end
        )
        avg, mn, mx, n = (await session.execute(stmt)).one()
        if n:
            out[s] = {"mean": round(avg, 2), "min": round(mn, 2), "max": round(mx, 2), "n": n}
    return out


async def error_counts(
    session: AsyncSession, machine_id: int, start: datetime, end: datetime
) -> dict[str, int]:
    """Number of each error type in (start, end]."""
    stmt = (
        select(ErrorEvent.error_id, func.count())
        .where(ErrorEvent.machine_id == machine_id, ErrorEvent.ts > start, ErrorEvent.ts <= end)
        .group_by(ErrorEvent.error_id)
        .order_by(ErrorEvent.error_id)
    )
    return {e: n for e, n in (await session.execute(stmt)).all()}


async def recent_errors(
    session: AsyncSession, machine_id: int, start: datetime, end: datetime,
    error_id: str | None = None, limit: int = 15,
) -> list[dict]:
    """Most recent error events, newest first."""
    stmt = select(ErrorEvent.ts, ErrorEvent.error_id).where(
        ErrorEvent.machine_id == machine_id, ErrorEvent.ts > start, ErrorEvent.ts <= end
    )
    if error_id:
        stmt = stmt.where(ErrorEvent.error_id == error_id)
    stmt = stmt.order_by(ErrorEvent.ts.desc()).limit(limit)
    return [{"ts": ts, "error": e} for ts, e in (await session.execute(stmt)).all()]


async def maintenance_events(
    session: AsyncSession, machine_id: int, start: datetime, end: datetime,
    component: str | None = None, limit: int = 20,
) -> list[dict]:
    """Component replacements in (start, end], newest first, marked failure or scheduled."""
    stmt = select(Maintenance.ts, Maintenance.component).where(
        Maintenance.machine_id == machine_id, Maintenance.ts > start, Maintenance.ts <= end
    )
    if component:
        stmt = stmt.where(Maintenance.component == component)
    stmt = stmt.order_by(Maintenance.ts.desc(), Maintenance.component).limit(limit)
    maint = (await session.execute(stmt)).all()
    fstmt = select(Failure.ts, Failure.component).where(
        Failure.machine_id == machine_id, Failure.ts > start, Failure.ts <= end
    )
    failures = {(ts, c) for ts, c in (await session.execute(fstmt)).all()}
    return [
        {"ts": ts, "component": c, "reason": "failure" if (ts, c) in failures else "scheduled"}
        for ts, c in maint
    ]


async def last_replacements(
    session: AsyncSession, machine_id: int, now: datetime
) -> dict[str, datetime | None]:
    """Latest replacement time per component at or before now."""
    stmt = (
        select(Maintenance.component, func.max(Maintenance.ts))
        .where(Maintenance.machine_id == machine_id, Maintenance.ts <= now)
        .group_by(Maintenance.component)
    )
    found = dict((await session.execute(stmt)).all())
    return {c: found.get(c) for c in ("comp1", "comp2", "comp3", "comp4")}


async def failures_between(
    session: AsyncSession, machine_id: int, start: datetime, end: datetime
) -> list[dict]:
    """Component failures in (start, end], newest first."""
    stmt = (
        select(Failure.ts, Failure.component)
        .where(Failure.machine_id == machine_id, Failure.ts > start, Failure.ts <= end)
        .order_by(Failure.ts.desc())
    )
    return [{"ts": ts, "component": c} for ts, c in (await session.execute(stmt)).all()]


async def open_tickets(session: AsyncSession, machine_id: int) -> list[dict]:
    """Open maintenance tickets for a machine."""
    stmt = (
        select(Ticket)
        .where(Ticket.machine_id == machine_id, Ticket.status == "open")
        .order_by(Ticket.id.desc())
    )
    return [
        {"ticket_id": t.id, "component": t.component, "priority": t.priority,
         "summary": t.summary, "created_at": t.created_at}
        for t in (await session.execute(stmt)).scalars()
    ]


async def fleet_ranking(
    session: AsyncSession, metric: str, start: datetime, end: datetime,
    top_n: int, filter_value: str | None = None,
) -> list[dict]:
    """Machines ranked by number of failures or errors in (start, end]."""
    table = Failure if metric == "failures" else ErrorEvent
    stmt = select(table.machine_id, func.count().label("n")).where(
        table.ts > start, table.ts <= end
    )
    if filter_value:
        col = Failure.component if metric == "failures" else ErrorEvent.error_id
        stmt = stmt.where(col == filter_value)
    stmt = (
        stmt.group_by(table.machine_id)
        .order_by(func.count().desc(), table.machine_id)
        .limit(top_n)
    )
    return [{"machine_id": m, "count": n} for m, n in (await session.execute(stmt)).all()]


async def history_frames(
    session: AsyncSession, machine_ids_: list[int] | None, now: datetime,
    history: timedelta = timedelta(hours=25),
) -> dict[str, pd.DataFrame]:
    """Raw frames the failure-risk features need, for one, several or all machines."""
    start = now - history

    def only(stmt: Select, col) -> Select:  # noqa: ANN001
        return stmt.where(col.in_(machine_ids_)) if machine_ids_ else stmt

    tel = await _frame(
        session,
        only(
            select(Telemetry.machine_id, Telemetry.ts, *[getattr(Telemetry, s) for s in SIGNALS])
            .where(Telemetry.ts > start, Telemetry.ts <= now),
            Telemetry.machine_id,
        ),
        ["machine_id", "ts", *SIGNALS],
    )
    err = await _frame(
        session,
        only(
            select(ErrorEvent.machine_id, ErrorEvent.ts, ErrorEvent.error_id)
            .where(ErrorEvent.ts > start, ErrorEvent.ts <= now),
            ErrorEvent.machine_id,
        ),
        ["machine_id", "ts", "error_id"],
    )
    maint = await _frame(
        session,
        only(
            select(Maintenance.machine_id, Maintenance.ts, Maintenance.component)
            .where(Maintenance.ts <= now),
            Maintenance.machine_id,
        ),
        ["machine_id", "ts", "component"],
    )
    machines = await _frame(
        session,
        only(select(Machine.machine_id, Machine.model, Machine.age), Machine.machine_id),
        ["machine_id", "model", "age"],
    )
    for df in (tel, err, maint):
        df["ts"] = pd.to_datetime(df["ts"])
    return {"telemetry": tel, "errors": err, "maint": maint, "machines": machines}
