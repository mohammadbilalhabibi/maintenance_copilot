"""Load the raw sample CSVs (machines 1..fleet_size) into the database."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from copilot.db.models import ErrorEvent, Failure, Machine, Maintenance, Telemetry

BATCH = 20_000


def read_events(path: Path, value_col: str, fleet_size: int) -> pd.DataFrame:
    """Read a small event CSV (errors, maint, failures) for machines 1..fleet_size."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df[df["machineID"] <= fleet_size].rename(
        columns={"datetime": "ts", "machineID": "machine_id", value_col: "value"}
    )


def iter_telemetry(path: Path, fleet_size: int, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield telemetry chunks for machines 1..fleet_size (file is sorted by machine)."""
    for chunk in pd.read_csv(path, parse_dates=["datetime"], chunksize=chunksize):
        part = chunk[chunk["machineID"] <= fleet_size]
        if not part.empty:
            yield part.rename(columns={"datetime": "ts", "machineID": "machine_id"})
        if chunk["machineID"].min() > fleet_size:
            break


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = df[cols].copy()
    if "ts" in out:
        out["ts"] = out["ts"].dt.to_pydatetime()
    return out.to_dict("records")


async def _insert(engine: AsyncEngine, table, rows: list[dict]) -> None:  # noqa: ANN001
    async with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            await conn.execute(insert(table), rows[i : i + BATCH])


async def fleet_loaded(engine: AsyncEngine) -> bool:
    """True when machine and telemetry rows already exist."""
    async with engine.connect() as conn:
        n = (await conn.execute(select(func.count()).select_from(Machine))).scalar_one()
    return n > 0


async def load_fleet(engine: AsyncEngine, raw_dir: Path, fleet_size: int) -> dict[str, int]:
    """Replace fleet tables with machines 1..fleet_size from raw_dir. Returns row counts."""
    async with engine.begin() as conn:
        for table in (Telemetry, ErrorEvent, Maintenance, Failure, Machine):
            await conn.execute(delete(table))

    machines = pd.read_csv(raw_dir / "machines.csv")
    machines = machines[machines["machineID"] <= fleet_size].rename(
        columns={"machineID": "machine_id"}
    )
    await _insert(engine, Machine.__table__, _records(machines, ["machine_id", "model", "age"]))
    counts = {"machines": len(machines)}

    for fname, col, model, attr in (
        ("errors.csv", "errorID", ErrorEvent, "error_id"),
        ("maint.csv", "comp", Maintenance, "component"),
        ("failures.csv", "failure", Failure, "component"),
    ):
        df = read_events(raw_dir / fname, col, fleet_size).rename(columns={"value": attr})
        await _insert(engine, model.__table__, _records(df, ["machine_id", "ts", attr]))
        counts[model.__tablename__] = len(df)

    n_tel = 0
    cols = ["machine_id", "ts", "volt", "rotate", "pressure", "vibration"]
    for chunk in iter_telemetry(raw_dir / "telemetry.csv", fleet_size):
        await _insert(engine, Telemetry.__table__, _records(chunk, cols))
        n_tel += len(chunk)
    counts["telemetry"] = n_tel
    return counts
