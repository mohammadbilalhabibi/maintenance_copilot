"""Shared fixtures: a tiny synthetic fleet in SQLite, a tiny risk model, a scripted LLM."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from copilot.agent.tools import ToolContext
from copilot.db.models import ErrorEvent, Failure, Machine, Maintenance, Telemetry
from copilot.db.session import Database
from copilot.ml.features import CLASSES, FEATURES
from copilot.ml.predict import RiskModel
from copilot.rag import BM25Retriever
from copilot.rag.chunking import load_chunks

NOW = datetime(2015, 3, 10, 6, 0)
START = NOW - timedelta(days=10)


def fleet_frames() -> dict[str, pd.DataFrame]:
    """Deterministic 3-machine fleet: 10 days of hourly telemetry up to NOW + 2 days."""
    rng = np.random.default_rng(0)
    hours = pd.date_range(START, NOW + timedelta(days=2), freq="h")
    tel = pd.concat(
        [
            pd.DataFrame({
                "machine_id": m, "ts": hours,
                "volt": 170 + rng.normal(0, 5, len(hours)),
                "rotate": 450 + rng.normal(0, 20, len(hours)),
                "pressure": 100 + rng.normal(0, 5, len(hours)),
                "vibration": 40 + rng.normal(0, 2, len(hours)),
            })
            for m in (1, 2, 3)
        ]
    )
    errors = pd.DataFrame({
        "machine_id": [1, 1, 1, 2, 1],
        "ts": [NOW - timedelta(hours=5), NOW - timedelta(hours=2), NOW - timedelta(days=3),
               NOW - timedelta(days=2), NOW + timedelta(hours=1)],  # last one is "future"
        "error_id": ["error4", "error4", "error1", "error2", "error5"],
    })
    maint = pd.DataFrame({
        "machine_id": [1, 1, 2, 3],
        "ts": [NOW - timedelta(days=9), NOW - timedelta(days=4), NOW - timedelta(days=6),
               NOW - timedelta(days=1)],
        "component": ["comp1", "comp3", "comp2", "comp4"],
    })
    failures = pd.DataFrame({"machine_id": [1], "ts": [NOW - timedelta(days=4)],
                             "component": ["comp3"]})
    machines = pd.DataFrame({"machine_id": [1, 2, 3], "model": ["model3", "model4", "model1"],
                             "age": [18, 7, 2]})
    return {"telemetry": tel, "errors": errors, "maint": maint, "failures": failures,
            "machines": machines}


async def _fill(db: Database, frames: dict[str, pd.DataFrame]) -> None:
    def rows(df: pd.DataFrame) -> list[dict]:
        out = df.copy()
        if "ts" in out:
            out["ts"] = out["ts"].dt.to_pydatetime()
        return out.to_dict("records")

    from sqlalchemy import insert

    async with db.engine.begin() as conn:
        await conn.execute(insert(Machine.__table__), rows(frames["machines"]))
        await conn.execute(insert(Telemetry.__table__), rows(frames["telemetry"]))
        await conn.execute(insert(ErrorEvent.__table__), rows(frames["errors"]))
        await conn.execute(insert(Maintenance.__table__), rows(frames["maint"]))
        await conn.execute(insert(Failure.__table__), rows(frames["failures"]))


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """SQLite database with the synthetic fleet."""
    database = Database(f"sqlite+aiosqlite:///{(tmp_path / 'test.sqlite').as_posix()}")
    await database.create_all()
    await _fill(database, fleet_frames())
    yield database
    await database.dispose()


def tiny_bundle() -> dict[str, Any]:
    """Small LightGBM models on random data, in the real bundle format."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
    y = np.arange(400) % len(CLASSES)
    horizons = {}
    for name, rate in (("24h", 0.02), ("7d", 0.10)):
        model = lgb.LGBMClassifier(n_estimators=5, verbose=-1, random_state=0).fit(x, y)
        horizons[name] = {"model": model, "threshold": 0.5, "alert_rate": rate,
                          "test_metrics": {}}
    return {"features": FEATURES, "classes": CLASSES, "horizons": horizons}


@pytest.fixture(scope="session")
def risk_model() -> RiskModel:
    """Tiny risk model (fast to build, same interface as the real one)."""
    return RiskModel(tiny_bundle())


@pytest.fixture(scope="session")
def knowledge_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two small knowledge documents."""
    d = tmp_path_factory.mktemp("knowledge")
    (d / "handbook.md").write_text(
        "---\ntitle: Test Handbook\nsynthetic: true\nlicence: CC BY 4.0\n---\n"
        "# Error codes\nerror4 means the pressure sensor reports a fault on comp3.\n"
        "## Ticket priorities\nUse high priority when a machine is on alert.\n",
        encoding="utf-8",
    )
    (d / "safety.md").write_text(
        "---\ntitle: Safety Guide\nsource_url: https://example.org\n---\n"
        "# Lockout\nAlways lock out and tag out energy sources before maintenance.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def ctx(db: Database, risk_model: RiskModel, knowledge_dir: Path) -> ToolContext:
    """Tool context over the synthetic fleet, with BM25 document search."""
    return ToolContext(db=db, risk_model=risk_model,
                       retriever=BM25Retriever(load_chunks(knowledge_dir)), now=NOW)


class ScriptedLLM(BaseChatModel):
    """Chat model that replays scripted AIMessages (and records what it was sent)."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> ScriptedLLM:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        msg = self.script.pop(0) if self.script else AIMessage("(script exhausted)")
        msg = msg.model_copy(update={
            "usage_metadata": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        })
        return ChatResult(generations=[ChatGeneration(message=msg)])


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    """AIMessage that requests one tool call."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id,
                                      "type": "tool_call"}])


@pytest.fixture
def scripted() -> Iterator[type[ScriptedLLM]]:
    """The ScriptedLLM class (tests build instances with their own script)."""
    yield ScriptedLLM
