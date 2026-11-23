"""Integration tests against a real Postgres (CI service container).

Run with:  TEST_POSTGRES_URL=postgresql+asyncpg://copilot:copilot@localhost:5432/copilot \
           pytest -m postgres
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command
from sqlalchemy import text

from copilot.agent import tools as t
from copilot.agent.graph import build_graph, new_question
from copilot.config import Settings
from copilot.db.models import Base
from copilot.db.session import Database
from copilot.runtime import open_checkpointer
from scripts.load_db import grant_read_only
from tests.conftest import NOW, ScriptedLLM, _fill, fleet_frames, tool_call

pytestmark = pytest.mark.postgres
URL = os.getenv("TEST_POSTGRES_URL", "")
RO_URL = URL.replace("://copilot:copilot@", "://copilot_ro:copilot_ro@")


@pytest.fixture
async def pg() -> Database:
    if not URL:
        pytest.skip("TEST_POSTGRES_URL not set")
    db = Database(URL, RO_URL)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text(
            "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='copilot_ro') "
            "THEN CREATE ROLE copilot_ro LOGIN PASSWORD 'copilot_ro'; END IF; END $$"
        ))
        await conn.execute(text("GRANT USAGE ON SCHEMA public TO copilot_ro"))
    await db.create_all()
    await _fill(db, fleet_frames())
    await grant_read_only(db)
    yield db
    await db.dispose()


async def test_tools_on_postgres_read_only_role(pg: Database, risk_model) -> None:
    ctx = t.ToolContext(db=pg, risk_model=risk_model, retriever=None, now=NOW)
    out = await t.get_machine_status(ctx, t.MachineArgs(machine_id=1))
    assert out["errors_last_24h"] == {"error4": 2}
    risk = await t.predict_failure_risk(ctx, t.MachineArgs(machine_id=1))
    assert set(risk["horizons"]) == {"24h", "7d"}
    # Separate sessions: after an error Postgres aborts the whole transaction.
    for statement in ("DELETE FROM machines", "SELECT * FROM users"):
        async with pg.read() as s:
            with pytest.raises(Exception, match="permission denied"):
                await s.execute(text(statement))


async def test_checkpointer_resumes_ticket_on_postgres(pg: Database, risk_model) -> None:
    from contextlib import AsyncExitStack

    settings = Settings(database_url=URL, _env_file=None)
    args = {"machine_id": 2, "priority": "medium", "summary": "check comp2 drift"}
    llm = ScriptedLLM(script=[tool_call("create_ticket", args), AIMessage("Done [S1].")], seen=[])
    ctx = t.ToolContext(db=pg, risk_model=risk_model, retriever=None, now=NOW)
    cfg = {"configurable": {"thread_id": "pg-thread", "user_id": None}}
    async with AsyncExitStack() as stack:
        saver = await open_checkpointer(settings, stack)
        graph = build_graph(llm, ctx, "system", checkpointer=saver)
        state = await graph.ainvoke(new_question("ticket for machine 2"), cfg)
        assert "__interrupt__" in state
        state = await graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert state["messages"][-1].content == "Done [S1]."
    async with pg.session() as s:
        assert (await s.execute(text("SELECT count(*) FROM tickets"))).scalar_one() == 1
