"""Build every long-lived resource once at startup (DB, models, index, LLM, graph)."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel

from copilot.agent.graph import build_graph
from copilot.agent.llm import make_llm
from copilot.agent.prompts import system_prompt
from copilot.agent.tools import ToolContext
from copilot.config import Settings
from copilot.db.session import Database
from copilot.ml.predict import RiskModel
from copilot.rag import open_hybrid
from copilot.rag.retriever import FastEmbedder, Retriever

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Resources shared by all requests."""

    settings: Settings
    db: Database
    risk_model: RiskModel | None
    retriever: Retriever | None
    llm: BaseChatModel
    graph: Any
    ctx: ToolContext


def load_risk_model(settings: Settings) -> RiskModel | None:
    """Load the failure-risk model, or None (tool then reports it unavailable)."""
    if not settings.model_path.exists():
        log.warning("risk model not found at %s - run scripts/train_model.py", settings.model_path)
        return None
    return RiskModel.load(settings.model_path)


def load_retriever(settings: Settings) -> Retriever | None:
    """Load the hybrid retriever, or None if the index is missing."""
    if not (settings.index_dir / "chunks.jsonl").exists():
        log.warning("document index missing at %s - run scripts/build_index.py", settings.index_dir)
        return None
    embedder = FastEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    return open_hybrid(settings.index_dir, embedder, settings.embedding_model, settings.rrf_k)


async def open_checkpointer(settings: Settings, stack: AsyncExitStack) -> Any:
    """LangGraph checkpointer in the same database (Postgres or SQLite)."""
    url = settings.database_url
    if url.startswith("postgresql"):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        conn = url.replace("postgresql+asyncpg://", "postgresql://")
        saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(conn))
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = url.split("///", 1)[1]
        saver = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(path.replace(".sqlite", ".checkpoints.sqlite"))
        )
    await saver.setup()
    return saver


async def build_runtime(
    settings: Settings,
    stack: AsyncExitStack,
    *,
    llm: BaseChatModel | None = None,
    risk_model: RiskModel | None = None,
    retriever: Retriever | None = None,
    checkpointer: Any = None,
) -> Runtime:
    """Create all resources; arguments override defaults (used by tests)."""
    db = Database(settings.database_url, settings.tools_url)
    stack.push_async_callback(db.dispose)
    await db.create_all()
    risk_model = risk_model if risk_model is not None else load_risk_model(settings)
    retriever = retriever if retriever is not None else load_retriever(settings)
    llm = llm or make_llm(settings)
    if checkpointer is None:
        checkpointer = await open_checkpointer(settings, stack)
    ctx = ToolContext(
        db=db, risk_model=risk_model, retriever=retriever, now=settings.sim_now,
        search_top_k=settings.search_top_k, timeout=settings.tool_timeout,
    )
    graph = build_graph(
        llm, ctx, system_prompt(settings.fleet_size, settings.sim_now),
        max_steps=settings.agent_max_steps, checkpointer=checkpointer,
    )
    return Runtime(settings, db, risk_model, retriever, llm, graph, ctx)
