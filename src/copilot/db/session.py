"""Async engines and sessions: one read-write, one read-only for agent tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from copilot.db.models import Base


def make_engine(url: str, *, read_only: bool = False) -> AsyncEngine:
    """Create an async engine. For SQLite, read_only opens the file in query-only mode."""
    if url.startswith("sqlite"):
        engine = create_async_engine(url, connect_args={"timeout": 30})

        @event.listens_for(engine.sync_engine, "connect")
        def _pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            if read_only:
                cur.execute("PRAGMA query_only=ON")
            cur.close()

        return engine
    return create_async_engine(url, pool_size=10, max_overflow=10, pool_pre_ping=True)


class Database:
    """Holds the read-write and read-only engines for the app's lifetime."""

    def __init__(self, url: str, tools_url: str | None = None) -> None:
        self.engine = make_engine(url)
        self.ro_engine = (
            make_engine(tools_url, read_only=True)
            if tools_url and tools_url != url
            else make_engine(url, read_only=True)
        )
        self.session = async_sessionmaker(self.engine, expire_on_commit=False)
        self.ro_session = async_sessionmaker(self.ro_engine, expire_on_commit=False)

    async def create_all(self) -> None:
        """Create missing tables (idempotent)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def read(self) -> AsyncIterator[AsyncSession]:
        """Read-only session for agent tools."""
        async with self.ro_session() as s:
            yield s

    async def dispose(self) -> None:
        """Close all connections."""
        await self.engine.dispose()
        await self.ro_engine.dispose()
