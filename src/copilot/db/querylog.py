"""Record the SQL statements a tool runs (shown as 'technical details' of a source)."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine

QUERY_LOG: ContextVar[list[dict] | None] = ContextVar("query_log", default=None)
MAX_QUERIES = 12


def _plain(value: Any) -> Any:
    return value.isoformat(sep=" ") if isinstance(value, datetime) else value


def attach(engine: Engine) -> None:
    """Append every statement on `engine` to QUERY_LOG while a log is active."""

    @event.listens_for(engine, "before_cursor_execute")
    def _record(conn, cursor, statement, params, context, executemany) -> None:  # noqa: ANN001
        log = QUERY_LOG.get()
        if log is None or len(log) >= MAX_QUERIES:
            return
        if isinstance(params, dict):
            shown: Any = {k: _plain(v) for k, v in params.items()}
        elif isinstance(params, list | tuple):
            shown = [_plain(v) for v in params][:50]
        else:
            shown = params
        log.append({"sql": " ".join(statement.split())[:1500], "params": shown})
