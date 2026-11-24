"""Helpers used by the cloud deployment: TLS URLs, Phoenix auth header, demo user script."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import select

from copilot.config import Settings
from copilot.db.models import User
from copilot.db.session import Database
from copilot.observability import exporter_headers
from copilot.runtime import psycopg_url


def test_psycopg_url_converts_tls_option() -> None:
    url = "postgresql+asyncpg://u:p@db.example:5432/copilot?ssl=require"
    assert psycopg_url(url) == "postgresql://u:p@db.example:5432/copilot?sslmode=require"
    assert psycopg_url("postgresql+asyncpg://u:p@h/db?a=1&ssl=require").endswith("&sslmode=require")
    assert psycopg_url("postgresql+asyncpg://u:p@h/db") == "postgresql://u:p@h/db"


def test_phoenix_header_only_with_key() -> None:
    assert exporter_headers(Settings(_env_file=None)) == {}
    s = Settings(_env_file=None, phoenix_api_key="abc.def")
    assert exporter_headers(s) == {"authorization": "Bearer abc.def"}


def test_create_user_script_creates_then_resets(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from copilot.api.auth import verify_password
    from copilot.config import get_settings
    from scripts import create_user

    url = f"sqlite+aiosqlite:///{(tmp_path / 'u.sqlite').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        assert asyncio.run(create_user.run("demo", "first-password")) == "created"
        assert asyncio.run(create_user.run("demo", "second-password")) == "updated"

        async def stored_hash() -> str:
            db = Database(url)
            async with db.session() as s:
                user = await s.scalar(select(User).where(User.username == "demo"))
            await db.dispose()
            return user.password_hash

        assert verify_password("second-password", asyncio.run(stored_hash()))
    finally:
        get_settings.cache_clear()
