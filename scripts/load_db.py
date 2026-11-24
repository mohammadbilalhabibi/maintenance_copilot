"""Create tables and load the fleet sample data into DATABASE_URL."""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url

from copilot.config import PROJECT_ROOT, get_settings
from copilot.db.load import fleet_loaded, load_fleet
from copilot.db.session import Database

READ_ONLY_TABLES = "machines, telemetry, errors, maintenance, failures, tickets"
ROLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


async def create_read_only_role(db: Database, tools_url: str) -> str:
    """Create (or update the password of) the tools role named in TOOLS_DATABASE_URL.

    Needed where no init SQL runs (e.g. Azure Database for PostgreSQL). Returns the role.
    """
    url = make_url(tools_url)
    role, password, database = url.username or "", url.password or "", url.database or ""
    if not ROLE_NAME.match(role) or not ROLE_NAME.match(database.lower()) or len(password) < 8:
        raise SystemExit("TOOLS_DATABASE_URL needs a simple role name, database and password")
    secret = password.replace("'", "''")
    async with db.engine.begin() as conn:
        exists = await conn.scalar(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role})
        verb = "ALTER" if exists else "CREATE"
        await conn.execute(text(f"{verb} ROLE {role} LOGIN PASSWORD '{secret}'"))
        await conn.execute(text(f'GRANT CONNECT ON DATABASE "{database}" TO {role}'))
        await conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
    print(f"read-only role {role}: {'updated' if exists else 'created'}")
    return role


async def grant_read_only(db: Database, role: str = "copilot_ro") -> None:
    """Let the read-only tools role SELECT the fleet tables (never users or chats)."""
    async with db.engine.begin() as conn:
        exists = await conn.scalar(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role})
        if exists:
            await conn.execute(text(f"GRANT SELECT ON {READ_ONLY_TABLES} TO {role}"))
            print(f"granted SELECT on fleet tables to {role}")


async def run(raw_dir: Path, fleet_size: int, if_empty: bool, create_ro_role: bool = False) -> None:
    """Create tables, then load fleet data (optionally only when the DB is empty)."""
    settings = get_settings()
    db = Database(settings.database_url)
    try:
        await db.create_all()
        if if_empty and await fleet_loaded(db.engine):
            print("fleet data already loaded - skipping")
        else:
            t0 = time.perf_counter()
            counts = await load_fleet(db.engine, raw_dir, fleet_size)
            print(f"loaded {counts} in {time.perf_counter() - t0:.1f} s")
        if settings.database_url.startswith("postgresql"):
            role = "copilot_ro"
            if create_ro_role and settings.tools_database_url:
                role = await create_read_only_role(db, settings.tools_database_url)
            await grant_read_only(db, role)
    finally:
        await db.dispose()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--fleet-size", type=int, default=get_settings().fleet_size)
    parser.add_argument("--if-empty", action="store_true", help="skip if data exists")
    parser.add_argument("--create-ro-role", action="store_true",
                        help="create the read-only role from TOOLS_DATABASE_URL (Postgres)")
    args = parser.parse_args()
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    asyncio.run(run(args.raw, args.fleet_size, args.if_empty, args.create_ro_role))


if __name__ == "__main__":
    main()
