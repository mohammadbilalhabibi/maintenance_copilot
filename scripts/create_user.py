"""Create a user, or reset its password (e.g. the demo account when registration is off)."""

from __future__ import annotations

import argparse
import asyncio
import os

from sqlalchemy import select

from copilot.api.auth import hash_password
from copilot.config import get_settings
from copilot.db.models import User
from copilot.db.session import Database


async def run(username: str, password: str) -> str:
    """Create or update the user; returns 'created' or 'updated'."""
    db = Database(get_settings().database_url)
    try:
        await db.create_all()
        async with db.session() as s:
            user = await s.scalar(select(User).where(User.username == username))
            if user is None:
                s.add(User(username=username, password_hash=hash_password(password)))
                outcome = "created"
            else:
                user.password_hash = hash_password(password)
                outcome = "updated"
            await s.commit()
        return outcome
    finally:
        await db.dispose()


def main() -> None:
    """CLI entry point. The password comes from --password or the DEMO_PASSWORD env var."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    parser.add_argument("--password", default=os.environ.get("DEMO_PASSWORD"))
    args = parser.parse_args()
    if not args.password or len(args.password) < 8:
        raise SystemExit("password missing or shorter than 8 characters")
    print(f"user {args.username}: {asyncio.run(run(args.username, args.password))}")


if __name__ == "__main__":
    main()
