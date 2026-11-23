"""Password hashing, JWT tokens and the current-user dependency."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash

from copilot.db.models import User

_hasher = PasswordHash.recommended()
_bearer = HTTPBearer(auto_error=False)
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Argon2 hash of a password."""
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Check a password against its hash."""
    return _hasher.verify(password, hashed)


def create_token(user_id: int, secret: str, minutes: int) -> str:
    """Signed JWT with the user id as subject."""
    now = datetime.now(UTC)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> int:
    """User id from a valid token; raises jwt exceptions otherwise."""
    return int(jwt.decode(token, secret, algorithms=[ALGORITHM])["sub"])


async def current_user(
    request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)
) -> User:
    """FastAPI dependency: the authenticated user (401 otherwise)."""
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                                 headers={"WWW-Authenticate": "Bearer"})
    if creds is None:
        raise unauthorized
    rt = request.app.state.runtime
    try:
        user_id = decode_token(creds.credentials, rt.settings.jwt_secret.get_secret_value())
    except jwt.PyJWTError:
        raise unauthorized from None
    async with rt.db.session() as s:
        user = await s.get(User, user_id)
    if user is None:
        raise unauthorized
    return user
