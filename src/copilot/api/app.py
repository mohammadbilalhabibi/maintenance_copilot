"""FastAPI app: auth, conversations with stored history, streaming chat (SSE), tickets."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, text

from copilot.agent.llm import cost_usd
from copilot.agent.runner import stream_events
from copilot.api.auth import create_token, current_user, hash_password, verify_password
from copilot.api.documents import render_document
from copilot.api.ratelimit import RateLimiter
from copilot.config import Settings, get_settings
from copilot.db.models import Conversation, Message, Ticket, User, utcnow
from copilot.observability import setup_tracing
from copilot.runtime import Runtime, build_runtime

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parents[1] / "static"


# ------------------------------------------------------------------ schemas
class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=8, max_length=128)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ConfirmIn(BaseModel):
    approved: bool


# ---------------------------------------------------------------- app setup
def create_app(settings: Settings | None = None, **runtime_overrides: Any) -> FastAPI:
    """Build the app. `runtime_overrides` go to build_runtime (tests inject fakes)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_tracing(settings)
        async with AsyncExitStack() as stack:
            app.state.runtime = await build_runtime(settings, stack, **runtime_overrides)
            app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
            app.state.locks = {}
            yield

    app = FastAPI(title="Maintenance Copilot", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def rt(request: Request) -> Runtime:
        return request.app.state.runtime

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        r = rt(request)
        async with r.db.session() as s:
            await s.execute(text("SELECT 1"))
        return {"status": "ok", "risk_model": r.risk_model is not None,
                "document_index": r.retriever is not None, "model": settings.llm_model,
                "sim_now": settings.sim_now.isoformat()}

    # --------------------------------------------------------------- auth
    def _token(user: User) -> Token:
        return Token(access_token=create_token(user.id, settings.jwt_secret.get_secret_value(),
                                               settings.jwt_expire_minutes),
                     username=user.username)

    @app.post("/api/auth/register", response_model=Token, status_code=201)
    async def register(body: Credentials, request: Request) -> Token:
        if not settings.allow_registration:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled")
        async with rt(request).db.session() as s:
            exists = await s.scalar(select(User).where(User.username == body.username))
            if exists:
                raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")
            user = User(username=body.username, password_hash=hash_password(body.password))
            s.add(user)
            await s.commit()
        return _token(user)

    @app.post("/api/auth/login", response_model=Token)
    async def login(body: Credentials, request: Request) -> Token:
        async with rt(request).db.session() as s:
            user = await s.scalar(select(User).where(User.username == body.username))
        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password")
        return _token(user)

    @app.get("/api/me")
    async def me(user: User = Depends(current_user)) -> dict:
        return {"id": user.id, "username": user.username}

    # ------------------------------------------------------ conversations
    async def _own_conversation(request: Request, conv_id: str, user: User) -> Conversation:
        async with rt(request).db.session() as s:
            conv = await s.get(Conversation, conv_id)
        if conv is None or conv.user_id != user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
        return conv

    @app.get("/api/conversations")
    async def list_conversations(request: Request, user: User = Depends(current_user)) -> list:
        async with rt(request).db.session() as s:
            rows = (await s.execute(
                select(Conversation).where(Conversation.user_id == user.id)
                .order_by(Conversation.updated_at.desc()).limit(100)
            )).scalars()
            return [{"id": c.id, "title": c.title, "updated_at": c.updated_at} for c in rows]

    @app.post("/api/conversations", status_code=201)
    async def new_conversation(request: Request, user: User = Depends(current_user)) -> dict:
        conv = Conversation(id=str(uuid.uuid4()), user_id=user.id)
        async with rt(request).db.session() as s:
            s.add(conv)
            await s.commit()
        return {"id": conv.id, "title": conv.title}

    @app.get("/api/conversations/{conv_id}")
    async def get_conversation(conv_id: str, request: Request,
                               user: User = Depends(current_user)) -> dict:
        conv = await _own_conversation(request, conv_id, user)
        async with rt(request).db.session() as s:
            msgs = (await s.execute(
                select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)
            )).scalars()
            messages = [{"role": m.role, "content": m.content, "meta": m.meta} for m in msgs]
        pending = await _pending_confirmation(request, conv_id)
        return {"id": conv.id, "title": conv.title, "messages": messages, "pending": pending}

    @app.delete("/api/conversations/{conv_id}", status_code=204)
    async def delete_conversation(conv_id: str, request: Request,
                                  user: User = Depends(current_user)) -> None:
        await _own_conversation(request, conv_id, user)
        async with rt(request).db.session() as s:
            await s.execute(delete(Message).where(Message.conversation_id == conv_id))
            await s.execute(delete(Conversation).where(Conversation.id == conv_id))
            await s.commit()

    async def _pending_confirmation(request: Request, conv_id: str) -> dict | None:
        snap = await rt(request).graph.aget_state({"configurable": {"thread_id": conv_id}})
        for task in getattr(snap, "tasks", ()) or ():
            for intr in getattr(task, "interrupts", ()) or ():
                return intr.value
        return None

    # --------------------------------------------------------------- chat
    def _sse(event: dict) -> str:
        return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"

    async def _run(request: Request, conv: Conversation, user: User,
                   question: str | None, resume: dict | None) -> StreamingResponse:
        r = rt(request)
        if not request.app.state.limiter.allow(user.id):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit reached, wait")
        locks: dict[str, asyncio.Lock] = request.app.state.locks
        lock = locks.setdefault(conv.id, asyncio.Lock())
        if lock.locked():
            raise HTTPException(status.HTTP_409_CONFLICT, "This chat is still answering")
        config = {"configurable": {"thread_id": conv.id, "user_id": user.id},
                  "metadata": {"user_id": user.id, "conversation_id": conv.id},
                  "run_name": "copilot-chat"}

        if question is not None:
            async with r.db.session() as s:
                s.add(Message(conversation_id=conv.id, role="user", content=question, meta={}))
                db_conv = await s.get(Conversation, conv.id)
                if db_conv.title == "New chat":
                    db_conv.title = question[:80]
                db_conv.updated_at = utcnow()
                await s.commit()

        async def gen() -> AsyncIterator[str]:
            async with lock:
                try:
                    async for event in stream_events(
                        r.graph, question, config, resume=resume,
                        durability=settings.checkpoint_durability,
                    ):
                        if event["type"] == "done":
                            event["cost_usd"] = round(cost_usd(event["usage"], settings), 6)
                            async with r.db.session() as s:
                                s.add(Message(
                                    conversation_id=conv.id, role="assistant",
                                    content=event["answer"],
                                    meta={k: event[k] for k in ("sources", "cited", "usage",
                                                                "tools", "latency_s", "cost_usd")},
                                ))
                                await s.commit()
                        yield _sse(event)
                except Exception as exc:  # noqa: BLE001 - reported to the client
                    log.exception("chat run failed")
                    yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"[:300]})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/chat/{conv_id}")
    async def chat(conv_id: str, body: ChatIn, request: Request,
                   user: User = Depends(current_user)) -> StreamingResponse:
        conv = await _own_conversation(request, conv_id, user)
        if await _pending_confirmation(request, conv_id):
            raise HTTPException(status.HTTP_409_CONFLICT, "Confirm or reject the ticket first")
        return await _run(request, conv, user, body.message, None)

    @app.post("/api/chat/{conv_id}/confirm")
    async def confirm(conv_id: str, body: ConfirmIn, request: Request,
                      user: User = Depends(current_user)) -> StreamingResponse:
        conv = await _own_conversation(request, conv_id, user)
        if not await _pending_confirmation(request, conv_id):
            raise HTTPException(status.HTTP_409_CONFLICT, "Nothing to confirm")
        return await _run(request, conv, user, None, {"approved": body.approved})

    # ---------------------------------------------------- sources / documents
    @app.get("/api/passages/{chunk_id:path}")
    async def passage(chunk_id: str, request: Request,
                      user: User = Depends(current_user)) -> dict:
        """One indexed passage with provenance (for sources of older answers)."""
        chunks = getattr(rt(request).retriever, "chunks", None) or []
        for c in chunks:
            if c.chunk_id == chunk_id:
                return c.to_dict()
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Passage not found")

    @app.get("/docs/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    async def document(doc_id: str) -> HTMLResponse:
        """Readable version of a knowledge document (public: synthetic or public domain)."""
        page = render_document(settings.knowledge_dir, doc_id)
        if page is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        return HTMLResponse(page)

    # ------------------------------------------------------------ tickets
    @app.get("/api/tickets")
    async def tickets(request: Request, user: User = Depends(current_user)) -> list:
        async with rt(request).db.session() as s:
            rows = (await s.execute(select(Ticket).order_by(Ticket.id.desc()).limit(50))).scalars()
            return [{"id": t.id, "machine_id": t.machine_id, "component": t.component,
                     "priority": t.priority, "summary": t.summary, "status": t.status,
                     "created_at": t.created_at, "mine": t.created_by == user.id} for t in rows]

    @app.get("/api/stats")
    async def stats(request: Request, user: User = Depends(current_user)) -> dict:
        async with rt(request).db.session() as s:
            n = await s.scalar(select(func.count()).select_from(Message)
                               .where(Message.role == "assistant"))
        return {"answers": n}

    return app
