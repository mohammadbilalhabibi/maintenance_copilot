"""HTTP API: auth, conversations, streamed chat, ticket confirmation, rate limit."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from copilot.api.app import create_app
from copilot.config import Settings
from copilot.db.session import Database
from copilot.rag import BM25Retriever
from copilot.rag.chunking import load_chunks
from tests.conftest import NOW, ScriptedLLM, _fill, fleet_frames, tool_call

SECRET = "test-secret-that-is-at-least-32-bytes"


def events(resp) -> list[dict]:  # noqa: ANN001
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def make_client(tmp_path: Path, risk_model, knowledge_dir):  # noqa: ANN001, ANN201
    url = f"sqlite+aiosqlite:///{(tmp_path / 'api.sqlite').as_posix()}"

    async def prepare() -> None:
        db = Database(url)
        await db.create_all()
        await _fill(db, fleet_frames())
        await db.dispose()

    asyncio.run(prepare())

    def build(script: list[AIMessage], **overrides) -> tuple[TestClient, ScriptedLLM]:  # noqa: ANN003
        settings = Settings(database_url=url, sim_now=NOW, fleet_size=3, jwt_secret=SECRET,
                            phoenix_collector_endpoint=None, knowledge_dir=knowledge_dir,
                            _env_file=None, **overrides)
        llm = ScriptedLLM(script=script, seen=[])
        app = create_app(settings, llm=llm, risk_model=risk_model,
                         retriever=BM25Retriever(load_chunks(knowledge_dir)),
                         checkpointer=InMemorySaver())
        return TestClient(app), llm

    return build


def login(client: TestClient, name: str = "tech1") -> dict:
    r = client.post("/api/auth/register", json={"username": name, "password": "password123"})
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_auth_required_and_login(make_client) -> None:  # noqa: ANN001
    client, _ = make_client([])
    with client:
        assert client.get("/api/conversations").status_code == 401
        login(client)
        bad = client.post("/api/auth/login", json={"username": "tech1", "password": "wrongpass1"})
        assert bad.status_code == 401
        ok = client.post("/api/auth/login", json={"username": "tech1", "password": "password123"})
        assert ok.status_code == 200 and ok.json()["username"] == "tech1"
        dup = client.post("/api/auth/register", json={"username": "tech1", "password": "x" * 9})
        assert dup.status_code == 409
        assert client.get("/api/health").json()["risk_model"] is True


def test_chat_streams_and_stores_history(make_client) -> None:  # noqa: ANN001
    client, _ = make_client([
        tool_call("get_machine_status", {"machine_id": 1}),
        AIMessage("Machine 1 is a model3 [S1]."),
    ])
    with client:
        h = login(client)
        conv = client.post("/api/conversations", headers=h).json()["id"]
        r = client.post(f"/api/chat/{conv}", json={"message": "status of machine 1?"}, headers=h)
        ev = events(r)
        types = [e["type"] for e in ev]
        assert types[0] == "tool_start" and "tool_end" in types and types[-1] == "done"
        done = ev[-1]
        assert done["answer"] == "Machine 1 is a model3 [S1]." and done["cited"] == ["S1"]
        assert done["usage"]["llm_calls"] == 2 and done["cost_usd"] > 0
        data = client.get(f"/api/conversations/{conv}", headers=h).json()
        assert data["title"] == "status of machine 1?"
        assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
        assert data["messages"][1]["meta"]["sources"][0]["tool"] == "get_machine_status"
        # Another user cannot see this conversation.
        other = login(client, "tech2")
        assert client.get(f"/api/conversations/{conv}", headers=other).status_code == 404


@pytest.mark.parametrize("durability", ["async", "exit"])
def test_ticket_confirmation_flow(make_client, durability: str) -> None:  # noqa: ANN001
    args = {"machine_id": 1, "component": "comp3", "priority": "high", "summary": "error4 twice"}
    client, _ = make_client([tool_call("create_ticket", args), AIMessage("Ticket created [S1].")],
                            checkpoint_durability=durability)
    with client:
        h = login(client)
        conv = client.post("/api/conversations", headers=h).json()["id"]
        ev = events(client.post(f"/api/chat/{conv}", json={"message": "open a ticket"}, headers=h))
        assert ev[-1]["type"] == "confirm" and ev[-1]["tickets"] == [args]
        assert client.get(f"/api/conversations/{conv}", headers=h).json()["pending"]
        blocked = client.post(f"/api/chat/{conv}", json={"message": "hello"}, headers=h)
        assert blocked.status_code == 409
        assert client.get("/api/tickets", headers=h).json() == []
        ev = events(client.post(f"/api/chat/{conv}/confirm", json={"approved": True}, headers=h))
        assert ev[-1]["type"] == "done" and ev[-1]["answer"] == "Ticket created [S1]."
        tickets = client.get("/api/tickets", headers=h).json()
        assert len(tickets) == 1 and tickets[0]["mine"] and tickets[0]["priority"] == "high"


def test_rate_limit(make_client) -> None:  # noqa: ANN001
    client, _ = make_client([AIMessage("a"), AIMessage("b")], rate_limit_per_minute=1)
    with client:
        h = login(client)
        conv = client.post("/api/conversations", headers=h).json()["id"]
        url = f"/api/chat/{conv}"
        assert client.post(url, json={"message": "q1"}, headers=h).status_code == 200
        assert client.post(url, json={"message": "q2"}, headers=h).status_code == 429


def test_passage_and_document_endpoints(make_client) -> None:  # noqa: ANN001
    client, _ = make_client([])
    with client:
        assert client.get("/api/passages/handbook%230.0").status_code == 401
        h = login(client)
        r = client.get("/api/passages/handbook#0.0".replace("#", "%23"), headers=h)
        assert r.status_code == 200 and r.json()["doc_id"] == "handbook"
        assert client.get("/api/passages/nope%230.0", headers=h).status_code == 404
        doc = client.get("/docs/handbook")
        assert doc.status_code == 200 and "Test Handbook" in doc.text
        assert client.get("/docs/..%2Fpyproject").status_code == 404


def test_machine_record_and_model_page(make_client) -> None:  # noqa: ANN001
    client, _ = make_client([])
    with client:
        assert client.get("/api/machines/1/record").status_code == 401
        h = login(client)
        r = client.get("/api/machines/1/record?start=2015-03-09T06:00&end=2015-03-10T06:00",
                       headers=h).json()
        assert r["window"] == {"from": "2015-03-09 06:00", "to": "2015-03-10 06:00"}
        assert r["shown"]["from"] == "2015-03-07 06:00"  # at least 3 days are shown
        assert len(r["telemetry"]) == 72 and r["telemetry"][-1]["ts"] == "2015-03-10 06:00"
        assert {e["error"] for e in r["errors"]} == {"error4"}  # future error5 is hidden
        future = client.get("/api/machines/1/record?end=2016-01-01T00:00", headers=h).json()
        assert future["window"]["to"] == "2015-03-10 06:00"  # never after the simulated now
        assert client.get("/api/machines/99/record", headers=h).status_code == 404
        page = client.get("/model")
        assert page.status_code == 200 and "Measured on the test period" in page.text


def test_index_page(make_client) -> None:  # noqa: ANN001
    client, _ = make_client([])
    with client:
        r = client.get("/")
        assert r.status_code == 200 and "Maintenance Copilot" in r.text
        assert client.get("/static/app.js").status_code == 200
