"""Smoke test a running deployment: health, register, chat (SSE), history, Phoenix traces."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

import httpx


def main() -> None:
    """CLI entry point; exits non-zero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--phoenix", default=None, help="Phoenix URL to check for traces")
    args = parser.parse_args()
    c = httpx.Client(base_url=args.base_url, timeout=60)

    health = c.get("/api/health").json()
    print("health:", health)
    assert health["status"] == "ok" and health["risk_model"] and health["document_index"]

    user = f"smoke_{uuid.uuid4().hex[:8]}"
    token = c.post("/api/auth/register", json={"username": user, "password": "smoke-password"})
    token.raise_for_status()
    h = {"Authorization": f"Bearer {token.json()['access_token']}"}
    conv = c.post("/api/conversations", headers=h).json()["id"]

    t0 = time.perf_counter()
    events = []
    with c.stream("POST", f"/api/chat/{conv}", json={"message": "What is the status of "
                  "machine 17?"}, headers=h) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    types = [e["type"] for e in events]
    compact = [t for i, t in enumerate(types) if t != "token" or types[i - 1] != "token"]
    print(f"chat events: {compact} ({types.count('token')} tokens) "
          f"in {time.perf_counter() - t0:.2f} s")
    assert "tool_start" in types and types[-1] == "done", events[-1]
    assert events[-1]["sources"], "no sources"

    hist = c.get(f"/api/conversations/{conv}", headers=h).json()
    assert [m["role"] for m in hist["messages"]] == ["user", "assistant"]
    print("history ok")

    if args.phoenix:
        time.sleep(6)  # batch span exporter flushes every few seconds
        query = {"query": "{ projects { edges { node { name traceCount } } } }"}
        resp = httpx.post(f"{args.phoenix}/graphql", json=query, timeout=20)
        assert resp.status_code == 200, f"Phoenix GraphQL returned {resp.status_code}"
        data = resp.json()
        projects = {e["node"]["name"]: e["node"]["traceCount"]
                    for e in data["data"]["projects"]["edges"]}
        print("phoenix projects:", projects)
        assert projects.get("maintenance-copilot", 0) > 0, "no traces in Phoenix"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - any failure fails the smoke test
        # "::error::" makes GitHub Actions show the reason as a public annotation.
        print(f"::error::SMOKE TEST FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
