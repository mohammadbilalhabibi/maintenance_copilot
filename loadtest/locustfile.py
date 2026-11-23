"""Locust scenario: technicians log in, open a chat and ask questions (streamed answers).

Each simulated user registers once, then asks a question every 3-8 s and sometimes reloads
the chat list. The chat request is timed until the final 'done' event; time to the first
streamed event is reported as a separate entry ("chat first event").
"""

from __future__ import annotations

import json
import random
import time
import uuid

from locust import HttpUser, between, events, task

QUESTIONS = [
    "What is the status of machine {m}?",
    "How many errors did machine {m} log in the last 7 days?",
    "What does error4 mean?",
    "When was comp2 of machine {m} last replaced?",
    "What is the normal vibration range?",
]


class Technician(HttpUser):
    """One technician using the chat."""

    wait_time = between(3, 8)

    def on_start(self) -> None:
        name = f"load_{uuid.uuid4().hex[:10]}"
        r = self.client.post("/api/auth/register", json={"username": name,
                                                          "password": "load-test-pw"},
                             name="register")
        self.headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        self.conv = self.client.post("/api/conversations", headers=self.headers,
                                     name="new conversation").json()["id"]

    @task(5)
    def ask(self) -> None:
        """Ask one question; time to first SSE event and to the final 'done' event."""
        question = random.choice(QUESTIONS).format(m=random.randint(1, 100))
        t0 = time.perf_counter()
        first: float | None = None
        error: Exception | None = None
        last: dict | None = None
        try:
            with self.client.post(f"/api/chat/{self.conv}", json={"message": question},
                                  headers=self.headers, stream=True, name="chat (headers)",
                                  catch_response=True) as r:
                if r.status_code != 200:
                    r.failure(f"status {r.status_code}")
                    error = RuntimeError(f"status {r.status_code}")
                else:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        if first is None:
                            first = time.perf_counter() - t0
                        if line.startswith(b"data: "):
                            last = json.loads(line[6:])
                    r.success()
        except Exception as exc:  # noqa: BLE001 - reported to Locust
            error = exc
        if error is None and (not last or last.get("type") != "done"):
            error = RuntimeError(f"no done event: {last}")
        total = time.perf_counter() - t0
        events.request.fire(request_type="SSE", name="chat (full answer)",
                            response_time=total * 1000, response_length=0, exception=error,
                            context={})
        if first is not None:
            events.request.fire(request_type="SSE", name="chat first event",
                                response_time=first * 1000, response_length=0, exception=None,
                                context={})

    @task(1)
    def history(self) -> None:
        self.client.get("/api/conversations", headers=self.headers, name="list conversations")
