"""Browser end-to-end: real API server + mock LLM, driven with Playwright (Chromium).

Run with: python -m playwright install chromium && pytest -m e2e
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import joblib
import pytest

from copilot.db.session import Database
from tests.conftest import NOW, _fill, fleet_frames, tiny_bundle

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up")


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    tmp = tmp_path_factory.mktemp("e2e")
    db_url = f"sqlite+aiosqlite:///{(tmp / 'e2e.sqlite').as_posix()}"

    async def prepare() -> None:
        db = Database(db_url)
        await db.create_all()
        await _fill(db, fleet_frames())
        await db.dispose()

    asyncio.run(prepare())
    joblib.dump(tiny_bundle(), tmp / "model.joblib")
    llm_port, api_port = _free_port(), _free_port()
    env = {**os.environ, "MOCK_LATENCY_MS": "50", "DATABASE_URL": db_url,
           "LLM_BASE_URL": f"http://127.0.0.1:{llm_port}/v1", "LLM_API_KEY": "mock",
           "LLM_MODEL": "mock", "MODEL_PATH": str(tmp / "model.joblib"),
           "INDEX_DIR": str(tmp / "no-index"), "SIM_NOW": NOW.isoformat(), "FLEET_SIZE": "3",
           "JWT_SECRET": "e2e-secret-e2e-secret-e2e-secret-1234", "PHOENIX_COLLECTOR_ENDPOINT": ""}
    procs = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "loadtest.mock_llm:app", "--port",
                          str(llm_port)], cwd=ROOT, env=env),
        subprocess.Popen([sys.executable, "-m", "uvicorn", "copilot.api.main:app", "--port",
                          str(api_port)], cwd=ROOT, env=env),
    ]
    try:
        _wait(f"http://127.0.0.1:{llm_port}/health")
        _wait(f"http://127.0.0.1:{api_port}/api/health")
        yield f"http://127.0.0.1:{api_port}"
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)


def test_login_chat_and_history(server: str) -> None:
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(server)
        page.fill("#username", "e2e_user")
        page.fill("#password", "e2e-password")
        page.click("#register-btn")
        expect(page.locator("#app")).to_be_visible()

        page.fill("#input", "What is the status of machine 1?")
        page.keyboard.press("Enter")
        answer = page.locator(".msg.assistant .answer").last
        expect(answer).to_contain_text("mock LLM", timeout=20_000)
        expect(page.locator(".chip.ok").first).to_have_text("get_machine_status")
        expect(page.locator(".sources summary").last).to_contain_text("1 sources, 1 cited")

        page.reload()
        expect(page.locator(".conv .title").first).to_have_text("What is the status of machine 1?")
        page.locator(".conv .title").first.click()
        expect(page.locator(".msg.user")).to_have_count(1)
        expect(page.locator(".msg.assistant .answer")).to_contain_text("mock LLM")
        browser.close()
