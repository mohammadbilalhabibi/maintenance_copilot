"""Run Locust at several user counts against a running API and save measured results.

The API should be configured with the mock LLM (LLM_BASE_URL -> loadtest/mock_llm.py) to
measure the backend itself, or with the real provider for a small run.
Writes loadtest/results/<name>.json (summary per user count) from Locust's CSV stats.

Example:
  python loadtest/run_loadtest.py --name mock_sqlite --host http://127.0.0.1:8000 \
      --users 10 25 50 100 --duration 90
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "loadtest" / "results"


def read_stats(prefix: Path) -> dict[str, dict]:
    """Per-endpoint stats from Locust's <prefix>_stats.csv."""
    out = {}
    with open(f"{prefix}_stats.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["Name"] == "Aggregated":
                continue

            def num(key: str, row: dict = row) -> float | None:
                v = row.get(key, "")
                return float(v) if v not in ("", "N/A") else None

            out[row["Name"]] = {
                "requests": int(row["Request Count"]), "failures": int(row["Failure Count"]),
                "rps": num("Requests/s"), "median_ms": num("Median Response Time"),
                "p95_ms": num("95%"), "p99_ms": num("99%"), "max_ms": num("Max Response Time"),
            }
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--duration", type=int, default=90, help="seconds per step")
    parser.add_argument("--note", default="", help="setup description stored with results")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    health = httpx.get(f"{args.host}/api/health", timeout=10).json()

    steps = []
    for users in args.users:
        prefix = RESULTS / f"{args.name}_u{users}"
        cmd = [sys.executable, "-m", "locust", "-f", str(ROOT / "loadtest" / "locustfile.py"),
               "--headless", "-u", str(users), "-r", str(max(users // 10, 1)),
               "-t", f"{args.duration}s", "--host", args.host, "--csv", str(prefix),
               "--only-summary"]
        print(f"== {users} users for {args.duration} s")
        subprocess.run(cmd, check=False, cwd=ROOT)
        stats = read_stats(prefix)
        chat = stats.get("chat (full answer)", {})
        print(f"   chat: {chat}")
        steps.append({"users": users, "endpoints": stats})

    payload = {
        "name": args.name, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": args.host, "duration_s_per_step": args.duration, "api_health": health,
        "client_machine": platform.processor(), "note": args.note,
        "think_time_s": [3, 8], "steps": steps,
    }
    out = RESULTS / f"{args.name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
