"""Download the Microsoft predictive-maintenance sample data and verify its checksums.

Source: github.com/microsoft/SQL-Server-R-Services-Samples, folder
PredictiveMaintenanceModelingGuide/Data (repo LICENSE: CC BY 4.0, code: MIT).
The files are simulated data for 1000 machines (hourly telemetry for 2015).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "9f08ba53b035a47d80ad861a9fe121dd4c6960a3"
BASE = (
    "https://media.githubusercontent.com/media/microsoft/SQL-Server-R-Services-Samples/"
    f"{COMMIT}/PredictiveMaintenanceModelingGuide/Data"
)
# sha256 and size from the Git LFS pointer files at COMMIT.
FILES: dict[str, tuple[str, int]] = {
    "machines.csv": ("d70736718dc48e90e42844c1f3a003dc332b5d40281a500b4ab1998eb05cb9bc", 16435),
    "errors.csv": ("f5817af3744ababd6af34802356f5698fc6755bba6c6ef2f93a162b5b9205ea0", 405509),
    "failures.csv": ("766ed7bf720510b88fdf4921903690513b836e710644a7e33fd4ed81513bf1d3", 221172),
    "maint.csv": ("e31a46a60c4eea067d694ef20651a453f1eae534d5ea731dd1028db9c9109582", 1072029),
    "telemetry.csv": (
        "7875052585c3a1f2ad8179963e00dd177da79153a6603a3b8e1d18d0120ce3d0",
        809939545,
    ),
}


def sha256_of(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, out_dir: Path) -> None:
    """Download one file (skipped when it already exists with the right hash)."""
    expected_sha, size = FILES[name]
    target = out_dir / name
    if target.exists() and target.stat().st_size == size and sha256_of(target) == expected_sha:
        print(f"ok (cached)  {name}")
        return
    tmp = target.with_suffix(".part")
    with httpx.stream("GET", f"{BASE}/{name}", follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        done = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                print(f"\r{name}: {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    actual = sha256_of(tmp)
    if actual != expected_sha:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {name}: {actual}")
    tmp.replace(target)
    print(f"ok           {name}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name, args.out)


if __name__ == "__main__":
    main()
