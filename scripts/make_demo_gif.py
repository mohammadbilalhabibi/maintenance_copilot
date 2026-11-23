"""Record docs/demo.gif from a running app (real LLM): question, streamed answer, ticket.

Needs: playwright + chromium, Pillow, and the app running (default http://127.0.0.1:8700).
"""

from __future__ import annotations

import argparse
import io
import time
import uuid
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
QUESTION = ("Machine 17 keeps throwing errors. What's wrong, how risky is it, and what does the "
            "handbook say?")
TICKET = ("Open a medium priority ticket for machine 17, comp4: inspect the bearings at the "
          "next visit.")


class Recorder:
    """Collects screenshots as GIF frames with per-frame durations."""

    def __init__(self, page: Page, width: int) -> None:
        self.page, self.width = page, width
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def shot(self, ms: int = 120) -> None:
        img = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        h = int(img.height * self.width / img.width)
        self.frames.append(img.resize((self.width, h), Image.LANCZOS))
        self.durations.append(ms)

    def hold(self, ms: int) -> None:
        self.durations[-1] += ms

    def until(self, done, timeout: float = 60, every: float = 0.35) -> None:  # noqa: ANN001
        end = time.time() + timeout
        while time.time() < end and not done():
            self.shot(int(every * 1000))
            time.sleep(every)
        self.shot()

    def save(self, path: Path) -> None:
        palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in self.frames]
        palette[0].save(path, save_all=True, append_images=palette[1:], duration=self.durations,
                        loop=0, optimize=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8700")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    parser.add_argument("--width", type=int, default=880)
    args = parser.parse_args()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1180, "height": 760}, color_scheme="light")
        rec = Recorder(page, args.width)
        page.goto(args.url)
        page.fill("#username", f"demo_{uuid.uuid4().hex[:6]}")
        page.fill("#password", "demo-password")
        rec.shot(900)
        page.click("#register-btn")
        page.wait_for_selector("#app:not(.hidden)")
        page.fill("#input", QUESTION)
        rec.shot(1200)
        page.keyboard.press("Enter")
        rec.until(lambda: page.locator(".meta").count() >= 1)
        rec.hold(2500)
        # Click a handbook citation: the side panel shows the exact passage and links.
        docs = page.locator(".msg.assistant").last.locator(".src", has_text="Handbook")
        sid = docs.first.get_attribute("data-sid") if docs.count() else "S1"
        page.locator(f".answer button.cite[data-sid={sid}]").first.click()
        rec.shot(3500)
        page.locator(".answer button.cite[data-sid=S2]").first.click()  # model output
        rec.shot(3500)
        page.keyboard.press("Escape")
        page.fill("#input", TICKET)
        rec.shot(1000)
        page.keyboard.press("Enter")
        rec.until(lambda: page.locator(".confirm").count() >= 1)
        rec.hold(2000)
        page.locator(".confirm .yes").click()
        rec.until(lambda: page.locator(".meta").count() >= 2)
        rec.hold(1500)
        page.click("#show-tickets")
        page.wait_for_selector("#tickets table")
        rec.shot(3000)
        browser.close()
    args.out.parent.mkdir(exist_ok=True)
    rec.save(args.out)
    print(f"wrote {args.out} ({len(rec.frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
