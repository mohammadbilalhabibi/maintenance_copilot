"""Download the public OSHA guides and convert them to Markdown in knowledge/.

Both are U.S. federal government works (public domain, 17 U.S.C. 105); OSHA 3170 also
states "This publication is in the public domain". Run once; the Markdown is committed.
Needs PyMuPDF (pip install pymupdf).
"""

from __future__ import annotations

import argparse
import re
import tempfile
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
GUIDES = {
    "osha3120_lockout_tagout": (
        "https://www.osha.gov/sites/default/files/publications/osha3120.pdf",
        "OSHA 3120 - Control of Hazardous Energy (Lockout/Tagout)",
    ),
    "osha3170_safeguarding_equipment": (
        "https://www.osha.gov/sites/default/files/publications/osha3170.pdf",
        "OSHA 3170 - Safeguarding Equipment and Protecting Employees from Amputations",
    ),
}
LICENCE = "Public domain (U.S. federal government work, 17 U.S.C. 105)"


def pdf_to_markdown(pdf: Path) -> str:
    """Extract text; short all-bold or larger-font lines become '##' headings."""
    import pymupdf

    doc = pymupdf.open(pdf)
    lines: list[tuple[float, bool, str]] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text and spans:
                    bold = all("Bold" in s["font"] or s["flags"] & 16 for s in spans)
                    lines.append((max(s["size"] for s in spans), bold, text))
    body = Counter(round(size) for size, _, _ in lines).most_common(1)[0][0]
    out: list[str] = []
    for size, bold, text in lines:
        if re.fullmatch(r"\d{1,3}", text) or "...." in text:  # page numbers, contents
            continue
        if (size >= body + 2 or bold) and len(text) < 100:
            out.append(f"\n## {text}\n")
        else:
            out.append(text)
    md = "\n".join(out)
    md = re.sub(r"-\n(?=[a-z])", "", md)  # re-join hyphenated words
    md = re.sub(r"(?<![\n.:;!?])\n(?![\n#•\-])", " ", md)  # re-flow paragraphs
    md = re.sub(r"\n## (.+)\n+## ", r"\n## \1 ", md)  # merge multi-line headings
    md = md.replace("�", "'").replace("’", "'")
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "knowledge")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        for doc_id, (url, title) in GUIDES.items():
            pdf = Path(tmp) / f"{doc_id}.pdf"
            resp = httpx.get(url, headers={"User-Agent": UA}, follow_redirects=True, timeout=60)
            resp.raise_for_status()
            pdf.write_bytes(resp.content)
            front = (
                f"---\ntitle: {title}\nsource_url: {url}\nlicence: {LICENCE}\n"
                "synthetic: false\n---\n\n"
            )
            (args.out / f"{doc_id}.md").write_text(front + pdf_to_markdown(pdf) + "\n",
                                                   encoding="utf-8")
            print(f"wrote {doc_id}.md")


if __name__ == "__main__":
    main()
