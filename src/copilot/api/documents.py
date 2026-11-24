"""Render the knowledge documents as simple HTML pages (with section and page anchors)."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from copilot.agent.provenance import DATASET
from copilot.rag.chunking import PAGE, parse_front_matter

DATA_URL = DATASET["url"]
DOC_ID = re.compile(r"^[a-z0-9_]+$")


def slugify(text: str) -> str:
    """Anchor id for a section heading (the chat page builds the same ids)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _inline(text: str) -> str:
    """Escape text, then mark page markers and **bold**."""
    out = html.escape(text)
    out = PAGE.sub(lambda m: f'<span class="page" id="page-{m.group(1)}">p. {m.group(1)}</span>',
                   out)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)


def render_markdown(body: str) -> str:
    """Small Markdown subset: headings, paragraphs, bullet lists, pipe tables."""
    parts: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []
    table: list[list[str]] = []

    def flush() -> None:
        if paragraph:
            parts.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()
        if items:
            parts.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
        if table:
            head, *rows = [r for r in table if not re.fullmatch(r"[\s|:-]+", "|".join(r))]
            cells = "".join(f"<th>{_inline(c)}</th>" for c in head)
            body_rows = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows
            )
            parts.append(f"<table><tr>{cells}</tr>{body_rows}</table>")
            table.clear()

    for raw in body.splitlines():
        line = raw.strip()
        heading = re.match(r"^(#{1,3})\s+(.*)", line)
        if heading:
            flush()
            level = len(heading.group(1)) + 1
            text = heading.group(2)
            parts.append(f'<h{level} id="{slugify(text)}">{_inline(text)}</h{level}>')
        elif line.startswith("|"):
            if paragraph or items:
                flush()
            table.append([c.strip() for c in line.strip("|").split("|")])
        elif re.match(r"^[-*•]\s+", line):
            if paragraph or table:
                flush()
            items.append(re.sub(r"^[-*•]\s+", "", line))
        elif not line:
            flush()
        else:
            if items or table:
                flush()
            paragraph.append(line)
    flush()
    return "\n".join(parts)


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><link rel="stylesheet" href="/static/style.css"></head>
<body class="doc-page"><main class="document">
<p><a href="/">&larr; Back to Maintenance Copilot</a></p>
<h1>{title}</h1>
<p class="doc-meta">{notice}Licence: {licence}{source}</p>
{content}
</main><script>
// Highlight the section or page the chat linked to.
const t = location.hash && document.getElementById(location.hash.slice(1));
if (t) {{ t.classList.add("target"); t.scrollIntoView({{block: "center"}}); }}
</script></body></html>"""


def render_document(knowledge_dir: Path, doc_id: str) -> str | None:
    """Full HTML page for knowledge/<doc_id>.md, or None if it does not exist."""
    if not DOC_ID.match(doc_id):
        return None
    path = knowledge_dir / f"{doc_id}.md"
    if not path.is_file():
        return None
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    url = meta.get("source_url", "")
    source = (f' &middot; <a href="{html.escape(url)}" target="_blank" rel="noopener">'
              "Original PDF</a>" if url.startswith("http") else "")
    notice = ("<strong>Synthetic document</strong> written for this project. "
              if meta.get("synthetic", "").lower() == "true" else "")
    return PAGE_TEMPLATE.format(
        title=html.escape(meta.get("title", doc_id)), notice=notice,
        licence=html.escape(meta.get("licence", "")), source=source,
        content=render_markdown(body),
    )


HORIZON_TEXT = {
    "24h": "a component fails within the next 24 hours",
    "7d": "a component fails within the next 7 days",
}


def render_model_card(risk_model: Any) -> str:
    """Model page built from the trained bundle (measured on the Oct-Dec 2015 test period)."""
    if risk_model is None:
        content = "<p>The failure-risk model is not loaded.</p>"
    else:
        rows = []
        for name, h in risk_model.horizons.items():
            t = h.get("test_metrics", {})

            def fmt(key: str, t: dict = t) -> str:
                v = t.get(key)
                return "-" if v is None or v != v else f"{v:.3f}"

            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{HORIZON_TEXT.get(name, name)}</td>"
                f"<td>{h.get('alert_rate', 0):.0%}</td><td>{fmt('pr_auc_any')}</td>"
                f"<td>{fmt('precision_at_rate')}</td><td>{fmt('recall_at_rate')}</td>"
                f"<td>{fmt('recall_early')}</td></tr>"
            )
        content = (
            "<p>LightGBM classifiers trained on the Microsoft predictive-maintenance sample "
            "(simulated, 1000 machines, hourly data for 2015). Inputs at time t: the last 3 h "
            "and 24 h of sensor readings (mean and spread), error counts in the last 24 h, "
            "hours since each component was last replaced, machine model and age. Only data "
            "up to t is used.</p>"
            "<p>Time-based split: trained on January-July, alert thresholds set on "
            "August-September, measured on October-December 2015, with a gap of one horizon "
            "at each boundary so labels cannot leak across the split.</p>"
            "<h2>Measured on the test period</h2>"
            "<table><tr><th>Horizon</th><th>Predicts that</th><th>Alerts on</th>"
            "<th>PR-AUC</th><th>Precision</th><th>Recall</th>"
            "<th>Recall &gt;24 h before failure</th></tr>" + "".join(rows) + "</table>"
            "<p>PR-AUC: 1.0 is perfect, the base rate is a random guess. 'Alerts on' is the "
            "share of machine-hours flagged.</p>"
            "<h2>Caveat</h2><p>On this simulated data a warning error appears shortly before "
            "almost every failure, which makes 24-hour prediction unrealistically easy. The "
            "7-day horizon is the harder, more realistic task.</p>"
            f'<p>Data: <a href="{DATA_URL}" target="_blank" rel="noopener">Microsoft '
            "sample data</a> (CC BY 4.0).</p>"
        )
    return PAGE_TEMPLATE.format(title="Failure-risk model", notice="",
                                licence="model trained in this project", source="",
                                content=content)
