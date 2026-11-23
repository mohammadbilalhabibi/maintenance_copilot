"""Render the knowledge documents as simple HTML pages (with section and page anchors)."""

from __future__ import annotations

import html
import re
from pathlib import Path

from copilot.rag.chunking import PAGE, parse_front_matter

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
