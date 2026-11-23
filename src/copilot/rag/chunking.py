"""Load knowledge documents (Markdown with a small front-matter block) and chunk them."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    """One searchable passage with its provenance."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source_url: str
    licence: str
    synthetic: bool
    page: int | None = None  # PDF page where the chunk starts (converted PDFs only)

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return asdict(self)


FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
PAGE = re.compile(r"\[\[page (\d+)\]\]")
PAGE_TOKEN = re.compile(r"\[\[page \d+\]\] ?")


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split 'key: value' front matter from the body."""
    m = FRONT.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end() :]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading, text) pairs at '#'/'##'/'###' headings."""
    sections: list[tuple[str, str]] = []
    heading, lines = "Introduction", []
    for line in body.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)", line)
        if m:
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = m.group(1).strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def paged_words(text: str, page: int | None) -> tuple[list[str], list[int | None], int | None]:
    """Split text into words, remove [[page N]] markers and note each word's page.

    Returns (words, page per word, page in effect at the end of the text).
    """
    words: list[str] = []
    pages: list[int | None] = []
    for i, part in enumerate(PAGE.split(text)):
        if i % 2:  # captured page number
            page = int(part)
            continue
        for word in part.split():
            words.append(word)
            pages.append(page)
    return words, pages, page


def window_words(words: list[str], max_words: int, overlap: int) -> list[tuple[int, list[str]]]:
    """Split a word list into overlapping windows: (start index, words)."""
    if len(words) <= max_words:
        return [(0, words)]
    step = max_words - overlap
    return [(i, words[i : i + max_words]) for i in range(0, len(words) - overlap, step)]


def load_chunks(knowledge_dir: Path, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    """Chunk every .md file in knowledge_dir (sorted, so ids are stable)."""
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = path.stem
        page: int | None = None
        for s_idx, (heading, text) in enumerate(split_sections(body)):
            words, pages, page_after = paged_words(text, page)
            for w_idx, (start, part) in enumerate(window_words(words, max_words, overlap)):
                if not part:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#{s_idx}.{w_idx}",
                        doc_id=doc_id,
                        title=meta.get("title", doc_id),
                        section=PAGE.sub("", heading).strip(),
                        # A section that fits in one chunk keeps its original text.
                        text=PAGE_TOKEN.sub("", text).strip() if len(part) == len(words)
                        else " ".join(part),
                        source_url=meta.get("source_url", ""),
                        licence=meta.get("licence", ""),
                        synthetic=meta.get("synthetic", "false").lower() == "true",
                        page=pages[start] if pages else page,
                    )
                )
            page = page_after
    return chunks
