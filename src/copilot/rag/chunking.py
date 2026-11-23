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

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return asdict(self)


FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


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


def window_words(text: str, max_words: int, overlap: int) -> list[str]:
    """Split long text into overlapping word windows."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words) - overlap, step)]


def load_chunks(knowledge_dir: Path, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    """Chunk every .md file in knowledge_dir (sorted, so ids are stable)."""
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = path.stem
        for s_idx, (heading, text) in enumerate(split_sections(body)):
            for w_idx, part in enumerate(window_words(text, max_words, overlap)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#{s_idx}.{w_idx}",
                        doc_id=doc_id,
                        title=meta.get("title", doc_id),
                        section=heading,
                        text=part,
                        source_url=meta.get("source_url", ""),
                        licence=meta.get("licence", ""),
                        synthetic=meta.get("synthetic", "false").lower() == "true",
                    )
                )
    return chunks
