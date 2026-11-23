"""Source provenance: page numbers in chunks, document pages, data kept with sources."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from copilot.agent.graph import build_graph, new_question
from copilot.api.documents import render_document, slugify
from copilot.rag.chunking import load_chunks
from tests.conftest import tool_call


def test_chunks_carry_pages_without_markers(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text(
        "---\ntitle: Guide\nsource_url: https://example.org/g.pdf\n---\n"
        "## First\n[[page 1]] Alpha beta gamma.\n\n## Second\nDelta [[page 2]] epsilon zeta.\n"
        "## Third\nEta theta.\n",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    assert [(c.section, c.page) for c in chunks] == [("First", 1), ("Second", 1), ("Third", 2)]
    assert chunks[0].text == "Alpha beta gamma." and "[[page" not in chunks[1].text


def test_long_section_windows_get_their_own_page(tmp_path: Path) -> None:
    words = " ".join(f"w{i}" for i in range(300))
    (tmp_path / "g.md").write_text(f"## S\n[[page 4]] {words[:600]} [[page 5]] {words[600:]}\n",
                                   encoding="utf-8")
    chunks = load_chunks(tmp_path, max_words=100, overlap=10)
    assert chunks[0].page == 4 and chunks[-1].page == 5


def test_render_document_has_anchors_and_escapes(knowledge_dir: Path) -> None:
    page = render_document(knowledge_dir, "handbook")
    assert f'id="{slugify("Ticket priorities")}"' in page and "Synthetic document" in page
    assert render_document(knowledge_dir, "../secrets") is None
    assert render_document(knowledge_dir, "missing") is None


async def test_sources_keep_passage_and_tool_data(ctx, scripted) -> None:
    llm = scripted(script=[
        AIMessage("", tool_calls=[
            {"name": "get_machine_status", "args": {"machine_id": 1}, "id": "a",
             "type": "tool_call"},
            {"name": "search_manuals", "args": {"query": "error4"}, "id": "b",
             "type": "tool_call"},
        ]),
        AIMessage("done [S1][S2]"),
    ], seen=[])
    state = await build_graph(llm, ctx, "system").ainvoke(new_question("q"))
    data_src, doc_src = state["sources"][0], state["sources"][1]
    assert data_src["data"]["model"] == "model3"
    assert doc_src["text"].startswith("error4") and doc_src["doc_id"] == "handbook"
    assert doc_src["synthetic"] is True
    # The LLM never sees the UI-only provenance block.
    search_msg = [m for m in state["messages"] if isinstance(m, ToolMessage)][1]
    assert "_meta" not in json.loads(search_msg.content)


async def test_sources_from_tool_errors_are_not_listed(ctx, scripted) -> None:
    llm = scripted(script=[tool_call("get_machine_status", {"machine_id": 999}),
                           AIMessage("no such machine")], seen=[])
    state = await build_graph(llm, ctx, "system").ainvoke(new_question("q"))
    assert state["sources"] == []
