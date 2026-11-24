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


async def test_data_sources_have_origin_and_sql(ctx, scripted) -> None:
    llm = scripted(script=[
        AIMessage("", tool_calls=[
            {"name": "get_error_history", "args": {"machine_id": 1, "days_back": 7}, "id": "a",
             "type": "tool_call"},
            {"name": "predict_failure_risk", "args": {"machine_id": 1}, "id": "b",
             "type": "tool_call"},
        ]),
        AIMessage("done [S1][S2]"),
    ], seen=[])
    state = await build_graph(llm, ctx, "system").ainvoke(new_question("q"))
    errors_src, risk_src = state["sources"]
    assert errors_src["origin"]["kind"] == "database"
    assert errors_src["origin"]["tables"] == ["errors"]
    assert errors_src["origin"]["window"] == {"from": "2015-03-03 06:00", "to": "2015-03-10 06:00"}
    assert errors_src["origin"]["machine_ids"] == [1]
    assert any("FROM errors" in q["sql"] for q in errors_src["queries"])
    assert risk_src["origin"]["kind"] == "model" and risk_src["origin"]["caveat"]
    assert risk_src["origin"]["window"]["to"] == "2015-03-10 06:00"
    # Each tool gets only its own queries, and the LLM never sees them.
    # The two tools ran concurrently; the risk tool's telemetry query must not leak.
    assert not any("telemetry" in q["sql"] for q in errors_src["queries"])
    assert any("telemetry" in q["sql"] for q in risk_src["queries"])
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            assert "_queries" not in msg.content and "_meta" not in msg.content


def test_describe_fleet_ranking_lists_machines() -> None:
    from copilot.agent.provenance import describe

    origin = describe("fleet_overview", {"metric": "failures"},
                      {"metric": "failures", "from": "2015-10-16 08:00",
                       "to": "2015-11-15 08:00", "ranking": [{"machine_id": 7, "count": 2},
                                                            {"machine_id": 3, "count": 1}]})
    assert origin["machine_ids"] == [7, 3] and origin["tables"] == ["failures"]
    assert origin["dataset"]["url"].startswith("https://github.com/microsoft/")
