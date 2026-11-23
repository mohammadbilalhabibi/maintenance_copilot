"""Agent graph with a scripted LLM: tool loop, sources, ticket confirmation, step limit."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import func, select

from copilot.agent.graph import build_graph, new_question
from copilot.db.models import Ticket
from tests.conftest import tool_call

CFG = {"configurable": {"thread_id": "t1", "user_id": None}}


async def test_tool_loop_numbers_sources(ctx, scripted) -> None:
    llm = scripted(script=[
        AIMessage("", tool_calls=[
            {"name": "get_machine_status", "args": {"machine_id": 1}, "id": "a",
             "type": "tool_call"},
            {"name": "search_manuals", "args": {"query": "error4"}, "id": "b",
             "type": "tool_call"},
        ]),
        AIMessage("Machine 1 had 2 error4 events [S1]; error4 is a pressure fault [S2]."),
    ], seen=[])
    graph = build_graph(llm, ctx, "system", checkpointer=InMemorySaver())
    state = await graph.ainvoke(new_question("What's up with machine 1?"), CFG)
    assert state["messages"][-1].content.startswith("Machine 1 had")
    ids = [s["id"] for s in state["sources"]]
    assert ids[0] == "S1" and state["sources"][0]["tool"] == "get_machine_status"
    assert state["sources"][1]["tool"] == "search_manuals"
    assert state["usage"]["llm_calls"] == 2 and state["usage"]["tool_calls"] == 2
    # Second question resets per-question sources but keeps the conversation.
    llm.script.append(AIMessage("Only maintenance questions, sorry."))
    state = await graph.ainvoke(new_question("Tell me a joke"), CFG)
    assert state["sources"] == [] and len(state["messages"]) == 7


async def test_tool_error_is_given_to_llm(ctx, scripted) -> None:
    llm = scripted(script=[tool_call("get_machine_status", {"machine_id": 500}),
                           AIMessage("Machine 500 does not exist.")], seen=[])
    graph = build_graph(llm, ctx, "system")
    state = await graph.ainvoke(new_question("status of machine 500"))
    tool_msg = [m for m in state["messages"] if isinstance(m, ToolMessage)][0]
    assert tool_msg.status == "error" and "does not exist" in tool_msg.content
    assert state["sources"] == []


async def _ticket_count(ctx) -> int:
    async with ctx.db.session() as s:
        return (await s.execute(select(func.count()).select_from(Ticket))).scalar_one()


async def test_ticket_needs_confirmation(ctx, scripted) -> None:
    args = {"machine_id": 1, "component": "comp3", "priority": "high", "summary": "error4 x2"}
    llm = scripted(script=[tool_call("create_ticket", args), AIMessage("Ticket 1 created [S1].")],
                   seen=[])
    graph = build_graph(llm, ctx, "system", checkpointer=InMemorySaver())
    state = await graph.ainvoke(new_question("open a ticket for machine 1"), CFG)
    assert state["__interrupt__"][0].value["tickets"] == [args]
    assert await _ticket_count(ctx) == 0  # nothing written before the user answers
    state = await graph.ainvoke(Command(resume={"approved": True}), CFG)
    assert await _ticket_count(ctx) == 1
    assert state["messages"][-1].content == "Ticket 1 created [S1]."


async def test_declined_ticket_is_not_written(ctx, scripted) -> None:
    args = {"machine_id": 2, "priority": "low", "summary": "check comp2"}
    llm = scripted(script=[tool_call("create_ticket", args), AIMessage("OK, no ticket.")],
                   seen=[])
    graph = build_graph(llm, ctx, "system", checkpointer=InMemorySaver())
    await graph.ainvoke(new_question("ticket for machine 2"), CFG)
    state = await graph.ainvoke(Command(resume={"approved": False}), CFG)
    assert await _ticket_count(ctx) == 0
    declined = [m for m in state["messages"] if isinstance(m, ToolMessage)][-1]
    assert "declined" in declined.content


async def test_step_limit_forces_answer(ctx, scripted) -> None:
    loop = [tool_call("get_machine_status", {"machine_id": 1}, f"c{i}") for i in range(5)]
    llm = scripted(script=[*loop, AIMessage("final")], seen=[])
    graph = build_graph(llm, ctx, "system", max_steps=2)
    state = await graph.ainvoke(new_question("loop forever"))
    # Steps 1 and 2 call tools; step 3 must answer. The scripted LLM still returns a tool
    # call there, which the agent drops so no tool call is left unanswered.
    assert state["steps"] == 3
    assert "Tool limit reached" in llm.seen[-1][-1].content
    assert not state["messages"][-1].tool_calls
