"""Alternative agent designs for the evaluation experiments.

- multi:  a router LLM call picks a specialist (data, documents, ticket, mixed); each
          specialist is the same agent loop with only its tools.
- routed: a router LLM call rates the question simple or complex; simple questions go to
          the small model (LLM_MODEL), complex ones to the large model (LARGE_MODEL).
Both keep the single agent's state, sources, step limit and ticket confirmation.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from copilot.agent.graph import AgentState, build_graph
from copilot.agent.llm import make_llm, usage_from_message
from copilot.agent.prompts import system_prompt

LARGE_MODEL = "gpt-4.1"
DATA_TOOLS = ["get_machine_status", "query_telemetry", "get_error_history",
              "get_maintenance_history", "predict_failure_risk", "fleet_overview"]
SPECIALISTS: dict[str, tuple[list[str] | None, str]] = {
    "data": (DATA_TOOLS, "You are the fleet-data specialist. Answer from the data tools."),
    "documents": (["search_manuals"], "You are the documentation specialist. Answer from the "
                  "handbook and safety guides via search_manuals."),
    "ticket": (["create_ticket", "search_manuals", "get_machine_status"],
               "You are the ticket specialist. Create the ticket the user asks for."),
    "mixed": (None, "Use all tools needed for this combined question."),
}


class Route(BaseModel):
    """Router decision for the multi-agent variant."""

    specialist: Literal["data", "documents", "ticket", "mixed"] = Field(
        description="data: facts about machines/fleet/risk. documents: error code meanings, "
        "procedures, ranges, safety, priorities. ticket: user asks to open a ticket. mixed: "
        "needs both data and documents, or anything else."
    )


class Complexity(BaseModel):
    """Router decision for the model-routing variant."""

    complexity: Literal["simple", "complex"] = Field(
        description="simple: one fact lookup or one document question. complex: several "
        "steps, combining data and documents, comparisons, or tickets."
    )


class RoutedState(AgentState, total=False):
    route: str


def _router(llm: Any, schema: type[BaseModel], instruction: str):  # noqa: ANN202
    structured = llm.with_structured_output(schema, include_raw=True)

    async def route(state: RoutedState) -> dict:
        question = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        out = await structured.ainvoke([SystemMessage(instruction), question])
        decision = out["parsed"]
        value = next(iter(decision.model_dump().values())) if decision else None
        return {"route": value, "usage": usage_from_message(out["raw"])}

    return route


def build_variant(name: str, rt: Any, settings: Any) -> Any:
    """Compile the named variant using the runtime's tools, LLM and settings."""
    prompt = system_prompt(settings.fleet_size, settings.sim_now)
    graph = StateGraph(RoutedState)
    if name == "multi":
        for key, (tools, extra) in SPECIALISTS.items():
            graph.add_node(key, build_graph(rt.llm, rt.ctx, f"{prompt}\n{extra}",
                                            max_steps=settings.agent_max_steps,
                                            tool_names=tools, name=key))
        graph.add_node("router", _router(rt.llm, Route, "Route the maintenance question."))
        targets = list(SPECIALISTS)
        default = "mixed"
    elif name == "routed":
        large = make_llm(settings, LARGE_MODEL)
        graph.add_node("simple", build_graph(rt.llm, rt.ctx, prompt, name="simple",
                                             max_steps=settings.agent_max_steps))
        graph.add_node("complex", build_graph(large, rt.ctx, prompt, name="complex",
                                              max_steps=settings.agent_max_steps))
        graph.add_node("router", _router(rt.llm, Complexity, "Rate the question."))
        targets = ["simple", "complex"]
        default = "complex"
    else:
        raise ValueError(f"unknown variant {name}")
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", lambda s: s.get("route") if s.get("route") in targets else default,
        {t: t for t in targets},
    )
    for t in targets:
        graph.add_edge(t, END)
    return graph.compile(checkpointer=rt.graph.checkpointer)
