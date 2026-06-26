from __future__ import annotations
from langgraph.graph import StateGraph, END
from checkers.state.state import CheckersState

# Simplified pipeline nodes
from checkers.nodes.scorer_node import scorer_agent
from checkers.nodes.proposer_node import proposer_node
from checkers.agents.explainer_agent import explainer_agent
from checkers.agents.updater_agent import updater_agent


def _updater_agent_routing(state: CheckersState) -> str:
    """
    Routing out of updater_agent.
    game_over → end
    default   → scorer_node (loop)
    """
    if state.game_over:
        return "end"
    return "scorer_agent"


def build_graph():
    graph = StateGraph(CheckersState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("scorer_agent", scorer_agent) 
    graph.add_node("proposer_agent", proposer_node)
    graph.add_node("explainer_agent", explainer_agent)
    graph.add_node("updater_agent", updater_agent)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("scorer_agent")

    # ── Flow: scorer → proposal → explainer → updater ────────────────────────
    graph.add_edge("scorer_agent", "proposer_agent")
    graph.add_edge("proposer_agent", "explainer_agent")
    graph.add_edge("explainer_agent", "updater_agent")

    # ── updater_agent: loop or exit ───────────────────────────────────────────
    graph.add_conditional_edges(
        "updater_agent",
        _updater_agent_routing,
        {
            "scorer_agent": "scorer_agent",
            "end": END,
        }
    )

    return graph.compile()


checkers_graph = build_graph()
