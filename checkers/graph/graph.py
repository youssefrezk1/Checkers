from __future__ import annotations
from langgraph.graph import StateGraph, END
from checkers.state.state import CheckersState

# Simplified pipeline nodes
from checkers.nodes.scorer_node import scorer_node
from checkers.nodes.deterministic_proposal_node import deterministic_proposal_node
from checkers.agents.ranker_agent import ranker_agent
from checkers.agents.update_agent import update_agent


def _update_agent_routing(state: CheckersState) -> str:
    """
    Routing out of update_agent.
    game_over → end
    default   → scorer_node (loop)
    """
    if state.game_over:
        return "end"
    return "scorer_node"


def build_graph():
    graph = StateGraph(CheckersState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("scorer_node", scorer_node)
    graph.add_node("deterministic_proposal_node", deterministic_proposal_node)
    graph.add_node("ranker_agent", ranker_agent)
    graph.add_node("update_agent", update_agent)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("scorer_node")

    # ── Flow: scorer → proposal → ranker → update ─────────────────────────────
    graph.add_edge("scorer_node", "deterministic_proposal_node")
    graph.add_edge("deterministic_proposal_node", "ranker_agent")
    graph.add_edge("ranker_agent", "update_agent")

    # ── update_agent: loop or exit ────────────────────────────────────────────
    graph.add_conditional_edges(
        "update_agent",
        _update_agent_routing,
        {
            "scorer_node": "scorer_node",
            "end": END,
        }
    )

    return graph.compile()


checkers_graph = build_graph()
