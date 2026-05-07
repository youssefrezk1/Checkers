"""
Routing unit tests for the simplified pipeline graph.

The only conditional routing function is _update_agent_routing.
All other edges in the simplified pipeline are direct (no routing function).
"""
from __future__ import annotations

from checkers.graph.graph import _update_agent_routing
from checkers.state.state import CheckersState


def _state(**kwargs) -> CheckersState:
    return CheckersState(**kwargs)


# ── _update_agent_routing ─────────────────────────────────────────────────────

def test_update_agent_game_over_routes_to_end():
    state = _state(game_over=True)
    assert _update_agent_routing(state) == "end"


def test_update_agent_game_continues_routes_to_scorer_node():
    state = _state(game_over=False)
    assert _update_agent_routing(state) == "scorer_node"


def test_update_agent_draw_routes_to_end():
    state = _state(game_over=True, draw=True)
    assert _update_agent_routing(state) == "end"


def test_update_agent_with_winner_routes_to_end():
    from checkers.engine.board import RED
    state = _state(game_over=True, winner=RED)
    assert _update_agent_routing(state) == "end"


# ── Graph compilation smoke tests ─────────────────────────────────────────────

def test_build_graph_compiles():
    from checkers.graph.graph import build_graph
    g = build_graph()
    assert g is not None


def test_graph_has_exactly_the_simplified_nodes():
    from checkers.graph.graph import build_graph
    g = build_graph()
    required = {"scorer_node", "deterministic_proposal_node", "ranker_agent", "update_agent"}
    for node in required:
        assert node in g.nodes, f"Node '{node}' missing from compiled graph"


def test_old_pipeline_nodes_not_in_graph():
    from checkers.graph.graph import build_graph
    g = build_graph()
    old_nodes = {
        "orchestrator", "inter_turn_memory", "symbolic_decision",
        "proposal_agent", "format_checker", "validator", "minimax_scorer",
        "state_manager", "win_condition", "logger_node",
    }
    present = old_nodes & set(g.nodes)
    assert not present, f"Old pipeline nodes still registered: {present}"


def test_graph_module_exports_update_agent_routing():
    import checkers.graph.graph as gg
    assert hasattr(gg, "_update_agent_routing")
    assert hasattr(gg, "build_graph")
    assert hasattr(gg, "checkers_graph")
