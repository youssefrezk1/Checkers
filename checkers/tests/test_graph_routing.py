"""
Routing unit tests for orchestrator_routing.

_orchestrator_routing is extracted to module level in graph.py specifically to
make these assertions cheap — no LangGraph invocation required.
"""
from __future__ import annotations

from checkers.graph.graph import _orchestrator_routing
from checkers.state.state import CheckersState


def _state(**kwargs) -> CheckersState:
    return CheckersState(**kwargs)


# ── ranker_agent exit paths ───────────────────────────────────────────────────

def test_ranker_agent_with_chosen_move_routes_to_state_manager():
    state = _state(
        last_completed_node="ranker_agent",
        chosen_move={"type": "simple", "path": [(5, 0), (4, 1)], "captured": []},
    )
    assert _orchestrator_routing(state) == "state_manager"


def test_ranker_agent_with_no_chosen_move_routes_to_win_condition():
    # ranker_agent returns chosen_move=None only when legal_moves was empty
    # (terminal position). Must route to win_condition, NOT back to ranker_agent.
    state = _state(
        last_completed_node="ranker_agent",
        chosen_move=None,
    )
    assert _orchestrator_routing(state) == "win_condition"


# ── Sanity checks for surrounding routing ────────────────────────────────────

def test_minimax_scorer_routes_to_ranker_agent():
    state = _state(last_completed_node="minimax_scorer")
    assert _orchestrator_routing(state) == "ranker_agent"


def test_state_manager_routes_to_win_condition():
    state = _state(last_completed_node="state_manager")
    assert _orchestrator_routing(state) == "win_condition"


def test_win_condition_routes_to_logger_node():
    state = _state(last_completed_node="win_condition")
    assert _orchestrator_routing(state) == "logger_node"


def test_logger_node_game_over_routes_to_end():
    state = _state(last_completed_node="logger_node", game_over=True)
    assert _orchestrator_routing(state) == "end"


def test_logger_node_not_over_routes_to_inter_turn_memory():
    state = _state(last_completed_node="logger_node", game_over=False)
    assert _orchestrator_routing(state) == "inter_turn_memory"
