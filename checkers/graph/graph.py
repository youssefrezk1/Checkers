from __future__ import annotations
from langgraph.graph import StateGraph, END
from checkers.state.state import CheckersState

#import nodes
from checkers.nodes.orchestrator import orchestrator
from checkers.nodes.inter_turn_memory import inter_turn_memory
from checkers.nodes.format_checker import format_checker
from checkers.nodes.validator import validator
from checkers.nodes.state_manager import state_manager
from checkers.nodes.win_condition import win_condition
from checkers.nodes.logger_node import logger_node
from checkers.nodes.minimax_scorer import minimax_scorer
from checkers.nodes.symbolic_decision import symbolic_decision

#import agents
from checkers.agents.proposal_agent import proposal_agent
from checkers.agents.ranker_agent import ranker_agent


def _orchestrator_routing(state: CheckersState) -> str:
    """
    All routing decisions for the hub-and-spoke pipeline.
    Extracted to module level so it can be unit-tested independently.
    """
    node = state.last_completed_node

    # Turn start
    if node is None or node == "orchestrator":
        return "inter_turn_memory"

    if node == "inter_turn_memory":
        return "symbolic_decision"

    if node == "symbolic_decision":
        # Always proceed to proposal — symbolic engine is support, not decision-maker.
        return "proposal_agent"

    if node == "proposal_agent":
        return "format_checker"

    # Loop 1: format checker → back to proposal if all removed
    if node == "format_checker":
        if len(state.proposed_moves) == 0:
            if state.retry_count >= state.retry_budget:
                return "end"
            return "proposal_agent"
        return "validator"

    # Loop 2: validator → back to proposal if all illegal
    if node == "validator":
        if len(state.legal_moves) == 0:
            if state.retry_count >= state.retry_budget:
                return "end"
            return "proposal_agent"
        return "minimax_scorer"

    if node == "minimax_scorer":
        return "ranker_agent"

    if node == "ranker_agent":
        if state.chosen_move is not None:
            return "state_manager"
        # chosen_move is None only when ranker_agent received empty legal_moves
        # (terminal position: no legal moves = loss). Route to win_condition.
        return "win_condition"

    if node == "state_manager":
        return "win_condition"

    if node == "win_condition":
        return "logger_node"

    # Game over check only after logger
    if node == "logger_node":
        if state.game_over:
            return "end"
        return "inter_turn_memory"

    return "inter_turn_memory"


def build_graph():
    graph = StateGraph(CheckersState)

    # ── Register all nodes ───────────────────────────────
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("inter_turn_memory", inter_turn_memory)
    graph.add_node("proposal_agent", proposal_agent)
    graph.add_node("format_checker", format_checker)
    graph.add_node("validator", validator)
    graph.add_node("ranker_agent", ranker_agent)
    graph.add_node("state_manager", state_manager)
    graph.add_node("win_condition", win_condition)
    graph.add_node("logger_node", logger_node)
    graph.add_node("minimax_scorer", minimax_scorer)
    graph.add_node("symbolic_decision", symbolic_decision)

    # ── Entry point ──────────────────────────────────────
    graph.set_entry_point("orchestrator")

    # ── Every node returns to Orchestrator ───────────────
    graph.add_edge("inter_turn_memory", "orchestrator")
    graph.add_edge("proposal_agent", "orchestrator")
    graph.add_edge("format_checker", "orchestrator")
    graph.add_edge("validator", "orchestrator")
    graph.add_edge("ranker_agent", "orchestrator")
    graph.add_edge("state_manager", "orchestrator")
    graph.add_edge("win_condition", "orchestrator")
    graph.add_edge("logger_node", "orchestrator")
    graph.add_edge("minimax_scorer", "orchestrator")
    graph.add_edge("symbolic_decision", "orchestrator")

    # ── Orchestrator conditional edge ────────────────────
    graph.add_conditional_edges(
        "orchestrator",
        _orchestrator_routing,
        {
            "inter_turn_memory": "inter_turn_memory",
            "symbolic_decision": "symbolic_decision",
            "proposal_agent": "proposal_agent",
            "format_checker": "format_checker",
            "validator": "validator",
            "minimax_scorer": "minimax_scorer",
            "ranker_agent": "ranker_agent",
            "state_manager": "state_manager",
            "win_condition": "win_condition",
            "logger_node": "logger_node",
            "end": END,
        }
    )

    return graph.compile()

checkers_graph = build_graph()
