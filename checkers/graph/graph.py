from __future__ import annotations
import os
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

#import simplified pipeline nodes
from checkers.nodes.scorer_node import scorer_node
from checkers.nodes.deterministic_proposal_node import deterministic_proposal_node
from checkers.agents.update_agent import update_agent


def _orchestrator_routing(state: CheckersState) -> str:
    """
    All routing decisions for the hub-and-spoke pipeline.
    Extracted to module level so it can be unit-tested independently.

    Simplified pipeline (USE_SIMPLIFIED_PIPELINE=true):
      scorer_node → deterministic_proposal_node → ranker_agent → update_agent
      update_agent computes next-turn strategic_context internally (Phase D),
      so inter_turn_memory is NOT called at turn start in this mode.
      scorer_node handles a missing strategic_context on the first turn.

    Old pipeline (flag absent or false):
      inter_turn_memory → symbolic_decision → proposal_agent → format_checker
      → validator → minimax_scorer → ranker_agent → state_manager
      → win_condition → logger_node → inter_turn_memory (loop)
    """
    _simplified = os.environ.get("USE_SIMPLIFIED_PIPELINE", "false").lower() == "true"
    node = state.last_completed_node

    # Turn start
    if node is None or node == "orchestrator":
        # Simplified: skip inter_turn_memory — scorer_node handles first-turn context.
        # Old pipeline: inter_turn_memory prepares context before symbolic_decision.
        return "scorer_node" if _simplified else "inter_turn_memory"

    if node == "inter_turn_memory":
        # Only reachable from the old pipeline's logger_node loop.
        return "symbolic_decision"

    # ── Simplified pipeline branch ───────────────────────────────────────────
    if node == "scorer_node":
        return "deterministic_proposal_node"

    if node == "deterministic_proposal_node":
        return "ranker_agent"

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
        # Simplified pipeline: update_agent handles the full end-of-turn sequence.
        if _simplified:
            return "update_agent"
        # Old pipeline: route based on whether a move was chosen.
        if state.chosen_move is not None:
            return "state_manager"
        # chosen_move is None only when ranker_agent received empty legal_moves
        # (terminal position: no legal moves = loss). Route to win_condition.
        return "win_condition"

    # ── Simplified pipeline: update_agent exit paths ─────────────────────────
    if node == "update_agent":
        if state.game_over:
            return "end"
        # Default (single-turn mode): stop after one completed move so that
        # run_full_trace.py / external callers can advance the game one ply at
        # a time without hitting LangGraph recursion limits.
        # Full-game autonomous loop: set AUTO_PLAY_UNTIL_GAME_OVER=true.
        _auto_play = (
            os.environ.get("AUTO_PLAY_UNTIL_GAME_OVER", "false").lower() == "true"
        )
        if _auto_play:
            return "scorer_node"
        return "end"

    # ── Old pipeline: post-ranker sequence ───────────────────────────────────
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
    # Simplified pipeline nodes
    graph.add_node("scorer_node", scorer_node)
    graph.add_node("deterministic_proposal_node", deterministic_proposal_node)
    graph.add_node("update_agent", update_agent)

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
    graph.add_edge("scorer_node", "orchestrator")
    graph.add_edge("deterministic_proposal_node", "orchestrator")
    graph.add_edge("update_agent", "orchestrator")

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
            "scorer_node": "scorer_node",
            "deterministic_proposal_node": "deterministic_proposal_node",
            "update_agent": "update_agent",
            "end": END,
        }
    )

    return graph.compile()

checkers_graph = build_graph()
