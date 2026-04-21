from __future__ import annotations
from langgraph.graph import StateGraph, END
from checkers.state.state import CheckersState

#import nodes 
from checkers.nodes.orchestrator import orchestrator
from checkers.nodes.inter_turn_memory import inter_turn_memory
from checkers.nodes.format_checker import format_checker
from checkers.nodes.validator import validator
from checkers.nodes.state_manager import state_manager
from checkers.nodes.ranker_fallback import ranker_fallback
from checkers.nodes.win_condition import win_condition
from checkers.nodes.logger_node import logger_node
from checkers.nodes.minimax_scorer import minimax_scorer

#import agents
from checkers.agents.proposal_agent import proposal_agent
from checkers.agents.ranker_agent import ranker_agent


def build_graph():
    graph = StateGraph(CheckersState)

    # ── Register all nodes ───────────────────────────────
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("inter_turn_memory", inter_turn_memory)
    graph.add_node("proposal_agent", proposal_agent)
    graph.add_node("format_checker", format_checker)
    graph.add_node("validator", validator)
    graph.add_node("ranker_agent", ranker_agent)
    graph.add_node("ranker_fallback", ranker_fallback)
    graph.add_node("state_manager", state_manager)
    graph.add_node("win_condition", win_condition)
    graph.add_node("logger_node", logger_node)
    graph.add_node("minimax_scorer", minimax_scorer)


    # ── Entry point ──────────────────────────────────────
    graph.set_entry_point("orchestrator")

    # ── Every node returns to Orchestrator ───────────────
    graph.add_edge("inter_turn_memory", "orchestrator")
    graph.add_edge("proposal_agent", "orchestrator")
    graph.add_edge("format_checker", "orchestrator")
    graph.add_edge("validator", "orchestrator")
    graph.add_edge("ranker_agent", "orchestrator")
    graph.add_edge("ranker_fallback", "orchestrator")
    graph.add_edge("state_manager", "orchestrator")
    graph.add_edge("win_condition", "orchestrator")
    graph.add_edge("logger_node", "orchestrator")
    graph.add_edge("minimax_scorer", "orchestrator")


    # ── Orchestrator conditional edge ────────────────────
    # This single function handles ALL routing decisions
   
    def orchestrator_routing(state: CheckersState) -> str:
        node = state.last_completed_node

        # Turn start
        if node is None or node == "orchestrator":
            return "inter_turn_memory"

        if node == "inter_turn_memory":
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
            return "ranker_agent"   # keep retrying forever until LLM succeeds

        if node == "ranker_fallback":
            if state.chosen_move is None:
                # No legal moves (should be rare after validator); regenerate proposals.
                return "proposal_agent"
            return "state_manager"

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

    graph.add_conditional_edges(
        "orchestrator",
        orchestrator_routing,
        {
            "inter_turn_memory": "inter_turn_memory",
            "proposal_agent": "proposal_agent",
            "format_checker": "format_checker",
            "validator": "validator",
            "minimax_scorer": "minimax_scorer",
            "ranker_agent": "ranker_agent",
            "ranker_fallback": "ranker_fallback",
            "state_manager": "state_manager",
            "win_condition": "win_condition",
            "logger_node": "logger_node",
            "end": END,
        }
    )

    return graph.compile()

checkers_graph = build_graph()
