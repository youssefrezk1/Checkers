# checkers/nodes/deterministic_proposal_node.py
#
# Simplified pipeline node — replaces proposal_agent + format_checker + validator.
#
# SOLE MOVE-SELECTION AUTHORITY for the simplified proposal-authoritative
# pipeline. Reads state.legal_moves (all enriched moves from scorer_node,
# sorted best-first) and deterministically selects the SINGLE BEST move by
# minimax ranking. Once written here, chosen_move is treated as immutable by
# every downstream node (ranker_agent, update_agent, state_manager).
#
# Outputs:
#   chosen_move         — the best move dict (full facts preserved)
#   chosen_move_score   — minimax_score of the chosen move
#   unchosen_moves      — all other legal moves (full facts preserved),
#                         supplied to ranker_agent only as comparative
#                         context for explanation generation
#   legal_moves         — PRESERVED unchanged (no destructive overwrite)
#   proposal_diagnostics — selection metadata
#
# Never raises: select_best_move is fully deterministic and handles
# edge cases (empty input) without retry logic.

from __future__ import annotations

from checkers.state.state import CheckersState
from checkers.agents.deterministic_proposal import select_best_move


def deterministic_proposal_node(state: CheckersState) -> dict:
    chosen, score, unchosen, meta = select_best_move(
        state.legal_moves,
        score_state=state.score_state,
    )

    return {
        "chosen_move": chosen,
        "chosen_move_score": score,
        "unchosen_moves": unchosen,
        "legal_moves": state.legal_moves,       # PRESERVE full list — no destructive overwrite
        "proposal_diagnostics": meta,
        "last_completed_node": "deterministic_proposal_node",
    }

