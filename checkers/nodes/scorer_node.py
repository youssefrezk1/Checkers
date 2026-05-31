# checkers/nodes/scorer_node.py
#
# Simplified pipeline node — replaces symbolic_decision + minimax_scorer.
#
# Calls score_all_legal_moves once and writes:
#
#   score_state              — whole-position balance ("CLEARLY_WINNING" …
#                              "CLEARLY_LOSING"), computed by compute_score_state
#                              from board facts only; no history required.
#                              Used by ranker_agent for adversity-seed gating.
#
#   symbolic_scored_moves    — all legal moves scored + ranked:
#                              [{"move": slim_move, "minimax_score": float, "rank": int}]
#                              "move" contains only {"type", "path", "captured"}.
#                              Sorted best-first (rank 1 = best).
#
#   legal_moves              — all enriched moves sorted best-first by minimax_score.
#                              Format: {"type", "path", "captured", "facts"}.
#
#   symbolic_best_move       — slim move dict, no facts
#   symbolic_best_score      — float, rounded to 2 dp
#   symbolic_second_best_score — float | None, rounded to 2 dp
#   symbolic_gap             — float, rounded to 2 dp

from __future__ import annotations

import math

from checkers.state.state import CheckersState
from checkers.agents.scorer_agent import score_all_legal_moves, compute_score_state
from checkers.engine.evaluation import LOSS_SCORE


def scorer_node(state: CheckersState) -> dict:
    # ── Score all legal moves ──────────────────────────────────────────────────
    enriched, best_score, second_best_score, gap = score_all_legal_moves(
        state.board,
        state.current_player,
        state.position_history,
    )

    # ── Score-state: whole-position balance classification ─────────────────────
    score_state = compute_score_state(state.board, state.current_player)

    # ── Build symbolic_scored_moves ────────────────────────────────────────────
    symbolic_scored_moves = [
        {
            "move": {
                "type": m["type"],
                "path": m["path"],
                "captured": m.get("captured", []),
            },
            "minimax_score": m["facts"]["minimax_score"],
            "rank": m["facts"]["symbolic_rank"],
        }
        for m in enriched
    ]

    # ── symbolic_best_move: slim (no facts) ───────────────────────────────────
    best_move = None
    if enriched:
        m0 = enriched[0]
        best_move = {
            "type": m0["type"],
            "path": m0["path"],
            "captured": m0.get("captured", []),
        }

    r_best = round(best_score, 2)
    r_second = round(second_best_score, 2) if second_best_score is not None else None

    if math.isinf(gap):
        r_gap = round(r_best - float(LOSS_SCORE), 2)
    else:
        r_gap = round(gap, 2)

    return {
        "score_state": score_state,
        "symbolic_scored_moves": symbolic_scored_moves,
        "legal_moves": enriched,
        "symbolic_best_score": r_best,
        "symbolic_second_best_score": r_second,
        "symbolic_gap": r_gap,
        "symbolic_best_move": best_move,
        "last_completed_node": "scorer_node",
    }
