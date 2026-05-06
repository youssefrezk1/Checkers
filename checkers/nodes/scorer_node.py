# checkers/nodes/scorer_node.py
#
# Simplified pipeline node — replaces symbolic_decision + minimax_scorer.
# Only active when USE_SIMPLIFIED_PIPELINE=true; the old nodes are untouched.
#
# Calls score_all_legal_moves once and writes:
#
#   symbolic_scored_moves    — all legal moves scored + ranked, in the same
#                              backward-compatible format as symbolic_decision:
#                              [{"move": slim_move, "minimax_score": float, "rank": int}]
#                              "move" contains only {"type", "path", "captured"} (no facts).
#                              Sorted best-first (rank 1 = best).
#
#   legal_moves              — all enriched moves sorted best-first by minimax_score.
#                              Format: {"type", "path", "captured", "facts"}.
#                              facts contains minimax_score, symbolic_rank, and all
#                              compute_move_facts fields.  Ranker-agent-compatible.
#
#   symbolic_best_move       — slim move dict ({"type", "path", "captured"}, no facts)
#   symbolic_best_score      — float, rounded to 2 dp (matches symbolic_decision)
#   symbolic_second_best_score — float | None, rounded to 2 dp
#   symbolic_gap             — float, rounded to 2 dp;
#                              when only 1 legal move: round(best - LOSS_SCORE, 2)
#                              instead of inf (matches symbolic_decision behaviour)
#
#   strategic_context        — injected ONLY on the very first turn, when
#                              update_agent has not yet produced a real context.
#                              All values are neutral/empty defaults; no engine
#                              calls are made.
#
# update_agent clears all symbolic fields between turns (no cleanup needed here).

from __future__ import annotations

import math

from checkers.state.state import CheckersState
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.engine.evaluation import LOSS_SCORE

# Minimal neutral context used on the very first turn of the simplified pipeline
# (before update_agent has produced a real one via Phase D).
# Keeps all downstream nodes (deterministic_proposal, ranker_agent) safe when
# they do `ctx or {}` reads on state.strategic_context.
_FIRST_TURN_CONTEXT: dict = {
    "game_phase": "OPENING",
    "score_state": "EQUAL",
    "winning_score": 0,
    "material_advantage": 0,
    "king_advantage": 0,
    "mobility_advantage": 0,
    "center_control_advantage": 0,
    "our_promotion_threats": 0,
    "opp_promotion_threats": 0,
    "our_vulnerable_pieces": 0,
    "opp_vulnerable_pieces": 0,
    "our_back_row_count": 0,
    "position_is_stable": False,
    "stagnation_detected": False,
    "material_trend": None,
    "mobility_trend": None,
    "center_trend": None,
    "active_patterns": [],
    "strategic_priorities": [],
    "turn_history": [],
    "archive_summary": [],
}


def scorer_node(state: CheckersState) -> dict:
    # ── First-turn context guard ───────────────────────────────────────────────
    # In the simplified pipeline, inter_turn_memory no longer runs before
    # scorer_node. On the very first turn strategic_context is None.
    # Inject a neutral default so downstream nodes always see a valid dict.
    # On every subsequent turn update_agent's Phase D provides a real context,
    # so this branch is never entered again.
    ctx_patch: dict = {}
    if not state.strategic_context:
        ctx_patch["strategic_context"] = dict(_FIRST_TURN_CONTEXT)

    # ── Score all legal moves ──────────────────────────────────────────────────
    enriched, best_score, second_best_score, gap = score_all_legal_moves(
        state.board,
        state.current_player,
        state.position_history,
    )

    # ── Build symbolic_scored_moves in symbolic_decision-compatible format ─────
    # Each entry: {"move": slim_move, "minimax_score": float, "rank": int}
    # "move" has only {"type", "path", "captured"} — no facts — matching
    # the format written by symbolic_decision (which sources moves from
    # get_all_legal_moves, plain move dicts without facts).
    # minimax_score is already rounded to 2 dp inside scorer_agent.
    symbolic_scored_moves = [
        {
            "move": {
                "type": m["type"],
                "path": m["path"],
                "captured": m.get("captured", []),
            },
            "minimax_score": m["facts"]["minimax_score"],  # already round(score, 2)
            "rank": m["facts"]["symbolic_rank"],            # 1-based, 1=best
        }
        for m in enriched
    ]

    # ── symbolic_best_move: slim (no facts), matches symbolic_decision ─────────
    best_move = None
    if enriched:
        m0 = enriched[0]
        best_move = {
            "type": m0["type"],
            "path": m0["path"],
            "captured": m0.get("captured", []),
        }

    # ── Round summary scores to 2 dp — match symbolic_decision exactly ─────────
    # scorer_agent already rounds facts["minimax_score"]; best_score /
    # second_best_score are read from those rounded values, so this is a no-op
    # in normal cases but makes the contract explicit.
    r_best = round(best_score, 2)
    r_second = round(second_best_score, 2) if second_best_score is not None else None

    # Gap: symbolic_decision converts float("inf") (single-move positions) to a
    # finite value. Mirror that exactly.
    if math.isinf(gap):
        r_gap = round(r_best - float(LOSS_SCORE), 2)
    else:
        r_gap = round(gap, 2)

    return {
        **ctx_patch,
        "symbolic_scored_moves": symbolic_scored_moves,
        "legal_moves": enriched,
        "symbolic_best_score": r_best,
        "symbolic_second_best_score": r_second,
        "symbolic_gap": r_gap,
        "symbolic_best_move": best_move,
        "last_completed_node": "scorer_node",
    }
