# nodes/symbolic_decision.py
#
# Phase 8 — Symbolic-assisted decision stage.
# Runs AFTER inter_turn_memory and BEFORE proposal_agent. ALWAYS leads to proposal_agent.
#
# Responsibilities:
#   1. Score all legal moves at SYMBOLIC_DECISION_DEPTH (== MINIMAX_DEPTH, env-overridable)
#      (use_tt=False for TT isolation, use_phase7a=True, use_tactical_extension=True)
#   2. Sort moves by score descending (best first)
#   3. Store sorted+scored list in state.symbolic_scored_moves
#   4. Store instrumentation fields (best_score, gap) for thesis evaluation
#
# What this node does NOT do:
#   - Does NOT set chosen_move
#   - Does NOT bypass proposal or ranker
#   - Does NOT set symbolic_bypass
#
# Proposal agent reads symbolic_scored_moves as its candidate pool (pre-sorted).
# Ranker agent remains the final decision-maker.
#
# Configuration (ENV-overridable):
#   SYMBOLIC_DECISION_DEPTH    default 3

from __future__ import annotations

import logging
import os
from typing import Any

from checkers.state.state import CheckersState
from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.evaluation import LOSS_SCORE
from checkers.engine.zobrist import compute_hash
from checkers.search.minimax_core import (
    negamax,
    SearchStats,
    search_root_all_scores,
    clear_transposition_table,
)
from checkers.engine.minimax import MINIMAX_DEPTH

# Penalty subtracted from a move's score for each time its resulting position
# already appears in position_history. 60 pts = just above the typical positional
# noise spread (~40-50 pts) so it reliably breaks king-shuffle ties without
# overriding genuine tactical differences measured in hundreds of points.
REPETITION_PENALTY = 60

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Depth is sourced from MINIMAX_DEPTH (env: MINIMAX_DEPTH, default 3) so that
# symbolic_decision, minimax_scorer (cache), and the diagnostic all score at
# the same depth — eliminating score inconsistency across pipeline components.
SYMBOLIC_DECISION_DEPTH = MINIMAX_DEPTH

_VALID_BACKENDS = {"per_move", "search_root_all_scores"}
SYMBOLIC_SCORING_BACKEND = os.environ.get("SYMBOLIC_SCORING_BACKEND", "search_root_all_scores")
if SYMBOLIC_SCORING_BACKEND not in _VALID_BACKENDS:
    raise ValueError(
        f"SYMBOLIC_SCORING_BACKEND={SYMBOLIC_SCORING_BACKEND!r} is not valid. "
        f"Allowed values: {sorted(_VALID_BACKENDS)}"
    )


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


def _score_all_moves(
    board: list[list[int]],
    legal: list[dict[str, Any]],
    player: int,
    depth: int,
    position_history: list[int] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """
    Score every legal move with full-config negamax at the given depth.
    Returns list of (move, score) sorted descending by score (best first).

    Repetition penalty: if the resulting board hash already appears N times in
    position_history, subtract REPETITION_PENALTY * N from the raw minimax score.
    This breaks king-shuffle ties without touching the search tree itself.
    """
    history_set: dict[int, int] = {}
    if position_history:
        for h in position_history:
            history_set[h] = history_set.get(h, 0) + 1

    scored: list[tuple[dict[str, Any], float]] = []
    for move in legal:
        try:
            child = apply_move(board, move)
            score = float(
                negamax(
                    board=child,
                    depth=max(0, depth - 1),
                    current_player=_opponent(player),
                    root_player=player,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    stats=SearchStats(),
                    use_tt=False,  # Isolate: do not contaminate _TT before diagnostic runs.
                    extension_depth=0,
                    use_tactical_extension=True,
                    use_phase7a=True,
                )
            )
            # Apply repetition penalty: count how many times this child board
            # has already appeared in the game's position history.
            if history_set:
                child_hash = compute_hash(child)
                repeat_count = history_set.get(child_hash, 0)
                if repeat_count > 0:
                    score -= REPETITION_PENALTY * repeat_count
                    logger.debug(
                        "[symbolic_decision] repetition_penalty path=%s count=%d penalty=%.0f",
                        move.get("path"), repeat_count, REPETITION_PENALTY * repeat_count,
                    )
        except Exception as exc:
            logger.warning("[symbolic_decision] scoring failed for %s: %s", move.get("path"), exc)
            score = float(LOSS_SCORE)
        scored.append((move, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _score_all_moves_search_root(
    board: list[list[int]],
    legal: list[dict[str, Any]],
    player: int,
    depth: int,
    position_history: list[int] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """
    Score every legal move using search_root_all_scores with TT sharing.

    Same contract as _score_all_moves: returns list of (move, score) sorted
    descending by score, with repetition penalty applied.
    """
    clear_transposition_table()
    _, _, raw_scored, _ = search_root_all_scores(
        board=board,
        current_player=player,
        depth=depth,
        legal_moves=legal,
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )

    history_set: dict[int, int] = {}
    if position_history:
        for h in position_history:
            history_set[h] = history_set.get(h, 0) + 1

    scored: list[tuple[dict[str, Any], float]] = []
    for move, score in raw_scored:
        if history_set:
            child = apply_move(board, move)
            child_hash = compute_hash(child)
            repeat_count = history_set.get(child_hash, 0)
            if repeat_count > 0:
                score -= REPETITION_PENALTY * repeat_count
                logger.debug(
                    "[symbolic_decision] repetition_penalty path=%s count=%d penalty=%.0f",
                    move.get("path"), repeat_count, REPETITION_PENALTY * repeat_count,
                )
        scored.append((move, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def symbolic_decision(state: CheckersState) -> dict[str, Any]:
    """
    Phase 8 symbolic-assisted decision stage.

    Scores all legal moves at depth=SYMBOLIC_DECISION_DEPTH with full engine config, sorts them
    best-first, and stores the result in state.symbolic_scored_moves.
    Proposal agent receives this sorted list as its candidate pool.
    Ranker agent remains the final decision-maker.

    Always routes to proposal_agent — never bypasses LLM pipeline.
    """
    board = state.board
    player = state.current_player
    legal = get_all_legal_moves(board, player)

    # ── No legal moves: terminal — let win_condition handle it ───────────────
    if not legal:
        logger.info("[symbolic_decision] no legal moves — terminal position")
        return {
            "symbolic_scored_moves": [],
            "symbolic_best_move": None,
            "symbolic_best_score": float(LOSS_SCORE),
            "symbolic_second_best_score": None,
            "symbolic_gap": 0.0,
            "last_completed_node": "symbolic_decision",
        }

    # ── Score and sort all legal moves ────────────────────────────────────────
    position_history: list[int] = getattr(state, "position_history", None) or []

    if SYMBOLIC_SCORING_BACKEND == "search_root_all_scores":
        scored = _score_all_moves_search_root(board, legal, player, SYMBOLIC_DECISION_DEPTH, position_history)
        logger.info("[symbolic_decision] backend=search_root_all_scores depth=%d", SYMBOLIC_DECISION_DEPTH)
    else:
        scored = _score_all_moves(board, legal, player, SYMBOLIC_DECISION_DEPTH, position_history)
        logger.info("[symbolic_decision] backend=per_move depth=%d", SYMBOLIC_DECISION_DEPTH)

    best_move, best_score = scored[0]
    second_best_score: float | None = scored[1][1] if len(scored) > 1 else None
    gap = (best_score - second_best_score) if second_best_score is not None else float("inf")

    logger.info(
        "[symbolic_decision] depth=%d best_score=%.2f gap=%.2f n_legal=%d",
        SYMBOLIC_DECISION_DEPTH, best_score, gap, len(legal),
    )

    # ── Build symbolic_scored_moves: sorted list with score and rank ─────────
    symbolic_scored_moves: list[dict[str, Any]] = []
    for rank, (move, score) in enumerate(scored, start=1):
        symbolic_scored_moves.append({
            "move": move,
            "minimax_score": round(score, 2),
            "rank": rank,                      # 1-based; rank=1 is the best move
        })

    return {
        "symbolic_scored_moves": symbolic_scored_moves,
        "symbolic_best_move": best_move,
        "symbolic_best_score": round(best_score, 2),
        "symbolic_second_best_score": round(second_best_score, 2) if second_best_score is not None else None,
        "symbolic_gap": round(gap, 2) if gap != float("inf") else round(best_score - float(LOSS_SCORE), 2),
        # Thesis instrumentation: LLM path is always taken
        "llm_invoked": True,
        "last_completed_node": "symbolic_decision",
    }
