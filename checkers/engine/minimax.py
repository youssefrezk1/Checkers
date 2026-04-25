# engine/minimax.py
#
# Compatibility wrapper around isolated Phase-1 search core.
#
# LEGACY NOTE: score_move_with_minimax is retained for backward compatibility
# (tests, one-off ablations) but must NOT be used in the live pipeline for
# ranking candidates.  It scores each move in isolation (no shared TT, no
# sibling context) so the resulting scores are not directly comparable across
# moves and produce an incorrect ordering.
#
# The canonical multi-move scorer is:
#   checkers.search.minimax_core.search_root_all_scores

from __future__ import annotations

import os

from checkers.engine.board import BLACK, RED
from checkers.engine.rules import apply_move
from checkers.search.minimax_core import SearchStats, negamax

# ── Configuration ─────────────────────────────────────────────────────────────
# Default changed from 3 to 6 to match the project's production search depth.
MINIMAX_DEPTH = int(os.environ.get("MINIMAX_DEPTH", "6"))


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


def minimax_score(
    board: list[list[int]],
    depth: int,
    current_player: int,
    root_player: int,
    alpha: float,
    beta: float,
) -> float:
    """
    Backward-compatible interface that delegates to negamax alpha-beta.
    """
    return float(
        negamax(
            board=board,
            depth=depth,
            current_player=current_player,
            root_player=root_player,
            alpha=alpha,
            beta=beta,
            stats=SearchStats(),
        )
    )


def score_move_with_minimax(
    board: list[list[int]],
    move: dict,
    current_player: int,
    depth: int = MINIMAX_DEPTH,
) -> float:
    """
    **LEGACY / DEBUG ONLY.**

    Score a single candidate move using fixed-depth negamax alpha-beta.

    Do NOT use this function to rank multiple moves against each other in the
    live pipeline.  Each call launches an independent search with no shared
    transposition table, so scores from different calls are not on a common
    scale and will produce a different (incorrect) ordering compared to the
    joint search performed by search_root_all_scores.

    Use search_root_all_scores for any ranking or scoring that affects move
    selection.
    """
    board_after = apply_move(board, move)
    return float(
        negamax(
            board=board_after,
            depth=max(0, depth - 1),
            current_player=_opponent(current_player),
            root_player=current_player,
            alpha=float("-inf"),
            beta=float("inf"),
            stats=SearchStats(),
            use_tt=False,  # Isolate per-candidate scoring: no cross-call TT contamination.
        )
    )