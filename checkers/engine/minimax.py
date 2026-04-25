# engine/minimax.py
#
# Compatibility wrapper around isolated Phase-1 search core.

from __future__ import annotations

import os

from checkers.engine.board import BLACK, RED
from checkers.engine.rules import apply_move
from checkers.search.minimax_core import SearchStats, negamax

# ── Configuration ─────────────────────────────────────────────────────────────
MINIMAX_DEPTH = int(os.environ.get("MINIMAX_DEPTH", "3"))


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
    Score a single candidate move using fixed-depth negamax alpha-beta.
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