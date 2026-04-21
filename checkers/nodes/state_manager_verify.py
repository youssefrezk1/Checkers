# Defense-in-depth checks before/after apply_move in state_manager.
# Validator + ranker already gate legality; this catches stale state, bugs, or drift.

from __future__ import annotations

from typing import Any

from checkers.state.state import CheckersState
from checkers.engine.board import EMPTY, BOARD_SIZE, in_bounds
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.move_facts import compute_move_facts
from checkers.nodes.validator import _moves_match

# Scalar facts cheap to recompute and compare (avoid dict identity / ordering noise).
_FACT_SCALAR_KEYS = (
    "move_type",
    "path_length",
    "captures_count",
    "jump_count",
    "is_multi_jump",
    "net_gain",
    "results_in_king",
    "kings_captured",
    "regulars_captured",
)


def _slim_move(chosen: dict[str, Any]) -> dict[str, Any]:
    """Engine-shaped move only (no facts); copies path/captured so history is immutable."""
    return {
        "type": chosen["type"],
        "path": [list(sq) for sq in chosen["path"]],
        "captured": [list(sq) for sq in chosen.get("captured", [])],
    }


def verify_chosen_move_for_state(state: CheckersState) -> None:
    """
    Raises ValueError if chosen_move is missing, not engine-legal on this board,
    or embedded facts disagree with a fresh compute_move_facts (when facts present).
    """
    chosen = state.chosen_move
    if chosen is None:
        raise ValueError("state_manager: chosen_move is None")

    for key in ("type", "path"):
        if key not in chosen:
            raise ValueError(f"state_manager: chosen_move missing required key {key!r}")

    board = state.board
    player = state.current_player
    slim = _slim_move(chosen)

    legal = get_all_legal_moves(board, player)
    if not any(_moves_match(slim, m) for m in legal):
        raise ValueError(
            "state_manager: chosen_move does not match any engine-legal move "
            f"for current_player={player} on this board (path={slim.get('path')!r})."
        )

    existing = chosen.get("facts")
    if isinstance(existing, dict) and existing:
        fresh = compute_move_facts(board, slim, player)
        for k in _FACT_SCALAR_KEYS:
            if k not in existing:
                continue
            if k not in fresh:
                continue
            if existing[k] != fresh[k]:
                raise ValueError(
                    f"state_manager: stale or inconsistent move facts for key {k!r}: "
                    f"stored={existing[k]!r} recomputed={fresh[k]!r}"
                )


def verify_board_after_move(
    board_before: list[list[int]],
    board_after: list[list[int]],
    move: dict[str, Any],
) -> None:
    """Light invariants: every path square in-bounds; captured cells empty; source empty; dest occupied."""
    path = move["path"]
    for sq in path:
        r, c = int(sq[0]), int(sq[1])
        if not in_bounds(r, c):
            raise ValueError(
                f"state_manager: path square [{r},{c}] is out of bounds"
            )

    from_row, from_col = path[0][0], path[0][1]
    to_row, to_col = path[-1][0], path[-1][1]

    for cap in move.get("captured", []):
        cr, cc = int(cap[0]), int(cap[1])
        if board_after[cr][cc] != EMPTY:
            raise ValueError(
                f"state_manager: post-apply invariant failed — square {cr},{cc} should be "
                f"EMPTY after capture, got {board_after[cr][cc]}"
            )

    if board_after[from_row][from_col] != EMPTY:
        raise ValueError(
            "state_manager: post-apply invariant failed — origin square should be EMPTY"
        )

    if board_after[to_row][to_col] == EMPTY:
        raise ValueError(
            "state_manager: post-apply invariant failed — landing square should not be EMPTY"
        )

    # Board dimensions
    if len(board_after) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board_after):
        raise ValueError("state_manager: board_after has wrong shape")
