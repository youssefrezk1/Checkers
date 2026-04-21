# engine/minimax.py
#
# Stronger minimax backend for scoring candidate moves.
# Architecture remains unchanged: this module only returns numeric scores.

from __future__ import annotations

import os
from typing import Any

from checkers.engine.board import BLACK, RED, BLACK_KING, RED_KING
from checkers.engine.evaluation import LOSS_SCORE, WIN_SCORE, evaluate_board
from checkers.engine.rules import apply_move, get_all_legal_moves

# ── Configuration ─────────────────────────────────────────────────────────────
MINIMAX_DEPTH = int(os.environ.get("MINIMAX_DEPTH", "2"))
QUIESCENCE_MAX_DEPTH = int(os.environ.get("QUIESCENCE_MAX_DEPTH", "8"))
MINIMAX_USE_TT = os.environ.get("MINIMAX_USE_TT", "true").lower() in ("1", "true", "yes", "on")
MINIMAX_TT_MAX_ENTRIES = int(os.environ.get("MINIMAX_TT_MAX_ENTRIES", "200000"))

_TT: dict[tuple[Any, ...], float] = {}


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


def _board_key(board: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in board)


def _tt_get(key: tuple[Any, ...]) -> float | None:
    if not MINIMAX_USE_TT:
        return None
    return _TT.get(key)


def _tt_put(key: tuple[Any, ...], value: float) -> None:
    if not MINIMAX_USE_TT:
        return
    if len(_TT) >= MINIMAX_TT_MAX_ENTRIES:
        _TT.pop(next(iter(_TT)))
    _TT[key] = value


def _promotion_after_move(board: list[list[int]], move: dict, player: int) -> bool:
    start_r, start_c = move["path"][0]
    end_r, _ = move["path"][-1]
    piece = board[start_r][start_c]
    if player == RED and piece == RED and end_r == 0:
        return True
    if player == BLACK and piece == BLACK and end_r == 7:
        return True
    return False


def _captures_kings(board: list[list[int]], move: dict, player: int) -> int:
    captured = move.get("captured", [])
    if not captured:
        return 0
    opp_king = BLACK_KING if player == RED else RED_KING
    count = 0
    for r, c in captured:
        if board[r][c] == opp_king:
            count += 1
    return count


def _move_order_score(board: list[list[int]], move: dict, player: int, tactical_only: bool) -> float:
    captures = len(move.get("captured", []))
    king_caps = _captures_kings(board, move, player)
    promotion = _promotion_after_move(board, move, player)
    end_r, end_c = move["path"][-1]
    center = 1 if (3 <= end_r <= 4 and 2 <= end_c <= 5) else 0

    score = 0.0
    if move.get("type") == "jump":
        score += 300.0
        score += captures * 40.0
        score += king_caps * 50.0
    elif tactical_only:
        score -= 100.0
    if promotion:
        score += 80.0
    score += center * 2.0
    return score


def _order_moves(board: list[list[int]], moves: list[dict], player: int, tactical_only: bool) -> list[dict]:
    return sorted(
        moves,
        key=lambda m: _move_order_score(board, m, player, tactical_only),
        reverse=True,
    )


def quiescence_score(
    board: list[list[int]],
    qdepth: int,
    current_player: int,
    root_player: int,
    alpha: float,
    beta: float,
) -> float:
    legal = get_all_legal_moves(board, current_player)
    if not legal:
        return float(LOSS_SCORE if current_player == root_player else WIN_SCORE)

    jumps = [m for m in legal if m.get("type") == "jump"]
    if not jumps:
        return evaluate_board(board, current_player, root_player)
    if qdepth <= 0:
        return evaluate_board(board, current_player, root_player)

    tt_key = ("q", _board_key(board), qdepth, current_player, root_player)
    cached = _tt_get(tt_key)
    if cached is not None:
        return cached

    ordered = _order_moves(board, jumps, current_player, tactical_only=True)
    maximizing = current_player == root_player
    best = float("-inf") if maximizing else float("inf")

    for move in ordered:
        child = apply_move(board, move)
        score = quiescence_score(
            child,
            qdepth - 1,
            _opponent(current_player),
            root_player,
            alpha,
            beta,
        )
        if maximizing:
            if score > best:
                best = score
            if best > alpha:
                alpha = best
        else:
            if score < best:
                best = score
            if best < beta:
                beta = best
        if beta <= alpha:
            break

    _tt_put(tt_key, best)
    return best


def minimax_score(
    board: list[list[int]],
    depth: int,
    current_player: int,
    root_player: int,
    alpha: float,
    beta: float,
) -> float:
    legal = get_all_legal_moves(board, current_player)
    if not legal:
        return float(LOSS_SCORE if current_player == root_player else WIN_SCORE)

    if depth <= 0:
        return quiescence_score(
            board,
            QUIESCENCE_MAX_DEPTH,
            current_player,
            root_player,
            alpha,
            beta,
        )

    tt_key = ("m", _board_key(board), depth, current_player, root_player)
    cached = _tt_get(tt_key)
    if cached is not None:
        return cached

    ordered = _order_moves(board, legal, current_player, tactical_only=False)
    maximizing = current_player == root_player
    best = float("-inf") if maximizing else float("inf")

    for move in ordered:
        child = apply_move(board, move)
        score = minimax_score(
            child,
            depth - 1,
            _opponent(current_player),
            root_player,
            alpha,
            beta,
        )
        if maximizing:
            if score > best:
                best = score
            if best > alpha:
                alpha = best
        else:
            if score < best:
                best = score
            if best < beta:
                beta = best
        if beta <= alpha:
            break

    _tt_put(tt_key, best)
    return best


def score_move_with_minimax(
    board: list[list[int]],
    move: dict,
    current_player: int,
    depth: int = MINIMAX_DEPTH,
) -> float:
    board_after = apply_move(board, move)
    return minimax_score(
        board_after,
        max(0, depth - 1),
        _opponent(current_player),
        current_player,
        float("-inf"),
        float("inf"),
    )