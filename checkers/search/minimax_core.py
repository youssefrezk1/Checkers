from __future__ import annotations

import logging
import time
from enum import Enum
from dataclasses import dataclass
from typing import Any

from checkers.engine.board import BLACK, RED
from checkers.engine.evaluation import LOSS_SCORE, WIN_SCORE, evaluate_board
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.engine.zobrist import compute_hash

logger = logging.getLogger(__name__)
MAX_TACTICAL_EXTENSION_PLIES = 2


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


@dataclass
class SearchStats:
    nodes: int = 0
    tt_hits: int = 0


class TTBoundType(str, Enum):
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


@dataclass
class TTEntry:
    value: float
    depth: int
    bound_type: TTBoundType
    best_move: dict[str, Any] | None = None


_TT: dict[tuple[int, int, int, int, int], TTEntry] = {}


def clear_transposition_table() -> None:
    _TT.clear()


def _tt_key(
    board: list[list[int]],
    current_player: int,
    root_player: int,
    extension_depth: int = 0,
    use_phase7a: bool = True,
) -> tuple[int, int, int, int, int]:
    """
    Conservative TT key:
    - board Zobrist hash
    - side to move (current_player)
    - root_player perspective (value orientation)
    - extension_depth (to separate bounded capture extensions)
    - use_phase7a flag (evaluation ablation separation)
    """
    # bool is normalized to int to keep key purely hashable primitives
    return (compute_hash(board), current_player, root_player, extension_depth, int(use_phase7a))


def _is_promotion_move(board: list[list[int]], move: dict[str, Any], player: int) -> bool:
    path = move.get("path", [])
    if len(path) < 2:
        return False
    start_r, start_c = path[0]
    end_r, _ = path[-1]
    piece = board[start_r][start_c]
    if player == RED and piece == RED and end_r == 0:
        return True
    if player == BLACK and piece == BLACK and end_r == 7:
        return True
    return False


def order_moves(board: list[list[int]], moves: list[dict[str, Any]], player: int) -> list[dict[str, Any]]:
    """
    Stable, conservative move ordering for alpha-beta pruning:
    1) captures before non-captures
    2) among captures, larger capture count first
    3) promotion-producing moves first
    4) preserve original order for ties
    """
    indexed = list(enumerate(moves))
    ordered = sorted(
        indexed,
        key=lambda pair: (
            pair[1].get("type") == "jump",
            len(pair[1].get("captured", [])),
            _is_promotion_move(board, pair[1], player),
            -pair[0],  # keep stable original order when reverse=True
        ),
        reverse=True,
    )
    return [move for _, move in ordered]


def negamax(
    board: list[list[int]],
    depth: int,
    current_player: int,
    root_player: int,
    alpha: float,
    beta: float,
    stats: SearchStats,
    use_tt: bool = True,
    extension_depth: int = 0,
    use_tactical_extension: bool = True,
    use_phase7a: bool = True,
) -> float:
    """
    Fixed-depth alpha-beta search score from root player's perspective.

    Convention:
    - All returned scores are in root_player perspective.
    - current_player == root_player is a maximizing layer.
    - current_player != root_player is a minimizing layer.
    """
    stats.nodes += 1
    alpha_orig = alpha
    beta_orig = beta
    key = _tt_key(board, current_player, root_player, extension_depth, use_phase7a=use_phase7a)

    if use_tt:
        entry = _TT.get(key)
        if entry is not None and entry.depth >= depth:
            stats.tt_hits += 1
            if entry.bound_type == TTBoundType.EXACT:
                return entry.value
            if entry.bound_type == TTBoundType.LOWER:
                if entry.value > alpha:
                    alpha = entry.value
            elif entry.bound_type == TTBoundType.UPPER:
                if entry.value < beta:
                    beta = entry.value
            if alpha >= beta:
                return entry.value

    legal = get_all_legal_moves(board, current_player)
    if not legal:
        value = float(LOSS_SCORE if current_player == root_player else WIN_SCORE)
        if use_tt:
            existing = _TT.get(key)
            if existing is None or depth >= existing.depth:
                _TT[key] = TTEntry(value=value, depth=depth, bound_type=TTBoundType.EXACT, best_move=None)
        return value
    if depth <= 0:
        jump_moves = [m for m in legal if m.get("type") == "jump"]
        if use_tactical_extension and jump_moves and extension_depth < MAX_TACTICAL_EXTENSION_PLIES:
            ordered_jumps = order_moves(board, jump_moves, current_player)
            if current_player == root_player:
                best = float("-inf")
                best_move: dict[str, Any] | None = None
                for move in ordered_jumps:
                    child = apply_move(board, move)
                    score = negamax(
                        board=child,
                        depth=depth - 1,
                        current_player=_opponent(current_player),
                        root_player=root_player,
                        alpha=alpha,
                        beta=beta,
                        stats=stats,
                        use_tt=use_tt,
                        extension_depth=extension_depth + 1,
                        use_tactical_extension=use_tactical_extension,
                        use_phase7a=use_phase7a,
                    )
                    if score > best:
                        best = score
                        best_move = move
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        break
                if use_tt:
                    bound = TTBoundType.EXACT
                    if best <= alpha_orig:
                        bound = TTBoundType.UPPER
                    elif best >= beta_orig:
                        bound = TTBoundType.LOWER
                    existing = _TT.get(key)
                    if existing is None or depth >= existing.depth:
                        _TT[key] = TTEntry(value=float(best), depth=depth, bound_type=bound, best_move=best_move)
                return best

            best = float("inf")
            best_move = None
            for move in ordered_jumps:
                child = apply_move(board, move)
                score = negamax(
                    board=child,
                    depth=depth - 1,
                    current_player=_opponent(current_player),
                    root_player=root_player,
                    alpha=alpha,
                    beta=beta,
                    stats=stats,
                    use_tt=use_tt,
                    extension_depth=extension_depth + 1,
                    use_tactical_extension=use_tactical_extension,
                    use_phase7a=use_phase7a,
                )
                if score < best:
                    best = score
                    best_move = move
                if score < beta:
                    beta = score
                if alpha >= beta:
                    break
            if use_tt:
                bound = TTBoundType.EXACT
                if best <= alpha_orig:
                    bound = TTBoundType.UPPER
                elif best >= beta_orig:
                    bound = TTBoundType.LOWER
                existing = _TT.get(key)
                if existing is None or depth >= existing.depth:
                    _TT[key] = TTEntry(value=float(best), depth=depth, bound_type=bound, best_move=best_move)
            return best

        value = float(evaluate_board(board, current_player, root_player, use_phase7a=use_phase7a))
        if use_tt:
            existing = _TT.get(key)
            if existing is None or depth >= existing.depth:
                _TT[key] = TTEntry(value=value, depth=depth, bound_type=TTBoundType.EXACT, best_move=None)
        return value

    ordered_legal = order_moves(board, legal, current_player)
    tt_entry = _TT.get(key) if use_tt else None
    if tt_entry is not None and tt_entry.best_move is not None and tt_entry.best_move in ordered_legal:
        ordered_legal = [tt_entry.best_move] + [m for m in ordered_legal if m != tt_entry.best_move]

    if current_player == root_player:
        best = float("-inf")
        best_move: dict[str, Any] | None = None
        for move in ordered_legal:
            child = apply_move(board, move)
            score = negamax(
                board=child,
                depth=depth - 1,
                current_player=_opponent(current_player),
                root_player=root_player,
                alpha=alpha,
                beta=beta,
                stats=stats,
                use_tt=use_tt,
                extension_depth=extension_depth,
                use_tactical_extension=use_tactical_extension,
                use_phase7a=use_phase7a,
            )
            if score > best:
                best = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        if use_tt:
            bound = TTBoundType.EXACT
            if best <= alpha_orig:
                bound = TTBoundType.UPPER
            elif best >= beta_orig:
                bound = TTBoundType.LOWER
            existing = _TT.get(key)
            if existing is None or depth >= existing.depth:
                _TT[key] = TTEntry(value=float(best), depth=depth, bound_type=bound, best_move=best_move)
        return best

    best = float("inf")
    best_move = None
    for move in ordered_legal:
        child = apply_move(board, move)
        score = negamax(
            board=child,
            depth=depth - 1,
            current_player=_opponent(current_player),
            root_player=root_player,
            alpha=alpha,
            beta=beta,
            stats=stats,
            use_tt=use_tt,
            extension_depth=extension_depth,
            use_tactical_extension=use_tactical_extension,
            use_phase7a=use_phase7a,
        )
        if score < best:
            best = score
            best_move = move
        if score < beta:
            beta = score
        if alpha >= beta:
            break
    if use_tt:
        bound = TTBoundType.EXACT
        if best <= alpha_orig:
            bound = TTBoundType.UPPER
        elif best >= beta_orig:
            bound = TTBoundType.LOWER
        existing = _TT.get(key)
        if existing is None or depth >= existing.depth:
            _TT[key] = TTEntry(value=float(best), depth=depth, bound_type=bound, best_move=best_move)
    return best


def search_root(
    board: list[list[int]],
    current_player: int,
    depth: int,
    legal_moves: list[dict[str, Any]] | None = None,
    use_tt: bool = True,
    use_tactical_extension: bool = True,
    use_phase7a: bool = True,
) -> tuple[dict[str, Any] | None, float, SearchStats]:
    """
    Search the current position and choose the best move in legal move order.
    Returns (best_move, best_score, stats).
    """
    root_legal = legal_moves if legal_moves is not None else get_all_legal_moves(board, current_player)
    stats = SearchStats(nodes=1)  # Count root once for easier debugging.
    started = time.perf_counter()

    if not root_legal:
        score = float(LOSS_SCORE)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "search_root move=None score=%.2f depth=%d nodes=%d elapsed_ms=%.2f",
            score,
            depth,
            stats.nodes,
            elapsed_ms,
        )
        return None, score, stats

    best_move: dict[str, Any] | None = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in order_moves(board, root_legal, current_player):
        child = apply_move(board, move)
        score = negamax(
            board=child,
            depth=max(0, depth - 1),
            current_player=_opponent(current_player),
            root_player=current_player,
            alpha=alpha,
            beta=beta,
            stats=stats,
            use_tt=use_tt,
            extension_depth=0,
            use_tactical_extension=use_tactical_extension,
            use_phase7a=use_phase7a,
        )
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "search_root move=%s score=%.2f depth=%d nodes=%d elapsed_ms=%.2f",
        best_move.get("path") if best_move else None,
        best_score,
        depth,
        stats.nodes,
        elapsed_ms,
    )
    return best_move, float(best_score), stats


def search_root_iterative(
    board: list[list[int]],
    current_player: int,
    target_depth: int,
    legal_moves: list[dict[str, Any]] | None = None,
    use_tt: bool = True,
    use_tactical_extension: bool = True,
    use_phase7a: bool = True,
) -> tuple[dict[str, Any] | None, float, SearchStats]:
    """
    Iterative deepening root driver.

    Runs search_root from depth 1 up to target_depth and returns the deepest
    fully completed result.
    """
    if target_depth <= 0:
        return search_root(
            board=board,
            current_player=current_player,
            depth=0,
            legal_moves=legal_moves,
            use_tt=use_tt,
            use_tactical_extension=use_tactical_extension,
            use_phase7a=use_phase7a,
        )

    last_result: tuple[dict[str, Any] | None, float, SearchStats] | None = None
    started = time.perf_counter()

    for depth in range(1, target_depth + 1):
        depth_started = time.perf_counter()
        best_move, best_score, stats = search_root(
            board=board,
            current_player=current_player,
            depth=depth,
            legal_moves=legal_moves,
            use_tt=use_tt,
            use_tactical_extension=use_tactical_extension,
            use_phase7a=use_phase7a,
        )
        depth_elapsed_ms = (time.perf_counter() - depth_started) * 1000.0
        logger.info(
            "search_root_iterative depth=%d move=%s score=%.2f nodes=%d elapsed_ms=%.2f",
            depth,
            best_move.get("path") if best_move else None,
            best_score,
            stats.nodes,
            depth_elapsed_ms,
        )
        last_result = (best_move, best_score, stats)

    total_elapsed_ms = (time.perf_counter() - started) * 1000.0
    if last_result is None:
        logger.info("search_root_iterative no_result elapsed_ms=%.2f", total_elapsed_ms)
        return None, float(LOSS_SCORE), SearchStats(nodes=0)

    logger.info("search_root_iterative completed target_depth=%d total_elapsed_ms=%.2f", target_depth, total_elapsed_ms)
    return last_result
