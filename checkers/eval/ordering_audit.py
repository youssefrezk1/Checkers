"""
checkers/eval/ordering_audit.py
────────────────────────────────
Diagnostic: measures node savings from current move ordering.

Compares alpha-beta with:
  A) normal ordering (order_moves + TT best-move hoist)
  B) no ordering (legal moves in engine-generation order)

Both must return the same best move and score (correctness check).

Usage:
    venv/bin/python3 -m checkers.eval.ordering_audit
    venv/bin/python3 -m checkers.eval.ordering_audit --depth 4
    venv/bin/python3 -m checkers.eval.ordering_audit --depth 6 --out logs/ordering_audit_d6.json

No engine logic is modified.  This file only reads from the engine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from checkers.engine.board import RED, BLACK, EMPTY
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.engine.evaluation import LOSS_SCORE, WIN_SCORE, evaluate_board
from checkers.search.minimax_core import (
    SearchStats,
    clear_transposition_table,
    order_moves,
    _is_promotion_move,
)
from checkers.eval.benchmark_positions import BENCHMARK_POSITIONS


PLAYER_NAME = {RED: "RED", BLACK: "BLACK"}


def _opp(player: int) -> int:
    return BLACK if player == RED else RED


# ── Unordered negamax (no move ordering, no TT) ─────────────────────────────

def _negamax_unordered(
    board: list[list[int]],
    depth: int,
    current_player: int,
    root_player: int,
    alpha: float,
    beta: float,
    stats: SearchStats,
) -> float:
    """Alpha-beta with NO move ordering and NO TT. For audit comparison only."""
    stats.nodes += 1

    legal = get_all_legal_moves(board, current_player)
    if not legal:
        return float(LOSS_SCORE if current_player == root_player else WIN_SCORE)
    if depth <= 0:
        return float(evaluate_board(board, current_player, root_player, use_phase7a=True))

    if current_player == root_player:
        best = float("-inf")
        for move in legal:
            child = apply_move(board, move)
            score = _negamax_unordered(child, depth - 1, _opp(current_player), root_player, alpha, beta, stats)
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best
    else:
        best = float("inf")
        for move in legal:
            child = apply_move(board, move)
            score = _negamax_unordered(child, depth - 1, _opp(current_player), root_player, alpha, beta, stats)
            if score < best:
                best = score
            if score < beta:
                beta = score
            if alpha >= beta:
                break
        return best


def _search_root_unordered(
    board: list[list[int]],
    player: int,
    depth: int,
) -> tuple[dict[str, Any] | None, float, SearchStats]:
    """search_root equivalent with no ordering and no TT."""
    legal = get_all_legal_moves(board, player)
    stats = SearchStats(nodes=1)
    if not legal:
        return None, float(LOSS_SCORE), stats

    best_move = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in legal:
        child = apply_move(board, move)
        score = _negamax_unordered(child, max(0, depth - 1), _opp(player), player, alpha, beta, stats)
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    return best_move, float(best_score), stats


# ── Ordered negamax (current ordering, no TT, no extensions) ─────────────────

def _negamax_ordered(
    board: list[list[int]],
    depth: int,
    current_player: int,
    root_player: int,
    alpha: float,
    beta: float,
    stats: SearchStats,
) -> float:
    """Alpha-beta with current order_moves but NO TT, NO extensions. For audit."""
    stats.nodes += 1

    legal = get_all_legal_moves(board, current_player)
    if not legal:
        return float(LOSS_SCORE if current_player == root_player else WIN_SCORE)
    if depth <= 0:
        return float(evaluate_board(board, current_player, root_player, use_phase7a=True))

    ordered = order_moves(board, legal, current_player)

    if current_player == root_player:
        best = float("-inf")
        for move in ordered:
            child = apply_move(board, move)
            score = _negamax_ordered(child, depth - 1, _opp(current_player), root_player, alpha, beta, stats)
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best
    else:
        best = float("inf")
        for move in ordered:
            child = apply_move(board, move)
            score = _negamax_ordered(child, depth - 1, _opp(current_player), root_player, alpha, beta, stats)
            if score < best:
                best = score
            if score < beta:
                beta = score
            if alpha >= beta:
                break
        return best


def _search_root_ordered(
    board: list[list[int]],
    player: int,
    depth: int,
) -> tuple[dict[str, Any] | None, float, SearchStats]:
    """search_root equivalent with order_moves but no TT."""
    legal = get_all_legal_moves(board, player)
    stats = SearchStats(nodes=1)
    if not legal:
        return None, float(LOSS_SCORE), stats

    best_move = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in order_moves(board, legal, player):
        child = apply_move(board, move)
        score = _negamax_ordered(child, max(0, depth - 1), _opp(player), player, alpha, beta, stats)
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    return best_move, float(best_score), stats


# ── No pruning at all (pure minimax, for node-count ceiling) ─────────────────

def _minimax_no_pruning(
    board: list[list[int]],
    depth: int,
    current_player: int,
    root_player: int,
    stats: SearchStats,
) -> float:
    """Pure minimax, no alpha-beta, no ordering. Maximum node count baseline."""
    stats.nodes += 1

    legal = get_all_legal_moves(board, current_player)
    if not legal:
        return float(LOSS_SCORE if current_player == root_player else WIN_SCORE)
    if depth <= 0:
        return float(evaluate_board(board, current_player, root_player, use_phase7a=True))

    if current_player == root_player:
        best = float("-inf")
        for move in legal:
            child = apply_move(board, move)
            score = _minimax_no_pruning(child, depth - 1, _opp(current_player), root_player, stats)
            if score > best:
                best = score
        return best
    else:
        best = float("inf")
        for move in legal:
            child = apply_move(board, move)
            score = _minimax_no_pruning(child, depth - 1, _opp(current_player), root_player, stats)
            if score < best:
                best = score
        return best


def _search_root_no_pruning(
    board: list[list[int]],
    player: int,
    depth: int,
) -> tuple[dict[str, Any] | None, float, SearchStats]:
    """search_root with no pruning at all — pure minimax."""
    legal = get_all_legal_moves(board, player)
    stats = SearchStats(nodes=1)
    if not legal:
        return None, float(LOSS_SCORE), stats

    best_move = None
    best_score = float("-inf")

    for move in legal:
        child = apply_move(board, move)
        score = _minimax_no_pruning(child, max(0, depth - 1), _opp(player), player, stats)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move, float(best_score), stats


# ── Run audit for one position ───────────────────────────────────────────────

def audit_position(pos: dict, depth: int, include_no_pruning: bool = False) -> dict:
    pid = pos["position_id"]
    board = pos["board"]
    player = pos["side_to_move"]
    n_legal = len(get_all_legal_moves(board, player))

    # A: ordered
    t0 = time.perf_counter()
    ord_move, ord_score, ord_stats = _search_root_ordered(board, player, depth)
    ord_rt = time.perf_counter() - t0

    # B: unordered
    t0 = time.perf_counter()
    unord_move, unord_score, unord_stats = _search_root_unordered(board, player, depth)
    unord_rt = time.perf_counter() - t0

    # Correctness check
    score_match = ord_score == unord_score
    move_match = (ord_move == unord_move) if (ord_move is not None and unord_move is not None) else (ord_move is None and unord_move is None)

    result = {
        "position_id": pid,
        "depth": depth,
        "legal_moves": n_legal,
        "ordered_nodes": ord_stats.nodes,
        "unordered_nodes": unord_stats.nodes,
        "ordered_runtime_s": round(ord_rt, 4),
        "unordered_runtime_s": round(unord_rt, 4),
        "node_reduction_pct": round(100.0 * (1.0 - ord_stats.nodes / max(1, unord_stats.nodes)), 1),
        "runtime_reduction_pct": round(100.0 * (1.0 - ord_rt / max(0.0001, unord_rt)), 1),
        "score_match": score_match,
        "move_match": move_match,
        "ordered_score": ord_score,
        "unordered_score": unord_score,
        "ordered_best_path": ord_move.get("path") if ord_move else None,
        "unordered_best_path": unord_move.get("path") if unord_move else None,
    }

    if include_no_pruning:
        t0 = time.perf_counter()
        np_move, np_score, np_stats = _search_root_no_pruning(board, player, depth)
        np_rt = time.perf_counter() - t0
        result["no_pruning_nodes"] = np_stats.nodes
        result["no_pruning_runtime_s"] = round(np_rt, 4)
        result["no_pruning_score"] = np_score
        result["pruning_savings_pct"] = round(100.0 * (1.0 - ord_stats.nodes / max(1, np_stats.nodes)), 1)

    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Move ordering audit for alpha-beta.")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-pruning", action="store_true", help="Also measure pure minimax (slow at depth>4)")
    p.add_argument("--id", type=str, default=None)
    args = p.parse_args(argv)

    positions = BENCHMARK_POSITIONS
    if args.id:
        positions = [p for p in positions if p["position_id"] == args.id]

    print(f"Move ordering audit — depth={args.depth}, positions={len(positions)}, no_pruning={args.no_pruning}")
    print()

    results = []
    for pos in positions:
        r = audit_position(pos, args.depth, include_no_pruning=args.no_pruning)
        results.append(r)

        ok = "OK" if (r["score_match"] and r["move_match"]) else "**MISMATCH**"
        np_str = f"  no_prune={r['no_pruning_nodes']:>10,}" if "no_pruning_nodes" in r else ""
        print(
            f"  {r['position_id']:<38} "
            f"ord={r['ordered_nodes']:>8,}  unord={r['unordered_nodes']:>8,}  "
            f"reduction={r['node_reduction_pct']:>5.1f}%  "
            f"{ok}"
            f"{np_str}"
        )

    # Summary
    total_ord = sum(r["ordered_nodes"] for r in results)
    total_unord = sum(r["unordered_nodes"] for r in results)
    total_ord_rt = sum(r["ordered_runtime_s"] for r in results)
    total_unord_rt = sum(r["unordered_runtime_s"] for r in results)
    all_correct = all(r["score_match"] and r["move_match"] for r in results)

    print()
    print(f"TOTAL ordered_nodes={total_ord:,}  unordered_nodes={total_unord:,}  "
          f"reduction={100.0 * (1.0 - total_ord / max(1, total_unord)):.1f}%")
    print(f"TOTAL ordered_rt={total_ord_rt:.2f}s  unordered_rt={total_unord_rt:.2f}s  "
          f"speedup={total_unord_rt / max(0.001, total_ord_rt):.2f}x")
    print(f"CORRECTNESS: {'ALL MATCH' if all_correct else '*** MISMATCHES FOUND ***'}")

    if args.no_pruning:
        total_np = sum(r.get("no_pruning_nodes", 0) for r in results)
        print(f"TOTAL no_pruning_nodes={total_np:,}  "
              f"pruning_savings={100.0 * (1.0 - total_ord / max(1, total_np)):.1f}%")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "depth": args.depth,
            "total_ordered_nodes": total_ord,
            "total_unordered_nodes": total_unord,
            "node_reduction_pct": round(100.0 * (1.0 - total_ord / max(1, total_unord)), 1),
            "all_correct": all_correct,
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
