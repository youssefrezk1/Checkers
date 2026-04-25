"""
checkers/eval/run_position_benchmark.py
────────────────────────────────────────
Position-level benchmark runner for the Checkers neuro-symbolic engine.

Usage
-----
    # Default depth 6, output to logs/position_benchmark_baseline.json
    python3 -m checkers.eval.run_position_benchmark

    # Custom depth and output
    python3 -m checkers.eval.run_position_benchmark --depth 4 --out logs/bench_d4.json

    # Depth sweep (1, 2, 4, 6)
    python3 -m checkers.eval.run_position_benchmark --sweep

    # Single position
    python3 -m checkers.eval.run_position_benchmark --id pos_t41_promo_tiebreak

No engine logic is modified.  This file only reads from the engine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from checkers.engine.board import RED, BLACK, EMPTY, RED_KING, BLACK_KING
from checkers.engine.evaluation import evaluate_board_breakdown
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.search.minimax_core import (
    SearchStats,
    clear_transposition_table,
    negamax,
)
from checkers.eval.benchmark_positions import BENCHMARK_POSITIONS


# ── Constants ─────────────────────────────────────────────────────────────────

PLAYER_NAME = {RED: "RED", BLACK: "BLACK"}
PIECE_SYM = {EMPTY: ".", RED: "r", BLACK: "b", RED_KING: "R", BLACK_KING: "B"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _opp(player: int) -> int:
    return BLACK if player == RED else RED


def _board_to_display(board: list[list[int]]) -> list[str]:
    lines = ["  0 1 2 3 4 5 6 7"]
    for ri, row in enumerate(board):
        lines.append(f"{ri} " + " ".join(PIECE_SYM[c] for c in row))
    return lines


def _score_move(
    board: list[list[int]],
    move: dict[str, Any],
    player: int,
    depth: int,
    stats: SearchStats,
) -> float:
    """Score a single move using depth-limited negamax, use_tt=False for isolation."""
    child = apply_move(board, move)
    clear_transposition_table()
    return float(
        negamax(
            child,
            depth - 1,
            _opp(player),
            player,
            float("-inf"),
            float("inf"),
            stats,
            use_tt=False,
        )
    )


def _paths_equal(a: list[list[int]] | None, b: list | None) -> bool:
    if a is None or b is None:
        return False
    return [list(x) for x in a] == [list(x) for x in b]


def _breakdown_compact(board: list[list[int]], next_player: int, root: int) -> dict[str, float]:
    """Return non-zero terms from evaluate_board_breakdown."""
    bd = evaluate_board_breakdown(board, next_player, root)
    return {k: float(v) for k, v in bd.items() if abs(float(v)) > 0.001}


# ── Core runner for one position ─────────────────────────────────────────────

def run_position(pos: dict[str, Any], depth: int) -> dict[str, Any]:
    position_id = pos["position_id"]
    board = pos["board"]
    player = pos["side_to_move"]
    expected_path = pos.get("expected_best_path")
    known_failure = pos.get("known_failure", False)

    legal = get_all_legal_moves(board, player)

    # Edge case: no legal moves (terminal)
    if not legal:
        return {
            "position_id": position_id,
            "category": pos["category"],
            "tags": pos["tags"],
            "side_to_move": PLAYER_NAME[player],
            "depth": depth,
            "legal_move_count": 0,
            "terminal": True,
            "error": "No legal moves — terminal position",
            "expected_best_path": expected_path,
            "expected_match": False,
            "known_failure": known_failure,
            "runtime_s": 0.0,
        }

    t0 = time.perf_counter()

    # Score all legal moves
    scored: list[tuple[float, dict[str, Any]]] = []
    total_stats = SearchStats()
    for m in legal:
        s = _score_move(board, m, player, depth, total_stats)
        scored.append((s, m))

    scored.sort(key=lambda x: -x[0])
    runtime_s = time.perf_counter() - t0

    best_score, best_move = scored[0]
    best_path = best_move["path"]

    # Score table
    score_table = [
        {
            "path": m["path"],
            "type": m.get("type", "simple"),
            "captured": m.get("captured", []),
            "score": round(s, 2),
            "is_best": (m["path"] == best_path),
            "is_expected": _paths_equal(m["path"], expected_path),
        }
        for s, m in scored
    ]

    # Expected-move match
    expected_match: bool | None = None
    expected_score: float | None = None
    score_gap_from_expected: float | None = None
    if expected_path is not None:
        matched = [s for s, m in scored if _paths_equal(m["path"], expected_path)]
        if matched:
            expected_match = _paths_equal(best_path, expected_path)
            expected_score = round(matched[0], 2)
            score_gap_from_expected = round(best_score - matched[0], 2)
        else:
            expected_match = False
            expected_score = None
            score_gap_from_expected = None

    # Evaluation breakdown for best move
    best_child = apply_move(board, best_move)
    breakdown = _breakdown_compact(best_child, _opp(player), player)

    # Jump safety check: does best move expose piece to immediate recapture?
    recaptures = [
        m for m in get_all_legal_moves(best_child, _opp(player))
        if m.get("type") == "jump"
    ]

    return {
        "position_id": position_id,
        "category": pos["category"],
        "tags": pos["tags"],
        "side_to_move": PLAYER_NAME[player],
        "depth": depth,
        "legal_move_count": len(legal),
        "terminal": False,
        "score_table": score_table,
        "best_minimax_path": best_path,
        "best_minimax_score": round(best_score, 2),
        "second_best_score": round(scored[1][0], 2) if len(scored) > 1 else None,
        "score_gap_top2": round(best_score - scored[1][0], 2) if len(scored) > 1 else None,
        "expected_best_path": expected_path,
        "expected_match": expected_match,
        "expected_score": expected_score,
        "score_gap_from_expected": score_gap_from_expected,
        "eval_breakdown_after_best": breakdown,
        "opponent_recaptures_after_best": len(recaptures),
        "nodes_searched": total_stats.nodes,
        "tt_hits": total_stats.tt_hits,
        "runtime_s": round(runtime_s, 4),
        "explanation": pos.get("explanation", ""),
        "known_failure": known_failure,
    }


# ── Summary aggregation ───────────────────────────────────────────────────────

def _summarize(results: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    total = len(results)
    terminal = sum(1 for r in results if r.get("terminal"))
    with_expected = [r for r in results if r.get("expected_best_path") is not None and not r.get("terminal")]
    exact_matches = sum(1 for r in with_expected if r.get("expected_match") is True)
    regressions = sum(
        1 for r in with_expected
        if r.get("expected_match") is False and not r.get("known_failure", False)
    )
    known_still_failing = sum(
        1 for r in results
        if r.get("known_failure") and r.get("expected_match") is False
    )
    known_now_passing = sum(
        1 for r in results
        if r.get("known_failure") and r.get("expected_match") is True
    )

    gaps = [
        r["score_gap_from_expected"]
        for r in with_expected
        if r.get("score_gap_from_expected") is not None
    ]
    avg_gap = round(sum(gaps) / len(gaps), 2) if gaps else None
    total_nodes = sum(r.get("nodes_searched", 0) for r in results)
    total_runtime = round(sum(r.get("runtime_s", 0.0) for r in results), 3)
    avg_runtime = round(total_runtime / max(1, total - terminal), 4)

    return {
        "depth": depth,
        "total_positions": total,
        "terminal_positions": terminal,
        "positions_with_expected": len(with_expected),
        "exact_expected_matches": exact_matches,
        "regressions_unexpected": regressions,
        "known_failures_still_failing": known_still_failing,
        "known_failures_now_passing": known_now_passing,
        "avg_score_gap_from_expected": avg_gap,
        "total_nodes_searched": total_nodes,
        "total_runtime_s": total_runtime,
        "avg_runtime_per_position_s": avg_runtime,
    }


# ── Depth sweep ───────────────────────────────────────────────────────────────

def run_sweep(
    positions: list[dict],
    depths: list[int],
) -> dict[str, Any]:
    sweep: dict[str, Any] = {"depths": {}}
    for d in depths:
        print(f"\n── Depth {d} ──────────────────────────────")
        results = []
        for pos in positions:
            r = run_position(pos, depth=d)
            results.append(r)
            match_str = (
                "✓" if r.get("expected_match") is True
                else "✗" if r.get("expected_match") is False
                else "~"
            )
            print(
                f"  {r['position_id']:<38} "
                f"best={str(r.get('best_minimax_path', '?')):<22} "
                f"score={r.get('best_minimax_score', '?'):>8}  "
                f"{match_str}  {r.get('runtime_s', 0):.2f}s"
            )
        summary = _summarize(results, depth=d)
        sweep["depths"][str(d)] = {"summary": summary, "results": results}
        print(
            f"  → matches={summary['exact_expected_matches']}/{summary['positions_with_expected']} "
            f"known_failing={summary['known_failures_still_failing']} "
            f"known_passing={summary['known_failures_now_passing']} "
            f"nodes={summary['total_nodes_searched']:,} "
            f"time={summary['total_runtime_s']:.1f}s"
        )
    return sweep


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Position-level benchmark for the Checkers neuro-symbolic engine."
    )
    p.add_argument("--depth", type=int, default=6, help="Minimax depth (default 6)")
    p.add_argument(
        "--out",
        type=str,
        default="logs/position_benchmark_baseline.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run at depths 1, 2, 4, 6 and report delta",
    )
    p.add_argument(
        "--id",
        type=str,
        default=None,
        help="Run only the position with this position_id",
    )
    args = p.parse_args(argv)

    # Filter positions
    positions = BENCHMARK_POSITIONS
    if args.id:
        positions = [p for p in positions if p["position_id"] == args.id]
        if not positions:
            print(f"ERROR: No position found with id={args.id!r}")
            return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CHECKERS ENGINE — POSITION BENCHMARK")
    print(f"Positions: {len(positions)}  |  depth: {args.depth}")
    print("=" * 65)

    if args.sweep:
        sweep_depths = [1, 2, 4, 6]
        report = run_sweep(positions, sweep_depths)
        report["meta"] = {
            "sweep_depths": sweep_depths,
            "position_count": len(positions),
            "output": str(out_path),
        }
    else:
        results = []
        for pos in positions:
            r = run_position(pos, depth=args.depth)
            results.append(r)
            match_str = (
                "✓" if r.get("expected_match") is True
                else "✗" if r.get("expected_match") is False
                else "~"
            )
            print(
                f"  {r['position_id']:<38} "
                f"best={str(r.get('best_minimax_path', '?')):<22} "
                f"score={r.get('best_minimax_score', '?'):>8}  "
                f"{match_str}  {r.get('runtime_s', 0):.2f}s"
            )

        summary = _summarize(results, depth=args.depth)

        print()
        print("── SUMMARY ────────────────────────────────────────────────")
        print(f"  Total positions      : {summary['total_positions']}")
        print(f"  With expected move   : {summary['positions_with_expected']}")
        print(f"  Exact matches        : {summary['exact_expected_matches']}")
        print(f"  Regressions          : {summary['regressions_unexpected']}")
        print(f"  Known failures still : {summary['known_failures_still_failing']}")
        print(f"  Known failures FIXED : {summary['known_failures_now_passing']}")
        print(f"  Avg gap from expected: {summary['avg_score_gap_from_expected']}")
        print(f"  Total nodes          : {summary['total_nodes_searched']:,}")
        print(f"  Total runtime        : {summary['total_runtime_s']:.1f}s")
        print(f"  Avg per position     : {summary['avg_runtime_per_position_s']:.2f}s")

        report = {
            "meta": {
                "depth": args.depth,
                "position_count": len(positions),
                "output": str(out_path),
            },
            "summary": summary,
            "results": results,
        }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
