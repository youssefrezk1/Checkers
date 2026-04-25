"""
checkers/eval/scoring_path_comparison.py
─────────────────────────────────────────
Diagnostic: compares the three scoring paths on benchmark positions.

A) Production-style: per-move negamax with use_tt=False, full (-inf,+inf) window
   (exactly as symbolic_decision._score_all_moves does)
B) search_root: ordered moves, progressive alpha narrowing, use_tt configurable
C) search_root_iterative: iterative deepening with PV feed-forward, use_tt=True

No engine logic is modified.  This file only reads from the engine.

Usage:
    venv/bin/python3 -m checkers.eval.scoring_path_comparison --depth 6
    venv/bin/python3 -m checkers.eval.scoring_path_comparison --depth 4 --out logs/scoring_path_comparison_d4.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from checkers.engine.board import RED, BLACK
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.engine.evaluation import LOSS_SCORE
from checkers.search.minimax_core import (
    SearchStats,
    clear_transposition_table,
    negamax,
    search_root,
    search_root_all_scores,
    search_root_iterative,
)
from checkers.eval.benchmark_positions import BENCHMARK_POSITIONS

PLAYER_NAME = {RED: "RED", BLACK: "BLACK"}


def _opp(player: int) -> int:
    return BLACK if player == RED else RED


# ── Path A: production-style per-move scoring ────────────────────────────────

def _score_all_production(
    board: list[list[int]],
    legal: list[dict[str, Any]],
    player: int,
    depth: int,
) -> tuple[dict[str, Any] | None, float, list[tuple[dict, float]], int]:
    """
    Replicates symbolic_decision._score_all_moves exactly:
    - per-move negamax
    - use_tt=False
    - alpha=-inf, beta=inf (full window, no cross-move pruning)
    - use_tactical_extension=True
    - use_phase7a=True
    - no position_history (benchmark has none)

    Returns (best_move, best_score, all_scored, total_nodes).
    """
    scored: list[tuple[dict, float]] = []
    total_nodes = 0

    for move in legal:
        stats = SearchStats()
        child = apply_move(board, move)
        score = float(
            negamax(
                board=child,
                depth=max(0, depth - 1),
                current_player=_opp(player),
                root_player=player,
                alpha=float("-inf"),
                beta=float("inf"),
                stats=stats,
                use_tt=False,
                extension_depth=0,
                use_tactical_extension=True,
                use_phase7a=True,
            )
        )
        total_nodes += stats.nodes
        scored.append((move, score))

    scored.sort(key=lambda x: -x[1])
    best_move, best_score = scored[0] if scored else (None, float(LOSS_SCORE))
    return best_move, best_score, scored, total_nodes


# ── Run comparison for one position ──────────────────────────────────────────

def compare_position(pos: dict, depth: int) -> dict[str, Any]:
    pid = pos["position_id"]
    board = pos["board"]
    player = pos["side_to_move"]
    legal = get_all_legal_moves(board, player)
    n_legal = len(legal)

    if not legal:
        return {
            "position_id": pid,
            "depth": depth,
            "legal_moves": 0,
            "terminal": True,
        }

    # ── A: Production-style ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    a_move, a_score, a_scored, a_nodes = _score_all_production(board, legal, player, depth)
    a_rt = time.perf_counter() - t0

    # ── B: search_root with use_tt=False (apples-to-apples TT comparison) ───
    clear_transposition_table()
    t0 = time.perf_counter()
    b_move, b_score, b_stats = search_root(
        board=board, current_player=player, depth=depth,
        use_tt=False, use_tactical_extension=True, use_phase7a=True,
    )
    b_rt = time.perf_counter() - t0

    # ── B2: search_root with use_tt=True ────────────────────────────────────
    clear_transposition_table()
    t0 = time.perf_counter()
    b2_move, b2_score, b2_stats = search_root(
        board=board, current_player=player, depth=depth,
        use_tt=True, use_tactical_extension=True, use_phase7a=True,
    )
    b2_rt = time.perf_counter() - t0

    # ── C: search_root_iterative (PV + TT) ──────────────────────────────────
    clear_transposition_table()
    t0 = time.perf_counter()
    c_move, c_score, c_stats = search_root_iterative(
        board=board, current_player=player, target_depth=depth,
        use_tt=True, use_tactical_extension=True, use_phase7a=True,
    )
    c_rt = time.perf_counter() - t0

    # ── D: search_root_all_scores with TT (exact per-move + TT sharing) ────
    clear_transposition_table()
    t0 = time.perf_counter()
    d_move, d_score, d_scored, d_stats = search_root_all_scores(
        board=board, current_player=player, depth=depth,
        use_tt=True, use_tactical_extension=True, use_phase7a=True,
    )
    d_rt = time.perf_counter() - t0

    # ── Comparisons ──────────────────────────────────────────────────────────
    a_path = a_move.get("path") if a_move else None
    b_path = b_move.get("path") if b_move else None
    b2_path = b2_move.get("path") if b2_move else None
    c_path = c_move.get("path") if c_move else None
    d_path = d_move.get("path") if d_move else None

    ab_move_match = a_path == b_path
    ab_score_match = a_score == b_score
    ab2_move_match = a_path == b2_path
    ab2_score_match = a_score == b2_score
    ac_move_match = a_path == c_path
    ac_score_match = a_score == c_score
    ad_move_match = a_path == d_path
    ad_score_match = a_score == d_score

    # All-move score comparison: A vs D (the key measurement)
    a_by_path = {
        tuple(map(tuple, m.get("path", []))): s for m, s in a_scored
    }
    d_by_path = {
        tuple(map(tuple, m.get("path", []))): s for m, s in d_scored
    }
    all_move_scores_match = True
    all_move_mismatches: list[dict] = []
    for path_key, a_s in a_by_path.items():
        d_s = d_by_path.get(path_key)
        if d_s is None:
            all_move_scores_match = False
            all_move_mismatches.append({"path": list(path_key), "a_score": round(a_s, 2), "d_score": None})
        elif round(a_s, 2) != round(d_s, 2):
            all_move_scores_match = False
            all_move_mismatches.append({"path": list(path_key), "a_score": round(a_s, 2), "d_score": round(d_s, 2)})

    # Build score table for path A (top 5)
    a_table = [
        {"path": m.get("path"), "score": round(s, 2)}
        for m, s in a_scored[:5]
    ]

    # Diagnose any mismatches
    mismatch_notes: list[str] = []
    if not ab_score_match:
        mismatch_notes.append(
            f"A vs B(no-TT) score differs: A={a_score} B={b_score}. "
            "Both use use_tt=False, same extensions. "
            "Difference must come from alpha-beta window narrowing "
            "(A uses full window per move; B uses progressive alpha)."
        )
    if not ab2_score_match:
        mismatch_notes.append(
            f"A vs B(TT) score differs: A={a_score} B2={b2_score}. "
            "TT may cause different pruning paths."
        )
    if not ac_score_match:
        mismatch_notes.append(
            f"A vs C(iterative+TT) score differs: A={a_score} C={c_score}. "
            "Iterative deepening PV + TT warm-up may alter pruning."
        )
    if not ad_score_match:
        mismatch_notes.append(
            f"A vs D(all_scores+TT) best score differs: A={a_score} D={d_score}."
        )
    if not all_move_scores_match:
        mismatch_notes.append(
            f"A vs D all-move score mismatch on {len(all_move_mismatches)} move(s): "
            f"{all_move_mismatches}"
        )

    return {
        "position_id": pid,
        "depth": depth,
        "legal_moves": n_legal,
        "terminal": False,

        "A_production_best_path": a_path,
        "A_production_best_score": round(a_score, 2),
        "A_production_nodes": a_nodes,
        "A_production_runtime_s": round(a_rt, 4),
        "A_production_top5": a_table,

        "B_search_root_noTT_best_path": b_path,
        "B_search_root_noTT_best_score": round(b_score, 2),
        "B_search_root_noTT_nodes": b_stats.nodes,
        "B_search_root_noTT_runtime_s": round(b_rt, 4),

        "B2_search_root_TT_best_path": b2_path,
        "B2_search_root_TT_best_score": round(b2_score, 2),
        "B2_search_root_TT_nodes": b2_stats.nodes,
        "B2_search_root_TT_runtime_s": round(b2_rt, 4),

        "C_iterative_TT_PV_best_path": c_path,
        "C_iterative_TT_PV_best_score": round(c_score, 2),
        "C_iterative_TT_PV_nodes": c_stats.nodes,
        "C_iterative_TT_PV_runtime_s": round(c_rt, 4),

        "D_all_scores_TT_best_path": d_path,
        "D_all_scores_TT_best_score": round(d_score, 2),
        "D_all_scores_TT_nodes": d_stats.nodes,
        "D_all_scores_TT_runtime_s": round(d_rt, 4),
        "D_all_move_scores_match_A": all_move_scores_match,
        "D_all_move_mismatches": all_move_mismatches,

        "A_vs_B_move_match": ab_move_match,
        "A_vs_B_score_match": ab_score_match,
        "A_vs_B2_move_match": ab2_move_match,
        "A_vs_B2_score_match": ab2_score_match,
        "A_vs_C_move_match": ac_move_match,
        "A_vs_C_score_match": ac_score_match,
        "A_vs_D_move_match": ad_move_match,
        "A_vs_D_score_match": ad_score_match,

        "node_savings_B_vs_A_pct": round(100.0 * (1 - b_stats.nodes / max(1, a_nodes)), 1),
        "node_savings_B2_vs_A_pct": round(100.0 * (1 - b2_stats.nodes / max(1, a_nodes)), 1),
        "node_savings_C_vs_A_pct": round(100.0 * (1 - c_stats.nodes / max(1, a_nodes)), 1),
        "node_savings_D_vs_A_pct": round(100.0 * (1 - d_stats.nodes / max(1, a_nodes)), 1),

        "mismatch_notes": mismatch_notes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scoring path comparison diagnostic.")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--id", type=str, default=None)
    args = p.parse_args(argv)

    positions = BENCHMARK_POSITIONS
    if args.id:
        positions = [pos for pos in positions if pos["position_id"] == args.id]

    if not args.out:
        args.out = f"logs/scoring_path_comparison_d{args.depth}.json"

    print(f"Scoring path comparison — depth={args.depth}, positions={len(positions)}")
    print()
    print(
        f"{'Position':<38} "
        f"{'A nodes':>8} {'B nodes':>8} {'B2 nodes':>8} {'C nodes':>8} {'D nodes':>8} | "
        f"{'A=B':>4} {'A=B2':>4} {'A=C':>4} {'A=D':>4} {'D all':>5} | "
        f"{'A rt':>6} {'D rt':>6}"
    )
    print("-" * 150)

    results = []
    totals = {"a_nodes": 0, "b_nodes": 0, "b2_nodes": 0, "c_nodes": 0, "d_nodes": 0,
              "a_rt": 0.0, "b_rt": 0.0, "b2_rt": 0.0, "c_rt": 0.0, "d_rt": 0.0}
    all_match = True
    all_d_all_move_match = True

    for pos in positions:
        r = compare_position(pos, args.depth)
        results.append(r)

        if r.get("terminal"):
            print(f"  {r['position_id']:<38} TERMINAL")
            continue

        ab = "OK" if r["A_vs_B_score_match"] else "DIFF"
        ab2 = "OK" if r["A_vs_B2_score_match"] else "DIFF"
        ac = "OK" if r["A_vs_C_score_match"] else "DIFF"
        ad = "OK" if r["A_vs_D_score_match"] else "DIFF"
        d_all = "OK" if r["D_all_move_scores_match_A"] else "DIFF"

        if not (r["A_vs_B_score_match"] and r["A_vs_B2_score_match"]
                and r["A_vs_C_score_match"] and r["A_vs_D_score_match"]):
            all_match = False
        if not r["D_all_move_scores_match_A"]:
            all_d_all_move_match = False

        totals["a_nodes"] += r["A_production_nodes"]
        totals["b_nodes"] += r["B_search_root_noTT_nodes"]
        totals["b2_nodes"] += r["B2_search_root_TT_nodes"]
        totals["c_nodes"] += r["C_iterative_TT_PV_nodes"]
        totals["d_nodes"] += r["D_all_scores_TT_nodes"]
        totals["a_rt"] += r["A_production_runtime_s"]
        totals["b_rt"] += r["B_search_root_noTT_runtime_s"]
        totals["b2_rt"] += r["B2_search_root_TT_runtime_s"]
        totals["c_rt"] += r["C_iterative_TT_PV_runtime_s"]
        totals["d_rt"] += r["D_all_scores_TT_runtime_s"]

        print(
            f"  {r['position_id']:<38} "
            f"{r['A_production_nodes']:>8,} {r['B_search_root_noTT_nodes']:>8,} "
            f"{r['B2_search_root_TT_nodes']:>8,} {r['C_iterative_TT_PV_nodes']:>8,} "
            f"{r['D_all_scores_TT_nodes']:>8,} | "
            f"{ab:>4} {ab2:>4} {ac:>4} {ad:>4} {d_all:>5} | "
            f"{r['A_production_runtime_s']:>5.2f}s {r['D_all_scores_TT_runtime_s']:>5.2f}s"
        )

        for note in r.get("mismatch_notes", []):
            print(f"    NOTE: {note}")

    print("-" * 150)
    print(
        f"  {'TOTAL':<38} "
        f"{totals['a_nodes']:>8,} {totals['b_nodes']:>8,} "
        f"{totals['b2_nodes']:>8,} {totals['c_nodes']:>8,} "
        f"{totals['d_nodes']:>8,} | "
        f"{'':>4} {'':>4} {'':>4} {'':>4} {'':>5} | "
        f"{totals['a_rt']:>5.2f}s {totals['d_rt']:>5.2f}s"
    )

    overall_b = 100.0 * (1 - totals["b_nodes"] / max(1, totals["a_nodes"]))
    overall_b2 = 100.0 * (1 - totals["b2_nodes"] / max(1, totals["a_nodes"]))
    overall_c = 100.0 * (1 - totals["c_nodes"] / max(1, totals["a_nodes"]))
    overall_d = 100.0 * (1 - totals["d_nodes"] / max(1, totals["a_nodes"]))

    print()
    print(f"Node savings vs production (A):")
    print(f"  B  (search_root, no TT):            {overall_b:.1f}%")
    print(f"  B2 (search_root, TT):               {overall_b2:.1f}%")
    print(f"  C  (iterative, TT+PV):              {overall_c:.1f}%")
    print(f"  D  (all_scores, TT, full window):   {overall_d:.1f}%")
    print(f"All best scores match: {all_match}")
    print(f"D all-move scores match A: {all_d_all_move_match}")
    print(f"Runtime speedup D vs A: {totals['a_rt'] / max(0.001, totals['d_rt']):.2f}x")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "depth": args.depth,
        "all_best_scores_match": all_match,
        "D_all_move_scores_match_A": all_d_all_move_match,
        "totals": totals,
        "node_savings_B_vs_A_pct": round(overall_b, 1),
        "node_savings_B2_vs_A_pct": round(overall_b2, 1),
        "node_savings_C_vs_A_pct": round(overall_c, 1),
        "node_savings_D_vs_A_pct": round(overall_d, 1),
        "runtime_speedup_D_vs_A": round(totals["a_rt"] / max(0.001, totals["d_rt"]), 2),
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
