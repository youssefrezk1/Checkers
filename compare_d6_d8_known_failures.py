"""
compare_d6_d8_known_failures.py

Read-only diagnostic: compare D6 (stored) vs D8 (live) on the 4 ambiguous
known-failure positions — T41, T43, T45, T55.

Input:  logs/known_failure_positions_20260425_144451.json
Output: logs/known_failure_d6_vs_d8_20260425_144451.json

Canonical scorer: search_root_all_scores only.
No evaluator, ranker, proposal, or minimax changes.

Usage:
    venv/bin/python3 compare_d6_d8_known_failures.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from checkers.engine.board import RED
from checkers.engine.rules import get_all_legal_moves
from checkers.search.minimax_core import search_root_all_scores, clear_transposition_table

IN_PATH  = Path("logs/known_failure_positions_20260425_144451.json")
OUT_PATH = Path("logs/known_failure_d6_vs_d8_20260425_144451.json")
TARGET_TURNS = {41, 43, 45, 55}
D8_DEPTH = 8

# ── helpers ───────────────────────────────────────────────────────────────────

def _pk(path: list) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _path_str(path: list | None) -> str:
    if path is None:
        return "None"
    return "→".join(f"({r},{c})" for r, c in path)


def _top3_keys(scored_table: list) -> list[tuple]:
    """Return up to 3 path-keys from a score table (sorted descending by score)."""
    return [_pk(row["path"]) for row in sorted(scored_table, key=lambda x: -x["score"])[:3]]


def _score_for_path(scored: list[tuple], path: list) -> float | None:
    """Find the score for a given path in a list of (move_dict, score) tuples."""
    key = _pk(path)
    for mv, sc in scored:
        if _pk(mv["path"]) == key:
            return float(sc)
    return None


def _top3_from_scored(scored: list[tuple]) -> list[tuple]:
    """Return top-3 path-keys from search_root_all_scores output (already sorted)."""
    return [_pk(mv["path"]) for mv, _ in scored[:3]]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input not found: {IN_PATH}")

    with open(IN_PATH) as f:
        kf = json.load(f)

    positions = {p["turn"]: p for p in kf["positions"] if p["turn"] in TARGET_TURNS}
    if not positions:
        print("[ERROR] None of the target turns found in input JSON.")
        return

    BAR = "─" * 88
    print(f"D6 vs D8 comparison — {IN_PATH.name}")
    print(f"Target turns: {sorted(TARGET_TURNS)}")
    print(f"D8 depth: {D8_DEPTH}")
    print(BAR)
    hdr = (
        f"{'Turn':>4}  {'Old chosen':>24}  {'D6 best':>24}  {'D8 best':>24}  "
        f"{'D6 gap':>7}  {'D8 gap':>7}  {'top1?':>6}  {'top3?':>6}  {'D8 sec':>7}"
    )
    print(hdr)
    print(BAR)

    results = []

    for turn in sorted(TARGET_TURNS):
        pos = positions.get(turn)
        if pos is None:
            print(f"  T{turn:>2}  (not in JSON — skipped)")
            continue

        board = pos["board"]
        old_chosen_path = pos["old_chosen_move"]["path"]
        d6_table = pos["d6_score_table"]   # list of {path, score, is_d6_best, is_old_chosen, ...}

        # D6 stored data
        d6_sorted = sorted(d6_table, key=lambda x: -x["score"])
        d6_best_path  = d6_sorted[0]["path"]
        d6_best_score = d6_sorted[0]["score"]
        d6_chosen_score = next(
            (row["score"] for row in d6_table if _pk(row["path"]) == _pk(old_chosen_path)),
            None,
        )
        d6_gap = round(d6_best_score - d6_chosen_score, 2) if d6_chosen_score is not None else None
        d6_top3 = [_pk(row["path"]) for row in d6_sorted[:3]]

        # Legal moves from rule engine (ground truth — do not use stored list)
        legal = get_all_legal_moves(board, RED)

        # D8 live search
        clear_transposition_table()
        t0 = time.perf_counter()
        d8_best_move, d8_best_score, d8_scored, d8_stats = search_root_all_scores(
            board=board,
            current_player=RED,
            depth=D8_DEPTH,
            legal_moves=legal,
            use_tt=True,
            use_tactical_extension=True,
            use_phase7a=True,
        )
        elapsed = round(time.perf_counter() - t0, 2)

        d8_best_path    = d8_best_move["path"] if d8_best_move else None
        d8_chosen_score = _score_for_path(d8_scored, old_chosen_path)
        d8_gap = (
            round(float(d8_best_score) - d8_chosen_score, 2)
            if d8_best_score is not None and d8_chosen_score is not None
            else None
        )
        d8_top3 = _top3_from_scored(d8_scored)

        # Change flags
        top1_changed = (_pk(d8_best_path) != _pk(d6_best_path)) if d8_best_path else True
        top3_changed = (d8_top3 != d6_top3)

        # Node count (exposed via stats if available)
        nodes = getattr(d8_stats, "nodes", None)

        # Full D8 score table
        d8_score_table = [
            {
                "path":         mv["path"],
                "score":        round(float(sc), 2),
                "is_d8_best":   i == 0,
                "is_old_chosen": _pk(mv["path"]) == _pk(old_chosen_path),
            }
            for i, (mv, sc) in enumerate(d8_scored)
        ]

        result = {
            "turn":              turn,
            "old_chosen_path":   old_chosen_path,
            "d6_best_path":      d6_best_path,
            "d6_best_score":     round(d6_best_score, 2),
            "d8_best_path":      d8_best_path,
            "d8_best_score":     round(float(d8_best_score), 2) if d8_best_score is not None else None,
            "d6_chosen_score":   round(d6_chosen_score, 2) if d6_chosen_score is not None else None,
            "d8_chosen_score":   round(d8_chosen_score, 2) if d8_chosen_score is not None else None,
            "d6_gap":            d6_gap,
            "d8_gap":            d8_gap,
            "top1_changed":      top1_changed,
            "top3_changed":      top3_changed,
            "d8_elapsed_s":      elapsed,
            "d8_nodes":          nodes,
            "d6_top3":           [list(list(sq) for sq in k) for k in d6_top3],
            "d8_top3":           [list(list(sq) for sq in k) for k in d8_top3],
            "d8_score_table":    d8_score_table,
        }
        results.append(result)

        # Print row
        old_s  = _path_str(old_chosen_path)
        d6_s   = _path_str(d6_best_path)
        d8_s   = _path_str(d8_best_path) if d8_best_path else "None"
        d6g    = f"{d6_gap:+.1f}" if d6_gap is not None else "  n/a"
        d8g    = f"{d8_gap:+.1f}" if d8_gap is not None else "  n/a"
        t1c    = "YES" if top1_changed else "no"
        t3c    = "YES" if top3_changed else "no"
        print(
            f"  T{turn:>2}  {old_s:>24}  {d6_s:>24}  {d8_s:>24}  "
            f"{d6g:>7}  {d8g:>7}  {t1c:>6}  {t3c:>6}  {elapsed:>7.1f}s"
        )

        # Per-position detail: D6 vs D8 full rankings side-by-side
        print(f"\n  --- T{turn} full ranking comparison ---")
        d6_map = {_pk(row["path"]): row["score"] for row in d6_table}
        d8_map = {_pk(mv["path"]): float(sc) for mv, sc in d8_scored}
        all_keys = list(d8_map.keys())
        print(f"  {'path':>30}  {'D6 score':>10}  {'D8 score':>10}  {'delta':>8}  {'note'}")
        for i, (mv, sc) in enumerate(d8_scored):
            pk = _pk(mv["path"])
            d6sc = d6_map.get(pk, float("nan"))
            d8sc = float(sc)
            delta = round(d8sc - d6sc, 1) if d6sc == d6sc else float("nan")  # nan check
            note = ""
            if _pk(mv["path"]) == _pk(old_chosen_path):
                note += "CHOSEN "
            if i == 0:
                note += "D8_BEST"
            if pk == _pk(d6_best_path):
                note += " D6_BEST" if "D8_BEST" not in note else "+D6_BEST"
            print(f"  {'→'.join(f'({r},{c})' for r,c in mv['path']):>30}  {d6sc:>10.1f}  {d8sc:>10.1f}  {delta:>8.1f}  {note}")
        print()

    print(BAR)

    # Summary
    any_top1 = any(r["top1_changed"] for r in results)
    any_top3 = any(r["top3_changed"] for r in results)
    print(f"\nSUMMARY")
    print(f"  Positions analysed : {len(results)}")
    print(f"  Any top-1 changed  : {'YES — D8 flips best move at ≥1 position' if any_top1 else 'no — D8 confirms D6 top-1 everywhere'}")
    print(f"  Any top-3 changed  : {'YES — ordering differs in ≥1 position' if any_top3 else 'no — D8 confirms D6 top-3 everywhere'}")
    if not any_top1:
        print(f"  Interpretation     : Depth is NOT the problem for these 4 positions.")
        print(f"                       The failure was determined before T41.")
        print(f"                       Next step: run D8 on T31–T39 to find the upstream error,")
        print(f"                       or investigate whether BLACK's moves forced the structure.")
    else:
        changed = [r["turn"] for r in results if r["top1_changed"]]
        print(f"  Top-1 flipped at   : T{changed}")
        print(f"  Interpretation     : D6 has a horizon problem at these turns.")
        print(f"                       Increasing MINIMAX_DEPTH to 8 may be justified.")

    # Write output JSON
    out = {
        "meta": {
            "source_json":  str(IN_PATH),
            "d6_source":    "stored in input JSON",
            "d8_depth":     D8_DEPTH,
            "target_turns": sorted(TARGET_TURNS),
            "scorer":       "search_root_all_scores",
        },
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(results)} results → {OUT_PATH}")


if __name__ == "__main__":
    main()
