"""
compare_d6_d8_known_failure.py

Read-only comparison script: for each position in the known-failure benchmark,
runs D6 and D8 searches and reports whether depth changes the best move or
closes the gap to the old chosen move.

Input:   logs/known_failure_positions_20260425_144451.json
Output:  logs/known_failure_d6_vs_d8_20260425_144451.json

Usage:
    venv/bin/python3 compare_d6_d8_known_failure.py

Expected runtime: 15–40 min (11 positions × D8; D8 is ~4-8× slower than D6).
Does NOT modify any engine, evaluator, ranker, minimax, rules, or proposal code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from checkers.engine.board import RED
from checkers.engine.rules import get_all_legal_moves
from checkers.search.minimax_core import search_root_all_scores, clear_transposition_table

IN_PATH  = "logs/known_failure_positions_20260425_144451.json"
OUT_PATH = "logs/known_failure_d6_vs_d8_20260425_144451.json"
DEPTHS   = (6, 8)


# ── helpers ───────────────────────────────────────────────────────────────────

def _path_key(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _fmt_path(path) -> str:
    if not path:
        return "?"
    return "→".join(f"({r},{c})" for r, c in path)


def _score_all(board: list[list[int]], player: int, depth: int, legal):
    """Run search_root_all_scores and return (scored_list, stats, elapsed_s)."""
    clear_transposition_table()
    t0 = time.time()
    _, _, scored, stats = search_root_all_scores(
        board=board,
        current_player=player,
        depth=depth,
        legal_moves=legal,
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )
    elapsed = time.time() - t0
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored, stats, elapsed


def _old_chosen_score(scored, old_path_key) -> float | None:
    for mv, sc in scored:
        if _path_key(mv["path"]) == old_path_key:
            return sc
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data = json.loads(Path(IN_PATH).read_text())
    positions = data["positions"]

    header = f"{'Turn':>4} | {'Old chosen':>18} | {'D6 best':>18} | {'D8 best':>18} | {'D6 gap':>7} | {'D8 gap':>7} | {'Changed?':>8} | Tags"
    sep = "-" * len(header)
    print(header)
    print(sep)

    results = []

    for pos in positions:
        turn      = pos["turn"]
        board     = pos["board"]
        old_path  = pos["old_chosen_move"]["path"]
        old_pk    = _path_key(old_path)
        tags      = pos.get("tags", [])
        legal     = get_all_legal_moves(board, RED)

        row: dict = {
            "turn": turn,
            "tags": tags,
            "old_chosen_path": old_path,
            "total_pieces": pos.get("total_pieces"),
            "legal_move_count": len(legal),
        }

        by_depth: dict[int, dict] = {}

        for depth in DEPTHS:
            scored, stats, elapsed = _score_all(board, RED, depth, legal)

            best_mv, best_sc = (scored[0][0], scored[0][1]) if scored else (None, None)
            old_sc = _old_chosen_score(scored, old_pk)
            gap = round(best_sc - old_sc, 2) if best_sc is not None and old_sc is not None else None

            full_table = [
                {
                    "path": mv["path"],
                    "score": round(sc, 2),
                    "is_old_chosen": _path_key(mv["path"]) == old_pk,
                    "is_best": i == 0,
                }
                for i, (mv, sc) in enumerate(scored)
            ]

            by_depth[depth] = {
                "best_path":          best_mv["path"] if best_mv else None,
                "best_score":         round(best_sc, 2) if best_sc is not None else None,
                "old_chosen_score":   round(old_sc, 2) if old_sc is not None else None,
                "gap_best_vs_chosen": gap,
                "nodes":              stats.nodes,
                "elapsed_s":          round(elapsed, 2),
                "score_table":        full_table,
            }

        row["d6"] = by_depth[6]
        row["d8"] = by_depth[8]

        # Did the best move change D6 → D8?
        d6_best_pk = _path_key(by_depth[6]["best_path"]) if by_depth[6]["best_path"] else None
        d8_best_pk = _path_key(by_depth[8]["best_path"]) if by_depth[8]["best_path"] else None
        changed = d6_best_pk != d8_best_pk
        row["d8_changes_best_move"] = changed

        results.append(row)

        # Compact summary line
        old_s    = _fmt_path(old_path)
        d6_s     = _fmt_path(by_depth[6]["best_path"] or [])
        d8_s     = _fmt_path(by_depth[8]["best_path"] or [])
        d6_gap_s = f"{by_depth[6]['gap_best_vs_chosen']:+.1f}" if by_depth[6]["gap_best_vs_chosen"] is not None else "  n/a"
        d8_gap_s = f"{by_depth[8]['gap_best_vs_chosen']:+.1f}" if by_depth[8]["gap_best_vs_chosen"] is not None else "  n/a"
        chg_s    = "YES" if changed else "no"
        tag_s    = ",".join(tags) if tags else "-"
        print(f"{turn:>4} | {old_s:>18} | {d6_s:>18} | {d8_s:>18} | {d6_gap_s:>7} | {d8_gap_s:>7} | {chg_s:>8} | {tag_s}")

    # ── Write output ──────────────────────────────────────────────────────────
    out = {
        "meta": {
            "input":       IN_PATH,
            "depths":      list(DEPTHS),
            "n_positions": len(results),
            "description": (
                "D6 vs D8 comparison for known-failure endgame positions from "
                "game_20260425_144451_493544. Read-only: no engine code modified."
            ),
        },
        "positions": results,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(sep)
    print(f"\nWrote {len(results)} positions → {OUT_PATH}")


if __name__ == "__main__":
    main()
