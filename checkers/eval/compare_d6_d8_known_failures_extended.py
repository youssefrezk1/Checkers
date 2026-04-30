"""
compare_d6_d8_known_failures_extended.py

Extended read-only diagnostic: D6 (canonical or live) vs D8 on turns
T35, T37, T39, T41, T43, T45, T47, T49, T51, T55.

- Turns in the JSON   (T35,T37,T41,T43,T45,T49,T51,T55): D6 loaded from stored
  score table; D8 run live.
- Turns NOT in JSON   (T39, T47): board reconstructed from JSONL; both D6 and
  D8 run live via search_root_all_scores.

Canonical scorer: search_root_all_scores only.
No evaluator / ranker / proposal / minimax changes.

Input:
    logs/known_failure_positions_20260425_144451.json
    logs/game_20260425_144451_493544.jsonl

Output:
    logs/known_failure_d6_vs_d8_extended_20260425_144451.json

Usage:
    venv/bin/python3 compare_d6_d8_known_failures_extended.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.search.minimax_core import search_root_all_scores, clear_transposition_table

IN_JSON   = Path("logs/known_failure_positions_20260425_144451.json")
IN_JSONL  = Path("logs/game_20260425_144451_493544.jsonl")
OUT_PATH  = Path("logs/known_failure_d6_vs_d8_extended_20260425_144451.json")

TARGET_TURNS = [35, 37, 39, 41, 43, 45, 47, 49, 51, 55]
D6_DEPTH = 6
D8_DEPTH = 8

# ── helpers ───────────────────────────────────────────────────────────────────

def _pk(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _path_str(path) -> str:
    if path is None:
        return "—"
    return "→".join(f"({r},{c})" for r, c in path)


def _score_for_path(scored_tuples: list, path: list) -> float | None:
    key = _pk(path)
    for mv, sc in scored_tuples:
        if _pk(mv["path"]) == key:
            return float(sc)
    return None


def _top3_keys(scored_tuples: list) -> list[tuple]:
    return [_pk(mv["path"]) for mv, _ in scored_tuples[:3]]


# ── board reconstruction from JSONL ──────────────────────────────────────────

def _make_start() -> list[list[int]]:
    b = [[0] * 8 for _ in range(8)]
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = BLACK
    return b


def _rebuild_boards(records: list[dict]) -> dict[int, list[list[int]]]:
    """
    Rebuild board-before-turn snapshots using the same logic as
    extract_known_failure_positions.py:
        boards[turn-1] = board after turn (turn-1) = board before turn.
    apply_move only — no manual promotion patch (matches extractor behaviour).
    """
    import copy
    board = _make_start()
    boards: dict[int, list[list[int]]] = {0: [row[:] for row in board]}
    for rec in records:
        t = rec["turn"]
        move = {
            "type":     rec["move_type"],
            "path":     rec["path"],
            "captured": rec.get("captured", []),
        }
        board = apply_move(board, move)
        boards[t] = [row[:] for row in board]
    return boards


# ── search wrapper ────────────────────────────────────────────────────────────

def _run_search(board, depth: int) -> tuple[list, list, float, int]:
    """
    Returns (best_path, all_scored_tuples, elapsed_s, node_count).
    all_scored_tuples is sorted descending by score.
    """
    legal = get_all_legal_moves(board, RED)
    clear_transposition_table()
    t0 = time.perf_counter()
    best_move, best_score, scored, stats = search_root_all_scores(
        board=board,
        current_player=RED,
        depth=depth,
        legal_moves=legal,
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    nodes = getattr(stats, "nodes", None)
    return best_move["path"] if best_move else None, scored, elapsed, nodes


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for p in [IN_JSON, IN_JSONL]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    # Load stored positions
    with open(IN_JSON) as f:
        kf = json.load(f)
    stored: dict[int, dict] = {p["turn"]: p for p in kf["positions"]}

    # Load JSONL for board reconstruction and old chosen paths
    with open(IN_JSONL) as f:
        records = [json.loads(l) for l in f if l.strip()]
    rec_by_turn = {r["turn"]: r for r in records}
    boards = _rebuild_boards(records)   # boards[t] = board AFTER turn t

    BAR = "─" * 96
    print(f"D6 vs D8 extended comparison")
    print(f"  JSON source : {IN_JSON.name}")
    print(f"  JSONL source: {IN_JSONL.name}")
    print(f"  Turns       : {TARGET_TURNS}")
    print(f"  D8 depth    : {D8_DEPTH}")
    print(BAR)
    print(
        f"{'Trn':>3}  {'Old chosen':>20}  {'D6 best':>20}  {'D8 best':>20}  "
        f"{'cD6':>8}  {'cD8':>8}  {'gD6':>7}  {'gD8':>7}  {'t1?':>4}  {'D8s':>6}  {'src':>5}"
    )
    print(BAR)

    results = []

    for turn in TARGET_TURNS:
        rec = rec_by_turn.get(turn)
        if rec is None:
            print(f"  T{turn:>2}  (turn not in JSONL — skipped)")
            continue

        old_chosen_path = rec["path"]
        board_before = boards[turn - 1]   # board after turn (turn-1) = before turn

        # ── D6 ────────────────────────────────────────────────────────────────
        if turn in stored:
            # Use stored D6 score table
            d6_table = stored[turn]["d6_score_table"]
            d6_sorted = sorted(d6_table, key=lambda x: -x["score"])
            d6_best_path  = d6_sorted[0]["path"]
            d6_best_score = float(d6_sorted[0]["score"])
            d6_chosen_score = next(
                (float(row["score"]) for row in d6_table
                 if _pk(row["path"]) == _pk(old_chosen_path)),
                None,
            )
            # Reconstruct scored tuples for top-3 key comparison
            d6_scored_keys = [_pk(row["path"]) for row in d6_sorted[:3]]
            d6_src = "json"
        else:
            # Reconstruct D6 live
            d6_best_path, d6_scored_live, d6_elapsed_live, _ = _run_search(board_before, D6_DEPTH)
            d6_best_score = float(d6_scored_live[0][1]) if d6_scored_live else float("-inf")
            d6_chosen_score = _score_for_path(d6_scored_live, old_chosen_path)
            d6_scored_keys = _top3_keys(d6_scored_live)
            d6_src = "live6"

        d6_gap = round(d6_best_score - d6_chosen_score, 2) if d6_chosen_score is not None else None

        # ── D8 ────────────────────────────────────────────────────────────────
        d8_best_path, d8_scored, d8_elapsed, d8_nodes = _run_search(board_before, D8_DEPTH)
        d8_best_score    = float(d8_scored[0][1]) if d8_scored else float("-inf")
        d8_chosen_score  = _score_for_path(d8_scored, old_chosen_path)
        d8_gap = (
            round(d8_best_score - d8_chosen_score, 2)
            if d8_best_score is not None and d8_chosen_score is not None
            else None
        )
        d8_top3_keys = _top3_keys(d8_scored)

        # Change flags
        top1_changed = (
            _pk(d8_best_path) != _pk(d6_best_path)
            if d8_best_path and d6_best_path else True
        )
        top3_changed = (d8_top3_keys != d6_scored_keys)

        # Full D8 score table
        d8_score_table = [
            {
                "path":          mv["path"],
                "score":         round(float(sc), 2),
                "is_d8_best":    i == 0,
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
            "d8_best_score":     round(d8_best_score, 2),
            "d6_chosen_score":   round(d6_chosen_score, 2) if d6_chosen_score is not None else None,
            "d8_chosen_score":   round(d8_chosen_score, 2) if d8_chosen_score is not None else None,
            "d6_gap":            d6_gap,
            "d8_gap":            d8_gap,
            "top1_changed":      top1_changed,
            "top3_changed":      top3_changed,
            "d8_elapsed_s":      d8_elapsed,
            "d8_nodes":          d8_nodes,
            "d6_source":         d6_src,
            "d8_score_table":    d8_score_table,
        }
        results.append(result)

        # Compact table row
        old_s = _path_str(old_chosen_path)
        d6_s  = _path_str(d6_best_path)
        d8_s  = _path_str(d8_best_path) if d8_best_path else "—"
        cd6   = f"{d6_chosen_score:+.1f}" if d6_chosen_score is not None else "  n/a"
        cd8   = f"{d8_chosen_score:+.1f}" if d8_chosen_score is not None else "  n/a"
        gd6   = f"{d6_gap:+.1f}"          if d6_gap  is not None else "  n/a"
        gd8   = f"{d8_gap:+.1f}"          if d8_gap  is not None else "  n/a"
        t1    = "YES" if top1_changed else " no"
        print(
            f"T{turn:>2}  {old_s:>20}  {d6_s:>20}  {d8_s:>20}  "
            f"{cd6:>8}  {cd8:>8}  {gd6:>7}  {gd8:>7}  {t1:>4}  {d8_elapsed:>5.1f}s  {d6_src:>5}"
        )

    print(BAR)

    # ── Summary ───────────────────────────────────────────────────────────────
    changed_turns  = [r["turn"] for r in results if r["top1_changed"]]
    any_top1       = bool(changed_turns)
    gap_grew       = [r["turn"] for r in results
                      if r["d6_gap"] is not None and r["d8_gap"] is not None
                      and r["d8_gap"] > r["d6_gap"] + 2]

    print()
    print("SUMMARY")
    print(f"  Positions analysed : {len(results)}")
    print(f"  top-1 changed      : {changed_turns if changed_turns else 'none'}")
    print(f"  gap grew at D8     : {gap_grew if gap_grew else 'none'}")
    if not any_top1 and not gap_grew:
        print("  Interpretation     : D8 confirms D6 everywhere.")
        print("                       Depth is NOT the problem for these turns.")
        print("                       The failure was determined before T35,")
        print("                       or by BLACK's moves — not RED evaluator depth.")
    elif any_top1:
        print(f"  Interpretation     : D8 flips best move at T{changed_turns}.")
        print(f"                       D6 has a horizon problem at those positions.")
        print(f"                       Increasing MINIMAX_DEPTH to 8 may be justified.")
    else:
        print(f"  Interpretation     : D8 does not flip top-1 but gap grew at T{gap_grew}.")
        print(f"                       Chosen move becomes comparatively weaker at D8.")

    # ── Write output ──────────────────────────────────────────────────────────
    out = {
        "meta": {
            "source_json":   str(IN_JSON),
            "source_jsonl":  str(IN_JSONL),
            "target_turns":  TARGET_TURNS,
            "d6_depth":      D6_DEPTH,
            "d8_depth":      D8_DEPTH,
            "scorer":        "search_root_all_scores",
            "d6_note":       "Stored from JSON where available; run live for T39, T47.",
        },
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(results)} results → {OUT_PATH}")


if __name__ == "__main__":
    main()
