#!/usr/bin/env python3
"""
audit_strength.py — Read-only strength audit for the checkers pipeline.

Reconstructs board states by replaying the full move history from the
standard initial position, then runs multi-depth search and reports
evaluator breakdowns, move_facts, and a classification for each position.

TRACE FORMAT NOTE
-----------------
The JSONL trace has no 'board' field — only move coordinates (path/captured).
Board state is recovered deterministically by replaying every prior move.
Fields present:  turn, player_who_moved, move_type, path, captured,
                 promotion, reasoning, material_advantage, winning_assessment,
                 strategic_priorities, metrics, game_over, winner, draw.
Fields absent:   board, legal_moves, minimax_scores per move,
                 symbolic_scored_moves, ranker_filtered_menu.

To enable exact-score replay without re-running the engine, add to logger_node:
  entry["board_after"]          = [[int(c) for c in row] for row in state.board]
  entry["symbolic_best_score"]  = state.symbolic_best_score
  entry["symbolic_gap"]         = state.symbolic_gap

Modes
-----
position
    Audit specific turns. For each turn, reconstructs the board BEFORE
    the move was made, runs D6 + D8 (and optional D10) search over ALL
    legal moves, shows evaluator breakdown and move_facts for the top 3
    moves, and emits a classification.

trajectory
    Audit a contiguous window of turns. For each turn, shows chosen move
    vs. D6 best, material before/after, king counts, promotion distances,
    and flags the first turn where RED's advantage starts declining.
    Also detects board-hash cycling within the window.

Usage
-----
    python3 audit_strength.py \\
        --trace logs/game_20260427_091537_540232.jsonl \\
        --mode position --turns 35,37

    python3 audit_strength.py \\
        --trace logs/game_20260427_091537_540232.jsonl \\
        --mode position --turns 35,37 --d10

    python3 audit_strength.py \\
        --trace logs/game_20260427_091537_540232.jsonl \\
        --mode trajectory --turns 83,84,85,86,87,88,89

    python3 audit_strength.py \\
        --trace logs/game_20260427_091537_540232.jsonl \\
        --mode trajectory --turns 30,31,32,33,34,35,36,37,38,39,40

    python3 audit_strength.py \\
        --trace logs/game_20260427_091537_540232.jsonl \\
        --mode position --turns 35 --depths 6,7,8,9,10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from checkers.engine.board import (
    BLACK, BLACK_KING, BOARD_SIZE, EMPTY, RED, RED_KING,
    create_initial_board,
)
from checkers.engine.evaluation import evaluate_board_breakdown
from checkers.engine.move_facts import compute_move_facts
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.engine.zobrist import compute_hash
from checkers.search.minimax_core import (
    clear_transposition_table,
    search_root_all_scores,
)

# ── move_facts keys surfaced in the report ────────────────────────────────────
_FACTS_KEYS = [
    "opponent_can_recapture", "results_in_king", "near_promotion",
    "captures_count", "net_gain",
    "mobility_reduction", "opponent_mobility_after",
    "creates_immediate_threat", "shot_sequence_available", "forces_exchange",
    "restriction_score", "counterplay_score",
    "winning_conversion_score", "simplification_value",
    "king_activity_score", "king_distance_pressure",
    "unsafe_simple_move", "quiet_move_role",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def normalize_path(path) -> tuple:
    """Convert any path format (list-of-lists, list-of-tuples, etc.) to tuple-of-tuples of ints."""
    return tuple(tuple(int(x) for x in step) for step in path)


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f]


def _counts(board: list[list[int]]) -> dict[str, int]:
    rm = rk = bm = bk = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if   p == RED:       rm += 1
            elif p == RED_KING:  rk += 1
            elif p == BLACK:     bm += 1
            elif p == BLACK_KING: bk += 1
    return {
        "red_men": rm, "red_kings": rk,
        "black_men": bm, "black_kings": bk,
        "red_total": rm + rk, "black_total": bm + bk,
        "total": rm + rk + bm + bk,
    }


def _promo_dist(board: list[list[int]], player: int) -> int:
    """Minimum rows-to-promotion for any man of `player`. 99 if none."""
    best = 99
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if player == RED and p == RED:
                best = min(best, r)           # promotes at row 0
            elif player == BLACK and p == BLACK:
                best = min(best, 7 - r)       # promotes at row 7
    return best


def reconstruct(records: list[dict], before_turn: int) -> tuple[list[list[int]], int]:
    """
    Apply all moves with turn < before_turn and return (board, player_to_move).
    """
    board = create_initial_board()
    for rec in records:
        if rec["turn"] >= before_turn:
            break
        move = {
            "type": rec["move_type"],
            "path": rec["path"],
            "captured": rec["captured"],
        }
        board = apply_move(board, move)
    player = next(r["player_who_moved"] for r in records if r["turn"] == before_turn)
    return board, player


def fmt_move(mv: dict) -> str:
    path = mv.get("path", [])
    cap  = mv.get("captured", [])
    s = "→".join(f"({r},{c})" for r, c in path)
    return s + (f" x{len(cap)}" if cap else "")


def _player_label(p: int) -> str:
    return "RED" if p == RED else "BLACK"


def score_depth(
    board: list[list[int]],
    player: int,
    depth: int,
) -> list[tuple[dict[str, Any], float]]:
    """
    Score all legal moves at `depth` with a fresh TT.
    Returns [(move, score), ...] sorted descending.
    """
    clear_transposition_table()
    _, _, scored, _ = search_root_all_scores(
        board=board,
        current_player=player,
        depth=depth,
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )
    return scored


def best_path(scored: list[tuple]) -> tuple | None:
    return normalize_path(scored[0][0]["path"]) if scored else None


def best_paths(scored: list[tuple]) -> list[tuple]:
    """Return all moves tied for the best score."""
    if not scored:
        return []
    top_score = scored[0][1]
    return [normalize_path(mv["path"]) for mv, sc in scored if sc == top_score]


def _find_scored(scored: list[tuple], path) -> float | None:
    needle = normalize_path(path)
    for mv, sc in scored:
        if normalize_path(mv["path"]) == needle:
            return sc
    return None


# ── classification ────────────────────────────────────────────────────────────

def classify(
    chosen_path,
    d6: list[tuple],
    d8: list[tuple],
    counts: dict,
    cycling: bool,
) -> tuple[str, str]:
    bp6 = best_path(d6)
    bp8 = best_path(d8) if d8 else bp6
    chosen_norm = normalize_path(chosen_path) if chosen_path else None

    if cycling:
        return (
            "KING_ENDGAME_PLAN_MISSING",
            "Board hash repeated within the audit window — engine is cycling "
            "(no progress; win-distance scoring and/or repetition penalty needed)",
        )

    if bp6 is not None and bp8 is not None and bp6 != bp8:
        return (
            "SEARCH_HORIZON_SUSPECT",
            f"D6 best {bp6} (score {d6[0][1]:.1f}) ≠ "
            f"D8 best {bp8} (score {d8[0][1]:.1f}) — "
            "deeper search reverses the ranking; D6 horizon is insufficient",
        )

    if chosen_norm and bp6:
        cobests6 = best_paths(d6)
        if chosen_norm in cobests6:
            if len(cobests6) > 1:
                return (
                    "NEAR_TIE",
                    f"Chosen {chosen_norm} is one of {len(cobests6)} co-best moves "
                    f"all scoring {d6[0][1]:.1f} — no error",
                )
            return ("NO_ISSUE", "Chosen equals unique D6 best; no anomaly detected")
        # chosen is genuinely not among the best
        gap = abs(d6[0][1] - (_find_scored(d6, chosen_path) or d6[0][1]))
        if gap <= 2.0:
            return (
                "NEAR_TIE",
                f"Chosen {chosen_norm} differs from D6 best {bp6} by only {gap:.1f} pts",
            )
        return (
            "RANKER_SUSPECT",
            f"Chosen {chosen_norm} ≠ D6/D8 best {bp6} (gap={gap:.1f}) — "
            "LLM or override deviated from the symbolic best",
        )

    all_kings = counts["red_men"] == 0 and counts["black_men"] == 0
    if all_kings and counts["total"] <= 6:
        return (
            "KING_ENDGAME_PLAN_MISSING",
            "Pure king endgame; flat WIN_SCORE means no preference for faster win",
        )

    return ("NO_ISSUE", "D6 = D8 = chosen; no anomaly detected at this depth pair")


# ── position audit ────────────────────────────────────────────────────────────

def run_position_audit(
    records: list[dict],
    turns: list[int],
    run_d10: bool,
) -> None:
    turn_index = {r["turn"]: r for r in records}
    for turn in turns:
        if turn not in turn_index:
            print(f"[SKIP] Turn {turn} not found in trace.\n")
            continue

        rec = turn_index[turn]
        board, player = reconstruct(records, turn)
        counts = _counts(board)
        legal = get_all_legal_moves(board, player)
        chosen_path = rec["path"]

        print("=" * 70)
        print(f"POSITION AUDIT  Turn {turn}  {_player_label(player)} to move")
        print(f"  Pieces: RED {counts['red_men']}m + {counts['red_kings']}K  "
              f"BLACK {counts['black_men']}m + {counts['black_kings']}K  "
              f"total={counts['total']}")
        print(f"  Trace chosen: {chosen_path}  (reasoning: {rec['reasoning'][:80]}…)")
        print(f"  material_advantage (after move): {rec['material_advantage']}")
        print(f"  winning_assessment:  {rec['winning_assessment']}")
        print(f"  Legal moves: {len(legal)}")
        print()

        # ── D6 ────────────────────────────────────────────────────────────────
        print(f"  Scoring D6 ({len(legal)} moves) …", flush=True)
        d6 = score_depth(board, player, 6)

        # ── D8 ────────────────────────────────────────────────────────────────
        print(f"  Scoring D8 ({len(legal)} moves) …", flush=True)
        d8 = score_depth(board, player, 8)

        # ── D10 (optional) ───────────────────────────────────────────────────
        d10: list[tuple] = []
        if run_d10:
            print(f"  Scoring D10 ({len(legal)} moves) — may take ~30s …", flush=True)
            d10 = score_depth(board, player, 10)

        # ── Ranking table ─────────────────────────────────────────────────────
        print()
        header = f"  {'#':>3}  {'Move':<28} {'D6':>9} {'D8':>9}"
        if d10:
            header += f" {'D10':>9}"
        header += "  CHOSEN"
        print(header)
        print("  " + "-" * (len(header) - 2))

        d6_paths = [mv["path"] for mv, _ in d6]
        d8_paths = [mv["path"] for mv, _ in d8]

        for rank, (mv, sc6) in enumerate(d6, 1):
            sc8 = _find_scored(d8, mv["path"])
            sc10 = _find_scored(d10, mv["path"]) if d10 else None
            chosen_flag = "  ← CHOSEN" if mv["path"] == chosen_path else ""
            row = (f"  {rank:>3}  {fmt_move(mv):<28} {sc6:>9.1f} "
                   f"{sc8:>9.1f}" if sc8 is not None else
                   f"  {rank:>3}  {fmt_move(mv):<28} {sc6:>9.1f} {'?':>9}")
            if sc8 is not None:
                row = f"  {rank:>3}  {fmt_move(mv):<28} {sc6:>9.1f} {sc8:>9.1f}"
            if d10:
                row += f" {sc10:>9.1f}" if sc10 is not None else f" {'?':>9}"
            row += chosen_flag
            print(row)
            if rank >= 12:
                print(f"  … ({len(d6) - 12} more)")
                break

        bp6 = best_path(d6)
        bp8 = best_path(d8)
        bp10 = best_path(d10) if d10 else None
        cobests6 = best_paths(d6)
        cobests8 = best_paths(d8) if d8 else []
        chosen_norm = normalize_path(chosen_path)
        print()
        if len(cobests6) > 1:
            print(f"  D6 co-best moves ({len(cobests6)}, all score {d6[0][1]:.1f}):")
            for p in cobests6:
                marker = "  ← CHOSEN" if p == chosen_norm else ""
                print(f"    {p}{marker}")
        else:
            print(f"  D6 best: {bp6}{'  ← CHOSEN' if bp6 == chosen_norm else ''}")
        print(f"  D8 best: {bp8}  {'← DIFFERS FROM D6' if bp6 != bp8 else '(same as D6)'}")
        if d10:
            print(f"  D10 best: {bp10}  {'← DIFFERS FROM D8' if bp10 != bp8 else '(same as D8)'}")

        # ── Evaluator breakdown for top-3 moves ───────────────────────────────
        print()
        print("  EVALUATOR BREAKDOWN  (board-after-move, root=player perspective)")
        for idx, (mv, sc6) in enumerate(d6[:3], 1):
            board_after = apply_move(board, mv)
            bd = evaluate_board_breakdown(board_after, player, player)
            non_zero = {k: v for k, v in bd.items() if v != 0.0}
            print(f"    Move #{idx} {fmt_move(mv)}  D6={sc6:.1f}")
            for k, v in non_zero.items():
                print(f"      {k:<35} {v:>10.1f}")

        # ── move_facts for top-3 moves ────────────────────────────────────────
        print()
        print("  MOVE FACTS  (for top-3 D6 moves)")
        for idx, (mv, sc6) in enumerate(d6[:3], 1):
            facts = compute_move_facts(board, mv, player)
            print(f"    Move #{idx} {fmt_move(mv)}")
            for k in _FACTS_KEYS:
                v = facts.get(k, "—")
                print(f"      {k:<35} {str(v):>12}")

        # ── Classification ────────────────────────────────────────────────────
        label, reason = classify(chosen_path, d6, d8, counts, cycling=False)
        print()
        print(f"  CLASSIFICATION: {label}")
        print(f"  Reason: {reason}")
        print()


# ── depth sweep ──────────────────────────────────────────────────────────────

_COBEST_MARGIN = 2.0   # points within best score to qualify as "co-best"


def run_depth_sweep(
    records: list[dict],
    turns: list[int],
    depths: list[int],
) -> None:
    """
    For each turn, score all legal moves at every depth in `depths` and report:
      - best move + score at each depth
      - full ranked list with gap from best
      - whether the best move changed from the previous depth
      - co-best moves within _COBEST_MARGIN points of the best
    Goal: diagnose odd/even depth instability vs genuine deeper-search correction.
    """
    turn_index = {r["turn"]: r for r in records}
    for turn in turns:
        if turn not in turn_index:
            print(f"[SKIP] Turn {turn} not found in trace.\n")
            continue

        rec = turn_index[turn]
        board, player = reconstruct(records, turn)
        counts = _counts(board)
        legal = get_all_legal_moves(board, player)
        chosen_norm = normalize_path(rec["path"])

        print("=" * 72)
        print(f"DEPTH SWEEP  Turn {turn}  {_player_label(player)} to move  "
              f"depths={depths}")
        print(f"  Pieces: RED {counts['red_men']}m + {counts['red_kings']}K  "
              f"BLACK {counts['black_men']}m + {counts['black_kings']}K  "
              f"total={counts['total']}")
        print(f"  Trace chosen: {rec['path']}")
        print(f"  Legal moves: {len(legal)}")
        print()

        # ── Run all depths and collect results ────────────────────────────────
        depth_results: list[tuple[int, list[tuple[dict, float]]]] = []
        for depth in depths:
            print(f"  Scoring D{depth} ({len(legal)} moves) …", flush=True)
            scored = score_depth(board, player, depth)
            depth_results.append((depth, scored))

        # ── Per-depth analysis ────────────────────────────────────────────────
        print()
        prev_best_p: tuple | None = None
        prev_depth: int | None = None
        for depth, scored in depth_results:
            if not scored:
                print(f"\n  ── D{depth}: no results")
                continue

            best_mv, best_sc = scored[0]
            best_p = best_path(scored)
            cobests = [(mv, sc) for mv, sc in scored
                       if abs(best_sc - sc) <= _COBEST_MARGIN]
            changed = prev_best_p is not None and best_p != prev_best_p
            change_tag = (f"  [CHANGED from D{prev_depth}]" if changed else "")

            print(f"\n  ── D{depth}  best: {fmt_move(best_mv)}  score={best_sc:.1f}{change_tag}")

            # Co-bests
            if len(cobests) > 1:
                print(f"     Co-best (within {_COBEST_MARGIN:.0f}pt): {len(cobests)} moves")
                for mv, sc in cobests:
                    p = normalize_path(mv["path"])
                    gap = best_sc - sc
                    chosen_flag = "  ← CHOSEN" if p == chosen_norm else ""
                    print(f"       {fmt_move(mv):<30}  score={sc:.1f}  gap={gap:.1f}{chosen_flag}")
            else:
                chosen_flag = ("  ← CHOSEN"
                               if best_p == chosen_norm else
                               "  (chosen differs)")
                print(f"     Unique best (no co-best within {_COBEST_MARGIN:.0f}pt){chosen_flag}")

            # Full ranking
            print(f"     Full ranking at D{depth}:")
            print(f"     {'#':>3}  {'Move':<30}  {'Score':>9}  {'Gap':>7}  Notes")
            print("     " + "-" * 62)
            for rank, (mv, sc) in enumerate(scored, 1):
                p = normalize_path(mv["path"])
                gap = best_sc - sc
                notes = []
                if p == chosen_norm:
                    notes.append("CHOSEN")
                if abs(best_sc - sc) <= _COBEST_MARGIN and rank > 1:
                    notes.append("co-best")
                note_s = ", ".join(notes)
                print(f"     {rank:>3}  {fmt_move(mv):<30}  {sc:>9.1f}  {gap:>7.1f}  {note_s}")
                if rank >= 20:
                    print(f"     … ({len(scored) - 20} more moves not shown)")
                    break

            prev_best_p = best_p
            prev_depth = depth

        # ── Stability summary table ───────────────────────────────────────────
        print()
        print(f"  STABILITY SUMMARY  Turn {turn}")
        print(f"  {'Depth':>6}  {'Best Move':<32}  {'Score':>9}  {'Changed':>8}  "
              f"{'#co-best':>8}")
        print("  " + "-" * 72)
        prev_best_p = None
        for depth, scored in depth_results:
            if not scored:
                print(f"  {depth:>6}  (no results)")
                continue
            best_mv, best_sc = scored[0]
            best_p = best_path(scored)
            n_cobest = sum(1 for _, sc in scored if abs(best_sc - sc) <= _COBEST_MARGIN)
            changed_s = "YES" if prev_best_p is not None and best_p != prev_best_p else "no"
            chosen_here = "  ← CHOSEN" if best_p == chosen_norm else ""
            print(f"  {depth:>6}  {fmt_move(best_mv):<32}  {best_sc:>9.1f}  "
                  f"{changed_s:>8}  {n_cobest:>8}{chosen_here}")
            prev_best_p = best_p

        print()


# ── trajectory audit ──────────────────────────────────────────────────────────

def run_trajectory_audit(
    records: list[dict],
    turns: list[int],
) -> None:
    turn_index = {r["turn"]: r for r in records}
    missing = [t for t in turns if t not in turn_index]
    if missing:
        print(f"[WARN] Turns not in trace: {missing}  — skipping them.\n")

    valid_turns = [t for t in turns if t in turn_index]
    if not valid_turns:
        print("No valid turns to audit.")
        return

    # Reconstruct board states and compute hashes ONCE in order.
    states: list[dict] = []
    seen_hashes: dict[int, int] = {}   # hash → first turn it appeared
    cycles_detected: list[str] = []

    # We need the board state BEFORE each turn's move.
    # Pre-build: board after each turn up to max(valid_turns).
    max_turn = max(valid_turns)
    board_snapshots: dict[int, list[list[int]]] = {}
    board = create_initial_board()
    board_snapshots[0] = [row[:] for row in board]
    for rec in records:
        if rec["turn"] > max_turn:
            break
        board_snapshots[rec["turn"]] = [row[:] for row in board]   # BEFORE this turn's move
        mv = {"type": rec["move_type"], "path": rec["path"], "captured": rec["captured"]}
        board = apply_move(board, mv)

    for turn in valid_turns:
        rec = turn_index[turn]
        board_before = board_snapshots[turn]
        board_after_rec = apply_move(
            board_before,
            {"type": rec["move_type"], "path": rec["path"], "captured": rec["captured"]},
        )
        cb = _counts(board_before)
        ca = _counts(board_after_rec)
        h_before = compute_hash(board_before)
        player = rec["player_who_moved"]
        promo_red_before = _promo_dist(board_before, RED)
        promo_black_before = _promo_dist(board_before, BLACK)
        promo_red_after = _promo_dist(board_after_rec, RED)
        promo_black_after = _promo_dist(board_after_rec, BLACK)

        if h_before in seen_hashes:
            cycles_detected.append(
                f"Turn {turn}: board-before hash={h_before} "
                f"already seen at turn {seen_hashes[h_before]}"
            )
        else:
            seen_hashes[h_before] = turn

        # D6 search only for trajectory (D8 would be slow over many turns)
        if player == RED:
            print(f"  Scoring D6 for Turn {turn} (RED) …", flush=True)
            d6 = score_depth(board_before, player, 6)
            bp6 = best_path(d6)
            d6_score = d6[0][1] if d6 else None
            chosen_score = _find_scored(d6, rec["path"])
            match = "YES" if bp6 == normalize_path(rec["path"]) else "NO"
        else:
            d6 = []
            bp6 = None
            d6_score = None
            chosen_score = None
            match = "—"

        states.append({
            "turn": turn,
            "player": player,
            "chosen": rec["path"],
            "d6_best": bp6,
            "d6_best_score": d6_score,
            "chosen_d6_score": chosen_score,
            "match": match,
            "mat_before": rec["material_advantage"],  # from trace (after move)
            "red_total_before": cb["red_total"],
            "black_total_before": cb["black_total"],
            "red_men_before": cb["red_men"],
            "red_kings_before": cb["red_kings"],
            "black_men_before": cb["black_men"],
            "black_kings_before": cb["black_kings"],
            "promo_red_before": promo_red_before,
            "promo_black_before": promo_black_before,
            "promo_red_after": promo_red_after,
            "promo_black_after": promo_black_after,
            "reasoning": rec["reasoning"][:80],
            "winning_assessment": rec["winning_assessment"],
        })

    # ── Print trajectory table ─────────────────────────────────────────────
    print("=" * 90)
    print(f"TRAJECTORY AUDIT  Turns {valid_turns[0]}–{valid_turns[-1]}")
    print()
    print(f"  {'T':>4}  {'Pl':<5} {'Chosen':<24} {'D6 best':<24} "
          f"{'Match':>5}  {'D6sc':>7}  {'Csc':>7}  {'Mat':>4}  {'Rd':>4}  {'Bk':>4}")
    print("  " + "-" * 88)

    advantage_declining_turn: int | None = None
    prev_mat: int | None = None

    for s in states:
        p_label = "RED" if s["player"] == RED else "blk"
        chosen_s = str(s["chosen"])[:23]
        d6_s = str(s["d6_best"])[:23] if s["d6_best"] else "—"
        d6sc_s = f"{s['d6_best_score']:.1f}" if s["d6_best_score"] is not None else "—"
        csc_s = f"{s['chosen_d6_score']:.1f}" if s["chosen_d6_score"] is not None else "—"
        mat = s["mat_before"]

        if s["player"] == RED and prev_mat is not None and mat < prev_mat:
            if advantage_declining_turn is None:
                advantage_declining_turn = s["turn"]
        if s["player"] == RED:
            prev_mat = mat

        cobests = best_paths(score_depth(board_snapshots[s["turn"]], s["player"], 6)) if False else []
        flag = "  ← D6≠CHOSEN" if s["match"] == "NO" and s["player"] == RED else ""
        print(f"  {s['turn']:>4}  {p_label:<5} {chosen_s:<24} {d6_s:<24} "
              f"{s['match']:>5}  {d6sc_s:>7}  {csc_s:>7}  {mat:>4}  "
              f"{s['red_total_before']:>4}  {s['black_total_before']:>4}{flag}")

    # ── Strategic progress summary ────────────────────────────────────────
    print()
    print("  STRATEGIC PROGRESS SUMMARY")

    if advantage_declining_turn:
        print(f"  First turn where RED material advantage declined: {advantage_declining_turn}")
    else:
        print(f"  RED material advantage did not decline within this window.")

    # Promotion distance trend
    red_promo = [(s["turn"], s["promo_red_before"]) for s in states if s["player"] == RED]
    blk_promo = [(s["turn"], s["promo_black_before"]) for s in states if s["player"] == RED]
    if red_promo:
        first_t, first_d = red_promo[0]
        last_t,  last_d  = red_promo[-1]
        delta = last_d - first_d
        print(f"  RED  promo-dist (rows-to-crown): {first_d} at T{first_t} → {last_d} at T{last_t}  "
              f"(Δ={delta:+d}; negative = getting closer)")
    if blk_promo:
        first_t, first_d = blk_promo[0]
        last_t,  last_d  = blk_promo[-1]
        delta = last_d - first_d
        print(f"  BLACK promo-dist (rows-to-crown): {first_d} at T{first_t} → {last_d} at T{last_t}  "
              f"(Δ={delta:+d})")

    # King count trajectory
    red_kings  = [(s["turn"], s["red_kings_before"])  for s in states if s["player"] == RED]
    blk_kings  = [(s["turn"], s["black_kings_before"]) for s in states if s["player"] == RED]
    if red_kings:
        k_vals = ", ".join(f"T{t}:{k}" for t, k in red_kings)
        print(f"  RED  king count:   {k_vals}")
    if blk_kings:
        k_vals = ", ".join(f"T{t}:{k}" for t, k in blk_kings)
        print(f"  BLACK king count:  {k_vals}")

    # Cycling
    print()
    if cycles_detected:
        print(f"  CYCLING DETECTED ({len(cycles_detected)} repeat(s)):")
        for c in cycles_detected:
            print(f"    {c}")
        label, reason = classify(None, [], [], {}, cycling=True)
        print(f"  CLASSIFICATION: {label}")
        print(f"  Reason: {reason}")
    else:
        print("  No board-state cycling detected in this window.")

    # D6 match rate for RED turns
    red_states = [s for s in states if s["player"] == RED]
    if red_states:
        match_count = sum(1 for s in red_states if s["match"] == "YES")
        print(f"  RED D6 agreement: {match_count}/{len(red_states)} turns chosen = D6 best")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read-only strength audit tool for the checkers pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--trace", required=True, help="Path to the .jsonl trace file")
    ap.add_argument(
        "--mode", required=True, choices=["position", "trajectory"],
        help="'position': per-turn D6/D8 deep-dive; 'trajectory': window trend analysis",
    )
    ap.add_argument(
        "--turns", required=True,
        help="Comma-separated turn numbers, e.g. 35,37 or 83,84,85,86,87,88,89",
    )
    ap.add_argument(
        "--d10", action="store_true",
        help="[position mode] Also run D10 search (~30 s/position). Off by default.",
    )
    ap.add_argument(
        "--depths",
        default=None,
        help=(
            "[position mode] Comma-separated depths for a depth sweep, e.g. 6,7,8,9,10. "
            "When provided, replaces the standard D6/D8/D10 analysis with a per-depth "
            "ranked table showing best move stability and co-bests within 2 pt."
        ),
    )
    args = ap.parse_args()

    turns = [int(t.strip()) for t in args.turns.split(",")]
    records = load_trace(args.trace)

    # Quick trace summary
    all_turns = [r["turn"] for r in records]
    print(f"Trace: {args.trace}")
    print(f"  Records: {len(records)}  turns {min(all_turns)}–{max(all_turns)}")
    print(f"  Fields: {sorted(records[0].keys())}")
    print(f"  Board field present: NO  (reconstructed by move replay)")
    print()

    if args.mode == "position":
        if args.depths is not None:
            depths = [int(d.strip()) for d in args.depths.split(",")]
            run_depth_sweep(records, turns, depths)
        else:
            run_position_audit(records, turns, run_d10=args.d10)
    else:
        run_trajectory_audit(records, turns)


if __name__ == "__main__":
    main()
