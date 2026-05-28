#!/usr/bin/env python3
"""
benchmark_dataset_generator.py
───────────────────────────────
Offline benchmark dataset generator.

Replays every PDN game/problem → filters non-trivial positions →
scores every legal move with KingsRow → writes a JSONL dataset for
benchmark_evaluator.py.

Key design decisions
  • Reuses the SAME proven KingsRow runtime path as
    run_kingsrow_benchmark_trace.py:
        for each of OUR legal moves:
            apply move → call KR on the resulting position from the
            opponent's perspective → negate the score to get
            side-to-move's score.
        Sort moves descending; KR's best = highest scored.
    KR is used purely as a position SCORER; the kr_path written into the
    dataset is taken directly from our own legal-move list, so no
    coordinate canonicalisation is required.
  • Board stored as a raw list-of-lists so the evaluator needs no FEN round-trip.
  • Position categories stored as boolean flags for category-sliced evaluation.

Usage
    python benchmark_dataset_generator.py [OPTIONS]

Options
    --output PATH         JSONL output path             (default: benchmark_positions.jsonl)
    --kr-depth N          KingsRow fixed search depth   (default: 6)
    --min-branching N     The ONLY triviality filter.   (default: 2)
                          Drops positions with fewer than N legal moves
                          (so single-move forced lines are filtered;
                          multi-jump-choice positions are KEPT — they are
                          still tactical decision points).
    --phase PHASE         all|opening|midgame|endgame   (default: all)
    --categories LIST     Comma-separated subset of     (default: all)
                          quiet,tactical,multi_jump
                          (alias: --category)
    --skip-existing       Reuse KR cache: skip (fen,turn) already in --output;
                          new rows are APPENDED instead of overwriting.
    --max-positions N     Cap on positions WRITTEN this run; 0 = unlimited.
    --deduplicate         Within-run FEN dedup (independent of --skip-existing).
    --kr-disable          Skip KR calls — writes position metadata only.
    --kr-dll PATH         Explicit path to Kingsrow64.dll.
    --quiet               Suppress per-position output.

Note
    `is_forced_capture` is recorded as informational metadata only — it
    flags positions where every legal move is a jump (per checkers rules).
    It is NOT a filter, because positions with 2+ legal jumps are still
    meaningful tactical choices.

Examples
    python benchmark_dataset_generator.py --phase midgame
    python benchmark_dataset_generator.py --category tactical
    python benchmark_dataset_generator.py --phase midgame --category tactical
    python benchmark_dataset_generator.py --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from checkers.engine.board import RED, BLACK, EMPTY, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.move_facts import count_pieces
from checkers.data.pdn_importer.pdn_parser import parse_pdn_file
from checkers.data.pdn_importer.fen_utils import (
    rowcol_to_square,
    str_to_side,
    side_to_str,
)

_HERE          = Path(__file__).parent
RAW_PDN_DIR    = _HERE / "checkers" / "data" / "raw_pdn" / "bob_newell"
DEFAULT_OUTPUT = _HERE / "benchmark_positions.jsonl"
DEFAULT_KR_DLL = r"C:\Program Files (x86)\CheckerBoard\engines\Kingsrow64.dll"


# ─── helpers ───────────────────────────────────────────────────────────────────

def _board_to_fen(board: list, side: int) -> str:
    side_char = "B" if side == BLACK else "W"
    black_parts: list = []
    red_parts:   list = []
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == EMPTY:
                continue
            try:
                sq = rowcol_to_square(r, c)
            except ValueError:
                continue
            if   piece == BLACK:      black_parts.append((sq, str(sq)))
            elif piece == BLACK_KING: black_parts.append((sq, f"K{sq}"))
            elif piece == RED:        red_parts.append((sq, str(sq)))
            elif piece == RED_KING:   red_parts.append((sq, f"K{sq}"))
    black_parts.sort()
    red_parts.sort()
    b_str = "B" + ",".join(p for _, p in black_parts)
    w_str = "W" + ",".join(p for _, p in red_parts)
    return f"{side_char}:{w_str}:{b_str}"


def _classify_phase(board: list) -> str:
    total = count_pieces(board, RED)["total"] + count_pieces(board, BLACK)["total"]
    if total >= 18: return "opening"
    if total >= 10: return "midgame"
    return "endgame"


def _classify_position(legal_moves: list) -> dict:
    """
    Compute boolean category flags for a set of legal moves.

    Note on `is_forced_capture`: this is the rules-driven label "every legal
    move is a jump" — it is NOT a triviality filter. A position with 2+
    capture options is still a meaningful tactical decision. Triviality
    (single legal move) is controlled separately by --min-branching.
    """
    has_capture  = any(m["type"] == "jump" for m in legal_moves)
    multi_jump   = any(
        m["type"] == "jump" and len(m.get("path", [])) >= 3
        for m in legal_moves
    )
    all_captures = has_capture and all(m["type"] == "jump" for m in legal_moves)
    return {
        "is_quiet":          not has_capture,
        "is_tactical":       has_capture,
        "is_forced_capture": all_captures,    # informational only
        "is_multi_jump":     multi_jump,
    }


def _norm_path(path) -> list:
    """Normalise path to list-of-[r,c] lists for comparison."""
    return [[int(sq[0]), int(sq[1])] for sq in (path or [])]


def _kr_rank_legal_moves(
    kr_engine,
    board: list,
    side: int,
    legal_moves: list,
    kr_time: float,
    kr_depth: int,
) -> list:
    """
    Score every legal move with KingsRow, using the SAME runtime path as
    run_kingsrow_benchmark_trace.py:

        for each of OUR legal moves:
            test_board = apply_move(board, lm)              # post-move position
            kr = kr_engine.get_best_move(test_board,
                                         opponent,
                                         kr_time, kr_depth) # opponent on move
            score_for_side = -kr["score"]                   # flip to side's view

    Returns a list of dicts: [{"move", "kr_score", "kr_depth"}, ...]
    sorted descending by kr_score (best for *side* first).
    """
    opponent = BLACK if side == RED else RED
    kr_evals: list = []
    for lm in legal_moves:
        # apply_move returns a NEW board (does not mutate); use the return value
        # so KR actually sees the post-move position.
        test_board = apply_move(board, lm)
        kr_after = kr_engine.get_best_move(
            test_board, opponent, kr_time, kr_depth,
        )
        kr_evals.append({
            "move":     lm,
            "kr_score": -kr_after["score"],
            "kr_depth": kr_after["depth"],
        })
    kr_evals.sort(key=lambda e: e["kr_score"], reverse=True)
    return kr_evals


# ─── main generator ────────────────────────────────────────────────────────────

_ALL_CATEGORIES = {"quiet", "tactical", "multi_jump"}


def _parse_categories(s: str) -> set[str]:
    """Parse comma-separated category list; 'all' → every category."""
    if not s or s.strip().lower() == "all":
        return set(_ALL_CATEGORIES)
    cats = {c.strip().lower() for c in s.split(",") if c.strip()}
    invalid = cats - _ALL_CATEGORIES
    if invalid:
        raise SystemExit(
            f"[generator] Invalid --categories value(s): {sorted(invalid)}. "
            f"Allowed: {sorted(_ALL_CATEGORIES)} or 'all'."
        )
    return cats


def _load_existing_keys(output_path: Path) -> set[tuple[str, str]]:
    """Return set of (fen, turn) already present in output_path JSONL."""
    seen: set[tuple[str, str]] = set()
    if not output_path.exists():
        return seen
    with open(output_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fen, turn = r.get("fen"), r.get("turn")
            if fen and turn:
                seen.add((fen, turn))
    return seen


def generate_benchmark(
    output_path:    Path,
    kr_engine,
    kr_depth:       int,
    min_branching:  int,
    phase_filter:   str,
    selected_cats:  set[str],
    skip_existing:  bool,
    max_positions:  int,
    deduplicate:    bool,
    quiet:          bool,
) -> dict:
    pdn_files = sorted(RAW_PDN_DIR.glob("*.pdn"))
    if not pdn_files:
        print(f"[generator] ERROR: no PDN files in {RAW_PDN_DIR}", file=sys.stderr)
        return {}

    print(f"[generator] {len(pdn_files)} PDN file(s):")
    for f in pdn_files:
        print(f"    {f.name}")
    print()

    stats: dict = {
        "total_raw":       0,
        "skip_terminal":   0,
        "skip_branching":  0,
        "skip_phase":      0,
        "skip_category":   0,
        "skip_dup":        0,
        "skip_cached":     0,
        "kr_errors":       0,
        "accepted":        0,
        "phase_opening":   0,
        "phase_midgame":   0,
        "phase_endgame":   0,
        "cat_quiet":       0,
        "cat_tactical":    0,
        "cat_multi_jump":  0,
    }
    seen_fens: set[str] = set()
    written   = 0

    # --skip-existing: pre-load (fen, turn) keys already evaluated in prior runs.
    # In that mode we APPEND to the existing JSONL instead of overwriting.
    existing_keys: set[tuple[str, str]] = set()
    if skip_existing:
        existing_keys = _load_existing_keys(output_path)
        print(
            f"[generator] --skip-existing: {len(existing_keys)} cached (fen,turn) "
            f"keys loaded from {output_path}; appending new rows."
        )

    output_mode = "a" if skip_existing else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with open(output_path, output_mode, encoding="utf-8") as out_f:
        for pdn_file in pdn_files:
            if not quiet:
                print(f"[generator] Parsing {pdn_file.name} ...", end=" ", flush=True)
            raw_positions = parse_pdn_file(str(pdn_file))
            if not quiet:
                print(f"{len(raw_positions)} raw positions")

            for pos in raw_positions:
                if max_positions > 0 and written >= max_positions:
                    break

                stats["total_raw"] += 1
                board = pos["board"]
                side  = str_to_side(pos["side_to_move"])
                legal = get_all_legal_moves(board, side)

                # ── position filters ──────────────────────────────────────────
                if len(legal) == 0:
                    stats["skip_terminal"] += 1
                    continue
                if len(legal) < min_branching:
                    stats["skip_branching"] += 1
                    continue

                phase = _classify_phase(board)
                if phase_filter != "all" and phase != phase_filter:
                    stats["skip_phase"] += 1
                    continue

                # Category classification for the --categories filter.
                # NOTE: is_forced_capture is informational only — multi-jump
                # forced positions are valid tactical decisions and must NOT
                # be filtered here. Triviality (single legal move) is owned by
                # --min-branching above.
                cats     = _classify_position(legal)
                turn_str = side_to_str(side)
                fen      = _board_to_fen(board, side)

                # --categories: keep only positions matching one of the requested
                # buckets (quiet / tactical / multi_jump). A position can satisfy
                # multiple buckets simultaneously (e.g. tactical + multi_jump).
                if not (
                    ("quiet"      in selected_cats and cats["is_quiet"])
                    or ("tactical"   in selected_cats and cats["is_tactical"])
                    or ("multi_jump" in selected_cats and cats["is_multi_jump"])
                ):
                    stats["skip_category"] += 1
                    continue

                # --skip-existing: if (fen, turn) already in prior dataset,
                # skip KR entirely (this is the reusable-KR cache).
                key = (fen, turn_str)
                if skip_existing and key in existing_keys:
                    stats["skip_cached"] += 1
                    continue

                # --deduplicate: dedup within this run by fen alone (legacy).
                if deduplicate:
                    if fen in seen_fens:
                        stats["skip_dup"] += 1
                        continue
                    seen_fens.add(fen)

                rc = count_pieces(board, RED)
                bc = count_pieces(board, BLACK)

                # ── KingsRow: score every legal move (proven runtime path) ────
                kr_path:         list | None = None
                kr_score:        float | None = None
                kr_depth_actual: int | None   = None
                kr_path_found                 = False

                if kr_engine is not None:
                    if not quiet:
                        print(
                            f"  [KR] game={pos['game_index']:>3} "
                            f"ply={pos['ply_index']:>2} "
                            f"n_legal={len(legal)} ...",
                            end=" ", flush=True,
                        )
                    t_kr = time.perf_counter()
                    try:
                        kr_evals = _kr_rank_legal_moves(
                            kr_engine, board, side, legal,
                            kr_time=1.0, kr_depth=kr_depth,
                        )
                        if kr_evals:
                            best            = kr_evals[0]
                            kr_path         = _norm_path(best["move"]["path"])
                            kr_score        = best["kr_score"]
                            kr_depth_actual = best["kr_depth"]
                            kr_path_found   = True

                        if not quiet:
                            elapsed_kr = time.perf_counter() - t_kr
                            print(
                                f"score={kr_score}  "
                                f"depth={kr_depth_actual}  "
                                f"ok  "
                                f"({elapsed_kr:.3f}s)"
                            )

                    except Exception as exc:
                        stats["kr_errors"] += 1
                        print(f"\n  [KR-error] {exc}", file=sys.stderr)

                # ── write record ──────────────────────────────────────────────
                record = {
                    # position
                    "fen":               fen,
                    "board":             board,          # 8×8 list-of-lists (direct use in evaluator)
                    "turn":              turn_str,
                    "n_legal":           len(legal),
                    "phase":             phase,
                    # category flags
                    "is_quiet":          cats["is_quiet"],
                    "is_tactical":       cats["is_tactical"],
                    "is_forced_capture": cats["is_forced_capture"],
                    "is_multi_jump":     cats["is_multi_jump"],
                    # KingsRow annotation
                    "kr_path":           kr_path,        # canonical [[r,c],...] or null
                    "kr_path_found":     kr_path_found,
                    "kr_score":          kr_score,
                    "kr_depth":          kr_depth_actual,
                    # provenance
                    "piece_count_red":   rc["total"],
                    "piece_count_black": bc["total"],
                    "source_file":       pos["source_file"],
                    "game_index":        pos["game_index"],
                    "ply_index":         pos["ply_index"],
                    "event":             pos.get("event", ""),
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                # Add to cache so a long --skip-existing run doesn't re-evaluate
                # a (fen, turn) we just emitted earlier in the same pass.
                existing_keys.add(key)
                written += 1

                stats["accepted"]          += 1
                stats[f"phase_{phase}"]    += 1
                if cats["is_quiet"]:      stats["cat_quiet"]      += 1
                if cats["is_tactical"]:   stats["cat_tactical"]   += 1
                if cats["is_multi_jump"]: stats["cat_multi_jump"] += 1

                if written % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = written / elapsed if elapsed > 0 else 0
                    print(f"  ... {written} positions written  ({rate:.1f}/s)")

            if max_positions > 0 and written >= max_positions:
                print(f"[generator] Cap reached ({max_positions}). Stopping.")
                break

    return stats


def _print_summary(stats: dict, output_path: Path, elapsed: float) -> None:
    n = stats.get("accepted", 0)
    print()
    print("=" * 62)
    print("  BENCHMARK DATASET GENERATION — SUMMARY")
    print("=" * 62)
    print(f"  Output           : {output_path}")
    print(f"  Elapsed          : {elapsed:.1f}s")
    print()
    print(f"  Raw positions    : {stats.get('total_raw', 0)}")
    print(f"  Skipped terminal : {stats.get('skip_terminal', 0)}")
    print(f"  Skipped branching: {stats.get('skip_branching', 0)}  "
          f"(triviality filter: len(legal) < --min-branching)")
    print(f"  Skipped phase    : {stats.get('skip_phase', 0)}")
    print(f"  Skipped category : {stats.get('skip_category', 0)}")
    print(f"  Skipped dup      : {stats.get('skip_dup', 0)}")
    print(f"  Skipped cached   : {stats.get('skip_cached', 0)}")
    print(f"  KR errors        : {stats.get('kr_errors', 0)}")
    print(f"  Accepted         : {n}")
    print()
    print(f"  Phase breakdown  :")
    print(f"    Opening        : {stats.get('phase_opening', 0)}")
    print(f"    Midgame        : {stats.get('phase_midgame', 0)}")
    print(f"    Endgame        : {stats.get('phase_endgame', 0)}")
    print()
    print(f"  Category breakdown:")
    print(f"    Quiet          : {stats.get('cat_quiet', 0)}")
    print(f"    Tactical       : {stats.get('cat_tactical', 0)}")
    print(f"    MultiJump      : {stats.get('cat_multi_jump', 0)}")
    print("=" * 62)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(
            {**stats, "output_file": str(output_path), "elapsed_s": round(elapsed, 2)},
            sf, indent=2,
        )
    print(f"  Summary JSON     : {summary_path}")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDN → filtered positions → KingsRow annotation → JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output",        default=str(DEFAULT_OUTPUT))
    parser.add_argument("--kr-depth",      type=int,   default=6)
    parser.add_argument("--min-branching", type=int,   default=2,
        help="Triviality filter: drop positions with fewer than N legal moves. "
             "Multi-jump-choice positions (2+ legal jumps) are KEPT.")
    parser.add_argument("--phase",
        choices=["all", "opening", "midgame", "endgame"], default="all")
    parser.add_argument("--categories",    "--category", dest="categories", default="all",
        help="Comma-separated subset of {quiet,tactical,multi_jump} (or 'all').")
    parser.add_argument("--skip-existing", action="store_true",
        help="Reuse KR cache: skip (fen,turn) already in --output; append new rows.")
    parser.add_argument("--max-positions", type=int,   default=0)
    parser.add_argument("--deduplicate",   action="store_true",
        help="Within-run FEN-only dedup (independent of --skip-existing).")
    parser.add_argument("--kr-disable",    action="store_true")
    parser.add_argument("--kr-dll",        default=None)
    parser.add_argument("--quiet",         action="store_true")
    args = parser.parse_args()

    selected_cats = _parse_categories(args.categories)

    kr_engine = None
    if not args.kr_disable:
        dll_path = (
            args.kr_dll
            or os.environ.get("KINGSROW_DLL_PATH", "")
            or DEFAULT_KR_DLL
        )
        if os.path.exists(dll_path):
            from checkers.engine.kingsrow_interface import KingsRowEngine
            kr_engine = KingsRowEngine(dll_path)
            print(f"[generator] KingsRow ready: {dll_path}")
            print(f"            depth={args.kr_depth}  time_budget=1.0s/call")
        else:
            print(
                f"[generator] WARNING: KR DLL not found at {dll_path!r}\n"
                f"            Continuing without KR (positions only).",
                file=sys.stderr,
            )
    else:
        print("[generator] --kr-disable: writing positions without KR annotation.")

    print()
    output_path = Path(args.output)
    t0 = time.perf_counter()

    stats = generate_benchmark(
        output_path   = output_path,
        kr_engine     = kr_engine,
        kr_depth      = args.kr_depth,
        min_branching = args.min_branching,
        phase_filter  = args.phase,
        selected_cats = selected_cats,
        skip_existing = args.skip_existing,
        max_positions = args.max_positions,
        deduplicate   = args.deduplicate,
        quiet         = args.quiet,
    )

    elapsed = time.perf_counter() - t0
    _print_summary(stats, output_path, elapsed)


if __name__ == "__main__":
    main()
