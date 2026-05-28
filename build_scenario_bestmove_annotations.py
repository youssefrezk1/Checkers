#!/usr/bin/env python3
"""
build_scenario_bestmove_annotations.py
───────────────────────────────────────
Offline annotation builder for checkers/data/legality_stress/scenarios.jsonl.

For every position in the dataset this script computes:

  engine_best_path   — the path chosen by score_all_legal_moves → select_best_move
                       (deterministic minimax; no LLM, no external DLL needed)
  engine_best_score  — minimax score of that move
  kr_path            — KingsRow best move path (requires Kingsrow64.dll)
  kr_score           — KingsRow evaluation score (from side's perspective)
  kr_depth           — search depth actually used by KR
  kr_available       — False when KR DLL was disabled or unavailable

Output  (default):
  checkers/data/legality_stress/scenarios_bestmove_annotations.json

The output is a JSON array of objects, keyed by scenario_id.  The evaluator
(proposal_seperation_eval.py) loads this file via --bestmove-annotations and
enriches each result with contains_engine_best / contains_kingsrow_best.

KingsRow call pattern:
  Reuses the EXACT same proven path as benchmark_dataset_generator.py:
      for each legal move lm:
          test_board = apply_move(board, lm)
          kr_after   = kr_engine.get_best_move(test_board, opponent, time, depth)
          score      = -kr_after["score"]   # flip to side's perspective
      best = highest scored move

Usage
─────
  # Engine annotations only (no DLL required):
  python build_scenario_bestmove_annotations.py --kr-disable

  # Full annotations (KR DLL required):
  python build_scenario_bestmove_annotations.py

  # Custom DLL path:
  python build_scenario_bestmove_annotations.py --kr-dll "C:/path/to/Kingsrow64.dll"

  # Resume interrupted run (skip already-annotated scenario_ids):
  python build_scenario_bestmove_annotations.py --skip-existing

  # Limit for testing:
  python build_scenario_bestmove_annotations.py --kr-disable --max-positions 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
    _ep = find_dotenv(usecwd=True)
    load_dotenv(_ep, override=True)
except ImportError:
    pass

# Engine imports — no LLM, no graph, no DLL at this level
from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.deterministic_proposal import select_best_move

_HERE = Path(__file__).parent
_DEFAULT_DATASET = _HERE / "checkers" / "data" / "legality_stress" / "scenarios.jsonl"
_DEFAULT_OUTPUT  = _HERE / "checkers" / "data" / "legality_stress" / "scenarios_bestmove_annotations.json"
_DEFAULT_KR_DLL  = r"C:\Program Files (x86)\CheckerBoard\engines\Kingsrow64.dll"


# ── path normalisation (mirrors benchmark_dataset_generator._norm_path) ────────

def _norm_path(path) -> list:
    """Normalise path to list-of-[r,c] int lists."""
    return [[int(sq[0]), int(sq[1])] for sq in (path or [])]


# ── KingsRow ranking (EXACT same pattern as benchmark_dataset_generator) ────────

def _kr_rank_legal_moves(
    kr_engine,
    board: list,
    side: int,
    legal_moves: list,
    kr_time: float,
    kr_depth: int,
) -> list:
    """
    Score every legal move with KingsRow — IDENTICAL to benchmark_dataset_generator.py.

        for each legal move lm:
            test_board = apply_move(board, lm)
            kr_after   = kr_engine.get_best_move(test_board, opponent, time, depth)
            score      = -kr_after["score"]   # flip to side's perspective
        Sort descending; best move = index 0.

    Returns list of {move, kr_score, kr_depth} sorted best-first for *side*.
    """
    opponent = BLACK if side == RED else RED
    kr_evals: list = []
    for lm in legal_moves:
        test_board = apply_move(board, lm)          # new board, no mutation
        kr_after = kr_engine.get_best_move(
            test_board, opponent, kr_time, kr_depth,
        )
        kr_evals.append({
            "move":     lm,
            "kr_score": -kr_after["score"],         # flip to side's perspective
            "kr_depth": kr_after["depth"],
        })
    kr_evals.sort(key=lambda e: e["kr_score"], reverse=True)
    return kr_evals


# ── dataset helpers ─────────────────────────────────────────────────────────────

def _player_from_str(s: str) -> int:
    return RED if s.upper() == "RED" else BLACK


def _load_existing_ids(output_path: Path) -> set[str]:
    """Return set of scenario_ids already in the output file (for --skip-existing)."""
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            sid = entry.get("scenario_id")
            if sid:
                seen.add(sid)
    except Exception:
        pass
    return seen


def _load_existing_entries(output_path: Path) -> list[dict]:
    """Load existing annotation entries from output file."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ── main annotation loop ────────────────────────────────────────────────────────

def build_annotations(
    dataset_path: Path,
    output_path: Path,
    kr_engine,
    kr_depth: int,
    skip_existing: bool,
    max_positions: int,
    quiet: bool,
) -> dict:
    if not dataset_path.exists():
        print(f"[annotations] ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Load existing entries for --skip-existing resume support
    existing_ids: set[str] = set()
    existing_entries: list[dict] = []
    if skip_existing:
        existing_ids    = _load_existing_ids(output_path)
        existing_entries = _load_existing_entries(output_path)
        print(f"[annotations] --skip-existing: {len(existing_ids)} already annotated, resuming.")

    stats = {
        "total":           0,
        "skipped_cached":  0,
        "engine_ok":       0,
        "engine_error":    0,
        "kr_ok":           0,
        "kr_error":        0,
        "kr_disabled":     0,
    }

    new_entries: list[dict] = []
    t0 = time.perf_counter()

    with open(dataset_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if max_positions > 0 and stats["total"] - stats["skipped_cached"] >= max_positions:
                break

            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[annotations] bad JSON line {line_no}: {e}", file=sys.stderr)
                continue

            scenario_id = entry.get("scenario_id", f"line_{line_no}")
            stats["total"] += 1

            if skip_existing and scenario_id in existing_ids:
                stats["skipped_cached"] += 1
                continue

            board = entry.get("board")
            side_str = entry.get("side_to_move", "RED")
            if board is None:
                print(f"[annotations] line {line_no}: missing board, skipping", file=sys.stderr)
                continue

            side = _player_from_str(side_str)
            legal = get_all_legal_moves(board, side)

            annotation: dict = {
                "scenario_id":      scenario_id,
                "side_to_move":     side_str,
                # engine best
                "engine_best_path": None,
                "engine_best_score": None,
                "engine_ok":        False,
                # kingsrow best
                "kr_path":          None,
                "kr_score":         None,
                "kr_depth":         None,
                "kr_available":     kr_engine is not None,
            }

            if not legal:
                # Terminal position — no moves to annotate
                new_entries.append(annotation)
                continue

            # ── Engine best move (live minimax, no external deps) ────────────
            try:
                enriched, _, _, _ = score_all_legal_moves(board, side)
                if enriched:
                    chosen, chosen_score, _, _ = select_best_move(enriched)
                    annotation["engine_best_path"]  = _norm_path(chosen.get("path", []))
                    annotation["engine_best_score"] = round(float(chosen_score), 3)
                    annotation["engine_ok"]         = True
                    stats["engine_ok"] += 1
                    if not quiet:
                        print(
                            f"  [{stats['total'] - stats['skipped_cached']:>4}] "
                            f"{scenario_id:<42}  engine_score={chosen_score:>8.2f}",
                            end="",
                        )
            except Exception as exc:
                stats["engine_error"] += 1
                if not quiet:
                    print(f"\n  [engine-error] {scenario_id}: {exc}", file=sys.stderr)

            # ── KingsRow best move (same pattern as benchmark_dataset_generator) ──
            if kr_engine is not None:
                try:
                    kr_evals = _kr_rank_legal_moves(
                        kr_engine, board, side, legal,
                        kr_time=1.0, kr_depth=kr_depth,
                    )
                    if kr_evals:
                        best = kr_evals[0]
                        annotation["kr_path"]  = _norm_path(best["move"]["path"])
                        annotation["kr_score"] = best["kr_score"]
                        annotation["kr_depth"] = best["kr_depth"]
                    stats["kr_ok"] += 1
                    if not quiet:
                        print(f"  kr_score={annotation['kr_score']}")
                except Exception as exc:
                    stats["kr_error"] += 1
                    if not quiet:
                        print(f"\n  [kr-error] {scenario_id}: {exc}", file=sys.stderr)
            else:
                stats["kr_disabled"] += 1
                if not quiet:
                    print()  # newline after engine line

            new_entries.append(annotation)

    # Merge: existing preserved entries + new
    all_entries = existing_entries + new_entries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(all_entries, out_f, indent=2)

    stats["written"] = len(all_entries)
    stats["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return stats


# ── summary printer ─────────────────────────────────────────────────────────────

def _print_summary(stats: dict, output_path: Path) -> None:
    print()
    print("=" * 62)
    print("  SCENARIO BEST-MOVE ANNOTATION — SUMMARY")
    print("=" * 62)
    print(f"  Output           : {output_path}")
    print(f"  Elapsed          : {stats.get('elapsed_s', 0):.1f}s")
    print()
    print(f"  Total lines read : {stats.get('total', 0)}")
    print(f"  Skipped (cached) : {stats.get('skipped_cached', 0)}")
    print(f"  Engine OK        : {stats.get('engine_ok', 0)}")
    print(f"  Engine errors    : {stats.get('engine_error', 0)}")
    if stats.get("kr_disabled", 0):
        print(f"  KingsRow         : DISABLED (--kr-disable)")
    else:
        print(f"  KingsRow OK      : {stats.get('kr_ok', 0)}")
        print(f"  KingsRow errors  : {stats.get('kr_error', 0)}")
    print(f"  Total written    : {stats.get('written', 0)}")
    print("=" * 62)


# ── entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline best-move annotations for scenarios.jsonl.\n"
            "Produces engine best-move (always) and KingsRow best-move (optional)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default=str(_DEFAULT_DATASET),
        help="Path to scenarios.jsonl (default: checkers/data/legality_stress/scenarios.jsonl)",
    )
    parser.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT),
        help="Output JSON path (default: checkers/data/legality_stress/scenarios_bestmove_annotations.json)",
    )
    parser.add_argument(
        "--kr-depth", type=int, default=6,
        help="KingsRow search depth per legal move (default: 6)",
    )
    parser.add_argument(
        "--kr-disable", action="store_true",
        help="Skip KingsRow calls; write engine annotations only.",
    )
    parser.add_argument(
        "--kr-dll", default=None,
        help=f"Explicit path to Kingsrow64.dll (default: {_DEFAULT_KR_DLL})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Resume: skip scenario_ids already in --output; append new entries.",
    )
    parser.add_argument(
        "--max-positions", type=int, default=0,
        help="Cap on NEW positions to annotate (0 = unlimited)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-position output",
    )
    args = parser.parse_args()

    # ── KingsRow engine setup ───────────────────────────────────────────────
    kr_engine = None
    if not args.kr_disable:
        dll_path = (
            args.kr_dll
            or os.environ.get("KINGSROW_DLL_PATH", "")
            or _DEFAULT_KR_DLL
        )
        if os.path.exists(dll_path):
            from checkers.engine.kingsrow_interface import KingsRowEngine
            kr_engine = KingsRowEngine(dll_path)
            print(f"[annotations] KingsRow ready : {dll_path}")
            print(f"              depth={args.kr_depth}  time_budget=1.0s/call")
        else:
            print(
                f"[annotations] WARNING: KR DLL not found at {dll_path!r}\n"
                f"              Continuing without KR (engine annotations only).",
                file=sys.stderr,
            )
    else:
        print("[annotations] --kr-disable: engine annotations only (no KingsRow).")

    print(f"[annotations] Dataset  : {args.dataset}")
    print(f"[annotations] Output   : {args.output}")
    print()

    stats = build_annotations(
        dataset_path   = Path(args.dataset),
        output_path    = Path(args.output),
        kr_engine      = kr_engine,
        kr_depth       = args.kr_depth,
        skip_existing  = args.skip_existing,
        max_positions  = args.max_positions,
        quiet          = args.quiet,
    )
    _print_summary(stats, Path(args.output))


if __name__ == "__main__":
    main()
