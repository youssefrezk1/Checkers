#!/usr/bin/env python3
"""
benchmark_evaluator.py
───────────────────────
Evaluates the move-selection pipeline against a precomputed KingsRow dataset.

Pipeline path (NO LangGraph, NO ranker_agent, NO LLM):
    board
    → score_all_legal_moves()       [= scorer_node logic]
    → select_best_move()            [= deterministic_proposal_node logic]
    → chosen_move
    → compare path vs kr_path

Metrics per position (written to benchmark_results.jsonl)
    kr_rank        rank of KR's best move in our minimax ordering (1 = best);
                   None if the move could not be located.
    kr_in_top1     kr_rank == 1   (our chosen move matches KR's best)
    kr_in_top2     kr_rank <= 2
    kr_in_top3     kr_rank <= 3

Aggregation (per label = overall / each phase / each category)
    n, top1_count, top2_count, top3_count, top1_pct, top2_pct, top3_pct,
    unranked_count, rank_histogram = {rank_int: count, "unranked": count}.
    Phases:     opening, midgame, endgame.
    Categories: quiet, tactical, multi_jump.

Usage
    python benchmark_evaluator.py [OPTIONS]

Options
    --dataset PATH        Input JSONL from generator  (default: benchmark_positions.jsonl)
    --output  PATH        Per-position results JSONL  (default: benchmark_results.jsonl)
    --max-positions N     Hard cap; 0 = unlimited     (default: 0)
    --phase   PHASE       all|opening|midgame|endgame (default: all)
    --category CAT        all|quiet|tactical|multi_jump (default: all)
    --quiet               Print only the final summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Set env vars BEFORE importing the pipeline so module-level reads pick them up.
os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
os.environ.setdefault("MINIMAX_ENABLED", "true")

from dotenv import load_dotenv
load_dotenv()

from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.deterministic_proposal import select_best_move
from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.evaluation import evaluate_board

_HERE           = Path(__file__).parent
DEFAULT_DATASET = _HERE / "benchmark_positions.jsonl"
DEFAULT_OUTPUT  = _HERE / "benchmark_results.jsonl"


# ─── path utilities ────────────────────────────────────────────────────────────

def _norm(path) -> list:
    """Normalise path to list-of-[r,c] for comparison."""
    return [[int(sq[0]), int(sq[1])] for sq in (path or [])]


def _paths_equal(p1, p2) -> bool:
    return _norm(p1) == _norm(p2)


def _find_path_rank(path, enriched: list) -> int | None:
    """
    Return the 1-based rank of the move matching *path* in *enriched*.
    enriched is sorted best-first by minimax score (rank 1 = index 0).
    Returns None if the path is not found.
    """
    target = _norm(path)
    for i, m in enumerate(enriched):
        if _norm(m.get("path", [])) == target:
            return i + 1
    # Fallback: match by first + last square only (handles minor captured diffs).
    if len(target) >= 2:
        first, last = target[0], target[-1]
        for i, m in enumerate(enriched):
            mp = _norm(m.get("path", []))
            if len(mp) >= 2 and mp[0] == first and mp[-1] == last:
                return i + 1
    return None


# ─── heuristic-only scoring (no minimax, no alpha-beta, no selective D8) ──────

def _score_heuristic(board, side):
    """
    Rank legal moves purely by the static heuristic — apply each legal move
    once and evaluate the resulting board with evaluate_board(). No search.

    Returns (enriched, best_score, second_best_score, gap) in the same shape
    as score_all_legal_moves() so the rest of the pipeline can stay unchanged.
    Each enriched dict mirrors that contract: {type, path, captured, facts}
    with facts["heuristic_score"] populated for inspection.
    """
    legal = get_all_legal_moves(board, side)
    if not legal:
        return [], 0.0, None, 0.0

    scored = []
    for mv in legal:
        next_board = apply_move(board, mv)
        # evaluate_board returns from root_player's perspective; side is root.
        score = evaluate_board(next_board, current_player=side, root_player=side)
        scored.append((mv, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)

    enriched = []
    for rank, (mv, sc) in enumerate(scored, start=1):
        enriched.append({
            "type":     mv["type"],
            "path":     mv["path"],
            "captured": mv.get("captured", []),
            "facts":    {
                "heuristic_score": round(sc, 2),
                "minimax_score":   round(sc, 2),  # alias so select_best_move works
                "symbolic_rank":   rank,
            },
        })

    best   = enriched[0]["facts"]["heuristic_score"]
    second = enriched[1]["facts"]["heuristic_score"] if len(enriched) > 1 else None
    gap    = (best - second) if second is not None else float("inf")
    return enriched, best, second, gap


# ─── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    dataset_path:    Path,
    output_path:     Path,
    max_positions:   int,
    phase_filter:    str,
    category_filter: str,
    quiet:           bool,
    mode:            str = "minimax",
) -> dict:
    if not dataset_path.exists():
        print(f"[evaluator] ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Accumulators keyed by label ("all", "opening", "midgame", "endgame",
    # "quiet", "tactical", "multi_jump"). Each value tracks the full
    # ranking-distribution metrics needed for thesis-grade reporting:
    #   n               total positions scored
    #   top1/2/3_count  KR best move landed inside our top-1/2/3 minimax order
    #   unranked_count  KR best move could not be located in our scored list
    #   rank_histogram  Counter of {rank_int: count}  (key "unranked" for misses)
    def _new_bucket() -> dict:
        return {
            "n":              0,
            "top1_count":     0,
            "top2_count":     0,
            "top3_count":     0,
            "unranked_count": 0,
            "rank_histogram": Counter(),
        }

    acc: dict = defaultdict(_new_bucket)

    def _accumulate(label: str, rank: int | None) -> None:
        a = acc[label]
        a["n"] += 1
        if rank == 1:                       a["top1_count"] += 1
        if rank is not None and rank <= 2:  a["top2_count"] += 1
        if rank is not None and rank <= 3:  a["top3_count"] += 1
        if rank is None:
            a["unranked_count"] += 1
            a["rank_histogram"]["unranked"] += 1
        else:
            a["rank_histogram"][rank] += 1

    errors        = 0
    processed     = 0
    skipped       = 0
    t0            = time.perf_counter()
    total_score_ns   = 0      # sum of per-position scoring durations (nanoseconds)
    total_legal      = 0      # sum of len(enriched) — used for avg legal moves
    total_gap        = 0.0    # sum of (top1 − top2); skipped when only one move
    gap_samples      = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, encoding="utf-8") as in_f, \
         open(output_path,  "w", encoding="utf-8") as out_f:

        for line_no, line in enumerate(in_f, 1):
            if max_positions > 0 and processed >= max_positions:
                break

            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[evaluator] bad JSON on line {line_no}: {e}", file=sys.stderr)
                errors += 1
                continue

            # ── filters ──────────────────────────────────────────────────────
            phase = rec.get("phase", "unknown")
            if phase_filter != "all" and phase != phase_filter:
                skipped += 1
                continue

            if category_filter != "all":
                flag = {
                    "quiet":      "is_quiet",
                    "tactical":   "is_tactical",
                    "multi_jump": "is_multi_jump",
                }.get(category_filter, "")
                if not rec.get(flag, False):
                    skipped += 1
                    continue

            # Skip positions where KR returned no valid move.
            kr_path = rec.get("kr_path")
            if not kr_path or not rec.get("kr_path_found", False):
                skipped += 1
                continue

            # ── reconstruct board ─────────────────────────────────────────────
            board = rec.get("board")
            if board is None:
                print(f"[evaluator] line {line_no}: missing 'board' field", file=sys.stderr)
                errors += 1
                continue

            turn_str = rec.get("turn", "RED").upper()
            side     = RED if turn_str == "RED" else BLACK

            # ── run pipeline (scorer_node + deterministic_proposal_node) ──────
            try:
                t_score_start = time.perf_counter_ns()
                if mode == "heuristic":
                    enriched, _, _, _ = _score_heuristic(board, side)
                else:
                    enriched, _, _, _ = score_all_legal_moves(board, side)
                total_score_ns += time.perf_counter_ns() - t_score_start
                chosen_move, our_score, _, _ = select_best_move(enriched)
            except Exception as exc:
                print(f"[evaluator] pipeline error line {line_no}: {exc}", file=sys.stderr)
                errors += 1
                continue

            if not chosen_move or not enriched:
                errors += 1
                continue

            # ── runtime / branching / gap tracking ────────────────────────────
            total_legal += len(enriched)
            if len(enriched) >= 2:
                top1 = enriched[0]["facts"].get("minimax_score", 0.0)
                top2 = enriched[1]["facts"].get("minimax_score", 0.0)
                total_gap   += (top1 - top2)
                gap_samples += 1

            # ── metrics ───────────────────────────────────────────────────────
            our_path = chosen_move.get("path", [])

            # Rank of KR's best move in our minimax ordering (1-based).
            # None => KR's path could not be located in our scored list.
            kr_rank    = _find_path_rank(kr_path, enriched)
            kr_in_top1 = (kr_rank == 1)
            kr_in_top2 = (kr_rank is not None and kr_rank <= 2)
            kr_in_top3 = (kr_rank is not None and kr_rank <= 3)

            # ── write per-position result ─────────────────────────────────────
            result = {
                "fen":            rec.get("fen"),
                "turn":           turn_str,
                "phase":          phase,
                "n_legal":        rec.get("n_legal"),
                "is_quiet":       rec.get("is_quiet"),
                "is_tactical":    rec.get("is_tactical"),
                "is_multi_jump":  rec.get("is_multi_jump"),
                "our_path":       _norm(our_path),
                "our_score":      our_score,
                "kr_path":        _norm(kr_path),
                "kr_score":       rec.get("kr_score"),
                "kr_rank":        kr_rank,
                "kr_in_top1":     kr_in_top1,
                "kr_in_top2":     kr_in_top2,
                "kr_in_top3":     kr_in_top3,
                "source_file":    rec.get("source_file"),
                "game_index":     rec.get("game_index"),
                "ply_index":      rec.get("ply_index"),
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            processed += 1

            # ── accumulate ────────────────────────────────────────────────────
            _accumulate("all",   kr_rank)
            _accumulate(phase,   kr_rank)
            for cat_label, cat_flag in (
                ("quiet",      "is_quiet"),
                ("tactical",   "is_tactical"),
                ("multi_jump", "is_multi_jump"),
            ):
                if rec.get(cat_flag):
                    _accumulate(cat_label, kr_rank)

            if not quiet and processed % 50 == 0:
                a    = acc["all"]
                n    = a["n"]
                p1   = a["top1_count"] / n * 100 if n else 0
                rate = n / (time.perf_counter() - t0)
                print(f"  ... {n:>5}  top1={p1:.1f}%  ({rate:.0f}/s)")

    # Convert accumulators to JSON-safe dicts and attach top1/2/3 percentages.
    by_label = {}
    for label, a in acc.items():
        n = a["n"]
        by_label[label] = {
            "n":              n,
            "top1_count":     a["top1_count"],
            "top2_count":     a["top2_count"],
            "top3_count":     a["top3_count"],
            "unranked_count": a["unranked_count"],
            "top1_pct":       (a["top1_count"] / n * 100) if n else 0.0,
            "top2_pct":       (a["top2_count"] / n * 100) if n else 0.0,
            "top3_pct":       (a["top3_count"] / n * 100) if n else 0.0,
            # Counter keys mix ints and 'unranked' string — normalise to strings
            # so the JSON output is portable.
            "rank_histogram": {str(k): v for k, v in a["rank_histogram"].items()},
        }

    avg_ms_per_position = (
        (total_score_ns / processed) / 1_000_000.0 if processed else 0.0
    )
    avg_legal_moves = total_legal / processed if processed else 0.0
    avg_top1_top2_gap = (total_gap / gap_samples) if gap_samples else 0.0

    return {
        "mode":                mode,
        "processed":           processed,
        "skipped":             skipped,
        "errors":              errors,
        "elapsed_s":           round(time.perf_counter() - t0, 2),
        "avg_ms_per_position": round(avg_ms_per_position, 3),
        "avg_legal_moves":     round(avg_legal_moves, 2),
        "avg_top1_top2_gap":   round(avg_top1_top2_gap, 3),
        "by_label":            by_label,
    }


# ─── summary printing ──────────────────────────────────────────────────────────

def _fmt_pct(num: int, denom: int) -> str:
    return "  N/A" if denom == 0 else f"{num / denom * 100:5.1f}%"


def _print_metrics_block(label: str, d: dict, indent: str) -> None:
    """Print a 'topN : x.x%  (k/n)' block for one label."""
    n = d.get("n", 0)
    if n == 0:
        print(f"{indent}{label}: (no positions)")
        return
    t1, t2, t3 = d.get("top1_count", 0), d.get("top2_count", 0), d.get("top3_count", 0)
    print(f"{indent}{label}  (n={n})")
    print(f"{indent}  top1 : {_fmt_pct(t1, n)}  ({t1}/{n})")
    print(f"{indent}  top2 : {_fmt_pct(t2, n)}  ({t2}/{n})")
    print(f"{indent}  top3 : {_fmt_pct(t3, n)}  ({t3}/{n})")


def _print_rank_histogram(label: str, hist: dict, indent: str) -> None:
    """Print sorted rank-frequency bars for a label's histogram."""
    if not hist:
        return
    numeric_keys = sorted(int(k) for k in hist.keys() if k != "unranked")
    total = sum(hist.values())
    print(f"{indent}{label}  (n={total})")
    for r in numeric_keys:
        c = hist[str(r)]
        bar = "#" * min(c, 40)
        print(f"{indent}  rank {r:>3}: {c:>5}  {bar}")
    if "unranked" in hist:
        c = hist["unranked"]
        bar = "?" * min(c, 40)
        print(f"{indent}  unranked: {c:>5}  {bar}")


def _print_summary(stats: dict, output_path: Path) -> None:
    by    = stats.get("by_label", {})
    d_all = by.get("all", {"n": 0})
    n_all = d_all.get("n", 0)

    mode_label = (
        "heuristic-only (no minimax)" if stats.get("mode") == "heuristic"
        else "minimax (scorer + select_best_move)"
    )

    print()
    print("=" * 64)
    print("  BENCHMARK EVALUATION — SUMMARY")
    print("=" * 64)
    print(f"  MODE             : {mode_label}")
    print(f"  Results JSONL    : {output_path}")
    print(f"  Elapsed          : {stats['elapsed_s']}s")
    print(f"  Processed        : {stats['processed']}  "
          f"(skipped: {stats['skipped']}  errors: {stats['errors']})")
    print(f"  avg ms/position  : {stats.get('avg_ms_per_position', 0):.3f}")
    print(f"  avg legal moves  : {stats.get('avg_legal_moves', 0):.2f}")
    print(f"  avg top1-top2 gap: {stats.get('avg_top1_top2_gap', 0):.3f}")
    print()

    # ── Overall ──────────────────────────────────────────────────────────────
    print(f"OVERALL  n={n_all}")
    if n_all:
        t1 = d_all.get("top1_count", 0)
        t2 = d_all.get("top2_count", 0)
        t3 = d_all.get("top3_count", 0)
        ur = d_all.get("unranked_count", 0)
        print(f"  top1 : {_fmt_pct(t1, n_all)}  ({t1}/{n_all})")
        print(f"  top2 : {_fmt_pct(t2, n_all)}  ({t2}/{n_all})")
        print(f"  top3 : {_fmt_pct(t3, n_all)}  ({t3}/{n_all})")
        if ur:
            print(f"  unranked : {ur}/{n_all}  (KR move not in our scored list)")
    print()

    # ── By phase ─────────────────────────────────────────────────────────────
    print("BY PHASE:")
    any_phase = False
    for phase in ("opening", "midgame", "endgame"):
        d = by.get(phase)
        if not d or d.get("n", 0) == 0:
            continue
        any_phase = True
        _print_metrics_block(phase, d, indent="  ")
    if not any_phase:
        print("  (no positions matched any phase bucket)")
    print()

    # ── By category ──────────────────────────────────────────────────────────
    print("BY CATEGORY:")
    any_cat = False
    for cat in ("quiet", "tactical", "multi_jump"):
        d = by.get(cat)
        if not d or d.get("n", 0) == 0:
            continue
        any_cat = True
        _print_metrics_block(cat, d, indent="  ")
    if not any_cat:
        print("  (no positions matched any category bucket)")
    print()

    # ── Rank histogram (overall) ─────────────────────────────────────────────
    hist_all = d_all.get("rank_histogram", {})
    if hist_all:
        print("RANK HISTOGRAM (overall):")
        _print_rank_histogram("all", hist_all, indent="  ")
    print("=" * 64)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    print(f"  Summary JSON     : {summary_path}")
    print("=" * 64)


# ─── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate scorer→proposal pipeline against KingsRow dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",       default=str(DEFAULT_DATASET))
    parser.add_argument("--output",        default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--phase",
        choices=["all", "opening", "midgame", "endgame"], default="all")
    parser.add_argument("--category",
        choices=["all", "quiet", "tactical", "multi_jump"], default="all")
    parser.add_argument("--mode",
        choices=["minimax", "heuristic"], default="minimax",
        help="minimax = scorer_node + select_best_move; "
             "heuristic = static evaluate_board on each child position (no search).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.mode == "heuristic":
        pipeline_line = "[evaluator] Pipeline: apply_move -> evaluate_board (heuristic-only, no minimax)"
    else:
        pipeline_line = "[evaluator] Pipeline: score_all_legal_moves -> select_best_move (minimax)"
    print(pipeline_line)
    print("[evaluator] NO ranker_agent  NO LangGraph  NO LLM")
    print(f"[evaluator] MODE     : {args.mode}")
    print(f"[evaluator] Dataset  : {args.dataset}")
    print(f"[evaluator] Filter   : phase={args.phase}  category={args.category}")
    print()

    stats = evaluate(
        dataset_path    = Path(args.dataset),
        output_path     = Path(args.output),
        max_positions   = args.max_positions,
        phase_filter    = args.phase,
        category_filter = args.category,
        quiet           = args.quiet,
        mode            = args.mode,
    )

    _print_summary(stats, Path(args.output))


if __name__ == "__main__":
    main()
