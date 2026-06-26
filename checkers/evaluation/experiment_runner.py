# checkers/evaluation/experiment_runner.py
#
# Minimal deterministic batch experiment runner for the reasoning-faithfulness
# evaluation pipeline.
#
# PURPOSE
# -------
# Run N games automatically (RED = full AI pipeline, BLACK = deterministic
# best-move via symbolic engine), collect the evaluation_source JSONL produced
# by logger_node for each game, run replay_evaluate_file() on each, aggregate
# the per-game summary dicts into a single cross-game result, and write one
# JSON file under logs/evaluation/experiments/.
#
# This module also exposes evaluate_existing() for running the same aggregation
# pipeline over pre-recorded evaluation_source files without launching new games
# (useful for smoke tests and post-hoc batch analysis).
#
# CONSTRAINTS
# -----------
# - No LLM changes.  Gameplay, ranker, scorer, proposal logic untouched.
# - No evaluator changes.  replay_evaluate_file() / summarize_records() used as-is.
# - Standard library only (plus existing project imports).
# - No plotting, pandas, markdown reports, or concurrency.
# - Deterministic: same source files → same aggregate always.
#
# USAGE (CLI)
# -----------
#   python -m checkers.evaluation.experiment_runner --games 3 --max-turns 80
#   python -m checkers.evaluation.experiment_runner --existing logs/evaluation_source/game_*.jsonl
#
# USAGE (API)
# -----------
#   from checkers.evaluation.experiment_runner import run_experiment, evaluate_existing
#   result = run_experiment(n_games=3, max_turns=80)
#   result = evaluate_existing(source_paths=["logs/evaluation_source/game_X.jsonl"])

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project-root guard — ensure imports resolve correctly when called as a module
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy runtime imports (only needed for run_single_game)
# These are deferred so that evaluate_existing() works without the LLM stack.
# ---------------------------------------------------------------------------

def _import_runtime():
    """Import the game-running components on first use."""
    # Must be set before graph import
    os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
    os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

    from dotenv import load_dotenv  # type: ignore
    load_dotenv()

    from checkers.graph.graph import checkers_graph
    from checkers.state.state import CheckersState
    from checkers.agents.updater_agent import updater_agent as _ua
    from checkers.engine.board import RED, BLACK, create_initial_board
    from checkers.engine.rules import get_all_legal_moves
    from checkers.engine.move_facts import compute_move_facts

    return checkers_graph, CheckersState, _ua, RED, BLACK, create_initial_board, get_all_legal_moves, compute_move_facts


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _pct(numerator: int, denominator: int) -> float:
    """Return percentage rounded to 1 decimal, or 0.0 if denominator is zero."""
    return round(100.0 * numerator / denominator, 1) if denominator > 0 else 0.0


def _aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine a list of per-game summary dicts (returned by replay_evaluate_file)
    into a single cross-game aggregate.

    Keys in each summary:
      total_turns, total_claims,
      supported_claims, unsupported_claims, contradicted_claims, vague_claims,
      turns_with_contradiction, turns_with_unsupported, turns_with_vague,
      reasoning_path_counts, trajectory_event_counts
    """
    if not summaries:
        return {}

    total_turns       = sum(s.get("total_turns", 0)       for s in summaries)
    total_claims      = sum(s.get("total_claims", 0)      for s in summaries)
    supported         = sum(s.get("supported_claims", 0)  for s in summaries)
    unsupported       = sum(s.get("unsupported_claims", 0) for s in summaries)
    contradicted      = sum(s.get("contradicted_claims", 0) for s in summaries)
    vague             = sum(s.get("vague_claims", 0)      for s in summaries)

    path_counts   = collections.Counter()
    event_counts  = collections.Counter()
    for s in summaries:
        path_counts.update(s.get("reasoning_path_counts", {}))
        event_counts.update(s.get("trajectory_event_counts", {}))

    return {
        "games_evaluated":         len(summaries),
        "total_turns":             total_turns,
        "total_claims":            total_claims,
        "claims_per_turn":         round(total_claims / total_turns, 2) if total_turns else 0.0,
        "supported_count":         supported,
        "unsupported_count":       unsupported,
        "contradicted_count":      contradicted,
        "vague_count":             vague,
        "supported_pct":           _pct(supported,    total_claims),
        "unsupported_pct":         _pct(unsupported,  total_claims),
        "contradicted_pct":        _pct(contradicted, total_claims),
        "vague_pct":               _pct(vague,        total_claims),
        "turns_with_contradiction": sum(s.get("turns_with_contradiction", 0) for s in summaries),
        "turns_with_unsupported":   sum(s.get("turns_with_unsupported",   0) for s in summaries),
        "turns_with_vague":         sum(s.get("turns_with_vague",         0) for s in summaries),
        "reasoning_path_counts":   dict(path_counts),
        "trajectory_event_counts": dict(event_counts),
    }


# ---------------------------------------------------------------------------
# Single-game runner
# ---------------------------------------------------------------------------

def run_single_game(
    max_turns: int = 80,
    verbose: bool = False,
) -> Optional[str]:
    """
    Run one headless game: RED = full AI pipeline, BLACK = deterministic
    best-move via compute_move_facts().

    Returns the game_log_id string (used to locate the evaluation_source JSONL),
    or None if the game failed to produce a log id.
    """
    (checkers_graph, CheckersState, _ua,
     RED, BLACK, create_initial_board,
     get_all_legal_moves, compute_move_facts) = _import_runtime()

    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()

    while not acc.get("game_over") and (acc.get("turn_number") or 0) < max_turns:
        player = acc["current_player"]

        if player == RED:
            # ── RED: full AI pipeline ─────────────────────────────────────
            acc["last_completed_node"] = None
            cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 60}
            ok = False
            try:
                for chunk in checkers_graph.stream(
                    acc, stream_mode="updates",
                    interrupt_after=["updater_agent"], config=cfg,
                ):
                    for node_name, delta in chunk.items():
                        if node_name in ("__interrupt__", "__end__"):
                            continue
                        if isinstance(delta, dict):
                            acc.update(delta)
                        if node_name == "updater_agent":
                            ok = True
            except Exception as exc:
                print(f"[experiment_runner] RED pipeline error: {exc}", file=sys.stderr)
                break
            if verbose and ok:
                print(f"  RED ply {acc.get('turn_number'):3d}")

        else:
            # ── BLACK: deterministic best-move ────────────────────────────
            board = acc["board"]
            legal = get_all_legal_moves(board, BLACK)
            if not legal:
                break
            best_move  = legal[0]
            best_score = float("-inf")
            for m in legal:
                f = compute_move_facts(board, m, BLACK)
                s = float(f.get("minimax_score") or 0.0)
                if s > best_score:
                    best_score = s
                    best_move  = m

            acc["chosen_move"]         = best_move
            acc["last_move_reasoning"] = "BLACK auto best-move (deterministic)"
            valid  = set(CheckersState.model_fields.keys())
            state  = CheckersState(**{k: v for k, v in acc.items() if k in valid})
            result = _ua(state)
            acc.update(result)
            if verbose:
                print(f"  BLK ply {acc.get('turn_number'):3d}")

    gid = acc.get("game_log_id")
    if verbose:
        print(
            f"  Game done — turns={acc.get('turn_number')} "
            f"winner={acc.get('winner')} game_log_id={gid}"
        )
    return gid


# ---------------------------------------------------------------------------
# Per-game evaluation
# ---------------------------------------------------------------------------

def _evaluate_game(
    game_log_id: str,
    eval_dir: Path,
    eval_source_dir: Path,
) -> Dict[str, Any]:
    """
    Locate the evaluation_source JSONL for game_log_id and run
    replay_evaluate_file() into eval_dir.

    Returns a dict with keys: game_log_id, source_path, eval_path, summary.
    """
    from checkers.evaluation.replay_evaluator import replay_evaluate_file

    src_path  = eval_source_dir / f"{game_log_id}.jsonl"
    eval_path = eval_dir / f"{game_log_id}.jsonl"

    summary = replay_evaluate_file(str(src_path), str(eval_path))
    return {
        "game_log_id": game_log_id,
        "source_path": str(src_path),
        "eval_path":   str(eval_path),
        "summary":     summary,
    }


# ---------------------------------------------------------------------------
# Public API — run_experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_games:  int  = 3,
    max_turns: int = 80,
    output_dir: str = "logs/evaluation/experiments",
    eval_dir:   str = "logs/evaluation",
    eval_source_dir: str = "logs/evaluation_source",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run n_games fresh games, evaluate each, aggregate, and write an experiment
    JSON to output_dir.

    Parameters
    ----------
    n_games : int
        Number of sequential games to run.
    max_turns : int
        Per-game turn cap (safety limit).
    output_dir : str
        Directory for the experiment aggregate JSON.
    eval_dir : str
        Directory where per-game eval JSONL files are written.
    eval_source_dir : str
        Directory where logger_node writes evaluation_source JSONL files.
    verbose : bool
        If True, print per-ply progress.

    Returns
    -------
    dict
        Full experiment result including per_game_results and aggregate.
    """
    out_dir  = Path(output_dir)
    ev_dir   = Path(eval_dir)
    src_dir  = Path(eval_source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    per_game: List[Dict[str, Any]] = []

    for game_no in range(1, n_games + 1):
        print(f"[experiment_runner] Starting game {game_no}/{n_games}...")
        t0 = time.monotonic()
        gid = run_single_game(max_turns=max_turns, verbose=verbose)
        elapsed = round(time.monotonic() - t0, 1)

        if not gid:
            print(f"[experiment_runner] Game {game_no} produced no game_log_id — skipped", file=sys.stderr)
            continue

        result = _evaluate_game(gid, ev_dir, src_dir)
        result["elapsed_seconds"] = elapsed
        per_game.append(result)
        print(
            f"[experiment_runner] Game {game_no} done in {elapsed}s — "
            f"turns={result['summary'].get('total_turns')} "
            f"claims={result['summary'].get('total_claims')}"
        )

    summaries  = [g["summary"] for g in per_game]
    aggregate  = _aggregate_summaries(summaries)

    experiment = {
        "experiment_id":    experiment_id,
        "n_games_requested": n_games,
        "n_games_completed": len(per_game),
        "max_turns_per_game": max_turns,
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "per_game_results": per_game,
        "aggregate":        aggregate,
    }

    out_path = out_dir / f"experiment_{experiment_id}.json"
    out_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print(f"[experiment_runner] Aggregate saved → {out_path}")
    return experiment


# ---------------------------------------------------------------------------
# Public API — evaluate_existing
# ---------------------------------------------------------------------------

def evaluate_existing(
    source_paths: List[str],
    output_dir:   str = "logs/evaluation/experiments",
    eval_dir:     str = "logs/evaluation",
) -> Dict[str, Any]:
    """
    Aggregate evaluation over pre-recorded evaluation_source JSONL files.
    No game runs are performed — useful for post-hoc analysis and smoke tests.

    Parameters
    ----------
    source_paths : list of str
        Absolute or relative paths to evaluation_source JSONL files.
    output_dir : str
        Directory for the experiment aggregate JSON.
    eval_dir : str
        Directory where per-game eval JSONL files are written.

    Returns
    -------
    dict
        Full experiment result including per_game_results and aggregate.
    """
    from checkers.evaluation.replay_evaluator import replay_evaluate_file

    out_dir = Path(output_dir)
    ev_dir  = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    per_game: List[Dict[str, Any]] = []

    for src_str in source_paths:
        src_path = Path(src_str)
        if not src_path.exists():
            print(f"[experiment_runner] Source not found: {src_path} — skipped", file=sys.stderr)
            continue

        gid       = src_path.stem          # game_YYYYMMDD_HHMMSS_XXXXXX
        eval_path = ev_dir / f"existing_{gid}.jsonl"

        t0      = time.monotonic()
        summary = replay_evaluate_file(str(src_path), str(eval_path))
        elapsed = round(time.monotonic() - t0, 2)

        result = {
            "game_log_id":     gid,
            "source_path":     str(src_path),
            "eval_path":       str(eval_path),
            "summary":         summary,
            "elapsed_seconds": elapsed,
        }
        per_game.append(result)
        print(
            f"[experiment_runner] Evaluated {gid} — "
            f"turns={summary.get('total_turns')} "
            f"claims={summary.get('total_claims')} "
            f"({elapsed}s)"
        )

    summaries  = [g["summary"] for g in per_game]
    aggregate  = _aggregate_summaries(summaries)

    experiment = {
        "experiment_id":    experiment_id,
        "mode":             "existing_sources",
        "n_games_evaluated": len(per_game),
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "per_game_results": per_game,
        "aggregate":        aggregate,
    }

    out_path = out_dir / f"experiment_existing_{experiment_id}.json"
    out_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print(f"[experiment_runner] Aggregate saved → {out_path}")
    return experiment


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reasoning-faithfulness experiment batch runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m checkers.evaluation.experiment_runner --games 3\n"
            "  python -m checkers.evaluation.experiment_runner --games 3 --max-turns 60 --verbose\n"
            "  python -m checkers.evaluation.experiment_runner --existing logs/evaluation_source/game_*.jsonl\n"
        ),
    )
    parser.add_argument("--games",      type=int,  default=3,
                        help="Number of fresh games to run (default: 3).")
    parser.add_argument("--max-turns",  type=int,  default=80,
                        help="Per-game turn cap (default: 80).")
    parser.add_argument("--output-dir", type=str,  default="logs/evaluation/experiments",
                        help="Directory for aggregate JSON output.")
    parser.add_argument("--eval-dir",   type=str,  default="logs/evaluation",
                        help="Directory for per-game eval JSONL files.")
    parser.add_argument("--existing",   nargs="+", default=None,
                        help="Evaluate pre-recorded source files instead of running new games.")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print per-ply progress during game runs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.existing:
        result = evaluate_existing(
            source_paths=args.existing,
            output_dir=args.output_dir,
            eval_dir=args.eval_dir,
        )
    else:
        result = run_experiment(
            n_games=args.games,
            max_turns=args.max_turns,
            output_dir=args.output_dir,
            eval_dir=args.eval_dir,
            verbose=args.verbose,
        )

    agg = result.get("aggregate", {})
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
