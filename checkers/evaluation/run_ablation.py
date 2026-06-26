# checkers/evaluation/run_ablation.py
#
# No-seed ablation harness for the proposal-authoritative pipeline.
#
# Produces paired evidence for the thesis question:
#     Do symbolic reasoning seeds reduce hallucinations and contradictions
#     in the explanations the ranker generates?
#
# How it works
# ------------
# 1. The simplified pipeline is fully deterministic at the proposal level:
#    given an identical board state, proposer_agent picks the
#    same move every time.  The ONLY non-deterministic component is the LLM
#    seed-prose / refinement call inside explainer_agent.
#
# 2. The runner therefore plays N AI-vs-AI games TWICE, with the same
#    starting boards each time, varying only RANKER_SEEDS_DISABLED.  Both
#    runs land on the same chosen_move for every turn; only the reasoning
#    text and ranker_diagnostics differ.
#
# 3. Eval-source records are partitioned by run_tag automatically
#    (logger_node nests `evaluation_source/seed_on/` and
#    `evaluation_source/seed_off/`).
#
# 4. After both batches finish the runner verifies the safeguards from the
#    spec — chosen_move, chosen_move_score, final_choice_source must match
#    on every paired turn.  Any divergence aborts with a non-zero exit code.
#
# 5. Optionally the runner invokes metrics.run_batch in --compare mode and
#    writes the comparative report to disk.
#
# Constraints honoured
#  - No prompt changes (ablation only suppresses seeds in the existing prompt).
#  - No move-selection logic touched (proposal-authoritative invariant intact).
#  - No evaluator semantics changed (run_batch metrics unchanged).
#  - No LLM judges.

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _set_env(var: str, val: Optional[str]) -> Optional[str]:
    """Set or clear an env var; return previous value for restoration."""
    prev = os.environ.get(var)
    if val is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = val
    return prev


def _restore_env(var: str, prev: Optional[str]) -> None:
    if prev is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = prev


# ─────────────────────────────────────────────────────────────────────────────
# Single-game play, deterministic at the proposal level
# ─────────────────────────────────────────────────────────────────────────────

def _play_one_game(
    *,
    log_dir: Path,
    run_tag: str,
    max_turns: int,
    seeds_disabled: bool,
    game_log_id: str,
) -> str:
    """
    Play one AI-vs-AI game and return the game_log_id used.

    Configures the runtime via env vars so that:
      - logger_node writes to `log_dir` (CHECKERS_LOG_DIR)
      - the eval-source file is partitioned by run_tag
      - the explainer_agent honours EXPLAINER_SEEDS_DISABLED on every turn
      - logger stdout chatter is suppressed (the runner prints its own)
    """
    # Imports deferred until env vars are in place: graph compilation reads
    # LOG_DIR at module load time in logger_node.
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "evaluation_source" / run_tag).mkdir(parents=True, exist_ok=True)

    prev_log_dir   = _set_env("CHECKERS_LOG_DIR", str(log_dir))
    prev_logger    = _set_env("CHECKERS_LOGGER_PRINT", "false")
    prev_seeds_dis = _set_env("EXPLAINER_SEEDS_DISABLED", "1" if seeds_disabled else "0")
    prev_run_tag   = _set_env("EXPLAINER_RUN_TAG", run_tag)

    try:
        # Import AFTER env is set so logger_node picks up CHECKERS_LOG_DIR.
        from checkers.graph.graph import build_graph
        from checkers.state.state import CheckersState
        from checkers.engine.board import create_initial_board, RED

        graph = build_graph()

        init_state = CheckersState(
            board=create_initial_board(),
            current_player=RED,
            turn_number=0,
            game_log_id=game_log_id,
        )

        cfg = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": max(50, max_turns * 6),
        }

        # Use stream so we can early-stop at max_turns even when game_over
        # never fires (large positions, draw-by-repetition not detected, …).
        final_state: Dict[str, Any] = init_state.model_dump()
        ply_count = 0
        for chunk in graph.stream(
            init_state, stream_mode="updates", config=cfg,
        ):
            for node_name, delta in chunk.items():
                if node_name in ("__interrupt__", "__end__"):
                    continue
                if not isinstance(delta, dict):
                    continue
                final_state.update(delta)
                if node_name == "updater_agent":
                    ply_count += 1
            if final_state.get("game_over"):
                break
            if ply_count >= max_turns:
                break

        return game_log_id
    finally:
        _restore_env("CHECKERS_LOG_DIR",        prev_log_dir)
        _restore_env("CHECKERS_LOGGER_PRINT",   prev_logger)
        _restore_env("EXPLAINER_SEEDS_DISABLED", prev_seeds_dis)
        _restore_env("EXPLAINER_RUN_TAG",       prev_run_tag)


# ─────────────────────────────────────────────────────────────────────────────
# Pair-safety verification
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{lineno}: bad JSON ({e})") from e
    return records


def _verify_pairs(
    seed_on_paths:  List[Path],
    seed_off_paths: List[Path],
) -> Tuple[int, int]:
    """
    Returns (n_pairs_checked, n_pairs_skipped).

    Raises PairingError on any divergence — chosen_move /
    chosen_move_score / final_choice_source must be identical between the
    two runs for every paired turn_id.
    """
    from checkers.evaluation.metrics.compare import (
        pair_by_turn_id,
        assert_records_paired,
    )

    # Build a flat list per side (multiple game files concatenated).
    a_recs: List[Dict[str, Any]] = []
    for p in seed_on_paths:
        a_recs.extend(_load_jsonl(p))
    b_recs: List[Dict[str, Any]] = []
    for p in seed_off_paths:
        b_recs.extend(_load_jsonl(p))

    pairs = pair_by_turn_id(a_recs, b_recs)
    if not pairs:
        return (0, len(a_recs) + len(b_recs))

    for ra, rb in pairs:
        assert_records_paired(ra, rb)

    skipped = (len(a_recs) - len(pairs)) + (len(b_recs) - len(pairs))
    return (len(pairs), skipped)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    *,
    n_games: int,
    max_turns_per_game: int,
    log_root: Path,
    write_compare_report: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a matched ablation experiment.

    Parameters
    ----------
    n_games : int
        Number of games to play under EACH condition. Total games = 2 * n_games.
    max_turns_per_game : int
        Safety cap on plies per game; the runner stops early if reached.
    log_root : Path
        Root directory passed as CHECKERS_LOG_DIR. Eval-source files land at
            <log_root>/evaluation_source/seed_on/<game_log_id>.jsonl
            <log_root>/evaluation_source/seed_off/<game_log_id>.jsonl
    write_compare_report : Path or None
        If provided, write the comparative metrics JSON to this path.

    Returns
    -------
    dict with keys: 'seed_on_files', 'seed_off_files', 'pair_check', 'report'.
    """
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    seed_on_dir  = log_root / "evaluation_source" / "seed_on"
    seed_off_dir = log_root / "evaluation_source" / "seed_off"
    seed_on_dir.mkdir(parents=True, exist_ok=True)
    seed_off_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Play matched games. Same game_log_id base for both conditions so
    #       the resulting JSONL filenames pair up trivially.
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    seed_on_ids:  List[str] = []
    seed_off_ids: List[str] = []
    for i in range(n_games):
        gid = f"ablation_{stamp}_g{i:03d}"
        print(f"[ablation] game {i + 1}/{n_games} — seed_on   gid={gid}")
        _play_one_game(
            log_dir=log_root, run_tag="seed_on",
            max_turns=max_turns_per_game, seeds_disabled=False, game_log_id=gid,
        )
        seed_on_ids.append(gid)
        print(f"[ablation] game {i + 1}/{n_games} — seed_off  gid={gid}")
        _play_one_game(
            log_dir=log_root, run_tag="seed_off",
            max_turns=max_turns_per_game, seeds_disabled=True, game_log_id=gid,
        )
        seed_off_ids.append(gid)

    seed_on_files  = [seed_on_dir  / f"{gid}.jsonl" for gid in seed_on_ids]
    seed_off_files = [seed_off_dir / f"{gid}.jsonl" for gid in seed_off_ids]

    # ── 2. Safeguard: chosen_move / chosen_move_score / final_choice_source
    #       must match per turn across the two runs.
    n_pairs, n_skipped = _verify_pairs(seed_on_files, seed_off_files)
    print(f"[ablation] pair check: {n_pairs} matched turns ({n_skipped} unpaired)")

    # ── 3. Build comparative report.
    from checkers.evaluation.metrics.run_batch import evaluate_batch
    from checkers.evaluation.metrics.compare import compare_summaries

    seed_on_summary  = evaluate_batch(seed_on_files)
    seed_off_summary = evaluate_batch(seed_off_files)
    report = compare_summaries(
        seed_on_summary, seed_off_summary,
        label_a="seed_on", label_b="seed_off",
    )

    if write_compare_report is not None:
        Path(write_compare_report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[ablation] wrote comparative report → {write_compare_report}")

    return {
        "seed_on_files":  [str(p) for p in seed_on_files],
        "seed_off_files": [str(p) for p in seed_off_files],
        "pair_check":     {"matched": n_pairs, "unpaired": n_skipped},
        "report":         report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=int, default=1,
        help="Games per condition (default: 1).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=80,
        help="Safety cap on plies per game (default: 80).",
    )
    parser.add_argument(
        "--log-root", default="logs",
        help="CHECKERS_LOG_DIR root; eval-source files land under "
             "<log-root>/evaluation_source/{seed_on,seed_off}/. "
             "(default: logs)",
    )
    parser.add_argument(
        "--report-out", default=None,
        help="Write comparative JSON report here.",
    )
    args = parser.parse_args(argv)

    if args.games <= 0:
        print("[ablation] --games must be >= 1", file=sys.stderr)
        return 1

    try:
        result = run_ablation(
            n_games=args.games,
            max_turns_per_game=args.max_turns,
            log_root=Path(args.log_root),
            write_compare_report=(
                Path(args.report_out) if args.report_out else None
            ),
        )
    except Exception as e:
        print(f"[ablation] FAILED: {e}", file=sys.stderr)
        return 2

    # Brief headline summary on stdout (full report is JSON on disk if requested).
    delta = (result.get("report") or {}).get("delta") or {}
    fac   = delta.get("factuality") or {}
    src   = (delta.get("by_claim_source") or {}).get("seed_vs_unsupported") or {}
    print()
    print("── HEADLINE DELTA (seed_on − seed_off) ─────────────────────────")
    for k in (
        "post_repair_contradiction_rate_micro",
        "post_repair_supported_rate_micro",
    ):
        if k in fac:
            print(f"  factuality.{k} = {fac[k]:+.4f}")
    for k in (
        "contradicted_delta",
        "hallucination_delta",
        "supported_delta",
    ):
        if k in src:
            print(f"  by_claim_source.seed_vs_unsupported.{k} = {src[k]:+.4f}")
    print(f"  matched_turns = {result['pair_check']['matched']}")
    print("────────────────────────────────────────────────────────────────")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
