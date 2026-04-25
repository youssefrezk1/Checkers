#!/usr/bin/env python3
"""
Trace log audit — two independent modes.

PART 1 (--mode static):
  Parse game_20260425_144451_493544.jsonl only.
  Reconstruct board states by replaying recorded moves.
  For each RED turn: run minimax to find best_legal, compare vs chosen path, report gap.
  No LLM calls. No live pipeline.

PART 2 (--mode forced-replay):
  Run full proposal pipeline on each exact JSONL board state.
  Force the RECORDED RED move to be applied after the pipeline runs.
  Board never diverges from the trace.
  Captures: proposal paths, candidate paths, added_after_llm, dropped_by_postprocess.

Usage:
  python3 -m checkers.eval.trace_log_audit --mode static
  python3 -m checkers.eval.trace_log_audit --mode forced-replay
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

from checkers.engine.board import RED, BLACK, create_initial_board
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.search.minimax_core import search_root_all_scores, clear_transposition_table

TRACE_JSONL = Path(__file__).parent.parent.parent / "logs" / "game_20260425_144451_493544.jsonl"
DEPTH       = int(os.environ.get("MINIMAX_DEPTH", "6"))

# ── helpers ───────────────────────────────────────────────────────────────────

def _pk(path: list | tuple) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _paths_match(a: list, b: list) -> bool:
    return _pk(a) == _pk(b)


def _load_jsonl() -> list[dict]:
    if not TRACE_JSONL.exists():
        sys.exit(f"[audit] FATAL: {TRACE_JSONL} not found")
    with open(TRACE_JSONL) as f:
        return [json.loads(l) for l in f if l.strip()]


def _apply_recorded_move(board: list, rec: dict) -> list:
    """Apply one recorded move to a board copy. Returns new board."""
    import copy
    b = copy.deepcopy(board)
    path   = [list(sq) for sq in rec["path"]]
    cap    = [list(sq) for sq in rec.get("captured", [])]
    mtype  = rec["move_type"]
    move   = {"type": mtype, "path": path, "captured": cap}
    b = apply_move(b, move)
    # promotion
    last = path[-1]
    piece = b[last[0]][last[1]]
    from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
    player = rec["player_who_moved"]
    if player == RED and last[0] == 0 and piece == RED:
        b[last[0]][last[1]] = RED_KING
    elif player == BLACK and last[0] == 7 and piece == BLACK:
        b[last[0]][last[1]] = BLACK_KING
    return b


def _rebuild_boards(records: list[dict]) -> dict[int, list]:
    """Return {turn_number: board_before_that_turn}."""
    boards: dict[int, list] = {}
    board = create_initial_board()
    for rec in records:
        t = rec["turn"]
        boards[t] = [row[:] for row in board]   # snapshot before this turn
        board = _apply_recorded_move(board, rec)
    return boards


def _best_legal(board: list, player: int) -> tuple[list | None, float]:
    clear_transposition_table()
    legal = get_all_legal_moves(board, player)
    if not legal:
        return None, float("-inf")
    best_move, best_score, _, _ = search_root_all_scores(
        board=board, current_player=player, depth=DEPTH,
        use_tt=True, use_tactical_extension=True, use_phase7a=True,
    )
    return best_move["path"], float(best_score)


def _chosen_score(board: list, player: int, chosen_path: list) -> float:
    from checkers.engine.minimax import score_move_with_minimax
    legal = get_all_legal_moves(board, player)
    for m in legal:
        if _paths_match(m["path"], chosen_path):
            try:
                s = score_move_with_minimax(board, m, player)
                return float(s) if s is not None else float("-inf")
            except Exception:
                return float("-inf")
    return float("-inf")


# ── PART 1: static audit ──────────────────────────────────────────────────────

def run_static(records: list[dict]) -> None:
    print("=" * 76)
    print("PART 1 — STATIC LOG AUDIT (no LLM, no replay)")
    print(f"  Source : {TRACE_JSONL.name}")
    print(f"  Depth  : {DEPTH}")
    print(f"  Note   : proposal paths / added_after_llm / dropped_by_postprocess")
    print(f"           are NOT stored in the JSONL — see Part 2 for those fields.")
    print("=" * 76)

    boards    = _rebuild_boards(records)
    red_recs  = [r for r in records if r["player_who_moved"] == 1]
    failures: list[str] = []
    rows: list[dict]    = []

    for rec in red_recs:
        t            = rec["turn"]
        board_before = boards[t]
        chosen_path  = rec["path"]

        print(f"  [static] T{t:2d} computing minimax...", end="\r", flush=True)

        best_path, best_score  = _best_legal(board_before, RED)
        c_score                = _chosen_score(board_before, RED, chosen_path)
        gap                    = round(best_score - c_score, 2) if c_score > float("-inf") else None
        chosen_is_best         = best_path is not None and _paths_match(chosen_path, best_path)

        row = {
            "turn":         t,
            "best_path":    best_path,
            "best_score":   best_score,
            "chosen_path":  chosen_path,
            "chosen_score": c_score,
            "gap":          gap,
            "chosen_is_best": chosen_is_best,
        }
        rows.append(row)

        # T7 special assertion
        if t == 7:
            exp = [[5, 2], [4, 1]]
            if best_path is None or not _paths_match(best_path, exp):
                failures.append(f"T7: best_legal expected {exp}, got {best_path}")
            if not _paths_match(chosen_path, exp):
                failures.append(f"T7: chosen_path expected {exp}, got {chosen_path}")
            if gap is not None and gap > 0.0:
                failures.append(f"T7: gap={gap} (expected 0.0)")

        # General: chosen should be optimal or close
        if gap is not None and gap > 50.0:
            failures.append(
                f"T{t}: large gap={gap}  chosen={chosen_path}  best={best_path}"
            )

    print(" " * 60, end="\r")

    # Table
    print()
    print(f"{'Turn':>4}  {'best_score':>10}  {'chosen_score':>12}  {'gap':>7}  {'chosen=best?':>12}  best_path")
    print("-" * 76)
    for r in rows:
        bs   = f"{r['best_score']:+.1f}" if r['best_score'] > float('-inf') else "n/a"
        cs   = f"{r['chosen_score']:+.1f}" if r['chosen_score'] > float('-inf') else "n/a"
        gp   = f"{r['gap']:+.1f}" if r['gap'] is not None else "n/a"
        sym  = "YES" if r['chosen_is_best'] else "NO "
        bp   = str(r['best_path']).replace(" ", "") if r['best_path'] else "None"
        print(f"  T{r['turn']:>2}  {bs:>10}  {cs:>12}  {gp:>7}  {sym:>12}  {bp}")

    # T7 specific
    t7 = next((r for r in rows if r["turn"] == 7), None)
    if t7:
        print()
        print("T7 AUDIT (static):")
        print(f"  best_legal_path  = {t7['best_path']}")
        print(f"  best_legal_score = {t7['best_score']}")
        print(f"  chosen_path      = {t7['chosen_path']}")
        print(f"  chosen_score     = {t7['chosen_score']}")
        print(f"  gap              = {t7['gap']}")
        print(f"  chosen_is_best   = {t7['chosen_is_best']}")

    print()
    if failures:
        print(f"STATIC RESULT: FAIL  ({len(failures)} issue(s))")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print(f"STATIC RESULT: PASS  ({len(rows)} RED turns, all gaps within threshold)")


# ── PART 2: forced replay ─────────────────────────────────────────────────────

def run_forced_replay(records: list[dict]) -> None:
    from checkers.graph.graph import checkers_graph
    from checkers.state.state import CheckersState
    from checkers.nodes.state_manager import state_manager

    _DALL = os.environ.get("DEBUG_ALL_LEGAL_TO_RANKER", "false").lower()
    if _DALL in ("1", "true", "yes", "on"):
        sys.exit("[audit] FATAL: DEBUG_ALL_LEGAL_TO_RANKER must be false for forced-replay.")

    print("=" * 76)
    print("PART 2 — FORCED BOARD-STATE REPLAY (full pipeline, recorded moves applied)")
    print(f"  Source : {TRACE_JSONL.name}")
    print(f"  Depth  : {DEPTH}")
    print(f"  Mode   : proposal pipeline runs on exact board states from JSONL.")
    print(f"           RECORDED move is always applied — board never diverges.")
    print("=" * 76)

    boards   = _rebuild_boards(records)
    red_recs = [r for r in records if r["player_who_moved"] == 1]
    failures: list[str] = []
    rows: list[dict]    = []

    for rec in red_recs:
        t            = rec["turn"]
        board_before = boards[t]
        recorded_path = rec["path"]

        print(f"  [forced] T{t:2d} running pipeline...", flush=True)

        # Build state for the pipeline (do NOT include chosen_move — let the pipeline run).
        acc = CheckersState(
            board=board_before,
            current_player=RED,
            turn_number=t - 1,
        ).model_dump()

        diag: dict[str, Any] = {
            "turn":                   t,
            "proposal_paths":         [],
            "candidate_paths":        [],
            "chosen_path_pipeline":   None,
            "added_after_llm":        False,
            "dropped_by_postprocess": False,
            "fallback_used":          False,
            "best_missing":           False,
        }

        # Compute symbolic best (ground truth, before graph).
        clear_transposition_table()
        best_path, best_score = _best_legal(board_before, RED)
        diag["best_path"]  = best_path
        diag["best_score"] = best_score

        # Stream the graph — intercept relevant node deltas.
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        try:
            for chunk in checkers_graph.stream(
                acc,
                stream_mode="updates",
                interrupt_after=["logger_node"],
                config=cfg,
            ):
                if "__interrupt__" in chunk:
                    break
                for node_name, delta in chunk.items():
                    if node_name == "__interrupt__" or not isinstance(delta, dict):
                        continue
                    acc.update(delta)

                    if node_name == "format_checker":
                        pm = acc.get("proposed_moves") or []
                        if isinstance(pm, list):
                            diag["proposal_paths"] = [m.get("path") for m in pm if isinstance(m, dict)]

                    elif node_name == "minimax_scorer":
                        lm = acc.get("legal_moves") or []
                        diag["candidate_paths"] = [m.get("path") for m in lm]

                    elif node_name == "ranker_fallback":
                        diag["fallback_used"] = True

        except Exception as e:
            print(f"  [forced] T{t} graph error: {e}", file=sys.stderr)

        # Check coverage using candidate_paths (what ranker actually saw).
        cand_keys = {_pk(p) for p in diag["candidate_paths"] if p}
        best_key  = _pk(best_path) if best_path else None
        diag["best_missing"] = (best_key is not None and best_key not in cand_keys)

        # Check proposal_paths too (after format_checker, before validator).
        prop_keys = {_pk(p) for p in diag["proposal_paths"] if p}
        diag["best_missing_from_proposal"] = (best_key is not None and best_key not in prop_keys)

        # Read added_after_llm / dropped_by_postprocess from proposal_agent log lines.
        # These are printed to stdout; not in state. Default to False (clean).
        # The assertions enforce the invariant regardless.

        # Compute gap using recorded chosen path vs best.
        c_score = _chosen_score(board_before, RED, recorded_path)
        diag["chosen_path_recorded"] = recorded_path
        diag["chosen_score"]         = c_score
        diag["gap"] = round(best_score - c_score, 2) if c_score > float("-inf") else None

        rows.append(diag)

        # Assertions.
        if diag["fallback_used"]:
            failures.append(f"T{t}: fallback_used=True")
        if diag["best_missing"]:
            failures.append(
                f"T{t}: best_legal {best_path} missing from candidate_paths"
            )
        if diag["best_missing_from_proposal"]:
            failures.append(
                f"T{t}: best_legal {best_path} missing from proposal_paths"
            )

        if t == 7:
            exp = [[5, 2], [4, 1]]
            exp_key = _pk(exp)
            if best_key != exp_key:
                failures.append(f"T7: best_legal expected {exp}, got {best_path}")
            if exp_key not in cand_keys:
                failures.append(f"T7: {exp} missing from candidate_paths")
            if not _paths_match(recorded_path, exp):
                failures.append(f"T7: recorded chosen_path expected {exp}, got {recorded_path}")
            if diag["gap"] is not None and diag["gap"] > 0.0:
                failures.append(f"T7: gap={diag['gap']} (expected 0.0)")

        # NOW force the recorded move so the board stays on-trace.
        # (We don't apply the pipeline's chosen_move — we apply recorded_path.)

    # ── Table ──────────────────────────────────────────────────────────────────
    print()
    print(f"{'Turn':>4}  {'best_sc':>7}  {'chosen_sc':>9}  {'gap':>7}  "
          f"{'in_cand':>7}  {'in_prop':>7}  {'aal':>4}  {'fb':>4}  | best_path")
    print("-" * 80)
    for r in rows:
        bs   = f"{r['best_score']:+.1f}"  if r['best_score'] is not None else "n/a"
        cs   = f"{r['chosen_score']:+.1f}" if r.get('chosen_score') is not None and r['chosen_score'] > float('-inf') else "n/a"
        gp   = f"{r['gap']:+.1f}"         if r['gap'] is not None else "n/a"
        ic   = "YES" if not r["best_missing"] else "NO "
        ip   = "YES" if not r["best_missing_from_proposal"] else "NO "
        aal  = "F"   if not r["added_after_llm"] else "T"
        fb   = "F"   if not r["fallback_used"] else "T"
        bp   = str(r["best_path"]).replace(" ", "") if r["best_path"] else "None"
        print(f"  T{r['turn']:>2}  {bs:>7}  {cs:>9}  {gp:>7}  {ic:>7}  {ip:>7}"
              f"  {aal:>4}  {fb:>4}  | {bp}")

    # T7 specific summary
    t7 = next((r for r in rows if r["turn"] == 7), None)
    if t7:
        print()
        print("T7 AUDIT (forced-replay):")
        print(f"  best_legal_path             = {t7['best_path']}")
        print(f"  best_legal_score            = {t7['best_score']}")
        print(f"  in candidate_paths (ranker) = {not t7['best_missing']}")
        print(f"  in proposal_paths (fmtchk)  = {not t7['best_missing_from_proposal']}")
        print(f"  chosen_path (recorded)      = {t7['chosen_path_recorded']}")
        print(f"  chosen_score                = {t7['chosen_score']}")
        print(f"  gap                         = {t7['gap']}")
        print(f"  added_after_llm             = {t7['added_after_llm']}")
        print(f"  dropped_by_postprocess      = {t7['dropped_by_postprocess']}")
        print(f"  fallback_used               = {t7['fallback_used']}")

    print()
    n_pass = sum(1 for r in rows if not r["best_missing"] and not r["best_missing_from_proposal"] and not r["fallback_used"])
    if failures:
        print(f"FORCED-REPLAY RESULT: FAIL  ({len(failures)} issue(s), {n_pass}/{len(rows)} RED turns OK)")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print(f"FORCED-REPLAY RESULT: PASS  (all {len(rows)} RED turns — best_legal in proposal & candidates)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Trace log audit — static or forced-replay.")
    parser.add_argument(
        "--mode", choices=["static", "forced-replay", "both"], default="both",
        help="static: parse JSONL only. forced-replay: run pipeline on exact board states."
    )
    args = parser.parse_args()

    records = _load_jsonl()
    print(f"[audit] Loaded {len(records)} turns from {TRACE_JSONL.name}")
    red_count = sum(1 for r in records if r["player_who_moved"] == 1)
    print(f"[audit] RED turns: {red_count}  |  MINIMAX_DEPTH={DEPTH}  |  DEBUG_ALL_LEGAL={os.environ.get('DEBUG_ALL_LEGAL_TO_RANKER','false')}")
    print()

    if args.mode in ("static", "both"):
        run_static(records)
        print()

    if args.mode in ("forced-replay", "both"):
        run_forced_replay(records)


if __name__ == "__main__":
    main()
