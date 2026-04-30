#!/usr/bin/env python3
"""
Proposal pipeline diagnostic trace.

RED plays via the full LangGraph pipeline.
BLACK plays topk_random: scores all legal moves with minimax, picks uniformly
from the top-k.

Per RED turn, captures:
  - presentation order + mm_pin state (from state.proposal_diagnostics)
  - raw LLM indices vs actual best
  - postprocess state
  - validator output
  - final choice vs symbolic best
  - proposal_classification + final_classification

Output: JSONL (one object per RED turn) + printed summary.

Usage:
  venv/bin/python3 run_proposal_trace.py \\
      --opponent-top-k 3 --seed 123 --max-plies 60 \\
      --out logs/proposal_trace.jsonl
"""

from __future__ import annotations

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import argparse
import copy
import json
import os
import random
import sys
import uuid
from collections import Counter
from typing import Any

os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from checkers.graph.graph import checkers_graph
from checkers.state.state import CheckersState
from checkers.engine.board import RED, BLACK, create_initial_board
from checkers.engine.rules import get_all_legal_moves
from checkers.nodes.state_manager import state_manager
from checkers.nodes.symbolic_decision import _score_all_moves


# ── Helpers ───────────────────────────────────────────────────────────────────

def _path_key(path: list) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _top3_minimax_paths(symbolic_scored_moves):
    scored = []
    for entry in symbolic_scored_moves or []:
        score = entry.get("minimax_score", entry.get("score"))
        move = entry.get("move")
        if score is None or not move or "path" not in move:
            continue
        scored.append((float(score), _path_key(move["path"])))
    scored.sort(key=lambda x: -x[0])
    return [path for _, path in scored[:3]]


def _top3_coverage(symbolic_scored_moves, candidate_moves):
    top3 = _top3_minimax_paths(symbolic_scored_moves)
    candidate_paths = {
        _path_key(m["path"])
        for m in (candidate_moves or [])
        if isinstance(m, dict) and "path" in m
    }
    present = [p in candidate_paths for p in top3]
    return {
        "top3_required_count": len(top3),
        "top3_present_count": sum(present),
        "proposal_contains_top1": present[0] if len(present) > 0 else None,
        "proposal_contains_top2": present[1] if len(present) > 1 else None,
        "proposal_contains_top3": present[2] if len(present) > 2 else None,
        "proposal_contains_all_top3": all(present) if top3 else None,
        "missing_top3_count": len(top3) - sum(present),
    }

def _chosen_top3_metrics(symbolic_scored_moves, chosen_path):
    top3 = _top3_minimax_paths(symbolic_scored_moves)
    if chosen_path is None or not top3:
        return {
            "chosen_is_top1": None,
            "chosen_is_top2": None,
            "chosen_is_top3": None,
            "chosen_in_top3": None,
        }
    ck = _path_key(chosen_path)
    hits = [ck == p for p in top3]
    return {
        "chosen_is_top1": hits[0] if len(hits) > 0 else None,
        "chosen_is_top2": hits[1] if len(hits) > 1 else None,
        "chosen_is_top3": hits[2] if len(hits) > 2 else None,
        "chosen_in_top3": any(hits),
    }


def _get_score_for_path(path: list | None, scored_moves: list[dict]) -> float | None:
    if path is None:
        return None
    pk = _path_key(path)
    for entry in scored_moves:
        if _path_key(entry["move"]["path"]) == pk:
            return entry["minimax_score"]
    return None


# ── topk_random opponent ──────────────────────────────────────────────────────

def _topk_random_move(
    board: list[list[int]],
    player: int,
    rng: random.Random,
    k: int,
    depth: int,
) -> dict | None:
    """Score all legal moves for player, pick uniformly from top-k."""
    legal = get_all_legal_moves(board, player)
    if not legal:
        return None
    scored = _score_all_moves(board, legal, player, depth=depth)
    top_k = scored[: min(k, len(scored))]
    move, _ = rng.choice(top_k)
    return move


# ── Classification ────────────────────────────────────────────────────────────

def _classify_proposal(diag: dict | None, n_legal: int) -> str:
    """
    Classify the proposal stage outcome.

    BEST_VISIBLE_AND_RAW_INCLUDED — best in window AND raw LLM selected it
    BEST_VISIBLE_BUT_RAW_MISSED   — best in window, raw LLM did not pick it
    BEST_HIDDEN_FROM_LLM          — best outside presentation window (pos >= min(5,n))
    POSTPROCESS_DROPPED_BEST      — raw LLM had best, postprocess dropped it
    PROPOSAL_DIAGNOSTICS_MISSING  — proposal_diagnostics is None (fallback/API error)
    """
    if diag is None:
        return "PROPOSAL_DIAGNOSTICS_MISSING"

    src = diag.get("proposal_source")
    if src == "fallback":
        return "FALLBACK_PROPOSAL"
    if src == "llm_parse_error":
        return "LLM_PARSE_ERROR_PROPOSAL"

    actual_best_pres_idx = diag.get("actual_best_pres_idx")
    effective_window = min(5, n_legal)

    if actual_best_pres_idx is None or actual_best_pres_idx >= effective_window:
        return "BEST_HIDDEN_FROM_LLM"

    # Best is visible (pos < window)
    if diag.get("dropped_by_postprocess"):
        return "POSTPROCESS_DROPPED_BEST"

    if diag.get("raw_llm_selected_actual_best"):
        return "BEST_VISIBLE_AND_RAW_INCLUDED"

    return "BEST_VISIBLE_BUT_RAW_MISSED"


def _classify_final(gap: float | None, chosen_path: list | None, best_path: list | None) -> str:
    """
    Classify the final decision.

    BEST_CHOSEN           — chosen == symbolic best (gap ≤ 0.5)
    RANKER_CHOSE_WORSE    — chosen != best (gap > 0.5)
    FINAL_DECISION_MISSING — chosen_path or best_path or gap is None
    """
    if chosen_path is None or best_path is None or gap is None:
        return "FINAL_DECISION_MISSING"
    if gap <= 0.5:
        return "BEST_CHOSEN"
    return "RANKER_CHOSE_WORSE"


# ── RED turn runner ───────────────────────────────────────────────────────────

def _run_red_turn(
    acc: dict[str, Any],
    turn_no: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run one RED turn through the LangGraph pipeline.
    Diagnostics are read from state.proposal_diagnostics (set by proposal_agent,
    snapshotted before state_manager clears it).
    Returns (updated_acc, turn_record).
    """
    board_snapshot = copy.deepcopy(acc["board"])
    n_legal_pre = len(get_all_legal_moves(board_snapshot, RED))

    symbolic_scored_snap: list[dict] = []
    legal_after_validator_snap: list[dict] = []
    proposal_diag_snap: dict | None = None
    proposal_source_snap: str | None = None
    fallback_reason_snap: str | None = None
    format_checker_rejection_count: int = 0
    format_checker_last_rejection: str | None = None
    chosen_path_snap: list | None = None

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
                if node_name == "__interrupt__":
                    continue
                if not isinstance(delta, dict):
                    continue
                acc.update(delta)
                if node_name == "symbolic_decision":
                    symbolic_scored_snap = copy.deepcopy(
                        acc.get("symbolic_scored_moves", [])
                    )
                elif node_name == "proposal_agent":
                    # Snapshot from delta — state_manager will clear this field later.
                    proposal_diag_snap = copy.deepcopy(
                        delta.get("proposal_diagnostics")
                    )
                    proposal_source_snap = (proposal_diag_snap or {}).get("proposal_source")
                    fallback_reason_snap = (proposal_diag_snap or {}).get("fallback_reason")
                elif node_name == "validator":
                    legal_after_validator_snap = copy.deepcopy(
                        acc.get("legal_moves", [])
                    )
                elif node_name == "format_checker":
                    fc_feedback = delta.get("feedback")
                    if fc_feedback:
                        format_checker_rejection_count += 1
                        fc_str = str(fc_feedback)
                        if "MISSING_REQUIRED_MINIMAX_RANKS" in fc_str:
                            format_checker_last_rejection = "MISSING_REQUIRED_MINIMAX_RANKS"
                        elif "INSUFFICIENT_PROPOSAL_COUNT" in fc_str:
                            format_checker_last_rejection = "INSUFFICIENT_PROPOSAL_COUNT"
                        elif "FORMAT_ERROR" in fc_str:
                            format_checker_last_rejection = "FORMAT_ERROR"
                        else:
                            format_checker_last_rejection = "OTHER"
                elif node_name == "state_manager":
                    # state_manager sets chosen_move=None but appends to move_history.
                    mh = acc.get("move_history", [])
                    if mh:
                        chosen_path_snap = mh[-1]["move"].get("path")
    except Exception as e:
        print(f"[run_proposal_trace] graph error turn={turn_no}: {e}", file=sys.stderr)

    # Symbolic best from the scored snapshot
    best_entry = symbolic_scored_snap[0] if symbolic_scored_snap else None
    best_path = best_entry["move"]["path"] if best_entry else None
    best_score = best_entry["minimax_score"] if best_entry else None

    chosen_score = _get_score_for_path(chosen_path_snap, symbolic_scored_snap)
    gap: float | None = None
    if best_score is not None and chosen_score is not None:
        gap = round(best_score - chosen_score, 2)

    # Did symbolic best survive the validator?
    best_survived_validator = False
    if best_path is not None:
        bk = _path_key(best_path)
        best_survived_validator = any(
            _path_key(m.get("path", [])) == bk
            for m in legal_after_validator_snap
        )

    n_legal = len(symbolic_scored_snap) if symbolic_scored_snap else n_legal_pre
    effective_window = min(5, n_legal)

    # best_visible: True/False when known, None when unknown
    if n_legal == 1:
        best_visible: bool | None = True  # single move always at pos 0
    elif proposal_diag_snap is None:
        best_visible = None  # diagnostics missing
    else:
        idx = proposal_diag_snap.get("actual_best_pres_idx")
        if idx is None:
            best_visible = None  # best path not matched in presentation
        else:
            best_visible = idx < effective_window

    mm_pin_pres_idx = (
        proposal_diag_snap.get("mm_pinned_pres_idx") if proposal_diag_snap else None
    )
    mm_pin_fired = mm_pin_pres_idx is not None
    mm_pin_points_to_best = mm_pin_fired and (
        mm_pin_pres_idx == (
            proposal_diag_snap.get("actual_best_pres_idx") if proposal_diag_snap else None
        )
    )

    proposal_cls = _classify_proposal(proposal_diag_snap, n_legal)
    final_cls = _classify_final(gap, chosen_path_snap, best_path)

    _d = proposal_diag_snap or {}
    top3_cov = _top3_coverage(symbolic_scored_snap, legal_after_validator_snap)
    chosen_top3 = _chosen_top3_metrics(symbolic_scored_snap, chosen_path_snap)
    record: dict[str, Any] = {
        "game_id": acc.get("game_log_id"),
        "turn": turn_no,
        "player": "RED",
        "board": board_snapshot,
        # Legal moves layer
        "n_legal": n_legal,
        "best_path": best_path,
        "best_score": best_score,
        # Presentation layer
        "best_pres_idx": _d.get("actual_best_pres_idx"),
        "effective_window": effective_window,
        "best_visible": best_visible,
        "mm_pin_fired": mm_pin_fired,
        "mm_pin_pres_idx": mm_pin_pres_idx,
        "mm_pin_points_to_best": mm_pin_points_to_best,
        # LLM raw output
        "raw_pres_indices": _d.get("raw_pres_indices"),
        "raw_llm_selected_actual_best": _d.get("raw_llm_selected_actual_best"),
        # Postprocess output
        "final_pres_indices": _d.get("final_pres_indices"),
        "final_contains_actual_best": _d.get("final_contains_actual_best"),
        "dropped_by_postprocess": _d.get("dropped_by_postprocess"),
        "added_after_llm": _d.get("added_after_llm"),
        # Validator output
        "validator_n_candidates": len(legal_after_validator_snap),
        "best_survived_validator": best_survived_validator,
        # Final decision
        "chosen_path": chosen_path_snap,
        "chosen_score": chosen_score,
        "gap": gap,
        # Classification (split into two stages)
        "proposal_classification": proposal_cls,
        "final_classification": final_cls,
        # Top-3 minimax coverage
        "top3_required_count": top3_cov["top3_required_count"],
        "top3_present_count": top3_cov["top3_present_count"],
        "proposal_contains_top1": top3_cov["proposal_contains_top1"],
        "proposal_contains_top2": top3_cov["proposal_contains_top2"],
        "proposal_contains_top3": top3_cov["proposal_contains_top3"],
        "proposal_contains_all_top3": top3_cov["proposal_contains_all_top3"],
        "missing_top3_count": top3_cov["missing_top3_count"],
        # Chosen-move rank coverage
        "chosen_is_top1": chosen_top3["chosen_is_top1"],
        "chosen_is_top2": chosen_top3["chosen_is_top2"],
        "chosen_is_top3": chosen_top3["chosen_is_top3"],
        "chosen_in_top3": chosen_top3["chosen_in_top3"],
        # Proposal origin
        "proposal_source": proposal_source_snap,
        "fallback_reason": fallback_reason_snap,
        "format_checker_rejections": format_checker_rejection_count,
        "format_checker_last_rejection": format_checker_last_rejection,
        "transient_retry_count": _d.get("transient_retry_count"),
        "used_retry": _d.get("used_retry"),
        "prompt_metrics": _d.get("prompt_metrics"),
    }
    return acc, record


# ── BLACK turn runner ─────────────────────────────────────────────────────────

def _run_black_turn(
    acc: dict[str, Any],
    rng: random.Random,
    k: int,
    depth: int,
) -> dict[str, Any]:
    """Apply one topk_random BLACK move and run logger."""
    board = acc["board"]
    move = _topk_random_move(board, BLACK, rng, k, depth)
    if move is None:
        return acc

    st = CheckersState.model_validate(acc)
    st.chosen_move = move
    st.last_move_reasoning = f"topk_random(k={k})"
    patch = state_manager(st)
    acc.update(patch)

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
                if node_name == "__interrupt__":
                    continue
                if isinstance(delta, dict):
                    acc.update(delta)
    except Exception as e:
        print(f"[run_proposal_trace] BLACK graph error: {e}", file=sys.stderr)

    return acc


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(records: list[dict[str, Any]], out_path: str) -> None:
    n = len(records)
    if n == 0:
        print("No RED turns recorded.")
        return

    llm_records = [r for r in records if r.get("proposal_source") == "llm"]
    visible = sum(1 for r in records if r.get("best_visible") is True)
    visible_known = sum(1 for r in records if r.get("best_visible") is not None)
    raw_ok = sum(1 for r in records if r.get("raw_llm_selected_actual_best") is True)
    raw_known = sum(1 for r in records if r.get("raw_llm_selected_actual_best") is not None)
    final_ok = sum(1 for r in records if r.get("final_contains_actual_best") is True)
    final_known = sum(1 for r in records if r.get("final_contains_actual_best") is not None)
    chosen_best = sum(1 for r in records if r.get("final_classification") == "BEST_CHOSEN")

    prop_counts = Counter(r["proposal_classification"] for r in records)
    final_counts = Counter(r["final_classification"] for r in records)

    first_fail: dict[str, int] = {}
    for r in records:
        cls = r["final_classification"]
        if cls not in ("BEST_CHOSEN", "FINAL_DECISION_MISSING") and cls not in first_fail:
            first_fail[cls] = r["turn"]

    bar = "═" * 60
    print()
    print(bar)
    print("PROPOSAL TRACE SUMMARY")
    print(bar)
    print(f"Total RED turns:         {n}")
    if visible_known:
        print(
            f"Best visible (top-5):    {visible}/{visible_known} = {visible/visible_known*100:.1f}%"
            + (f"  ({n - visible_known} unknown)" if n > visible_known else "")
        )
    else:
        print("Best visible:            (all unknown — diagnostics missing)")
    if raw_known:
        print(
            f"Raw LLM had best:        {raw_ok}/{raw_known} = {raw_ok/raw_known*100:.1f}%"
            + (f"  ({n - raw_known} missing)" if n > raw_known else "")
        )
    else:
        print("Raw LLM had best:        (no data)")
    if final_known:
        print(f"Final had best:          {final_ok}/{final_known} = {final_ok/final_known*100:.1f}%")
    print(f"Chosen == best (top1):   {chosen_best}/{n} = {chosen_best/n*100:.1f}%")
    if llm_records:
        chosen_best_llm = sum(1 for r in llm_records if r.get("final_classification") == "BEST_CHOSEN")
        print(f"  (LLM turns only: {chosen_best_llm}/{len(llm_records)} = {chosen_best_llm/len(llm_records)*100:.1f}%)")
    top3_known = [r for r in records if r.get("proposal_contains_all_top3") is not None]
    top3_all = sum(1 for r in top3_known if r.get("proposal_contains_all_top3"))
    if top3_known:
        print(
            f"Proposal contains all top-3: {top3_all}/{len(top3_known)} = "
            f"{100 * top3_all / len(top3_known):.1f}%"
        )
    else:
        print("Proposal contains all top-3: no data")
    chosen_in_top3_known = [r for r in records if r.get("chosen_in_top3") is not None]
    chosen_in_top3_cnt = sum(1 for r in chosen_in_top3_known if r.get("chosen_in_top3"))
    if chosen_in_top3_known:
        print(
            f"Chosen in top-3:         {chosen_in_top3_cnt}/{len(chosen_in_top3_known)} = "
            f"{100 * chosen_in_top3_cnt / len(chosen_in_top3_known):.1f}%"
        )
    else:
        print("Chosen in top-3:         no data")
    ranker_outside_top3 = sum(1 for r in records if r.get("chosen_in_top3") is False)
    print(f"Ranker chose outside top-3: {ranker_outside_top3}/{len(chosen_in_top3_known)}")
    print()
    src_llm = sum(1 for r in records if r.get("proposal_source") == "llm")
    src_fallback = sum(1 for r in records if r.get("proposal_source") == "fallback")
    src_parse_err = sum(1 for r in records if r.get("proposal_source") == "llm_parse_error")
    src_unknown = sum(1 for r in records if r.get("proposal_source") not in ("llm", "fallback", "llm_parse_error"))
    print("Proposal source:")
    print(f"  LLM proposal used:    {src_llm}")
    print(f"  Fallback proposal:    {src_fallback}")
    print(f"  Parse-error proposal: {src_parse_err}")
    print(f"  Unknown source:       {src_unknown}")
    if src_unknown > 0:
        first_unknown = next(
            (r["turn"] for r in records if r.get("proposal_source") not in ("llm", "fallback", "llm_parse_error")),
            None,
        )
        print(f"    (Unknown = proposal_diagnostics missing entirely; first at turn {first_unknown})")
    fallback_records = [r for r in records if r.get("proposal_source") == "fallback"]
    if fallback_records:
        from collections import Counter as _Counter
        reason_counts = _Counter(r.get("fallback_reason") or "unknown" for r in fallback_records)
        print("  Fallback reasons:")
        for reason, cnt in sorted(reason_counts.items()):
            print(f"    {reason}: {cnt}")
    print()
    fc_total = sum(r.get("format_checker_rejections") or 0 for r in records)
    fc_turns = sum(1 for r in records if (r.get("format_checker_rejections") or 0) > 0)
    fc_types = Counter(r.get("format_checker_last_rejection") for r in records if r.get("format_checker_last_rejection"))
    print(f"Format checker rejections: {fc_total} total, {fc_turns} turns affected")
    for fc_type, cnt in sorted(fc_types.items()):
        print(f"  {fc_type}: {cnt} turns")
    print()
    llm_no_retry = sum(1 for r in llm_records if r.get("used_retry") is False)
    llm_with_retry = sum(1 for r in llm_records if r.get("used_retry") is True)
    total_retry_attempts = sum(r.get("transient_retry_count") or 0 for r in records)
    print("Retry usage (LLM turns):")
    print(f"  LLM without retry:    {llm_no_retry}")
    print(f"  LLM after retry:      {llm_with_retry}")
    print(f"  Total retry attempts: {total_retry_attempts}")
    print()
    print("Proposal classification:")
    for cls, cnt in sorted(prop_counts.items()):
        print(f"  {cls}: {cnt}")
    print()
    print("Final classification:")
    for cls, cnt in sorted(final_counts.items()):
        marker = "  " if cls == "BEST_CHOSEN" else "! "
        first = f"  (first at turn {first_fail[cls]})" if cls in first_fail else ""
        print(f"  {marker}{cls}: {cnt}{first}")
    print()
    print(f"JSONL written to: {out_path}")
    print(bar)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proposal pipeline diagnostic trace (RED=LLM, BLACK=topk_random)."
    )
    parser.add_argument("--max-plies", type=int, default=120,
                        help="Safety cap on total plies (both sides).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for topk_random opponent.")
    parser.add_argument("--opponent-top-k", type=int, default=3,
                        help="Pick BLACK move uniformly from top-k minimax moves.")
    parser.add_argument("--depth", type=int, default=None,
                        help="Minimax depth for topk_random scoring (default: MINIMAX_DEPTH env).")
    parser.add_argument("--out", default="logs/proposal_trace.jsonl",
                        help="Output JSONL path.")
    args = parser.parse_args()

    depth = args.depth or int(os.environ.get("MINIMAX_DEPTH", "3"))
    rng = random.Random(args.seed)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    print(
        f"[run_proposal_trace] seed={args.seed}  top_k={args.opponent_top_k}  "
        f"depth={depth}  max_plies={args.max_plies}  out={args.out}"
    )

    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()

    records: list[dict[str, Any]] = []
    ply = 0

    with open(args.out, "w", encoding="utf-8") as f_out:
        while True:
            if acc.get("game_over"):
                break
            if ply >= args.max_plies:
                print(
                    f"[run_proposal_trace] max_plies={args.max_plies} reached.",
                    file=sys.stderr,
                )
                break

            if acc["current_player"] == RED:
                turn_no = acc.get("turn_number", 0) + 1
                print(f"[run_proposal_trace] RED turn {turn_no} ...", end=" ", flush=True)
                acc, record = _run_red_turn(acc, turn_no)
                record["game_id"] = acc.get("game_log_id")
                records.append(record)
                f_out.write(json.dumps(record, default=str) + "\n")
                f_out.flush()
                gap = record.get("gap")
                gap_str = f"gap={gap:.1f}" if gap is not None else "gap=?"
                print(
                    f"prop={record['proposal_classification']}  "
                    f"final={record['final_classification']}  {gap_str}"
                )
                if record.get("final_classification") == "FINAL_DECISION_MISSING":
                    print(
                        f"[run_proposal_trace] STOP: RED turn {turn_no} did not complete "
                        f"(proposal={record.get('proposal_classification')}, "
                        f"source={record.get('proposal_source')}, "
                        f"format_checker_rejections={record.get('format_checker_rejections')}).",
                        file=sys.stderr,
                    )
                    break
            else:
                acc = _run_black_turn(acc, rng, args.opponent_top_k, depth)

            ply += 1

    _print_summary(records, args.out)


if __name__ == "__main__":
    main()
