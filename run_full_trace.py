#!/usr/bin/env python3
# run_full_trace.py — original pipeline runner
# (symbolic_decision → proposal_agent → format_checker → validator → minimax_scorer → ranker_agent → state_manager → win_condition → logger_node)
"""
Full game trace: RED via LangGraph (proposal + ranker pipeline), BLACK via human input.
Logs still go to logs/ via logger_node (terminal logger output suppressed by default env).
"""

from __future__ import annotations
from dotenv import load_dotenv # type: ignore
load_dotenv()  # loads .env automatically before anything else

import argparse
import copy
import json
import os
import random
import sys
import uuid
from typing import Any, Optional

# Quiet logger_node stdout while tracing here
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from checkers.graph.graph import checkers_graph
from checkers.state.state import CheckersState
from checkers.engine.board import RED, BLACK, create_initial_board, print_board
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves
from checkers.nodes.state_manager import state_manager

BAR = "═" * 56
RULE = "─" * 56

_ANSI_RED = "\033[91m"
_ANSI_RESET = "\033[0m"


def red_if(val, cond):
    return f"{_ANSI_RED}{val}{_ANSI_RESET}" if cond else val


def _as_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    return dict(chunk)


def _merge_stream_updates(
    acc: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """
    Run graph until interrupt_after logger_node. Merge each node's delta into acc.
    Returns (acc, interrupted_ok) where interrupted_ok is False if graph ended before logger.
    """
    saw_logger = False
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
                if node_name == "logger_node":
                    saw_logger = True
    except Exception as e:
        print(f"[run_full_trace] graph stream error: {e}", file=sys.stderr)
        return acc, False
    return acc, saw_logger


def _player_label(p: int) -> str:
    return "RED" if p == RED else "BLACK"


def _fmt_engine_move(i: int, m: dict[str, Any]) -> str:
    return (
        f"[{i}] type={m.get('type')} path={m.get('path')} "
        f"captured={m.get('captured', [])}"
    )


def _fmt_proposed_after_format(i: int, m: dict[str, Any]) -> str:
    t = m.get("type", "?")
    path = m.get("path", [])
    cap = m.get("captured") or []
    if cap:
        return f"  [{i}] {t}   {path} captures {cap}"
    return f"  [{i}] {t} {path}"


def _fmt_validator_move(i: int, m: dict[str, Any]) -> str:
    facts = m.get("facts") or {}
    ng = facts.get("net_gain", "n/a")
    orc = facts.get("opponent_can_recapture", "n/a")
    mm = facts.get("minimax_score", "n/a")
    cps = facts.get("counterplay_score", "n/a")
    kas = facts.get("king_activity_score", "n/a")
    path = m.get("path", [])
    t = m.get("type", "?")
    return (
        f"  [{i}] {t} {path}  net_gain={ng}  opp_recapture={orc}  "
        f"minimax={mm}  counterplay={cps}  king_activity={kas}"
    )

def _print_decision_audit(
    legal_moves: list[dict[str, Any]],
    chosen_move: Optional[dict[str, Any]],
) -> None:
    print("── DECISION AUDIT ──")
    if not legal_moves:
        print("  (no legal moves)")
        print()
        return

    chosen_idx = -1
    for i, m in enumerate(legal_moves):
        if (
            chosen_move
            and m.get("type") == chosen_move.get("type")
            and m.get("path") == chosen_move.get("path")
        ):
            chosen_idx = i
            break

    for i, m in enumerate(legal_moves):
        facts = m.get("facts") or {}
        marker = "★" if i == chosen_idx else " "
        print(
            f"{marker} [{i}] {m.get('path')} | "
            f"minimax={facts.get('minimax_score', 'n/a')} | "
            f"threat_after={facts.get('our_pieces_threatened_after', 'n/a')} | "
            f"counterplay={facts.get('counterplay_score', 'n/a')} | "
            f"king_act={facts.get('king_activity_score', 'n/a')} | "
            f"convert={facts.get('winning_conversion_score', 'n/a')} | "
            f"threat={facts.get('creates_immediate_threat', 'n/a')} | "
            f"shot={facts.get('shot_sequence_available', 'n/a')} | "
            f"isolated={facts.get('leaves_piece_isolated', 'n/a')} | "
            f"center={facts.get('center_control', 'n/a')} | "
            f"role={facts.get('quiet_move_role', 'n/a')}"
        )
    print()


def _fmt_applied_move(player: int, move: dict[str, Any], promotion: bool) -> str:
    path = move.get("path") or []
    cap = move.get("captured") or []
    if len(path) >= 2:
        a, b = path[0], path[-1]
        seg = f"{move.get('type', '?')} from [{a[0]},{a[1]}] to [{b[0]},{b[1]}] captured {cap}"
    else:
        seg = str(move)
    return f"{_player_label(player)} played: {seg}\nPromotion: {'Yes' if promotion else 'No'}"


def _strategic_block(ctx: Optional[dict[str, Any]]) -> None:
    if not ctx:
        print("── STRATEGIC CONTEXT ──")
        print("  (none)")
        return
    phase = ctx.get("game_phase", "unknown")
    winning = ctx.get("winning_score", "unknown")
    priorities = ctx.get("strategic_priorities") or []
    patterns = ctx.get("active_patterns") or []
    print("── STRATEGIC CONTEXT ──")
    print(f"Game phase: {phase}")
    print(f"Winning score: {winning}")
    print("Strategic priorities:")
    if priorities:
        for i, p in enumerate(priorities, 1):
            print(f"  {i}. {p}")
    else:
        print("  (none)")
    if patterns:
        print(f"Active patterns: {', '.join(str(x) for x in patterns)}")
    else:
        print("Active patterns: none")
    mat = ctx.get("material_trend", "?")
    ctr = ctx.get("center_trend", "?")
    print(f"Trends: material={mat} center={ctr}")


def _compute_game_totals(state: dict[str, Any]) -> dict[str, Any]:
    mh = state.get("move_history") or []
    total_captures_red = sum(
        len(r.get("move", {}).get("captured", []))
        for r in mh
        if r.get("player") == RED
    )
    total_captures_black = sum(
        len(r.get("move", {}).get("captured", []))
        for r in mh
        if r.get("player") == BLACK
    )
    total_promotions = sum(1 for r in mh if r.get("promotion", False))
    tn = state.get("turn_number") or 0
    fb = state.get("ranker_fallback_count") or 0
    rate = (fb / tn * 100.0) if tn > 0 else 0.0
    return {
        "total_captures_red": total_captures_red,
        "total_captures_black": total_captures_black,
        "total_promotions": total_promotions,
        "ranker_fallback_rate_pct": rate,
    }


def _winner_text(state: dict[str, Any]) -> str:
    if state.get("draw"):
        return "Draw"
    w = state.get("winner")
    if w == RED:
        return "RED"
    if w == BLACK:
        return "BLACK"
    return "N/A"


def _print_final_summary(state: dict[str, Any], quiet: bool) -> None:
    totals = _compute_game_totals(state)
    gid = state.get("game_log_id") or "(no game_log_id yet)"
    if quiet:
        print(BAR)
        print(f"Winner: {_winner_text(state)}  turns={state.get('turn_number')}")
        print(BAR)
        return
    print(BAR)
    print("GAME OVER")
    print(BAR)
    print(f"Winner: {_winner_text(state)}")
    print(f"Total turns: {state.get('turn_number')}")
    print(f"Total captures RED: {totals['total_captures_red']}")
    print(f"Total captures BLACK: {totals['total_captures_black']}")
    print(f"Total promotions: {totals['total_promotions']}")
    print(f"format_error_count: {state.get('format_error_count', 0)}")
    print(f"ranker_failure_count: {state.get('ranker_failure_count', 0)}")
    print(f"ranker_fallback_count: {state.get('ranker_fallback_count', 0)}")
    print(f"ranker_fallback_rate: {totals['ranker_fallback_rate_pct']:.1f}%")
    print("\nFinal board:")
    print_board(state["board"])
    print("\nLogs saved to:")
    print(f"  logs/{gid}.jsonl")
    print(f"  logs/summary_{gid}.json")
    print(BAR)


def _run_red_ply(acc: dict[str, Any], quiet: bool) -> dict[str, Any]:
    ctx_snapshot = copy.deepcopy(acc.get("strategic_context"))
    turn_no = acc.get("turn_number", 0)
    display_turn = turn_no + 1
    legal = get_all_legal_moves(acc["board"], RED)
    peak_retry = 0  # reset per-ply; never inherit stale retry_count from prior turn
    fe_before = acc.get("format_error_count", 0)  # snapshot for per-turn delta
    fb_before = acc.get("ranker_fallback_count", 0)
    ranker_fallback_used = False

    if not quiet:
        print(BAR)
        print(f"TURN {display_turn} | RED to move  (AI)")
        print(BAR)
        print("── BOARD BEFORE MOVE ──")
        print_board(acc["board"])
        print(f"\n── ENGINE LEGAL MOVES ({len(legal)} available) ──")
        for i, m in enumerate(legal):
            print(_fmt_engine_move(i, m))
        print()

    saw_logger = False

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
                peak_retry = max(peak_retry, acc.get("retry_count", 0))

                if quiet:
                    continue

                if node_name == "proposal_agent":
                    pass  # internal [proposal_agent] logs already printed by the node itself

                elif node_name == "format_checker":
                    pm = acc.get("proposed_moves")
                    print("── PROPOSAL AGENT ──")
                    print("Proposed moves after format_checker:")
                    if isinstance(pm, list) and pm:
                        for i, m in enumerate(pm):
                            print(_fmt_proposed_after_format(i, m))
                    else:
                        print("  (empty or invalid)")
                    # print rejection feedback when format_checker sent one
                    _fc_fb = acc.get("feedback")
                    if _fc_fb:
                        print(f"  [rejection feedback]: {str(_fc_fb)[:200]}")
                    print()

                elif node_name == "minimax_scorer":
                    lm = acc.get("legal_moves") or []
                    print("── MINIMAX SCORER ──")
                    print("Candidate moves after shallow search scoring:")
                    if lm:
                        for i, m in enumerate(lm):
                            facts = m.get("facts") or {}
                            print(
                                f"  [{i}] {m.get('type')} {m.get('path')}  "
                                f"minimax_score={facts.get('minimax_score', 'n/a')}  "
                                f"counterplay_score={facts.get('counterplay_score', 'n/a')}  "
                                f"king_activity_score={facts.get('king_activity_score', 'n/a')}"
                            )
                    else:
                        print("  (none)")
                    print()

                    # ── TRACE DIAGNOSTIC (READ-ONLY) ─────────────────────────
                    # Scores all legal moves with minimax (read-only, never sent
                    # to Ranker) and reports whether the best legal move overall
                    # was missing from the proposal shortlist.
                    _all_legal = get_all_legal_moves(acc["board"], RED)
                    _proposal_paths = frozenset(
                        tuple(tuple(sq) for sq in m.get("path", []))
                        for m in lm
                    )
                    # Find best proposed (from scored list)
                    _best_prop_score = float("-inf")
                    _best_prop_path  = None
                    for _m in lm:
                        _s = _m.get("facts", {}).get("minimax_score")
                        _s = float(_s) if _s is not None else float("-inf")
                        if _s > _best_prop_score:
                            _best_prop_score = _s
                            _best_prop_path  = _m.get("path")

                    # Score all legal moves with ONE joint search_root_all_scores call
                    # at PIPELINE_SCORER_DEPTH — same function and depth as minimax_scorer,
                    # shared TT, full window — so best_legal_score is on the same scale as
                    # the pipeline scores in lm[*].facts["minimax_score"].
                    from checkers.search.minimax_core import search_root_all_scores as _sra
                    from checkers.oldfiles.minimax_scorer import PIPELINE_SCORER_DEPTH as _diag_depth
                    _best_legal_score = float("-inf")
                    _best_legal_path  = None
                    _best_legal_missing = False
                    try:
                        _, _, _all_scored, _ = _sra(acc["board"], RED, _diag_depth, _all_legal)
                        if _all_scored:
                            _best_legal_path = _all_scored[0][0].get("path")
                            _best_legal_score = float(_all_scored[0][1])
                            _p = tuple(tuple(sq) for sq in _all_scored[0][0].get("path", []))
                            _best_legal_missing = _p not in _proposal_paths
                    except Exception:
                        pass

                    print("── TRACE DIAGNOSTIC (READ-ONLY) ──")
                    print(f"best_legal_overall:   path={_best_legal_path}  minimax_score={_best_legal_score:.1f}")
                    print(f"best_in_proposal:     path={_best_prop_path}  minimax_score={_best_prop_score:.1f}")
                    print(f"best_legal_missing_from_proposal: {_best_legal_missing}")
                    print(f"legal_count: {len(_all_legal)}  proposal_count: {len(lm)}")
                    print()
                    # ── END TRACE DIAGNOSTIC ─────────────────────────────────

                elif node_name == "ranker_agent":
                    cm = acc.get("chosen_move")
                    lm = acc.get("legal_moves") or []
                    print("── RANKER AGENT ──")
                    if cm and lm:
                        idx = -1
                        chosen_facts = {}
                        for i, m in enumerate(lm):
                            if m.get("type") == cm.get("type") and m.get("path") == cm.get("path"):
                                idx = i
                                chosen_facts = m.get("facts") or {}
                                break

                        # Determine decision source from ranker_diagnostics
                        _rd_now = acc.get("ranker_diagnostics") or {}
                        _orr_now = _rd_now.get("override_retry_resolved", False)
                        _ofa_now = _rd_now.get("override_fallback_applied", False)
                        if _orr_now:
                            _via_tag = "retry_llm"
                        elif _ofa_now:
                            _via_tag = "python_fallback"
                        else:
                            _via_tag = "original_llm"

                        print(f"Chose index: {idx}  [via={_via_tag}]")
                        cap = cm.get("captured") or []
                        path = cm.get("path", [])
                        if cap:
                            print(f"Move: {cm.get('type')} {path} captures {cap}")
                        else:
                            print(f"Move: {cm.get('type')} {path}")

                        print(
                            f"Chosen facts: minimax_score={chosen_facts.get('minimax_score', 'n/a')}  "
                            f"net_gain={chosen_facts.get('net_gain', 'n/a')}  "
                            f"opp_recapture={chosen_facts.get('opponent_can_recapture', 'n/a')}  "
                            f"counterplay_score={chosen_facts.get('counterplay_score', 'n/a')}  "
                            f"king_activity_score={chosen_facts.get('king_activity_score', 'n/a')}  "
                            f"Opening/Conversion facts: "
                            f"winning_conversion_score={chosen_facts.get('winning_conversion_score', 'n/a')}  "
                            f"creates_immediate_threat={chosen_facts.get('creates_immediate_threat', 'n/a')}  "
                            f"shot_sequence_available={chosen_facts.get('shot_sequence_available', 'n/a')}  "
                            f"restriction_score={chosen_facts.get('restriction_score', 'n/a')}  "
                            f"frozen_enemy_pieces={chosen_facts.get('frozen_enemy_pieces', 'n/a')}  "
                            f"center_control={chosen_facts.get('center_control', 'n/a')}  "
                            f"quiet_move_role={chosen_facts.get('quiet_move_role', 'n/a')}"
                        )

                        r = acc.get("last_move_reasoning") or ""
                        print(f"Reasoning: {r!r}")
                                                # ── RANKER QUALITY CHECK ──
                        best_idx = None
                        best_score = float("-inf")
                        best_move = None

                        for j, cand in enumerate(lm):
                            cand_facts = cand.get("facts") or {}
                            cand_score = cand_facts.get("minimax_score", float("-inf"))
                            try:
                                cand_score = float(cand_score)
                            except (TypeError, ValueError):
                                cand_score = float("-inf")

                            if cand_score > best_score:
                                best_score = cand_score
                                best_idx = j
                                best_move = cand

                        chosen_score = chosen_facts.get("minimax_score", float("-inf"))
                        try:
                            chosen_score = float(chosen_score)
                        except (TypeError, ValueError):
                            chosen_score = float("-inf")

                        best_facts = (best_move or {}).get("facts") or {}
                        minimax_gap = best_score - chosen_score if best_idx is not None else 0.0

                        print("── RANKER QUALITY CHECK ──")
                        print(f"best_legal_idx: {best_idx}")
                        print(f"best_legal_minimax: {best_score}")
                        print(f"chosen_idx: {idx}")
                        print(f"chosen_minimax: {chosen_score}")
                        print(f"minimax_gap_from_best: {minimax_gap}")
                        print(
                            f"chosen_threat_after: "
                            f"{chosen_facts.get('our_pieces_threatened_after', 'n/a')}"
                        )
                        print(
                            f"best_threat_after: "
                            f"{best_facts.get('our_pieces_threatened_after', 'n/a')}"
                        )
                        print(
                            f"chosen_center: {chosen_facts.get('center_control', False)}"
                        )
                        print(
                            f"best_center: {best_facts.get('center_control', False)}"
                        )
                        print(
                            f"chosen_isolated: "
                            f"{chosen_facts.get('leaves_piece_isolated', False)}"
                        )
                        print(
                            f"best_isolated: "
                            f"{best_facts.get('leaves_piece_isolated', False)}"
                        )
                        print(
                            f"chosen_weakens_back: "
                            f"{chosen_facts.get('weakens_king_row', False)}"
                        )
                        print(
                            f"best_weakens_back: "
                            f"{best_facts.get('weakens_king_row', False)}"
                        )
                        print()
                        _print_decision_audit(lm, cm)
                    else:
                        print("Chose index: (none — ranker failure, may retry)")
                    print()

                elif node_name == "state_manager":
                    mh = acc.get("move_history") or []
                    if mh:
                        last = mh[-1]
                        chosen_for_applied = last.get("move")
                        promotion_flag = bool(last.get("promotion"))
                        player_who_moved = last.get("player", RED)
                    else:
                        chosen_for_applied = None
                        promotion_flag = False
                        player_who_moved = RED
                    print("── MOVE APPLIED ──")
                    if chosen_for_applied:
                        print(
                            _fmt_applied_move(
                                int(player_who_moved),
                                chosen_for_applied,
                                promotion_flag,
                            )
                        )
                    print("\n── BOARD AFTER MOVE ──")
                    print_board(acc["board"])
                    rc = count_pieces(acc["board"], RED)
                    bc = count_pieces(acc["board"], BLACK)
                    mat = (ctx_snapshot or {}).get("material_advantage", "n/a")
                    print("\n── PIECE COUNTS ──")
                    print(
                        f"RED:   {rc['total']} pieces ({rc['regular']} regular, {rc['kings']} kings)"
                    )
                    print(
                        f"BLACK: {bc['total']} pieces ({bc['regular']} regular, {bc['kings']} kings)"
                    )
                    print(f"Material advantage: {mat}")
                    print()
                    _strategic_block(ctx_snapshot)
                    print()
                    print("── METRICS THIS TURN ──")
                    _fe_this_turn = acc.get('format_error_count', 0) - fe_before
                    _ranker_failures = acc.get('ranker_failure_count', 0)
                    _ranker_fallbacks = acc.get('ranker_fallback_count', 0)
                    print(
                        f"format_errors_this_turn={red_if(_fe_this_turn, _fe_this_turn > 0)}  "
                        f"format_errors_cumulative={acc.get('format_error_count', 0)}  "
                        f"ranker_failures={red_if(_ranker_failures, _ranker_failures > 0)}  "
                        f"ranker_fallbacks={red_if(_ranker_fallbacks, _ranker_fallbacks > 0)}"
                    )
                    print(f"format_checker_retry_count={red_if(peak_retry, peak_retry > 0)}")
                    print(
                        f"ranker_fallback_used_this_turn="
                        f"{ranker_fallback_used or (acc.get('ranker_fallback_count', 0) > fb_before)}"
                    )
                    # Fix 3: read proposal_diagnostics for LLM source / retry visibility
                    _pd = acc.get("proposal_diagnostics") or {}
                    if _pd:
                        print(
                            f"proposal_source={_pd.get('proposal_source')}  "
                            f"transient_retry_count={_pd.get('transient_retry_count', 0)}  "
                            f"used_retry={_pd.get('used_retry', False)}  "
                            f"fallback_reason={_pd.get('fallback_reason')}"
                        )
                    _rd = acc.get("ranker_diagnostics") or {}
                    if _rd:
                        _ora = _rd.get('override_retry_attempts', 0)
                        _orr = _rd.get('override_retry_resolved', False)
                        _ofa = _rd.get('override_fallback_applied', False)
                        _rufp = _rd.get('retry_used_full_proposal', False)
                        _retry_menu_label = (
                            "n/a" if _ora == 0
                            else ("full_proposal" if _rufp else "filtered")
                        )
                        print(
                            f"override_retry_attempts={red_if(_ora, _ora > 0)}  "
                            f"override_retry_resolved={red_if(_orr, not _orr and _ora > 0)}  "
                            f"override_fallback_applied={red_if(_ofa, _ofa)}  "
                            f"override_branch_name={_rd.get('override_branch_name')}  "
                            f"retry_menu={red_if(_retry_menu_label, _rufp)}"
                        )
                    print(RULE)
                    print()

                elif node_name == "logger_node":
                    saw_logger = True

    except Exception as e:
        print(f"[run_full_trace] RED stream error: {e}", file=sys.stderr)

    if not saw_logger and not quiet:
        print(
            "[run_full_trace] warning: graph did not complete logger_node "
            "(retry budget, error, or early end).",
            file=sys.stderr,
        )

    return acc


def _black_random_ply(
    acc: dict[str, Any], quiet: bool, rng: random.Random
) -> dict[str, Any]:
    turn_no = acc.get("turn_number", 0)
    display_turn = turn_no + 1
    board = acc["board"]
    legal = get_all_legal_moves(board, BLACK)
    if not legal:
        print("[run_full_trace] BLACK has no legal moves.", file=sys.stderr)
        return acc

    # ── Human picks BLACK's move ──────────────────────────────────────
    print(BAR)
    print(f"TURN {display_turn} | BLACK to move  (YOU)")
    print(BAR)
    print("── BOARD BEFORE MOVE ──")
    print_board(board)
    print()
    print(f"── YOUR AVAILABLE MOVES ({len(legal)} available) ──")
    for i, m in enumerate(legal):
        print(_fmt_engine_move(i, m))

    while True:
        try:
            raw = input(f"\nEnter move index [0-{len(legal) - 1}]: ").strip()
            k = int(raw)
            if 0 <= k < len(legal):
                break
            print(f"  Invalid — enter a number between 0 and {len(legal) - 1}.")
        except (ValueError, EOFError):
            print("  Invalid input, please enter a number.")

    move = legal[k]
    # ─────────────────────────────────────────────────────────────────

    if not quiet:
        path = move.get("path") or []
        if len(path) >= 2:
            a, b = path[0], path[-1]
            print(
                f"\nApplied: {move.get('type')} from [{a[0]},{a[1]}] to [{b[0]},{b[1]}]"
            )
        else:
            print(f"\nApplied: {move}")
        print(RULE)
        print()

    st = CheckersState.model_validate(acc)
    st.chosen_move = move
    st.last_move_reasoning = "BLACK human move"
    patch = state_manager(st)
    acc.update(patch)

    acc, ok = _merge_stream_updates(acc)
    if not ok:
        print(
            "[run_full_trace] warning: win/logger stream did not reach logger_node.",
            file=sys.stderr,
        )
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Full checkers trace: RED=LLM graph, BLACK=human.")
    parser.add_argument("--max-turns", type=int, default=200, help="Safety cap on plies (half-moves).")
    parser.add_argument("--seed", type=int, default=None, help="(unused, kept for CLI compatibility)")
    parser.add_argument("--quiet", action="store_true", help="Only final summary / incomplete message.")
    args = parser.parse_args()
    rng = random.Random(args.seed)  # kept in case needed elsewhere

    if not args.quiet:
        print(BAR)
        print(
            f"TRACE CONFIG | MINIMAX_ENABLED={os.environ.get('MINIMAX_ENABLED', 'unset')} "
            f"| MINIMAX_DEPTH={os.environ.get('MINIMAX_DEPTH', 'unset')} "
            f"| DEBUG_ALL_LEGAL_TO_RANKER={os.environ.get('DEBUG_ALL_LEGAL_TO_RANKER', 'false')}"
        )
        print(BAR)
        
    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()

    while True:
        if acc.get("game_over"):
            _print_final_summary(acc, quiet=args.quiet)
            return

        if (acc.get("turn_number") or 0) >= args.max_turns:
            print("GAME INCOMPLETE: max turns reached", file=sys.stderr)
            if not args.quiet:
                print(BAR)
                print("GAME INCOMPLETE: max turns reached")
                print(BAR)
                print("\nFinal board:")
                print_board(acc["board"])
                print("\n── METRICS ──")
                print(f"format_error_count: {acc.get('format_error_count', 0)}")
                print(f"ranker_failure_count: {acc.get('ranker_failure_count', 0)}")
                print(f"ranker_fallback_count: {acc.get('ranker_fallback_count', 0)}")
                t = _compute_game_totals(acc)
                print(f"ranker_fallback_rate: {t['ranker_fallback_rate_pct']:.1f}%")
                gid = acc.get("game_log_id") or "?"
                print(f"\nLogs (if any): logs/{gid}.jsonl")
            return

        if acc["current_player"] == RED:
            acc = _run_red_ply(acc, args.quiet)
        else:
            acc = _black_random_ply(acc, args.quiet, rng)


if __name__ == "__main__":
    main()