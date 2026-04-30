#!/usr/bin/env python3
"""
Batch ranker evaluation.

Two minimax comparison bases (same board, same ply, before ranker applies the move):

* **Filtered** — ``legal_moves`` passed to the ranker (proposal shortlist or all-legal list after validator).
* **Full legal** — all engine-legal moves rescored with validator + minimax (coverage vs global best).

Primary mismatch categories:

* ``near_tie`` — |filtered_best − chosen| < MINIMAX_NEAR_TIE_EPSILON (default 0.5).
* ``justified_deviation`` — override or symbolic rejection (incl. concrete defensive better).
* ``proposal_or_filtering_gap`` — worse than global best but **not** worse than best among the ranker menu.
* ``true_ranker_error`` — worse than best among moves the ranker actually saw; logged in ``suspicious_mismatch_details``.

Recommended iterative workflow (fast -> thorough):

1) Quick smoke test
   venv/bin/python3 evaluate_ranker_batch.py --games 1 --max-plies 40 --seed 42 --out logs/quick_eval.json

2) Medium check
   venv/bin/python3 evaluate_ranker_batch.py --games 2 --max-plies 60 --seed 42 --out logs/medium_eval.json

3) Full verification
   venv/bin/python3 evaluate_ranker_batch.py --games 5 --max-plies 120 --seed 42 --out logs/final_eval.json
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import random
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from typing import Any, Optional
from dotenv import load_dotenv  # type: ignore

load_dotenv()

# Strict minimax near-tie band: |best − chosen| <= this → not a true error (overrides other labels).
MINIMAX_NEAR_TIE_EPSILON = float(os.environ.get("MINIMAX_NEAR_TIE_EPSILON", "2.0"))

from checkers.engine.board import BLACK, RED, create_initial_board
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves
from checkers.agents.proposal_agent import proposal_agent
from checkers.nodes.format_checker import format_checker
from checkers.nodes.inter_turn_memory import inter_turn_memory
from checkers.nodes.minimax_scorer import minimax_scorer
from checkers.nodes.validator import validator
from checkers.nodes.state_manager import state_manager
from checkers.nodes.win_condition import win_condition
from checkers.state.state import CheckersState
from checkers.agents.ranker_agent import ranker_agent

include_exact_ties = (
    os.environ.get("SELECTIVE_D8_INCLUDE_EXACT_TIES", "false").lower()
    in ("1", "true", "yes", "on")
)
@dataclass
class RedDecisionStats:
    override_triggered: Optional[bool]
    override_branch_name: Optional[str]
    override_block_reason: Optional[str]
    best_move_rejected_reason: Optional[str]
    chosen_not_best_minimax: bool
    full_legal_not_best: bool
    filtered_not_best: bool
    mismatch_classification: Optional[str]


@dataclass
class GameEval:
    game_index: int
    winner: str
    total_plies: int
    override_triggers: int
    chosen_not_best_minimax_count: int
    full_legal_not_best_count: int
    filtered_not_best_count: int
    true_error_count: int
    near_tie_count: int
    justified_count: int
    proposal_or_filtering_gap_count: int
    first_material_deficit_turn: Optional[int]
    red_pipeline_failures: int


def _winner_label(state: dict[str, Any]) -> str:
    if state.get("draw"):
        return "DRAW"
    if state.get("winner") == RED:
        return "RED"
    if state.get("winner") == BLACK:
        return "BLACK"
    return "UNKNOWN"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("-inf")


def _parse_decision_debug_token(raw: str) -> Any:
    if raw in ("None", "null"):
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    # Best-effort numeric parsing for new debug tokens.
    try:
        if re.fullmatch(r"[+-]?\d+", raw):
            return int(raw)
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", raw):
            return float(raw)
    except re.error:
        pass
    return raw


def _parse_decision_debug(stdout_text: str) -> dict[str, Any]:
    full = _parse_decision_debug_full(stdout_text)
    if not full:
        return {}
    return {
        "override_triggered": full.get("override_triggered"),
        "override_branch_name": full.get("override_branch_name"),
        "override_block_reason": full.get("override_block_reason"),
        "best_move_rejected_reason": full.get("best_move_rejected_reason"),
    }


def _parse_decision_debug_full(stdout_text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.startswith("[DECISION_DEBUG]")]
    if not lines:
        return {}

    line1 = next((ln for ln in lines if "gap=" in ln and "chosen=" in ln), "")
    line2 = next((ln for ln in lines if "chosen_move_facts=" in ln), "")
    out: dict[str, Any] = {}

    if line1:
        m = re.search(r"chosen=(.+?)\s+best=", line1)
        if m:
            cs = m.group(1).strip()
            try:
                out["chosen_path"] = ast.literal_eval(cs) if cs != "None" else None
            except (ValueError, SyntaxError):
                out["chosen_path"] = None
        m = re.search(r"best=(.+?)\s+gap=", line1)
        if m:
            bs = m.group(1).strip()
            try:
                out["best_path"] = ast.literal_eval(bs) if bs != "None" else None
            except (ValueError, SyntaxError):
                out["best_path"] = None
        m = re.search(r"gap=([0-9.+-]+|None)", line1)
        if m and m.group(1) != "None":
            try:
                out["gap"] = float(m.group(1))
            except ValueError:
                out["gap"] = None
        else:
            out["gap"] = None

        for key in (
            "llm_idx",
            "best_idx",
            "filtered_menu_size",
            "chosen_minimax_internal",
            "best_minimax_internal",
            "chosen_path_internal",
            "best_path_internal",
            "chosen_path_matches_llm_idx",
            "best_idx_is_argmax",
            "best_score_tie_count",
            "chosen_passive_safe",
            "best_passive_safe",
            "chosen_low_danger",
            "best_low_danger",
            "override_triggered",
            "override_branch_name",
            "override_block_reason",
            "best_move_rejected_reason",
        ):
            m = re.search(rf"{key}=([^\s]+)", line1)
            out[key] = _parse_decision_debug_token(m.group(1)) if m else None

        m = re.search(r"threat_delta=([^\s]+)", line1)
        if m:
            raw = m.group(1)
            if raw in ("None", "null"):
                out["threat_delta"] = None
            else:
                try:
                    out["threat_delta"] = float(raw)
                except ValueError:
                    out["threat_delta"] = raw

    if line2:
        scores = re.findall(r"'minimax_score': ([-0-9.]+|None)", line2)

        def _to_score(x: str) -> Optional[float]:
            if x == "None":
                return None
            try:
                return float(x)
            except ValueError:
                return None

        if len(scores) >= 2:
            out["chosen_minimax_from_debug"] = _to_score(scores[0])
            out["best_minimax_from_debug"] = _to_score(scores[1])
        elif len(scores) == 1:
            out["chosen_minimax_from_debug"] = _to_score(scores[0])
            out["best_minimax_from_debug"] = None

    return out


def _slim_move(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": m.get("type"),
        "path": m.get("path"),
        "captured": m.get("captured", []),
    }


def _normalize_move_path(path: Any) -> Any:
    """
    Canonicalize move path for evaluator-only equality checks.
    Accepts tuple/list nesting and normalizes to list-of-lists ints when possible.
    """
    if isinstance(path, (list, tuple)):
        out: list[Any] = []
        for item in path:
            if isinstance(item, (list, tuple)):
                norm_item: list[Any] = []
                for v in item:
                    if isinstance(v, bool):
                        norm_item.append(v)
                    elif isinstance(v, (int, float)):
                        norm_item.append(int(v))
                    elif isinstance(v, str) and re.fullmatch(r"[+-]?\d+", v.strip()):
                        norm_item.append(int(v.strip()))
                    else:
                        norm_item.append(v)
                out.append(norm_item)
            elif isinstance(item, bool):
                out.append(item)
            elif isinstance(item, (int, float)):
                out.append(int(item))
            elif isinstance(item, str) and re.fullmatch(r"[+-]?\d+", item.strip()):
                out.append(int(item.strip()))
            else:
                out.append(item)
        return out
    return path


def _move_matches_menu_entry(menu_move: dict[str, Any], chosen_move: dict[str, Any]) -> bool:
    menu_path = _normalize_move_path(menu_move.get("path"))
    chosen_path = _normalize_move_path(chosen_move.get("path"))
    if menu_path != chosen_path:
        return False
    menu_type = menu_move.get("type")
    chosen_type = chosen_move.get("type")
    if menu_type is None or chosen_type is None:
        return True
    return menu_type == chosen_type


def _chosen_matches_menu(menu: list[dict[str, Any]], chosen_move: dict[str, Any]) -> bool:
    for m in menu:
        if _move_matches_menu_entry(m, chosen_move):
            return True
    return False


def _find_move_by_path(legal_moves: list[dict[str, Any]], path: Any) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    normalized_path = _normalize_move_path(path)
    for m in legal_moves:
        if _normalize_move_path(m.get("path")) == normalized_path:
            return m
    return None


def _best_minimax_legal(legal_moves: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], float]:
    best_m: Optional[dict[str, Any]] = None
    best_s = float("-inf")
    for m in legal_moves:
        s = _safe_float((m.get("facts") or {}).get("minimax_score"))
        if s > best_s:
            best_s = s
            best_m = m
    return best_m, best_s


def _best_minimax_on_menu(menu: list[dict[str, Any]]) -> float:
    _, s = _best_minimax_legal(menu)
    return s


def _merge_state_patch(state: CheckersState, patch: dict[str, Any]) -> CheckersState:
    return CheckersState(**{**state.model_dump(), **patch})


def _score_moves_with_validator_minimax(
    st: CheckersState,
    proposed_moves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Re-run validator + minimax_scorer on a fresh copy of ``st`` with the given proposed_moves.
    Used to build the full-engine-legal scored menu in parallel with the ranker's filtered menu.
    """
    base = CheckersState.model_validate(st.model_dump())
    base = _merge_state_patch(base, {"proposed_moves": list(proposed_moves)})
    base = _merge_state_patch(base, validator(base))
    legal = base.legal_moves or []
    if not legal:
        return []
    base = _merge_state_patch(base, minimax_scorer(base))
    return list(base.legal_moves or [])


def _chosen_minimax_on_menu(
    menu: list[dict[str, Any]],
    chosen_move: dict[str, Any],
) -> float:
    for m in menu:
        if _move_matches_menu_entry(m, chosen_move):
            return _safe_float((m.get("facts") or {}).get("minimax_score"))
    return float("-inf")


def _build_suspicious_mismatch_record(
    game_index: int,
    turn_number: int,
    decision_snapshot: dict[str, Any],
    capture_text: str,
    stats: RedDecisionStats,
) -> Optional[dict[str, Any]]:
    if stats.mismatch_classification != "true_ranker_error":
        return None

    ranker_menu = decision_snapshot.get("ranker_filtered_menu") or []
    legal = ranker_menu if ranker_menu else (decision_snapshot.get("legal_moves") or [])
    full_menu = decision_snapshot.get("full_legal_scored_moves") or []
    chosen_raw = decision_snapshot.get("chosen_move") or {}
    full = _parse_decision_debug_full(capture_text)

    chosen_move = _slim_move(chosen_raw) if chosen_raw else {}
    best_m, best_score = _best_minimax_legal(legal)
    best_move = _slim_move(best_m) if best_m else {}

    bp = full.get("best_path")
    cp = full.get("chosen_path")
    if bp is not None:
        alt = _find_move_by_path(legal, bp)
        if alt is not None:
            best_move = _slim_move(alt)
    if cp is not None:
        altc = _find_move_by_path(legal, cp)
        if altc is not None:
            chosen_move = _slim_move(altc)

    chosen_score = _safe_float((chosen_raw.get("facts") or {}).get("minimax_score"))
    if full.get("chosen_minimax_from_debug") is not None:
        chosen_minimax = full["chosen_minimax_from_debug"]
    else:
        chosen_minimax = chosen_score if chosen_score > float("-inf") else None

    if full.get("best_minimax_from_debug") is not None:
        best_minimax = full["best_minimax_from_debug"]
    else:
        best_minimax = best_score if best_score > float("-inf") else None

    gap = full.get("gap")
    if gap is None and isinstance(best_minimax, (int, float)) and isinstance(chosen_minimax, (int, float)):
        gap = float(best_minimax) - float(chosen_minimax)

    g_best_m, g_best_s = _best_minimax_legal(full_menu)
    g_chosen = _chosen_minimax_on_menu(full_menu, chosen_raw)
    f_chosen = _chosen_minimax_on_menu(legal, chosen_raw)
    f_best_s = _best_minimax_on_menu(legal)
    filtered_menu_gap = float(f_best_s) - float(f_chosen) if f_chosen > float("-inf") else None
    full_legal_gap = float(g_best_s) - float(g_chosen) if g_chosen > float("-inf") else None

    chosen_path_norm = _normalize_move_path(chosen_move.get("path"))
    best_path_norm = _normalize_move_path(best_move.get("path"))

    diagnostic_inconsistency = False
    if chosen_path_norm == best_path_norm:
        diagnostic_inconsistency = True
    elif gap == 0.0 and ((filtered_menu_gap is not None and filtered_menu_gap > 0) or (full_legal_gap is not None and full_legal_gap > 0)):
        diagnostic_inconsistency = True
    elif chosen_minimax is not None and f_chosen > float("-inf") and not math.isclose(chosen_minimax, f_chosen, abs_tol=1e-5):
        diagnostic_inconsistency = True

    rec = {
        "classification": "diagnostic_inconsistency" if diagnostic_inconsistency else "true_ranker_error",
        "diagnostic_inconsistency": diagnostic_inconsistency,
        "game_index": game_index,
        "turn_number": turn_number,
        # Override-time instrumentation captured from DECISION_DEBUG (ranker stdout).
        "llm_idx": full.get("llm_idx"),
        "best_idx": full.get("best_idx"),
        "filtered_menu_size": full.get("filtered_menu_size"),
        "chosen_minimax_internal": full.get("chosen_minimax_internal"),
        "best_minimax_internal": full.get("best_minimax_internal"),
        "chosen_path_internal": full.get("chosen_path_internal"),
        "best_path_internal": full.get("best_path_internal"),
        "chosen_path_matches_llm_idx": full.get("chosen_path_matches_llm_idx"),
        "best_idx_is_argmax": full.get("best_idx_is_argmax"),
        "best_score_tie_count": full.get("best_score_tie_count"),
        "chosen_move": chosen_move,
        "best_move": best_move,
        "chosen_minimax": chosen_minimax,
        "best_minimax": best_minimax,
        "minimax_gap": gap,
        "full_legal_not_best": stats.full_legal_not_best,
        "filtered_not_best": stats.filtered_not_best,
        "filtered_best_minimax": f_best_s if f_best_s > float("-inf") else None,
        "full_legal_best_minimax": g_best_s if g_best_s > float("-inf") else None,
        "full_legal_chosen_minimax": g_chosen if g_chosen > float("-inf") else None,
        "filtered_menu_gap": filtered_menu_gap,
        "full_legal_minimax_gap": full_legal_gap,
        "full_legal_best_move": _slim_move(g_best_m) if g_best_m else {},
        "override_triggered": full.get("override_triggered", stats.override_triggered),
        "override_branch_name": full.get("override_branch_name", stats.override_branch_name),
        "override_block_reason": full.get("override_block_reason", stats.override_block_reason),
        "best_move_rejected_reason": full.get("best_move_rejected_reason", stats.best_move_rejected_reason),
        "chosen_passive_safe": full.get("chosen_passive_safe"),
        "best_passive_safe": full.get("best_passive_safe"),
        "chosen_low_danger": full.get("chosen_low_danger"),
        "best_low_danger": full.get("best_low_danger"),
    }
    return rec


_JUSTIFIED_DEVIATION_REASONS = frozenset(
    {
        "protect_endgame_activity",
        "protect_king_activation",
        "protect_losing_mode_counterplay",
        "opening_safe_lane_preserved",
        "chosen_better_immediate_danger",
        "best_uniquely_worse_structure",
        "chosen_bad_danger",
    }
)


def _justified_from_debug(debug: dict[str, Any]) -> bool:
    br = debug.get("best_move_rejected_reason")
    obr = debug.get("override_block_reason")
    if br == "chosen_concrete_defensive_better" or obr == "chosen_concrete_defensive_better":
        return True
    if debug.get("override_triggered") is True:
        return True
    if br in _JUSTIFIED_DEVIATION_REASONS:
        return True
    return False


def _classify_ranker_mismatch(
    *,
    full_legal_not_best: bool,
    filtered_not_best: bool,
    f_chosen: float,
    f_best: float,
    g_chosen: float,
    g_best: float,
    debug: dict[str, Any],
) -> Optional[str]:
    """
    Single primary label per RED ply (mutually exclusive buckets for batch totals).

    * ``near_tie`` — tiny gap on the comparison set that matters for that bucket
      (filtered menu if ``filtered_not_best``, else full-legal menu for proposal-only gaps).
    * ``justified_deviation`` — symbolic / override reasons (incl. concrete defensive better).
    * ``proposal_or_filtering_gap`` — globally suboptimal but best among the filtered candidate set.
    * ``true_ranker_error`` — strictly worse than the best minimax move the ranker actually saw.
    """
    if not full_legal_not_best and not filtered_not_best:
        return None

    if filtered_not_best:
        diff_f = abs(f_best - f_chosen)
        if diff_f <= MINIMAX_NEAR_TIE_EPSILON:
            return "near_tie"
        if _justified_from_debug(debug):
            return "justified_deviation"
        return "true_ranker_error"

    # Globally suboptimal but not worse than the best move on the filtered menu.
    if not full_legal_not_best:
        return None
    diff_g = abs(g_best - g_chosen)
    if diff_g <= MINIMAX_NEAR_TIE_EPSILON:
        return "near_tie"
    return "proposal_or_filtering_gap"


def _extract_red_decision(snapshot: dict[str, Any], debug: dict[str, Any]) -> RedDecisionStats:
    ranker_menu = snapshot.get("ranker_filtered_menu") or []
    legal_moves = ranker_menu if ranker_menu else (snapshot.get("legal_moves") or [])
    full_menu = snapshot.get("full_legal_scored_moves") or []
    chosen_move = snapshot.get("chosen_move") or {}

    f_chosen = _chosen_minimax_on_menu(legal_moves, chosen_move)
    f_best_m, f_best = _best_minimax_legal(legal_moves)
    g_chosen = _chosen_minimax_on_menu(full_menu, chosen_move)
    g_best_m, g_best = _best_minimax_legal(full_menu)

    chosen_p = _normalize_move_path(chosen_move.get("path"))
    f_best_p = _normalize_move_path(f_best_m.get("path")) if f_best_m else None
    g_best_p = _normalize_move_path(g_best_m.get("path")) if g_best_m else None

    # Fix 1: If chosen path matches best path, it's not a mismatch.
    if chosen_p == f_best_p:
        filtered_not_best = False
    else:
        filtered_not_best = f_chosen < f_best

    if g_chosen > float("-inf"):
        if chosen_p == g_best_p:
            full_legal_not_best = False
        else:
            full_legal_not_best = g_chosen < g_best
    else:
        # Chosen move missing from the rescored full-legal menu (validator drop) — treat as global gap.
        full_legal_not_best = bool(full_menu) and g_best > float("-inf")

    chosen_not_best_minimax = filtered_not_best

    mismatch_classification = _classify_ranker_mismatch(
        full_legal_not_best=full_legal_not_best,
        filtered_not_best=filtered_not_best,
        f_chosen=f_chosen,
        f_best=f_best,
        g_chosen=g_chosen,
        g_best=g_best,
        debug=debug,
    )

    if filtered_not_best:
        diff_abs = abs(f_best - f_chosen)
        if math.isclose(diff_abs, 0.0, abs_tol=1e-12) and mismatch_classification != "near_tie":
            print(
                "[evaluate_ranker_batch] INVALID_CLASSIFICATION: "
                f"filtered chosen<best but |minimax_gap|≈0 (chosen={f_chosen}, best={f_best}, "
                f"class={mismatch_classification})",
                flush=True,
            )
        if mismatch_classification == "true_ranker_error" and diff_abs < MINIMAX_NEAR_TIE_EPSILON:
            print(
                "[evaluate_ranker_batch] INVALID_CLASSIFICATION: "
                f"true_ranker_error with |filtered_best−chosen| < epsilon "
                f"(chosen={f_chosen}, best={f_best})",
                flush=True,
            )

    return RedDecisionStats(
        override_triggered=debug.get("override_triggered"),
        override_branch_name=debug.get("override_branch_name"),
        override_block_reason=debug.get("override_block_reason"),
        best_move_rejected_reason=debug.get("best_move_rejected_reason"),
        chosen_not_best_minimax=chosen_not_best_minimax,
        full_legal_not_best=full_legal_not_best,
        filtered_not_best=filtered_not_best,
        mismatch_classification=mismatch_classification,
    )


def _run_red_ply(
    state_dict: dict[str, Any],
    *,
    pipeline: str = "all_legal",
) -> tuple[dict[str, Any], Optional[RedDecisionStats], bool, Optional[str], str, dict[str, Any]]:
    """
    Runs one RED ply.

    pipeline ``all_legal`` (default): inter_turn_memory -> validator(engine legal)
    -> minimax_scorer -> ranker_agent -> state_manager -> win_condition

    pipeline ``proposal``: inter_turn_memory -> proposal_agent -> format_checker
    -> validator -> minimax_scorer -> ranker_agent -> state_manager -> win_condition
    (matches the main graph ordering; ranker sees only the proposal shortlist.)

    Returns (updated_state, decision_stats_or_none, ok, reason_or_none, ranker_stdout,
              decision_snapshot_for_eval).
    """
    empty_snap: dict[str, Any] = {}
    st = CheckersState.model_validate(state_dict)
    engine_legal = get_all_legal_moves(st.board, RED)
    if not engine_legal:
        return {
            **state_dict,
            "game_over": True,
            "winner": BLACK,
            "draw": False,
        }, None, True, None, "", empty_snap

    def _merge_state(state: CheckersState, patch: dict[str, Any]) -> CheckersState:
        return CheckersState(**{**state.model_dump(), **patch})

    st = _merge_state(st, inter_turn_memory(st))

    if pipeline == "proposal":
        while True:
            if st.retry_count >= st.retry_budget:
                return (
                    st.model_dump(),
                    None,
                    False,
                    "proposal pipeline: retry_budget exhausted before validator accepted proposals",
                    "",
                    empty_snap,
                )
            st = _merge_state(st, proposal_agent(st))
            st = _merge_state(st, format_checker(st))
            if isinstance(st.proposed_moves, list) and len(st.proposed_moves) == 0:
                if st.retry_count >= st.retry_budget:
                    return (
                        st.model_dump(),
                        None,
                        False,
                        "proposal pipeline: format_checker returned no proposals after retries",
                        "",
                        empty_snap,
                    )
                continue
            st = _merge_state(st, validator(st))
            if st.legal_moves:
                break
            if st.retry_count >= st.retry_budget:
                return (
                    st.model_dump(),
                    None,
                    False,
                    "proposal pipeline: validator had no legal moves after retries",
                    "",
                    empty_snap,
                )
            continue
    else:
        # Bypass proposal: pass all engine-legal moves directly to validator.
        st = _merge_state(st, {"proposed_moves": engine_legal})
        st = _merge_state(st, validator(st))

    if not st.legal_moves:
        return (
            st.model_dump(),
            None,
            False,
            "validator produced no legal_moves",
            "",
            empty_snap,
        )

    full_legal_scored_moves = _score_moves_with_validator_minimax(st, engine_legal)
    st = _merge_state(st, minimax_scorer(st))

    decision_snapshot: dict[str, Any] = {
        "legal_moves": list(st.legal_moves or []),
        "full_legal_scored_moves": full_legal_scored_moves,
        "chosen_move": None,
        "ranker_filtered_menu": None,
    }
    capture = io.StringIO()
    try:
        with redirect_stdout(capture):
            st = _merge_state(st, ranker_agent(st))
    except Exception as e:
        return st.model_dump(), None, False, f"ranker_agent failed: {e}", "", empty_snap
    if st.chosen_move is None:
        return st.model_dump(), None, False, "ranker_agent returned no chosen_move", "", empty_snap

    decision_snapshot["chosen_move"] = dict(st.chosen_move)
    decision_snapshot["ranker_filtered_menu"] = (
        list(st.ranker_filtered_menu or []) if getattr(st, "ranker_filtered_menu", None) else None
    )

    capture_text = capture.getvalue()
    debug = _parse_decision_debug(capture_text)
    stats = _extract_red_decision(decision_snapshot, debug)
    st = _merge_state(st, state_manager(st))
    st = _merge_state(st, win_condition(st))
    return st.model_dump(), stats, True, None, capture_text, decision_snapshot


def _run_black_random_ply(state_dict: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], bool]:
    acc = dict(state_dict)
    legal = get_all_legal_moves(acc["board"], BLACK)
    if not legal:
        acc.update({"game_over": True, "winner": RED, "draw": False})
        return acc, True
    move = legal[rng.randrange(len(legal))]
    st = CheckersState.model_validate(acc)
    st.chosen_move = move
    st.last_move_reasoning = "BLACK random move (evaluation harness)"
    acc.update(state_manager(st))
    st2 = CheckersState.model_validate(acc)
    acc.update(win_condition(st2))
    return acc, True


def _first_red_material_deficit_turn(move_history: list[dict[str, Any]], initial_board: list[list[int]]) -> Optional[int]:
    board = [row[:] for row in initial_board]
    for rec in move_history:
        move = rec.get("move") or {}
        path = move.get("path") or []
        if not path:
            continue
        from checkers.engine.rules import apply_move  # local import to keep top clean
        board = apply_move(board, move)
        counts_red = count_pieces(board, RED)["total"]
        counts_black = count_pieces(board, BLACK)["total"]
        if counts_red < counts_black:
            return rec.get("turn")
    return None


def run_batch(
    games: int,
    max_plies: int,
    seed: Optional[int],
    *,
    pipeline: str = "all_legal",
) -> dict[str, Any]:
    rng = random.Random(seed)
    all_results: list[GameEval] = []
    red_failure_reasons: list[dict[str, Any]] = []
    suspicious_mismatch_details: list[dict[str, Any]] = []
    red_decisions_with_ranker_filtered_menu = 0
    red_decisions_missing_ranker_filtered_menu = 0
    chosen_move_unmatched_on_ranker_filtered_menu = 0
    true_error_rows_filtered_menu_gap_null = 0

    single_safe_positions_count = 0
    single_safe_unsafe_survivor_3_to_14_count = 0
    single_safe_unsafe_chosen_count = 0
    single_safe_examples: list[dict[str, Any]] = []

    for game_idx in range(1, games + 1):
        state = CheckersState(
            board=create_initial_board(),
            current_player=RED,
            turn_number=0,
        ).model_dump()
        override_triggers = 0
        mismatches = 0
        full_fl = 0
        filtered_fl = 0
        true_err = 0
        near_tie = 0
        justified = 0
        proposal_gap = 0
        red_failures = 0
        initial_board = [row[:] for row in state["board"]]

        while not state.get("game_over") and (state.get("turn_number", 0) < max_plies):
            if state["current_player"] == RED:
                legal = get_all_legal_moves(state["board"], RED)
                if not legal:
                    state.update({"game_over": True, "winner": BLACK, "draw": False})
                    break
                turn_before = state.get("turn_number", 0)
                state, decision, ok, reason, capture_text, decision_snapshot = _run_red_ply(
                    state,
                    pipeline=pipeline,
                )
                if not ok:
                    red_failures += 1
                    red_failure_reasons.append(
                        {
                            "game_index": game_idx,
                            "turn_number": state.get("turn_number", 0),
                            "reason": reason or "unknown RED failure",
                        }
                    )
                    state.update({"game_over": True, "winner": BLACK, "draw": False})
                    break
                if decision is not None:
                    ranker_menu = decision_snapshot.get("ranker_filtered_menu") or []
                    chosen_move = decision_snapshot.get("chosen_move") or {}
                    if ranker_menu:
                        red_decisions_with_ranker_filtered_menu += 1
                        if not _chosen_matches_menu(ranker_menu, chosen_move):
                            chosen_move_unmatched_on_ranker_filtered_menu += 1
                    else:
                        red_decisions_missing_ranker_filtered_menu += 1
                    if decision.override_triggered is True:
                        override_triggers += 1
                    if decision.filtered_not_best:
                        mismatches += 1
                    if decision.full_legal_not_best:
                        full_fl += 1
                    if decision.filtered_not_best:
                        filtered_fl += 1
                    cat = decision.mismatch_classification
                    if cat == "near_tie":
                        near_tie += 1
                    elif cat == "justified_deviation":
                        justified += 1
                    elif cat == "proposal_or_filtering_gap":
                        proposal_gap += 1
                    elif cat == "true_ranker_error":
                        rec = _build_suspicious_mismatch_record(
                            game_idx,
                            turn_before,
                            decision_snapshot,
                            capture_text,
                            decision,
                        )
                        if rec is not None:
                            if not rec.get("diagnostic_inconsistency"):
                                true_err += 1
                                if rec.get("filtered_menu_gap") is None:
                                    true_error_rows_filtered_menu_gap_null += 1
                            suspicious_mismatch_details.append(rec)
                            
                    # Safety audit
                    menu = ranker_menu if ranker_menu else (decision_snapshot.get("legal_moves") or [])
                    safe_moves = [m for m in menu if not m.get("facts", {}).get("opponent_can_recapture", False)]
                    if len(safe_moves) == 1:
                        single_safe_positions_count += 1
                        best_safe_score = _safe_float(safe_moves[0].get("facts", {}).get("minimax_score"))
                        
                        unsafe_survivors = False
                        for m in menu:
                            if m.get("facts", {}).get("opponent_can_recapture", False):
                                gap = _safe_float(m.get("facts", {}).get("minimax_score")) - best_safe_score
                                if 3.0 <= gap <= 14.0:
                                    unsafe_survivors = True
                                    break
                                    
                        if unsafe_survivors:
                            single_safe_unsafe_survivor_3_to_14_count += 1
                            
                        if chosen_move.get("facts", {}).get("opponent_can_recapture", False):
                            single_safe_unsafe_chosen_count += 1
                            if len(single_safe_examples) < 5:
                                chosen_score = _safe_float(chosen_move.get("facts", {}).get("minimax_score"))
                                single_safe_examples.append({
                                    "game_index": game_idx,
                                    "turn_number": turn_before,
                                    "chosen_move": _slim_move(chosen_move),
                                    "chosen_minimax": chosen_score,
                                    "best_safe_minimax": best_safe_score,
                                    "chosen_opponent_can_recapture": True,
                                    "gap_over_best_safe": chosen_score - best_safe_score,
                                    "override_triggered": decision.override_triggered,
                                    "classification": decision.mismatch_classification,
                                })
            else:
                state, _ = _run_black_random_ply(state, rng)

        first_deficit = _first_red_material_deficit_turn(state.get("move_history") or [], initial_board)
        all_results.append(
            GameEval(
                game_index=game_idx,
                winner=_winner_label(state),
                total_plies=state.get("turn_number", 0),
                override_triggers=override_triggers,
                chosen_not_best_minimax_count=mismatches,
                full_legal_not_best_count=full_fl,
                filtered_not_best_count=filtered_fl,
                true_error_count=true_err,
                near_tie_count=near_tie,
                justified_count=justified,
                proposal_or_filtering_gap_count=proposal_gap,
                first_material_deficit_turn=first_deficit,
                red_pipeline_failures=red_failures,
            )
        )

    summary = {
        "pipeline": pipeline,
        "games": games,
        "max_plies": max_plies,
        "seed": seed,
        "wins_red": sum(1 for r in all_results if r.winner == "RED"),
        "wins_black": sum(1 for r in all_results if r.winner == "BLACK"),
        "draws": sum(1 for r in all_results if r.winner == "DRAW"),
        "avg_override_triggers_per_game": round(sum(r.override_triggers for r in all_results) / max(1, len(all_results)), 2),
        "avg_mismatch_per_game": round(sum(r.chosen_not_best_minimax_count for r in all_results) / max(1, len(all_results)), 2),
        "full_legal_not_best_count": sum(r.full_legal_not_best_count for r in all_results),
        "filtered_not_best_count": sum(r.filtered_not_best_count for r in all_results),
        "true_error_count": sum(r.true_error_count for r in all_results),
        "true_ranker_error_count": sum(r.true_error_count for r in all_results),
        "near_tie_count": sum(r.near_tie_count for r in all_results),
        "justified_count": sum(r.justified_count for r in all_results),
        "proposal_or_filtering_gap_count": sum(r.proposal_or_filtering_gap_count for r in all_results),
        "minimax_near_tie_epsilon": MINIMAX_NEAR_TIE_EPSILON,
        "games_with_material_deficit": sum(1 for r in all_results if r.first_material_deficit_turn is not None),
        "total_red_pipeline_failures": sum(r.red_pipeline_failures for r in all_results),
        "red_failure_reasons": red_failure_reasons,
        "suspicious_mismatch_details": suspicious_mismatch_details,
        "debug_ranker_menu_presence": {
            "red_decisions_with_ranker_filtered_menu": red_decisions_with_ranker_filtered_menu,
            "red_decisions_missing_ranker_filtered_menu": red_decisions_missing_ranker_filtered_menu,
            "chosen_move_unmatched_on_ranker_filtered_menu": chosen_move_unmatched_on_ranker_filtered_menu,
            "true_error_rows_filtered_menu_gap_null": true_error_rows_filtered_menu_gap_null,
        },
        "single_safe_audit": {
            "single_safe_positions_count": single_safe_positions_count,
            "single_safe_unsafe_survivor_3_to_14_count": single_safe_unsafe_survivor_3_to_14_count,
            "single_safe_unsafe_chosen_count": single_safe_unsafe_chosen_count,
            "examples": single_safe_examples,
        },
        "per_game": [asdict(r) for r in all_results],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation harness for ranker policy quality.",
        epilog=(
            "Recommended run tiers:\n"
            "  Quick smoke:  venv/bin/python3 evaluate_ranker_batch.py --games 1 --max-plies 40 --seed 42 --out logs/quick_eval.json\n"
            "  Medium check: venv/bin/python3 evaluate_ranker_batch.py --games 2 --max-plies 60 --seed 42 --out logs/medium_eval.json\n"
            "  Full verify:  venv/bin/python3 evaluate_ranker_batch.py --games 5 --max-plies 120 --seed 42 --out logs/final_eval.json\n"
            "Use quick/medium during iteration; reserve full runs for confirmation."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=5, help="Number of full games to run.")
    parser.add_argument("--max-plies", type=int, default=120, help="Safety cap per game.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for BLACK random policy.")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON file path.")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=("all_legal", "proposal"),
        default="all_legal",
        help=(
            "all_legal: pass every engine-legal move to validator (no proposal shortlist). "
            "proposal: real proposal_agent + format_checker shortlist, then minimax + ranker."
        ),
    )
    args = parser.parse_args()

    summary = run_batch(
        games=max(1, args.games),
        max_plies=max(10, args.max_plies),
        seed=args.seed,
        pipeline=args.pipeline,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()