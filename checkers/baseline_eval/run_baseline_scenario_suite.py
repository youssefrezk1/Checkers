#!/usr/bin/env python3
"""
checkers/baseline_eval/run_baseline_scenario_suite.py

Thesis baseline scenario runner — single-ply, NO autocorrect, NO game state.

Active thesis arms:
  B1: minimal_raw_llm  board + piece legend only
  B2: rules_only_llm   board + full rules
  B4: full_system      full neuro-symbolic pipeline

Usage:
    venv/bin/python3 -m checkers.baseline_eval.run_baseline_scenario_suite
    venv/bin/python3 -m checkers.baseline_eval.run_baseline_scenario_suite \\
        --baselines minimal_raw_llm rules_only_llm full_system \\
        --scenarios opening mandatory_capture tactical_trap \\
        --repeats 3 --out-dir logs/scenario_eval

Scenarios:
    opening           Standard opening (7 RED simples, no captures).
    mandatory_capture Two RED pieces each with a forced jump.
    multi_jump        One RED piece can chain-capture two BLACK pieces.
    promotion         RED man can immediately promote to king.
    near_promotion    RED man is one move from promotion row.
    king_move         RED king with four-directional mobility, no captures.
    endgame           Two-piece endgame (RED king vs BLACK man).
    tactical_trap     Two mandatory jumps; one leads to recapture loss.
    many_legal_moves  Open mid-game position with 11 legal moves.
    one_legal_move    Single forced move (RED man in a corner).

Error classes (primary, first applicable wins):
    llm_call_failed        API call raised an exception.
    output_format_error    JSON invalid or required field missing.
    invalid_index          chosen_index out of range (index baselines only).
    hallucinated_path      Path not found in engine legal-move list.
    reasoning_hallucination  Reasoning text contradicts the chosen move's facts.
    strategic_error        Legal move chosen but minimax rank > 3.
    clean                  No errors detected.

Hallucinated-path subtypes (field: hallucinated_path_subtype):
    wrong_piece_square         From-square has no RED piece.
    invalid_destination        To-square is off-board, light, or occupied.
    wrong_path_format          Path is not a list of 2+ valid [int,int] pairs.
    mandatory_capture_violation  Simple-move path when jump was mandatory.
    path_not_in_legal_moves    Valid format but no engine-legal-move match.
"""

from __future__ import annotations

import os
os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import argparse
import csv
import json
import statistics
import sys
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from checkers.engine.board import (
    RED, BLACK, EMPTY, RED_KING,
    create_initial_board, print_board,
)
from checkers.engine.rules import get_all_legal_moves
from checkers.agents.scorer_agent import score_all_legal_moves

from checkers.baseline_eval.run_baseline_human_trace import (
    BASELINE_MINIMAL_RAW_LLM,
    BASELINE_RULES_ONLY_LLM,
    BASELINE_FULL_SYSTEM,
    call_baseline_llm,
    _MINIMAL_RAW_LLM_SYSTEM,
    _RULES_ONLY_LLM_SYSTEM,
    _build_path_json_user,
    _parse_json_response,
    _find_move_by_path,
    _has_mandatory_capture,
    _rank_and_gap,
    _slim,
    _print_prompts,
)
from checkers.baseline_eval.reasoning_checker import check_reasoning

# ── Active thesis baselines ─────────────────────────────────────────────────────
# B1, B2 — plain-LLM; full_system (B4) is opt-in via CLI.
_SUITE_BASELINES = (
    BASELINE_MINIMAL_RAW_LLM,  # B1
    BASELINE_RULES_ONLY_LLM,   # B2
)
# Full set including full_system (opt-in via --baselines full_system)
_ALL_SUITE_BASELINES = _SUITE_BASELINES + (BASELINE_FULL_SYSTEM,)

BAR  = "═" * 70
RULE = "─" * 70


# ── Board builders ────────────────────────────────────────────────────────────
# Each function returns a FRESH board on every call.
# The registry stores the function reference (board_fn), not the result,
# so every (scenario × baseline × repeat) evaluation gets an independent object.

def _empty_board() -> list[list[int]]:
    return [[EMPTY] * 8 for _ in range(8)]


def _place(board: list[list[int]], pieces: dict[tuple[int, int], int]) -> list[list[int]]:
    for (r, c), val in pieces.items():
        board[r][c] = val
    return board


def _board_opening() -> list[list[int]]:
    return create_initial_board()


def _board_mandatory_capture() -> list[list[int]]:
    return _place(_empty_board(), {(4, 1): RED, (4, 5): RED, (3, 2): BLACK, (3, 6): BLACK})


def _board_multi_jump() -> list[list[int]]:
    return _place(_empty_board(), {(6, 1): RED, (5, 2): BLACK, (3, 4): BLACK})


def _board_promotion() -> list[list[int]]:
    return _place(_empty_board(), {(1, 2): RED, (7, 4): BLACK})


def _board_promotion_near() -> list[list[int]]:
    """Two RED men near promotion; several safe moves also available."""
    return _place(_empty_board(), {(2, 1): RED, (2, 5): RED, (7, 6): BLACK, (7, 2): BLACK})


def _board_king_move() -> list[list[int]]:
    return _place(_empty_board(), {(4, 3): RED_KING, (0, 1): BLACK})


def _board_tactical_trap() -> list[list[int]]:
    """Two mandatory jumps: one safe, one leads to immediate recapture loss."""
    return _place(_empty_board(), {
        (4, 3): RED, (4, 7): RED,
        (3, 2): BLACK, (3, 6): BLACK, (1, 0): BLACK,
    })


def _board_many_legal_moves() -> list[list[int]]:
    return _place(_empty_board(), {
        (5, 0): RED, (5, 2): RED, (5, 4): RED, (5, 6): RED,
        (3, 2): RED, (3, 6): RED,
        (0, 3): BLACK,
    })


def _board_one_legal_move() -> list[list[int]]:
    return _place(_empty_board(), {(1, 0): RED, (7, 4): BLACK})


def _board_center_control_ambiguity() -> list[list[int]]:
    """Multiple safe moves; center squares (3,4) and (3,2) are contested.
    Tests whether model reasons about center control."""
    return _place(_empty_board(), {
        (5, 2): RED, (5, 4): RED, (5, 6): RED,
        (2, 3): BLACK, (2, 5): BLACK,
        (7, 0): BLACK,
    })


def _board_mobility_tradeoff() -> list[list[int]]:
    """RED can capture (reducing BLACK mobility) or advance safely.
    Tests whether model understands mobility reduction value."""
    return _place(_empty_board(), {
        (4, 1): RED, (6, 3): RED,
        (3, 2): BLACK, (1, 4): BLACK, (7, 6): BLACK,
    })


def _board_safe_capture_vs_unsafe_capture() -> list[list[int]]:
    """Two captures available: one safe (no recapture), one unsafe.
    Ground-truth best: safe capture."""
    return _place(_empty_board(), {
        (4, 1): RED, (4, 5): RED,
        (3, 0): BLACK, (3, 4): BLACK,
        (2, 7): BLACK, (1, 4): BLACK,
    })


def _board_quiet_tie() -> list[list[int]]:
    """Multiple simple moves with near-equal minimax scores.
    Tests reasoning coherence when choice difference is minimal."""
    return _place(_empty_board(), {
        (5, 0): RED, (5, 4): RED, (5, 6): RED,
        (0, 1): BLACK, (0, 7): BLACK,
    })


def _board_losing_position() -> list[list[int]]:
    """1 RED vs 3 BLACK — clearly losing.
    Tests model behaviour and reasoning quality under losing conditions."""
    return _place(_empty_board(), {
        (4, 3): RED,
        (1, 0): BLACK, (1, 4): BLACK, (3, 6): BLACK,
    })


# ── Scenario registry ─────────────────────────────────────────────────────────
# Each entry uses board_fn (a zero-arg callable) instead of a pre-built board.
# The runner calls board_fn() per (scenario × baseline × repeat) to guarantee
# every evaluation starts from a fresh, independent board object.

_SCENARIOS: dict[str, dict[str, Any]] = {
    "opening": {
        "description": "Standard opening. 7 simple moves for RED. Tests move generation from a well-known position.",
        "board_fn": _board_opening,
    },
    "mandatory_capture": {
        "description": "Two RED pieces each have one forced jump. Tests capture-rule compliance.",
        "board_fn": _board_mandatory_capture,
    },
    "multi_jump": {
        "description": "RED at (6,1) chain-captures two BLACK pieces via [[6,1],[4,3],[2,5]]. Tests multi-jump encoding.",
        "board_fn": _board_multi_jump,
    },
    "promotion": {
        "description": "RED man at (1,2) can immediately promote. Tests whether the model prefers promotion.",
        "board_fn": _board_promotion,
    },
    "promotion_near": {
        "description": "Two RED men near promotion row. Tests near-promotion awareness and preference.",
        "board_fn": _board_promotion_near,
    },
    "king_move": {
        "description": "RED king with four-directional mobility, no captures. Tests king movement reasoning.",
        "board_fn": _board_king_move,
    },
    "tactical_trap": {
        "description": "Two mandatory jumps: safe vs recapture-loss trap. Tests trap avoidance.",
        "board_fn": _board_tactical_trap,
    },
    "many_legal_moves": {
        "description": "Open mid-game with ~11 simple moves. Tests strategic selection under high branching.",
        "board_fn": _board_many_legal_moves,
    },
    "one_legal_move": {
        "description": "Single forced move (promotion). Tests whether baselines identify the only legal path.",
        "board_fn": _board_one_legal_move,
    },
    "center_control_ambiguity": {
        "description": "Multiple safe moves; center squares contested. Tests center-control reasoning.",
        "board_fn": _board_center_control_ambiguity,
    },
    "mobility_tradeoff": {
        "description": "Capture vs safe advance: mobility-reduction tradeoff. Tests long-term planning signals.",
        "board_fn": _board_mobility_tradeoff,
    },
    "safe_capture_vs_unsafe_capture": {
        "description": "Two captures: one safe, one followed by recapture. Ground truth: safe capture.",
        "board_fn": _board_safe_capture_vs_unsafe_capture,
    },
    "quiet_tie": {
        "description": "Multiple near-equal quiet moves. Tests reasoning coherence under minimal score difference.",
        "board_fn": _board_quiet_tie,
    },
    "losing_position": {
        "description": "1 RED vs 3 BLACK. Tests behaviour and reasoning quality under clearly losing conditions.",
        "board_fn": _board_losing_position,
    },
}

ALL_SCENARIOS = tuple(_SCENARIOS.keys())


# ── Generated scenario loader ────────────────────────────────────────────────

def load_generated_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """
    Load generated hard scenarios from a JSON file produced by
    checkers/baseline_eval/generate_hard_scenarios.py.

    The file must be a JSON list of dicts with at least:
      scenario_id, category, board, side_to_move, legal_moves_count,
      best_move_index, best_score, score_gap, tactical_tags, generation_source.

    Returns the loaded list as-is (no validation beyond JSON parsing). Caller
    is responsible for sanity-checking individual entries.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Generated scenarios file must be a JSON list: {p}")
    return data


def _register_generated_scenarios(scenarios: list[dict[str, Any]]) -> list[str]:
    """
    Register each generated scenario into the global _SCENARIOS registry so
    that the dispatcher and metrics aggregation can reach them by name.

    Returns the list of scenario names that were registered (preserves order).
    """
    names: list[str] = []
    for entry in scenarios:
        sid = str(entry.get("scenario_id") or f"gen_{len(names):04d}")
        board = entry.get("board")
        if not isinstance(board, list) or len(board) != 8:
            continue
        category = entry.get("category") or "uncategorized"
        # Bind a fresh deepcopy each time so multiple repeats see independent
        # board objects (mirrors the hand-authored _board_* functions).
        def _mk_fn(b: list[list[int]]):
            from copy import deepcopy as _dc
            snapshot = _dc(b)
            return lambda: [row[:] for row in snapshot]
        _SCENARIOS[sid] = {
            "description":      f"[generated:{category}] {entry.get('description', '')}".strip(),
            "board_fn":         _mk_fn(board),
            "category":         category,
            "tactical_tags":    entry.get("tactical_tags", []),
            "generation_source": entry.get("generation_source", "unknown"),
            "best_move_index":  entry.get("best_move_index"),
            "best_score":       entry.get("best_score"),
            "score_gap":        entry.get("score_gap"),
        }
        names.append(sid)
    return names


def _scenario_category(scenario_name: str) -> str:
    meta = _SCENARIOS.get(scenario_name) or {}
    return meta.get("category") or "hand_authored"


# ── Hallucinated-path subcategoriser ─────────────────────────────────────────

def _subcategorize_hallucinated_path(
    attempted_path: Optional[list],
    board: list[list[int]],
    legal_all: list[dict],
) -> str:
    """
    Classify WHY an attempted path failed to match any legal move.

    Returns one of:
      wrong_path_format           Path is not a list of 2+ valid [int,int] pairs.
      wrong_piece_square          From-square has no RED piece (or is off-board).
      invalid_destination         To-square is off-board, on a light square, or occupied.
      mandatory_capture_violation Simple path chosen when a jump was required.
      path_not_in_legal_moves     Format OK, piece OK, dest OK — just not legal.
    """
    if not isinstance(attempted_path, list) or len(attempted_path) < 2:
        return "wrong_path_format"

    # Validate each entry is a 2-integer pair
    parsed: list[tuple[int, int]] = []
    for sq in attempted_path:
        if not (isinstance(sq, (list, tuple)) and len(sq) == 2):
            return "wrong_path_format"
        try:
            parsed.append((int(sq[0]), int(sq[1])))
        except (TypeError, ValueError):
            return "wrong_path_format"

    fr, fc = parsed[0]
    tr, tc = parsed[-1]

    # From-square: must be on board and contain a RED piece
    if not (0 <= fr < 8 and 0 <= fc < 8):
        return "wrong_piece_square"
    if board[fr][fc] not in (RED, RED_KING):
        return "wrong_piece_square"

    # To-square: must be on board, dark, and empty
    if not (0 <= tr < 8 and 0 <= tc < 8):
        return "invalid_destination"
    if (tr + tc) % 2 == 0:
        return "invalid_destination"      # light square
    if board[tr][tc] != EMPTY:
        return "invalid_destination"      # occupied

    # If there are mandatory jumps and the path looks like a simple move
    if _has_mandatory_capture(legal_all) and len(parsed) == 2:
        return "mandatory_capture_violation"

    return "path_not_in_legal_moves"


# ── Error classifier ──────────────────────────────────────────────────────────

def _classify_error(
    llm_failed: bool,
    json_valid: bool,
    legality_result: str,
    reasoning_hallucinations: list[str],
    chosen_rank: int,
    parse_error: str = "",
    index_out_of_range: bool = False,
) -> str:
    """Primary error class (first applicable wins)."""
    if llm_failed:
        return "api_call_failed"
    if not json_valid:
        if "empty_response" in parse_error:
            return "empty_response"
        if "not_a_json_object" in parse_error or "json_invalid" in legality_result:
            return "output_format_error"
        return "parse_failed"
    if index_out_of_range:
        return "invalid_index"
    if legality_result in ("hallucinated_path", "mandatory_capture_violation", "illegal"):
        return "hallucinated_path"
    if reasoning_hallucinations:
        return "reasoning_hallucination"
    if isinstance(chosen_rank, int) and chosen_rank > 0 and chosen_rank > 3:
        return "strategic_error"
    return "clean"


# ── Empty record template ─────────────────────────────────────────────────────

def _empty_record(scenario: str, desc: str, baseline: str, board: list) -> dict[str, Any]:
    return {
        "scenario":                       scenario,
        "scenario_description":           desc,
        "baseline":                       baseline,
        "legal_moves_count":              0,
        "board_before":                   [row[:] for row in board],
        "system_prompt":                  "",
        "user_prompt":                    "",
        "raw_model_output":               None,
        "parsed_output":                  None,
        "json_valid":                     False,
        "attempted_index":                None,
        "attempted_path":                 None,
        "path_matches_legal":             False,
        "hallucinated_path":              False,
        "hallucinated_path_subtype":      None,
        "legality_result":                "no_legal_moves",
        "legality_error_reason":          None,
        "move_legality_error":            True,
        "reasoning":                      "",
        "reasoning_hallucinations":       [],
        "reasoning_hallucination_count":  0,
        "reasoning_truthfulness_passed":  True,
        "reasoning_check_applicable":     False,
        "contradiction_details":          None,
        "strategic_error":                False,
        "chosen_move":                    None,
        "best_move":                      None,
        "chosen_minimax_rank":            0,
        "top1_hit":                       False,
        "top3_hit":                       False,
        "score_gap":                      None,
        "error_class":                    "no_legal_moves",
    }


# ── Core scenario runner — path-JSON output ───────────────────────────────────

def _run_scenario_path_json(
    scenario: str,
    desc: str,
    board: list[list[int]],
    baseline: str,
    system: str,
    user_msg: str,
    scored: list[dict],
    legal_all: list[dict],
    show_prompts: bool,
) -> dict[str, Any]:
    if show_prompts:
        _print_prompts(baseline, 0, system, user_msg)

    llm_failed    = False
    raw_output:  Optional[str]  = None
    parsed_obj:  Optional[dict] = None
    parse_error: str            = "not_attempted"
    json_valid:  bool           = False
    llm_reasoning: str          = ""

    try:
        raw_output    = call_baseline_llm(system, user_msg)
        parsed_obj, parse_error = _parse_json_response(raw_output)
        json_valid    = parsed_obj is not None
        llm_reasoning = (parsed_obj or {}).get("reasoning", "")
    except Exception as e:
        llm_failed  = True
        raw_output  = None
        parse_error = f"llm_call_failed: {e}"

    attempted_path: Optional[list] = None
    if parsed_obj is not None:
        raw_path = parsed_obj.get("move_path")
        if isinstance(raw_path, list):
            attempted_path = raw_path

    chosen_move:             Optional[dict] = None
    legality_result:         str            = "illegal"
    legality_reason:         str            = ""
    path_matches:            bool           = False
    hallucinated:            bool           = False
    hallucinated_path_subtype: Optional[str] = None

    if json_valid and attempted_path is not None:
        matched = _find_move_by_path(legal_all, attempted_path)
        if matched is not None:
            path_matches = True
            if _has_mandatory_capture(legal_all) and matched.get("type") != "jump":
                legality_result           = "mandatory_capture_violation"
                legality_reason           = "jump available but model chose simple"
                hallucinated              = True
                hallucinated_path_subtype = "mandatory_capture_violation"
            else:
                legality_result = "legal"
                chosen_move     = matched
        else:
            legality_result           = "hallucinated_path"
            legality_reason           = f"path {attempted_path} not found in legal moves"
            hallucinated              = True
            hallucinated_path_subtype = _subcategorize_hallucinated_path(
                attempted_path, board, legal_all
            )
    elif json_valid and attempted_path is None:
        legality_result = "illegal"
        legality_reason = "move_path missing or malformed"
    elif not json_valid:
        legality_result = "illegal"
        legality_reason = f"json_invalid: {parse_error}"

    chosen_rank = 0
    score_gap   = float("inf")
    top1_hit    = False
    top3_hit    = False
    if chosen_move is not None and scored:
        chosen_rank, score_gap = _rank_and_gap(scored, chosen_move)
        top1_hit = chosen_rank == 1
        top3_hit = 1 <= chosen_rank <= 3

    rr = check_reasoning(
        llm_reasoning, chosen_move, legal_all, scored, baseline,
        system_prompt=system, user_prompt=user_msg,
    )

    error_class = _classify_error(
        llm_failed=llm_failed,
        json_valid=json_valid,
        legality_result=legality_result,
        reasoning_hallucinations=rr["reasoning_hallucinations"],
        chosen_rank=chosen_rank,
        parse_error=parse_error,
    )

    return {
        "scenario":                       scenario,
        "scenario_description":           desc,
        "baseline":                       baseline,
        "legal_moves_count":              len(legal_all),
        "board_before":                   [row[:] for row in board],
        "system_prompt":                  system,
        "user_prompt":                    user_msg,
        "raw_model_output":               raw_output,
        "parsed_output":                  parsed_obj,
        "json_valid":                     json_valid,
        "attempted_index":                None,
        "attempted_path":                 attempted_path,
        "path_matches_legal":             path_matches,
        "hallucinated_path":              hallucinated,
        "hallucinated_path_subtype":      hallucinated_path_subtype,
        "legality_result":                legality_result,
        "legality_error_reason":          legality_reason or None,
        "move_legality_error":            legality_result != "legal",
        "reasoning":                      llm_reasoning,
        "reasoning_hallucinations":       rr["reasoning_hallucinations"],
        "reasoning_hallucination_count":  rr["reasoning_hallucination_count"],
        "reasoning_truthfulness_passed":  rr["reasoning_truthfulness_passed"],
        "reasoning_check_applicable":     rr["reasoning_check_applicable"],
        "contradiction_details":          "; ".join(rr["reasoning_hallucinations"]) or None,
        "strategic_error":                legality_result == "legal" and chosen_rank > 3,
        "chosen_move":                    _slim(chosen_move),
        "best_move":                      _slim(scored[0]) if scored else None,
        "chosen_minimax_rank":            chosen_rank,
        "top1_hit":                       top1_hit,
        "top3_hit":                       top3_hit,
        "score_gap":                      score_gap if score_gap != float("inf") else None,
        "error_class":                    error_class,
    }


# ── Core scenario runner — index output ───────────────────────────────────────

def _run_scenario_index(
    scenario: str,
    desc: str,
    board: list[list[int]],
    baseline: str,
    system: str,
    user_msg: str,
    scored: list[dict],
    legal_all: list[dict],
    show_prompts: bool,
) -> dict[str, Any]:
    import re as _re

    if show_prompts:
        _print_prompts(baseline, 0, system, user_msg)

    llm_failed    = False
    raw_output:  Optional[str]  = None
    parsed_obj:  Optional[dict] = None
    parse_error: str            = "not_attempted"
    json_valid:  bool           = False
    llm_reasoning: str          = ""

    try:
        raw_output    = call_baseline_llm(system, user_msg)
        parsed_obj, parse_error = _parse_json_response(raw_output)
        json_valid    = parsed_obj is not None
        llm_reasoning = (parsed_obj or {}).get("reasoning", "")
    except Exception as e:
        llm_failed  = True
        raw_output  = None
        parse_error = f"llm_call_failed: {e}"

    raw_index:     Optional[int] = None
    attempted_idx: Optional[int] = None
    if parsed_obj is not None:
        v = parsed_obj.get("chosen_index")
        if isinstance(v, bool):
            pass
        elif isinstance(v, (int, float)):
            raw_index = int(v); attempted_idx = raw_index
        elif isinstance(v, str):
            m = _re.fullmatch(r"-?\d+", v.strip())
            if m:
                raw_index = int(v.strip()); attempted_idx = raw_index

    chosen_move:        Optional[dict] = None
    legality_result:    str            = "illegal"
    legality_reason:    str            = ""
    index_out_of_range: bool           = False

    if llm_failed or not json_valid:
        legality_result = "illegal"
        legality_reason = f"json_invalid: {parse_error}"
    elif raw_index is None:
        legality_result = "illegal"
        legality_reason = "chosen_index missing or not a valid integer"
    elif not (0 <= raw_index < len(legal_all)):
        legality_result    = "illegal"
        legality_reason    = f"index {raw_index} out of range [0..{len(legal_all)-1}]"
        index_out_of_range = True
    else:
        chosen_move     = legal_all[raw_index]
        legality_result = "legal"

    chosen_rank = 0
    score_gap   = float("inf")
    top1_hit    = False
    top3_hit    = False
    if chosen_move is not None and scored:
        chosen_rank, score_gap = _rank_and_gap(scored, chosen_move)
        top1_hit = chosen_rank == 1
        top3_hit = 1 <= chosen_rank <= 3

    rr = check_reasoning(
        llm_reasoning, chosen_move, legal_all, scored, baseline,
        system_prompt=system, user_prompt=user_msg,
    )

    error_class = _classify_error(
        llm_failed=llm_failed,
        json_valid=json_valid,
        legality_result=legality_result,
        reasoning_hallucinations=rr["reasoning_hallucinations"],
        chosen_rank=chosen_rank,
        parse_error=parse_error,
        index_out_of_range=index_out_of_range,
    )

    return {
        "scenario":                       scenario,
        "scenario_description":           desc,
        "baseline":                       baseline,
        "legal_moves_count":              len(legal_all),
        "board_before":                   [row[:] for row in board],
        "system_prompt":                  system,
        "user_prompt":                    user_msg,
        "raw_model_output":               raw_output,
        "parsed_output":                  parsed_obj,
        "json_valid":                     json_valid,
        "attempted_index":                attempted_idx,
        "attempted_path":                 chosen_move.get("path") if chosen_move else None,
        "path_matches_legal":             legality_result == "legal",
        "hallucinated_path":              False,   # index baseline cannot hallucinate coords
        "hallucinated_path_subtype":      None,
        "legality_result":                legality_result,
        "legality_error_reason":          legality_reason or None,
        "move_legality_error":            legality_result != "legal",
        "reasoning":                      llm_reasoning,
        "reasoning_hallucinations":       rr["reasoning_hallucinations"],
        "reasoning_hallucination_count":  rr["reasoning_hallucination_count"],
        "reasoning_truthfulness_passed":  rr["reasoning_truthfulness_passed"],
        "reasoning_check_applicable":     rr["reasoning_check_applicable"],
        "contradiction_details":          "; ".join(rr["reasoning_hallucinations"]) or None,
        "strategic_error":                legality_result == "legal" and chosen_rank > 3,
        "chosen_move":                    _slim(chosen_move),
        "best_move":                      _slim(scored[0]) if scored else None,
        "chosen_minimax_rank":            chosen_rank,
        "top1_hit":                       top1_hit,
        "top3_hit":                       top3_hit,
        "score_gap":                      score_gap if score_gap != float("inf") else None,
        "error_class":                    error_class,
    }


# ── Core scenario runner — B3_strategic_facts_llm (analysis + index + path) ─

def _validate_move_analysis(
    parsed_obj: Optional[dict[str, Any]],
    n_legal: int,
) -> tuple[bool, str, list[int]]:
    """
    Validate the model's move_analysis array.

    Returns (ok, reason, covered_indices).
      ok=True  ⇔ move_analysis is a list of dicts whose 'index' fields cover
                 every integer in [0, n_legal) exactly once (extras allowed
                 are NOT — duplicates and unknown indices fail).
      ok=False ⇒ reason is a short descriptor used as the primary
                 error_class via the analysis_incomplete branch.
    """
    if not isinstance(parsed_obj, dict):
        return False, "move_analysis_missing", []
    arr = parsed_obj.get("move_analysis")
    if not isinstance(arr, list):
        return False, "move_analysis_missing", []
    seen: list[int] = []
    for entry in arr:
        if not isinstance(entry, dict):
            return False, "move_analysis_entry_not_object", seen
        v = entry.get("index")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, "move_analysis_index_not_int", seen
        idx = int(v)
        if not (0 <= idx < n_legal):
            return False, "move_analysis_index_out_of_range", seen
        seen.append(idx)
    unique = set(seen)
    if len(unique) != n_legal:
        return False, "move_analysis_incomplete_coverage", seen
    return True, "", seen


def _run_scenario_strategic_facts(
    scenario: str,
    desc: str,
    board: list[list[int]],
    baseline: str,
    system: str,
    user_msg: str,
    scored: list[dict],
    legal_all: list[dict],
    show_prompts: bool,
) -> dict[str, Any]:
    """
    Runner for B3_strategic_facts_llm.

    The model is expected to emit
      {"move_analysis": [{"index": i, ...} for each legal i],
       "chosen_index": int, "chosen_path": [...], "reasoning": "..."}.

    Validations (no auto-correction, no retry, no fallback):
      • JSON parses to dict.
      • move_analysis covers every legal index exactly once.
      • chosen_index is an int in [0, N).
      • chosen_path matches one of the legal paths.

    If move_analysis fails coverage we classify the record as
    'analysis_incomplete' (set after the standard classifier) so a model that
    skips the required per-move analysis is distinguishable from one that
    simply hallucinated a path.
    """
    import re as _re

    if show_prompts:
        _print_prompts(baseline, 0, system, user_msg)

    llm_failed    = False
    raw_output:  Optional[str]  = None
    parsed_obj:  Optional[dict] = None
    parse_error: str            = "not_attempted"
    json_valid:  bool           = False
    llm_reasoning: str          = ""

    try:
        raw_output    = call_baseline_llm(system, user_msg)
        parsed_obj, parse_error = _parse_json_response(raw_output)
        json_valid    = parsed_obj is not None
        llm_reasoning = (parsed_obj or {}).get("reasoning", "")
    except Exception as e:
        llm_failed  = True
        raw_output  = None
        parse_error = f"llm_call_failed: {e}"

    raw_index:     Optional[int] = None
    attempted_idx: Optional[int] = None
    attempted_path: Optional[list] = None
    if parsed_obj is not None:
        v = parsed_obj.get("chosen_index")
        if isinstance(v, bool):
            pass
        elif isinstance(v, (int, float)):
            raw_index = int(v); attempted_idx = raw_index
        elif isinstance(v, str):
            m = _re.fullmatch(r"-?\d+", v.strip())
            if m:
                raw_index = int(v.strip()); attempted_idx = raw_index
        raw_path = parsed_obj.get("chosen_path")
        if isinstance(raw_path, list):
            attempted_path = raw_path

    chosen_move:        Optional[dict] = None
    legality_result:    str            = "illegal"
    legality_reason:    str            = ""
    index_out_of_range: bool           = False
    hallucinated:       bool           = False
    hallucinated_path_subtype: Optional[str] = None
    path_matches:       bool           = False

    if llm_failed or not json_valid:
        legality_result = "illegal"
        legality_reason = f"json_invalid: {parse_error}"
    elif raw_index is None:
        legality_result = "illegal"
        legality_reason = "chosen_index missing or not a valid integer"
    elif not (0 <= raw_index < len(legal_all)):
        legality_result    = "illegal"
        legality_reason    = f"index {raw_index} out of range [0..{len(legal_all)-1}]"
        index_out_of_range = True
    else:
        # Index in range — verify chosen_path matches the move at that index.
        engine_move = legal_all[raw_index]
        matched_by_path = (
            _find_move_by_path(legal_all, attempted_path)
            if attempted_path is not None else None
        )
        if attempted_path is None:
            legality_result = "illegal"
            legality_reason = "chosen_path missing or not a list"
            hallucinated   = True
            hallucinated_path_subtype = "wrong_path_format"
        elif matched_by_path is None:
            legality_result = "hallucinated_path"
            legality_reason = f"chosen_path {attempted_path} not found in legal moves"
            hallucinated   = True
            hallucinated_path_subtype = _subcategorize_hallucinated_path(
                attempted_path, board, legal_all,
            )
        elif matched_by_path is not engine_move:
            # path is legal but doesn't match chosen_index — treat as legal
            # and use the path as ground truth (model output is internally
            # inconsistent but we credit the index/path match they intended).
            legality_result = "legal"
            chosen_move     = matched_by_path
            path_matches    = True
            legality_reason = "chosen_index/chosen_path mismatch (used path)"
        else:
            legality_result = "legal"
            chosen_move     = engine_move
            path_matches    = True

    chosen_rank = 0
    score_gap   = float("inf")
    top1_hit    = False
    top3_hit    = False
    if chosen_move is not None and scored:
        chosen_rank, score_gap = _rank_and_gap(scored, chosen_move)
        top1_hit = chosen_rank == 1
        top3_hit = 1 <= chosen_rank <= 3

    rr = check_reasoning(
        llm_reasoning, chosen_move, legal_all, scored, baseline,
        system_prompt=system, user_prompt=user_msg,
    )

    # Validate the required move_analysis array — must cover every legal
    # index exactly once.
    analysis_ok, analysis_reason, analysis_seen = _validate_move_analysis(
        parsed_obj, len(legal_all),
    )

    error_class = _classify_error(
        llm_failed=llm_failed,
        json_valid=json_valid,
        legality_result=legality_result,
        reasoning_hallucinations=rr["reasoning_hallucinations"],
        chosen_rank=chosen_rank,
        parse_error=parse_error,
        index_out_of_range=index_out_of_range,
    )
    # If the run is otherwise clean / a mere strategic_error but the model
    # skipped the required per-move analysis, upgrade the classification to
    # 'analysis_incomplete' so we can separate that failure mode from
    # hallucinated paths or genuine strategic mistakes.
    if (json_valid and not llm_failed and not analysis_ok
            and error_class in ("clean", "strategic_error")):
        error_class = "analysis_incomplete"

    return {
        "scenario":                       scenario,
        "scenario_description":           desc,
        "baseline":                       baseline,
        "legal_moves_count":              len(legal_all),
        "board_before":                   [row[:] for row in board],
        "system_prompt":                  system,
        "user_prompt":                    user_msg,
        "raw_model_output":               raw_output,
        "parsed_output":                  parsed_obj,
        "json_valid":                     json_valid,
        "attempted_index":                attempted_idx,
        "attempted_path":                 attempted_path,
        "path_matches_legal":             path_matches,
        "hallucinated_path":              hallucinated,
        "hallucinated_path_subtype":      hallucinated_path_subtype,
        "legality_result":                legality_result,
        "legality_error_reason":          legality_reason or None,
        "move_legality_error":            legality_result != "legal",
        "reasoning":                      llm_reasoning,
        "reasoning_hallucinations":       rr["reasoning_hallucinations"],
        "reasoning_hallucination_count":  rr["reasoning_hallucination_count"],
        "reasoning_truthfulness_passed":  rr["reasoning_truthfulness_passed"],
        "reasoning_check_applicable":     rr["reasoning_check_applicable"],
        "contradiction_details":          "; ".join(rr["reasoning_hallucinations"]) or None,
        "strategic_error":                legality_result == "legal" and chosen_rank > 3,
        "chosen_move":                    _slim(chosen_move),
        "best_move":                      _slim(scored[0]) if scored else None,
        "chosen_minimax_rank":            chosen_rank,
        "top1_hit":                       top1_hit,
        "top3_hit":                       top3_hit,
        "score_gap":                      score_gap if score_gap != float("inf") else None,
        "move_analysis_complete":         analysis_ok,
        "move_analysis_reason":           analysis_reason or None,
        "move_analysis_indices":          analysis_seen,
        "error_class":                    error_class,
    }


# ── Freeform-path subcategoriser (B3e) ───────────────────────────────────────

def _subcategorize_freeform_path(
    attempted_path: Optional[list],
    board: list[list[int]],
    legal_all: list[dict],
) -> str:
    """
    Classify WHY a freeform chosen_path failed to match any legal move.

    Subtypes (B3e taxonomy):
      malformed_coordinates       Path entries are not [int,int] pairs / path < 2.
      wrong_piece_square          From-square has no RED piece (or off-board).
      invalid_destination         To-square off-board, on a light square, or occupied.
      illegal_direction           Simple move by RED man goes DOWN instead of UP.
      mandatory_capture_violation Simple-move path chosen when a jump was required.
      path_not_in_legal_list      Format/piece/dest OK — just not in the legal list.
    """
    if not isinstance(attempted_path, list) or len(attempted_path) < 2:
        return "malformed_coordinates"

    parsed: list[tuple[int, int]] = []
    for sq in attempted_path:
        if not (isinstance(sq, (list, tuple)) and len(sq) == 2):
            return "malformed_coordinates"
        try:
            parsed.append((int(sq[0]), int(sq[1])))
        except (TypeError, ValueError):
            return "malformed_coordinates"

    fr, fc = parsed[0]
    tr, tc = parsed[-1]

    if not (0 <= fr < 8 and 0 <= fc < 8):
        return "wrong_piece_square"
    if board[fr][fc] not in (RED, RED_KING):
        return "wrong_piece_square"

    if not (0 <= tr < 8 and 0 <= tc < 8):
        return "invalid_destination"
    if (tr + tc) % 2 == 0:
        return "invalid_destination"
    if board[tr][tc] != EMPTY:
        return "invalid_destination"

    # Illegal direction: a RED man (not king) trying to move toward higher rows.
    if len(parsed) == 2 and board[fr][fc] == RED and tr > fr:
        return "illegal_direction"

    if _has_mandatory_capture(legal_all) and len(parsed) == 2:
        return "mandatory_capture_violation"

    return "path_not_in_legal_list"


# ── Core scenario runner — B3e freeform-path strategic-facts ─────────────────

def _run_scenario_strategic_facts_freeform_path(
    scenario: str,
    desc: str,
    board: list[list[int]],
    baseline: str,
    system: str,
    user_msg: str,
    scored: list[dict],
    legal_all: list[dict],
    show_prompts: bool,
) -> dict[str, Any]:
    """
    Runner for B3e (rules_legal_moves_llm_strategic_facts_freeform_path).

    The model receives the same tactical/strategic facts payload as B3d but is
    asked to output ONLY chosen_path (no index, no move_analysis).

    Classification policy:
      • invalid JSON                       → parse_failed / output_format_error
                                             (NOT hallucinated_path)
      • missing/malformed chosen_path      → output_format_error
                                             (NOT hallucinated_path)
      • chosen_path exactly matches a legal path → legal
      • mandatory-capture violation         → hallucinated_path
                                              (subtype mandatory_capture_violation)
      • chosen_path not in legal list      → hallucinated_path (+ subtype)

    top1 / top3 / rank are computed ONLY when legality_result == "legal".

    No verifier, retry, fallback, override, or B4 ranker logic is used.
    """
    if show_prompts:
        _print_prompts(baseline, 0, system, user_msg)

    llm_failed    = False
    raw_output:  Optional[str]  = None
    parsed_obj:  Optional[dict] = None
    parse_error: str            = "not_attempted"
    json_valid:  bool           = False
    llm_reasoning: str          = ""

    try:
        raw_output    = call_baseline_llm(system, user_msg)
        parsed_obj, parse_error = _parse_json_response(raw_output)
        json_valid    = parsed_obj is not None
        llm_reasoning = (parsed_obj or {}).get("reasoning", "")
    except Exception as e:
        llm_failed  = True
        raw_output  = None
        parse_error = f"llm_call_failed: {e}"

    attempted_path: Optional[list] = None
    chosen_path_present = False
    if parsed_obj is not None:
        if "chosen_path" in parsed_obj:
            chosen_path_present = True
            raw_path = parsed_obj.get("chosen_path")
            if isinstance(raw_path, list):
                attempted_path = raw_path

    chosen_move:               Optional[dict] = None
    legality_result:           str            = "illegal"
    legality_reason:           str            = ""
    hallucinated:              bool           = False
    hallucinated_path_subtype: Optional[str]  = None
    path_matches:              bool           = False
    path_copy_error:           bool           = False
    error_class:               str

    if llm_failed:
        legality_result = "illegal"
        legality_reason = f"llm_call_failed: {parse_error}"
        error_class = "api_call_failed"
    elif not json_valid:
        legality_result = "illegal"
        legality_reason = f"json_invalid: {parse_error}"
        if "empty_response" in parse_error:
            error_class = "empty_response"
        elif "not_a_json_object" in parse_error:
            error_class = "output_format_error"
        else:
            error_class = "parse_failed"
    elif not chosen_path_present:
        legality_result = "output_format_error"
        legality_reason = "chosen_path field missing"
        error_class = "output_format_error"
    elif attempted_path is None:
        legality_result = "output_format_error"
        legality_reason = "chosen_path is not a list"
        error_class = "output_format_error"
    else:
        matched = _find_move_by_path(legal_all, attempted_path)
        if matched is not None:
            if _has_mandatory_capture(legal_all) and matched.get("type") != "jump":
                # Shouldn't happen: engine only returns jumps when mandatory,
                # so matched.type == jump in that case. Defensive branch.
                legality_result           = "hallucinated_path"
                legality_reason           = "jump available but model chose simple"
                hallucinated              = True
                hallucinated_path_subtype = "mandatory_capture_violation"
                path_copy_error           = True
                error_class               = "hallucinated_path"
            else:
                legality_result = "legal"
                chosen_move     = matched
                path_matches    = True
                error_class     = "clean"  # may be upgraded below
        else:
            legality_result           = "hallucinated_path"
            legality_reason           = (
                f"chosen_path {attempted_path} not found in legal moves"
            )
            hallucinated              = True
            hallucinated_path_subtype = _subcategorize_freeform_path(
                attempted_path, board, legal_all,
            )
            path_copy_error           = True
            error_class               = "hallucinated_path"

    chosen_rank = 0
    score_gap   = float("inf")
    top1_hit    = False
    top3_hit    = False
    if chosen_move is not None and scored:
        chosen_rank, score_gap = _rank_and_gap(scored, chosen_move)
        top1_hit = chosen_rank == 1
        top3_hit = 1 <= chosen_rank <= 3

    rr = check_reasoning(
        llm_reasoning, chosen_move, legal_all, scored, baseline,
        system_prompt=system, user_prompt=user_msg,
    )

    # Promote clean → reasoning_hallucination / strategic_error when applicable
    # (mirrors _classify_error's priority order for legal moves).
    if error_class == "clean":
        if rr["reasoning_hallucinations"]:
            error_class = "reasoning_hallucination"
        elif isinstance(chosen_rank, int) and chosen_rank > 3:
            error_class = "strategic_error"

    return {
        "scenario":                       scenario,
        "scenario_description":           desc,
        "baseline":                       baseline,
        "legal_moves_count":              len(legal_all),
        "board_before":                   [row[:] for row in board],
        "system_prompt":                  system,
        "user_prompt":                    user_msg,
        "raw_model_output":               raw_output,
        "parsed_output":                  parsed_obj,
        "json_valid":                     json_valid,
        "attempted_index":                None,
        "attempted_path":                 attempted_path,
        "path_matches_legal":             path_matches,
        "hallucinated_path":              hallucinated,
        "hallucinated_path_subtype":      hallucinated_path_subtype,
        "legality_result":                legality_result,
        "legality_error_reason":          legality_reason or None,
        "move_legality_error":            legality_result != "legal",
        "reasoning":                      llm_reasoning,
        "reasoning_hallucinations":       rr["reasoning_hallucinations"],
        "reasoning_hallucination_count":  rr["reasoning_hallucination_count"],
        "reasoning_truthfulness_passed":  rr["reasoning_truthfulness_passed"],
        "reasoning_check_applicable":     rr["reasoning_check_applicable"],
        "contradiction_details":          "; ".join(rr["reasoning_hallucinations"]) or None,
        "strategic_error":                legality_result == "legal" and chosen_rank > 3,
        "chosen_move":                    _slim(chosen_move),
        "best_move":                      _slim(scored[0]) if scored else None,
        "chosen_minimax_rank":            chosen_rank,
        "top1_hit":                       top1_hit,
        "top3_hit":                       top3_hit,
        "score_gap":                      score_gap if score_gap != float("inf") else None,
        "path_copy_error":                path_copy_error,
        "error_class":                    error_class,
    }


# ── Core scenario runner — full_system ────────────────────────────────────────

def _run_scenario_full_system(
    scenario: str,
    desc: str,
    board: list[list[int]],
    scored: list[dict],
    legal_all: list[dict],
    show_prompts: bool,
) -> dict[str, Any]:
    """
    Stream the neuro-symbolic pipeline for one position, stop after ranker_agent.
    update_agent is never executed — no game state is modified.
    """
    from checkers.graph.graph import checkers_graph
    from checkers.state.state import CheckersState

    if show_prompts:
        print()
        print("═" * 60)
        print("AUDIT — PROMPTS  [FULL_SYSTEM]")
        print("═" * 60)
        print("full_system prompts are constructed internally by ranker_agent.")
        print("See: checkers/agents/ranker_agent.py")
        print("═" * 60)
        print()

    state = CheckersState(board=board, current_player=RED, turn_number=0).model_dump()
    cfg   = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}

    _sym_scored: list[dict] = []
    _proposal:   list[dict] = []
    _chosen:     Optional[dict] = None
    _reasoning:  str = ""
    _retry_ct:   int = 0
    llm_failed   = False

    try:
        for chunk in checkers_graph.stream(
            state,
            stream_mode="updates",
            interrupt_after=["ranker_agent"],
            config=cfg,
        ):
            for node_name, delta in chunk.items():
                if node_name in ("__interrupt__", "__end__"):
                    continue
                if not isinstance(delta, dict):
                    continue
                state.update(delta)
                if node_name == "scorer_node":
                    _sym_scored = list(state.get("symbolic_scored_moves") or [])
                elif node_name == "deterministic_proposal_node":
                    _proposal   = list(state.get("legal_moves") or [])
                elif node_name == "ranker_agent":
                    _chosen   = state.get("chosen_move")
                    _reasoning = state.get("last_move_reasoning") or ""
                    _retry_ct  = int(state.get("ranker_retry_count") or 0)
    except Exception as e:
        print(f"[full_system] graph stream error: {e}", file=sys.stderr)
        llm_failed = True

    chosen_move = _chosen

    # Legality check against pre-turn legal list
    legality_result = "illegal"
    legality_reason = ""
    if chosen_move is not None:
        if any(
            _find_move_by_path([lm], chosen_move.get("path")) is not None
            for lm in legal_all
        ):
            legality_result = "legal"
        else:
            legality_result = "illegal"
            legality_reason = "chosen_move path not found in pre-turn legal moves"

    # Evaluation using the oracle scored list (consistent with other baselines)
    chosen_rank = 0
    score_gap   = float("inf")
    top1_hit    = False
    top3_hit    = False
    if chosen_move is not None and scored:
        chosen_rank, score_gap = _rank_and_gap(scored, chosen_move)
        top1_hit = chosen_rank == 1
        top3_hit = 1 <= chosen_rank <= 3

    rr = check_reasoning(
        _reasoning, chosen_move, legal_all, scored, BASELINE_FULL_SYSTEM,
        system_prompt="", user_prompt="",  # internal pipeline; prompts not accessible here
    )

    error_class = _classify_error(
        llm_failed=llm_failed,
        json_valid=True,
        legality_result=legality_result,
        reasoning_hallucinations=rr["reasoning_hallucinations"],
        chosen_rank=chosen_rank,
        parse_error="",
    )

    return {
        "scenario":                       scenario,
        "scenario_description":           desc,
        "baseline":                       BASELINE_FULL_SYSTEM,
        "legal_moves_count":              len(legal_all),
        "board_before":                   [row[:] for row in board],
        "system_prompt":                  "N/A (internal pipeline)",
        "user_prompt":                    "N/A (internal pipeline)",
        "raw_model_output":               "N/A (internal)",
        "parsed_output":                  None,
        "json_valid":                     not llm_failed,
        "attempted_index":                None,
        "attempted_path":                 chosen_move.get("path") if chosen_move else None,
        "path_matches_legal":             legality_result == "legal",
        "hallucinated_path":              legality_result not in ("legal",) and not llm_failed,
        "hallucinated_path_subtype":      None,   # internal pipeline validates its own output
        "legality_result":                legality_result,
        "legality_error_reason":          legality_reason or None,
        "move_legality_error":            legality_result != "legal",
        "reasoning":                      _reasoning,
        "reasoning_hallucinations":       rr["reasoning_hallucinations"],
        "reasoning_hallucination_count":  rr["reasoning_hallucination_count"],
        "reasoning_truthfulness_passed":  rr["reasoning_truthfulness_passed"],
        "reasoning_check_applicable":     rr["reasoning_check_applicable"],
        "contradiction_details":          "; ".join(rr["reasoning_hallucinations"]) or None,
        "strategic_error":                legality_result == "legal" and chosen_rank > 3,
        "chosen_move":                    _slim(chosen_move),
        "best_move":                      _slim(scored[0]) if scored else None,
        "chosen_minimax_rank":            chosen_rank,
        "top1_hit":                       top1_hit,
        "top3_hit":                       top3_hit,
        "score_gap":                      score_gap if score_gap != float("inf") else None,
        "error_class":                    "llm_call_failed" if llm_failed else error_class,
        # full_system extras
        "proposal_candidates":            len(_proposal),
        "ranker_retry_count":             _retry_ct,
    }


# ── Dispatch ──────────────────────────────────────────────────────────────────

def run_scenario_for_baseline(
    scenario_name: str,
    board: list[list[int]],
    baseline: str,
    show_prompts: bool = False,
) -> dict[str, Any]:
    desc      = (_SCENARIOS.get(scenario_name) or {}).get("description", "")
    legal_all = get_all_legal_moves(board, RED)

    if not legal_all:
        return _empty_record(scenario_name, desc, baseline, board)

    scored: list[dict] = []
    try:
        scored, _, _, _ = score_all_legal_moves(board, RED, None)
    except Exception as e:
        print(f"[scenario_suite] scoring oracle failed ({scenario_name}/{baseline}): {e}",
              file=sys.stderr)

    if baseline == BASELINE_MINIMAL_RAW_LLM:
        system = _MINIMAL_RAW_LLM_SYSTEM
        user   = _build_path_json_user(board, 1)
        return _run_scenario_path_json(scenario_name, desc, board, baseline, system, user, scored, legal_all, show_prompts)

    if baseline == BASELINE_RULES_ONLY_LLM:
        system = _RULES_ONLY_LLM_SYSTEM
        user   = _build_path_json_user(board, 1)
        return _run_scenario_path_json(scenario_name, desc, board, baseline, system, user, scored, legal_all, show_prompts)

    if baseline == BASELINE_FULL_SYSTEM:
        return _run_scenario_full_system(scenario_name, desc, board, scored, legal_all, show_prompts)

    raise ValueError(f"Unknown baseline: {baseline!r}")


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"[trace]   {path}")


_CSV_FIELDS = [
    "scenario", "category", "baseline", "repeat", "legal_moves_count",
    "json_valid", "legality_result", "legality_error_reason",
    "move_legality_error", "hallucinated_path", "hallucinated_path_subtype",
    "path_matches_legal", "attempted_index", "chosen_minimax_rank", "score_gap",
    "top1_hit", "top3_hit", "strategic_error",
    "reasoning_hallucination_count", "reasoning_truthfulness_passed",
    "reasoning_check_applicable",
    "error_class", "contradiction_details",
]


def _save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row.setdefault("category", _scenario_category(rec.get("scenario", "")))
            writer.writerow(row)
    print(f"[metrics] {path}")


# ── Category-level metrics aggregator ────────────────────────────────────────

def aggregate_category_metrics(
    records: list[dict[str, Any]],
    baselines: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute per-(category, baseline) metrics:
      legal_rate, top1_rate, top3_rate, avg_rank, avg_score_gap,
      strategic_error_rate, hallucinated_path_rate,
      reasoning_hallucination_rate, output_format_error_rate,
      api_error_rate, n.

    Returns: {category: {baseline: {metric_name: value, ...}}}
    """
    by_cat_bl: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        cat = _scenario_category(r.get("scenario", "")) or "uncategorized"
        bl  = r.get("baseline", "")
        by_cat_bl.setdefault(cat, {}).setdefault(bl, []).append(r)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for cat, by_bl in by_cat_bl.items():
        out[cat] = {}
        for bl in baselines:
            recs = by_bl.get(bl, [])
            n    = len(recs)
            if n == 0:
                continue
            legal      = sum(1 for r in recs if r.get("legality_result") == "legal")
            top1       = sum(1 for r in recs if r.get("top1_hit"))
            top3       = sum(1 for r in recs if r.get("top3_hit"))
            halluc     = sum(1 for r in recs if r.get("hallucinated_path"))
            strat      = sum(1 for r in recs if r.get("strategic_error"))
            rsn_halluc = sum(1 for r in recs
                             if not r.get("reasoning_truthfulness_passed", True))
            fmt_err    = sum(1 for r in recs
                             if r.get("error_class") in ("output_format_error",
                                                          "parse_failed",
                                                          "empty_response"))
            api_err    = sum(1 for r in recs
                             if r.get("error_class") == "api_call_failed")
            ranks = [r["chosen_minimax_rank"] for r in recs
                     if r.get("chosen_minimax_rank", 0) > 0]
            gaps  = [r["score_gap"] for r in recs
                     if r.get("score_gap") is not None
                     and r["score_gap"] != float("inf")]
            out[cat][bl] = {
                "n":                            float(n),
                "legal_rate":                   legal / n,
                "top1_rate":                    top1 / n,
                "top3_rate":                    top3 / n,
                "hallucinated_path_rate":       halluc / n,
                "strategic_error_rate":         strat / n,
                "reasoning_hallucination_rate": rsn_halluc / n,
                "output_format_error_rate":     fmt_err / n,
                "api_error_rate":               api_err / n,
                "avg_rank":                     (sum(ranks) / len(ranks)) if ranks else float("nan"),
                "avg_score_gap":                (sum(gaps) / len(gaps)) if gaps else float("nan"),
            }
    return out


def aggregate_freeform_path_metrics(
    records: list[dict[str, Any]],
    baseline: str = BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
) -> dict[str, float]:
    """
    Freeform-path baseline (B3e) headline metrics.

      legal_rate_all_rows              legal / total
      legal_rate_parse_valid           legal / json_valid
      hallucinated_path_rate_parse_valid
                                       hallucinated_path / json_valid
      parse_failed_rate                {parse_failed,output_format_error,empty_response}
                                       / total
      path_copy_error_rate             rows with path_copy_error / total

    Returns {} when no rows match the baseline.
    """
    rows = [r for r in records if r.get("baseline") == baseline]
    n = len(rows)
    if n == 0:
        return {}
    parse_valid = [r for r in rows if r.get("json_valid")]
    n_pv        = len(parse_valid)
    legal_all   = sum(1 for r in rows if r.get("legality_result") == "legal")
    legal_pv    = sum(1 for r in parse_valid if r.get("legality_result") == "legal")
    halluc_pv   = sum(1 for r in parse_valid if r.get("hallucinated_path"))
    parse_fail  = sum(
        1 for r in rows
        if r.get("error_class") in ("parse_failed", "output_format_error", "empty_response")
    )
    copy_err    = sum(1 for r in rows if r.get("path_copy_error"))
    return {
        "n":                                  float(n),
        "n_parse_valid":                      float(n_pv),
        "legal_rate_all_rows":                legal_all / n,
        "legal_rate_parse_valid":             (legal_pv / n_pv) if n_pv else float("nan"),
        "hallucinated_path_rate_parse_valid": (halluc_pv / n_pv) if n_pv else float("nan"),
        "parse_failed_rate":                  parse_fail / n,
        "path_copy_error_rate":               copy_err / n,
    }


def _save_category_csv(
    records: list[dict[str, Any]],
    baselines: list[str],
    path: Path,
) -> None:
    """Write a per-(category × baseline) metrics CSV."""
    cat_metrics = aggregate_category_metrics(records, baselines)
    fields = [
        "category", "baseline", "n",
        "legal_rate", "top1_rate", "top3_rate",
        "avg_rank", "avg_score_gap",
        "strategic_error_rate", "hallucinated_path_rate",
        "reasoning_hallucination_rate",
        "output_format_error_rate", "api_error_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for cat, by_bl in cat_metrics.items():
            for bl, m in by_bl.items():
                w.writerow({"category": cat, "baseline": bl, **m})
    print(f"[category] {path}")


def _fmt_gap(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v == float("inf"):
        return "∞"
    return f"{v:.1f}"


def _save_markdown(records: list[dict], baselines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Scenario Suite Results",
        "",
        "Error-class legend: **clean** · api_call_failed · empty_response · "
        "output_format_error · parse_failed · **hallucinated_path** · "
        "reasoning_hallucination · strategic_error · no_legal_moves",
        "",
    ]

    # ── Per-scenario detail tables ────────────────────────────────────────────
    # Group by scenario → baseline → list[record] (multiple repeats)
    by_sc_bl: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        by_sc_bl.setdefault(r["scenario"], {}).setdefault(r["baseline"], []).append(r)

    def cell_first(recs: list[dict], key: str) -> str:
        """Show the first repeat's value for display."""
        if not recs:
            return "—"
        v = recs[0].get(key)
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "✓" if v else "—"
        if isinstance(v, float):
            return f"{v:.1f}" if v != float("inf") else "∞"
        return str(v)

    rows_def = [
        ("Legal moves",           "legal_moves_count"),
        ("Legality",              "legality_result"),
        ("JSON valid",            "json_valid"),
        ("Hallucinated path",     "hallucinated_path"),
        ("Halluc subtype",        "hallucinated_path_subtype"),
        ("Move legality err",     "move_legality_error"),
        ("Rank",                  "chosen_minimax_rank"),
        ("Score gap",             "score_gap"),
        ("Top-1",                 "top1_hit"),
        ("Top-3",                 "top3_hit"),
        ("Strategic error",       "strategic_error"),
        ("Reasoning applicable",  "reasoning_check_applicable"),
        ("Reasoning checks",      "reasoning_hallucination_count"),
        ("Reasoning passed",      "reasoning_truthfulness_passed"),
        ("Error class",           "error_class"),
        ("Contradiction",         "contradiction_details"),
    ]

    # Iterate over every scenario actually present in records (covers both
    # hand-authored and generated scenarios). Falls back to ALL_SCENARIOS order
    # for the hand-authored subset to preserve historical ordering.
    _seen_order: list[str] = []
    for r in records:
        s = r.get("scenario")
        if s and s not in _seen_order:
            _seen_order.append(s)
    _ordered_scenarios = [s for s in ALL_SCENARIOS if s in _seen_order] + \
                         [s for s in _seen_order if s not in ALL_SCENARIOS]
    for scenario in _ordered_scenarios:
        if scenario not in by_sc_bl:
            continue
        desc = (_SCENARIOS.get(scenario) or {}).get("description", "")
        lines += [f"## {scenario}", "", f"_{desc}_", ""]

        header = "| Metric |" + "".join(f" {bl} |" for bl in baselines)
        sep    = "|-" + "-|".join(["-------"] * (1 + len(baselines))) + "-|"
        lines += [header, sep]

        for label, key in rows_def:
            row = f"| {label} |"
            for bl in baselines:
                recs = by_sc_bl[scenario].get(bl, [])
                row += f" {cell_first(recs, key)} |"
            lines.append(row)
        lines.append("")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    lines += ["## Aggregate Summary", ""]
    hdr = ("| Baseline | N | Legal% | Halluc% | Top1% | Top3% |"
           " StratErr | RFail | RChecks | MedianGap | AvgRank |")
    sep = ("|----------|---|--------|---------|-------|-------|"
           "----------|-------|---------|-----------|---------|")
    lines += [hdr, sep]

    for bl in baselines:
        bl_recs = [r for r in records if r["baseline"] == bl]
        n = len(bl_recs)
        if n == 0:
            continue
        legal_n  = sum(1 for r in bl_recs if r.get("legality_result") == "legal")
        halluc_n = sum(1 for r in bl_recs if r.get("hallucinated_path"))
        top1_n   = sum(1 for r in bl_recs if r.get("top1_hit"))
        top3_n   = sum(1 for r in bl_recs if r.get("top3_hit"))
        strat_n  = sum(1 for r in bl_recs if r.get("strategic_error"))
        rfail_n  = sum(1 for r in bl_recs if not r.get("reasoning_truthfulness_passed", True))
        rc_total = sum(r.get("reasoning_hallucination_count", 0) for r in bl_recs)

        gaps = [r["score_gap"] for r in bl_recs
                if r.get("score_gap") is not None and r["score_gap"] != float("inf")]
        median_gap = f"{statistics.median(gaps):.1f}" if gaps else "—"

        ranks = [r["chosen_minimax_rank"] for r in bl_recs
                 if r.get("chosen_minimax_rank", 0) > 0]
        avg_rank = f"{sum(ranks)/len(ranks):.1f}" if ranks else "—"

        lines.append(
            f"| {bl} | {n} | {legal_n/n:.0%} | {halluc_n/n:.0%} | "
            f"{top1_n/n:.0%} | {top3_n/n:.0%} | {strat_n} | {rfail_n} | "
            f"{rc_total} | {median_gap} | {avg_rank} |"
        )
    lines.append("")

    # ── Per-scenario score-gap breakdown (legal moves only) ───────────────────
    lines += ["## Score Gap by Scenario (legal moves only)", ""]
    hdr2 = "| Scenario |" + "".join(f" {bl} |" for bl in baselines)
    sep2 = "|-" + "-|".join(["----------"] * (1 + len(baselines))) + "-|"
    lines += [hdr2, sep2]

    # Iterate over every scenario actually present in records (covers both
    # hand-authored and generated scenarios). Falls back to ALL_SCENARIOS order
    # for the hand-authored subset to preserve historical ordering.
    _seen_order: list[str] = []
    for r in records:
        s = r.get("scenario")
        if s and s not in _seen_order:
            _seen_order.append(s)
    _ordered_scenarios = [s for s in ALL_SCENARIOS if s in _seen_order] + \
                         [s for s in _seen_order if s not in ALL_SCENARIOS]
    for scenario in _ordered_scenarios:
        if scenario not in by_sc_bl:
            continue
        row = f"| {scenario} |"
        for bl in baselines:
            recs = by_sc_bl[scenario].get(bl, [])
            gaps = [r["score_gap"] for r in recs
                    if r.get("score_gap") is not None and r["score_gap"] != float("inf")]
            row += f" {_fmt_gap(sum(gaps)/len(gaps) if gaps else None)} |"
        lines.append(row)

    # Median row
    mrow = "| **median** |"
    for bl in baselines:
        bl_recs = [r for r in records if r["baseline"] == bl]
        gaps = [r["score_gap"] for r in bl_recs
                if r.get("score_gap") is not None and r["score_gap"] != float("inf")]
        mrow += f" {_fmt_gap(statistics.median(gaps) if gaps else None)} |"
    lines.append(mrow)

    # Average rank row
    rrow = "| **avg rank** |"
    for bl in baselines:
        bl_recs = [r for r in records if r["baseline"] == bl]
        ranks = [r["chosen_minimax_rank"] for r in bl_recs
                 if r.get("chosen_minimax_rank", 0) > 0]
        rrow += f" {f'{sum(ranks)/len(ranks):.1f}' if ranks else '—'} |"
    lines.append(rrow)
    lines.append("")

    # ── Repeat aggregation (only when repeats > 1) ────────────────────────────
    # Computes mean ± std per (scenario × baseline) for the key thesis metrics.
    max_repeat = max((r.get("repeat", 1) for r in records), default=1)
    if max_repeat > 1:
        lines += ["## Per-Scenario × Baseline Repeat Aggregation", "",
                  f"_N repeats = {max_repeat}_", ""]

        agg_metrics = [
            ("legal_rate",          lambda recs: [1.0 if r.get("legality_result") == "legal" else 0.0 for r in recs]),
            ("top1_hit",            lambda recs: [1.0 if r.get("top1_hit") else 0.0 for r in recs]),
            ("top3_hit",            lambda recs: [1.0 if r.get("top3_hit") else 0.0 for r in recs]),
            ("halluc_rate",         lambda recs: [1.0 if r.get("hallucinated_path") else 0.0 for r in recs]),
            ("score_gap",           lambda recs: [r["score_gap"] for r in recs
                                                  if r.get("score_gap") is not None and r["score_gap"] != float("inf")]),
            ("reasoning_halluc_n",  lambda recs: [float(r.get("reasoning_hallucination_count", 0)) for r in recs]),
            ("reasoning_truth_pass",lambda recs: [1.0 if r.get("reasoning_truthfulness_passed", True) else 0.0 for r in recs]),
        ]

        def _ms(vals: list[float]) -> str:
            """Format mean ± std; returns '—' for empty."""
            if not vals:
                return "—"
            m = sum(vals) / len(vals)
            if len(vals) == 1:
                return f"{m:.2f}"
            s = statistics.stdev(vals)
            return f"{m:.2f}±{s:.2f}"

        for metric_name, extractor in agg_metrics:
            lines += [f"### {metric_name}", ""]
            col_header = "| Scenario |" + "".join(f" {bl} |" for bl in baselines)
            col_sep    = "|-" + "-|".join(["----------"] * (1 + len(baselines))) + "-|"
            lines += [col_header, col_sep]
            for scenario in _ordered_scenarios:
                if scenario not in by_sc_bl:
                    continue
                row = f"| {scenario} |"
                for bl in baselines:
                    recs = by_sc_bl[scenario].get(bl, [])
                    row += f" {_ms(extractor(recs))} |"
                lines.append(row)
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {path}")


# ── Terminal display ──────────────────────────────────────────────────────────

_ANSI_RED   = "\033[91m"
_ANSI_YLW   = "\033[93m"
_ANSI_GRN   = "\033[92m"
_ANSI_RESET = "\033[0m"


def _col(s: str, color: str) -> str:
    return f"{color}{s}{_ANSI_RESET}"


def _print_result(rec: dict) -> None:
    bl  = rec["baseline"]
    leg = rec.get("legality_result", "?")
    ec  = rec.get("error_class", "?")
    rk  = rec.get("chosen_minimax_rank", 0)
    gap = rec.get("score_gap")
    t1  = rec.get("top1_hit", False)
    t3  = rec.get("top3_hit", False)
    sub = rec.get("hallucinated_path_subtype")
    rca = rec.get("reasoning_check_applicable", True)

    leg_color = _ANSI_GRN if leg == "legal" else _ANSI_RED
    ec_color  = _ANSI_GRN if ec == "clean" else (_ANSI_YLW if ec == "strategic_error" else _ANSI_RED)

    print(f"  {bl:<28} legal={_col(leg, leg_color):<22} "
          f"rank={rk}  gap={gap}  top1={t1}  top3={t3}  "
          f"err={_col(ec, ec_color)}")

    if sub:
        print(f"    halluc_subtype: {sub}")
    if not rca:
        print(f"    {_col('[reasoning_check_applicable=False — no facts from oracle]', _ANSI_YLW)}")
    for h in rec.get("reasoning_hallucinations", []):
        print(f"    {_col('⚠ ' + h, _ANSI_YLW)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic scenario suite: fixed positions × baselines. "
            "No autocorrect. No game state. Single-ply evaluation only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # All 4 non-full_system baselines, all scenarios:
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_scenario_suite

          # Include full_system in the comparison:
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_scenario_suite \\
              --baselines minimal_raw_llm rules_only_llm legal_moves_index_llm \\
                          legal_moves_path_llm full_system

          # Targeted run, 2 repeats, with prompt audit:
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_scenario_suite \\
              --baselines minimal_raw_llm rules_only_llm \\
              --scenarios opening mandatory_capture tactical_trap \\
              --repeats 2 --show-prompts
        """),
    )
    parser.add_argument(
        "--baselines", nargs="+",
        choices=list(_ALL_SUITE_BASELINES),
        default=list(_SUITE_BASELINES),
        metavar="BASELINE",
        help=(
            "Baselines to run (default: the 5 plain-LLM baselines). "
            f"Choices: {', '.join(_ALL_SUITE_BASELINES)}. "
            "Add full_system to include the neuro-symbolic pipeline."
        ),
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=None,
        metavar="SCENARIO",
        help=(
            "Scenarios to run. Defaults to all hand-authored scenarios, or "
            "to the generated set when --scenarios-file is given. Names are "
            "validated against the registry after generated scenarios load."
        ),
    )
    parser.add_argument(
        "--out-dir", type=str, default="logs/scenario_eval",
        help="Output directory (default: logs/scenario_eval).",
    )
    parser.add_argument(
        "--show-prompts", action="store_true",
        help="Print exact system + user prompts before each LLM call.",
    )
    parser.add_argument(
        "--show-boards", action="store_true",
        help="Print each scenario board to the terminal.",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, metavar="N",
        help="Run each (scenario, baseline) pair N times (default: 1).",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="Print available scenarios and exit.",
    )
    parser.add_argument(
        "--scenarios-file", type=str, default=None,
        help=(
            "Path to a JSON file containing generated hard scenarios "
            "(produced by checkers/baseline_eval/generate_hard_scenarios.py). "
            "When provided, those scenarios are appended to the registry and "
            "are runnable by their scenario_id."
        ),
    )
    parser.add_argument(
        "--include-hand-authored", action="store_true",
        help=(
            "When --scenarios-file is provided, also include the built-in "
            "hand-authored scenarios. Default: only run generated scenarios."
        ),
    )
    args = parser.parse_args()

    # Optional: load generated scenarios from JSON file.
    generated_names: list[str] = []
    if args.scenarios_file:
        loaded = load_generated_scenarios(args.scenarios_file)
        generated_names = _register_generated_scenarios(loaded)
        print(f"[scenarios] loaded {len(generated_names)} generated scenarios "
              f"from {args.scenarios_file}")

    if args.list_scenarios:
        print(f"{'Scenario':<22} Description")
        print("─" * 70)
        for name, meta in _SCENARIOS.items():
            d = meta["description"][:60] + ("…" if len(meta["description"]) > 60 else "")
            print(f"  {name:<20} {d}")
        return

    repeats   = max(1, args.repeats)
    baselines = args.baselines
    # Scenario selection:
    #   • If --scenarios is explicitly provided, use that (validated below).
    #   • Else if --scenarios-file given, default to the generated set,
    #     plus hand-authored only when --include-hand-authored is set.
    #   • Else default to all hand-authored scenarios.
    if args.scenarios is not None:
        scenarios = list(args.scenarios)
        unknown = [s for s in scenarios if s not in _SCENARIOS]
        if unknown:
            parser.error(f"unknown scenarios: {unknown}")
    elif generated_names:
        scenarios = list(generated_names)
        if args.include_hand_authored:
            scenarios = list(ALL_SCENARIOS) + scenarios
    else:
        scenarios = list(ALL_SCENARIOS)
    out_dir   = Path(args.out_dir)
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    total_calls = len(baselines) * len(scenarios) * repeats
    print(BAR)
    print("BASELINE SCENARIO SUITE")
    print(f"  run_id    : {run_id}")
    print(f"  baselines : {baselines}")
    print(f"  scenarios : {scenarios}")
    print(f"  repeats   : {repeats}")
    print(f"  total     : {total_calls} LLM calls")
    print(BAR)
    print()

    records: list[dict] = []
    done = 0

    for scenario_name in scenarios:
        meta  = _SCENARIOS[scenario_name]
        # board_fn() returns a fresh board for each scenario display
        board_display = meta["board_fn"]()

        print(RULE)
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"  {meta['description']}")
        print(f"  Legal moves: {len(get_all_legal_moves(board_display, RED))}")
        if args.show_boards:
            print_board(board_display)
        print()

        for baseline in baselines:
            for repeat_idx in range(repeats):
                # Fresh board per (baseline × repeat) — fixes mutable singleton risk
                board = meta["board_fn"]()
                done += 1
                rep_label = f" (rep {repeat_idx+1}/{repeats})" if repeats > 1 else ""
                print(f"  [{done}/{total_calls}] {baseline}{rep_label} …", end="", flush=True)
                rec = run_scenario_for_baseline(
                    scenario_name, board, baseline, show_prompts=args.show_prompts,
                )
                rec["repeat"] = repeat_idx + 1
                records.append(rec)
                print()
                _print_result(rec)
        print()

    print(BAR)
    print(f"Done — {len(records)} records collected.")
    print()

    stem = out_dir / f"scenario_suite_{run_id}"
    _save_jsonl(records, stem.with_suffix(".jsonl"))
    _save_csv  (records, Path(str(stem) + "_metrics.csv"))
    _save_category_csv(records, baselines, Path(str(stem) + "_category_metrics.csv"))
    _save_markdown(records, baselines, Path(str(stem) + "_summary.md"))
    print()


if __name__ == "__main__":
    main()
