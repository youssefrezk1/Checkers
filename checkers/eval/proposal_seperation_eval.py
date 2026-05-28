# checkers/eval/proposal_seperation_eval.py
#
# PURE RAW LLM BASELINE — Separated Proposal Evaluation
#
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE
# ══════════════════════════════════════════════════════════════════════════════
# Evaluates RAW LLM OUTPUT against GROUND TRUTH LEGAL MOVES.
#
# Scanner correctness is evaluated SEPARATELY from proposal correctness.
# Even if the scanner predicts incorrectly, the evaluator routes using the
# CORRECT GROUND TRUTH branch to separately evaluate proposal quality.
# This prevents scanner errors from contaminating proposal evaluation.
#
# This module must NEVER:
#   - repair malformed proposals
#   - reconstruct intended moves
#   - infer missing continuations
#   - inflate benchmark results
#
# ══════════════════════════════════════════════════════════════════════════════
# USAGE
# ══════════════════════════════════════════════════════════════════════════════
#   # Full dataset (default):
#   python3 -m checkers.eval.proposal_seperation_eval
#
#   # With mode filter:
#   python3 -m checkers.eval.proposal_seperation_eval --mode hard
#   python3 -m checkers.eval.proposal_seperation_eval --mode jump_only --limit 50
#
#   # Custom dataset:
#   python3 -m checkers.eval.proposal_seperation_eval \
#       --dataset checkers/data/legality_stress/eval_subset_balanced.jsonl
#

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv, find_dotenv
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path, override=True)
except ImportError:
    pass

from checkers.engine.board import RED, BLACK, BOARD_SIZE
from checkers.engine.rules import get_all_legal_moves
from checkers.agents.proposal_seperation import (
    run_proposal_seperation,
    run_proposer_only,
    run_strategic_selection,
    parse_best_move_output,
    SCANNER_MODEL,
    PROPOSAL_MODEL,
)
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.deterministic_proposal import select_best_move

logger = logging.getLogger(__name__)

# ── Default dataset path ─────────────────────────────────────────────────────
_DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "data", "legality_stress", "scenarios.jsonl"
)

# ── Default best-move annotations path (optional; eval degrades gracefully) ──
_DEFAULT_ANNOTATIONS = os.path.join(
    os.path.dirname(__file__), "..", "data", "legality_stress",
    "scenarios_bestmove_annotations.json"
)


# ══════════════════════════════════════════════════════════════════════════════
# BEST-MOVE ANNOTATION LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_bestmove_annotations(path: str | None) -> dict[str, dict]:
    """
    Load optional scenario_id → annotation mapping from a JSON file.

    Returns an empty dict (not an error) when:
      - path is None
      - file does not exist
      - file is malformed

    The evaluator degrades gracefully: coverage fields are None when the
    annotation is absent, and all other metrics are unaffected.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        logger.warning(
            "[annotations] file not found: %s — coverage metrics will be skipped", path
        )
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("[annotations] expected JSON array, got %s — skipping", type(data))
            return {}
        mapping: dict[str, dict] = {}
        for entry in data:
            sid = entry.get("scenario_id")
            if sid:
                mapping[sid] = entry
        logger.info("[annotations] loaded %d entries from %s", len(mapping), path)
        return mapping
    except Exception as exc:
        logger.warning("[annotations] could not load %s: %s — skipping", path, exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# BEST-MOVE COVERAGE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _path_in_proposals(
    target_path: list[list[int]] | None,
    proposed_moves: list[dict] | None,
) -> bool | None:
    """
    Check whether *target_path* appears as the path of any move in
    *proposed_moves*.  Returns:

      True  — at least one proposed move has the matching path
      False — no proposed move matches
      None  — target_path is unknown (annotation absent) or proposed_moves is None

    Comparison is path-only (type and captured are ignored), normalised to
    list-of-[int,int] so minor representation differences do not cause mismatches.
    This is purely a membership test; it does NOT repair or infer moves.
    """
    if target_path is None:
        return None
    if not proposed_moves:
        return False
    # Normalise target
    try:
        t_norm = [[int(sq[0]), int(sq[1])] for sq in target_path]
    except (TypeError, IndexError, ValueError):
        return None
    # Check each proposed move
    for m in proposed_moves:
        try:
            p_norm = [[int(sq[0]), int(sq[1])] for sq in m.get("path", [])]
        except (TypeError, IndexError, ValueError, AttributeError):
            continue
        if p_norm == t_norm:
            return True
    return False


def _is_top1_match(
    target_path: list[list[int]] | None,
    proposed_moves: list[dict] | None,
) -> bool | None:
    """
    Check whether the top-1 (first) proposed move matches target_path.
    Returns:
      True  — the first proposed move matches target_path
      False — the first proposed move does not match target_path
      None  — target_path is unknown or proposed_moves is empty/None
    """
    if target_path is None:
        return None
    if not proposed_moves:
        return False
    # Check only the first proposed move
    m = proposed_moves[0]
    try:
        p_norm = [[int(sq[0]), int(sq[1])] for sq in m.get("path", [])]
    except (TypeError, IndexError, ValueError, AttributeError):
        return False
    try:
        t_norm = [[int(sq[0]), int(sq[1])] for sq in target_path]
    except (TypeError, IndexError, ValueError):
        return None
    return p_norm == t_norm



# ══════════════════════════════════════════════════════════════════════════════
# MOVE NORMALIZATION — minimal, no repair
# ══════════════════════════════════════════════════════════════════════════════

def _norm_path(path: list) -> list[list[int]]:
    """Normalize a path to list of [int, int]. No repair."""
    return [list(map(int, x)) for x in path]


def _norm_captured(captured: list) -> list[list[int]]:
    """Normalize captured to list of [int, int]. No repair."""
    return [list(map(int, x)) for x in captured]


def _norm_move(move: dict) -> dict:
    """Normalize a move dict to canonical form for comparison."""
    return {
        "type": move["type"],
        "path": _norm_path(move["path"]),
        "captured": _norm_captured(move.get("captured", [])),
    }


def _move_key(move: dict) -> str:
    """Deterministic string key for a normalized move (for set operations)."""
    nm = _norm_move(move)
    return json.dumps(nm, sort_keys=True)


# ══════════════════════════════════════════════════════════════════════════════
# FAILURE TAXONOMY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _is_out_of_bounds(coord: list[int]) -> bool:
    """Check if a coordinate is outside the 8×8 board."""
    return coord[0] < 0 or coord[0] >= BOARD_SIZE or coord[1] < 0 or coord[1] >= BOARD_SIZE


def _check_duplicate_moves(moves: list[dict]) -> int:
    """Count duplicate moves in the proposal."""
    keys = [_move_key(m) for m in moves]
    return len(keys) - len(set(keys))


def _check_partial_jumps(
    proposed_moves: list[dict],
    legal_moves: list[dict],
) -> int:
    """
    Count partial jump sequences.

    A partial jump: proposed path is a strict prefix of a legal path.
    Example: proposed [[5,0],[3,2]] when legal is [[5,0],[3,2],[1,4]].
    """
    count = 0
    legal_jump_paths = [
        _norm_path(m["path"]) for m in legal_moves if m["type"] == "jump"
    ]

    for move in proposed_moves:
        if move["type"] != "jump":
            continue
        prop_path = _norm_path(move["path"])
        for legal_path in legal_jump_paths:
            if (len(prop_path) < len(legal_path)
                    and legal_path[:len(prop_path)] == prop_path):
                count += 1
                break

    return count


def _check_illegal_geometry(proposed_moves: list[dict]) -> int:
    """
    Count moves with illegal geometry:
    - Path entries not on dark squares (row+col must be odd)
    - Jump legs not exactly 2 diagonal steps apart
    - Simple moves not exactly 1 diagonal step apart
    """
    count = 0
    for move in proposed_moves:
        path = _norm_path(move["path"])
        is_illegal = False

        # Check dark square constraint
        for coord in path:
            if not _is_out_of_bounds(coord) and (coord[0] + coord[1]) % 2 == 0:
                is_illegal = True
                break

        # Check step distances
        if not is_illegal:
            for i in range(len(path) - 1):
                dr = abs(path[i+1][0] - path[i][0])
                dc = abs(path[i+1][1] - path[i][1])
                if move["type"] == "jump":
                    if dr != 2 or dc != 2:
                        is_illegal = True
                        break
                elif move["type"] == "simple":
                    if dr != 1 or dc != 1:
                        is_illegal = True
                        break

        if is_illegal:
            count += 1

    return count


def _check_out_of_bounds(proposed_moves: list[dict]) -> int:
    """Count moves with any out-of-bounds coordinates."""
    count = 0
    for move in proposed_moves:
        path = _norm_path(move["path"])
        captured = _norm_captured(move.get("captured", []))
        all_coords = path + captured
        if any(_is_out_of_bounds(c) for c in all_coords):
            count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
# PROPOSAL CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_proposal(
    proposed_moves: list[dict],
    legal_moves: list[dict],
) -> dict[str, Any]:
    """
    Classify a proposal against ground truth legal moves.

    Returns:
      classification: str — one of:
        "perfect"               — all legal moves generated, no illegal
        "legal_but_incomplete"  — all generated are legal, some missing
        "proposal_illegal"      — at least one generated move is illegal
        "proposal_fully_illegal"— all generated moves are illegal
        "empty_proposal"        — no moves generated

      legal_count:    int — total ground truth legal moves
      proposed_count: int — total proposed moves
      legal_proposed: int — proposed moves that are legal
      illegal_proposed: int — proposed moves that are illegal
      missing_legal:  int — legal moves not proposed
    """
    if not proposed_moves:
        return {
            "classification": "empty_proposal",
            "legal_count": len(legal_moves),
            "proposed_count": 0,
            "legal_proposed": 0,
            "illegal_proposed": 0,
            "missing_legal": len(legal_moves),
        }

    legal_keys = {_move_key(m) for m in legal_moves}

    legal_proposed = 0
    illegal_proposed = 0

    for pm in proposed_moves:
        if _move_key(pm) in legal_keys:
            legal_proposed += 1
        else:
            illegal_proposed += 1

    proposed_keys = {_move_key(m) for m in proposed_moves}
    missing_legal = len(legal_keys - proposed_keys)

    if illegal_proposed == 0 and missing_legal == 0:
        classification = "perfect"
    elif illegal_proposed == 0 and missing_legal > 0:
        classification = "legal_but_incomplete"
    elif legal_proposed == 0:
        classification = "proposal_fully_illegal"
    else:
        classification = "proposal_illegal"

    return {
        "classification": classification,
        "legal_count": len(legal_moves),
        "proposed_count": len(proposed_moves),
        "legal_proposed": legal_proposed,
        "illegal_proposed": illegal_proposed,
        "missing_legal": missing_legal,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_scanner(
    scanner_prediction: bool | None,
    ground_truth_has_captures: bool,
) -> dict[str, Any]:
    """
    Evaluate scanner correctness independently.

    Returns:
      scanner_correct: bool
      scanner_prediction: True/False/None
      ground_truth_has_captures: bool
    """
    if scanner_prediction is None:
        return {
            "scanner_correct": False,
            "scanner_prediction": None,
            "ground_truth_has_captures": ground_truth_has_captures,
        }

    return {
        "scanner_correct": scanner_prediction == ground_truth_has_captures,
        "scanner_prediction": scanner_prediction,
        "ground_truth_has_captures": ground_truth_has_captures,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PER-POSITION EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_position(
    board: list[list[int]],
    current_player: int,
    scenario_id: str = "",
    category: str = "",
    difficulty: str = "",
    bestmove_annotations: dict[str, dict] | None = None,
    single_best: bool = False,
) -> dict[str, Any]:
    """
    Run the full pipeline and evaluate one position.

    Uses GROUND TRUTH routing: even if the scanner is wrong, the proposal
    is evaluated against the correct branch (jumps or simples).

    bestmove_annotations: optional mapping scenario_id -> annotation dict
    loaded by load_bestmove_annotations().  When present, each result is
    enriched with engine_best_move / kingsrow_best_move / contains_* fields.
    Absent or missing entry -> those fields are None (graceful degradation).
    """
    t0 = time.perf_counter()

    # ── Best-move annotation lookup (evaluation reporting only) ───────────
    # Engine best: computed live using the same scorer chain as
    # benchmark_evaluator.py — no external dependencies.
    # KR best: looked up from pre-computed annotation file.
    _ann = (bestmove_annotations or {}).get(scenario_id, {})

    engine_best_path: list | None = None
    try:
        _enriched, _, _, _ = score_all_legal_moves(board, current_player)
        if _enriched:
            _chosen, _, _, _ = select_best_move(_enriched)
            engine_best_path = [[int(sq[0]), int(sq[1])] for sq in _chosen.get("path", [])]
    except Exception as _exc:
        logger.debug("[coverage] engine best-move error for %s: %s", scenario_id, _exc)

    kr_best_path: list | None = None
    if _ann.get("kr_path"):
        try:
            kr_best_path = [[int(sq[0]), int(sq[1])] for sq in _ann["kr_path"]]
        except Exception:
            kr_best_path = None

    # ── Ground truth ──────────────────────────────────────────────────────
    legal_moves = get_all_legal_moves(board, current_player)
    ground_truth_has_captures = any(m["type"] == "jump" for m in legal_moves)

    # Normalize legal moves for comparison
    legal_norm = [_norm_move(m) for m in legal_moves]

    # ── Run separated proposal pipeline ──────────────────────────────────
    pipeline_result = run_proposal_seperation(board, current_player, single_best=single_best)
    elapsed = round(time.perf_counter() - t0, 3)

    # ── Handle API failures ──────────────────────────────────────────────
    if pipeline_result["api_failure"]:
        return {
            "scenario_id": scenario_id,
            "category": category,
            "difficulty": difficulty,
            "side_to_move": "RED" if current_player == RED else "BLACK",
            "legal_move_count": len(legal_moves),
            "ground_truth_has_captures": ground_truth_has_captures,
            "api_failure": True,
            "scanner_api_ok": pipeline_result["scanner_api_ok"],
            "proposal_api_ok": pipeline_result["proposal_api_ok"],
            "elapsed_s": elapsed,
            "quadrant": "api_failure",
            "classification": "api_failure",
            # Coverage — no proposals exist on API failure
            "engine_best_move": engine_best_path,
            "kingsrow_best_move": kr_best_path,
            "contains_engine_best": None,
            "contains_kingsrow_best": None,
            "top1_engine_match": None,
            "top1_kingsrow_match": None,
        }

    # ── Scanner evaluation (independent) ─────────────────────────────────
    scanner_eval = evaluate_scanner(
        scanner_prediction=pipeline_result["scanner_prediction"],
        ground_truth_has_captures=ground_truth_has_captures,
    )

    # ── Handle parse failures ────────────────────────────────────────────
    if pipeline_result["scanner_parse_failure"]:
        return {
            "scenario_id": scenario_id,
            "category": category,
            "difficulty": difficulty,
            "side_to_move": "RED" if current_player == RED else "BLACK",
            "legal_move_count": len(legal_moves),
            "ground_truth_has_captures": ground_truth_has_captures,
            "api_failure": False,
            "scanner_eval": scanner_eval,
            "scanner_parse_failure": True,
            "proposal_parse_failure": False,
            "parse_failure": True,
            "elapsed_s": elapsed,
            "quadrant": "scanner_wrong_proposal_wrong",
            "classification": "parse_failure",
            "scanner_raw": pipeline_result["scanner_raw"][:500],
            # Coverage — parse failure means no usable proposals
            "engine_best_move": engine_best_path,
            "kingsrow_best_move": kr_best_path,
            "contains_engine_best": None,
            "contains_kingsrow_best": None,
            "top1_engine_match": None,
            "top1_kingsrow_match": None,
        }

    # ── Branch mismatch detection ────────────────────────────────────────
    # branch_mismatch = True when the scanner routed to the WRONG proposer.
    # When this happens the scanner is still marked wrong, but we make a
    # second proposer call using the correct (ground-truth) branch so that
    # the proposal is evaluated on the right task.
    #
    # scanner_routed_proposal_raw: the original wrong-branch output (auditable).
    # The evaluation uses the ground-truth-routed call only.
    gt_branch  = "jump"  if ground_truth_has_captures else "quiet"
    act_branch = pipeline_result["proposal_branch"]
    branch_mismatch = (act_branch != gt_branch)

    # Preserve the scanner-routed output for the audit log
    scanner_routed_proposal_raw = pipeline_result["proposal_raw"]

    if branch_mismatch:
        # Scanner routed wrong — call the correct branch for fair evaluation
        logger.info(
            "[branch_mismatch] scanner=%s gt=%s — re-calling %s proposer",
            act_branch, gt_branch, gt_branch,
        )
        gt_result = run_proposer_only(board, current_player, gt_branch, single_best=single_best)

        if gt_result["api_failure"]:
            # Ground-truth proposer call failed — fall back to api_failure result
            return {
                "scenario_id": scenario_id,
                "category": category,
                "difficulty": difficulty,
                "side_to_move": "RED" if current_player == RED else "BLACK",
                "legal_move_count": len(legal_moves),
                "ground_truth_has_captures": ground_truth_has_captures,
                "api_failure": True,
                "branch_mismatch": True,
                "scanner_eval": scanner_eval,
                "scanner_api_ok": pipeline_result["scanner_api_ok"],
                "proposal_api_ok": False,
                "scanner_routed_proposal_raw": scanner_routed_proposal_raw[:500],
                "elapsed_s": elapsed,
                "quadrant": "api_failure",
                "classification": "api_failure",
                # Coverage — no proposals
                "engine_best_move": engine_best_path,
                "kingsrow_best_move": kr_best_path,
                "contains_engine_best": None,
                "contains_kingsrow_best": None,
                "top1_engine_match": None,
                "top1_kingsrow_match": None,
            }

        # Use gt_result for proposal evaluation
        proposal_source       = gt_result
        eval_proposal_branch  = gt_branch
    else:
        # Scanner was correct — use the original pipeline result
        proposal_source      = pipeline_result
        eval_proposal_branch = act_branch

    # ── Proposal evaluation ──────────────────────────────────────────────
    # Always compared against the engine ground truth.
    # When branch_mismatch=True, proposal_source is the gt-routed call.
    proposed_moves = proposal_source["proposal_moves"]

    if single_best:
        parsed_moves = proposed_moves
        print(len(parsed_moves) if parsed_moves else 0, parsed_moves)

    # Strict single_best validation
    is_violation = False
    original_len = proposal_source.get("original_proposal_moves_len", len(proposed_moves) if proposed_moves is not None else 0)
    if single_best and proposed_moves is not None and original_len != 1:
        is_violation = True

    if proposal_source["proposal_parse_failure"] or proposed_moves is None:
        proposal_class = {
            "classification": "parse_failure",
            "legal_count": len(legal_moves),
            "proposed_count": 0,
            "legal_proposed": 0,
            "illegal_proposed": 0,
            "missing_legal": len(legal_moves),
        }
        failure_taxonomy = {
            "duplicate_moves_generated": 0,
            "partial_jump_sequences": 0,
            "illegal_geometry_moves": 0,
            "out_of_bounds_coordinates": 0,
            "missing_legal_moves": len(legal_moves),
            "parse_failures": 1,
            "wrong_branch_called": 1 if branch_mismatch else 0,
            "api_failures": 0,
        }
    elif is_violation:
        proposal_class = {
            "classification": "single_best_violation",
            "legal_count": len(legal_moves),
            "proposed_count": original_len,
            "legal_proposed": sum(1 for m in proposed_moves if _norm_move(m) in legal_norm),
            "illegal_proposed": sum(1 for m in proposed_moves if _norm_move(m) not in legal_norm),
            "missing_legal": len(legal_moves),
        }
        failure_taxonomy = {
            "duplicate_moves_generated": _check_duplicate_moves(proposed_moves),
            "partial_jump_sequences": _check_partial_jumps(proposed_moves, legal_norm),
            "illegal_geometry_moves": _check_illegal_geometry(proposed_moves),
            "out_of_bounds_coordinates": _check_out_of_bounds(proposed_moves),
            "missing_legal_moves": len(legal_moves),
            "parse_failures": 0,
            "wrong_branch_called": 1 if branch_mismatch else 0,
            "api_failures": 0,
        }
    else:
        proposal_class = classify_proposal(proposed_moves, legal_norm)
        failure_taxonomy = {
            "duplicate_moves_generated": _check_duplicate_moves(proposed_moves),
            "partial_jump_sequences": _check_partial_jumps(proposed_moves, legal_norm),
            "illegal_geometry_moves": _check_illegal_geometry(proposed_moves),
            "out_of_bounds_coordinates": _check_out_of_bounds(proposed_moves),
            "missing_legal_moves": proposal_class["missing_legal"],
            "parse_failures": 0,
            "wrong_branch_called": 1 if branch_mismatch else 0,
            "api_failures": 0,
        }

    # ── Determine quadrant ───────────────────────────────────────────────
    scanner_correct  = scanner_eval["scanner_correct"]
    proposal_correct = (proposal_class["classification"] == "perfect") and not is_violation

    if scanner_correct and proposal_correct:
        quadrant = "scanner_correct_proposal_correct"
    elif scanner_correct and not proposal_correct:
        quadrant = "scanner_correct_proposal_wrong"
    elif not scanner_correct and proposal_correct:
        quadrant = "scanner_wrong_proposal_correct"
    else:
        quadrant = "scanner_wrong_proposal_wrong"

    return {
        "scenario_id": scenario_id,
        "category": category,
        "difficulty": difficulty,
        "side_to_move": "RED" if current_player == RED else "BLACK",
        "legal_move_count": len(legal_moves),
        "ground_truth_has_captures": ground_truth_has_captures,
        # Scanner
        "scanner_eval": scanner_eval,
        "scanner_prediction": pipeline_result["scanner_prediction"],
        "proposal_branch": eval_proposal_branch,
        "ground_truth_branch": gt_branch,
        "branch_mismatch": branch_mismatch,
        # Proposal (always from the correctly-routed call)
        "proposal_classification": proposal_class,
        "proposal_raw": proposal_source["proposal_raw"][:2000],
        # Original scanner-routed proposal raw (audit only, present when mismatch)
        "scanner_routed_proposal_raw": scanner_routed_proposal_raw[:500] if branch_mismatch else "",
        # Quadrant
        "quadrant": quadrant,
        "classification": proposal_class["classification"],
        # Failure taxonomy
        "failure_taxonomy": failure_taxonomy,
        # API
        "api_failure": False,
        "scanner_api_ok": pipeline_result["scanner_api_ok"],
        "proposal_api_ok": proposal_source["proposal_api_ok"],
        "parse_failure": proposal_source["proposal_parse_failure"],
        "scanner_parse_failure": pipeline_result["scanner_parse_failure"],
        "proposal_parse_failure": proposal_source["proposal_parse_failure"],
        # Raw scanner output
        "scanner_raw": pipeline_result["scanner_raw"][:500],
        # Timing
        "elapsed_s": elapsed,
        # Best-move coverage (evaluation reporting only — does not affect proposal)
        "engine_best_move": engine_best_path,
        "kingsrow_best_move": kr_best_path,
        "contains_engine_best": _path_in_proposals(engine_best_path, proposed_moves),
        "contains_kingsrow_best": _path_in_proposals(kr_best_path, proposed_moves),
        "top1_engine_match": _is_top1_match(engine_best_path, proposed_moves),
        "top1_kingsrow_match": _is_top1_match(kr_best_path, proposed_moves),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _player_from_str(s: str) -> int:
    """Convert side string to player constant."""
    if s.upper() == "RED":
        return RED
    return BLACK


def load_dataset(path: str) -> list[dict[str, Any]]:
    """
    Load a JSONL dataset file. Each line must have:
      board, side_to_move, scenario_id
    Optional: hidden_legal_moves, category, difficulty
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d: JSON parse error: %s", line_num, e)
                continue

            if "board" not in entry or "side_to_move" not in entry:
                logger.warning("Skipping line %d: missing board or side_to_move", line_num)
                continue

            entries.append(entry)

    return entries


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODES — filter from existing dataset properties
# ══════════════════════════════════════════════════════════════════════════════
# All modes filter scenarios from the existing dataset ONLY.
# No synthetic data is created.

# Categories that indicate jump-related complexity
_JUMP_CATEGORIES = {"multi_jump_required", "mandatory_capture"}

# Categories that indicate quiet/positional complexity
_QUIET_CATEGORIES = {
    "king_vs_man_confusion", "wrong_direction_trap",
    "crowded_board", "promotion_state_update",
    "wrong_player_piece_trap", "occupied_destination_trap",
}


def _has_captures(entry: dict) -> bool:
    """Check if a scenario has any capture moves in its hidden_legal_moves."""
    moves = entry.get("hidden_legal_moves", [])
    return any(m.get("type") == "jump" for m in moves)


def _is_multi_jump(entry: dict) -> bool:
    """Check if a scenario has any multi-jump sequences (path length > 2)."""
    moves = entry.get("hidden_legal_moves", [])
    return any(
        m.get("type") == "jump" and len(m.get("path", [])) > 2
        for m in moves
    )


def _has_kings(entry: dict) -> bool:
    """Check if a scenario has any king pieces on the board."""
    board = entry.get("board", [])
    for row in board:
        for cell in row:
            if cell in (3, 4):  # RED_KING=3, BLACK_KING=4
                return True
    return False


def _piece_count(entry: dict) -> int:
    """Count total pieces on the board."""
    board = entry.get("board", [])
    count = 0
    for row in board:
        for cell in row:
            if cell != 0:
                count += 1
    return count


def _legal_move_count(entry: dict) -> int:
    """Count legal moves from hidden_legal_moves."""
    return len(entry.get("hidden_legal_moves", []))


def filter_by_mode(
    dataset: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """
    Filter dataset by evaluation mode. All modes filter from existing
    scenario properties — no synthetic data.

    Modes:
      full       — all scenarios (no filter)
      balanced   — equal sample from easy/medium/hard difficulty
      easy       — difficulty == "easy"
      medium     — difficulty == "medium"
      hard       — difficulty == "hard"
      jump_only  — scenarios where ground truth has captures
      quiet_only — scenarios where ground truth has no captures
    """
    if mode == "full":
        return dataset

    if mode == "easy":
        return [e for e in dataset if e.get("difficulty") == "easy"]

    if mode == "medium":
        return [e for e in dataset if e.get("difficulty") == "medium"]

    if mode == "hard":
        return [e for e in dataset if e.get("difficulty") == "hard"]

    if mode == "medium_hard":
        return [e for e in dataset if e.get("difficulty") in ("medium", "hard")]

    if mode == "jump_only":
        return [e for e in dataset if _has_captures(e)]

    if mode == "quiet_only":
        return [e for e in dataset if not _has_captures(e)]

    if mode == "balanced":
        # Equal sample from each difficulty level, capped at min count
        by_diff: dict[str, list[dict]] = {}
        for e in dataset:
            d = e.get("difficulty", "unknown")
            by_diff.setdefault(d, []).append(e)
        # Use the smallest group size to balance
        known_diffs = ["easy", "medium", "hard"]
        available = {d: by_diff.get(d, []) for d in known_diffs if d in by_diff}
        if not available:
            return dataset
        min_count = min(len(v) for v in available.values())
        balanced: list[dict] = []
        for d in known_diffs:
            if d in available:
                balanced.extend(available[d][:min_count])
        return balanced

    # Unknown mode — return full dataset with warning
    logger.warning("Unknown mode %r — using full dataset", mode)
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate evaluation results into a summary report.
    """
    total = len(results)
    if total == 0:
        return {"total": 0}

    single_best_violations = sum(
        1 for r in results
        if r.get("classification") == "single_best_violation"
    )

    # Quadrant counts
    quadrants = {
        "scanner_correct_proposal_correct": 0,
        "scanner_correct_proposal_wrong": 0,
        "scanner_wrong_proposal_correct": 0,
        "scanner_wrong_proposal_wrong": 0,
        "api_failure": 0,
    }
    for r in results:
        q = r.get("quadrant", "api_failure")
        if q in quadrants:
            quadrants[q] += 1
        else:
            quadrants["api_failure"] += 1

    # Classification counts
    classifications = {}
    for r in results:
        c = r.get("classification", "unknown")
        classifications[c] = classifications.get(c, 0) + 1

    # Failure taxonomy totals
    taxonomy_totals = {
        "duplicate_moves_generated": 0,
        "partial_jump_sequences": 0,
        "illegal_geometry_moves": 0,
        "out_of_bounds_coordinates": 0,
        "missing_legal_moves": 0,
        "parse_failures": 0,
        "wrong_branch_called": 0,
        "api_failures": 0,
    }
    for r in results:
        ft = r.get("failure_taxonomy", {})
        for key in taxonomy_totals:
            taxonomy_totals[key] += ft.get(key, 0)
    # Count API failures from results level too
    taxonomy_totals["api_failures"] = quadrants.get("api_failure", 0)

    # Average missing legal moves per evaluated position
    # (across all non-api-failure positions, including parse failures)
    evaluated_count = sum(1 for r in results if not r.get("api_failure"))
    taxonomy_totals["missing_legal_moves_avg"] = round(
        taxonomy_totals["missing_legal_moves"] / evaluated_count, 2
    ) if evaluated_count else 0.0

    # Scanner accuracy
    scanner_results = [r for r in results if not r.get("api_failure")]
    scanner_total = len(scanner_results)
    scanner_correct = sum(
        1 for r in scanner_results
        if r.get("scanner_eval", {}).get("scanner_correct", False)
    )

    # Branch mismatch stats
    # wrong_branch_called: scanner routed the wrong proposer (cascading failure)
    branch_mismatch_count = sum(
        1 for r in results if r.get("branch_mismatch", False)
    )
    # Proposal accuracy split:
    #   isolated — scanner was correct (right proposer was called)
    #   cascading — scanner was wrong (wrong proposer was called)
    isolated_proposal_results = [
        r for r in results
        if not r.get("api_failure")
        and not r.get("parse_failure")
        and not r.get("branch_mismatch", False)
    ]
    isolated_perfect = sum(
        1 for r in isolated_proposal_results
        if r.get("classification") == "perfect"
    )

    # Proposal accuracy (all, excluding api/parse failures) — original metric
    proposal_results = [
        r for r in results
        if not r.get("api_failure") and not r.get("parse_failure")
    ]
    proposal_total = len(proposal_results)
    proposal_perfect = sum(
        1 for r in proposal_results
        if r.get("classification") == "perfect"
    )

    # Category breakdown
    category_stats: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "perfect": 0, "scanner_correct": 0, "branch_mismatch": 0}
        category_stats[cat]["total"] += 1
        if r.get("classification") == "perfect":
            category_stats[cat]["perfect"] += 1
        if r.get("scanner_eval", {}).get("scanner_correct", False):
            category_stats[cat]["scanner_correct"] += 1
        if r.get("branch_mismatch", False):
            category_stats[cat]["branch_mismatch"] += 1

    # Difficulty breakdown
    difficulty_stats: dict[str, dict[str, int]] = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "perfect": 0}
        difficulty_stats[diff]["total"] += 1
        if r.get("classification") == "perfect":
            difficulty_stats[diff]["perfect"] += 1

    # Calculate Top-1 match percentages
    eng_top1_eval = [r for r in results if isinstance(r.get("top1_engine_match"), bool)]
    eng_top1_denom = len(eng_top1_eval)
    eng_top1_num = sum(1 for r in eng_top1_eval if r.get("top1_engine_match") is True)
    
    kr_top1_eval = [r for r in results if isinstance(r.get("top1_kingsrow_match"), bool)]
    kr_top1_denom = len(kr_top1_eval)
    kr_top1_num = sum(1 for r in kr_top1_eval if r.get("top1_kingsrow_match") is True)

    top1_engine_match_pct = round(100 * eng_top1_num / eng_top1_denom, 1) if eng_top1_denom > 0 else 0.0
    top1_kingsrow_match_pct = round(100 * kr_top1_num / kr_top1_denom, 1) if kr_top1_denom > 0 else 0.0


    return {
        "total_positions": total,
        "quadrants": quadrants,
        "classifications": classifications,
        "failure_taxonomy": taxonomy_totals,
        "scanner_accuracy": {
            "total": scanner_total,
            "correct": scanner_correct,
            "accuracy_pct": round(100 * scanner_correct / scanner_total, 1) if scanner_total else 0,
        },
        "branch_routing": {
            "total": scanner_total,
            "branch_mismatch": branch_mismatch_count,
            "branch_correct": scanner_total - branch_mismatch_count,
            "mismatch_pct": round(100 * branch_mismatch_count / scanner_total, 1) if scanner_total else 0,
        },
        "proposal_accuracy": {
            "total": proposal_total,
            "perfect": proposal_perfect,
            "perfect_pct": round(100 * proposal_perfect / proposal_total, 1) if proposal_total else 0,
        },
        "proposal_accuracy_isolated": {
            # Proposal quality when the RIGHT proposer was called (scanner correct)
            # This is the true measurement of proposer ability.
            "total": len(isolated_proposal_results),
            "perfect": isolated_perfect,
            "perfect_pct": round(100 * isolated_perfect / len(isolated_proposal_results), 1)
                           if isolated_proposal_results else 0,
        },
        "by_category": category_stats,
        "by_difficulty": difficulty_stats,
        "difficulty_distribution": {d: stats["total"] for d, stats in difficulty_stats.items()},
        "best_move_coverage": _build_coverage_stats(results),
        "branching_factor_breakdown": _build_branching_factor_breakdown(results),
        "top1_engine_match_pct": top1_engine_match_pct,
        "top1_kingsrow_match_pct": top1_kingsrow_match_pct,
        "single_best_violations": single_best_violations,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BEST-MOVE COVERAGE AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _coverage_pct(num: int, denom: int) -> float | None:
    """Return coverage percentage or None when no data exists."""
    return round(100 * num / denom, 1) if denom > 0 else None


def _coverage_block(
    results: list[dict],
    field: str,
) -> dict:
    """
    Build a coverage stats block for one field ('contains_engine_best' or
    'contains_kingsrow_best').  Only positions where the field is a bool
    (not None) are included in the denominator — positions where the
    annotation was absent, or where proposals were unavailable (api/parse
    failures), are excluded so as not to inflate or deflate the metric.
    """
    # Positions with a definite True/False answer
    evaluated = [r for r in results if isinstance(r.get(field), bool)]
    n = len(evaluated)
    contains = sum(1 for r in evaluated if r[field])

    # scanner_correct / scanner_wrong split
    sc_yes  = [r for r in evaluated if r.get("scanner_eval", {}).get("scanner_correct")]
    sc_no   = [r for r in evaluated if not r.get("scanner_eval", {}).get("scanner_correct")]
    sc_yes_c = sum(1 for r in sc_yes if r[field])
    sc_no_c  = sum(1 for r in sc_no  if r[field])

    # quiet / tactical split (ground_truth_has_captures is always computed)
    quiet    = [r for r in evaluated if not r.get("ground_truth_has_captures", False)]
    tactical = [r for r in evaluated if r.get("ground_truth_has_captures", False)]
    quiet_c    = sum(1 for r in quiet    if r[field])
    tactical_c = sum(1 for r in tactical if r[field])

    # category breakdown
    by_cat: dict[str, dict] = {}
    for r in evaluated:
        cat = r.get("category", "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"n": 0, "contains": 0}
        by_cat[cat]["n"] += 1
        if r[field]:
            by_cat[cat]["contains"] += 1
    cat_summary = {
        cat: {
            "n": v["n"],
            "contains": v["contains"],
            "coverage_pct": _coverage_pct(v["contains"], v["n"]),
        }
        for cat, v in sorted(by_cat.items())
    }

    # difficulty breakdown
    by_diff: dict[str, dict] = {}
    for r in evaluated:
        diff = r.get("difficulty", "unknown")
        if diff not in by_diff:
            by_diff[diff] = {"n": 0, "contains": 0}
        by_diff[diff]["n"] += 1
        if r[field]:
            by_diff[diff]["contains"] += 1
    diff_summary = {
        diff: {
            "n": v["n"],
            "contains": v["contains"],
            "coverage_pct": _coverage_pct(v["contains"], v["n"]),
        }
        for diff, v in sorted(by_diff.items())
    }

    return {
        "evaluated": n,
        "contains": contains,
        "coverage_pct": _coverage_pct(contains, n),
        "scanner_correct": {
            "n": len(sc_yes),
            "contains": sc_yes_c,
            "coverage_pct": _coverage_pct(sc_yes_c, len(sc_yes)),
        },
        "scanner_wrong": {
            "n": len(sc_no),
            "contains": sc_no_c,
            "coverage_pct": _coverage_pct(sc_no_c, len(sc_no)),
        },
        "quiet": {
            "n": len(quiet),
            "contains": quiet_c,
            "coverage_pct": _coverage_pct(quiet_c, len(quiet)),
        },
        "tactical": {
            "n": len(tactical),
            "contains": tactical_c,
            "coverage_pct": _coverage_pct(tactical_c, len(tactical)),
        },
        "by_category": cat_summary,
        "by_difficulty": diff_summary,
    }


def _build_coverage_stats(results: list[dict]) -> dict:
    """Build the full best_move_coverage block for summarize_results()."""
    return {
        "engine": _coverage_block(results, "contains_engine_best"),
        "kingsrow": _coverage_block(results, "contains_kingsrow_best"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BRANCHING-FACTOR BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
# Buckets: exact counts 1–10, then "11+" for all larger values.
# Provides thesis evidence that LLM proposal quality degrades with
# combinatorial branching complexity.
_BF_MAX_EXACT = 10   # buckets 1..10 are exact; >=11 is collapsed into "11+"


def _bf_bucket(n: int) -> str:
    """Return the bucket label for a legal-move count."""
    if n <= _BF_MAX_EXACT:
        return str(n)
    return "11+"


def _build_branching_factor_breakdown(results: list[dict]) -> dict:
    """
    Compute per-legal-move-count statistics over all evaluated results.

    Each bucket contains:
      n_positions              — positions in this bucket
      perfect_proposal_pct     — % with classification=="perfect"
      scanner_correct_pct      — % where scanner was correct
      engine_best_coverage_pct — % containing engine best move (None if no data)
      kingsrow_best_coverage_pct — same for KR (None if no annotations)

    legal_move_count is read from result["legal_move_count"] which is always
    populated by evaluate_position() on every code path.
    """
    # Ordered bucket labels for deterministic output
    _LABELS = [str(i) for i in range(1, _BF_MAX_EXACT + 1)] + ["11+"]
    buckets: dict[str, dict] = {
        lbl: {
            "n": 0,
            "perfect": 0,
            "scanner_correct": 0,
            "engine_contains": 0, "engine_evaluated": 0,
            "kr_contains":     0, "kr_evaluated":     0,
        }
        for lbl in _LABELS
    }

    for r in results:
        n_legal = r.get("legal_move_count")
        if n_legal is None:
            continue
        try:
            n_legal = int(n_legal)
        except (TypeError, ValueError):
            continue
        lbl = _bf_bucket(n_legal)
        b = buckets[lbl]
        b["n"] += 1
        if r.get("classification") == "perfect":
            b["perfect"] += 1
        if r.get("scanner_eval", {}).get("scanner_correct"):
            b["scanner_correct"] += 1
        # Engine coverage (bool only — None excluded)
        ce = r.get("contains_engine_best")
        if isinstance(ce, bool):
            b["engine_evaluated"] += 1
            if ce:
                b["engine_contains"] += 1
        # KR coverage
        ck = r.get("contains_kingsrow_best")
        if isinstance(ck, bool):
            b["kr_evaluated"] += 1
            if ck:
                b["kr_contains"] += 1

    # Build output dict with computed percentages
    out: dict[str, dict] = {}
    for lbl in _LABELS:
        b = buckets[lbl]
        n = b["n"]
        if n == 0:
            continue   # omit empty buckets from output
        out[lbl] = {
            "n_positions":               n,
            "perfect_proposal_pct":      _coverage_pct(b["perfect"], n),
            "scanner_correct_pct":       _coverage_pct(b["scanner_correct"], n),
            "engine_best_coverage_pct":  _coverage_pct(b["engine_contains"], b["engine_evaluated"]),
            "kingsrow_best_coverage_pct": _coverage_pct(b["kr_contains"], b["kr_evaluated"]),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

_VALID_MODES = ["full", "balanced", "easy", "medium", "hard", "medium_hard", "jump_only", "quiet_only"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate pure raw LLM baseline (separated scanner + proposal) "
            "against ground truth legal moves."
        )
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help=(
            "Path to JSONL dataset file "
            "(default: checkers/data/legality_stress/scenarios.jsonl)"
        ),
    )
    p.add_argument(
        "--mode", type=str, default="full",
        choices=_VALID_MODES,
        help=(
            "Evaluation mode: full (all scenarios), balanced (equal sample "
            "per difficulty), easy/medium/hard (filter by difficulty), "
            "jump_only (captures only), quiet_only (no captures). "
            "All modes filter from the existing dataset. Default: full."
        ),
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of positions to evaluate (applied after mode filter)",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Output JSON path (default: logs/proposal_seperation_results.json)",
    )
    p.add_argument(
        "--scenario-id", type=str, default=None, dest="scenario_id",
        help="Evaluate a single scenario by ID",
    )
    p.add_argument(
        "--inter-test-delay", type=float, default=0.0, dest="inter_test_delay",
        metavar="SECONDS",
        help=(
            "Seconds to sleep between positions (reduces 429 rate-limit errors). "
            "Example: --inter-test-delay 1.0. Default: 0.0 (no delay)."
        ),
    )
    p.add_argument(
        "--bestmove-annotations", type=str, default=None, dest="bestmove_annotations",
        help=(
            "Path to scenarios_bestmove_annotations.json produced by "
            "build_scenario_bestmove_annotations.py.  When supplied, each "
            "result is enriched with engine_best_move, kingsrow_best_move, "
            "contains_engine_best, contains_kingsrow_best fields and a "
            "best-move coverage section is printed in the summary.  "
            "The evaluator degrades gracefully if the file is absent."
        ),
    )
    p.add_argument(
        "--min-legal-moves", type=int, default=None, dest="min_legal_moves",
        metavar="N",
        help=(
            "Keep only positions with at least N legal moves "
            "(uses hidden_legal_moves length from dataset). "
            "Applied after mode/scenario-id filter, before --limit."
        ),
    )
    p.add_argument(
        "--max-legal-moves", type=int, default=None, dest="max_legal_moves",
        metavar="N",
        help=(
            "Keep only positions with at most N legal moves "
            "(uses hidden_legal_moves length from dataset). "
            "Applied after mode/scenario-id filter, before --limit."
        ),
    )
    p.add_argument(
        "--sample-size", type=int, default=None, dest="sample_size",
        metavar="N",
        help=(
            "Randomly sample exactly N positions (seed=42, deterministic). "
            "Applied AFTER legal-move filtering, BEFORE --limit. "
            "If N >= remaining positions the full filtered set is used unchanged."
        ),
    )
    p.add_argument(
        "--single-best", action="store_true", dest="single_best",
        help="Use single_best proposer mode (outputs only the single strategically best move)",
    )

    args = p.parse_args(argv)

    # Resolve dataset path
    dataset_path = args.dataset or os.path.abspath(_DEFAULT_DATASET)
    out_path = Path(args.out or "logs/proposal_seperation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset(dataset_path)
    if not dataset:
        print(f"ERROR: no valid entries in {dataset_path}")
        return 1

    # Load optional best-move annotations
    # Tries --bestmove-annotations path first; falls back to default path;
    # degrades silently to empty dict when neither exists.
    _ann_path = (
        args.bestmove_annotations
        or (os.path.abspath(_DEFAULT_ANNOTATIONS) if os.path.exists(os.path.abspath(_DEFAULT_ANNOTATIONS)) else None)
    )
    bestmove_annotations = load_bestmove_annotations(_ann_path)

    # Filter by scenario_id (takes priority over mode)
    if args.scenario_id:
        dataset = [e for e in dataset if e.get("scenario_id") == args.scenario_id]
        if not dataset:
            print(f"ERROR: no scenario with id={args.scenario_id!r}")
            return 1
    else:
        # Apply mode filter
        pre_filter_count = len(dataset)
        dataset = filter_by_mode(dataset, args.mode)
        post_filter_count = len(dataset)

    # Apply limit
    if args.limit is not None:
        dataset = dataset[:args.limit]

    # ── Branching-factor filter (--min/max-legal-moves) ──────────────────────────
    # Applied AFTER mode filter and --limit so the limit is on the mode-filtered
    # pool, not the legal-move-filtered subset (consistent with all other filters).
    _min_lm = args.min_legal_moves
    _max_lm = args.max_legal_moves
    _bf_filter_active = (_min_lm is not None) or (_max_lm is not None)
    _before_bf = len(dataset)
    if _bf_filter_active:
        dataset = [
            e for e in dataset
            if (
                (_min_lm is None or len(e.get("hidden_legal_moves", [])) >= _min_lm)
                and (_max_lm is None or len(e.get("hidden_legal_moves", [])) <= _max_lm)
            )
        ]
    _after_bf = len(dataset)

    # Persist filtering metadata for the JSON report
    _filtering_meta = {
        "min_legal_moves": _min_lm,
        "max_legal_moves": _max_lm,
        "active": _bf_filter_active,
        "positions_before": _before_bf,
        "positions_after":  _after_bf,
        "positions_removed": _before_bf - _after_bf,
    }

    # ── Sampling (--sample-size) ───────────────────────────────────────────────
    # Applied AFTER BF filter, BEFORE --limit.
    # seed=42 is fixed so every run with the same dataset + same N gives
    # the same subset — essential for thesis reproducibility.
    _sample_size = args.sample_size
    _before_sample = len(dataset)
    _sample_active = _sample_size is not None and _sample_size < _before_sample
    if _sample_active:
        rng = random.Random(42)
        dataset = rng.sample(dataset, _sample_size)
    _after_sample = len(dataset)

    _sampling_meta = {
        "active": _sample_active,
        "sample_size": _sample_size,
        "positions_before_sampling": _before_sample,
        "positions_after_sampling":  _after_sample,
    }

    # ── Limit (--limit) applied last ─────────────────────────────────────────
    if args.limit is not None:
        dataset = dataset[:args.limit]

    # Print header
    print("=" * 74)
    print("PROPOSAL SEPARATION EVAL — Pure Raw LLM Baseline")
    print(f"  Dataset   : {dataset_path}")
    print(f"  Mode      : {args.mode}")
    if not args.scenario_id:
        print(f"  Filtered  : {post_filter_count}/{pre_filter_count} scenarios")
    if args.limit is not None:
        print(f"  Limit     : {args.limit}")
    print(f"  Positions : {len(dataset)}")
    print(f"  Scanner   : {SCANNER_MODEL}")
    print(f"  Proposal  : {PROPOSAL_MODEL}")
    print(f"  Output    : {out_path}")
    if _bf_filter_active:
        _fmin = f">={_min_lm}" if _min_lm is not None else ""
        _fmax = f"<={_max_lm}" if _max_lm is not None else ""
        _fstr = " ".join(filter(None, [_fmin, _fmax]))
        print(f"  BF filter : legal_moves {_fstr}  ({_after_bf}/{_before_bf} kept)")
    if _sample_active:
        print(f"  Sample    : {_after_sample}/{_before_sample} sampled (seed=42)")
    elif _sample_size is not None:
        print(f"  Sample    : N={_sample_size} >= pool ({_before_sample}) — no-op, using full set")
    if bestmove_annotations:
        print(f"  Annotations: {len(bestmove_annotations)} entries loaded (best-move coverage ON)")
    else:
        print(f"  Annotations: none — best-move coverage metrics will be skipped")
    if args.inter_test_delay > 0:
        print(f"  Delay     : {args.inter_test_delay:.1f}s between positions")
    print("=" * 74)
    print()
    print(f"  {'#':>4}  {'scenario_id':<42}  {'quad':<10}  {'class':<25}  {'time':>6}")
    print("  " + "-" * 94)

    results: list[dict[str, Any]] = []
    for i, entry in enumerate(dataset, 1):
        board          = entry["board"]
        current_player = _player_from_str(entry["side_to_move"])
        scenario_id    = entry.get("scenario_id", f"pos_{i}")
        category       = entry.get("category", "")
        difficulty     = entry.get("difficulty", "")

        result = evaluate_position(
            board=board,
            current_player=current_player,
            scenario_id=scenario_id,
            category=category,
            difficulty=difficulty,
            bestmove_annotations=bestmove_annotations,
            single_best=args.single_best,
        )
        results.append(result)

        # Print per-position result
        q   = result.get("quadrant", "?")[:10]
        cls = result.get("classification", "?")[:25]
        t   = result.get("elapsed_s", 0)
        print(f"  {i:>4}  {scenario_id:<42}  {q:<10}  {cls:<25}  {t:>5.1f}s")

        # Inter-test delay — applied after every position except the last
        if args.inter_test_delay > 0 and i < len(dataset):
            time.sleep(args.inter_test_delay)

    # Summary
    summary = summarize_results(results)
    summary["filtering"] = _filtering_meta
    summary["sampling"]  = _sampling_meta

    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)

    if not results:
        print("  Total positions       : 0")
        print("  No positions were evaluated (dataset empty or limited to 0).")
        print("=" * 74)
        report = {
            "meta": {
                "dataset": dataset_path,
                "mode": args.mode,
                "position_count": len(dataset),
                "scanner_model": SCANNER_MODEL,
                "proposal_model": PROPOSAL_MODEL,
                "annotations_path": _ann_path,
                "output": str(out_path),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            "summary": summary,
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")
        return 0

    print(f"  Total positions       : {summary['total_positions']}")
    if "single_best_violations" in summary:
        print(f"  Single-Best Violations: {summary['single_best_violations']}")
    print()
    print("  ── Quadrant Breakdown ──")
    for q_name, q_count in summary["quadrants"].items():
        pct = round(100 * q_count / summary["total_positions"], 1) if summary["total_positions"] else 0
        print(f"    {q_name:<42}: {q_count:>4}  ({pct:>5.1f}%)")
    print()
    print("  ── Classification Breakdown ──")
    for c_name, c_count in sorted(summary["classifications"].items()):
        pct = round(100 * c_count / summary["total_positions"], 1) if summary["total_positions"] else 0
        print(f"    {c_name:<30}: {c_count:>4}  ({pct:>5.1f}%)")
    print()
    print("  ── Scanner Accuracy ──")
    sa = summary["scanner_accuracy"]
    print(f"    {sa['correct']}/{sa['total']} correct ({sa['accuracy_pct']}%)")
    print()
    print("  ── Branch Routing ──")
    br = summary["branch_routing"]
    print(f"    Correct branch called : {br['branch_correct']}/{br['total']} ({100-br['mismatch_pct']:.1f}%)")
    print(f"    Wrong branch called   : {br['branch_mismatch']}/{br['total']} ({br['mismatch_pct']:.1f}%)  ← scanner mismatch")
    print()
    print("  ── Proposal Accuracy (all, excl. api/parse failures) ──")
    pa = summary["proposal_accuracy"]
    print(f"    {pa['perfect']}/{pa['total']} perfect ({pa['perfect_pct']}%)  [includes wrong-branch positions]")
    print()
    print("  ── Proposal Accuracy — Isolated (right branch only) ──")
    pi = summary["proposal_accuracy_isolated"]
    print(f"    {pi['perfect']}/{pi['total']} perfect ({pi['perfect_pct']}%)  ← true proposer quality")
    print()
    print("  ── Failure Taxonomy ──")
    ft = summary["failure_taxonomy"]
    avg_missing = ft.get("missing_legal_moves_avg", 0.0)
    for tax_name, tax_count in ft.items():
        if tax_name == "missing_legal_moves_avg":
            continue  # printed inline with missing_legal_moves
        if tax_name == "missing_legal_moves":
            marker = f"  (avg {avg_missing:.2f} per position)"
        elif tax_name == "wrong_branch_called" and tax_count > 0:
            marker = "  ← cascading (scanner error)"
        else:
            marker = ""
        print(f"    {tax_name:<30}: {tax_count:>4}{marker}")
    print()

    if summary.get("by_category"):
        print("  ── By Category ──")
        for cat, cs in sorted(summary["by_category"].items()):
            pct = round(100 * cs["perfect"] / cs["total"], 1) if cs["total"] else 0
            s_pct = round(100 * cs["scanner_correct"] / cs["total"], 1) if cs["total"] else 0
            print(
                f"    {cat:<30}: {cs['perfect']:>3}/{cs['total']:<3} perfect ({pct:>5.1f}%)  "
                f"scanner={s_pct:>5.1f}%"
            )
        print()

    if summary.get("by_difficulty"):
        print("  ── By Difficulty ──")
        for diff, ds in sorted(summary["by_difficulty"].items()):
            pct = round(100 * ds["perfect"] / ds["total"], 1) if ds["total"] else 0
            print(f"    {diff:<15}: {ds['perfect']:>3}/{ds['total']:<3} perfect ({pct:>5.1f}%)")
        print()

    if summary.get("difficulty_distribution"):
        print("  ── Difficulty Distribution ──")
        total_pos = summary.get("total_positions", 0)
        for diff, count in sorted(summary["difficulty_distribution"].items()):
            pct = round(100 * count / total_pos, 1) if total_pos else 0
            print(f"    {diff:<15}: {count:>4} ({pct:>5.1f}%)")
        print()

    # Best-move coverage summary (only printed when data is available)
    cov = summary.get("best_move_coverage", {})
    _eng = cov.get("engine", {})
    _kr  = cov.get("kingsrow", {})
    if _eng.get("evaluated", 0) > 0 or _kr.get("evaluated", 0) > 0:
        print("  ── Best-Move Coverage ──")
        print("  (% of positions where the LLM's proposal list included the best move)")
        print()

        def _cov_line(label: str, block: dict) -> None:
            n   = block.get("evaluated", 0)
            c   = block.get("contains", 0)
            pct = block.get("coverage_pct")
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
            print(f"    {label:<30}: {c:>4}/{n:<4} ({pct_str:>7})")

        _cov_line("engine best move", _eng)
        _cov_line("kingsrow best move", _kr)
        print()

        # Print top-1 match pct if single-best or annotations present
        if "top1_engine_match_pct" in summary or "top1_kingsrow_match_pct" in summary:
            print("  ── Top-1 Match Pct ──")
            print("  (% of positions where the first proposed move matches the best move)")
            print()
            print(f"    engine best move match        : {summary.get('top1_engine_match_pct', 0.0)}%")
            print(f"    kingsrow best move match      : {summary.get('top1_kingsrow_match_pct', 0.0)}%")
            print()

        # Scanner breakdown
        for _lbl, _blk in (("Engine", _eng), ("KingsRow", _kr)):
            sc_y = _blk.get("scanner_correct", {})
            sc_n = _blk.get("scanner_wrong", {})
            if sc_y.get("n", 0) + sc_n.get("n", 0) > 0:
                sy_p = sc_y.get("coverage_pct") or 0
                sn_p = sc_n.get("coverage_pct") or 0
                print(
                    f"    {_lbl} — scanner correct : "
                    f"{sc_y.get('contains',0)}/{sc_y.get('n',0)}  ({sy_p:.1f}%)  "
                    f"| scanner wrong : "
                    f"{sc_n.get('contains',0)}/{sc_n.get('n',0)}  ({sn_p:.1f}%)"
                )
        print()

        # Quiet / Tactical breakdown
        for _lbl, _blk in (("Engine", _eng), ("KingsRow", _kr)):
            qt = _blk.get("quiet", {})
            tc = _blk.get("tactical", {})
            if qt.get("n", 0) + tc.get("n", 0) > 0:
                qp = qt.get("coverage_pct") or 0
                tp = tc.get("coverage_pct") or 0
                print(
                    f"    {_lbl} — quiet    : "
                    f"{qt.get('contains',0)}/{qt.get('n',0)}  ({qp:.1f}%)  "
                    f"| tactical : "
                    f"{tc.get('contains',0)}/{tc.get('n',0)}  ({tp:.1f}%)"
                )
        print()

        # Category breakdown (engine only; omit if empty)
        eng_cats = _eng.get("by_category", {})
        if eng_cats:
            print("    Engine coverage by category:")
            for cat, cv in sorted(eng_cats.items()):
                cp = cv.get("coverage_pct")
                cp_str = f"{cp:.1f}%" if cp is not None else "N/A"
                print(f"      {cat:<30}: {cv['contains']:>3}/{cv['n']:<3}  ({cp_str:>7})")
            print()

        # Difficulty breakdown
        eng_diffs = _eng.get("by_difficulty", {})
        if eng_diffs:
            print("    Engine coverage by difficulty:")
            for diff, dv in sorted(eng_diffs.items()):
                dp = dv.get("coverage_pct")
                dp_str = f"{dp:.1f}%" if dp is not None else "N/A"
                print(f"      {diff:<15}: {dv['contains']:>3}/{dv['n']:<3}  ({dp_str:>7})")
            print()

    # Branching-factor breakdown
    bf = summary.get("branching_factor_breakdown", {})
    if bf:
        print("  ── Branching-Factor Breakdown ──")
        print("  (metrics by exact legal-move count; 11+ = all positions with ≥11 moves)")
        print()
        _HAS_ENG = any(v.get("engine_best_coverage_pct") is not None for v in bf.values())
        _HAS_KR  = any(v.get("kingsrow_best_coverage_pct") is not None for v in bf.values())
        # Header
        _hdr = f"    {'n_legal':>7}  {'n':>5}  {'perfect%':>9}  {'scan_ok%':>9}"
        if _HAS_ENG: _hdr += f"  {'eng_cov%':>9}"
        if _HAS_KR:  _hdr += f"  {'kr_cov%':>8}"
        print(_hdr)
        print("    " + "-" * (len(_hdr) - 4))
        for lbl, bv in bf.items():
            n   = bv["n_positions"]
            pp  = bv["perfect_proposal_pct"]
            sp  = bv["scanner_correct_pct"]
            ep  = bv["engine_best_coverage_pct"]
            kp  = bv["kingsrow_best_coverage_pct"]
            def _f(v): return f"{v:.1f}%" if v is not None else "  N/A"
            row = f"    {lbl:>7}  {n:>5}  {_f(pp):>9}  {_f(sp):>9}"
            if _HAS_ENG: row += f"  {_f(ep):>9}"
            if _HAS_KR:  row += f"  {_f(kp):>8}"
            print(row)
        print()

    # Write report
    report = {
        "meta": {
            "dataset": dataset_path,
            "mode": args.mode,
            "single_best": args.single_best,
            "position_count": len(dataset),
            "scanner_model": SCANNER_MODEL,
            "proposal_model": PROPOSAL_MODEL,
            "annotations_path": _ann_path,
            "output": str(out_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "summary": summary,
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
