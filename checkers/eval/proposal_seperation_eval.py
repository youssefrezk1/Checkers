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
    SCANNER_MODEL,
    PROPOSAL_MODEL,
)

logger = logging.getLogger(__name__)

# ── Default dataset path ─────────────────────────────────────────────────────
_DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "data", "legality_stress", "scenarios.jsonl"
)


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
) -> dict[str, Any]:
    """
    Run the full pipeline and evaluate one position.

    Uses GROUND TRUTH routing: even if the scanner is wrong, the proposal
    is evaluated against the correct branch (jumps or simples).
    """
    t0 = time.perf_counter()

    # ── Ground truth ──────────────────────────────────────────────────────
    legal_moves = get_all_legal_moves(board, current_player)
    ground_truth_has_captures = any(m["type"] == "jump" for m in legal_moves)

    # Normalize legal moves for comparison
    legal_norm = [_norm_move(m) for m in legal_moves]

    # ── Run separated proposal pipeline ──────────────────────────────────
    pipeline_result = run_proposal_seperation(board, current_player)
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
        gt_result = run_proposer_only(board, current_player, gt_branch)

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
    proposal_correct = proposal_class["classification"] == "perfect"

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
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

_VALID_MODES = ["full", "balanced", "easy", "medium", "hard", "jump_only", "quiet_only"]


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

    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  Total positions       : {summary['total_positions']}")
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

    # Write report
    report = {
        "meta": {
            "dataset": dataset_path,
            "mode": args.mode,
            "position_count": len(dataset),
            "scanner_model": SCANNER_MODEL,
            "proposal_model": PROPOSAL_MODEL,
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
