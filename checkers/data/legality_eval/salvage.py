"""
checkers/data/legality_eval/salvage.py
=======================================
Secondary "raw output salvage analysis" for parse_failure records.

Purpose
-------
When a record's result_type is "parse_failure", the required output schema
was incorrect, so we exclude the response from the main legality evaluation.
This module attempts to recover any move/selection the LLM actually produced
and checks whether it would have been legal — purely for diagnostic purposes.

Design contract
---------------
- NEVER mutates result_type.
- NEVER replaces parse_success or legal fields.
- NEVER affects legal_move_rate.
- Returns a separate salvage dict that is merged into the record as extra fields.
- Works post-hoc on saved JSONL records as well as live in run_pilot.

Salvage types
-------------
  "none"              no salvage was attempted (record already parsed OK)
  "chosen_index"      extracted a valid chosen_index from raw response (B8/B8c)
  "selected_move"     extracted selected_move coordinates from raw response
  "coordinate_path"   extracted bare coordinate array from raw response

Failure reasons (raw_salvage_failure_reason)
--------------------------------------------
  "no_move_output"
  "no_index_output"
  "invalid_index"
  "malformed_coordinates"
  "ambiguous_multiple_moves"
  "cannot_parse_raw_output"
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Index-style baselines — salvage looks for chosen_index first.
_INDEX_STYLE_BASELINES = {
    "B8a_ranker_shortlist_no_safety",
    "B8b_ranker_full_legal_shuffled_no_safety",
    "B8c_ranker_compare_no_safety",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _path_key(path: list) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _norm_coord(sq) -> Optional[list]:
    """Normalise one coordinate entry to [row, col] or None."""
    if isinstance(sq, (list, tuple)) and len(sq) == 2:
        try:
            return [int(sq[0]), int(sq[1])]
        except (TypeError, ValueError):
            return None
    return None


def _norm_path(raw) -> Optional[list]:
    """Normalise a raw path value to a list of [row, col] pairs, or None."""
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    coords = [_norm_coord(sq) for sq in raw]
    if any(c is None for c in coords):
        return None
    return coords


def _is_legal_path(path: list, hidden_legal_moves: list) -> bool:
    key = _path_key(path)
    for m in hidden_legal_moves:
        p = _norm_path(m.get("path"))
        if p is not None and _path_key(p) == key:
            return True
    return False


def _extract_chosen_index_from_raw(raw: str) -> Optional[int]:
    """
    Try to extract a chosen_index integer from a raw LLM string.
    Attempts JSON parse first, then regex.
    """
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            ci = obj.get("chosen_index")
            if isinstance(ci, int):
                return ci
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Regex fallback
    m = re.search(r'"chosen_index"\s*:\s*(\d+)', raw)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    m = re.search(r'\bchosen_index\s*[=:]\s*(\d+)', raw)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _extract_selected_move_from_raw(raw: str) -> Optional[list]:
    """
    Try to extract a selected_move coordinate array from a raw LLM string.
    Attempts JSON parse first, then regex scan for [[r,c],[r,c]...].
    """
    # JSON parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            mv = obj.get("selected_move")
            if isinstance(mv, list) and len(mv) >= 2:
                path = _norm_path(mv)
                if path is not None:
                    return path
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Regex: find the first [[row,col],...] pattern in the string
    pattern = re.compile(
        r'\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*\])+)\s*\]'
    )
    matches = pattern.findall(raw)
    if len(matches) == 1:
        try:
            arr = json.loads(matches[0])
            path = _norm_path(arr)
            if path is not None:
                return path
        except (json.JSONDecodeError, ValueError):
            pass
    elif len(matches) > 1:
        # Try to pick the one tagged as selected_move; otherwise ambiguous
        sm_tag = re.search(r'"selected_move"\s*:\s*(\[\s*\[.*?\]\s*\])', raw, re.DOTALL)
        if sm_tag:
            try:
                arr = json.loads(sm_tag.group(1))
                path = _norm_path(arr)
                if path is not None:
                    return path
            except (json.JSONDecodeError, ValueError):
                pass
        return None   # ambiguous — do not salvage

    return None


# ── public API ────────────────────────────────────────────────────────────────

def salvage_parse_failure(
    record: dict[str, Any],
    raw_llm_response: str,
    hidden_legal_moves: list,
    board: list,
    side_str: str,
    candidates: Optional[list] = None,   # scored move dicts for B8/B8c index resolution
) -> dict[str, Any]:
    """
    Attempt to recover a move from a parse_failure record.

    Parameters
    ----------
    record              Existing result record (read-only, never mutated).
    raw_llm_response    The actual LLM output string (may be "" on API failure).
    hidden_legal_moves  Ground-truth legal moves (NEVER shown to LLM).
    board               8×8 board state.
    side_str            "RED" | "BLACK".
    candidates          Ranked candidate dicts (for B8/B8c index→path resolution).

    Returns
    -------
    Dict with raw_salvage_* fields.  Merge into record as extra diagnostics.
    """
    # Only salvage actual parse failures
    if record.get("result_type") != "parse_failure":
        return {
            "raw_salvage_attempted":      False,
            "raw_salvage_type":           "none",
            "raw_salvage_success":        False,
            "raw_salvaged_move":          None,
            "raw_salvaged_move_legal":    None,
            "raw_salvage_illegal_type":   None,
            "raw_salvage_failure_reason": None,
        }

    baseline = record.get("baseline", "")
    raw = raw_llm_response.strip() if raw_llm_response else ""

    if not raw:
        return {
            "raw_salvage_attempted":      True,
            "raw_salvage_type":           "none",
            "raw_salvage_success":        False,
            "raw_salvaged_move":          None,
            "raw_salvaged_move_legal":    None,
            "raw_salvage_illegal_type":   None,
            "raw_salvage_failure_reason": "cannot_parse_raw_output",
        }

    # ── Index-style baselines (B8a/B8b/B8c) ─────────────────────────────────
    if baseline in _INDEX_STYLE_BASELINES:
        return _salvage_index_style(raw, hidden_legal_moves, candidates)

    # ── Path-style baselines (B1-B7, B9, etc.) ──────────────────────────────
    return _salvage_path_style(raw, hidden_legal_moves)


def _salvage_index_style(
    raw: str,
    hidden_legal_moves: list,
    candidates: Optional[list],
) -> dict[str, Any]:
    """Salvage for index-output baselines (B8a/B8b/B8c)."""

    # 1. Try chosen_index → resolve to candidate path
    ci = _extract_chosen_index_from_raw(raw)
    if ci is not None:
        if candidates and 0 <= ci < len(candidates):
            path = _norm_path(candidates[ci].get("path"))
            if path:
                legal = _is_legal_path(path, hidden_legal_moves)
                return {
                    "raw_salvage_attempted":      True,
                    "raw_salvage_type":           "chosen_index",
                    "raw_salvage_success":        True,
                    "raw_salvaged_move":          path,
                    "raw_salvaged_move_legal":    legal,
                    "raw_salvage_illegal_type":   None if legal else "salvaged_candidate_illegal",
                    "raw_salvage_failure_reason": None,
                }
        # Index found but out of range or no candidates
        return {
            "raw_salvage_attempted":      True,
            "raw_salvage_type":           "chosen_index",
            "raw_salvage_success":        False,
            "raw_salvaged_move":          None,
            "raw_salvaged_move_legal":    None,
            "raw_salvage_illegal_type":   None,
            "raw_salvage_failure_reason": "invalid_index",
        }

    # 2. Try selected_move / coordinate path (schema confusion — model wrote
    #    coordinates instead of an index)
    path = _extract_selected_move_from_raw(raw)
    if path:
        legal = _is_legal_path(path, hidden_legal_moves)
        return {
            "raw_salvage_attempted":      True,
            "raw_salvage_type":           "selected_move",
            "raw_salvage_success":        True,
            "raw_salvaged_move":          path,
            "raw_salvaged_move_legal":    legal,
            "raw_salvage_illegal_type":   None if legal else "salvaged_coordinate_illegal",
            "raw_salvage_failure_reason": None,
        }

    # 3. Nothing usable
    return {
        "raw_salvage_attempted":      True,
        "raw_salvage_type":           "none",
        "raw_salvage_success":        False,
        "raw_salvaged_move":          None,
        "raw_salvaged_move_legal":    None,
        "raw_salvage_illegal_type":   None,
        "raw_salvage_failure_reason": "no_usable_selection",
    }


def _salvage_path_style(raw: str, hidden_legal_moves: list) -> dict[str, Any]:
    """Salvage for coordinate-output baselines (B1-B7, B9)."""
    path = _extract_selected_move_from_raw(raw)

    if path is None:
        # Check if there were multiple coordinate arrays (ambiguous)
        pattern = re.compile(
            r'\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*\])+)\s*\]'
        )
        matches = pattern.findall(raw)
        reason = "ambiguous_multiple_moves" if len(matches) > 1 else "no_move_output"
        return {
            "raw_salvage_attempted":      True,
            "raw_salvage_type":           "none",
            "raw_salvage_success":        False,
            "raw_salvaged_move":          None,
            "raw_salvaged_move_legal":    None,
            "raw_salvage_illegal_type":   None,
            "raw_salvage_failure_reason": reason,
        }

    legal = _is_legal_path(path, hidden_legal_moves)
    return {
        "raw_salvage_attempted":      True,
        "raw_salvage_type":           "selected_move",
        "raw_salvage_success":        True,
        "raw_salvaged_move":          path,
        "raw_salvaged_move_legal":    legal,
        "raw_salvage_illegal_type":   None if legal else "salvaged_path_illegal",
        "raw_salvage_failure_reason": None,
    }


# ── Aggregate salvage metrics from a list of records ─────────────────────────

def aggregate_salvage(results: list[dict[str, Any]], n_total: int) -> dict[str, Any]:
    """
    Compute salvage summary metrics for a single baseline's result list.
    Called by metrics.aggregate() — results are already for one baseline.
    """
    pf = [r for r in results if r.get("result_type") == "parse_failure"]
    n_pf = len(pf)

    attempted       = [r for r in pf if r.get("raw_salvage_attempted", False)]
    success         = [r for r in pf if r.get("raw_salvage_success", False)]
    salvage_legal   = [r for r in success if r.get("raw_salvaged_move_legal") is True]
    salvage_illegal = [r for r in success if r.get("raw_salvaged_move_legal") is False]
    no_usable       = [r for r in pf
                       if not r.get("raw_salvage_success", False)
                       and r.get("raw_salvage_attempted", False)]

    n_normal_legal = sum(1 for r in results if r.get("result_type") == "legal")

    def _rate(a, b):
        return round(a / b, 4) if b > 0 else None

    return {
        "parse_failure_count":           n_pf,
        "salvage_attempted_count":        len(attempted),
        "salvage_success_count":          len(success),
        "salvage_legal_count":            len(salvage_legal),
        "salvage_illegal_count":          len(salvage_illegal),
        "no_usable_output_count":         len(no_usable),
        # Adjusted end-to-end rates (denominator: n_total including parse failures)
        "adjusted_e2e_legal_if_salvaged": _rate(n_normal_legal + len(salvage_legal), n_total),
        "adjusted_e2e_unusable":          _rate(len(no_usable) + len(salvage_illegal) +
                                                sum(1 for r in results if r.get("result_type") == "illegal"),
                                                n_total),
    }
