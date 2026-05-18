"""
checkers/data/legality_eval/candidate_moves.py
──────────────────────────────────────────────
Candidate move generator for B5_candidate_moves_rule_filter.

Generates PHYSICALLY-POSSIBLE moves from a board + side_to_move WITHOUT
applying the global mandatory-capture filter that get_all_legal_moves() uses.

Crucially:
  - When a jump is available, simple moves are still included as candidates.
  - The LLM receives both jump candidates AND tempting simple-move candidates.
  - The LLM must apply the MANDATORY CAPTURE rule to reject simple moves.
  - Candidates are NOT labelled "legal" or "illegal".
  - hidden_legal_moves are NEVER exposed.

Design:
  1. _simple_candidates() — all one-step diagonal moves to empty squares,
     grouped by origin piece. No mandatory-capture filter applied.
  2. _jump_candidates()   — all single-leg and multi-leg jump sequences,
     using the existing get_all_jump_sequences() for geometry correctness.
  3. get_candidates()     — union of both, deduplicated by path, with metadata.

Each candidate dict:
  {
    "id":          "C0",           # display label
    "move_type":   "simple" | "jump",
    "path":        [[r,c], ...],   # 2 squares for simple, 3+ for multi-jump
    "captured":    [[r,c], ...],   # empty for simple
    "from_piece":  "r"/"R"/"b"/"B",
  }
"""

from __future__ import annotations

from typing import Any

from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING,
    in_bounds, is_own_piece, is_opponent_piece, is_king,
)
from checkers.engine.rules import (
    get_move_directions,
    get_all_jump_sequences,
    get_single_jumps,
)

_PIECE_CHAR = {
    RED: "r", RED_KING: "R",
    BLACK: "b", BLACK_KING: "B",
    EMPTY: ".",
}

BOARD_SIZE = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_path(path) -> list[list[int]]:
    return [list(sq) for sq in path]


def _path_key(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


# ── Simple candidates (no mandatory-capture filter) ───────────────────────────

def _simple_candidates(
    board: list[list[int]],
    player: int,
) -> list[dict[str, Any]]:
    """
    All one-step diagonal moves to empty dark squares for every own piece.
    Does NOT apply mandatory-capture filter — simple moves are included even
    when a jump is available.
    """
    candidates = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if not is_own_piece(piece, player):
                continue
            for dr, dc in get_move_directions(piece):
                tr, tc = row + dr, col + dc
                if in_bounds(tr, tc) and board[tr][tc] == EMPTY:
                    candidates.append({
                        "move_type":  "simple",
                        "path":       [[row, col], [tr, tc]],
                        "captured":   [],
                        "from_piece": _PIECE_CHAR.get(piece, "?"),
                    })
    return candidates


# ── Jump candidates (full multi-leg sequences) ────────────────────────────────

def _jump_candidates(
    board: list[list[int]],
    player: int,
) -> list[dict[str, Any]]:
    """
    All complete jump sequences (single-leg or multi-leg) for every own piece.
    Uses get_all_jump_sequences() for geometric correctness (anti-recapture,
    promotion termination).
    """
    candidates = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if not is_own_piece(piece, player):
                continue
            seqs = get_all_jump_sequences(
                board, row, col, player,
                path_so_far=[(row, col)],
                captured_so_far=[],
            )
            for seq in seqs:
                path = seq.get("path", [])
                cap  = seq.get("captured", [])
                if len(path) >= 2:
                    candidates.append({
                        "move_type":  "jump",
                        "path":       _norm_path(path),
                        "captured":   _norm_path(cap),
                        "from_piece": _PIECE_CHAR.get(piece, "?"),
                    })
    return candidates


# ── Public API ────────────────────────────────────────────────────────────────

def get_candidates(
    board: list[list[int]],
    player: int,
) -> dict[str, Any]:
    """
    Return all physically-possible move candidates for *player* on *board*
    WITHOUT applying the mandatory-capture filter.

    Returns a dict:
    {
      "candidates":               list[dict]   — all candidates with id field
      "capture_candidate_count":  int          — jump candidates
      "simple_candidate_count":   int          — simple candidates
      "any_jump_available":       bool         — True if at least one jump candidate
    }

    IMPORTANT: This function does NOT expose hidden_legal_moves.
    The mandatory-capture filter is intentionally omitted so that simple-move
    candidates remain visible when jumps also exist.
    """
    simples = _simple_candidates(board, player)
    jumps   = _jump_candidates(board, player)

    # Deduplicate by path key (shouldn't overlap, but be safe)
    seen: set[tuple] = set()
    combined: list[dict] = []
    for c in jumps + simples:      # jumps first so they appear first in list
        k = _path_key(c["path"])
        if k not in seen:
            seen.add(k)
            combined.append(c)

    # Assign display IDs
    for i, c in enumerate(combined):
        c["id"] = f"C{i}"

    any_jump = any(c["move_type"] == "jump" for c in combined)

    return {
        "candidates":               combined,
        "capture_candidate_count":  sum(1 for c in combined if c["move_type"] == "jump"),
        "simple_candidate_count":   sum(1 for c in combined if c["move_type"] == "simple"),
        "any_jump_available":       any_jump,
    }


# ── Diagnostic helpers (for JSONL records) ───────────────────────────────────

def match_selected_to_candidate(
    selected_path: list[list[int]] | None,
    candidates: list[dict],
) -> dict[str, Any]:
    """
    Given the LLM's selected_move path and the candidate list, return
    B5-specific diagnostic fields.

    Returns:
      {
        "selected_candidate_id":                    str | None
        "selected_candidate_move_type":             str | None
        "selected_path_not_in_candidates":          bool
        "selected_candidate_was_simple_when_jump_available": bool
      }
    """
    if not selected_path:
        return {
            "selected_candidate_id":                         None,
            "selected_candidate_move_type":                  None,
            "selected_path_not_in_candidates":               True,
            "selected_candidate_was_simple_when_jump_available": False,
        }

    sel_key = _path_key(selected_path)
    any_jump = any(c["move_type"] == "jump" for c in candidates)

    for c in candidates:
        if _path_key(c["path"]) == sel_key:
            return {
                "selected_candidate_id":    c["id"],
                "selected_candidate_move_type": c["move_type"],
                "selected_path_not_in_candidates": False,
                "selected_candidate_was_simple_when_jump_available": (
                    c["move_type"] == "simple" and any_jump
                ),
            }

    return {
        "selected_candidate_id":                         None,
        "selected_candidate_move_type":                  None,
        "selected_path_not_in_candidates":               True,
        "selected_candidate_was_simple_when_jump_available": False,
    }


def match_b6_response(
    raw_selected_id: str | None,
    selected_path: list[list[int]] | None,
    candidates: list[dict],
) -> dict:
    """
    B6-specific diagnostic: cross-validate the LLM's claimed candidate ID
    against the path it actually output.

    Returns:
    {
      "b6_selected_candidate_id":                 str | None  — claimed ID
      "b6_selected_candidate_id_valid":           bool        — ID exists in list
      "b6_selected_move_matches_candidate_id":    bool        — path == claimed ID's path
      "b6_selected_path_not_in_candidates":       bool        — path not in any candidate
      "b6_selected_candidate_was_simple_when_jump_available": bool
    }
    """
    any_jump = any(c["move_type"] == "jump" for c in candidates)
    by_id    = {c["id"]: c for c in candidates}

    id_valid      = raw_selected_id in by_id if raw_selected_id else False
    claimed_cand  = by_id.get(raw_selected_id) if id_valid else None

    # Does the output path match what the claimed ID's path is?
    path_matches_id = False
    if claimed_cand and selected_path:
        path_matches_id = _path_key(claimed_cand["path"]) == _path_key(selected_path)

    # Is the output path in the candidate list at all (regardless of claimed ID)?
    in_cands = False
    was_simple_with_jump = False
    if selected_path:
        sel_key = _path_key(selected_path)
        for c in candidates:
            if _path_key(c["path"]) == sel_key:
                in_cands = True
                was_simple_with_jump = (c["move_type"] == "simple" and any_jump)
                break

    return {
        "b6_selected_candidate_id":               raw_selected_id,
        "b6_selected_candidate_id_valid":          id_valid,
        "b6_selected_move_matches_candidate_id":   path_matches_id,
        "b6_selected_path_not_in_candidates":      not in_cands,
        "b6_selected_candidate_was_simple_when_jump_available": was_simple_with_jump,
    }


def match_b7_response(
    selected_path: list[list[int]] | None,
    candidates: list[dict],
) -> dict:
    """
    B7-specific diagnostic: check whether the selected_move path matches any
    displayed candidate (path-only — no candidate ID to cross-validate).

    Returns:
    {
      "b7_selected_path_not_in_candidates":       bool
      "b7_selected_candidate_match_count":        int   — 0 or 1 (paths are unique)
      "b7_selected_candidate_move_type":          str | None
      "b7_selected_candidate_was_simple_when_jump_available": bool
    }
    """
    any_jump = any(c["move_type"] == "jump" for c in candidates)

    if not selected_path:
        return {
            "b7_selected_path_not_in_candidates":               True,
            "b7_selected_candidate_match_count":                0,
            "b7_selected_candidate_move_type":                  None,
            "b7_selected_candidate_was_simple_when_jump_available": False,
        }

    sel_key = _path_key(selected_path)
    for c in candidates:
        if _path_key(c["path"]) == sel_key:
            return {
                "b7_selected_path_not_in_candidates":               False,
                "b7_selected_candidate_match_count":                1,
                "b7_selected_candidate_move_type":                  c["move_type"],
                "b7_selected_candidate_was_simple_when_jump_available": (
                    c["move_type"] == "simple" and any_jump
                ),
            }

    return {
        "b7_selected_path_not_in_candidates":               True,
        "b7_selected_candidate_match_count":                0,
        "b7_selected_candidate_move_type":                  None,
        "b7_selected_candidate_was_simple_when_jump_available": False,
    }
