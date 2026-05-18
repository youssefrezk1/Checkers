"""
checkers/data/legality_eval/evaluator.py
=========================================
Evaluates one LLM response against the symbolic engine's hidden_legal_moves.

Schema (new)
------------
The LLM outputs:
  {
    "selected_move": [[row, col], [row, col], ...],
    "reasoning": "..."
  }

The evaluator checks whether selected_move is a member of hidden_legal_moves
and classifies the failure mode when it is not.

Privacy guarantee
-----------------
hidden_legal_moves are loaded by the CALLER after the LLM call.
This module never receives or emits them as part of any prompt.

Per-scenario metrics
--------------------
  parse_success     bool   JSON parsed and selected_move is present and well-formed
  legal             bool   selected_move is in hidden_legal_moves
  illegal_move_type str    failure category when legal=False (see ILLEGAL_TYPES)
  wrong_direction   str|None   direction violation detail, or None
  mandatory_violation bool  simple move chosen when a jump was mandatory
  multi_jump_incomplete bool  jump path shorter than matching hidden jump
  n_legal           int    cardinality of hidden_legal_moves (ground truth)
  parse_error       str    "" when parse_success=True, else error reason

ILLEGAL_TYPES (field: illegal_move_type)
-----------------------------------------
  ""                         legal move — no error
  "parse_failed"             JSON invalid or selected_move field missing/malformed
  "wrong_piece_square"       from-square is off-board or holds no friendly piece
  "invalid_destination"      to-square is off-board, light, or occupied
  "wrong_direction"          direction violates piece-type rule
  "mandatory_capture_violation"  simple chosen when jump was mandatory
  "multi_jump_incomplete"    jump chain shorter than required
  "path_not_in_legal_moves"  valid format but path not found in hidden_legal_moves
"""

from __future__ import annotations

import json
from typing import Any, Optional

# Engine piece constants (mirrors engine/board.py)
_EMPTY    = 0
_RED      = 1
_BLACK    = 2
_RED_KING = 3
_BLACK_K  = 4


# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------

def _norm_path(path: Any) -> Optional[list[list[int]]]:
    """Normalise a path value to list-of-[int,int]. Returns None if invalid."""
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        return None
    result: list[list[int]] = []
    for sq in path:
        if not isinstance(sq, (list, tuple)) or len(sq) != 2:
            return None
        try:
            result.append([int(sq[0]), int(sq[1])])
        except (TypeError, ValueError):
            return None
    return result


def _path_key(path: list[list[int]]) -> tuple:
    return tuple(tuple(sq) for sq in path)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def parse_llm_output(raw: str) -> tuple[Optional[dict], str]:
    """
    Parse the LLM's raw string response.
    Returns (parsed_dict | None, error_reason).  error_reason="" on success.
    """
    if not raw or not raw.strip():
        return None, "empty_response"
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.split("\n") if not ln.strip().startswith("```")
        ).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
        return None, "not_a_json_object"
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        end = text.rfind("}")
        if end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj, ""
            except json.JSONDecodeError:
                pass
    return None, "json_decode_error"


# ---------------------------------------------------------------------------
# Board helpers for subcategorisation
# ---------------------------------------------------------------------------

def _on_board(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _is_dark(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


def _piece_at(board: list[list[int]], r: int, c: int) -> int:
    return board[r][c] if _on_board(r, c) else _EMPTY


def _is_friendly(piece: int, side: str) -> bool:
    """True if piece belongs to side ('RED' or 'BLACK')."""
    if side == "RED":
        return piece in (_RED, _RED_KING)
    return piece in (_BLACK, _BLACK_K)


def _check_direction(path: list[list[int]], board: list[list[int]], side: str) -> Optional[str]:
    """
    Check the first diagonal step for direction violations.
    Returns an error string or None if OK.
    """
    r0, c0 = path[0]
    r1, c1 = path[1]
    if not _on_board(r0, c0):
        return "from_square_off_board"
    piece = _piece_at(board, r0, c0)
    dr = r1 - r0
    if piece == _RED and dr > 0:          # RED man moving down
        return f"red_man_moves_down (Δrow=+{dr})"
    if piece == _BLACK and dr < 0:        # BLACK man moving up
        return f"black_man_moves_up (Δrow={dr})"
    return None


# ---------------------------------------------------------------------------
# Subcategorise why a path is illegal
# ---------------------------------------------------------------------------

def _subcategorize(
    path: list[list[int]],
    board: list[list[int]],
    side: str,
    hidden_legal_moves: list[dict],
) -> tuple[str, Optional[str], bool, bool]:
    """
    Classify why `path` is not in hidden_legal_moves.

    Returns
    -------
    (illegal_move_type, wrong_direction_detail, mandatory_violation, multi_jump_incomplete)
    """
    r0, c0 = path[0]

    # 1. From-square validity
    if not _on_board(r0, c0):
        return "wrong_piece_square", None, False, False
    piece = _piece_at(board, r0, c0)
    if not _is_friendly(piece, side):
        return "wrong_piece_square", None, False, False

    # 2. To-square validity (last step)
    r1, c1 = path[-1]
    if not _on_board(r1, c1):
        return "invalid_destination", None, False, False
    if not _is_dark(r1, c1):
        return "invalid_destination", None, False, False
    if _piece_at(board, r1, c1) != _EMPTY:
        return "invalid_destination", None, False, False

    # 3. Direction check (step between first two waypoints)
    dir_err = _check_direction(path, board, side)
    if dir_err:
        return "wrong_direction", dir_err, False, False

    # 4. Mandatory capture violation
    has_jumps = any(m.get("type") == "jump" for m in hidden_legal_moves)
    is_simple_attempt = len(path) == 2 and abs(path[1][0] - path[0][0]) == 1
    if has_jumps and is_simple_attempt:
        return "mandatory_capture_violation", None, True, False

    # 5. Multi-jump incomplete: path prefix matches a legal jump but is shorter
    hidden_jump_prefixes: dict[tuple, int] = {}
    for m in hidden_legal_moves:
        if m.get("type") == "jump":
            p = _norm_path(m.get("path"))
            if p and len(p) >= 3:
                prefix = (_path_key([p[0], p[1]]),)
                hidden_jump_prefixes[_path_key([p[0], p[1]])] = len(p)

    if len(path) >= 2:
        prefix_key = _path_key([path[0], path[1]])
        if prefix_key in hidden_jump_prefixes and len(path) < hidden_jump_prefixes[prefix_key]:
            return "multi_jump_incomplete", None, False, True

    return "path_not_in_legal_moves", None, False, False


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_scenario(
    raw_output: str,
    hidden_legal_moves: list[dict],
    board: list[list[int]],
    side_to_move: str = "RED",
) -> dict[str, Any]:
    """
    Compare the LLM's selected_move against hidden_legal_moves ground truth.

    Parameters
    ----------
    raw_output         : LLM's raw response string.
    hidden_legal_moves : Symbolic engine ground truth — NEVER shown to LLM.
    board              : 8×8 int board.
    side_to_move       : "RED" or "BLACK".

    Returns
    -------
    Per-scenario metric dict.
    """
    n_legal = len(hidden_legal_moves)

    # Build ground-truth key set
    hidden_keys: set[tuple] = set()
    for m in hidden_legal_moves:
        p = _norm_path(m.get("path"))
        if p is not None:
            hidden_keys.add(_path_key(p))

    # -- Parse
    parsed, parse_error = parse_llm_output(raw_output)
    reasoning = (parsed or {}).get("reasoning", "") if parsed else ""

    if parsed is None:
        return _fail_record("parse_failed", parse_error, reasoning,
                            n_legal, raw_output, parsed)

    # -- Extract selected_move
    raw_move = parsed.get("selected_move")
    path = _norm_path(raw_move)

    if path is None:
        return _fail_record("parse_failed", "selected_move missing or malformed",
                            reasoning, n_legal, raw_output, parsed)

    # -- Check legality
    key = _path_key(path)
    if key in hidden_keys:
        return {
            "parse_success":          True,
            "parse_error":            "",
            "legal":                  True,
            "illegal_move_type":      "",
            "wrong_direction":        None,
            "mandatory_violation":    False,
            "multi_jump_incomplete":  False,
            "selected_path":          path,
            "n_legal":                n_legal,
            "raw_output":             raw_output,
            "parsed_output":          parsed,
            "reasoning":              reasoning,
        }

    # -- Illegal: classify why
    illegal_type, wrong_dir, mand_viol, multi_inc = _subcategorize(
        path, board, side_to_move, hidden_legal_moves
    )
    return {
        "parse_success":          True,
        "parse_error":            "",
        "legal":                  False,
        "illegal_move_type":      illegal_type,
        "wrong_direction":        wrong_dir,
        "mandatory_violation":    mand_viol,
        "multi_jump_incomplete":  multi_inc,
        "selected_path":          path,
        "n_legal":                n_legal,
        "raw_output":             raw_output,
        "parsed_output":          parsed,
        "reasoning":              reasoning,
    }


def _fail_record(
    illegal_type: str,
    parse_error: str,
    reasoning: str,
    n_legal: int,
    raw_output: str,
    parsed: Optional[dict],
) -> dict[str, Any]:
    return {
        "parse_success":          False,
        "parse_error":            parse_error,
        "legal":                  False,
        "illegal_move_type":      illegal_type,
        "wrong_direction":        None,
        "mandatory_violation":    False,
        "multi_jump_incomplete":  False,
        "selected_path":          None,
        "n_legal":                n_legal,
        "raw_output":             raw_output,
        "parsed_output":          parsed,
        "reasoning":              reasoning,
    }
