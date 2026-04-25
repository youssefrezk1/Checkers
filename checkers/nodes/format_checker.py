# nodes/format_checker.py
#
# Runs after proposal_agent and before validator.
#
# The proposal LLM outputs only JSON indices into get_all_legal_moves (see
# FORMAT_FEEDBACK_INDICES). This node parses that payload, expands indices to
# engine move dicts (ground-truth paths/captures), dedupes, enforces 3–5
# candidates when the engine has ≥3 moves, and tracks format_error_count.
#
# Alternatively accepts a list of ints (tests) or list of dicts that exactly
# match engine paths (programmatic use).
#
# Strategic legality is still checked in validator (enrichment, dedupe).

from __future__ import annotations

import copy
import json
import re
from checkers.state.state import CheckersState
from checkers.engine.board import (
    BLACK,
    RED,
    BLACK_KING,
    RED_KING,
    in_bounds,
    is_own_piece,
)
from checkers.engine.rules import get_all_legal_moves
from checkers.nodes.validator import _moves_match


# Required fields every proposed move must have
REQUIRED_FIELDS = ["type", "from", "to", "path", "captured", "piece_type"]

# Valid move types
VALID_TYPES = ["simple", "jump"]

# Proposal LLM outputs indices into the engine legal-move list only (no coordinates).
FORMAT_FEEDBACK_INDICES = (
    "FORMAT_ERROR: output must be a JSON object with a single key \"selected_indices\".\n"
    "Value: a JSON array of distinct integers in 0..N-1 where N is the number of "
    "legal moves listed in the prompt (0-based indexing).\n"
    "Pick between 3 and 5 indices when N >= 3; if N is 1 or 2, pick that many distinct valid indices.\n"
    "Example: {\"selected_indices\": [2, 0, 4]}\n"
    "No markdown fences, no commentary, no move objects — indices only.\n"
    "Alternatively a bare JSON array of integers is accepted: [2, 0, 4]"
)


def _strip_markdown(raw):
    """
    Strips markdown code blocks from LLM output.
    LLMs often wrap JSON in ```json ... ``` blocks.
    Returns the cleaned string.
    """
    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    return raw


def _sanitize_json_noise(text: str) -> str:
    """
    Remove junk tokens LLMs inject between brackets (e.g. ]正当] -> ]]).
    Only CJK / fullwidth punctuation between ] and ] — avoids eating ,"piece_type".
    """
    if not text:
        return text
    out = text
    # CJK Unified Ideographs + common CJK punctuation (no ASCII comma/quotes).
    junk_between = r"\]([\u4e00-\u9fff\u3000-\u303f\uff01-\uff60]+)\]"
    for _ in range(8):
        next_out = re.sub(junk_between, "]]", out)
        if next_out == out:
            break
        out = next_out
    return out


def _extract_outer_json_array(text: str) -> str:
    """
    Take the first balanced [...] slice, ignoring brackets inside JSON strings.
    Stops models from breaking parse when they append prose after the array.
    """
    text = text.strip()
    start = text.find("[")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _extract_outer_json_object(text: str) -> str | None:
    """First balanced {...} slice, ignoring braces inside JSON strings."""
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_selected_indices_payload(raw: str) -> tuple[dict[str, object] | None, str | None, bool]:
    """
    Parse LLM output into {"kind": "indices", "indices": list[int]}.
    Returns (bundle, error_message, parse_intervention).
    """
    if not isinstance(raw, str):
        return None, "proposed_moves must be a string from proposal_agent", False

    stripped_in = raw.strip()
    after_md = _strip_markdown(raw)
    parse_intervention = stripped_in != after_md
    text = after_md.strip()

    val: object | None = None
    try:
        val = json.loads(text)
    except json.JSONDecodeError:
        obj_slice = _extract_outer_json_object(text)
        if obj_slice:
            parse_intervention = True
            try:
                val = json.loads(obj_slice)
            except json.JSONDecodeError as e:
                return None, f"JSON decode detail: {e.msg} (char {e.pos})", parse_intervention
        else:
            arr_slice = _extract_outer_json_array(text)
            parse_intervention = True
            try:
                val = json.loads(arr_slice)
            except json.JSONDecodeError as e:
                return None, f"JSON decode detail: {e.msg} (char {e.pos})", parse_intervention

    indices_src: object | None = None
    if isinstance(val, dict):
        for key in ("selected_indices", "selectedIndices", "indices", "choices"):
            if key in val:
                indices_src = val[key]
                break
        if indices_src is None:
            return None, "object must contain key \"selected_indices\" (array of integers)", parse_intervention
    elif isinstance(val, list):
        indices_src = val
    else:
        return None, "top-level JSON must be an object or an array of integers", parse_intervention

    if not isinstance(indices_src, list):
        return None, "selected_indices must be a JSON array", parse_intervention

    out: list[int] = []
    for x in indices_src:
        if isinstance(x, bool):
            return None, "boolean values are not valid indices", parse_intervention
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, float) and x == int(x) and abs(x) < 1e9:
            out.append(int(x))
        else:
            return None, f"non-integer index: {x!r}", parse_intervention

    return {"kind": "indices", "indices": out}, None, parse_intervention


def _expand_indices_to_engine_moves(indices: list[int], legal: list[dict]) -> list[dict]:
    """Map 0-based indices to deep copies of engine moves; dedupe by path; skip out-of-range."""
    n = len(legal)
    seen_i: set[int] = set()
    seen_path: set[tuple[tuple[int, int], ...]] = set()
    out: list[dict] = []
    for i in indices:
        if i < 0 or i >= n or i in seen_i:
            continue
        seen_i.add(i)
        m = copy.deepcopy(legal[i])
        m["path"] = [list(sq) for sq in m["path"]]
        m["captured"] = [list(sq) for sq in m["captured"]]
        key = tuple((int(s[0]), int(s[1])) for s in m["path"])
        if key in seen_path:
            continue
        seen_path.add(key)
        out.append(m)
    return out


def _finalize_cleaned_moves(
    *,
    cleaned_moves: list[dict],
    board: list[list[int]],
    current_player: int,
    state: CheckersState,
    format_error_count: int,
    parse_intervention: bool,
    index_extra_intervention: bool,
    failure_reasons: list[str],
    any_repaired: bool,
    flatten_used: bool,
) -> dict:
    """Shared tail: min/max counts, insufficient_proposals, return patch dict."""
    engine_legal_n = len(get_all_legal_moves(board, current_player))
    # Proposal count rule: min(5, n_legal) — never require more than what exists.
    n_to_propose = min(5, engine_legal_n)
    max_proposals_forward = n_to_propose
    min_proposals_required = n_to_propose

    if len(cleaned_moves) > 0:
        truncated = False
        if len(cleaned_moves) > max_proposals_forward:
            cleaned_moves = cleaned_moves[:max_proposals_forward]
            truncated = True

        if len(cleaned_moves) < min_proposals_required and engine_legal_n >= min_proposals_required:
            feedback = (
                "INSUFFICIENT_PROPOSAL_COUNT: only "
                f"{len(cleaned_moves)} distinct valid index(es) resolved, but at least "
                f"{min_proposals_required} are required when the engine lists "
                f"{engine_legal_n} legal move(s).\n"
                f"Output {{\"selected_indices\": [...]}} with between {min_proposals_required} and "
                f"{max_proposals_forward} distinct integers from 0 to {engine_legal_n - 1}."
            )
            bump = 1 + (1 if parse_intervention else 0)
            return {
                "proposed_moves": [],
                "feedback": feedback,
                "retry_count": state.retry_count + 1,
                "insufficient_proposals": True,
                "format_error_count": format_error_count + bump,
                "last_completed_node": "format_checker",
            }

        intervention = (
            any_repaired
            or len(failure_reasons) > 0
            or flatten_used
            or truncated
            or parse_intervention
            or index_extra_intervention
        )
        insufficient = (
            len(cleaned_moves) < min_proposals_required
            and engine_legal_n >= min_proposals_required
        )
        return {
            "proposed_moves": cleaned_moves,
            "feedback": None,
            "insufficient_proposals": insufficient,
            "format_error_count": format_error_count + (1 if intervention else 0),
            "last_completed_node": "format_checker",
        }

    unique_reasons = list(set(failure_reasons))
    extra = "\n" + "\n".join(unique_reasons) if unique_reasons else ""
    bump = 1 + (1 if parse_intervention else 0)
    return {
        "proposed_moves": [],
        "feedback": FORMAT_FEEDBACK_INDICES + extra,
        "retry_count": state.retry_count + 1,
        "format_error_count": format_error_count + bump,
        "last_completed_node": "format_checker",
    }


def _parse_json(raw):
    """
    Parse raw LLM output as JSON.

    Returns (parsed, decode_error_message, parse_intervention).
    - If raw is not a str, returns (raw, None, False).
    - parse_intervention: True if markdown strip, CJK sanitize, or outer-array
      extraction changed the text before a successful parse (thesis metrics).

    LLMs often omit closing bracket(s); we retry with extra `]`.
    """
    if not isinstance(raw, str):
        return raw, None, False

    stripped_in = raw.strip()
    after_md = _strip_markdown(raw)
    after_sanitize = _sanitize_json_noise(after_md)
    sliced = _extract_outer_json_array(after_sanitize)

    parse_intervention = (
        stripped_in != after_md
        or after_md != after_sanitize
        or after_sanitize != sliced
    )

    last_err: str | None = None
    for extra in range(0, 4):
        candidate = sliced + ("]" * extra)
        try:
            return json.loads(candidate), None, parse_intervention
        except json.JSONDecodeError as e:
            last_err = f"{e.msg} (char {e.pos})"
            continue
    return None, last_err or "unknown JSON decode error", parse_intervention


def _wrap_if_single(parsed):
    """
    If the LLM returned a single move object instead of a list,
    wrap it in a list so downstream processing is consistent.
    """
    if isinstance(parsed, dict):
        return [parsed]
    return parsed


def _flatten_llm_move_list(moves):
    """
    Normalize LLM output to a flat list of move dicts.

    Models often emit:
      [[{a}], [{b},{c}]]     — one level of wrappers
      [[[{a}]], [{b}]]       — deeper nesting (single-pass flatten used to keep lists → empty downstream)
    We DFS any list and collect dict leaves only.

    Returns (flat_list, changed: bool).
    """
    if not isinstance(moves, list):
        return moves, False

    out: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            out.append(node)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(moves)
    changed = any(isinstance(x, list) for x in moves) or (len(out) != len(moves))
    return out, changed


def _fix_coordinates(value):
    """
    Attempts to convert coordinate values to integers.
    Handles string digits, floats, and nested lists.
    Returns fixed value or None if unfixable.
    """
    if isinstance(value, list):
        fixed = []
        for item in value:
            if isinstance(item, list):
                inner = _fix_coordinates(item)
                if inner is None:
                    return None
                fixed.append(inner)
            else:
                try:
                    fixed.append(int(item))
                except (ValueError, TypeError):
                    return None
        return fixed

    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _validate_and_repair_move(move, board, current_player):
    """
    Attempts to validate and auto-repair a single proposed move.

    Repair attempts (silent, no feedback):
        - Convert string coordinates to integers
        - Convert float coordinates to integers
        - Lowercase the type field

    Hard failures (move is discarded):
        - Missing required fields that cannot be inferred
        - type is not simple or jump after lowercasing
        - from or to coordinates out of 0-7 range
        - path is not a list
        - captured is not a list
        - No own piece at the from square
        - piece_type is not regular or king
        - piece_type does not match the actual piece at from (man vs king)

    Returns (repaired_move, was_repaired, failure_reason)
        repaired_move   : the cleaned move dict or None if unsalvageable
        was_repaired    : True if any auto-repair was applied
        failure_reason  : string describing why it failed or None
    """
    if not isinstance(move, dict):
        return None, False, "INVALID_STRUCTURE: move is not a JSON object"

    was_repaired = False
    repaired = {}

    # Check all required fields exist
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in move:
            missing.append(field)
    if len(missing) > 0:
        return None, False, f"MISSING_FIELDS: {missing}"

    # Validate and repair type field
    move_type = move["type"]
    if isinstance(move_type, str):
        move_type = move_type.lower().strip()
        if move_type != move["type"]:
            was_repaired = True
    if move_type not in VALID_TYPES:
        return None, False, f"INVALID_VALUE: type must be simple or jump, got {move_type}"
    repaired["type"] = move_type

    # Validate and repair from field
    from_fixed = _fix_coordinates(move["from"])
    if from_fixed is None or len(from_fixed) != 2:
        return None, False, "INVALID_TYPE: from must be a list of 2 integers"
    if not in_bounds(from_fixed[0], from_fixed[1]):
        return None, False, f"BOUNDARY: from {from_fixed} is outside the board"
    if from_fixed != move["from"]:
        was_repaired = True
    repaired["from"] = from_fixed

    # Validate and repair to field
    to_fixed = _fix_coordinates(move["to"])
    if to_fixed is None or len(to_fixed) != 2:
        return None, False, "INVALID_TYPE: to must be a list of 2 integers"
    if not in_bounds(to_fixed[0], to_fixed[1]):
        return None, False, f"BOUNDARY: to {to_fixed} is outside the board"
    if to_fixed != move["to"]:
        was_repaired = True
    repaired["to"] = to_fixed

    # Validate and repair path field
    path_fixed = _fix_coordinates(move["path"])
    if path_fixed is None or not isinstance(path_fixed, list) or len(path_fixed) < 2:
        return None, False, "INVALID_TYPE: path must be a list of at least 2 coordinate pairs"
    if path_fixed != move["path"]:
        was_repaired = True
    repaired["path"] = path_fixed

    # Validate and repair captured field
    captured = move["captured"]
    if not isinstance(captured, list):
        return None, False, "INVALID_TYPE: captured must be a list"
    captured_fixed = _fix_coordinates(captured) if len(captured) > 0 else []
    if captured_fixed is None:
        return None, False, "INVALID_TYPE: captured contains invalid coordinates"
    if captured_fixed != captured:
        was_repaired = True
    repaired["captured"] = captured_fixed

    # Validate piece_type field
    piece_type = move["piece_type"]
    if isinstance(piece_type, str):
        piece_type = piece_type.lower().strip()
        if piece_type != move["piece_type"]:
            was_repaired = True
    if piece_type not in ["regular", "king"]:
        return None, False, f"INVALID_VALUE: piece_type must be regular or king, got {piece_type}"
    repaired["piece_type"] = piece_type

    # Check own piece exists at from square
    from_row = repaired["from"][0]
    from_col = repaired["from"][1]
    piece_at_from = board[from_row][from_col]
    if not is_own_piece(piece_at_from, current_player):
        return None, False, f"PIECE_MISMATCH: no own piece at {repaired['from']}"

    if piece_at_from in (RED_KING, BLACK_KING):
        expected_piece_type = "king"
    elif piece_at_from in (RED, BLACK):
        expected_piece_type = "regular"
    else:
        return None, False, f"PIECE_MISMATCH: unexpected piece code at {repaired['from']}"

    if piece_type != expected_piece_type:
        return (
            None,
            False,
            "PIECE_TYPE_MISMATCH: "
            f"square {repaired['from']} has a {expected_piece_type} "
            f"but piece_type was {piece_type!r}",
        )

    return repaired, was_repaired, None


def format_checker(state: CheckersState) -> dict:
    """
    Parses index-based LLM output and expands to engine move dicts, or accepts
    a list of dicts that exactly match engine paths (tests / programmatic use).
    """

    board = state.board
    current_player = state.current_player
    proposed_moves = state.proposed_moves
    format_error_count = state.format_error_count
    parse_intervention = False
    index_extra_intervention = False
    any_repaired = False
    flatten_used = False
    failure_reasons: list[str] = []

    # Full legal move list (always used for n_legal feedback and final validator)
    legal_basis = get_all_legal_moves(board, current_player)
    n_legal = len(legal_basis)

    # ── Phase 8: when symbolic_scored_moves is present, proposal indices map
    # into the scored list (which is the full legal list, sorted best-first).
    # index 0 = best symbolic move. Validator still checks all expanded moves
    # against the full legal list from get_all_legal_moves.
    if state.symbolic_scored_moves:
        scored_basis = [entry["move"] for entry in state.symbolic_scored_moves]
        expansion_basis = scored_basis
        expansion_n = len(scored_basis)
    else:
        expansion_basis = legal_basis
        expansion_n = n_legal

    if isinstance(proposed_moves, list) and len(proposed_moves) == 0:
        return {
            "proposed_moves": [],
            "feedback": (
                "No proposal output (empty list). Respond with JSON only, e.g. "
                '{"selected_indices": [0, 2, 4]} — see FORMAT_ERROR in prior message.'
            ),
            "retry_count": state.retry_count + 1,
            "format_error_count": format_error_count + 1,
            "last_completed_node": "format_checker",
        }

    # ── A) Raw string: {"selected_indices": [...]} or [...] ints ─────────
    if isinstance(proposed_moves, str):
        bundle, err, parse_intervention = _parse_selected_indices_payload(proposed_moves)
        if bundle is None:
            bump = 1 + (1 if parse_intervention else 0)
            detail = f"\n{err}" if err else ""
            return {
                "proposed_moves": [],
                "feedback": FORMAT_FEEDBACK_INDICES + detail,
                "retry_count": state.retry_count + 1,
                "format_error_count": format_error_count + bump,
                "last_completed_node": "format_checker",
            }
        raw_indices = bundle["indices"]
        assert isinstance(raw_indices, list)
        for i in raw_indices:
            if isinstance(i, int) and (i < 0 or i >= expansion_n):
                index_extra_intervention = True
        cleaned_moves = _expand_indices_to_engine_moves(raw_indices, expansion_basis)
        if len(raw_indices) > 0 and len(cleaned_moves) == 0:
            bump = 1 + (1 if parse_intervention else 0)
            hi = max(0, expansion_n - 1)
            return {
                "proposed_moves": [],
                "feedback": (
                    f"Every index was out of range. This position has {n_legal} legal move(s); "
                    f"use only integers from 0 to {hi} inclusive."
                ),
                "retry_count": state.retry_count + 1,
                "format_error_count": format_error_count + bump,
                "last_completed_node": "format_checker",
            }
        return _finalize_cleaned_moves(
            cleaned_moves=cleaned_moves,
            board=board,
            current_player=current_player,
            state=state,
            format_error_count=format_error_count,
            parse_intervention=parse_intervention,
            index_extra_intervention=index_extra_intervention,
            failure_reasons=failure_reasons,
            any_repaired=any_repaired,
            flatten_used=flatten_used,
        )

    # ── B) List of ints (tests) ────────────────────────────────────────
    if isinstance(proposed_moves, list) and len(proposed_moves) > 0:
        if all(isinstance(x, int) and not isinstance(x, bool) for x in proposed_moves):
            for i in proposed_moves:
                if i < 0 or i >= expansion_n:
                    index_extra_intervention = True
            cleaned_moves = _expand_indices_to_engine_moves(list(proposed_moves), expansion_basis)
            return _finalize_cleaned_moves(
                cleaned_moves=cleaned_moves,
                board=board,
                current_player=current_player,
                state=state,
                format_error_count=format_error_count,
                parse_intervention=False,
                index_extra_intervention=index_extra_intervention,
                failure_reasons=failure_reasons,
                any_repaired=any_repaired,
                flatten_used=flatten_used,
            )

        # ── C) List of dicts: must match engine moves exactly (path/type) ─
        if all(isinstance(x, dict) for x in proposed_moves):
            cleaned_moves = []
            for move in proposed_moves:
                hit = next((e for e in legal_basis if _moves_match(move, e)), None)
                if hit is None:
                    bump = 1
                    return {
                        "proposed_moves": [],
                        "feedback": (
                            FORMAT_FEEDBACK_INDICES
                            + "\nOne or more move objects do not match the engine legal list — "
                            "use JSON indices only, not hand-written paths."
                        ),
                        "retry_count": state.retry_count + 1,
                        "format_error_count": format_error_count + bump,
                        "last_completed_node": "format_checker",
                    }
                mc = copy.deepcopy(hit)
                mc["path"] = [list(sq) for sq in mc["path"]]
                mc["captured"] = [list(sq) for sq in mc["captured"]]
                cleaned_moves.append(mc)
            # Dedupe paths (same as index expansion)
            seen_path: set[tuple[tuple[int, int], ...]] = set()
            deduped: list[dict] = []
            for m in cleaned_moves:
                key = tuple((int(s[0]), int(s[1])) for s in m["path"])
                if key not in seen_path:
                    seen_path.add(key)
                    deduped.append(m)
            cleaned_moves = deduped
            return _finalize_cleaned_moves(
                cleaned_moves=cleaned_moves,
                board=board,
                current_player=current_player,
                state=state,
                format_error_count=format_error_count,
                parse_intervention=False,
                index_extra_intervention=False,
                failure_reasons=failure_reasons,
                any_repaired=any_repaired,
                flatten_used=flatten_used,
            )

    bump = 1
    return {
        "proposed_moves": [],
        "feedback": FORMAT_FEEDBACK_INDICES + f"\nUnexpected proposed_moves type: {type(proposed_moves).__name__}.",
        "retry_count": state.retry_count + 1,
        "format_error_count": format_error_count + bump,
        "last_completed_node": "format_checker",
    }
