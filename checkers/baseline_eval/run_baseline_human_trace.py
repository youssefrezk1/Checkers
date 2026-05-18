#!/usr/bin/env python3
"""
checkers/baseline_eval/run_baseline_human_trace.py

Interactive evaluation runner: RED = selected baseline, BLACK = human terminal input.

Usage:
    venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline minimal_raw_llm
    venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline rules_only_llm
    venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline legal_moves_index_llm
    venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline legal_moves_path_llm
    venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline full_system

Baselines:
    minimal_raw_llm       — board + piece legend + JSON format ONLY.
                            No movement rules, no legality hints, no path encoding guide.
                            Measures pure board-reading + move-generation capability.

    rules_only_llm        — board + full written checkers rules + JSON format.
                            No legal move list, no scores, no evaluator output.
                            Measures whether written rules reduce hallucination.

    legal_moves_index_llm — board + enumerated legal moves (indices, no scores) + JSON.
                            Output: chosen_index. No minimax, no facts, no candidates.
                            Measures strategic selection from legal candidates.

    legal_moves_path_llm  — board + enumerated legal moves with explicit path coords + JSON.
                            Output: move_path. No minimax, no facts, no candidates.
                            Measures whether path copying eliminates coordinate errors.

    full_system           — existing neuro-symbolic pipeline unchanged:
                            scorer_node → deterministic_proposal_node → ranker_agent → update_agent

Auto-correction (minimal_raw_llm, rules_only_llm, legal_moves_index_llm, legal_moves_path_llm ONLY):
    On invalid JSON / illegal move / invalid index / mandatory capture violation /
    hallucinated path:
      1. Log exact raw output
      2. Log attempted move / index / path
      3. Log legality_error_reason
      4. Apply minimax-best legal move automatically
      5. Mark correction_used=True, correction_source="minimax_best_legal"
    Does NOT silently fix. Does NOT count corrected move as LLM success.

    full_system NEVER uses this external autocorrect.  Its internal
    retry / override / fallback architecture handles all failures.

Outputs (written at game end):
    logs/baseline_eval/<baseline>_<game_id>.jsonl          — per-turn trace
    logs/baseline_eval/<baseline>_<game_id>_metrics.csv    — aggregate CSV
    logs/baseline_eval/<baseline>_<game_id>_summary.md     — Markdown summary

Debug flags:
    --show-prompts   Print exact system + user prompts before each LLM call.
    --show-table     Print the baseline information-visibility comparison table.
"""

from __future__ import annotations

# ── Force simplified pipeline env before any graph import ────────────────────
import os
os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import argparse
import csv
import json
import re
import sys
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from checkers.engine.board import (
    RED, BLACK, EMPTY, RED_KING, BLACK_KING,
    create_initial_board, print_board,
)
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves
from checkers.state.state import CheckersState
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.update_agent import update_agent as _update_agent_fn

# ── Isolated baseline LLM client ──────────────────────────────────────────────
# Uses BASELINE_MISTRAL_* env vars exclusively.
# Never reads MISTRAL_API_KEY — that key belongs to the neuro-symbolic pipeline.

import json as _json_b
import urllib.error as _urllib_error_b
import urllib.request as _urllib_request_b

_BASELINE_API_KEY  = os.environ.get("BASELINE_MISTRAL_API_KEY", "")
_BASELINE_MODEL    = os.environ.get("BASELINE_MISTRAL_MODEL", "mistral-large-latest")
_BASELINE_TEMP     = float(os.environ.get("BASELINE_MISTRAL_TEMPERATURE", "0.2"))
_BASELINE_MAX_TOK  = int(os.environ.get("BASELINE_MISTRAL_MAX_TOKENS", "1200"))
_BASELINE_API_URL  = "https://api.mistral.ai/v1/chat/completions"


def call_baseline_llm(system: str, user: str) -> str:
    """
    Isolated LLM client for baseline evaluation.

    Reads BASELINE_MISTRAL_API_KEY / BASELINE_MISTRAL_MODEL /
    BASELINE_MISTRAL_TEMPERATURE / BASELINE_MISTRAL_MAX_TOKENS.
    Never touches the main MISTRAL_API_KEY used by the game pipeline.

    Raises:
        RuntimeError -- BASELINE_MISTRAL_API_KEY not set.
        ValueError   -- Non-200 API response or bad response structure.
        OSError      -- Network-level failure.
    """
    if not _BASELINE_API_KEY:
        raise RuntimeError(
            "BASELINE_MISTRAL_API_KEY is not set. "
            "Add it to your .env file before running baseline evaluations. "
            "This key is intentionally separate from the main MISTRAL_API_KEY "
            "so that baseline runs never consume pipeline quota."
        )

    payload: dict = {
        "model":           _BASELINE_MODEL,
        "temperature":     _BASELINE_TEMP,
        "max_tokens":      _BASELINE_MAX_TOK,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    body = _json_b.dumps(payload).encode("utf-8")
    req  = _urllib_request_b.Request(
        _BASELINE_API_URL,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {_BASELINE_API_KEY}",
            "Accept":        "application/json",
        },
        method="POST",
    )

    import time as _time_b
    import sys as _sys_b
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            with _urllib_request_b.urlopen(req, timeout=60.0) as resp:
                data = _json_b.loads(resp.read().decode("utf-8"))
                break
        except _urllib_error_b.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_attempts - 1:
                sleep_sec = 10 * (2 ** attempt)
                print(f"[baseline_eval] HTTP 429 Rate Limit. Attempt {attempt + 1} failed. Sleeping {sleep_sec}s...", file=_sys_b.stderr)
                _time_b.sleep(sleep_sec)
                continue
            raise ValueError(
                f"Baseline Mistral API HTTP {exc.code}: {body_text[:300]}"
            ) from exc
        except _urllib_error_b.URLError as exc:
            if attempt < max_attempts - 1:
                print(f"[baseline_eval] Network Error. Attempt {attempt + 1} failed. Sleeping 5s...", file=_sys_b.stderr)
                _time_b.sleep(5)
                continue
            raise ValueError(f"Baseline Mistral Network Error: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Unexpected baseline Mistral response: {str(data)[:300]}"
        ) from exc

    if not isinstance(content, str):
        raise ValueError(f"Baseline Mistral content is not a string: {type(content)}")

    return content

# ── Constants ─────────────────────────────────────────────────────────────────

# ── Active thesis baseline names ─────────────────────────────────────────────
BASELINE_MINIMAL_RAW_LLM        = "minimal_raw_llm"             # B1
BASELINE_RULES_ONLY_LLM         = "rules_only_llm"              # B2
BASELINE_FULL_SYSTEM             = "full_system"                  # B4

# Legacy ablation constants — NOT in any active suite run.
BASELINE_LEGAL_MOVES_INDEX_LLM = "legal_moves_index_llm"
BASELINE_LEGAL_MOVES_PATH_LLM  = "legal_moves_path_llm"

# Active thesis arms in canonical order (B1 → B2 → B4).
ALL_BASELINES = (
    BASELINE_MINIMAL_RAW_LLM,
    BASELINE_RULES_ONLY_LLM,
    BASELINE_FULL_SYSTEM,
)

# Baselines that output move_path JSON (all plain-LLM variants)
_PATH_JSON_BASELINES = (
    BASELINE_MINIMAL_RAW_LLM,
    BASELINE_RULES_ONLY_LLM,
)

BAR  = "═" * 60
RULE = "─" * 60

_ANSI_RED   = "\033[91m"
_ANSI_YLW   = "\033[93m"
_ANSI_GRN   = "\033[92m"
_ANSI_RESET = "\033[0m"


# ── Formatting helpers ────────────────────────────────────────────────────────

def _red_if(val: Any, cond: bool) -> str:
    return f"{_ANSI_RED}{val}{_ANSI_RESET}" if cond else str(val)


def _ylw(val: Any) -> str:
    return f"{_ANSI_YLW}{val}{_ANSI_RESET}"


def _player_label(p: int) -> str:
    return "RED" if p == RED else "BLACK"


def _fmt_engine_move(i: int, m: dict[str, Any]) -> str:
    cap = m.get("captured", [])
    cap_s = f"  captures={cap}" if cap else ""
    return f"  [{i}] {m.get('type')} path={m.get('path')}{cap_s}"


def _fmt_scored_move(i: int, m: dict[str, Any]) -> str:
    facts = m.get("facts") or {}
    score = facts.get("minimax_score", "n/a")
    rank  = facts.get("symbolic_rank", "?")
    cap   = m.get("captured") or []
    cap_s = f"  captures={cap}" if cap else ""
    return (
        f"  [{i}] rank={rank} score={score:>8}  "
        f"{m.get('type')} {m.get('path')}{cap_s}"
    )


def _wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    return textwrap.fill(
        text, width=width,
        initial_indent=indent, subsequent_indent=indent,
        break_long_words=False, break_on_hyphens=False,
    )


# ── Board renderer for LLM prompts ───────────────────────────────────────────

def _render_board_text(board: list[list[int]]) -> str:
    """Compact text board suitable for LLM prompts."""
    symbols = {EMPTY: ".", RED: "r", BLACK: "b", RED_KING: "R", BLACK_KING: "B"}
    lines = ["  " + " ".join(str(c) for c in range(8))]
    for row in range(8):
        row_str = str(row) + " " + " ".join(symbols[board[row][col]] for col in range(8))
        lines.append(row_str)
    return "\n".join(lines)


def _fmt_move_for_prompt(m: dict[str, Any]) -> str:
    """Human-readable move description for legal_moves_index_llm user prompt (no scores)."""
    mtype = m.get("type", "?")
    path  = m.get("path", [])
    cap   = m.get("captured", [])
    if mtype == "simple":
        return f"simple: {path[0]} → {path[-1]}" if len(path) >= 2 else f"simple: {path}"
    parts = " → ".join(str(sq) for sq in path)
    cap_s = f"  (captures: {cap})" if cap else ""
    return f"jump: {parts}{cap_s}"


# ── Baseline information-visibility comparison table ─────────────────────────

_BASELINE_COMPARISON_TABLE = """\
BASELINE INFORMATION VISIBILITY
{bar}

 Information received by LLM               min_raw  rules_only  legal_idx  legal_path  full_system
 ──────────────────────────────────────── ──────── ─────────── ─────────  ──────────  ─────────────
 Board position                              ✓          ✓           ✓          ✓            ✓
 Piece legend (r / R / b / B / .)           ✓          ✓           ✓          ✓            ✓
 Movement rules (direction, distance)       ✗          ✓           ✗          ✗        internal
 Dark-square constraint (row+col odd)       ✗          ✓           ✗          ✗        internal
 Mandatory capture rule (explicit)          ✗          ✓        implicit   implicit    internal
 Multi-jump rule                            ✗          ✓           ✗          ✗        internal
 Promotion rule (RED man → king)            ✗          ✓           ✗          ✗        internal
 Path encoding guide (2 / 3+ entries)       ✗          ✓           ✗          ✗        internal
 Legal moves: index + description           ✗          ✗           ✓          ✗       shortlist
 Legal moves: index + explicit path         ✗          ✗           ✗          ✓       shortlist
 Output format                             path       path        index      path         path
 Move scores (minimax)                      ✗          ✗           ✗          ✗            ✓
 Symbolic rank                              ✗          ✗           ✗          ✗            ✓
 Move facts / evaluator output              ✗          ✗           ✗          ✗            ✓
 Strategic context (phase, score_state)     ✗          ✗           ✗          ✗            ✓
 Proposal candidates (shortlisted)          ✗          ✗           ✗          ✗            ✓
 Previous corrections / turn history        ✗          ✗           ✗          ✗            ✗

{bar}

Contamination notes:
  minimal_raw_llm       — Cleanest baseline. No legality guidance. Measures raw
                          board-reading + move-generation capability.
  rules_only_llm        — Full rules + path encoding guide provide structural scaffolding.
                          Measures rule-following vs pure generation.
  legal_moves_index_llm — Index-only selection. Cannot hallucinate coordinates.
                          Failure modes: out-of-range index, non-integer field.
  legal_moves_path_llm  — Sees legal moves with explicit path coords; must output a path.
                          Measures whether path copying eliminates coordinate errors vs
                          legal_moves_index_llm and vs rules_only_llm.
  full_system           — Reference architecture. Scores / facts / context are intentional.
""".format(bar="═" * 82)


# ── LLM system prompts ────────────────────────────────────────────────────────
#
# STRICT SEPARATION RULES:
#   minimal_raw_llm : piece legend + "you are RED" + JSON output format ONLY.
#                     No rules. No hints. No path encoding guide.
#   rules_only_llm  : full checkers rules + path encoding + JSON output format.
#                     No legal moves, no scores, no evaluator output.
#   legal_moves_index_llm : legal move list + JSON output format; output = chosen_index.
#                           No rules, no scores, no facts.
#   legal_moves_path_llm  : legal move list with explicit path coords + JSON output format;
#                           output = move_path. No rules, no scores, no facts.
#   full_system           : internal pipeline (ranker_agent.py builds its own prompts).

_MINIMAL_RAW_LLM_SYSTEM = """\
You are playing American Checkers (8x8) as the RED player.

PIECES:
  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty

You are RED.  Choose your next move.

Respond with valid JSON only, no prose before or after:
{
  "move_path": [[row, col], [row, col], ...],
  "reasoning": "brief explanation"
}\
"""

_RULES_ONLY_LLM_SYSTEM = """\
You are playing American Checkers (8x8) as the RED player.

PIECES:
  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty

MOVEMENT:
- Only dark squares are playable: (row + col) must be ODD.
- RED men move diagonally UP (toward row 0). BLACK men move DOWN (toward row 7).
- Kings (R or B) move diagonally in all four directions.

CAPTURES (jumps):
- Jump over an adjacent enemy piece onto the vacant square beyond it (2 diagonals).
- The jumped piece is immediately removed.
- MANDATORY CAPTURE RULE: If any jump exists, you MUST jump. No exceptions.
- Multi-jump: if the jumping piece can keep jumping after landing, it MUST continue
  in the same turn. List all landing squares in order.

PROMOTION:
- A RED man (r) reaching row 0 becomes a RED king (R).

Path encoding:
  Simple move  : [[r_from, c_from], [r_to, c_to]]                    (2 entries)
  Single jump  : [[r_from, c_from], [r_landing, c_landing]]          (2 entries)
  Multi-jump   : [[r_from, c_from], [land1], [land2], ...]           (3+ entries)
  (The captured piece is always at the midpoint between consecutive path entries.)

OUTPUT — respond with valid JSON only, no prose before or after:
{
  "move_path": [[row, col], [row, col], ...],
  "reasoning": "brief explanation"
}

IMPORTANT: Only move YOUR pieces (r or R). Output JSON only.\
"""

_LEGAL_MOVES_INDEX_LLM_SYSTEM = """\
You are playing American Checkers (8x8) as the RED player.

You will receive the current board and a numbered list of every legal move available
to you this turn.  All listed moves are valid.  Mandatory capture is already enforced:
if any jump exists, only jumps appear in the list.

Choose the move you believe is strategically strongest.

OUTPUT — respond with valid JSON only, no prose before or after:
{
  "chosen_index": <integer from 0 to N-1>,
  "reasoning": "brief explanation"
}

IMPORTANT: chosen_index must be an integer within the range shown.\
"""

_LEGAL_MOVES_PATH_LLM_SYSTEM = """\
You are playing American Checkers (8x8) as the RED player.

You will receive the current board and a numbered list of every legal move available
to you this turn.  Each entry shows the move type and the exact path coordinates.
All listed moves are valid.  Mandatory capture is already enforced:
if any jump exists, only jumps appear in the list.

Choose the move you believe is strategically strongest.
Your move_path must exactly match one of the paths shown in the list.

OUTPUT — respond with valid JSON only, no prose before or after:
{
  "move_path": [[row, col], [row, col], ...],
  "reasoning": "brief explanation"
}

IMPORTANT: move_path must be copied exactly from one of the listed paths. Output JSON only.\
"""


# ── LLM prompt user messages ──────────────────────────────────────────────────

def _build_path_json_user(board: list[list[int]], turn_number: int) -> str:
    """Shared user prompt for minimal_raw_llm and rules_only_llm (board + JSON instruction)."""
    return (
        f"Turn {turn_number} — You are RED.  Choose your move.\n\n"
        f"Board:\n{_render_board_text(board)}\n\n"
        'Respond with JSON: {"move_path": [[row,col],...], "reasoning": "..."}'
    )


def _build_legal_moves_index_llm_user(
    board: list[list[int]],
    legal_moves: list[dict[str, Any]],
    turn_number: int,
) -> str:
    """User prompt for legal_moves_index_llm: move list shows type+description, output=index."""
    moves_text = "\n".join(
        f"[{i}] {_fmt_move_for_prompt(m)}" for i, m in enumerate(legal_moves)
    )
    return (
        f"Turn {turn_number} — You are RED.  Choose your move.\n\n"
        f"Board:\n{_render_board_text(board)}\n\n"
        f"Legal moves ({len(legal_moves)} available):\n{moves_text}\n\n"
        'Respond with JSON: {"chosen_index": N, "reasoning": "..."}'
    )


def _build_legal_moves_path_llm_user(
    board: list[list[int]],
    legal_moves: list[dict[str, Any]],
    turn_number: int,
) -> str:
    """User prompt for legal_moves_path_llm: move list shows explicit path coords, output=path."""
    moves_text = "\n".join(
        f"[{i}] {m.get('type', '?')}  {m.get('path')}"
        for i, m in enumerate(legal_moves)
    )
    return (
        f"Turn {turn_number} — You are RED.  Choose your move.\n\n"
        f"Board:\n{_render_board_text(board)}\n\n"
        f"Legal moves ({len(legal_moves)} available):\n{moves_text}\n\n"
        '{"move_path": [[row,col],...], "reasoning": "..."}'
    )


# ── Move matching helpers ─────────────────────────────────────────────────────

def _norm_path(path: Any) -> list:
    """Normalise a path to list-of-[int,int] for equality checks."""
    if not isinstance(path, (list, tuple)):
        return []
    out: list = []
    for sq in path:
        if isinstance(sq, (list, tuple)) and len(sq) == 2:
            try:
                out.append([int(sq[0]), int(sq[1])])
            except (TypeError, ValueError):
                return []
        else:
            return []
    return out


def _find_move_by_path(
    legal_moves: list[dict[str, Any]],
    path: Any,
) -> Optional[dict[str, Any]]:
    """Return the legal move whose path matches path, or None."""
    norm = _norm_path(path)
    if not norm:
        return None
    for m in legal_moves:
        if _norm_path(m.get("path")) == norm:
            return m
    return None


def _has_mandatory_capture(legal_moves: list[dict[str, Any]]) -> bool:
    """True when the engine only returns jumps (mandatory capture enforced)."""
    return any(m.get("type") == "jump" for m in legal_moves)


# ── Scoring oracle helpers ────────────────────────────────────────────────────

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("-inf")


def _best_from_scored(scored: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the move with the highest minimax_score (scored_moves format)."""
    if not scored:
        return None
    return max(scored, key=lambda m: _safe_float((m.get("facts") or {}).get("minimax_score")))


def _rank_and_gap(
    scored: list[dict[str, Any]],
    chosen: dict[str, Any],
) -> tuple[int, float]:
    """
    Find symbolic_rank and score_gap of chosen in the scored list.
    scored has format: [{"type", "path", "captured", "facts": {...minimax_score, symbolic_rank...}}]
    Returns (rank, gap) where rank=0 means the move was not found.
    """
    if not scored or not chosen:
        return 0, float("inf")
    chosen_path = _norm_path(chosen.get("path"))
    best_score  = _safe_float((scored[0].get("facts") or {}).get("minimax_score"))
    for m in scored:
        if _norm_path(m.get("path")) == chosen_path:
            chosen_score = _safe_float((m.get("facts") or {}).get("minimax_score"))
            rank = int((m.get("facts") or {}).get("symbolic_rank", 0))
            gap  = round(best_score - chosen_score, 2)
            return rank, gap
    return 0, float("inf")


def _rank_and_gap_slim(
    symbolic_scored: list[dict[str, Any]],
    chosen: dict[str, Any],
) -> tuple[int, float]:
    """
    Like _rank_and_gap but for symbolic_scored_moves format:
    [{"move": slim_dict, "minimax_score": float, "rank": int}, ...]
    """
    if not symbolic_scored or not chosen:
        return 0, float("inf")
    chosen_path = _norm_path(chosen.get("path"))
    best_score  = _safe_float(symbolic_scored[0].get("minimax_score"))
    for entry in symbolic_scored:
        move_path = _norm_path((entry.get("move") or {}).get("path"))
        if move_path == chosen_path:
            rank         = int(entry.get("rank", 0))
            chosen_score = _safe_float(entry.get("minimax_score"))
            gap          = round(best_score - chosen_score, 2)
            return rank, gap
    return 0, float("inf")


def _enrich_from_scored(
    plain: dict[str, Any],
    scored: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the enriched version of plain (with facts) from scored, or plain itself."""
    path = _norm_path(plain.get("path"))
    for m in scored:
        if _norm_path(m.get("path")) == path:
            return m
    return plain


# ── JSON response parser ──────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> tuple[Optional[dict[str, Any]], str]:
    """
    Parse raw LLM string as a JSON dict.
    Returns (parsed_dict, error_reason) — error_reason is "" on success.
    """
    if not raw or not raw.strip():
        return None, "empty_response"
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("```")).strip()
    try:
        obj = json.loads(text)
        return (obj, "") if isinstance(obj, dict) else (None, "not_a_json_object")
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        end = text.rfind("}")
        if end > start:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    return obj, ""
            except json.JSONDecodeError:
                pass
    return None, "json_decode_error"


# ── Contradiction checker ─────────────────────────────────────────────────────

def _check_contradictions(reasoning: str, chosen: dict[str, Any]) -> Optional[str]:
    """
    Spot obvious contradictions between the LLM's reasoning and the move's facts.
    Returns a description string or None if clean.
    """
    if not reasoning or not chosen:
        return None
    facts = chosen.get("facts") or {}
    r     = reasoning.lower()
    hits: list[str] = []

    if (
        any(w in r for w in ("captur", "jump", "jumping"))
        and facts.get("captures_count", 0) == 0
    ):
        hits.append("claims capture but captures_count=0")

    if (
        any(w in r for w in ("king", "promot", "becomes king"))
        and not facts.get("results_in_king", False)
        and not facts.get("near_promotion", False)
    ):
        hits.append("claims king/promotion but results_in_king=False and near_promotion=False")

    if (
        any(w in r for w in ("safe", "not threatened", "no threat", "secure", "unthreatened"))
        and facts.get("opponent_can_recapture", False)
    ):
        hits.append("claims safe but opponent_can_recapture=True")

    if (
        any(w in r for w in ("center", "central"))
        and not facts.get("center_control", False)
    ):
        hits.append("claims center_control but center_control=False")

    return "; ".join(hits) if hits else None


# ── Slim move and empty trace helpers ────────────────────────────────────────

def _slim(m: Any) -> Optional[dict[str, Any]]:
    if not m:
        return None
    return {"type": m.get("type"), "path": m.get("path"), "captured": m.get("captured", [])}


def _make_empty_trace(baseline: str, turn_no: int, board: list) -> dict[str, Any]:
    return {
        "baseline":              baseline,
        "turn_number":           turn_no,
        "board_before":          [row[:] for row in board],
        "legal_moves_count":     0,
        "legal_moves":           [],
        "raw_model_output":      None,
        "parsed_output":         None,
        "reasoning":             "",
        "attempted_path":        None,
        "attempted_index":       None,
        "attempted_move":        None,
        "chosen_move":           None,
        "best_move":             None,
        "chosen_rank":           0,
        "score_gap":             0.0,
        "legality_result":       "no_legal_moves",
        "legality_error_reason": None,
        "contradiction_result":  None,
        "json_valid":            False,
        "mandatory_capture_respected": True,
        "retry_count":           0,
        "fallback_used":         False,
        "override_used":         False,
        "override_branch":       None,
        "proposal_candidates":   None,
        "raw_llm_choice":        None,
        "retry_llm_choice":      None,
        "final_choice_source":   "none",
        "correction_used":       False,
        "correction_source":     None,
        "board_after":           [row[:] for row in board],
        "top1_hit":              False,
        "top3_hit":              False,
    }


# ── State application helper ──────────────────────────────────────────────────

def _apply_move(acc: dict[str, Any], move: dict[str, Any], reasoning: str) -> dict[str, Any]:
    """Apply a chosen move via update_agent (identical to run_simplified_trace.py BLACK path)."""
    acc["chosen_move"]         = move
    acc["last_move_reasoning"] = reasoning
    _valid  = set(CheckersState.model_fields.keys())
    _state  = CheckersState(**{k: v for k, v in acc.items() if k in _valid})
    _result = _update_agent_fn(_state)
    acc.update(_result)
    return acc


# ── Audit prompt printer ──────────────────────────────────────────────────────

def _print_prompts(baseline: str, turn_no: int, system: str, user: str) -> None:
    sep = "═" * 60
    print()
    print(sep)
    print(f"AUDIT — EXACT PROMPTS  [{baseline.upper()}]  turn={turn_no + 1}")
    print(sep)
    print("── SYSTEM PROMPT " + "─" * 43)
    print(system)
    print("── USER PROMPT " + "─" * 45)
    print(user)
    print(sep)
    print()


# ── Shared RED ply: path-JSON baselines (minimal_raw_llm + rules_only_llm) ───
#
# Both baselines output {"move_path": [...], "reasoning": "..."}.
# The ONLY difference between them is the system prompt passed in.
# This function is never called directly — use the thin wrappers below.

def _run_red_ply_path_json(
    acc: dict[str, Any],
    game_traces: list[dict],
    quiet: bool,
    baseline_name: str,
    system_prompt: str,
    show_prompts: bool,
) -> dict[str, Any]:
    turn_no   = acc.get("turn_number", 0)
    board     = acc["board"]
    legal_all = get_all_legal_moves(board, RED)

    if not legal_all:
        trace = _make_empty_trace(baseline_name, turn_no, board)
        game_traces.append(trace)
        return acc

    # Scoring oracle — runs minimax BEFORE the LLM call; result is NEVER sent to LLM.
    # Used only for post-hoc evaluation and auto-correction fallback.
    scored, _, _, _ = score_all_legal_moves(board, RED, acc.get("position_history"))
    best_scored     = _best_from_scored(scored)

    # ── Build prompts ─────────────────────────────────────────────────────────
    system = system_prompt
    user   = _build_path_json_user(board, turn_no + 1)

    if show_prompts:
        _print_prompts(baseline_name, turn_no, system, user)

    # ── Call LLM (single attempt, no retry) ──────────────────────────────────
    raw_output:  Optional[str]  = None
    parsed_obj:  Optional[dict] = None
    parse_error: str            = "not_attempted"
    json_valid:  bool           = False
    llm_reasoning: str          = ""

    try:
        raw_output  = call_baseline_llm(system, user)
        parsed_obj, parse_error = _parse_json_response(raw_output)
        json_valid  = parsed_obj is not None
        llm_reasoning = (parsed_obj or {}).get("reasoning", "")
    except Exception as e:
        raw_output  = None
        parse_error = f"llm_call_failed: {e}"

    # ── Parse move_path ───────────────────────────────────────────────────────
    attempted_path: Optional[list] = None
    if parsed_obj is not None:
        raw_path = parsed_obj.get("move_path")
        if isinstance(raw_path, list):
            attempted_path = raw_path

    # ── Validate ──────────────────────────────────────────────────────────────
    chosen_move:         Optional[dict] = None
    legality_result:     str            = "illegal"
    legality_reason:     str            = ""
    correction_used:     bool           = False
    correction_source:   Optional[str]  = None
    final_choice_src:    str            = "llm"
    attempted_move_slim: Optional[dict] = None

    if not json_valid:
        legality_result = "illegal"
        legality_reason = f"json_invalid: {parse_error}"

    elif attempted_path is None:
        legality_result = "illegal"
        legality_reason = "move_path missing or malformed in response"

    else:
        matched = _find_move_by_path(legal_all, attempted_path)
        attempted_move_slim = _slim(matched) if matched else None

        if matched is None:
            legality_result = "hallucinated_path"
            legality_reason = f"path {attempted_path} does not match any legal move"

        elif _has_mandatory_capture(legal_all) and matched.get("type") != "jump":
            legality_result = "mandatory_capture_violation"
            legality_reason = "a jump was available but model chose a simple move"

        else:
            legality_result  = "legal"
            chosen_move      = matched
            final_choice_src = "llm"

    # ── Auto-correction ───────────────────────────────────────────────────────
    if chosen_move is None:
        chosen_move       = best_scored
        correction_used   = True
        correction_source = "minimax_best_legal"
        final_choice_src  = "autocorrect"

    # ── Evaluate ──────────────────────────────────────────────────────────────
    chosen_enriched          = _enrich_from_scored(chosen_move, scored)
    chosen_rank, score_gap   = _rank_and_gap(scored, chosen_move)
    best_move_disp           = _slim(scored[0]) if scored else None
    contradiction            = _check_contradictions(llm_reasoning, chosen_enriched)

    board_before = [row[:] for row in board]

    # ── Apply move ────────────────────────────────────────────────────────────
    reasoning_text = llm_reasoning or (f"[autocorrect: {legality_reason}]" if correction_used else "")
    acc = _apply_move(acc, chosen_move, reasoning_text)
    board_after = acc["board"]

    trace: dict[str, Any] = {
        "baseline":              baseline_name,
        "turn_number":           turn_no,
        "board_before":          board_before,
        "legal_moves_count":     len(legal_all),
        "legal_moves":           [_slim(m) for m in legal_all],
        "raw_model_output":      raw_output,
        "parsed_output":         parsed_obj,
        "reasoning":             llm_reasoning,
        "attempted_path":        attempted_path,
        "attempted_index":       None,
        "attempted_move":        attempted_move_slim,
        "chosen_move":           _slim(chosen_move),
        "best_move":             best_move_disp,
        "chosen_rank":           chosen_rank,
        "score_gap":             score_gap,
        "legality_result":       legality_result,
        "legality_error_reason": legality_reason or None,
        "contradiction_result":  contradiction,
        "json_valid":            json_valid,
        "mandatory_capture_respected": legality_result not in (
            "mandatory_capture_violation",
        ) and not correction_used,
        "retry_count":           0,
        "fallback_used":         False,
        "override_used":         False,
        "override_branch":       None,
        "proposal_candidates":   None,
        "raw_llm_choice":        None,
        "retry_llm_choice":      None,
        "final_choice_source":   final_choice_src,
        "correction_used":       correction_used,
        "correction_source":     correction_source,
        "board_after":           board_after,
        "top1_hit":              chosen_rank == 1 and not correction_used,
        "top3_hit":              1 <= chosen_rank <= 3 and not correction_used,
    }
    game_traces.append(trace)

    if not quiet:
        _print_red_summary(trace, scored_moves=scored)

    return acc


# ── RED ply wrappers for path-JSON baselines ──────────────────────────────────

def _run_red_ply_minimal_raw_llm(
    acc: dict[str, Any],
    game_traces: list[dict],
    quiet: bool,
    show_prompts: bool = False,
) -> dict[str, Any]:
    return _run_red_ply_path_json(
        acc, game_traces, quiet,
        BASELINE_MINIMAL_RAW_LLM, _MINIMAL_RAW_LLM_SYSTEM, show_prompts,
    )


def _run_red_ply_rules_only_llm(
    acc: dict[str, Any],
    game_traces: list[dict],
    quiet: bool,
    show_prompts: bool = False,
) -> dict[str, Any]:
    return _run_red_ply_path_json(
        acc, game_traces, quiet,
        BASELINE_RULES_ONLY_LLM, _RULES_ONLY_LLM_SYSTEM, show_prompts,
    )


def _run_red_ply_legal_moves_index_llm(
    acc: dict[str, Any],
    game_traces: list[dict],
    quiet: bool,
    show_prompts: bool = False,
) -> dict[str, Any]:
    turn_no   = acc.get("turn_number", 0)
    board     = acc["board"]
    legal_all = get_all_legal_moves(board, RED)

    if not legal_all:
        trace = _make_empty_trace(BASELINE_LEGAL_MOVES_INDEX_LLM, turn_no, board)
        game_traces.append(trace)
        return acc

    # Scoring oracle — runs minimax BEFORE the LLM call; result is NEVER sent to LLM.
    scored, _, _, _ = score_all_legal_moves(board, RED, acc.get("position_history"))
    best_scored     = _best_from_scored(scored)

    # ── Build prompts ─────────────────────────────────────────────────────────
    system = _LEGAL_MOVES_INDEX_LLM_SYSTEM
    user   = _build_legal_moves_index_llm_user(board, legal_all, turn_no + 1)

    if show_prompts:
        _print_prompts(BASELINE_LEGAL_MOVES_INDEX_LLM, turn_no, system, user)

    # ── Call LLM ─────────────────────────────────────────────────────────────
    raw_output:   Optional[str]  = None
    parsed_obj:   Optional[dict] = None
    parse_error:  str            = "not_attempted"
    json_valid:   bool           = False
    llm_reasoning: str           = ""

    try:
        raw_output  = call_baseline_llm(system, user)
        parsed_obj, parse_error = _parse_json_response(raw_output)
        json_valid  = parsed_obj is not None
        llm_reasoning = (parsed_obj or {}).get("reasoning", "")
    except Exception as e:
        raw_output  = None
        parse_error = f"llm_call_failed: {e}"

    # ── Parse chosen_index ────────────────────────────────────────────────────
    raw_index:     Optional[int] = None
    attempted_idx: Optional[int] = None

    if parsed_obj is not None:
        v = parsed_obj.get("chosen_index")
        if isinstance(v, bool):
            pass  # reject bool masquerading as int
        elif isinstance(v, (int, float)):
            raw_index     = int(v)
            attempted_idx = raw_index
        elif isinstance(v, str):
            m = re.fullmatch(r"-?\d+", v.strip())
            if m:
                raw_index     = int(v.strip())
                attempted_idx = raw_index

    # ── Validate ──────────────────────────────────────────────────────────────
    chosen_move:         Optional[dict] = None
    legality_result:     str            = "illegal"
    legality_reason:     str            = ""
    correction_used:     bool           = False
    correction_source:   Optional[str]  = None
    final_choice_src:    str            = "llm"
    attempted_move_slim: Optional[dict] = None

    if not json_valid:
        legality_result = "illegal"
        legality_reason = f"json_invalid: {parse_error}"

    elif raw_index is None:
        legality_result = "illegal"
        legality_reason = "chosen_index missing or not a valid integer"

    elif not (0 <= raw_index < len(legal_all)):
        legality_result = "illegal"
        legality_reason = f"index {raw_index} out of range [0..{len(legal_all) - 1}]"

    else:
        # Legal list already enforces mandatory capture — valid index is always legal
        chosen_move         = legal_all[raw_index]
        attempted_move_slim = _slim(chosen_move)
        legality_result     = "legal"
        final_choice_src    = "llm"

    # ── Auto-correction ───────────────────────────────────────────────────────
    if chosen_move is None:
        chosen_move       = best_scored
        correction_used   = True
        correction_source = "minimax_best_legal"
        final_choice_src  = "autocorrect"

    # ── Evaluate ──────────────────────────────────────────────────────────────
    chosen_enriched        = _enrich_from_scored(chosen_move, scored)
    chosen_rank, score_gap = _rank_and_gap(scored, chosen_move)
    best_move_disp         = _slim(scored[0]) if scored else None
    contradiction          = _check_contradictions(llm_reasoning, chosen_enriched)

    board_before = [row[:] for row in board]

    reasoning_text = llm_reasoning or (f"[autocorrect: {legality_reason}]" if correction_used else "")
    acc = _apply_move(acc, chosen_move, reasoning_text)
    board_after = acc["board"]

    trace: dict[str, Any] = {
        "baseline":              BASELINE_LEGAL_MOVES_INDEX_LLM,
        "turn_number":           turn_no,
        "board_before":          board_before,
        "legal_moves_count":     len(legal_all),
        "legal_moves":           [_slim(m) for m in legal_all],
        "raw_model_output":      raw_output,
        "parsed_output":         parsed_obj,
        "reasoning":             llm_reasoning,
        "attempted_path":        None,
        "attempted_index":       attempted_idx,
        "attempted_move":        attempted_move_slim,
        "chosen_move":           _slim(chosen_move),
        "best_move":             best_move_disp,
        "chosen_rank":           chosen_rank,
        "score_gap":             score_gap,
        "legality_result":       legality_result,
        "legality_error_reason": legality_reason or None,
        "contradiction_result":  contradiction,
        "json_valid":            json_valid,
        "mandatory_capture_respected": legality_result == "legal" and not correction_used,
        "retry_count":           0,
        "fallback_used":         False,
        "override_used":         False,
        "override_branch":       None,
        "proposal_candidates":   None,
        "raw_llm_choice":        raw_index,
        "retry_llm_choice":      None,
        "final_choice_source":   final_choice_src,
        "correction_used":       correction_used,
        "correction_source":     correction_source,
        "board_after":           board_after,
        "top1_hit":              chosen_rank == 1 and not correction_used,
        "top3_hit":              1 <= chosen_rank <= 3 and not correction_used,
    }
    game_traces.append(trace)

    if not quiet:
        _print_red_summary(trace, scored_moves=scored)

    return acc


# ── RED ply: legal_moves_path_llm ────────────────────────────────────────────
#
# Sees board + legal move list with explicit path coords; outputs move_path JSON.
# Validation identical to path-JSON baselines (_find_move_by_path).
# No autocorrect — same as the other game-runner baselines (uses minimax fallback).

def _run_red_ply_legal_moves_path_llm(
    acc: dict[str, Any],
    game_traces: list[dict],
    quiet: bool,
    show_prompts: bool = False,
) -> dict[str, Any]:
    turn_no   = acc.get("turn_number", 0)
    board     = acc["board"]
    legal_all = get_all_legal_moves(board, RED)

    if not legal_all:
        trace = _make_empty_trace(BASELINE_LEGAL_MOVES_PATH_LLM, turn_no, board)
        game_traces.append(trace)
        return acc

    # Scoring oracle — runs minimax BEFORE the LLM call; result is NEVER sent to LLM.
    scored, _, _, _ = score_all_legal_moves(board, RED, acc.get("position_history"))
    best_scored     = _best_from_scored(scored)

    # ── Build prompts ─────────────────────────────────────────────────────────
    system = _LEGAL_MOVES_PATH_LLM_SYSTEM
    user   = _build_legal_moves_path_llm_user(board, legal_all, turn_no + 1)

    if show_prompts:
        _print_prompts(BASELINE_LEGAL_MOVES_PATH_LLM, turn_no, system, user)

    # ── Call LLM (single attempt, no retry) ──────────────────────────────────
    raw_output:  Optional[str]  = None
    parsed_obj:  Optional[dict] = None
    parse_error: str            = "not_attempted"
    json_valid:  bool           = False
    llm_reasoning: str          = ""

    try:
        raw_output  = call_baseline_llm(system, user)
        parsed_obj, parse_error = _parse_json_response(raw_output)
        json_valid  = parsed_obj is not None
        llm_reasoning = (parsed_obj or {}).get("reasoning", "")
    except Exception as e:
        raw_output  = None
        parse_error = f"llm_call_failed: {e}"

    # ── Parse move_path ───────────────────────────────────────────────────────
    attempted_path: Optional[list] = None
    if parsed_obj is not None:
        raw_path = parsed_obj.get("move_path")
        if isinstance(raw_path, list):
            attempted_path = raw_path

    # ── Validate ──────────────────────────────────────────────────────────────
    chosen_move:         Optional[dict] = None
    legality_result:     str            = "illegal"
    legality_reason:     str            = ""
    correction_used:     bool           = False
    correction_source:   Optional[str]  = None
    final_choice_src:    str            = "llm"
    attempted_move_slim: Optional[dict] = None

    if not json_valid:
        legality_result = "illegal"
        legality_reason = f"json_invalid: {parse_error}"

    elif attempted_path is None:
        legality_result = "illegal"
        legality_reason = "move_path missing or malformed in response"

    else:
        matched = _find_move_by_path(legal_all, attempted_path)
        attempted_move_slim = _slim(matched) if matched else None

        if matched is None:
            legality_result = "hallucinated_path"
            legality_reason = f"path {attempted_path} not found in legal move list"

        elif _has_mandatory_capture(legal_all) and matched.get("type") != "jump":
            legality_result = "mandatory_capture_violation"
            legality_reason = "a jump was available but model chose a simple move"

        else:
            legality_result  = "legal"
            chosen_move      = matched
            final_choice_src = "llm"

    # ── Auto-correction ───────────────────────────────────────────────────────
    if chosen_move is None:
        chosen_move       = best_scored
        correction_used   = True
        correction_source = "minimax_best_legal"
        final_choice_src  = "autocorrect"

    # ── Evaluate ──────────────────────────────────────────────────────────────
    chosen_enriched          = _enrich_from_scored(chosen_move, scored)
    chosen_rank, score_gap   = _rank_and_gap(scored, chosen_move)
    best_move_disp           = _slim(scored[0]) if scored else None
    contradiction            = _check_contradictions(llm_reasoning, chosen_enriched)

    board_before = [row[:] for row in board]

    reasoning_text = llm_reasoning or (f"[autocorrect: {legality_reason}]" if correction_used else "")
    acc = _apply_move(acc, chosen_move, reasoning_text)
    board_after = acc["board"]

    trace: dict[str, Any] = {
        "baseline":              BASELINE_LEGAL_MOVES_PATH_LLM,
        "turn_number":           turn_no,
        "board_before":          board_before,
        "legal_moves_count":     len(legal_all),
        "legal_moves":           [_slim(m) for m in legal_all],
        "raw_model_output":      raw_output,
        "parsed_output":         parsed_obj,
        "reasoning":             llm_reasoning,
        "attempted_path":        attempted_path,
        "attempted_index":       None,
        "attempted_move":        attempted_move_slim,
        "chosen_move":           _slim(chosen_move),
        "best_move":             best_move_disp,
        "chosen_rank":           chosen_rank,
        "score_gap":             score_gap,
        "legality_result":       legality_result,
        "legality_error_reason": legality_reason or None,
        "contradiction_result":  contradiction,
        "json_valid":            json_valid,
        "mandatory_capture_respected": legality_result not in (
            "mandatory_capture_violation",
        ) and not correction_used,
        "retry_count":           0,
        "fallback_used":         False,
        "override_used":         False,
        "override_branch":       None,
        "proposal_candidates":   None,
        "raw_llm_choice":        None,
        "retry_llm_choice":      None,
        "final_choice_source":   final_choice_src,
        "correction_used":       correction_used,
        "correction_source":     correction_source,
        "board_after":           board_after,
        "top1_hit":              chosen_rank == 1 and not correction_used,
        "top3_hit":              1 <= chosen_rank <= 3 and not correction_used,
    }
    game_traces.append(trace)

    if not quiet:
        _print_red_summary(trace, scored_moves=scored)

    return acc


# ── RED ply: full_system ──────────────────────────────────────────────────────

def _run_red_ply_full_system(
    acc: dict[str, Any],
    game_traces: list[dict],
    quiet: bool,
    show_prompts: bool = False,
) -> dict[str, Any]:
    # Lazy import: graph is only constructed when this baseline is actually used
    from checkers.graph.graph import checkers_graph

    if show_prompts:
        print()
        print("═" * 60)
        print("AUDIT — PROMPTS  [FULL_SYSTEM]")
        print("═" * 60)
        print("full_system prompts are constructed internally by ranker_agent.")
        print("See: checkers/agents/ranker_agent.py")
        print("═" * 60)
        print()

    turn_no      = acc.get("turn_number", 0)
    board_before = [row[:] for row in acc["board"]]
    legal_before = get_all_legal_moves(acc["board"], RED)

    if not legal_before:
        trace = _make_empty_trace(BASELINE_FULL_SYSTEM, turn_no, acc["board"])
        game_traces.append(trace)
        return acc

    # ── Stream graph: scorer → proposal → ranker → update_agent ───────────────
    acc["last_completed_node"] = None
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 50,
    }

    # Snapshots captured during the stream (before update_agent overwrites/clears)
    _sym_scored:      list[dict] = []   # from scorer_node
    _proposal_list:   list[dict] = []   # from deterministic_proposal_node
    _snap_chosen:     Optional[dict] = None
    _snap_reasoning:  str = ""
    _snap_retry_ct:   int = 0
    saw_update_agent  = False

    try:
        for chunk in checkers_graph.stream(
            acc,
            stream_mode="updates",
            interrupt_after=["update_agent"],
            config=cfg,
        ):
            for node_name, delta in chunk.items():
                if node_name in ("__interrupt__", "__end__"):
                    continue
                if not isinstance(delta, dict):
                    continue
                acc.update(delta)

                if node_name == "scorer_node":
                    _sym_scored = list(acc.get("symbolic_scored_moves") or [])

                elif node_name == "deterministic_proposal_node":
                    _proposal_list = list(acc.get("legal_moves") or [])

                elif node_name == "ranker_agent":
                    # Capture before update_agent potentially clears these
                    _snap_chosen    = acc.get("chosen_move")
                    _snap_reasoning = acc.get("last_move_reasoning") or ""
                    _snap_retry_ct  = int(acc.get("ranker_retry_count") or 0)

                elif node_name == "update_agent":
                    saw_update_agent = True

    except Exception as e:
        print(f"[full_system] graph stream error: {e}", file=sys.stderr)

    if not saw_update_agent and not quiet:
        print("[full_system] warning: update_agent did not complete.", file=sys.stderr)

    # ── Collect diagnostics ────────────────────────────────────────────────────
    chosen_move   = _snap_chosen or acc.get("chosen_move")
    reasoning     = _snap_reasoning or acc.get("last_move_reasoning") or ""
    ranker_diag   = acc.get("ranker_diagnostics") or {}
    override_used   = bool(ranker_diag.get("override_branch_name"))
    override_branch = ranker_diag.get("override_branch_name")
    fallback_used   = bool(ranker_diag.get("override_fallback_applied"))
    retry_count     = _snap_retry_ct

    # ── Evaluate via symbolic_scored_moves (full list, rank-sorted) ───────────
    chosen_rank, score_gap = _rank_and_gap_slim(_sym_scored, chosen_move)
    best_slim = next(
        (e.get("move") for e in _sym_scored if e.get("rank") == 1), None
    )

    # ── Legality check against pre-turn legal list ────────────────────────────
    legality_result = "illegal"
    legality_reason = ""
    if chosen_move is not None:
        if _find_move_by_path(legal_before, chosen_move.get("path")) is not None:
            legality_result = "legal"
        else:
            legality_result = "illegal"
            legality_reason = "chosen_move path not found in pre-turn legal moves"

    # ── Contradiction check (uses full chosen_move which has facts from ranker) ─
    contradiction = _check_contradictions(reasoning, chosen_move or {})

    board_after = acc["board"]

    trace: dict[str, Any] = {
        "baseline":              BASELINE_FULL_SYSTEM,
        "turn_number":           turn_no,
        "board_before":          board_before,
        "legal_moves_count":     len(legal_before),
        "legal_moves":           [_slim(m) for m in legal_before],
        # ranker_agent does not expose the raw Mistral API response in state
        "raw_model_output":      "N/A (internal)",
        "parsed_output":         None,
        "reasoning":             reasoning,
        "attempted_path":        None,
        "attempted_index":       None,
        "attempted_move":        None,
        "chosen_move":           _slim(chosen_move),
        "best_move":             best_slim,
        "chosen_rank":           chosen_rank,
        "score_gap":             score_gap,
        "legality_result":       legality_result,
        "legality_error_reason": legality_reason or None,
        "contradiction_result":  contradiction,
        "json_valid":            True,   # internal pipeline handles its own parsing
        "mandatory_capture_respected": legality_result == "legal",
        "retry_count":           retry_count,
        "fallback_used":         fallback_used,
        "override_used":         override_used,
        "override_branch":       override_branch,
        "proposal_candidates":   [_slim(m) for m in _proposal_list],
        "raw_llm_choice":        None,
        "retry_llm_choice":      None,
        "final_choice_source":   (
            "override" if override_used else
            "fallback" if fallback_used else
            "llm"
        ),
        "correction_used":       False,
        "correction_source":     None,
        "board_after":           board_after,
        "top1_hit":              chosen_rank == 1,
        "top3_hit":              1 <= chosen_rank <= 3,
    }
    game_traces.append(trace)

    if not quiet:
        _print_red_summary_full_system(trace, _proposal_list)

    return acc


# ── Post-move display: path-JSON and legal_moves_index/path_llm baselines ─────

def _print_red_summary(
    trace: dict[str, Any],
    scored_moves: list[dict[str, Any]],
) -> None:
    print()
    print(RULE)
    print(f"── RED MOVE DIAGNOSTICS  [{trace['baseline'].upper()}]  turn={trace['turn_number'] + 1} ──")

    cm = trace.get("chosen_move") or {}
    bm = trace.get("best_move")   or {}
    print(f"  Chosen : {cm.get('type')} {cm.get('path')}  captures={cm.get('captured', [])}")
    print(f"  Best   : {bm.get('type')} {bm.get('path')}")

    rank = trace.get("chosen_rank", 0)
    gap  = trace.get("score_gap", "?")
    n    = trace.get("legal_moves_count", "?")
    print(
        f"  Rank   : {_red_if(rank, isinstance(rank, int) and rank > 3)} / {n} legal"
        f"   |   Score gap : {_red_if(gap, isinstance(gap, float) and gap > 50)}"
    )

    leg = trace.get("legality_result", "?")
    leg_ok = leg == "legal"
    leg_str = _red_if(leg, not leg_ok)
    reason  = trace.get("legality_error_reason")
    print(f"  Legal  : {leg_str}" + (f"  ({reason})" if reason else ""))

    print(f"  JSON valid : {trace.get('json_valid')}")
    print(f"  top1_hit   : {trace.get('top1_hit')}   |   top3_hit : {trace.get('top3_hit')}")

    contr = trace.get("contradiction_result")
    if contr:
        print(f"  {_ylw('Contradiction: ' + contr)}")

    reasoning = trace.get("reasoning") or ""
    if reasoning:
        print("  Reasoning:")
        print(_wrap(reasoning, width=100, indent="    "))

    if trace.get("correction_used"):
        print(
            f"  {_red_if('[CORRECTION APPLIED]', True)}"
            f"  source={trace.get('correction_source')}"
            f"  reason={trace.get('legality_error_reason')}"
        )

    print()
    print("── BOARD AFTER RED MOVE ──")
    board_after = trace.get("board_after")
    if board_after:
        print_board(board_after)
    print(RULE)
    print()


# ── Post-move display: full_system ────────────────────────────────────────────

def _print_red_summary_full_system(
    trace: dict[str, Any],
    proposal_list: list[dict[str, Any]],
) -> None:
    print()
    print(RULE)
    print(f"── RED MOVE DIAGNOSTICS  [FULL_SYSTEM]  turn={trace['turn_number'] + 1} ──")

    cm = trace.get("chosen_move") or {}
    bm = trace.get("best_move")   or {}
    print(f"  Chosen : {cm.get('type')} {cm.get('path')}  captures={cm.get('captured', [])}")
    print(f"  Best   : {bm.get('type')} {bm.get('path')}")

    rank = trace.get("chosen_rank", 0)
    gap  = trace.get("score_gap", "?")
    n    = trace.get("legal_moves_count", "?")
    print(
        f"  Rank   : {_red_if(rank, isinstance(rank, int) and rank > 3)} / {n} legal"
        f"   |   Score gap : {_red_if(gap, isinstance(gap, float) and gap > 50)}"
    )

    leg = trace.get("legality_result", "?")
    print(f"  Legal  : {_red_if(leg, leg != 'legal')}")

    print(f"  top1_hit : {trace.get('top1_hit')}   |   top3_hit : {trace.get('top3_hit')}")

    if trace.get("override_used"):
        print(f"  {_ylw('Override : ' + str(trace.get('override_branch')))}")
    if trace.get("fallback_used"):
        print(f"  {_ylw('Fallback used')}")
    if trace.get("retry_count", 0) > 0:
        print(f"  Retries  : {trace.get('retry_count')}")

    contr = trace.get("contradiction_result")
    if contr:
        print(f"  {_ylw('Contradiction: ' + contr)}")

    reasoning = trace.get("reasoning") or ""
    if reasoning and reasoning != "BLACK human move":
        print("  Reasoning:")
        print(_wrap(reasoning, width=100, indent="    "))

    if proposal_list:
        print(f"\n── PROPOSAL SHORTLIST ({len(proposal_list)} candidates) ──")
        for i, m in enumerate(proposal_list):
            facts = m.get("facts") or {}
            sc    = facts.get("minimax_score", "n/a")
            rk    = facts.get("symbolic_rank", "?")
            print(f"  [{i}] rank={rk} score={sc}  {m.get('type')} {m.get('path')}")

    print(RULE)
    print()


# ── BLACK ply (human input — identical to run_simplified_trace.py) ────────────

def _run_black_ply(acc: dict[str, Any], quiet: bool) -> dict[str, Any]:
    turn_no = acc.get("turn_number", 0)
    board   = acc["board"]
    legal   = get_all_legal_moves(board, BLACK)

    if not legal:
        print("[baseline_runner] BLACK has no legal moves.", file=sys.stderr)
        return acc

    print(BAR)
    print(f"TURN {turn_no + 1} | BLACK to move  (YOU)")
    print(BAR)
    print("── BOARD ──")
    print_board(board)
    print()
    print(f"── YOUR MOVES ({len(legal)} available) ──")
    for i, m in enumerate(legal):
        print(_fmt_engine_move(i, m))

    while True:
        try:
            raw = input(f"\nEnter move index [0-{len(legal) - 1}]: ").strip()
            k   = int(raw)
            if 0 <= k < len(legal):
                break
            print(f"  Invalid — enter a number between 0 and {len(legal) - 1}.")
        except (ValueError, EOFError):
            print("  Invalid input, please enter a number.")

    move = legal[k]
    if not quiet:
        path = move.get("path") or []
        if len(path) >= 2:
            a, b = path[0], path[-1]
            print(f"\nApplied: {move.get('type')} from [{a[0]},{a[1]}] to [{b[0]},{b[1]}]")
        print(RULE)
        print()

    acc = _apply_move(acc, move, "BLACK human move")
    return acc


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_jsonl(traces: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in traces:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"[trace]   {path}")


_CSV_FIELDS = [
    "baseline", "turn_number", "legal_moves_count",
    "json_valid", "legality_result", "legality_error_reason",
    "correction_used", "correction_source",
    "chosen_rank", "score_gap",
    "top1_hit", "top3_hit",
    "mandatory_capture_respected",
    "retry_count", "fallback_used", "override_used", "override_branch",
    "final_choice_source", "contradiction_result",
]


def _save_csv(traces: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in traces:
            writer.writerow(rec)
    print(f"[metrics] {path}")


def _save_markdown(
    traces: list[dict],
    game_result: dict[str, Any],
    baseline: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    red_traces = [t for t in traces if t.get("baseline") == baseline]
    n = len(red_traces)

    winner = game_result.get("winner", "?")
    turns  = game_result.get("turns",  "?")

    if n == 0:
        path.write_text(f"# {baseline} — no RED moves recorded\n", encoding="utf-8")
        print(f"[summary] {path}")
        return

    def _pct(num: int) -> str:
        return f"{num}/{n} ({num / n:.1%})"

    legal_n        = sum(1 for t in red_traces if t.get("legality_result") == "legal")
    json_n         = sum(1 for t in red_traces if t.get("json_valid"))
    corr_n         = sum(1 for t in red_traces if t.get("correction_used"))
    top1_n         = sum(1 for t in red_traces if t.get("top1_hit"))
    top3_n         = sum(1 for t in red_traces if t.get("top3_hit"))
    halluc_n       = sum(1 for t in red_traces if t.get("legality_result") in (
        "hallucinated_path", "mandatory_capture_violation"))
    contr_n        = sum(1 for t in red_traces if t.get("contradiction_result"))
    retry_total    = sum(t.get("retry_count",   0) for t in red_traces)
    fallback_total = sum(1 for t in red_traces if t.get("fallback_used"))
    override_total = sum(1 for t in red_traces if t.get("override_used"))
    gaps           = [
        t.get("score_gap") for t in red_traces
        if isinstance(t.get("score_gap"), (int, float))
        and t.get("score_gap") not in (float("inf"), float("-inf"))
    ]
    avg_gap = round(sum(gaps) / len(gaps), 2) if gaps else "N/A"

    override_branches: dict[str, int] = {}
    for t in red_traces:
        b = t.get("override_branch")
        if b:
            override_branches[b] = override_branches.get(b, 0) + 1

    lines = [
        f"# Baseline Evaluation: `{baseline}`",
        "",
        f"**Game result**: {winner}  |  **Total plies**: {turns}",
        "",
        "## RED Performance Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| RED plies evaluated | {n} |",
        f"| Legality rate | {_pct(legal_n)} |",
        f"| JSON valid rate | {_pct(json_n)} |",
        f"| Correction rate | {_pct(corr_n)} |",
        f"| Hallucination / mandatory-capture-violation rate | {_pct(halluc_n)} |",
        f"| Reasoning contradiction rate | {_pct(contr_n)} |",
        f"| Top-1 agreement (uncorrected) | {_pct(top1_n)} |",
        f"| Top-3 agreement (uncorrected) | {_pct(top3_n)} |",
        f"| Avg minimax score gap | {avg_gap} |",
        f"| Total retries | {retry_total} |",
        f"| Fallback uses | {fallback_total} |",
        f"| Override uses | {override_total} |",
        "",
    ]

    if override_branches:
        lines += [
            "### Override branches triggered",
            "",
            "| Branch | Count |",
            "|--------|-------|",
        ]
        for branch, count in sorted(override_branches.items(), key=lambda x: -x[1]):
            lines.append(f"| `{branch}` | {count} |")
        lines.append("")

    lines += [
        "## Per-Turn Detail",
        "",
        "| Turn | Legal | Rank | Gap | Correction | Override | Contradiction |",
        "|------|-------|------|-----|------------|----------|---------------|",
    ]
    for t in red_traces:
        ov = t.get("override_branch") or ("✓" if t.get("override_used") else "—")
        lines.append(
            f"| {t.get('turn_number', '') + 1} "
            f"| {t.get('legality_result', '')} "
            f"| {t.get('chosen_rank', '')} "
            f"| {t.get('score_gap', '')} "
            f"| {'✓' if t.get('correction_used') else '—'} "
            f"| {ov} "
            f"| {'✓' if t.get('contradiction_result') else '—'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {path}")


# ── Game end summary ──────────────────────────────────────────────────────────

def _print_final_summary(
    acc: dict[str, Any],
    game_traces: list[dict],
    baseline: str,
) -> None:
    draw   = acc.get("draw", False)
    winner = acc.get("winner")
    if draw:
        winner_str = "Draw"
    elif winner == RED:
        winner_str = "RED"
    elif winner == BLACK:
        winner_str = "BLACK"
    else:
        winner_str = "N/A (max turns reached)"

    red_t = [t for t in game_traces if t.get("baseline") == baseline]
    n     = len(red_t)

    print(BAR)
    print("GAME OVER")
    print(BAR)
    print(f"Baseline : {baseline}")
    print(f"Winner   : {winner_str}")
    print(f"Plies    : {acc.get('turn_number', '?')}")
    if n:
        legal_n = sum(1 for t in red_t if t.get("legality_result") == "legal")
        corr_n  = sum(1 for t in red_t if t.get("correction_used"))
        top1_n  = sum(1 for t in red_t if t.get("top1_hit"))
        top3_n  = sum(1 for t in red_t if t.get("top3_hit"))
        print(f"\nRED ({baseline}) — {n} plies")
        print(f"  Legality    : {legal_n}/{n} = {legal_n/n:.1%}")
        print(f"  Corrections : {corr_n}/{n}  = {corr_n/n:.1%}")
        print(f"  Top-1 rate  : {top1_n}/{n}  = {top1_n/n:.1%}")
        print(f"  Top-3 rate  : {top3_n}/{n}  = {top3_n/n:.1%}")
    print()
    print("Final board:")
    print_board(acc["board"])
    print(BAR)


# ── Main game loop ────────────────────────────────────────────────────────────

# ── Manual-trace dispatch (one runner per --baseline value) ──────────────────
# Explicit one-to-one mapping; unknown values raise so nothing silently falls
# through to _run_red_ply_full_system. Only BASELINE_FULL_SYSTEM is allowed to
# call _run_red_ply_full_system.
_RED_PLY_DISPATCH: dict[str, Any] = {
    BASELINE_MINIMAL_RAW_LLM: _run_red_ply_minimal_raw_llm,
    BASELINE_RULES_ONLY_LLM:  _run_red_ply_rules_only_llm,
    BASELINE_FULL_SYSTEM:     _run_red_ply_full_system,
}


def _dispatch_red_ply(
    baseline: str,
    acc: dict[str, Any],
    game_traces: list[dict],
    quiet: bool,
    show_prompts: bool,
) -> dict[str, Any]:
    runner = _RED_PLY_DISPATCH.get(baseline)
    if runner is None:
        raise SystemExit(
            f"[manual_trace] unsupported baseline: {baseline!r}. "
            f"Choose one of: {sorted(_RED_PLY_DISPATCH)}"
        )
    return runner(acc, game_traces, quiet, show_prompts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline evaluation runner: RED = selected baseline, BLACK = human.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline minimal_raw_llm
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline rules_only_llm
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline legal_moves_index_llm
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline legal_moves_path_llm
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline full_system --max-turns 100
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --baseline minimal_raw_llm --show-prompts
          venv/bin/python3 -m checkers.baseline_eval.run_baseline_human_trace --show-table
        """),
    )
    parser.add_argument(
        "--baseline",
        choices=list(ALL_BASELINES),
        help=(
            "RED baseline: minimal_raw_llm | rules_only_llm | "
            "legal_moves_index_llm | legal_moves_path_llm | full_system"
        ),
    )
    parser.add_argument(
        "--max-turns", type=int, default=200,
        help="Safety cap on total plies (default 200).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="logs/baseline_eval",
        help="Output directory for trace files (default: logs/baseline_eval).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-node verbose output (still prints board and diagnostics).",
    )
    parser.add_argument(
        "--show-prompts", action="store_true",
        help="Print exact system + user prompts before each LLM call (audit mode).",
    )
    parser.add_argument(
        "--show-table", action="store_true",
        help="Print the baseline information-visibility comparison table and exit.",
    )
    args = parser.parse_args()

    if args.show_table:
        print(_BASELINE_COMPARISON_TABLE)
        return

    if not args.baseline:
        parser.error("--baseline is required (or use --show-table to see the comparison table).")

    baseline = args.baseline
    out_dir  = Path(args.out_dir)
    game_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    print(BAR)
    print(f"BASELINE EVALUATION RUNNER")
    print(f"  baseline      : {baseline}")
    print(f"  game_id       : {game_id}")
    print(f"  MINIMAX_DEPTH : {os.environ.get('MINIMAX_DEPTH', 'default')}")
    print(f"  MINIMAX_ENABLED: {os.environ.get('MINIMAX_ENABLED', 'true')}")
    print(f"  show_prompts  : {args.show_prompts}")
    print(f"  RED = {baseline}  |  BLACK = human")
    print(BAR)
    print()

    if args.show_prompts:
        print(_BASELINE_COMPARISON_TABLE)

    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()

    game_traces: list[dict[str, Any]] = []

    # ── Main loop ──────────────────────────────────────────────────────────────
    while True:
        if acc.get("game_over"):
            break

        if (acc.get("turn_number") or 0) >= args.max_turns:
            print(f"\n[baseline_runner] Max turns ({args.max_turns}) reached.", file=sys.stderr)
            break

        cp = acc["current_player"]

        # ── RED turn ───────────────────────────────────────────────────────────
        if cp == RED:
            turn_no = acc.get("turn_number", 0)
            legal   = get_all_legal_moves(acc["board"], RED)

            print(BAR)
            print(f"TURN {turn_no + 1} | RED to move  [{baseline}]")
            print(BAR)
            print("── BOARD ──")
            print_board(acc["board"])

            # For non-full_system baselines, show the engine legal moves to the
            # human viewer (these are NOT sent to the LLM in any of the 3 modes).
            if baseline != BASELINE_FULL_SYSTEM:
                print(f"\n── ENGINE LEGAL MOVES ({len(legal)}) ──")
                for i, m in enumerate(legal):
                    print(_fmt_engine_move(i, m))
            print()

            acc = _dispatch_red_ply(
                baseline, acc, game_traces, args.quiet, args.show_prompts,
            )

        # ── BLACK turn ─────────────────────────────────────────────────────────
        else:
            acc = _run_black_ply(acc, args.quiet)

    # ── Game over ──────────────────────────────────────────────────────────────
    draw   = acc.get("draw", False)
    winner = acc.get("winner")
    game_result = {
        "winner": "Draw" if draw else (
            "RED" if winner == RED else
            "BLACK" if winner == BLACK else
            "N/A"
        ),
        "turns": acc.get("turn_number", "?"),
    }

    _print_final_summary(acc, game_traces, baseline)

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{baseline}_{game_id}"

    print()
    _save_jsonl(game_traces,  stem.with_suffix(".jsonl"))
    _save_csv  (game_traces,  Path(str(stem) + "_metrics.csv"))
    _save_markdown(
        game_traces, game_result, baseline,
        Path(str(stem) + "_summary.md"),
    )


if __name__ == "__main__":
    main()
