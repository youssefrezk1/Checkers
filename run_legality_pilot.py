#!/usr/bin/env python3
"""
run_legality_pilot.py
=====================
Pilot baseline evaluation for LLM legality testing.

Schema: { "selected_move": [[row,col],...], "reasoning": "..." }

result_type field (authoritative discriminator)
-----------------------------------------------
  "api_failure"    API failed after all retries — excluded from all eval metrics
  "parse_failure"  API succeeded, JSON invalid — counted in invalid_format_rate
  "legal"          API succeeded, parsed, move in hidden_legal_moves
  "illegal"        API succeeded, parsed, move NOT in hidden_legal_moves

Privacy invariant
-----------------
  hidden_legal_moves are NEVER passed to build_user_prompt or the API client.

Usage
-----
  python run_legality_pilot.py                              # live, all baselines
  python run_legality_pilot.py --dry-run                    # canned, no API call
  python run_legality_pilot.py --request-delay 3            # 3s between calls
  python run_legality_pilot.py --baselines B2_rules --n-eval 5 --show-prompts
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import textwrap
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from checkers.data.legality_eval.prompts import (
    BASELINES, build_user_prompt,
    build_b5_user_prompt, build_b6_user_prompt, build_b7_user_prompt,
)
from checkers.data.legality_eval.evaluator import evaluate_scenario
from checkers.data.legality_eval.candidate_moves import (
    get_candidates, match_selected_to_candidate, match_b6_response, match_b7_response,
)
from checkers.data.legality_eval.metrics import aggregate, format_report
from checkers.data.legality_eval.salvage import salvage_parse_failure, aggregate_salvage
from checkers.engine.rules import get_all_legal_moves
from checkers.engine.board import RED, BLACK
from checkers.data.pdn_importer.fen_utils import str_to_side

# Full-System-Trace imports (real pipeline — lazy to avoid loading LangGraph
# at import time when only B1–B4 baselines are selected).
def _fst_imports() -> tuple:
    """
    Lazily import the real full-system graph and single-ply runner.
    Avoids loading LangGraph at module import time when only B1-B4 are used.
    Sets required env vars before first import.
    Returns (checkers_graph, _stream_one_ply_fn, CheckersState).
    """
    import os
    os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
    os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")
    # Import the compiled graph and the single-ply runner from the real trace file
    import importlib.util, sys
    from pathlib import Path as _Path
    _trace_path = str(_Path(__file__).parent / "run_simplified_trace.py")
    _spec = importlib.util.spec_from_file_location("_run_simplified_trace", _trace_path)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    from checkers.state.state import CheckersState
    return _mod.checkers_graph, _mod._stream_one_ply, CheckersState

# ── B8 lazy imports ──────────────────────────────────────────────────────────
def _b8_imports():
    """
    Lazily import all modules needed by B8a/B8b so they are not loaded at
    module-import time when B8 baselines are not selected.
    """
    import os as _os
    _os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
    _os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

    from checkers.agents.scorer_agent import score_all_legal_moves
    from checkers.agents.deterministic_proposal import select_proposal_candidates
    from checkers.agents.ranker_agent import (
        RANKER_SYSTEM_PROMPT,
        build_ranker_user_prompt,
        call_ranker,
        _parse_ranker_json,
        _extract_chosen_index,
        _regex_extract_chosen_index,
        _resolve_ranker_index,
    )
    from checkers.state.state import CheckersState
    return (
        score_all_legal_moves,
        select_proposal_candidates,
        RANKER_SYSTEM_PROMPT,
        build_ranker_user_prompt,
        call_ranker,
        _parse_ranker_json,
        _extract_chosen_index,
        _regex_extract_chosen_index,
        _resolve_ranker_index,
        CheckersState,
    )


def _run_b8(
    board: list,
    side_str: str,
    scenario_id: str,
    variant: str,          # "shortlist_no_safety" | "full_legal_shuffled_no_safety"
    run_seed: int,
    dry_run: bool = False,
) -> tuple[str, dict, dict]:
    """
    B8 single-ply ranker call with NO safety filter, NO retry, NO override,
    NO fallback, NO repair, NO update_agent.

    variant="shortlist_no_safety"  (B8a):
        Uses select_proposal_candidates(k=5) on scored legal moves.
        Preserves shortlist order (minimax-sorted + diversity pins).

    variant="full_legal_shuffled_no_safety"  (B8b):
        Uses all scored legal moves.
        Shuffles presentation order deterministically via (run_seed, scenario_id).

    Returns (raw_json_str, api_meta, b8_diag).
    raw_json_str uses the legality-pilot selected_move schema so evaluate_scenario works.
    """
    (
        score_all_legal_moves,
        select_proposal_candidates,
        RANKER_SYSTEM_PROMPT,
        build_ranker_user_prompt,
        call_ranker,
        _parse_ranker_json,
        _extract_chosen_index,
        _regex_extract_chosen_index,
        _resolve_ranker_index,
        CheckersState,
    ) = _b8_imports()

    from checkers.data.pdn_importer.fen_utils import str_to_side as _str_to_side

    side_int = _str_to_side(side_str)

    # ── Step 1: score all legal moves (engine, deterministic) ─────────────────
    enriched, _, _, _ = score_all_legal_moves(board, side_int)
    # enriched is sorted best-first by minimax_score (scorer_agent guarantee)

    # ── Step 2: build candidate list ─────────────────────────────────────────
    if variant == "shortlist_no_safety":
        candidates = select_proposal_candidates(enriched, strategic_context=None, k=5)
        order_mode = "shortlist_order"
        used_shortlist = True
        used_full = False
        shuffle_seed_used = None
    else:  # full_legal_shuffled_no_safety
        candidates = list(enriched)   # copy, do not mutate enriched
        # Deterministic shuffle: combine run seed with scenario_id hash so
        # different scenarios get different orderings while staying reproducible.
        shuffle_seed_used = run_seed ^ (hash(scenario_id) & 0xFFFFFFFF)
        rng = random.Random(shuffle_seed_used)
        rng.shuffle(candidates)
        order_mode = "shuffled_seeded"
        used_shortlist = False
        used_full = True

    n_candidates = len(candidates)
    # index_map: identity (1:1) — no safety filter, no removal
    index_map = list(range(n_candidates))

    # ── Step 3: build synthetic CheckersState (minimal, no game history) ──────
    state = CheckersState(
        board=board,
        current_player=side_int,
        turn_number=0,
        # strategic_context=None → prompt appends "(strategic_context omitted)"
    )
    # Temporarily put candidates into state.legal_moves so build_ranker_user_prompt
    # can read them via its 'filtered' parameter (we pass candidates directly).
    # The state object itself is only used for current_player + turn_number.

    # ── Step 4: build prompt (same template as real ranker) ───────────────────
    system = RANKER_SYSTEM_PROMPT
    # build_ranker_user_prompt(state, filtered, index_map)
    user = build_ranker_user_prompt(state, candidates, index_map)

    # ── Step 5: one raw LLM call — no retry, no fallback ────────────────────
    raw: Optional[str] = None
    api_meta: dict[str, Any] = {
        "api_reached":            False,
        "api_success":            False,
        "api_error_type":         None,
        "api_try_group_count":    0,
        "api_attempt_count":      0,
        "api_retry_count":        0,
        "rate_limit_retry_count": 0,
        "raw_response_present":   False,
    }

    if dry_run:
        raw = json.dumps({"chosen_index": 0, "reasoning": "DRY-RUN"})
        api_meta.update({"api_reached": False, "api_success": True,
                         "raw_response_present": True})
    else:
        try:
            raw = call_ranker(system, user)
            api_meta.update({
                "api_reached": True, "api_success": True,
                "api_attempt_count": 1, "api_try_group_count": 1,
                "raw_response_present": bool(raw and raw.strip()),
            })
        except Exception as exc:
            api_meta.update({
                "api_reached": True,
                "api_error_type": f"b8_call_failed:{type(exc).__name__}",
                "api_attempt_count": 1, "api_try_group_count": 1,
            })
            raw = None

    # ── Step 6: parse — same normalization as FST, no fallback on failure ────
    parsed = _parse_ranker_json(raw) if raw else None
    raw_idx_from_json: Optional[int] = _extract_chosen_index(parsed) if parsed else None
    regex_recovery_used = False

    if raw_idx_from_json is None and raw:
        raw_idx_from_json = _regex_extract_chosen_index(raw)
        regex_recovery_used = raw_idx_from_json is not None

    raw_idx = raw_idx_from_json   # before 1-based correction

    # _resolve_ranker_index: accepts both 0-based and 1-based, returns 0-based
    resolved_idx = _resolve_ranker_index(raw_idx, n_candidates)
    index_was_1_based_corrected = (
        raw_idx is not None
        and resolved_idx is not None
        and raw_idx != resolved_idx          # only true when 1-based → 0-based
    )

    # ── Step 7: resolve to path, no fallback on bad index ────────────────────
    chosen_index_valid = resolved_idx is not None and 0 <= resolved_idx < n_candidates
    if chosen_index_valid:
        chosen_move = candidates[resolved_idx]
        selected_path = [list(sq) for sq in chosen_move.get("path", [])]
        invalid_reason = None
    else:
        selected_path = []
        invalid_reason = (
            "no_valid_index_parsed" if raw_idx is None
            else f"index_out_of_range:{raw_idx}(n={n_candidates})"
        )

    # Produce selected_move schema for evaluate_scenario
    reasoning = ""
    if parsed and isinstance(parsed.get("reasoning"), str):
        reasoning = parsed["reasoning"].strip()

    raw_json_out = json.dumps({
        "selected_move": selected_path,
        "reasoning": reasoning,
    })

    # Save candidate order for diagnostics (paths only, keep compact)
    candidate_order_saved = [
        list(list(sq) for sq in c.get("path", [])) for c in candidates
    ]

    b8_diag: dict[str, Any] = {
        "b8_variant":                       variant,
        "b8_candidate_count":               n_candidates,
        "b8_candidate_order_mode":          order_mode,
        "b8_used_safety_filter":            False,
        "b8_used_proposal_shortlist":       used_shortlist,
        "b8_used_full_legal_set":           used_full,
        "b8_shuffle_seed":                  shuffle_seed_used,
        "b8_raw_chosen_index":              raw_idx,
        "b8_resolved_chosen_index":         resolved_idx,
        "b8_chosen_index_valid":            chosen_index_valid,
        "b8_selected_move":                 selected_path or None,
        "b8_invalid_selection_reason":      invalid_reason,
        "b8_final_legal":                   None,  # filled after evaluate_scenario
        "b8_no_retry_confirmed":            True,
        "b8_no_override_confirmed":         True,
        "b8_no_fallback_confirmed":         True,
        "b8_no_update_agent_confirmed":     True,
        "b8_regex_recovery_used":           regex_recovery_used,
        "b8_index_was_1_based_corrected":   index_was_1_based_corrected,
        "b8_candidate_order_saved":         candidate_order_saved,
        "b8_raw_llm_response":              raw or "",
    }

    return raw_json_out, api_meta, b8_diag




# ── B9 helper — ranker facts, path-output, no safety ─────────────────────────

# System prompt for B9: same facts/context as ranker, but requests selected_move
# coordinates directly instead of chosen_index. Verbatim-copy instruction matches
# the scientific intent of B6/B7 but applied to the richer ranker candidate table.
_B9_SYSTEM_PROMPT = """\
You are selecting ONE legal move for the CURRENT SIDE TO MOVE.

PIECE LEGEND:
  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty

BOARD ORIENTATION:
  Row 0 = top edge.  Row 7 = bottom edge.
  RED pieces move toward lower row numbers (toward row 0).
  BLACK pieces move toward higher row numbers (toward row 7).

CANDIDATE MOVES:
You are given a scored list of legal candidates. Each entry contains:
  - path     : the move expressed as a sequence of [[row, col], ...] squares
  - type     : "simple" or "jump"
  - captured : list of captured-piece squares (jumps only)
  - facts    : symbolic engine facts including minimax_score, safety, and tactics

TASK:
Study the candidates and choose the ONE you judge best.

OUTPUT RULES — CRITICAL:
  You MUST output selected_move as the EXACT path of one displayed candidate.
  Copy the path character-for-character. Do NOT invent new coordinates.
  Do NOT modify, extend, shorten, or "improve" any candidate path.
  Do NOT output a path that does not appear in the candidate list above.

OUTPUT FORMAT — respond with valid JSON only, no prose before or after:
{
  "selected_move": [[row, col], [row, col], ...],
  "reasoning": "brief explanation"
}

selected_move MUST be the exact path of one listed candidate.
Output JSON only.
"""


def _run_b9(
    board: list,
    side_str: str,
    scenario_id: str,
    run_seed: int,
    dry_run: bool = False,
) -> tuple[str, dict, dict]:
    """
    B9: ranker candidate information, path-output schema, NO safety wrapper.

    Uses the same scored shortlist as B8a (score_all_legal_moves +
    select_proposal_candidates, k=5), but asks the LLM to output the
    selected_move path directly instead of chosen_index.

    Bypasses:  _apply_safety_filter  _audit_override  retry  fallback
               repair  update_agent

    Returns (raw_json_str, api_meta, b9_diag).
    raw_json_str uses the legality-pilot selected_move schema so
    evaluate_scenario() works identically to B1-B7.
    """
    (
        score_all_legal_moves,
        select_proposal_candidates,
        _RANKER_SYSTEM_PROMPT_UNUSED,  # not used — B9 has its own prompt
        build_ranker_user_prompt,
        call_ranker,
        _parse_ranker_json,
        _extract_chosen_index,
        _regex_extract_chosen_index,
        _resolve_ranker_index,
        CheckersState,
    ) = _b8_imports()

    from checkers.data.pdn_importer.fen_utils import str_to_side as _str_to_side

    side_int = _str_to_side(side_str)

    # ── Step 1: scored shortlist (same as B8a) ────────────────────────────────
    enriched, _, _, _ = score_all_legal_moves(board, side_int)
    candidates = select_proposal_candidates(enriched, strategic_context=None, k=5)
    n_candidates = len(candidates)

    # ── Step 2: build user prompt (facts table identical to B8a) ─────────────
    # We reuse build_ranker_user_prompt for the candidate facts section,
    # but pair it with _B9_SYSTEM_PROMPT (path-output schema) instead.
    state = CheckersState(
        board=board,
        current_player=side_int,
        turn_number=0,
    )
    index_map = list(range(n_candidates))   # identity, no safety filter
    user = build_ranker_user_prompt(state, candidates, index_map)

    # ── Step 3: one raw LLM call — no retry, no fallback ────────────────────
    raw: Optional[str] = None
    api_meta: dict[str, Any] = {
        "api_reached":            False,
        "api_success":            False,
        "api_error_type":         None,
        "api_try_group_count":    0,
        "api_attempt_count":      0,
        "api_retry_count":        0,
        "rate_limit_retry_count": 0,
        "raw_response_present":   False,
    }

    if dry_run:
        raw = json.dumps({
            "selected_move": candidates[0].get("path", []) if candidates else [],
            "reasoning": "DRY-RUN",
        })
        api_meta.update({"api_reached": False, "api_success": True,
                         "raw_response_present": True})
    else:
        try:
            raw = call_ranker(_B9_SYSTEM_PROMPT, user)
            api_meta.update({
                "api_reached": True, "api_success": True,
                "api_attempt_count": 1, "api_try_group_count": 1,
                "raw_response_present": bool(raw and raw.strip()),
            })
        except Exception as exc:
            api_meta.update({
                "api_reached": True,
                "api_error_type": f"b9_call_failed:{type(exc).__name__}",
                "api_attempt_count": 1, "api_try_group_count": 1,
            })
            raw = None

    # ── Step 4: parse selected_move (B1-B7 schema) ───────────────────────────
    parsed_json: Optional[dict] = None
    selected_path: list = []
    reasoning_text: str = ""
    parse_success = False
    invalid_reason: Optional[str] = None

    if raw:
        try:
            parsed_json = json.loads(raw)
        except json.JSONDecodeError:
            invalid_reason = "json_decode_error"

    if parsed_json is not None:
        mv = parsed_json.get("selected_move")
        if isinstance(mv, list) and len(mv) >= 2:
            # validate every entry is a [row, col] pair
            if all(isinstance(sq, (list, tuple)) and len(sq) == 2 for sq in mv):
                selected_path = [list(sq) for sq in mv]
                parse_success = True
            else:
                invalid_reason = "malformed_path_entries"
        else:
            invalid_reason = "missing_or_short_selected_move"
        r = parsed_json.get("reasoning", "")
        reasoning_text = r.strip() if isinstance(r, str) else ""
    elif raw and invalid_reason is None:
        invalid_reason = "no_raw_response"

    # ── Step 5: check whether selected path matches a candidate ──────────────
    def _path_key(p) -> tuple:
        return tuple(tuple(sq) for sq in p)

    candidate_path_keys = {_path_key(c.get("path", [])) for c in candidates}

    selected_key = _path_key(selected_path) if selected_path else None
    path_matches_candidate = (
        selected_key is not None and selected_key in candidate_path_keys
    )
    path_not_in_candidates = (
        selected_key is not None and selected_key not in candidate_path_keys
    )

    # ── Step 6: assemble raw_json for evaluate_scenario ──────────────────────
    raw_json_out = json.dumps({
        "selected_move": selected_path,
        "reasoning":     reasoning_text,
    })

    b9_diag: dict[str, Any] = {
        "b9_candidate_source":           "shortlist_no_safety",
        "b9_candidate_count":            n_candidates,
        "b9_raw_response":               raw or "",
        "b9_parse_success":              parse_success,
        "b9_selected_move":              selected_path or None,
        "b9_selected_path_matches_candidate": path_matches_candidate,
        "b9_selected_path_not_in_candidates": path_not_in_candidates,
        "b9_final_legal":                None,    # filled after evaluate_scenario
        "b9_invalid_reason":             invalid_reason,
        "b9_no_retry_confirmed":         True,
        "b9_no_override_confirmed":      True,
        "b9_no_fallback_confirmed":      True,
        "b9_no_update_agent_confirmed":  True,
    }

    return raw_json_out, api_meta, b9_diag



# ── B8c helper — ranker facts, comparison-output, no safety ──────────────────

_B8C_SYSTEM_PROMPT = """\
You are a checkers ranker. Your task is to SELECT the BEST move for the CURRENT SIDE TO MOVE.

PIECE LEGEND:
  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty

BOARD ORIENTATION:
  Row 0 = top edge.  Row 7 = bottom edge.
  RED pieces move toward lower row numbers (toward row 0).
  BLACK pieces move toward higher row numbers (toward row 7).

CANDIDATE MOVES:
You are given a scored list of legal move candidates numbered 0, 1, 2, ....
Each candidate has:
  - path     : move coordinates (for reference only — do NOT copy into your output)
  - type     : "simple" or "jump"
  - captured : captured squares (jumps only)
  - facts    : minimax_score, opponent_can_recapture, captures_count, and other symbolic facts

MANDATORY TASK:
1. Analyse EACH candidate individually (pros and cons) using the visible scores and facts.
2. Compare at least two candidates before choosing.
3. Choose the single best candidate.

CRITICAL OUTPUT RULES:
  - Do NOT output selected_move.
  - Do NOT output any coordinate path in your answer.
  - NEVER automatically choose candidate 0 without comparing it to others.
  - You MUST use at least one visible score or fact keyword in your reasoning
    (e.g. minimax_score, opponent_can_recapture, captures_count, safe, threat).
  - If only one candidate exists output chosen_index: 0 and state no comparison was possible.
  - chosen_index MUST be a valid integer index matching one displayed candidate.
  - JSON must be valid and must include the chosen_index field.

OUTPUT FORMAT — respond with valid JSON only, absolutely no prose before or after:
{
  "candidate_analysis": [
    {"index": 0, "pros": "...", "cons": "..."},
    {"index": 1, "pros": "...", "cons": "..."}
  ],
  "chosen_index": <integer>,
  "reasoning": "why this candidate is better than at least one alternative"
}
"""


def _run_b8c(
    board: list,
    side_str: str,
    scenario_id: str,
    dry_run: bool = False,
) -> tuple[str, dict, dict]:
    """
    B8c: ranker-facts candidate list, comparison-output protocol, NO safety.

    Uses the same scored shortlist as B8a but demands:
      - explicit candidate_analysis for each candidate
      - a chosen_index (not selected_move)
      - reasoning that mentions an alternative and a score/fact keyword

    Bypasses: _apply_safety_filter  _audit_override  retry  fallback
              repair  update_agent

    Returns (raw_json_str, api_meta, b8c_diag).
    raw_json_str uses the legality-pilot selected_move schema so
    evaluate_scenario() works identically.
    """
    (
        score_all_legal_moves,
        select_proposal_candidates,
        _RANKER_SYSTEM_PROMPT_UNUSED,
        build_ranker_user_prompt,
        call_ranker,
        _parse_ranker_json,
        _extract_chosen_index,
        _regex_extract_chosen_index,
        _resolve_ranker_index,
        CheckersState,
    ) = _b8_imports()

    from checkers.data.pdn_importer.fen_utils import str_to_side as _str_to_side

    side_int = _str_to_side(side_str)

    # ── Step 1: scored shortlist (same as B8a) ────────────────────────────────
    enriched, _, _, _ = score_all_legal_moves(board, side_int)
    candidates = select_proposal_candidates(enriched, strategic_context=None, k=5)
    n_candidates = len(candidates)

    # ── Step 2: build user prompt (same facts table as B8a) ──────────────────
    state = CheckersState(board=board, current_player=side_int, turn_number=0)
    index_map = list(range(n_candidates))
    user = build_ranker_user_prompt(state, candidates, index_map)

    # ── Step 3: one raw LLM call ─────────────────────────────────────────────
    raw: Optional[str] = None
    api_meta: dict[str, Any] = {
        "api_reached":            False,
        "api_success":            False,
        "api_error_type":         None,
        "api_try_group_count":    0,
        "api_attempt_count":      0,
        "api_retry_count":        0,
        "rate_limit_retry_count": 0,
        "raw_response_present":   False,
    }

    if dry_run:
        # Dry-run: produce a compliant stub with candidate_analysis for all candidates
        analysis_stub = [
            {"index": i, "pros": "DRY-RUN", "cons": "DRY-RUN"}
            for i in range(n_candidates)
        ]
        raw = json.dumps({
            "candidate_analysis": analysis_stub,
            "chosen_index": 0,
            "reasoning": "DRY-RUN comparison using minimax_score",
        })
        api_meta.update({"api_reached": False, "api_success": True,
                         "raw_response_present": True})
    else:
        try:
            raw = call_ranker(_B8C_SYSTEM_PROMPT, user)
            api_meta.update({
                "api_reached": True, "api_success": True,
                "api_attempt_count": 1, "api_try_group_count": 1,
                "raw_response_present": bool(raw and raw.strip()),
            })
        except Exception as exc:
            api_meta.update({
                "api_reached": True,
                "api_error_type": f"b8c_call_failed:{type(exc).__name__}",
                "api_attempt_count": 1, "api_try_group_count": 1,
            })

    # ── Step 4: parse ─────────────────────────────────────────────────────────
    parsed: Optional[dict] = None
    schema_failure_type: Optional[str] = None

    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            schema_failure_type = "json_decode_error"

    # chosen_index extraction (same helpers as B8a — parsing, not safety)
    raw_idx: Optional[int] = None
    if parsed:
        raw_idx = _extract_chosen_index(parsed)
    if raw_idx is None and raw and schema_failure_type is None:
        raw_idx = _regex_extract_chosen_index(raw)

    missing_chosen_index = raw_idx is None
    resolved_idx = _resolve_ranker_index(raw_idx, n_candidates)
    invalid_index = raw_idx is not None and resolved_idx is None

    if missing_chosen_index and schema_failure_type is None:
        schema_failure_type = "missing_chosen_index"
    elif invalid_index and schema_failure_type is None:
        schema_failure_type = "invalid_index"

    chosen_index_valid = resolved_idx is not None

    # candidate_analysis diagnostics
    analysis_list: list = []
    if parsed and isinstance(parsed.get("candidate_analysis"), list):
        analysis_list = parsed["candidate_analysis"]
    candidate_analysis_present = bool(analysis_list)
    candidate_analysis_count = len(analysis_list)

    if not candidate_analysis_present and schema_failure_type is None:
        schema_failure_type = "missing_candidate_analysis"

    # reasoning diagnostics
    reasoning_text: str = ""
    if parsed:
        r = parsed.get("reasoning", "")
        reasoning_text = r.strip() if isinstance(r, str) else ""

    _ALTERNATIVE_KEYWORDS = {
        "better than", "vs", "versus", "compared", "over", "unlike",
        "alternative", "prefer", "instead", "rather",
    }
    _FACT_KEYWORDS = {
        "minimax", "minimax_score", "opponent_can_recapture", "captures_count",
        "safe", "threat", "score", "recapture", "threatened", "king",
        "mobility", "center", "isolated", "blocked",
    }
    rl = reasoning_text.lower()
    reasoning_mentions_alternative = any(k in rl for k in _ALTERNATIVE_KEYWORDS)
    reasoning_mentions_score_or_fact = any(k in rl for k in _FACT_KEYWORDS)

    # also scan candidate_analysis text for fact keywords
    if not reasoning_mentions_score_or_fact:
        for entry in analysis_list:
            combined = (str(entry.get("pros","")) + str(entry.get("cons",""))).lower()
            if any(k in combined for k in _FACT_KEYWORDS):
                reasoning_mentions_score_or_fact = True
                break

    # selected_first_candidate
    selected_first_candidate = (resolved_idx == 0)

    # ── Step 5: build selected_move path for evaluate_scenario ───────────────
    if chosen_index_valid:
        chosen_move = candidates[resolved_idx]
        selected_path = [list(sq) for sq in chosen_move.get("path", [])]
    else:
        selected_path = []

    parse_success = chosen_index_valid   # for the legality evaluator

    raw_json_out = json.dumps({
        "selected_move": selected_path,
        "reasoning":     reasoning_text,
    })

    b8c_diag: dict[str, Any] = {
        "b8c_candidate_count":               n_candidates,
        "b8c_raw_response":                  raw or "",
        "b8c_parse_success":                 parse_success,
        "b8c_missing_chosen_index":          missing_chosen_index,
        "b8c_invalid_index":                 invalid_index,
        "b8c_chosen_index":                  resolved_idx,
        "b8c_selected_first_candidate":      selected_first_candidate,
        "b8c_candidate_analysis_present":    candidate_analysis_present,
        "b8c_candidate_analysis_count":      candidate_analysis_count,
        "b8c_reasoning_mentions_alternative":reasoning_mentions_alternative,
        "b8c_reasoning_mentions_score_or_fact": reasoning_mentions_score_or_fact,
        "b8c_schema_failure_type":           schema_failure_type,
        "b8c_final_legal":                   None,   # filled after evaluate_scenario
        "b8c_no_retry_confirmed":            True,
        "b8c_no_override_confirmed":         True,
        "b8c_no_fallback_confirmed":         True,
        "b8c_no_update_agent_confirmed":     True,
    }

    return raw_json_out, api_meta, b8c_diag

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_JSONL = PROJECT_ROOT / "checkers/data/legality_stress/eval_subset_balanced.jsonl"
HARD_JSONL = PROJECT_ROOT / "checkers/data/legality_stress/hard_subset_balanced.jsonl"
OUT_DIR    = PROJECT_ROOT / "logs/legality_pilot"

# Symbolic control arm — not an LLM baseline, uses no API key.
SYMBOLIC_BASELINE_NAME = "Full_System_Symbolic"
# Real neuro-symbolic pipeline arm.
FST_BASELINE_NAME = "Full_System_Trace"
B5_BASELINE_NAME  = "B5_candidate_moves_rule_filter"
B6_BASELINE_NAME  = "B6_candidate_moves_verbatim"
B7_BASELINE_NAME  = "B7_candidate_moves_path_only"
B8A_BASELINE_NAME = "B8a_ranker_shortlist_no_safety"
B8B_BASELINE_NAME = "B8b_ranker_full_legal_shuffled_no_safety"
B9_BASELINE_NAME  = "B9_ranker_raw_path_no_safety"
B8C_BASELINE_NAME = "B8c_ranker_compare_no_safety"

# Non-LLM arms (share branching logic).
NON_LLM_BASELINES = {SYMBOLIC_BASELINE_NAME, FST_BASELINE_NAME}

# Primary baseline keys only — excludes backward-compat aliases.
ACTIVE_BASELINES = [
    "B1_board_only",
    "B2_rules",
    "B3_rules_structured_checklist",
    "B4_rules_engine_checking",
    B5_BASELINE_NAME,
    B6_BASELINE_NAME,
    B7_BASELINE_NAME,
    B8A_BASELINE_NAME,
    B8B_BASELINE_NAME,
    B9_BASELINE_NAME,
    B8C_BASELINE_NAME,
    SYMBOLIC_BASELINE_NAME,
    FST_BASELINE_NAME,
]
ALL_BASELINES  = ACTIVE_BASELINES   # kept for backward compat with any imports

DEFAULT_N_EVAL = 10
DEFAULT_N_HARD = 10
DEFAULT_SEED   = 42
DEFAULT_DELAY  = 3   # seconds between live API calls

PIECE_SYMBOLS  = {0: ".", 1: "r", 2: "b", 3: "R", 4: "B"}

# ── API client constants ──────────────────────────────────────────────────────
# run_baseline_human_trace.py is kept unchanged; the pilot uses its own
# metadata-aware client to track per-call try-group and rate-limit counts.
_API_KEY  = os.environ.get("BASELINE_MISTRAL_API_KEY", "")
_MODEL    = os.environ.get("BASELINE_MISTRAL_MODEL", "mistral-large-latest")
_TEMP     = float(os.environ.get("BASELINE_MISTRAL_TEMPERATURE", "0.2"))
_MAX_TOK  = int(os.environ.get("BASELINE_MISTRAL_MAX_TOKENS", "1200"))
_API_URL  = "https://api.mistral.ai/v1/chat/completions"

# Retry architecture
# ------------------
# MAX 3 try groups. Each group makes up to 4 attempts using the intra-group
# backoff schedule [20, 30, 40] seconds between the 1st/2nd/3rd/4th attempts.
# A success inside any group stops retrying immediately.
# Non-retriable errors (401, 403, malformed response) exit all groups at once.
_MAX_TRY_GROUPS      = 3
_INTRA_GROUP_WAITS   = [20, 30, 40]   # wait before 2nd, 3rd, 4th attempt in group
_NON_RETRIABLE_CODES = {401, 403}


# ── Metadata-aware API client ─────────────────────────────────────────────────

def _call_api_with_metadata(system: str, user: str) -> tuple[str, dict[str, Any]]:
    """
    Make one Mistral API call using the try-group retry architecture.

    Returns (raw_content, metadata_dict). Never raises.
    raw_content is "" on any failure.

    Retry architecture
    ------------------
    Up to _MAX_TRY_GROUPS (3) independent try groups.
    Within each group, up to (1 + len(_INTRA_GROUP_WAITS)) = 4 attempts,
    with waits of 20 / 30 / 40 seconds between successive failures.
    A success at any attempt stops all retrying immediately.
    Non-retriable errors (401, 403, malformed response) exit all groups.

    metadata keys
    -------------
      api_reached             bool   True when ≥1 HTTP request was sent
      api_success             bool   True when a 200 response with content received
      api_error_type          str|None
      api_try_group_count     int    try groups started (1–3)
      api_attempt_count       int    total HTTP attempts across all groups
      api_retry_count         int    intra-group retries (= api_attempt_count
                                     - api_try_group_count when all groups fail)
      rate_limit_retry_count  int    number of 429 responses received
      raw_response_present    bool   True when raw_content is non-empty
    """
    meta: dict[str, Any] = {
        "api_reached":            False,
        "api_success":            False,
        "api_error_type":         None,
        "api_try_group_count":    0,
        "api_attempt_count":      0,
        "api_retry_count":        0,
        "rate_limit_retry_count": 0,
        "raw_response_present":   False,
    }

    if not _API_KEY:
        meta["api_error_type"] = "no_api_key"
        return "", meta

    payload = {
        "model":           _MODEL,
        "temperature":     _TEMP,
        "max_tokens":      _MAX_TOK,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        _API_URL,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {_API_KEY}",
            "Accept":        "application/json",
        },
        method="POST",
    )

    last_error: str = "unknown"
    max_attempts_per_group = 1 + len(_INTRA_GROUP_WAITS)  # 4

    for group_idx in range(_MAX_TRY_GROUPS):
        meta["api_try_group_count"] = group_idx + 1

        for attempt_in_group in range(max_attempts_per_group):
            # Intra-group wait before 2nd, 3rd, 4th attempt
            if attempt_in_group > 0:
                time.sleep(_INTRA_GROUP_WAITS[attempt_in_group - 1])
                meta["api_retry_count"] += 1

            meta["api_reached"]       = True
            meta["api_attempt_count"] += 1

            try:
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    data    = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        # Malformed response — non-retriable
                        meta["api_error_type"] = "response_malformed"
                        return "", meta
                    meta["api_success"]          = True
                    meta["raw_response_present"] = bool(content.strip())
                    return content, meta

            except urllib.error.HTTPError as exc:
                try:
                    exc.read()
                except Exception:
                    pass
                if exc.code in _NON_RETRIABLE_CODES:
                    # Auth errors — exit all groups immediately
                    meta["api_error_type"] = f"http_{exc.code}_non_retriable"
                    return "", meta
                if exc.code == 429:
                    meta["rate_limit_retry_count"] += 1
                last_error = f"http_{exc.code}"
                # Continue within group (next attempt_in_group iteration)

            except urllib.error.URLError:
                last_error = "network_error"
                # Continue within group

            except (KeyError, IndexError):
                # Malformed response structure — non-retriable
                meta["api_error_type"] = "response_malformed"
                return "", meta

            except Exception as exc:
                last_error = f"unexpected:{type(exc).__name__}"
                # Continue within group

        # Try group exhausted — fall through to next group

    # All try groups exhausted
    meta["api_error_type"] = last_error
    return "", meta


def _run_symbolic(board: list, side_str: str) -> tuple[str, dict]:
    """
    Symbolic control arm: recompute legal moves from board + side using the
    rule engine, then deterministically pick the lex-first move.

    Returns (raw_json, api_meta) with the same shape as _call_api_with_metadata.
    api_reached=False (no network), api_success=True (pipeline treats as success).
    """
    side_int  = str_to_side(side_str)          # "RED" -> RED(1), "BLACK" -> BLACK(2)
    moves     = get_all_legal_moves(board, side_int)

    if not moves:
        # Shouldn't happen in a well-formed scenario; return a parse-able sentinel
        raw = json.dumps({
            "selected_move": [],
            "reasoning": "Full_System_Symbolic: no legal moves found (terminal position).",
        })
    else:
        # Deterministic selection: sort by path tuples, pick first
        moves_sorted = sorted(moves, key=lambda m: m["path"])
        chosen = moves_sorted[0]
        raw = json.dumps({
            "selected_move": [list(rc) for rc in chosen["path"]],
            "reasoning": (
                f"Full_System_Symbolic: rule-engine selected "
                f"{chosen['type']} move {chosen['path']}."
            ),
        })

    meta: dict = {
        "api_reached":            False,   # no network contact
        "api_success":            True,    # pipeline treats as success
        "api_error_type":         None,
        "api_try_group_count":    0,
        "api_attempt_count":      0,
        "api_retry_count":        0,
        "rate_limit_retry_count": 0,
        "raw_response_present":   True,
    }
    return raw, meta


def _run_full_system_trace(board: list, side_str: str) -> tuple[str, dict, dict]:
    """
    Full-System-Trace adapter.

    Calls _stream_one_ply() from run_simplified_trace.py — the real compiled
    LangGraph single-ply runner:
        scorer_node -> deterministic_proposal_node -> ranker_agent -> update_agent

    hidden_legal_moves are NEVER passed to this function or to the graph.
    The graph computes its own legal moves via scorer_node internally.

    After _stream_one_ply, state_manager inside update_agent CLEARS chosen_move
    from the state (to prevent reuse on the next turn).  The applied move is
    preserved in move_history[-1]["move"] — that is where we read it from.

    Returns
    -------
    (raw_json, api_meta, trace_meta)

    raw_json   — {"selected_move": [[r,c],...], "reasoning": ...}
    api_meta   — api_* keys for the result record
    trace_meta — fst_* diagnostic keys
    """
    _checkers_graph, _stream_one_ply, CheckersState = _fst_imports()

    side_int = str_to_side(side_str)

    # Build a fresh, clean state from scenario board + side.
    # No hidden_legal_moves, no move_history, no prior context.
    # game_log_id tags update_agent logs as legality-eval entries.
    state = CheckersState(
        board=board,
        current_player=side_int,
        turn_number=0,
        game_log_id="legality_eval_fst",
    )
    acc = state.model_dump()
    acc["last_completed_node"] = None   # same reset _run_red_ply does

    # Run exactly one ply via the real compiled graph
    acc, success = _stream_one_ply(acc, quiet=True)

    # Extract diagnostics — ranker_diagnostics persists after update_agent
    diag      = acc.get("ranker_diagnostics") or {}
    reasoning = acc.get("last_move_reasoning") or ""

    # Extract the applied move.
    # update_agent / state_manager clears chosen_move after applying it;
    # the move survives in move_history[-1]["move"].
    chosen = None
    mh = acc.get("move_history") or []
    if mh:
        chosen = mh[-1].get("move")          # primary: move actually applied
    if not chosen:
        chosen = acc.get("chosen_move")      # fallback A: not yet cleared
    if not chosen:
        _rp = diag.get("raw_llm_choice_path")
        if _rp:
            chosen = {"path": _rp}           # fallback B: ranker choice path

    # FST diagnostic fields
    raw_llm_idx      = diag.get("raw_llm_idx")
    raw_llm_path     = diag.get("raw_llm_choice_path")
    final_choice_src = diag.get("final_choice_source", "unknown")
    override_fired   = bool(diag.get("override_retry_attempts", 0) > 0)
    retry_count      = int(diag.get("override_retry_attempts", 0))
    fallback_used    = bool(diag.get("override_fallback_applied", False))
    api_fail_count   = int(diag.get("api_call_failure_count", 0))
    raw_valid        = raw_llm_idx is not None

    if chosen:
        final_path = [list(rc) for rc in chosen.get("path", [])]
    else:
        final_path = []

    raw_json = json.dumps({
        "selected_move": final_path,
        "reasoning":     reasoning,
    })

    # api_meta mirrors B1-B4 shape
    api_meta: dict = {
        "api_reached":            True,
        "api_success":            bool(final_path),
        "api_error_type":         None if final_path else "ranker_no_chosen_move",
        "api_try_group_count":    1,
        "api_attempt_count":      1 + api_fail_count,
        "api_retry_count":        retry_count,
        "rate_limit_retry_count": 0,
        "raw_response_present":   True,
    }

    trace_meta: dict = {
        "fst_stream_one_ply_success":             success,
        "fst_raw_llm_candidate_selection_valid":  raw_valid,
        "fst_raw_llm_selected_index":             raw_llm_idx,
        "fst_raw_llm_selected_candidate_path":    (
            [list(rc) for rc in raw_llm_path] if raw_llm_path else None
        ),
        "fst_final_selected_move":                final_path,
        "fst_final_choice_source":                final_choice_src,
        "fst_override_fired":                     override_fired,
        "fst_retry_count":                        retry_count,
        "fst_fallback_used":                      fallback_used,
        "fst_api_call_failure_count":             api_fail_count,
    }

    return raw_json, api_meta, trace_meta


def _dry_run_with_meta() -> tuple[str, dict[str, Any]]:
    """Return a canned wrong response for dry-run (no network call)."""
    raw = json.dumps({
        "selected_move": [[5, 0], [4, 1]],
        "reasoning": "DRY-RUN: canned response — not a real LLM call.",
    })
    meta: dict[str, Any] = {
        "api_reached":            False,   # no network contact
        "api_success":            True,    # pipeline treats it as success
        "api_error_type":         None,
        "api_try_group_count":    0,
        "api_attempt_count":      0,
        "api_retry_count":        0,
        "rate_limit_retry_count": 0,
        "raw_response_present":   True,
    }
    return raw, meta


# ── Loaders & samplers ────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_scenarios(scenarios: list[dict], n: int, seed: int, label: str) -> list[dict]:
    rng  = random.Random(seed)
    pool = list(scenarios)
    rng.shuffle(pool)
    if n > len(pool):
        print(
            f"  [{label}] WARNING: requested {n} but only {len(pool)} available — "
            f"using all {len(pool)}."
        )
    selected = pool[:n]
    print(f"  [{label}] sampled {len(selected)} / {len(scenarios)} (seed={seed})")
    return selected


def filter_by_side(scenarios: list[dict], side_filter: str | None) -> list[dict]:
    """
    Return only scenarios where side_to_move matches *side_filter*.
    ``side_filter`` must be "RED", "BLACK", or None (keep all).
    Matching is case-insensitive.
    """
    if side_filter is None or side_filter.upper() == "BOTH":
        return scenarios
    target = side_filter.upper()
    if target not in ("RED", "BLACK"):
        raise ValueError(f"--side-filter must be RED, BLACK, or BOTH; got {side_filter!r}")
    return [sc for sc in scenarios if sc.get("side_to_move", "").upper() == target]


# ── Display helpers ───────────────────────────────────────────────────────────

def _move_str(mv: dict) -> str:
    path = "→".join(f"({r},{c})" for r, c in mv["path"])
    caps = mv.get("captured", [])
    return f"[{mv['type'].upper()}] {path}" + (f"  captures={caps}" if caps else "")


def preview_prompts(sample: list[dict], baselines: list[str],
                    show_ground_truth: bool = False) -> None:
    """Print a prompt preview for the first scenario, restricted to selected baselines."""
    if not sample:
        return
    sc  = sample[0]
    sid = sc["scenario_id"]
    print("\n" + "═" * 70)
    print(f"PROMPT PREVIEW — {sid}  (side: {sc['side_to_move']})")
    print("  hidden_legal_moves are NOT in any prompt below.")
    print("═" * 70)
    for bname in baselines:                          # ← only selected baselines
        print(f"\n{'─'*70}\nBASELINE: {bname}\n{'─'*70}")
        if bname in NON_LLM_BASELINES:
            if bname == SYMBOLIC_BASELINE_NAME:
                print("  [Symbolic control arm — no LLM prompt]")
                print("  Moves recomputed by rule engine at evaluation time.")
            else:
                print("  [Full-System-Trace — uses real compiled graph]")
                print("  scorer_node → proposal → ranker_agent → update_agent")
                print("  No system prompt shown here; ranker builds its own internally.")
        elif bname == B5_BASELINE_NAME:
            cand_info = get_candidates(sc["board"], RED if sc["side_to_move"] == "RED" else BLACK)
            cands = cand_info["candidates"]
            user  = build_b5_user_prompt(sc["board"], sc["side_to_move"], sid, cands)
            system = BASELINES[bname]
            print("── SYSTEM (first 5 lines) ──")
            for line in system.splitlines()[:5]:
                print("  " + line)
            print("  [...]")
            print(f"── USER PROMPT (candidates: {len(cands)}, "
                  f"jumps: {cand_info['capture_candidate_count']}, "
                  f"simples: {cand_info['simple_candidate_count']}) ──")
            print(user)
        elif bname == B6_BASELINE_NAME:
            cand_info = get_candidates(sc["board"], RED if sc["side_to_move"] == "RED" else BLACK)
            cands = cand_info["candidates"]
            user  = build_b6_user_prompt(sc["board"], sc["side_to_move"], sid, cands)
            system = BASELINES[bname]
            print("── SYSTEM (first 5 lines) ──")
            for line in system.splitlines()[:5]:
                print("  " + line)
            print("  [...]")
            print(f"── USER PROMPT (candidates: {len(cands)}, "
                  f"jumps: {cand_info['capture_candidate_count']}, "
                  f"simples: {cand_info['simple_candidate_count']}) ──")
            print(user)
        elif bname == B7_BASELINE_NAME:
            cand_info = get_candidates(sc["board"], RED if sc["side_to_move"] == "RED" else BLACK)
            cands = cand_info["candidates"]
            user  = build_b7_user_prompt(sc["board"], sc["side_to_move"], sid, cands)
            system = BASELINES[bname]
            print("── SYSTEM (first 5 lines) ──")
            for line in system.splitlines()[:5]:
                print("  " + line)
            print("  [...]")
            print(f"── USER PROMPT (candidates: {len(cands)}, "
                  f"jumps: {cand_info['capture_candidate_count']}, "
                  f"simples: {cand_info['simple_candidate_count']}) ──")
            print(user)
        elif bname in (B8A_BASELINE_NAME, B8B_BASELINE_NAME):
            variant = (
                "shortlist_no_safety" if bname == B8A_BASELINE_NAME
                else "full_legal_shuffled_no_safety"
            )
            _raw, _, _diag = _run_b8(
                sc["board"], sc["side_to_move"], sid, variant,
                run_seed=DEFAULT_SEED, dry_run=True,
            )
            print(f"── B8 variant: {variant} ──")
            print(f"   candidates : {_diag['b8_candidate_count']}  "
                  f"order : {_diag['b8_candidate_order_mode']}")
            print(f"   (prompt built inside _run_b8 via build_ranker_user_prompt)")
        elif bname == B9_BASELINE_NAME:
            _raw9, _, _diag9 = _run_b9(
                sc["board"], sc["side_to_move"], sid,
                run_seed=DEFAULT_SEED, dry_run=True,
            )
            print(f"── B9 (ranker facts, path-output, no safety) ──")
            print(f"   candidates : {_diag9['b9_candidate_count']}  "
                  f"source : {_diag9['b9_candidate_source']}")
            print(f"   (prompt built inside _run_b9; output schema: selected_move)")
        elif bname == B8C_BASELINE_NAME:
            _rawc, _, _diagc = _run_b8c(
                sc["board"], sc["side_to_move"], sid, dry_run=True,
            )
            print(f"── B8c (ranker facts, comparison-output, no safety) ──")
            print(f"   candidates : {_diagc['b8c_candidate_count']}  "
                  f"schema : candidate_analysis + chosen_index")
            print(f"   (prompt built inside _run_b8c via build_ranker_user_prompt)")
        else:
            system = BASELINES[bname]

            user   = build_user_prompt(sc["board"], sc["side_to_move"], sid)
            print("── SYSTEM (first 5 lines) ──")
            for line in system.splitlines()[:5]:
                print("  " + line)

            print("  [...]")
            print("── USER PROMPT ──")
            print(user)
    print(f"\n{'─'*70}")
    if show_ground_truth:
        print(f"GROUND TRUTH (evaluator only) — {len(sc['hidden_legal_moves'])} hidden moves:")
        for mv in sc["hidden_legal_moves"]:
            print(f"  {_move_str(mv)}")
    else:
        print(f"GROUND TRUTH hidden ({len(sc['hidden_legal_moves'])} moves) "
              f"— use --show-ground-truth to display.")
    print("═" * 70 + "\n")


def show_sample_results(results: list[dict], n: int = 5) -> None:
    print("\n" + "═" * 70)
    print(f"SAMPLE RESULTS — first {n} records")
    print("═" * 70)
    for i, r in enumerate(results[:n], 1):
        rt = r["result_type"]
        print(
            f"\n[{i:02d}] {r['scenario_id']}  baseline={r['baseline']}\n"
            f"      cat={r['category']}  diff={r['difficulty']}  n_legal={r['n_legal']}\n"
            f"      result_type={rt}  api_ok={r['api_success']}  "
            f"parse_ok={r['parse_success']}\n"
            f"      selected_path={r.get('selected_path')}  "
            f"rate_limit_retries={r['rate_limit_retry_count']}\n"
            f"      reasoning: {textwrap.shorten(str(r.get('reasoning','')), 70)}"
        )


# ── Core runner ───────────────────────────────────────────────────────────────



def _apply_salvage(
    record: dict,
    raw_llm_response: str,
    hidden: list,
    board: list,
    side: str,
    candidates: Optional[list] = None,
) -> dict:
    """
    Attach raw_salvage_* fields to record.
    Called after every record build; only does work for parse_failure records.
    Never mutates result_type, legal, parse_success, or legal_move_rate.
    """
    salvage = salvage_parse_failure(
        record=record,
        raw_llm_response=raw_llm_response,
        hidden_legal_moves=hidden,
        board=board,
        side_str=side,
        candidates=candidates,
    )
    return {**record, **salvage}

def run_pilot(
    scenarios:    list[dict],
    baselines:    list[str],
    dry_run:      bool = False,
    show_prompts: bool = False,
    verbose:      bool = False,
    delay_sec:    float = 3.0,
    seed:         int   = DEFAULT_SEED,
) -> list[dict]:
    """
    Run each scenario through each baseline and return result records.

    hidden_legal_moves are NEVER passed to LLM prompts.
    Only the evaluator (evaluate_scenario) reads hidden_legal_moves.
    """
    results: list[dict[str, Any]] = []
    args_seed: int = seed   # threaded into _run_b8 for B8b shuffle
    first_call = True

    for sc in scenarios:
        sid        = sc["scenario_id"]
        board      = sc["board"]
        side       = sc["side_to_move"]
        hidden     = sc["hidden_legal_moves"]
        category   = sc.get("category",   "unknown")
        difficulty = sc.get("difficulty", "unknown")
        source     = sc.get("source_file", "unknown")

        if verbose:
            print(
                f"  ▶ {sid}  cat={category}  diff={difficulty}  n_legal={len(hidden)}"
            )

        for bname in baselines:
            if bname == SYMBOLIC_BASELINE_NAME:
                raw_output, api_meta = _run_symbolic(board, side)
                if show_prompts:
                    print(f"\n[{bname}] (symbolic — no LLM call)\n")
                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )
                record: dict[str, Any] = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   "",
                    "dry_run":       False,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                }
                results.append(record)
                continue

            # ── Full-System-Trace arm (real pipeline LLM call) ─────────────
            if bname == FST_BASELINE_NAME:
                raw_output, api_meta, trace_meta = _run_full_system_trace(board, side)
                if show_prompts:
                    print(f"\n[{bname}] (real pipeline — scorer→proposal→ranker)\n")
                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )
                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   "",   # ranker prompt not logged here (internal)
                    "dry_run":       False,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                    **trace_meta,   # fst_* diagnostic fields
                }
                if verbose:
                    src  = trace_meta.get("fst_final_choice_source", "?")
                    ovr  = trace_meta.get("fst_override_fired", False)
                    fb   = trace_meta.get("fst_fallback_used", False)
                    print(
                        f"      [{bname}] result={result_type}  "
                        f"src={src}  override={ovr}  fallback={fb}"
                    )
                results.append(record)
                continue

            # ── B5 candidate-assisted LLM arm ────────────────────────────────
            if bname == B5_BASELINE_NAME:
                side_int = RED if side == "RED" else BLACK
                cand_info = get_candidates(board, side_int)
                cands     = cand_info["candidates"]
                user      = build_b5_user_prompt(board, side, sid, cands)
                system    = BASELINES[bname]
                if show_prompts:
                    print(f"\n[{bname}] USER:\n{user}\n")

                # Rate-limit delay
                if not dry_run and delay_sec > 0 and not first_call:
                    time.sleep(delay_sec)
                first_call = False

                if dry_run:
                    raw_output, api_meta = _dry_run_with_meta()
                else:
                    raw_output, api_meta = _call_api_with_metadata(system, user)

                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )

                # B5-specific diagnostics
                sel_path = eval_result.get("selected_path")   # list[list] or None
                cand_diag = match_selected_to_candidate(sel_path, cands)

                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   user,
                    "dry_run":       dry_run,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                    # B5 candidate diagnostics
                    "b5_candidate_count":         len(cands),
                    "b5_capture_candidate_count": cand_info["capture_candidate_count"],
                    "b5_simple_candidate_count":  cand_info["simple_candidate_count"],
                    "b5_any_jump_available":      cand_info["any_jump_available"],
                    **{f"b5_{k}": v for k, v in cand_diag.items()},
                }
                if verbose:
                    print(
                        f"      [{bname}] result={result_type}  "
                        f"candidates={len(cands)}  "
                        f"jumps={cand_info['capture_candidate_count']}  "
                        f"simples={cand_info['simple_candidate_count']}"
                    )
                results.append(record)
                continue

            # ── B6 candidate-verbatim arm ─────────────────────────────────────
            if bname == B6_BASELINE_NAME:
                side_int  = RED if side == "RED" else BLACK
                cand_info = get_candidates(board, side_int)
                cands     = cand_info["candidates"]
                user      = build_b6_user_prompt(board, side, sid, cands)
                system    = BASELINES[bname]
                if show_prompts:
                    print(f"\n[{bname}] USER:\n{user}\n")

                if not dry_run and delay_sec > 0 and not first_call:
                    time.sleep(delay_sec)
                first_call = False

                if dry_run:
                    raw_output, api_meta = _dry_run_with_meta()
                else:
                    raw_output, api_meta = _call_api_with_metadata(system, user)

                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )

                # Extract selected_candidate_id from raw JSON (B6-specific field)
                try:
                    _raw_dict = json.loads(raw_output)
                    raw_cand_id = _raw_dict.get("selected_candidate_id")
                except Exception:
                    raw_cand_id = None

                sel_path  = eval_result.get("selected_path")
                b6_diag   = match_b6_response(raw_cand_id, sel_path, cands)

                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   user,
                    "dry_run":       dry_run,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                    # B6 candidate pool metadata
                    "b6_candidate_count":         len(cands),
                    "b6_capture_candidate_count": cand_info["capture_candidate_count"],
                    "b6_simple_candidate_count":  cand_info["simple_candidate_count"],
                    "b6_any_jump_available":      cand_info["any_jump_available"],
                    # B6 per-response diagnostics
                    **b6_diag,
                }
                if verbose:
                    cid   = b6_diag.get("b6_selected_candidate_id", "?")
                    valid = b6_diag.get("b6_selected_candidate_id_valid", False)
                    match = b6_diag.get("b6_selected_move_matches_candidate_id", False)
                    print(
                        f"      [{bname}] result={result_type}  "
                        f"claimed_id={cid}  id_valid={valid}  path_matches={match}"
                    )
                results.append(record)
                continue

            # ── B7 candidate-verbatim, path-only arm ─────────────────────────
            if bname == B7_BASELINE_NAME:
                side_int  = RED if side == "RED" else BLACK
                cand_info = get_candidates(board, side_int)
                cands     = cand_info["candidates"]
                user      = build_b7_user_prompt(board, side, sid, cands)
                system    = BASELINES[bname]
                if show_prompts:
                    print(f"\n[{bname}] USER:\n{user}\n")

                if not dry_run and delay_sec > 0 and not first_call:
                    time.sleep(delay_sec)
                first_call = False

                if dry_run:
                    raw_output, api_meta = _dry_run_with_meta()
                else:
                    raw_output, api_meta = _call_api_with_metadata(system, user)

                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )

                sel_path = eval_result.get("selected_path")
                b7_diag  = match_b7_response(sel_path, cands)

                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   user,
                    "dry_run":       dry_run,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                    "b7_candidate_count":         len(cands),
                    "b7_capture_candidate_count": cand_info["capture_candidate_count"],
                    "b7_simple_candidate_count":  cand_info["simple_candidate_count"],
                    "b7_any_jump_available":      cand_info["any_jump_available"],
                    **b7_diag,
                }
                if verbose:
                    not_found = b7_diag.get("b7_selected_path_not_in_candidates", True)
                    mtype     = b7_diag.get("b7_selected_candidate_move_type", "?")
                    print(
                        f"      [{bname}] result={result_type}  "
                        f"path_in_cands={not not_found}  type={mtype}"
                    )
                results.append(record)
                continue

            # ── B8a / B8b ranker-no-safety arms ─────────────────────────────
            if bname in (B8A_BASELINE_NAME, B8B_BASELINE_NAME):
                variant = (
                    "shortlist_no_safety" if bname == B8A_BASELINE_NAME
                    else "full_legal_shuffled_no_safety"
                )

                if not dry_run and delay_sec > 0 and not first_call:
                    time.sleep(delay_sec)
                first_call = False

                raw_output, api_meta, b8_diag = _run_b8(
                    board, side, sid, variant,
                    run_seed=args_seed,
                    dry_run=dry_run,
                )

                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )
                b8_diag["b8_final_legal"] = eval_result["legal"]

                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   "",   # ranker prompt not re-logged here
                    "dry_run":       dry_run,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                    **b8_diag,
                }
                _b8_raw = b8_diag.get("b8_raw_llm_response", "")
                _b8_cands = (
                    # reconstruct from candidate_order_saved paths if present
                    [{"path": p} for p in b8_diag.get("b8_candidate_order_saved", [])]
                    if b8_diag.get("b8_candidate_order_saved") else None
                )
                record = _apply_salvage(record, _b8_raw, hidden, board, side, _b8_cands)
                if verbose:
                    idx   = b8_diag.get("b8_resolved_chosen_index", "?")
                    valid = b8_diag.get("b8_chosen_index_valid", False)
                    cands = b8_diag.get("b8_candidate_count", 0)
                    regex = b8_diag.get("b8_regex_recovery_used", False)
                    print(
                        f"      [{bname}] result={result_type}  "
                        f"idx={idx}/{cands}  valid={valid}  "
                        f"regex_recovery={regex}"
                    )
                results.append(record)
                continue

            # ── B9 ranker facts / path-output / no safety arm ───────────────
            if bname == B9_BASELINE_NAME:
                if not dry_run and delay_sec > 0 and not first_call:
                    time.sleep(delay_sec)
                first_call = False

                raw_output, api_meta, b9_diag = _run_b9(
                    board, side, sid,
                    run_seed=args_seed,
                    dry_run=dry_run,
                )

                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )
                b9_diag["b9_final_legal"] = eval_result["legal"]

                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   "",   # ranker prompt internal
                    "dry_run":       dry_run,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                    **b9_diag,
                }
                record = _apply_salvage(record, b9_diag.get("b9_raw_response", ""),
                                        hidden, board, side)
                if verbose:
                    in_cands = b9_diag.get("b9_selected_path_matches_candidate", False)
                    not_cand = b9_diag.get("b9_selected_path_not_in_candidates", False)
                    cands    = b9_diag.get("b9_candidate_count", 0)
                    print(
                        f"      [{bname}] result={result_type}  "
                        f"cands={cands}  "
                        f"path_in_cands={in_cands}  path_invented={not_cand}"
                    )
                results.append(record)
                continue

            # ── B8c ranker comparison arm ────────────────────────────────────
            if bname == B8C_BASELINE_NAME:
                if not dry_run and delay_sec > 0 and not first_call:
                    time.sleep(delay_sec)
                first_call = False

                raw_output, api_meta, b8c_diag = _run_b8c(
                    board, side, sid, dry_run=dry_run,
                )

                eval_result = evaluate_scenario(raw_output, hidden, board, side)
                result_type = (
                    "legal" if eval_result["legal"]
                    else "parse_failure" if not eval_result["parse_success"]
                    else "illegal"
                )
                b8c_diag["b8c_final_legal"] = eval_result["legal"]

                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   "",   # ranker prompt internal
                    "dry_run":       dry_run,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                    **b8c_diag,
                }
                # B8c saves its raw LLM string as b8c_raw_response
                record = _apply_salvage(record, b8c_diag.get("b8c_raw_response", ""),
                                        hidden, board, side)
                if verbose:
                    analysis_ok = b8c_diag.get("b8c_candidate_analysis_present", False)
                    chosen      = b8c_diag.get("b8c_chosen_index", "?")
                    first       = b8c_diag.get("b8c_selected_first_candidate", False)
                    alt_mention = b8c_diag.get("b8c_reasoning_mentions_alternative", False)
                    print(
                        f"      [{bname}] result={result_type}  "
                        f"chosen={chosen}  first={first}  "
                        f"analysis={analysis_ok}  alt_mention={alt_mention}"
                    )
                results.append(record)
                continue

            # ── LLM baselines ───────────────────────────────────────────────
            system = BASELINES[bname]

            # Build prompt — hidden_legal_moves NOT passed
            user = build_user_prompt(board, side, sid)

            if show_prompts:
                print(f"\n[{bname}] USER:\n{user}\n")

            # Rate-limit delay (live mode only, not before the very first call)
            if not dry_run and delay_sec > 0 and not first_call:
                time.sleep(delay_sec)
            first_call = False

            # API call
            if dry_run:
                raw_output, api_meta = _dry_run_with_meta()
            else:
                raw_output, api_meta = _call_api_with_metadata(system, user)

            # Evaluate — only when API succeeded
            if not api_meta["api_success"]:
                # API failure: do NOT evaluate, do NOT count as legal or illegal
                record: dict[str, Any] = {
                    "scenario_id":           sid,
                    "baseline":              bname,
                    "category":              category,
                    "difficulty":            difficulty,
                    "source_file":           source,
                    "side_to_move":          side,
                    "n_legal":               len(hidden),
                    "user_prompt":           user,
                    "dry_run":               dry_run,
                    "result_type":           "api_failure",
                    "parse_success":         False,
                    "parse_error":           api_meta.get("api_error_type", "api_failure"),
                    "legal":                 False,
                    "illegal_move_type":     "",
                    "wrong_direction":       None,
                    "mandatory_violation":   False,
                    "multi_jump_incomplete": False,
                    "selected_path":         None,
                    "raw_output":            "",
                    "reasoning":             "",
                    **api_meta,
                }
            else:
                eval_result = evaluate_scenario(raw_output, hidden, board, side)

                if not eval_result["parse_success"]:
                    result_type = "parse_failure"
                elif eval_result["legal"]:
                    result_type = "legal"
                else:
                    result_type = "illegal"

                record = {
                    "scenario_id":   sid,
                    "baseline":      bname,
                    "category":      category,
                    "difficulty":    difficulty,
                    "source_file":   source,
                    "side_to_move":  side,
                    "n_legal":       len(hidden),
                    "user_prompt":   user,
                    "dry_run":       dry_run,
                    "result_type":   result_type,
                    **api_meta,
                    **eval_result,
                }
                # salvage: raw_output IS the raw LLM string for B1-B7 arms
                record = _apply_salvage(record, raw_output, hidden, board, side)

            results.append(record)

    return results


# ── Export ────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def save_report(text: str, out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ── Privacy assertion ─────────────────────────────────────────────────────────

def assert_privacy(results: list[dict]) -> None:
    FORBIDDEN = ["hidden_legal_moves", "hidden_legal", "ground_truth"]
    violations = []
    for r in results:
        for kw in FORBIDDEN:
            if kw in r.get("user_prompt", ""):
                violations.append(f"{r['scenario_id']}/{r['baseline']}: '{kw}'")
    if violations:
        for v in violations:
            print(f"[PRIVACY VIOLATION] {v}")
        raise AssertionError("hidden_legal_moves leaked into prompts!")
    print("✓ Privacy check: hidden_legal_moves confirmed absent from all prompts.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legality-stress pilot evaluation runner")
    p.add_argument("--baselines", nargs="+", default=ACTIVE_BASELINES,
                   choices=ACTIVE_BASELINES)
    p.add_argument("--n-eval",          type=int,   default=DEFAULT_N_EVAL)
    p.add_argument("--n-hard",          type=int,   default=DEFAULT_N_HARD)
    p.add_argument("--seed",            type=int,   default=DEFAULT_SEED)
    p.add_argument("--request-delay",   type=float, default=DEFAULT_DELAY,
                   help="Seconds between live API calls (default 3; ignored in dry-run)")
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--show-prompts",    action="store_true")
    p.add_argument("--show-ground-truth", action="store_true",
                   help="Print hidden_legal_moves in the prompt preview (never sent to LLM).")
    p.add_argument("--verbose",         action="store_true")
    p.add_argument(
        "--side-filter",
        choices=["RED", "BLACK", "BOTH"],
        default=None,
        metavar="{RED,BLACK,BOTH}",
        help=(
            "Filter scenarios by side_to_move before sampling. "            "RED → only RED scenarios; BLACK → only BLACK; default = both sides."        ),
    )
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("LEGALITY-STRESS PILOT  (schema: selected_move)")
    print(f"  Baselines     : {args.baselines}")
    print(f"  n_eval/n_hard : {args.n_eval}/{args.n_hard}  seed={args.seed}")
    _sf_display = args.side_filter if args.side_filter else "BOTH (default)"
    print(f"  side_filter   : {_sf_display}")
    print(f"  dry_run       : {args.dry_run}")
    print(f"  request_delay : {args.request_delay}s  "
          f"({'ignored — dry-run' if args.dry_run else 'applied between calls'})")
    print("=" * 70)

    print("\nLoading scenario files …")
    eval_all = load_jsonl(EVAL_JSONL)
    hard_all = load_jsonl(HARD_JSONL)
    print(f"  eval_subset_balanced : {len(eval_all)} scenarios")
    print(f"  hard_subset_balanced : {len(hard_all)} scenarios")

    # Apply side filter BEFORE sampling so seed behaviour is deterministic
    # within the filtered pool.
    eval_pool = filter_by_side(eval_all, args.side_filter)
    hard_pool = filter_by_side(hard_all, args.side_filter)
    if args.side_filter:
        print(f"  After side_filter={args.side_filter}: "
              f"eval={len(eval_pool)}  hard={len(hard_pool)}")

    print("\nSampling …")
    eval_sample = sample_scenarios(eval_pool, args.n_eval, args.seed,     "eval")
    hard_sample = sample_scenarios(hard_pool, args.n_hard, args.seed + 1, "hard")
    pilot       = eval_sample + hard_sample
    print(f"  Total : {len(pilot)} scenarios")

    preview_prompts(pilot, baselines=args.baselines,
                    show_ground_truth=args.show_ground_truth)

    mode = "[DRY RUN]" if args.dry_run else "[LIVE]"
    print(f"\n{mode} Running {len(args.baselines)} baseline(s) × {len(pilot)} scenarios …\n")
    results = run_pilot(
        pilot,
        baselines=args.baselines,
        dry_run=args.dry_run,
        show_prompts=args.show_prompts,
        verbose=args.verbose,
        delay_sec=args.request_delay,
        seed=args.seed,
    )
    print(f"Total records: {len(results)}")

    show_sample_results(results, n=5)

    baseline_metrics: dict[str, dict] = {}
    for bname in args.baselines:
        subset = [r for r in results if r["baseline"] == bname]
        baseline_metrics[bname] = aggregate(subset)

    # Run label: INCOMPLETE if any baseline had API failures
    any_api_failure = any(
        m.get("total_api_failures", 0) > 0
        for m in baseline_metrics.values()
    )
    run_label = "INCOMPLETE_FOR_FINAL_EVALUATION" if any_api_failure else "API_COMPLETE"

    _sf_tag = f"_side={args.side_filter}" if args.side_filter else ""
    report = format_report(
        pilot_name=(
            f"pilot_{ts}{_sf_tag}"
            + (" [DRY RUN]" if args.dry_run else "")
            + (f" [side={args.side_filter}]" if args.side_filter else "")
        ),
        baseline_metrics=baseline_metrics,
        n_scenarios=len(pilot),
        run_label=run_label,
    )
    print(f"\nRun label: {run_label}")
    print("\n" + report)

    ts_tag = ts + (_sf_tag if args.side_filter else "")
    rp = save_results(results, OUT_DIR, ts_tag)
    rr = save_report(report,   OUT_DIR, ts_tag)
    print(f"\nSaved results → {rp}")
    print(f"Saved report  → {rr}")

    assert_privacy(results)


if __name__ == "__main__":
    main()
