#!/usr/bin/env python3
"""
checkers/tests/probe_board_proposal_quality.py  (v6)

Standalone quality probe for board_proposal_agent.

Runs board_proposal_agent on 25 hard board scenarios (both RED and BLACK),
then post-hoc evaluates every proposal against get_all_legal_moves() + _moves_match().
The engine is used ONLY here for testing — it is never exposed to the agent.

Metrics per scenario / per run:
  precision                    = legal_proposed / total_proposed
  recall                       = legal_proposed / true_legal_count
  f1                           = harmonic mean of precision and recall
  any_legal_found              — at least one proposal matches a legal move
  fake_jump_count              — proposals with type='jump' that have no matching legal jump
  quiet_when_capture_mandatory — proposals with type='simple' during mandatory capture
  capture_estimate_correct     — whether LLM's capture_available_estimate matched engine truth
  scan_present                 — LLM included a non-empty scan field
  scan_board_mismatch_count    — scan entries where mid_val or land_val doesn't match actual board
  scan_contradiction_count     — scan entries with valid=true but claimed values don't support it
  capture_scan_consistent      — capture_available_estimate agrees with scan's valid=true entries
  proposal_uses_unverified_jump — jump proposals with no backing valid=true scan entry

Usage:
    python checkers/tests/probe_board_proposal_quality.py                   # all 25 scenarios
    python checkers/tests/probe_board_proposal_quality.py <name>            # single scenario
    python checkers/tests/probe_board_proposal_quality.py --repeat 3        # 3 runs per scenario
    python checkers/tests/probe_board_proposal_quality.py <name> --repeat 5 --verbose
    python checkers/tests/probe_board_proposal_quality.py --delay 3.0       # inter-call delay (s)
    python checkers/tests/probe_board_proposal_quality.py --list            # list names and exit
    python checkers/tests/probe_board_proposal_quality.py --dry-run         # show legal counts, no LLM

Environment:
    GITHUB_MODELS_API_KEY (or MISTRAL_API_KEY) — loaded from .env automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any

# ── Make "checkers.*" importable when running directly ────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv optional; set MISTRAL_API_KEY manually if absent

from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING,
    create_initial_board,
)
from checkers.engine.rules import get_all_legal_moves, _moves_match
from checkers.agents.board_proposal_agent import board_proposal_agent, _count_grounding_failures
from checkers.state.state import CheckersState

# Set by main() from --verbose flag; read by _print_result / _print_aggregated.
_VERBOSE: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Board helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eb() -> list[list[int]]:
    """Return a fresh empty 8×8 board."""
    return [[EMPTY] * 8 for _ in range(8)]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions  (13 hard boards, covering both RED and BLACK)
# ─────────────────────────────────────────────────────────────────────────────

def _build_scenarios() -> list[dict]:
    """
    Returns a list of 25 scenario dicts, each with:
      name               — unique slug (pass to CLI to run just this one)
      description        — one-line summary
      board              — 8×8 list[list[int]]
      current_player     — RED or BLACK constant
      expected_has_jumps — True if engine mandates capture on this board
      notes              — what failure pattern to watch for
    """
    s: list[dict] = []

    # ── S1: Initial board, BLACK plays ────────────────────────────────────────
    # 12 BLACK pieces rows 0-2, 12 RED rows 5-7. No captures (6 rows apart).
    # 7 legal forward simples for BLACK (from the 4 row-2 pieces).
    b = create_initial_board()
    s.append({
        "name":               "initial_no_capture_black",
        "description":        "Standard opening, BLACK plays — 7 legal simples, zero captures",
        "board":              b,
        "current_player":     BLACK,
        "expected_has_jumps": False,
        "notes":              "Any type='jump' = fake jump. Core regression (no v3 regressions allowed).",
    })

    # ── S2: Initial board, RED plays ──────────────────────────────────────────
    b = create_initial_board()
    s.append({
        "name":               "initial_no_capture_red",
        "description":        "Standard opening, RED plays — 7 legal simples, zero captures",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Any type='jump' = fake jump. Core regression test.",
    })

    # ── S3: Forced single capture — RED ───────────────────────────────────────
    # RED [5,2] jumps BLACK [4,3] → lands [3,4] (dark 3+4=7).  1 legal move.
    b = _eb()
    b[5][2] = RED
    b[4][3] = BLACK
    s.append({
        "name":               "forced_single_capture_red",
        "description":        "RED man [5,2] can jump BLACK [4,3] to [3,4] — 1 legal jump",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "Must propose type='jump'. Any simple = quiet_when_mandatory error.",
    })

    # ── S4: Forced single capture — BLACK ─────────────────────────────────────
    # BLACK [2,3] jumps RED [3,4] → lands [4,5] (dark 4+5=9).  1 legal move.
    b = _eb()
    b[2][3] = BLACK
    b[3][4] = RED
    s.append({
        "name":               "forced_single_capture_black",
        "description":        "BLACK man [2,3] jumps RED [3,4] to [4,5] — 1 legal jump",
        "board":              b,
        "current_player":     BLACK,
        "expected_has_jumps": True,
        "notes":              "Tests BLACK jump direction (row+1). Any simple = quiet_mandatory.",
    })

    # ── S5: Multiple capture choices ──────────────────────────────────────────
    # RED [5,0] → BLACK [4,1] → land [3,2] (dark 3+2=5).
    # RED [5,4] → BLACK [4,5] → land [3,6] (dark 3+6=9).
    b = _eb()
    b[5][0] = RED;  b[4][1] = BLACK
    b[5][4] = RED;  b[4][5] = BLACK
    s.append({
        "name":               "multiple_capture_choices",
        "description":        "2 RED men with distinct captures — 2 legal jumps, zero simples allowed",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "LLM must propose both captures. Simples = quiet_mandatory error.",
    })

    # ── S6: Multi-jump available — RED ────────────────────────────────────────
    # RED [5,2] → BLACK [4,3] → land [3,4] → BLACK [2,5] → land [1,6] (dark 1+6=7).
    # Only 1 legal move: full 2-jump path [[5,2],[3,4],[1,6]].
    # Single-jump [[5,2],[3,4]] is INVALID (engine forces continuation).
    b = _eb()
    b[5][2] = RED
    b[4][3] = BLACK
    b[2][5] = BLACK
    s.append({
        "name":               "multi_jump_available_red",
        "description":        "RED [5,2] chains 2 jumps →[3,4]→[1,6] — partial path is INVALID",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "Only full 2-jump path is legal. Partial [[5,2],[3,4]] = illegal.",
    })

    # ── S7: Multi-jump available — BLACK ──────────────────────────────────────
    # BLACK [2,5] → RED [3,4] → land [4,3] → RED [5,2] → land [6,1] (dark 6+1=7).
    # Only 1 legal move: full 2-jump path [[2,5],[4,3],[6,1]].
    b = _eb()
    b[2][5] = BLACK
    b[3][4] = RED
    b[5][2] = RED
    s.append({
        "name":               "forced_multi_jump_black",
        "description":        "BLACK [2,5] chains 2 jumps →[4,3]→[6,1] — tests BLACK multi-jump (row+1)",
        "board":              b,
        "current_player":     BLACK,
        "expected_has_jumps": True,
        "notes":              "Only full 2-jump path is legal. Tests BLACK direction AND multi-jump.",
    })

    # ── S8: Fake capture over own piece ───────────────────────────────────────
    # RED [5,2], own RED at [4,3] (midpoint!), empty [3,4].
    # [4,3] is OWN piece → condition ② fails → NOT a jump.
    # Legal: 5 simple moves (from [5,2]→[4,1], [4,3]→{[3,2],[3,4]}, [5,6]→{[4,5],[4,7]}).
    b = _eb()
    b[5][2] = RED
    b[4][3] = RED   # own piece at "midpoint" position
    b[5][6] = RED
    s.append({
        "name":               "fake_capture_over_own_piece",
        "description":        "RED [5,2] with OWN RED at [4,3] — midpoint is own piece, NOT a jump",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Jump [5,2]→[3,4] is ILLEGAL. Tests condition ②: midpoint must be OPPONENT.",
    })

    # ── S9: King backward move ─────────────────────────────────────────────────
    # RED_KING at [4,3] (dark 4+3=7), all 4 diagonal neighbors empty.
    # Legal: 4 simple moves to [3,2],[3,4],[5,2],[5,4].
    b = _eb()
    b[4][3] = RED_KING
    s.append({
        "name":               "king_backward_move",
        "description":        "RED_KING at [4,3] — 4 legal moves including backward diagonals [5,2] and [5,4]",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Missing [5,2] or [5,4] = king direction bug. Recall < 100% = partial miss.",
    })

    # ── S10: Man backward move illegal ─────────────────────────────────────────
    # RED man at [4,3] with forward [3,2] and [3,4] both blocked by own RED.
    # Backward [5,2] and [5,4] are dark+empty but ILLEGAL for a man.
    # RED [6,1] provides 2 legal moves ([5,0] and [5,2]).
    # RED [3,2] and [3,4] each provide 2 more legal moves. Total: 6.
    b = _eb()
    b[4][3] = RED     # forward blocked
    b[3][2] = RED
    b[3][4] = RED
    b[6][1] = RED
    s.append({
        "name":               "man_backward_illegal",
        "description":        "RED man [4,3] with forward blocked — backward moves [5,2]/[5,4] are ILLEGAL",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Proposals [4,3]→[5,2] or [4,3]→[5,4] are illegal backward moves for a man.",
    })

    # ── S11: Promotion position ────────────────────────────────────────────────
    # RED man at [1,2] (dark 1+2=3) — one step from row 0 (promotion row).
    # Moves [1,2]→[0,1] and [1,2]→[0,3] both promote to RED_KING.
    # RED [3,4] adds [2,3] and [2,5]. Total: 4 legal simples.
    b = _eb()
    b[1][2] = RED
    b[3][4] = RED
    s.append({
        "name":               "promotion_position",
        "description":        "RED man [1,2] one step from promotion row 0 — 4 legal simples",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Moves [1,2]→[0,1] and [1,2]→[0,3] promote. LLM must include them.",
    })

    # ── S12: Promotion during jump — must stop ─────────────────────────────────
    # RED [2,1] jumps BLACK [1,2] → lands [0,3] (dark 0+3=3) → PROMOTES → stop.
    # BLACK [1,4] is temptation: as RED_KING from [0,3] one might try [1,4]→[2,5],
    # but promotion ends the turn. Exactly 1 legal move: [[2,1],[0,3]].
    b = _eb()
    b[2][1] = RED
    b[1][2] = BLACK    # captured; RED lands at [0,3] and promotes
    b[1][4] = BLACK    # temptation — continuation after promotion is ILLEGAL
    s.append({
        "name":               "promotion_during_jump_stop",
        "description":        "RED [2,1] captures [1,2], promotes at [0,3] — must stop; no continuation",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "Only [[2,1],[0,3]] is legal. Multi-jump past [0,3] = INVALID.",
    })

    # ── S13: Edge-of-board pieces ──────────────────────────────────────────────
    # BLACK men at left/right edges — each has exactly 1 valid forward move.
    # [5,0] dark (5+0=5): only [6,1] forward; [6,-1] is out of bounds.
    # [2,7] dark (2+7=9): only [3,6] forward; [3,8] is out of bounds.
    # [1,2] dark (1+2=3): [2,1] and [2,3] — 2 moves. Total: 4 legal simples.
    b = _eb()
    b[5][0] = BLACK    # left edge — 1 forward move
    b[2][7] = BLACK    # right edge — 1 forward move
    b[1][2] = BLACK    # interior — 2 forward moves
    s.append({
        "name":               "edge_of_board",
        "description":        "BLACK men at edges [5,0] and [2,7] — each has exactly 1 valid forward move",
        "board":              b,
        "current_player":     BLACK,
        "expected_has_jumps": False,
        "notes":              "Out-of-bounds targets (col -1, col 8, row -1, row 8) are illegal.",
    })

    # ── S14: Dense board, no captures — ownership confusion test ────────────
    # Row 5: 4 RED men; row 4: 4 BLACK men; row 3: 4 RED men (12 pieces total).
    # Every potential RED jump from row 5 has its landing square in row 3 blocked by
    # own RED.  Only row-3 RED pieces can move forward (to row 2).  7 legal simples.
    b = _eb()
    b[5][0] = RED;   b[5][2] = RED;   b[5][4] = RED;   b[5][6] = RED
    b[4][1] = BLACK; b[4][3] = BLACK; b[4][5] = BLACK; b[4][7] = BLACK
    b[3][0] = RED;   b[3][2] = RED;   b[3][4] = RED;   b[3][6] = RED
    s.append({
        "name":               "ownership_dense_no_capture",
        "description":        "12 pieces in rows 3-5 — every jump landing blocked by own RED; 7 legal simples",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Any fake_jump = ownership confusion. Row-5 pieces contribute 0 proposals.",
    })

    # ── S15: Both forward diagonals of focus piece blocked by own pieces ─────
    # RED [5,4] has [4,3] and [4,5] occupied by own RED → 0 valid simple moves
    # from [5,4].  Other RED pieces ([4,3],[4,5],[6,1]) provide 6 legal simples.
    # Tests SIMPLE-GATE: simple_checks for [5,4] must show valid=false on both dirs.
    b = _eb()
    b[5][4] = RED   # focus piece — both forward diagonals blocked
    b[4][3] = RED   # blocks [5,4]→[4,3]
    b[4][5] = RED   # blocks [5,4]→[4,5]
    b[6][1] = RED   # anchor piece providing valid simples
    s.append({
        "name":               "occupied_simple_target",
        "description":        "RED [5,4] both NW/NE blocked by own pieces — must contribute 0 proposals",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Any proposal from [5,4] is illegal. Tests simple_checks valid=false for blocked.",
    })

    # ── S16: Landing occupied — fake jump test ────────────────────────────────
    # RED [5,2]: midpoint [4,3]=BLACK (opponent ✓), but landing [3,4]=own RED (✗).
    # Jump [5,2]→[3,4] must NOT be proposed (jump condition 3 fails: landing occupied).
    # 5 legal simples: [5,2]→[4,1], [5,6]→{[4,5],[4,7]}, [3,4]→{[2,3],[2,5]}.
    b = _eb()
    b[5][2] = RED
    b[4][3] = BLACK   # opponent midpoint — tempting but landing is blocked
    b[3][4] = RED     # own piece at landing square — NOT a valid jump
    b[5][6] = RED
    s.append({
        "name":               "landing_occupied_fake_jump",
        "description":        "RED [5,2]: opponent at [4,3] but landing [3,4] is own RED — jump is ILLEGAL",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Jump [5,2]→[3,4] is illegal (landing occupied). Tests jump condition 3.",
    })

    # ── S17: Midpoint empty — fake jump test ──────────────────────────────────
    # BLACK [3,2] and [3,6]: RED pieces sit 2 rows below ([5,0],[5,4]).
    # Midpoint squares [4,1],[4,3],[4,5],[4,7] are all empty — no valid jump.
    # LLM might spatially confuse distant RED pieces as adjacent midpoints.
    # 4 legal simples (forward); 0 captures.
    b = _eb()
    b[3][2] = BLACK
    b[3][6] = BLACK
    b[5][4] = RED   # distractor: 2 rows below [3,2/3,6] — midpoints are EMPTY
    b[5][0] = RED   # distractor
    s.append({
        "name":               "midpoint_empty_fake_jump",
        "description":        "BLACK [3,2]/[3,6] face RED 2 rows away — midpoints [4,*] are empty, no jump valid",
        "board":              b,
        "current_player":     BLACK,
        "expected_has_jumps": False,
        "notes":              "Fake jump [3,2]→[5,4] treats empty [4,3] as opponent. Tests jump condition 2.",
    })

    # ── S18: King backward capture ────────────────────────────────────────────
    # RED_KING [4,3]: opponent BLACK [5,2] is on the SW diagonal (backward for RED).
    # Landing [6,1] is empty.  Mandatory capture → 1 legal move: jump [4,3]→[6,1].
    # Tests that the LLM recognises backward king captures (SW/SE for RED_KING).
    b = _eb()
    b[4][3] = RED_KING
    b[5][2] = BLACK   # backward midpoint (SW of king)
    # [6,1] = EMPTY landing (dark: 6+1=7)
    s.append({
        "name":               "king_jump_backward",
        "description":        "RED_KING [4,3] captures backward (SW): jumps BLACK [5,2] to land [6,1]",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "Backward king capture. Any simple = quiet_mandatory. Miss = king direction bug.",
    })

    # ── S19: King must NOT jump over own piece ────────────────────────────────
    # RED_KING [4,3]: own RED [3,4] sits on the NE diagonal; [2,5] is empty beyond.
    # Jump [4,3]→[2,5] via own [3,4] is ILLEGAL (condition 2: midpoint must be opponent).
    # Legal: [4,3]→{[3,2],[5,2],[5,4]} + [3,4]→{[2,3],[2,5]} = 5 legal simples.
    b = _eb()
    b[4][3] = RED_KING
    b[3][4] = RED   # own piece — blocks NE jump; [2,5] is empty beyond
    s.append({
        "name":               "king_no_fake_jump_over_own",
        "description":        "RED_KING [4,3]: own RED [3,4] with empty [2,5] beyond — must NOT jump own piece",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Jump [4,3]→[2,5] via own [3,4] is ILLEGAL. Tests king jump condition 2.",
    })

    # ── S20: 3-leg multi-jump — RED ───────────────────────────────────────────
    # RED [7,0] chains 3 NE jumps along the main diagonal: →[5,2]→[3,4]→[1,6].
    # Captured: [6,1],[4,3],[2,5].  Only 1 legal move: the full 3-leg path.
    # Partial paths (1- or 2-leg) are INVALID (engine forces continuation).
    b = _eb()
    b[7][0] = RED
    b[6][1] = BLACK   # mid-1
    b[4][3] = BLACK   # mid-2
    b[2][5] = BLACK   # mid-3
    # landings [5,2],[3,4],[1,6] all empty
    s.append({
        "name":               "three_leg_multi_jump_red",
        "description":        "RED [7,0] chains 3 NE jumps →[5,2]→[3,4]→[1,6] — full path required",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "Partial paths (1- or 2-leg) are ILLEGAL. len(path)=4, len(captured)=3.",
    })

    # ── S21: 3-leg multi-jump — BLACK ─────────────────────────────────────────
    # BLACK [0,7] chains 3 SW jumps along the main diagonal: →[2,5]→[4,3]→[6,1].
    # Captured: [1,6],[3,4],[5,2].  Tests multi-jump in the BLACK forward direction.
    b = _eb()
    b[0][7] = BLACK
    b[1][6] = RED   # mid-1
    b[3][4] = RED   # mid-2
    b[5][2] = RED   # mid-3
    # landings [2,5],[4,3],[6,1] all empty
    s.append({
        "name":               "three_leg_multi_jump_black",
        "description":        "BLACK [0,7] chains 3 SW jumps →[2,5]→[4,3]→[6,1] — tests BLACK multi-jump",
        "board":              b,
        "current_player":     BLACK,
        "expected_has_jumps": True,
        "notes":              "Partial paths (1- or 2-leg) are ILLEGAL. len(path)=4, len(captured)=3.",
    })

    # ── S22: Promotion during jump — must stop ────────────────────────────────
    # RED [2,5] jumps BLACK [1,4] → lands [0,3] (row 0 = promotion row) → STOP.
    # BLACK [1,2] is a temptation: as RED_KING from [0,3], SW→cap[1,2]→land[2,1]
    # appears valid, but American checkers rule: promotion ends the turn immediately.
    # Only 1 legal move: [[2,5],[0,3]].
    b = _eb()
    b[2][5] = RED
    b[1][4] = BLACK   # captured; RED promotes at [0,3]
    b[1][2] = BLACK   # temptation — multi-leg continuation after promotion is ILLEGAL
    s.append({
        "name":               "promotion_jump_must_stop",
        "description":        "RED [2,5] captures [1,4], promotes at [0,3] — must stop despite BLACK [1,2] nearby",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "Only [[2,5],[0,3]] is legal. Continuation past promotion row = INVALID.",
    })

    # ── S23: King at corner — out-of-bounds filtering ─────────────────────────
    # RED_KING [0,7] sits in the top-right corner: 3 of 4 diagonals are OOB.
    # Only SW=[1,6] is in-bounds.  RED_KING [2,1] (interior) contributes 4 simples.
    # Total: 5 legal simples.  Tests that OOB squares are never proposed.
    b = _eb()
    b[0][7] = RED_KING   # corner — only [1,6] valid
    b[2][1] = RED_KING   # interior king — 4 valid directions
    s.append({
        "name":               "edge_king_corner",
        "description":        "RED_KING [0,7] at corner — 3/4 diagonals OOB; only [1,6] valid",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "Proposals with row<0/row>7/col<0/col>7 from [0,7] are OOB and illegal.",
    })

    # ── S24: All-kings endgame — exhaustive recall test ───────────────────────
    # 3 RED_KINGs spread across the board, no opponents.
    # Each king can move in all 4 diagonal directions → 12 legal simples total.
    # Tests exhaustive recall for kings in a quiet position.
    b = _eb()
    b[1][2] = RED_KING   # 4 moves: [0,1],[0,3],[2,1],[2,3]
    b[3][4] = RED_KING   # 4 moves: [2,3],[2,5],[4,3],[4,5]
    b[5][6] = RED_KING   # 4 moves: [4,5],[4,7],[6,5],[6,7]
    s.append({
        "name":               "all_kings_endgame_no_capture",
        "description":        "3 RED_KINGs — 12 legal simples total, no captures; tests king recall",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": False,
        "notes":              "recall < 100% = king direction or enumeration miss. No captures to distract.",
    })

    # ── S25: Mandatory capture with many simples available ────────────────────
    # 4 RED men in row 5; BLACK [4,3] creates 2 forced captures:
    #   [5,2] NE → cap[4,3] → land[3,4]  and  [5,4] NW → cap[4,3] → land[3,2].
    # The other row-5 pieces ([5,0],[5,6]) have apparent simples that are ILLEGAL
    # during mandatory capture.  2 legal jumps; 0 legal simples.
    b = _eb()
    b[5][0] = RED
    b[5][2] = RED   # can jump NE: mid=[4,3]=BLACK, land=[3,4]
    b[5][4] = RED   # can jump NW: mid=[4,3]=BLACK, land=[3,2]
    b[5][6] = RED
    b[4][3] = BLACK
    s.append({
        "name":               "mandatory_capture_many_quiets_available",
        "description":        "4 RED men with apparent simples, but BLACK [4,3] forces 2 captures only",
        "board":              b,
        "current_player":     RED,
        "expected_has_jumps": True,
        "notes":              "2 legal jumps: [5,2]→[3,4] and [5,4]→[3,2]. Any simple = quiet_mandatory.",
    })

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_legal(proposal: dict, legal_moves: list[dict]) -> bool:
    """True if proposal matches any engine legal move by type + full path."""
    return any(_moves_match(proposal, lm) for lm in legal_moves)


def _fmt_move(m: dict) -> str:
    """One-line move description: type:from→to  cap [r,c],..."""
    try:
        path_str = "→".join(f"[{r},{c}]" for r, c in m["path"])
    except Exception:
        path_str = str(m.get("path", "?"))
    cap = m.get("captured", [])
    cap_str = ("  cap " + ",".join(f"[{r},{c}]" for r, c in cap)) if cap else ""
    return f"{m.get('type', '?')}:{path_str}{cap_str}"


def _check_path_format(move: dict) -> list[str]:
    """
    Returns a list of path-format error strings for a jump move.
    Only applies to type='jump'; returns [] for simple moves.

    Errors detected:
      length_mismatch  — len(path) != len(captured) + 1
      step_size_error  — consecutive path entries are not exactly 2 diagonal steps apart
                         (happens when the LLM includes the midpoint inside path)
      midpoint_in_path — a captured midpoint square appears inside path
    """
    if move.get("type") != "jump":
        return []

    errors: list[str] = []
    path     = move.get("path",     [])
    captured = move.get("captured", [])

    # 1. Length invariant
    if len(path) != len(captured) + 1:
        errors.append(
            f"length_mismatch: len(path)={len(path)} != len(captured)+1={len(captured)+1}"
        )

    # 2. Consecutive path entries must be 2 diagonal steps apart
    for i in range(len(path) - 1):
        try:
            dr = abs(int(path[i + 1][0]) - int(path[i][0]))
            dc = abs(int(path[i + 1][1]) - int(path[i][1]))
            if dr != 2 or dc != 2:
                errors.append(
                    f"step_size_error: path[{i}]={list(path[i])}→path[{i+1}]={list(path[i+1])}"
                    f"  |Δrow|={dr} |Δcol|={dc}  (expected 2,2 — midpoint was included?)"
                )
        except (IndexError, TypeError, ValueError):
            errors.append(f"step_error: cannot compute delta for path[{i}]→path[{i+1}]")

    # 3. Captured midpoints must NOT appear in path
    try:
        path_set = {(int(sq[0]), int(sq[1])) for sq in path}
        for mid in captured:
            key = (int(mid[0]), int(mid[1]))
            if key in path_set:
                errors.append(f"midpoint_in_path: captured {list(mid)} appears in path")
    except (TypeError, IndexError, ValueError):
        errors.append("midpoint_check_error: could not compare captured vs path")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Scan analysis helpers  (Python reads board AFTER proposal — never leaked)
# ─────────────────────────────────────────────────────────────────────────────

_INT_TO_SYM: dict[int, str] = {
    EMPTY:      ".",
    RED:        "r",
    BLACK:      "b",
    RED_KING:   "R",
    BLACK_KING: "B",
}
_OPP_SYMS: dict[int, frozenset] = {
    RED:   frozenset({"b", "B"}),
    BLACK: frozenset({"r", "R"}),
}


def _board_sym(board: list[list[int]], row: int, col: int) -> str:
    """Symbol at (row, col), or '#' if out-of-bounds or light square."""
    if not (0 <= row <= 7 and 0 <= col <= 7):
        return "#"
    return _INT_TO_SYM.get(board[row][col], "?")


def _analyze_scan(scan_data: Any, board: list[list[int]]) -> dict:
    """
    Compares the LLM's scan claims against the actual board.
    Called AFTER the proposal — board is never passed to the agent.

    Returns:
      scan_present              — bool
      n_scan_entries            — int
      n_jump_checks             — int
      n_valid_scan_jumps        — int   (jump_checks with valid=true in scan)
      scan_board_mismatch_count — int   (mid_val or land_val disagrees with board)
      mismatches                — list of detail dicts for verbose output
    """
    if not isinstance(scan_data, list) or not scan_data:
        return {
            "scan_present":              False,
            "n_scan_entries":            0,
            "n_jump_checks":             0,
            "n_valid_scan_jumps":        0,
            "scan_board_mismatch_count": 0,
            "mismatches":                [],
        }

    mismatches: list[dict] = []
    n_checks = 0
    n_valid  = 0

    for entry in scan_data:
        if not isinstance(entry, dict):
            continue
        piece      = entry.get("piece", [])
        piece_type = entry.get("piece_type", "?")

        for chk in entry.get("jump_checks", []):
            if not isinstance(chk, dict):
                continue
            n_checks += 1
            direction = chk.get("dir", "?")

            mid = chk.get("mid")
            if isinstance(mid, (list, tuple)) and len(mid) == 2:
                mr, mc            = int(mid[0]), int(mid[1])
                mid_val_actual    = _board_sym(board, mr, mc)
                mid_val_claimed   = str(chk.get("mid_val", "?"))
                if mid_val_actual != mid_val_claimed:
                    mismatches.append({
                        "piece": piece, "piece_type": piece_type, "dir": direction,
                        "field": "mid_val", "at": [mr, mc],
                        "claimed": mid_val_claimed, "actual": mid_val_actual,
                    })

            land = chk.get("land")
            if isinstance(land, (list, tuple)) and len(land) == 2:
                lr, lc             = int(land[0]), int(land[1])
                land_val_actual    = _board_sym(board, lr, lc)
                land_val_claimed   = str(chk.get("land_val", "?"))
                if land_val_actual != land_val_claimed:
                    mismatches.append({
                        "piece": piece, "piece_type": piece_type, "dir": direction,
                        "field": "land_val", "at": [lr, lc],
                        "claimed": land_val_claimed, "actual": land_val_actual,
                    })

            if chk.get("valid") is True:
                n_valid += 1

    return {
        "scan_present":              True,
        "n_scan_entries":            len(scan_data),
        "n_jump_checks":             n_checks,
        "n_valid_scan_jumps":        n_valid,
        "scan_board_mismatch_count": len(mismatches),
        "mismatches":                mismatches,
    }


def _count_scan_contradictions(scan_data: Any, player: int) -> int:
    """
    Counts jump_check entries where valid=true but the claimed mid_val/land_val
    don't actually satisfy the jump conditions (internal inconsistency).
    valid=true requires mid_val ∈ {opponent symbols} AND land_val == '.'.
    """
    if not isinstance(scan_data, list):
        return 0
    opp_syms = _OPP_SYMS[player]
    count = 0
    for entry in scan_data:
        if not isinstance(entry, dict):
            continue
        for chk in entry.get("jump_checks", []):
            if not isinstance(chk, dict) or chk.get("valid") is not True:
                continue
            mid_val  = str(chk.get("mid_val",  "?"))
            land_val = str(chk.get("land_val", "?"))
            if mid_val not in opp_syms or land_val != ".":
                count += 1
    return count


def _count_unverified_jumps(proposals: list[dict], scan_data: Any) -> int:
    """
    Counts jump proposals whose first captured square has no matching
    valid=true jump_check in scan from the same starting piece.
    A proposal is "unverified" when the LLM skipped the scan gate.
    """
    if not isinstance(scan_data, list):
        return sum(1 for p in proposals if p.get("type") == "jump")

    valid_pairs: set[tuple] = set()
    for entry in scan_data:
        if not isinstance(entry, dict):
            continue
        piece = entry.get("piece", [])
        if not (isinstance(piece, (list, tuple)) and len(piece) == 2):
            continue
        start_key = (int(piece[0]), int(piece[1]))
        for chk in entry.get("jump_checks", []):
            if not isinstance(chk, dict) or chk.get("valid") is not True:
                continue
            mid = chk.get("mid")
            if isinstance(mid, (list, tuple)) and len(mid) == 2:
                valid_pairs.add((start_key, (int(mid[0]), int(mid[1]))))

    count = 0
    for prop in proposals:
        if prop.get("type") != "jump":
            continue
        path     = prop.get("path",     [])
        captured = prop.get("captured", [])
        if not path or not captured:
            count += 1
            continue
        try:
            start     = (int(path[0][0]),     int(path[0][1]))
            first_cap = (int(captured[0][0]), int(captured[0][1]))
        except (IndexError, TypeError, ValueError):
            count += 1
            continue
        if (start, first_cap) not in valid_pairs:
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Simple-checks analysis helpers  (Python reads board AFTER proposal — never leaked)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_simple_checks(simple_checks_data: Any, board: list[list[int]]) -> dict:
    """
    Compares the LLM's simple_checks entries against the actual board (post-hoc).
    Never called before the proposal; board is never exposed to the agent.

    Returns:
      simple_checks_present      — bool
      n_simple_checks            — int
      simple_board_mismatch_count— int  (to_val claimed by LLM ≠ actual board)
      simple_mismatches          — list of detail dicts for verbose output
    """
    if not isinstance(simple_checks_data, list) or not simple_checks_data:
        return {
            "simple_checks_present":       False,
            "n_simple_checks":             0,
            "simple_board_mismatch_count": 0,
            "simple_mismatches":           [],
        }

    mismatches: list[dict] = []
    for chk in simple_checks_data:
        if not isinstance(chk, dict):
            continue
        to = chk.get("to")
        if isinstance(to, (list, tuple)) and len(to) == 2:
            tr, tc          = int(to[0]), int(to[1])
            actual          = _board_sym(board, tr, tc)
            claimed         = str(chk.get("to_val", "?"))
            if actual != claimed:
                mismatches.append({
                    "from":    chk.get("from"),
                    "to":      [tr, tc],
                    "dir":     chk.get("dir"),
                    "claimed": claimed,
                    "actual":  actual,
                })

    return {
        "simple_checks_present":       True,
        "n_simple_checks":             len(simple_checks_data),
        "simple_board_mismatch_count": len(mismatches),
        "simple_mismatches":           mismatches,
    }


def _count_probe_simple_unbacked(proposals: list[dict], simple_checks_data: Any) -> int:
    """
    Counts type='simple' proposals that have no backing valid=true simple_checks entry
    (from == path[0], to == path[1]).  Mirrors _count_unverified_jumps for simples.
    """
    if not isinstance(simple_checks_data, list):
        return sum(1 for p in proposals if p.get("type") == "simple")

    valid_pairs: set[tuple] = set()
    for chk in simple_checks_data:
        if not isinstance(chk, dict) or chk.get("valid") is not True:
            continue
        frm = chk.get("from")
        to  = chk.get("to")
        if (isinstance(frm, (list, tuple)) and len(frm) == 2
                and isinstance(to, (list, tuple)) and len(to) == 2):
            valid_pairs.add(((int(frm[0]), int(frm[1])), (int(to[0]), int(to[1]))))

    count = 0
    for prop in proposals:
        if prop.get("type") != "simple":
            continue
        path = prop.get("path", [])
        if len(path) < 2:
            count += 1
            continue
        try:
            frm_key = (int(path[0][0]), int(path[0][1]))
            to_key  = (int(path[1][0]), int(path[1][1]))
        except (IndexError, TypeError, ValueError):
            count += 1
            continue
        if (frm_key, to_key) not in valid_pairs:
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (single run)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scenario(scenario: dict, delay_s: float = 0.0) -> dict:
    """
    Runs board_proposal_agent once on *scenario*, then evaluates proposals
    against get_all_legal_moves() + _moves_match().  Returns a metrics dict.
    """
    if delay_s > 0:
        time.sleep(delay_s)

    board  = scenario["board"]
    player = scenario["current_player"]
    state  = CheckersState(board=board, current_player=player)

    # Ground truth — computed AFTER proposal call to prevent any leakage
    true_legal            = get_all_legal_moves(board, player)
    has_mandatory_capture = any(m["type"] == "jump" for m in true_legal)

    agent_error: str | None = None
    agent_output: dict = {}
    try:
        agent_output = board_proposal_agent(state)
    except Exception:
        agent_error = traceback.format_exc()

    raw       = agent_output.get("board_proposal_raw", "")
    proposals = agent_output.get("board_proposal_moves", [])
    diag      = agent_output.get("board_proposal_diagnostics", {})

    legal_proposals:   list[dict] = []
    illegal_proposals: list[dict] = []
    fake_jumps:        list[dict] = []
    quiet_mandatory:   list[dict] = []
    path_format_errors: list[dict] = []    # {"move": …, "errors": [str, …]}

    for prop in proposals:
        if _is_legal(prop, true_legal):
            legal_proposals.append(prop)
        else:
            illegal_proposals.append(prop)
            if prop.get("type") == "jump":
                fake_jumps.append(prop)
        if has_mandatory_capture and prop.get("type") == "simple":
            quiet_mandatory.append(prop)
        fmt_errs = _check_path_format(prop)
        if fmt_errs:
            path_format_errors.append({"move": prop, "errors": fmt_errs})

    missed = [
        lm for lm in true_legal
        if not any(_moves_match(prop, lm) for prop in proposals)
    ]

    n_prop  = len(proposals)
    n_legal = len(true_legal)
    prec    = len(legal_proposals) / n_prop  if n_prop  > 0 else 0.0
    rec     = len(legal_proposals) / n_legal if n_legal > 0 else 0.0
    f1      = 2 * prec * rec / (prec + rec)  if (prec + rec) > 0 else 0.0

    llm_est = diag.get("llm_capture_estimate")
    est_ok: bool | None = (
        None if llm_est is None else (llm_est == has_mandatory_capture)
    )

    # ── Scan analysis (post-hoc, board never exposed to agent) ───────────────
    llm_scan     = diag.get("llm_scan")
    scan_anal    = _analyze_scan(llm_scan, board)
    scan_contrad = _count_scan_contradictions(llm_scan, player)
    unverif_j    = _count_unverified_jumps(proposals, llm_scan)

    n_valid_in_scan = scan_anal["n_valid_scan_jumps"]
    if llm_scan is not None:
        cap_scan_consistent: bool | None = (
            bool(llm_est) == (n_valid_in_scan > 0)
        )
    else:
        cap_scan_consistent = None

    # ── Simple-checks analysis (post-hoc, board never exposed to agent) ───────
    llm_simple_checks = diag.get("llm_simple_checks")
    simple_anal       = _analyze_simple_checks(llm_simple_checks, board)
    simple_unbacked   = _count_probe_simple_unbacked(proposals, llm_simple_checks)

    # ── Grounding analysis (source_check_id/ids — uses raw pre-normalise list) ─
    llm_final_raw = diag.get("llm_final_raw")
    grounding     = _count_grounding_failures(llm_final_raw, llm_scan, llm_simple_checks)

    n_contradiction_retries      = diag.get("contradiction_retry_count", 0)
    contradiction_reasons        = diag.get("contradiction_reasons", [])
    post_retry_still_contr       = diag.get("post_retry_still_contradictory", False)
    dropped_unverified_after     = diag.get("dropped_unverified_after_retry_count", 0)
    dropped_bad_source_after     = diag.get("dropped_bad_source_after_retry_count", 0)
    safe_rejection_count         = diag.get("safe_rejection_count", 0)

    valid_sc_count        = diag.get("valid_simple_checks_count", 0)
    missing_final_from_sc = diag.get("missing_final_moves_from_valid_simple_checks_count", 0)
    final_sc_rate         = diag.get("final_simple_completeness_rate", 1.0)

    return {
        "name":                               scenario["name"],
        "description":                        scenario["description"],
        "notes":                              scenario["notes"],
        "expected_has_jumps":                 scenario["expected_has_jumps"],
        "current_player":                     "RED" if player == RED else "BLACK",
        # Ground truth
        "true_legal_moves":                   true_legal,
        "n_true_legal":                       n_legal,
        "has_mandatory_capture":              has_mandatory_capture,
        # Proposals
        "raw_response":                       raw,
        "proposals":                          proposals,
        "n_proposals":                        n_prop,
        # Classified proposals
        "legal_proposals":                    legal_proposals,
        "illegal_proposals":                  illegal_proposals,
        "missed_legal":                       missed,
        "fake_jumps":                         fake_jumps,
        "quiet_when_capture_mandatory":       quiet_mandatory,
        "path_format_errors":                 path_format_errors,
        # Core metrics
        "precision":                          prec,
        "recall":                             rec,
        "f1":                                 f1,
        "any_legal_found":                    len(legal_proposals) > 0,
        "fake_jump_count":                    len(fake_jumps),
        "quiet_when_capture_mandatory_count": len(quiet_mandatory),
        "path_format_error_count":            len(path_format_errors),
        "llm_capture_estimate":               llm_est,
        "llm_capture_estimate_correct":       est_ok,
        # Scan metrics (diagnostic only — Python never repairs proposals)
        "scan_present":                       scan_anal["scan_present"],
        "n_scan_entries":                     scan_anal["n_scan_entries"],
        "n_jump_checks":                      scan_anal["n_jump_checks"],
        "n_valid_scan_jumps":                 n_valid_in_scan,
        "scan_board_mismatch_count":          scan_anal["scan_board_mismatch_count"],
        "scan_mismatches":                    scan_anal["mismatches"],
        "scan_contradiction_count":           scan_contrad,
        "capture_scan_consistent":            cap_scan_consistent,
        "proposal_uses_unverified_jump":      unverif_j,
        # Simple-checks metrics (diagnostic only — Python never repairs proposals)
        "simple_checks_present":              simple_anal["simple_checks_present"],
        "n_simple_checks":                    simple_anal["n_simple_checks"],
        "simple_board_mismatch_count":        simple_anal["simple_board_mismatch_count"],
        "simple_mismatches":                  simple_anal["simple_mismatches"],
        "simple_unbacked_count":              simple_unbacked,
        # Grounding metrics (source_check_id/ids — diagnostic only)
        "unlinked_jump_count":                grounding["unlinked_jump_count"],
        "bad_source_jump_count":              grounding["bad_source_jump_count"],
        "unlinked_simple_count":              grounding["unlinked_simple_count"],
        "bad_source_simple_count":            grounding["bad_source_simple_count"],
        # Contradiction retry + safe rejection metrics
        "contradiction_retry_count":          n_contradiction_retries,
        "contradiction_reasons":              contradiction_reasons,
        "post_retry_still_contradictory":     post_retry_still_contr,
        "dropped_unverified_after_retry":     dropped_unverified_after,
        "dropped_bad_source_after_retry":     dropped_bad_source_after,
        "safe_rejection_count":               safe_rejection_count,
        # Simple-completeness diagnostics (Phase 8)
        "valid_simple_checks_count":          valid_sc_count,
        "missing_final_from_sc_count":        missing_final_from_sc,
        "final_sc_rate":                      final_sc_rate,
        # Diagnostics
        "api_call_succeeded":                 diag.get("api_call_succeeded", False),
        "agent_error":                        agent_error,
        "diagnostics":                        diag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation for --repeat N
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_repeat(runs: list[dict]) -> dict:
    """Aggregate metrics across N repeat runs of the same scenario."""
    n = len(runs)

    def _rate(pred):
        return sum(1 for r in runs if pred(r)) / n

    def _avg(field):
        return sum(r[field] for r in runs) / n

    non_none = [r for r in runs if r["llm_capture_estimate_correct"] is not None]
    est_acc = (
        sum(1 for r in non_none if r["llm_capture_estimate_correct"]) / len(non_none)
        if non_none else None
    )

    non_none_csc = [r for r in runs if r["capture_scan_consistent"] is not None]
    csc_acc = (
        sum(1 for r in non_none_csc if r["capture_scan_consistent"]) / len(non_none_csc)
        if non_none_csc else None
    )

    return {
        "n_runs":                        n,
        "avg_precision":                 _avg("precision"),
        "avg_recall":                    _avg("recall"),
        "avg_f1":                        _avg("f1"),
        "pass_rate":                     _rate(
            lambda r: r["any_legal_found"]
                      and r["fake_jump_count"] == 0
                      and r["quiet_when_capture_mandatory_count"] == 0
                      and r["path_format_error_count"] == 0
        ),
        "any_legal_rate":                _rate(lambda r: r["any_legal_found"]),
        "fake_jump_rate":                _avg("fake_jump_count"),
        "quiet_mandatory_rate":          _avg("quiet_when_capture_mandatory_count"),
        "path_format_error_rate":        _avg("path_format_error_count"),
        "capture_est_accuracy":          est_acc,
        # Scan averages
        "scan_present_rate":             _rate(lambda r: r["scan_present"]),
        "avg_scan_board_mismatches":     _avg("scan_board_mismatch_count"),
        "avg_scan_contradictions":       _avg("scan_contradiction_count"),
        "capture_scan_consistent_rate":  csc_acc,
        "avg_unverified_jumps":          _avg("proposal_uses_unverified_jump"),
        # Simple-checks averages
        "simple_checks_present_rate":    _rate(lambda r: r["simple_checks_present"]),
        "avg_simple_board_mismatches":   _avg("simple_board_mismatch_count"),
        "avg_simple_unbacked":           _avg("simple_unbacked_count"),
        # Grounding averages
        "avg_unlinked_jumps":            _avg("unlinked_jump_count"),
        "avg_bad_source_jumps":          _avg("bad_source_jump_count"),
        "avg_unlinked_simples":          _avg("unlinked_simple_count"),
        # Retry + safe rejection averages
        "avg_contradiction_retries":     _avg("contradiction_retry_count"),
        "post_retry_contradiction_rate": _rate(lambda r: r.get("post_retry_still_contradictory", False)),
        "avg_safe_rejections":           _avg("safe_rejection_count"),
        # Simple-completeness averages (Phase 8)
        "avg_valid_simple_checks_count": _avg("valid_simple_checks_count"),
        "avg_missing_final_from_sc":     _avg("missing_final_from_sc_count"),
        "avg_final_sc_rate":             _avg("final_sc_rate"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing — single run
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict, idx: int, total: int) -> None:
    W    = 72
    sep  = "═" * W
    sep2 = "─" * W

    print(f"\n{sep}")
    print(f"  [{idx}/{total}]  {result['name']}  ({result['current_player']})")
    print(f"  {result['description']}")
    print(sep2)

    if result["agent_error"]:
        print(f"  !! AGENT EXCEPTION:\n{result['agent_error']}")
        return

    if not result["api_call_succeeded"]:
        print("  !! API call failed — proposals empty")

    print(
        f"  true_legal ({result['n_true_legal']} moves, "
        f"mandatory_capture={result['has_mandatory_capture']}, "
        f"expected_has_jumps={result['expected_has_jumps']}):"
    )
    for m in result["true_legal_moves"]:
        print(f"      {_fmt_move(m)}")

    print()
    print(f"  proposals ({result['n_proposals']} moves):")
    for prop in result["proposals"]:
        tag = "✓" if _is_legal(prop, result["true_legal_moves"]) else "✗"
        print(f"    {tag} {_fmt_move(prop)}")

    print()
    prec_s = f"{result['precision']*100:.0f}%"
    rec_s  = f"{result['recall']*100:.0f}%"
    f1_s   = f"{result['f1']*100:.0f}%"
    est    = result["llm_capture_estimate"]
    est_ok = result["llm_capture_estimate_correct"]
    est_s  = (
        f"{'✓' if est_ok else '✗'}  llm={est}  truth={result['has_mandatory_capture']}"
        if est_ok is not None else "(field absent in LLM response)"
    )

    print(f"  precision:                       {prec_s}")
    print(f"  recall:                          {rec_s}")
    print(f"  f1:                              {f1_s}")
    print(f"  any_legal_found:                 {result['any_legal_found']}")
    print(f"  fake_jump_count:                 {result['fake_jump_count']}")
    print(f"  quiet_when_capture_mandatory:    {result['quiet_when_capture_mandatory_count']}")
    print(f"  path_format_error_count:         {result['path_format_error_count']}")
    print(f"  capture_estimate_correct:        {est_s}")

    # ── Scan analysis ────────────────────────────────────────────────────────
    print()
    if result["scan_present"]:
        csc = result["capture_scan_consistent"]
        csc_s = ("✓" if csc is True else ("✗" if csc is False else "-"))
        print(f"  scan_entries:                    {result['n_scan_entries']}  "
              f"({result['n_jump_checks']} checks, {result['n_valid_scan_jumps']} valid)")
        print(f"  scan_board_mismatches:           {result['scan_board_mismatch_count']}")
        print(f"  scan_contradictions:             {result['scan_contradiction_count']}"
              f"  (valid=true but claimed values wrong)")
        print(f"  capture_scan_consistent:         {csc_s}")
        print(f"  proposal_uses_unverified_jump:   {result['proposal_uses_unverified_jump']}")
        if result["scan_mismatches"] and _VERBOSE:
            print(f"\n  SCAN BOARD MISMATCHES ({result['scan_board_mismatch_count']}):")
            for mm in result["scan_mismatches"]:
                print(f"      piece={mm['piece']} ({mm['piece_type']}) dir={mm['dir']} "
                      f"field={mm['field']} at={mm['at']}  "
                      f"claimed={mm['claimed']!r} actual={mm['actual']!r}")
    else:
        print("  scan:                            ABSENT (LLM did not include scan field)")

    # ── Simple-checks analysis ────────────────────────────────────────────────
    n_simples_proposed = sum(1 for p in result["proposals"] if p.get("type") == "simple")
    if result["simple_checks_present"]:
        print(f"  simple_checks:                   {result['n_simple_checks']} entries")
        print(f"  simple_board_mismatches:         {result['simple_board_mismatch_count']}"
              f"  (to_val claimed ≠ actual board)")
        print(f"  simple_unbacked_count:           {result['simple_unbacked_count']}"
              f"  (simples without valid=true check)")
        print(f"  valid_simple_checks_count:       {result['valid_simple_checks_count']}"
              f"  (valid=true entries in simple_checks)")
        print(f"  missing_final_from_sc:           {result['missing_final_from_sc_count']}"
              f"  (valid checks absent from final_proposed_moves)")
        print(f"  final_sc_rate:                   {result['final_sc_rate']:.3f}"
              f"  (completeness: present/total valid checks)")
        if result["simple_mismatches"] and _VERBOSE:
            print(f"\n  SIMPLE BOARD MISMATCHES ({result['simple_board_mismatch_count']}):")
            for mm in result["simple_mismatches"]:
                print(f"      from={mm['from']} to={mm['to']} dir={mm['dir']}  "
                      f"claimed={mm['claimed']!r} actual={mm['actual']!r}")
    elif n_simples_proposed > 0:
        print(f"  simple_checks:                   ABSENT  "
              f"({n_simples_proposed} simple(s) proposed without check table)")

    # ── Grounding analysis (source_check_id/ids) ─────────────────────────────
    ulj = result["unlinked_jump_count"]
    bsj = result["bad_source_jump_count"]
    uls = result["unlinked_simple_count"]
    bss = result["bad_source_simple_count"]
    n_jumps_proposed = sum(1 for p in result["proposals"] if p.get("type") == "jump")
    if n_jumps_proposed > 0 or ulj > 0 or bsj > 0:
        print(f"  unlinked_jump_count:             {ulj}  (jumps with no source_check_ids)")
        print(f"  bad_source_jump_count:           {bsj}  (id invalid/missing/wrong length)")
    if uls > 0 or bss > 0:
        print(f"  unlinked_simple_count:           {uls}")
        print(f"  bad_source_simple_count:         {bss}")

    # ── Contradiction retry + safe rejection ─────────────────────────────────
    n_crt = result.get("contradiction_retry_count", 0)
    if n_crt > 0:
        print(f"  contradiction_retry_count:       {n_crt}")
        for i, reason in enumerate(result.get("contradiction_reasons", []), 1):
            print(f"    [{i}] {reason[:120]}")
        still = result.get("post_retry_still_contradictory", False)
        sc    = result.get("safe_rejection_count", 0)
        du    = result.get("dropped_unverified_after_retry", 0)
        db    = result.get("dropped_bad_source_after_retry", 0)
        print(f"  post_retry_still_contradictory:  {still}")
        if sc > 0:
            print(f"  safe_rejection_count:            {sc}"
                  f"  (unverified={du} bad_source={db})")

    if result["path_format_errors"]:
        print(f"\n  PATH FORMAT ERRORS ({result['path_format_error_count']} moves):")
        for pfe in result["path_format_errors"]:
            print(f"      {_fmt_move(pfe['move'])}")
            for err in pfe["errors"]:
                print(f"        ↳ {err}")

    if result["fake_jumps"]:
        print("\n  FAKE JUMPS (type=jump proposals with no matching legal jump):")
        for fj in result["fake_jumps"]:
            print(f"      {_fmt_move(fj)}")

    if result["quiet_when_capture_mandatory"]:
        print("\n  QUIET WHEN MANDATORY CAPTURE (illegal simples during forced capture):")
        for qm in result["quiet_when_capture_mandatory"]:
            print(f"      {_fmt_move(qm)}")

    if result["missed_legal"]:
        print(f"\n  missed legal ({len(result['missed_legal'])}):")
        for ml in result["missed_legal"]:
            print(f"      {_fmt_move(ml)}")

    if result["notes"]:
        print(f"\n  note: {result['notes']}")

    if _VERBOSE and result["raw_response"]:
        print(f"\n  RAW LLM OUTPUT (first 3 000 chars):")
        print(result["raw_response"][:3000])


# ─────────────────────────────────────────────────────────────────────────────
# Printing — repeat mode
# ─────────────────────────────────────────────────────────────────────────────

def _print_aggregated(scenario: dict, agg: dict, runs: list[dict]) -> None:
    W    = 72
    sep  = "═" * W
    sep2 = "─" * W
    n    = agg["n_runs"]
    player = "RED" if scenario["current_player"] == RED else "BLACK"

    print(f"\n{sep}")
    print(f"  {scenario['name']}  ({player})  ×{n} runs")
    print(f"  {scenario['description']}")
    print(sep2)

    # True legal moves (constant across runs)
    true_legal = runs[0]["true_legal_moves"]
    has_cap    = runs[0]["has_mandatory_capture"]
    print(f"  true_legal ({len(true_legal)} moves, mandatory_capture={has_cap}):")
    for m in true_legal:
        print(f"      {_fmt_move(m)}")

    print()
    print(f"  avg_precision:    {agg['avg_precision']*100:.0f}%")
    print(f"  avg_recall:       {agg['avg_recall']*100:.0f}%")
    print(f"  avg_f1:           {agg['avg_f1']*100:.0f}%")
    print(f"  pass_rate:        {agg['pass_rate']*100:.0f}%  "
          f"({int(round(agg['pass_rate']*n))}/{n} runs passed)")
    print(f"  any_legal_rate:   {agg['any_legal_rate']*100:.0f}%")
    print(f"  fake_jump_rate:    {agg['fake_jump_rate']:.2f} per run")
    print(f"  quiet_mandatory:   {agg['quiet_mandatory_rate']:.2f} per run")
    print(f"  path_fmt_err_rate: {agg['path_format_error_rate']:.2f} per run")
    est_acc = agg["capture_est_accuracy"]
    est_s   = f"{est_acc*100:.0f}%" if est_acc is not None else "n/a"
    print(f"  capture_est_acc:   {est_s}")
    print(f"  scan_present_rate: {agg['scan_present_rate']*100:.0f}%")
    print(f"  avg_sbm:           {agg['avg_scan_board_mismatches']:.2f} per run  "
          f"(scan board mismatches)")
    print(f"  avg_scan_contrad:  {agg['avg_scan_contradictions']:.2f} per run")
    csc = agg["capture_scan_consistent_rate"]
    print(f"  cap_scan_consist:  {f'{csc*100:.0f}%' if csc is not None else 'n/a'}")
    print(f"  avg_unverif_jump:  {agg['avg_unverified_jumps']:.2f} per run")
    print(f"  sc_present_rate:   {agg['simple_checks_present_rate']*100:.0f}%"
          f"  (simple_checks field present)")
    print(f"  avg_simple_sbm:    {agg['avg_simple_board_mismatches']:.2f} per run"
          f"  (simple to_val ≠ board)")
    print(f"  avg_simple_sub:    {agg['avg_simple_unbacked']:.2f} per run"
          f"  (simples without valid check)")
    print(f"  avg_unlinked_jump: {agg['avg_unlinked_jumps']:.2f} per run"
          f"  (jumps with no source_check_ids)")
    print(f"  avg_bad_src_jump:  {agg['avg_bad_source_jumps']:.2f} per run"
          f"  (invalid/wrong-length source id)")
    print(f"  avg_contradiction_retries: {agg['avg_contradiction_retries']:.2f} per run"
          f"  (retry triggered by scan/source contradiction)")
    print(f"  post_retry_contradiction:  {agg['post_retry_contradiction_rate']*100:.0f}%"
          f"  (retry output still contradictory)")
    print(f"  avg_safe_rejections:       {agg['avg_safe_rejections']:.2f} per run"
          f"  (proposals dropped by post-retry filter)")
    print(f"  avg_valid_sc_count:    {agg['avg_valid_simple_checks_count']:.1f} per run"
          f"  (valid=true entries in simple_checks)")
    print(f"  avg_missing_final_sc:  {agg['avg_missing_final_from_sc']:.1f} per run"
          f"  (valid checks absent from final_proposed_moves)")
    print(f"  avg_final_sc_rate:     {agg['avg_final_sc_rate']:.3f}"
          f"  (simple completeness: present/total valid checks)")

    print("\n  Per-run detail:")
    for i, r in enumerate(runs, 1):
        prec  = f"{r['precision']*100:.0f}%"
        rec   = f"{r['recall']*100:.0f}%"
        f1    = f"{r['f1']*100:.0f}%"
        fj    = r["fake_jump_count"]
        qm    = r["quiet_when_capture_mandatory_count"]
        pfe   = r["path_format_error_count"]
        alf   = "✓" if r["any_legal_found"] else "✗"
        ec    = r["llm_capture_estimate_correct"]
        est   = "✓" if ec is True else ("✗" if ec is False else "-")
        sbm   = r["scan_board_mismatch_count"]
        puj   = r["proposal_uses_unverified_jump"]
        sc    = "✓" if r["scan_present"] else "✗"
        sub   = r["simple_unbacked_count"]
        vsc   = r["valid_simple_checks_count"]
        mfsc  = r["missing_final_from_sc_count"]
        scr   = f"{r['final_sc_rate']:.2f}"
        print(f"    run {i:2d}: prec={prec} rec={rec} f1={f1} "
              f"fake={fj} quiet={qm} pfe={pfe} legal={alf} est={est} "
              f"scan={sc} sbm={sbm} puj={puj} sub={sub} "
              f"vsc={vsc} mfsc={mfsc} scr={scr}")

    if scenario["notes"]:
        print(f"\n  note: {scenario['notes']}")

    if _VERBOSE:
        for i, r in enumerate(runs, 1):
            if r["raw_response"]:
                print(f"\n  ── run {i} RAW LLM OUTPUT (first 2 000 chars) ──")
                print(r["raw_response"][:2000])


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary (end of run)
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results_or_aggs: list, repeat_n: int) -> None:
    """
    results_or_aggs:
      repeat_n == 1 → list of single-run result dicts
      repeat_n >  1 → list of (scenario_name, agg_dict) tuples
    """
    W   = 90
    sep = "═" * W
    print(f"\n{sep}")
    print("  AGGREGATE SUMMARY")
    print(sep)
    print(f"  {'STATUS':6}  {'PREC':5}  {'REC':5}  {'F1':5}  "
          f"{'FAKE':4}  {'QUIET':5}  {'PFE':4}  {'EST':3}  {'SBM':4}  {'PUJ':4}  {'SUB':4}  "
          f"{'ULJ':4}  {'BSJ':4}  NAME")
    print("  " + "─" * (W - 2))

    passed = 0
    rows   = []

    if repeat_n == 1:
        for r in results_or_aggs:
            status = (
                "PASS" if r["any_legal_found"]
                          and r["fake_jump_count"] == 0
                          and r["quiet_when_capture_mandatory_count"] == 0
                          and r["path_format_error_count"] == 0
                else "FAIL"
            )
            if status == "PASS":
                passed += 1
            prec  = f"{r['precision']*100:3.0f}%"
            rec   = f"{r['recall']*100:3.0f}%"
            f1_s  = f"{r['f1']*100:3.0f}%"
            fj    = r["fake_jump_count"]
            qm    = r["quiet_when_capture_mandatory_count"]
            pfe   = r["path_format_error_count"]
            ec    = r["llm_capture_estimate_correct"]
            est   = "✓" if ec is True else ("✗" if ec is False else "-")
            sbm   = r["scan_board_mismatch_count"]
            puj   = r["proposal_uses_unverified_jump"]
            sub   = r["simple_unbacked_count"]
            ulj   = r["unlinked_jump_count"]
            bsj   = r["bad_source_jump_count"]
            rows.append((status, prec, rec, f1_s, fj, qm, pfe, est, sbm, puj, sub, ulj, bsj, r["name"]))
    else:
        for (name, agg) in results_or_aggs:
            status = f"{int(round(agg['pass_rate']*agg['n_runs']))}/{agg['n_runs']}"
            passed += agg["pass_rate"] == 1.0
            prec  = f"{agg['avg_precision']*100:3.0f}%"
            rec   = f"{agg['avg_recall']*100:3.0f}%"
            f1_s  = f"{agg['avg_f1']*100:3.0f}%"
            fj    = f"{agg['fake_jump_rate']:.1f}"
            qm    = f"{agg['quiet_mandatory_rate']:.1f}"
            pfe   = f"{agg['path_format_error_rate']:.1f}"
            ea    = agg["capture_est_accuracy"]
            est   = f"{ea*100:.0f}%" if ea is not None else "-"
            sbm   = f"{agg['avg_scan_board_mismatches']:.1f}"
            puj   = f"{agg['avg_unverified_jumps']:.1f}"
            sub   = f"{agg['avg_simple_unbacked']:.1f}"
            ulj   = f"{agg['avg_unlinked_jumps']:.1f}"
            bsj   = f"{agg['avg_bad_source_jumps']:.1f}"
            rows.append((status, prec, rec, f1_s, fj, qm, pfe, est, sbm, puj, sub, ulj, bsj, name))

    total = len(rows)
    for status, prec, rec, f1_s, fj, qm, pfe, est, sbm, puj, sub, ulj, bsj, name in rows:
        print(f"  {status:6}  {prec:5}  {rec:5}  {f1_s:5}  "
              f"{str(fj):4}  {str(qm):5}  {str(pfe):4}  {est:3}  "
              f"{str(sbm):4}  {str(puj):4}  {str(sub):4}  {str(ulj):4}  {str(bsj):4}  {name}")

    print("  " + "─" * (W - 2))
    if repeat_n == 1:
        print(f"  PASS: {passed}/{total}")
    else:
        print(f"  Perfect pass_rate: {passed}/{total} scenarios")
    print(sep)

    print("""
  INTERPRETATION
  ─────────────────────────────────────────────────────────────────────────
  STATUS  PASS  = any_legal_found AND fake_jump_count=0 AND quiet_mandatory=0
                  AND path_format_error_count=0
          FAIL  = at least one condition violated
          N/M   = N out of M repeat runs passed (in --repeat mode)

  PREC    precision = legal_proposed / total_proposed
          Low → LLM invents moves. Validator overhead but not fatal.

  REC     recall = legal_proposed / true_legal_count
          Low → LLM misses moves. Acceptable: validator falls back to engine
          when no proposals are legal.

  F1      harmonic mean of precision and recall.

  FAKE    fake_jump_count — type='jump' proposals with no matching legal jump.
          Goal: 0. LLM invented a capture that doesn't exist.

  QUIET   quiet_when_capture_mandatory — type='simple' during forced capture.
          Goal: 0. LLM failed mandatory-capture detection.

  PFE     path_format_error_count — jump moves with structural path errors:
            length_mismatch  : len(path) != len(captured)+1
            step_size_error  : consecutive path entries 1 step apart (midpoint in path)
            midpoint_in_path : a captured square also appears in path
          Goal: 0.

  EST     capture_estimate_correct — LLM's capture_available_estimate vs truth.
          ✓ = correct  ✗ = wrong  - = field absent

  SBM     scan_board_mismatch_count — scan entries where mid_val or land_val
          don't match the actual board. Goal: 0. Shows board-reading errors.
          High SBM with low FAKE = LLM guesses wrong squares but self-corrects.
          High SBM with high FAKE = LLM hallucinates captures from wrong reads.

  PUJ     proposal_uses_unverified_jump — jump proposals with no backing
          valid=true scan entry. Goal: 0. Shows LLM bypassed the scan gate.

  SUB     simple_unbacked_count — type='simple' proposals with no backing
          valid=true simple_checks entry (from=path[0], to=path[1]).
          Goal: 0. Shows LLM proposed a simple without recording its check.
          Non-zero = simple hallucination or SIMPLE-GATE bypassed.

  ULJ     unlinked_jump_count — type='jump' proposals with no source_check_ids field.
          Goal: 0. Shows LLM bypassed SOURCE-CHECK LINK grounding for jump moves.
          High ULJ + high FAKE = ungrounded jump hallucinations.

  BSJ     bad_source_jump_count — type='jump' proposals with source_check_ids present but
          invalid: any cited id has valid=false, id not found, or len≠len(captured).
          Goal: 0. Shows LLM cited checks that don't support the proposed jump.
          Non-zero includes partial-path jumps (too few ids) and mismatched ids.
  ─────────────────────────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _VERBOSE

    parser = argparse.ArgumentParser(
        description="board_proposal_agent quality probe (25 scenarios)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "scenario", nargs="?", default=None,
        help="Run only this scenario by name. Omit to run all scenarios.",
    )
    parser.add_argument(
        "--repeat", "-r", type=int, default=1, metavar="N",
        help="Number of times to repeat each scenario (default 1).\n"
             "Use ≥3 to measure fake_jump_rate stability.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print raw LLM output for each run.",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, metavar="S",
        help="Seconds to wait between consecutive API calls (default 2.0).",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all scenario names and exit (no API calls).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show engine legal-move counts for every scenario and exit (no API calls).",
    )
    args = parser.parse_args()
    _VERBOSE = args.verbose

    all_scenarios = _build_scenarios()

    if args.list:
        for sc in all_scenarios:
            player = "RED" if sc["current_player"] == RED else "BLACK"
            print(f"  {sc['name']:45s}  player={player}")
        sys.exit(0)

    if args.dry_run:
        print(f"\n{'':2}{'✓/✗':4}  {'NAME':45s}  {'PLAYER':5}  {'N':3}  HAS_JUMPS")
        print("  " + "─" * 70)
        all_ok = True
        for sc in all_scenarios:
            legal    = get_all_legal_moves(sc["board"], sc["current_player"])
            has_j    = any(m["type"] == "jump" for m in legal)
            player   = "RED" if sc["current_player"] == RED else "BLACK"
            expected = sc["expected_has_jumps"]
            ok       = has_j == expected
            all_ok   = all_ok and ok
            mark     = "✓" if ok else "✗"
            print(f"  {mark:4}  {sc['name']:45s}  {player:5}  {len(legal):3d}  {str(has_j):5s}")
        print("  " + "─" * 70)
        print(f"  {'All OK' if all_ok else 'MISMATCH DETECTED'}")
        sys.exit(0 if all_ok else 1)

    if args.scenario:
        scenarios = [sc for sc in all_scenarios if sc["name"] == args.scenario]
        if not scenarios:
            print(f"ERROR: scenario '{args.scenario}' not found.")
            print("Available names (use --list for full table):")
            for sc in all_scenarios:
                print(f"  {sc['name']}")
            sys.exit(1)
    else:
        scenarios = all_scenarios

    repeat_n  = max(1, args.repeat)
    delay_s   = max(0.0, args.delay)
    total_sc  = len(scenarios)
    total_api = total_sc * repeat_n

    print(f"\nboard_proposal_agent quality probe — "
          f"{total_sc} scenario(s) × {repeat_n} run(s) = {total_api} API call(s)")
    if _VERBOSE:
        print("  (--verbose: raw LLM output will be shown)")

    summary_entries = []   # (name, agg) for repeat mode, or result dicts for single mode
    call_index = 0

    for sc_idx, scenario in enumerate(scenarios):
        runs: list[dict] = []

        for _ in range(repeat_n):
            # No delay on the very first call; delay_s between all subsequent calls
            delay = 0.0 if call_index == 0 else delay_s
            call_index += 1

            result = evaluate_scenario(scenario, delay_s=delay)
            runs.append(result)

        if repeat_n > 1:
            agg = aggregate_repeat(runs)
            _print_aggregated(scenario, agg, runs)
            summary_entries.append((scenario["name"], agg))
        else:
            _print_result(runs[0], sc_idx + 1, total_sc)
            summary_entries.append(runs[0])

    if total_sc > 1:
        _print_summary(summary_entries, repeat_n)


if __name__ == "__main__":
    main()
