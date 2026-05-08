"""
checkers/tests/test_ranker_attribution_logging.py

Tests for DECISION_DEBUG field correctness and ranker_diagnostics attribution.

Three scenarios pinned here — do not change without updating the corresponding
DECISION_DEBUG field names in ranker_agent.py:

  1. raw_llm   — LLM's first choice passes the override audit → final = raw choice.
  2. retry_llm — First choice fails audit; retry LLM corrects it → final = retry choice.
  3. python_fallback — First choice fails audit; retry LLM also fails; Python fallback
                       picks best → final ≠ raw ≠ retry.

Turn 11-style diagnostic comparison
------------------------------------
Before this fix (mixed pre/post-retry snapshot):

  [DECISION_DEBUG] chosen=[[5,0],[4,1]] best=[[5,0],[4,1]]
    llm_idx=1                           ← filtered-space index of raw LLM
    chosen_path_matches_llm_idx=True    ← always True (tautology, useless)
    llm_choice_path=[[5,2],[4,3]]       ← raw choice (inconsistently named)
    ...

After this fix (clean separation):

  [DECISION_DEBUG]
    raw_llm_idx=1                       ← filtered-space index of raw LLM
    raw_llm_idx_legal=1                 ← same move in legal[] space
    raw_llm_choice_path=[[5,2],[4,3]]   ← what the first LLM call chose
    retry_llm_idx=0                     ← legal[] index retry chose (None if no retry)
    retry_llm_choice_path=[[5,0],[4,1]] ← what the retry LLM chose (None if no retry)
    final_chosen_idx=0                  ← legal[] index of the move played
    final_chosen_path=[[5,0],[4,1]]     ← path of the move played
    final_choice_source=retry_llm       ← raw_llm | retry_llm | python_fallback |
                                           single_candidate | tiebreak
    final_matches_raw_llm=False         ← did final == raw LLM choice?
    final_matches_retry_llm=True        ← did final == retry LLM choice?

Run:
    pytest checkers/tests/test_ranker_attribution_logging.py -v
"""
from __future__ import annotations

import json
from typing import Any

import pytest

import checkers.agents.ranker_agent as ranker_module
from checkers.engine.board import EMPTY, RED, BLACK
from checkers.state.state import CheckersState


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _empty_board() -> list[list[int]]:
    return [[EMPTY] * 8 for _ in range(8)]


def _make_move(
    path: list[list[int]],
    minimax_score: float,
    *,
    opponent_can_recapture: bool = False,
    moved_piece_is_threatened: bool = False,
    our_pieces_threatened_after: int = 0,
    forced_opponent_jump_reply: bool = False,
    max_opponent_jump_captures: int = 0,
    opponent_jump_count: int = 0,
) -> dict[str, Any]:
    return {
        "type": "simple",
        "path": path,
        "captured": [],
        "facts": {
            "minimax_score": minimax_score,
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": opponent_can_recapture,
            "moved_piece_is_threatened": moved_piece_is_threatened,
            "our_pieces_threatened_after": our_pieces_threatened_after,
            "forced_opponent_jump_reply": forced_opponent_jump_reply,
            "max_opponent_jump_captures": max_opponent_jump_captures,
            "opponent_jump_count": opponent_jump_count,
            "net_gain": 0,
            "leaves_piece_isolated": False,
            "weakens_king_row": False,
            "center_control": False,
            "quiet_move_role": "QUIET_DEFAULT",
            "counterplay_score": 0,
            "king_activity_score": 0,
            "simplification_value": 0,
            "results_in_king": False,
            "near_promotion": False,
            "mobility_reduction": 0,
            "winning_conversion_score": 0,
        },
    }


def _make_state(legal_moves: list[dict], turn_number: int = 15) -> CheckersState:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[0][1] = BLACK
    return CheckersState(
        board=board,
        current_player=RED,
        turn_number=turn_number,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "MIDGAME",
            "score_state": "EQUAL",
            "strategic_priorities": [],
        },
    )


def _patch_sequence(monkeypatch, responses: list[int]) -> None:
    """Fake call_ranker returning chosen_index from responses in order; last entry repeats."""
    _call = [0]

    def _fake(system: str, user: str) -> str:
        i = min(_call[0], len(responses) - 1)
        _call[0] += 1
        return json.dumps({"chosen_index": responses[i], "reasoning": f"call {i}"})

    monkeypatch.setattr(ranker_module, "call_ranker", _fake)


# ── Scenario 1: raw LLM accepted ──────────────────────────────────────────────

def test_attribution_raw_llm_accepted(monkeypatch, capsys):
    """
    LLM picks the best move on its first call. Audit passes. No retry.

    Expected attribution:
      final_choice_source  = "raw_llm"
      final_matches_raw_llm  = True
      final_matches_retry_llm = False   (no retry happened)
      retry_llm_idx          = None
      retry_llm_choice_path  = None
    """
    # Two safe, low-danger moves. LLM picks the better one (index 0).
    # low_danger_minimax_dominance requires gap >= 3.0; gap = 40 here — but the
    # LLM picks the best move so audit doesn't trigger.
    m_best  = _make_move([[5, 0], [4, 1]], minimax_score=50.0)
    m_worse = _make_move([[5, 2], [4, 3]], minimax_score=10.0)
    legal   = [m_best, m_worse]

    _patch_sequence(monkeypatch, [0])   # LLM picks best (filtered idx 0)
    patch = ranker_module.ranker_agent(_make_state(legal))

    diag = patch["ranker_diagnostics"]
    assert diag["final_choice_source"]  == "raw_llm"
    assert diag["final_matches_raw_llm"]  is True
    assert diag["final_matches_retry_llm"] is False
    assert diag["retry_llm_idx"]   is None
    assert diag["retry_llm_choice_path"] is None
    assert diag["override_retry_attempts"] == 0
    assert diag["override_fallback_applied"] is False
    assert patch["chosen_move"]["path"] == [[5, 0], [4, 1]]

    out = capsys.readouterr().out
    assert "final_choice_source=raw_llm" in out
    assert "final_matches_raw_llm=True" in out
    assert "final_matches_retry_llm=False" in out
    assert "retry_llm_idx=None" in out


# ── Scenario 2: retry LLM corrects bad raw choice ─────────────────────────────

def test_attribution_retry_llm_corrects_raw(monkeypatch, capsys):
    """
    LLM picks the worse move (audit triggers: low_danger_minimax_dominance, gap=40).
    Retry LLM picks the better move (audit passes).

    Expected attribution:
      final_choice_source    = "retry_llm"
      final_matches_raw_llm  = False
      final_matches_retry_llm = True
      retry_llm_idx          = 0   (legal[] index of best)
      retry_llm_choice_path  = [[5, 0], [4, 1]]
    """
    m_best  = _make_move([[5, 0], [4, 1]], minimax_score=50.0)
    m_worse = _make_move([[5, 2], [4, 3]], minimax_score=10.0)
    legal   = [m_best, m_worse]

    # Call 1 (raw): LLM picks worse (filtered idx 1 → legal idx 1)
    # Call 2 (retry): LLM picks best (legal idx 0)
    _patch_sequence(monkeypatch, [1, 0])
    patch = ranker_module.ranker_agent(_make_state(legal))

    diag = patch["ranker_diagnostics"]
    assert diag["final_choice_source"]     == "retry_llm"
    assert diag["final_matches_raw_llm"]   is False
    assert diag["final_matches_retry_llm"] is True
    assert diag["retry_llm_idx"]           == 0
    assert diag["retry_llm_choice_path"]   == [[5, 0], [4, 1]]
    assert diag["raw_llm_idx"]             == 1
    assert diag["raw_llm_choice_path"]     == [[5, 2], [4, 3]]
    assert diag["override_retry_attempts"] == 1
    assert diag["override_retry_resolved"] is True
    assert diag["override_fallback_applied"] is False
    assert patch["chosen_move"]["path"] == [[5, 0], [4, 1]]

    out = capsys.readouterr().out
    assert "final_choice_source=retry_llm" in out
    assert "final_matches_raw_llm=False" in out
    assert "final_matches_retry_llm=True" in out


# ── Scenario 3: Python fallback overrides both raw and retry LLM ──────────────

def test_attribution_python_fallback(monkeypatch, capsys):
    """
    LLM picks worse on first call (audit triggers). Retry LLM also picks worse
    (audit still triggers). OVERRIDE_MAX_RETRIES capped at 1 so fallback fires.
    Python fallback picks best.

    Expected attribution:
      final_choice_source    = "python_fallback"
      final_matches_raw_llm  = False
      final_matches_retry_llm = False  (retry also chose the wrong move)
      retry_llm_idx           = 1      (retry LLM also chose worse)
      final_chosen_path       = [[5, 0], [4, 1]]   (best, chosen by Python)
    """
    m_best  = _make_move([[5, 0], [4, 1]], minimax_score=50.0)
    m_worse = _make_move([[5, 2], [4, 3]], minimax_score=10.0)
    legal   = [m_best, m_worse]

    # Both raw and retry calls return the worse move (index 1).
    _patch_sequence(monkeypatch, [1, 1])
    monkeypatch.setattr(ranker_module, "OVERRIDE_MAX_RETRIES", 1)
    patch = ranker_module.ranker_agent(_make_state(legal))

    diag = patch["ranker_diagnostics"]
    assert diag["final_choice_source"]     == "python_fallback"
    assert diag["final_matches_raw_llm"]   is False
    assert diag["final_matches_retry_llm"] is False
    assert diag["retry_llm_idx"]           == 1          # retry also picked worse
    assert diag["retry_llm_choice_path"]   == [[5, 2], [4, 3]]
    assert diag["raw_llm_idx"]             == 1
    assert diag["raw_llm_choice_path"]     == [[5, 2], [4, 3]]
    assert diag["override_fallback_applied"] is True
    assert diag["override_retry_resolved"]   is False
    assert diag["final_chosen_path"]        == [[5, 0], [4, 1]]
    assert patch["chosen_move"]["path"]     == [[5, 0], [4, 1]]

    out = capsys.readouterr().out
    assert "final_choice_source=python_fallback" in out
    assert "final_matches_raw_llm=False" in out
    assert "final_matches_retry_llm=False" in out
