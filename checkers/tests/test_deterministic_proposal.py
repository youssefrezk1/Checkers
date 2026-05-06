"""
checkers/tests/test_deterministic_proposal.py

Tests for select_proposal_candidates in deterministic_proposal.py.

Run:
    pytest checkers/tests/test_deterministic_proposal.py -v
"""
from __future__ import annotations

import pytest

from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.deterministic_proposal import select_proposal_candidates


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _start_board() -> list[list[int]]:
    """Standard opening board (12 RED + 12 BLACK pieces)."""
    b = [[0] * 8 for _ in range(8)]
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = BLACK
    return b


def _board_with_one_legal_move() -> list[list[int]]:
    """A board where RED has exactly one legal move (and BLACK can't capture it)."""
    b = [[0] * 8 for _ in range(8)]
    # Single RED piece at (5, 0) — can only go to (4, 1)
    b[5][0] = RED
    # BLACK piece far away at (0, 7)
    b[0][7] = BLACK
    return b


def _board_with_capture() -> list[list[int]]:
    """
    Board with a forced capture sequence for RED.

    RED at (4, 3) can jump BLACK at (3, 4) -> landing at (2, 5).
    RED also has a quiet move at (5, 2).
    """
    b = [[0] * 8 for _ in range(8)]
    b[4][3] = RED
    b[5][2] = RED
    b[3][4] = BLACK
    b[0][7] = BLACK   # lone BLACK so game is not over
    return b


def _scored(board, player=RED):
    """Helper: return enriched scored moves from scorer_agent."""
    enriched, _, _, _ = score_all_legal_moves(board, player)
    return enriched


# ── Count correctness ──────────────────────────────────────────────────────────

def test_returns_five_from_start():
    board = _start_board()
    scored = _scored(board)
    assert len(scored) >= 5, "opening position needs >= 5 legal moves for this test"
    result = select_proposal_candidates(scored)
    assert len(result) == 5


def test_returns_all_when_fewer_than_five():
    board = _board_with_one_legal_move()
    scored = _scored(board)
    assert len(scored) == 1
    result = select_proposal_candidates(scored)
    assert len(result) == 1


def test_returns_min_k_n():
    board = _start_board()
    scored = _scored(board)
    n = len(scored)
    for k in range(1, n + 2):
        result = select_proposal_candidates(scored, k=k)
        assert len(result) == min(k, n), f"expected {min(k, n)} for k={k}, n={n}"


def test_empty_scored_moves_returns_empty():
    result = select_proposal_candidates([])
    assert result == []


# ── Source integrity — all returned moves come from scored_moves ──────────────

def _path_key(m: dict) -> tuple:
    return tuple(tuple(sq) for sq in m["path"])


def test_all_results_from_scored_moves():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    scored_keys = {_path_key(m) for m in scored}
    for r in result:
        assert _path_key(r) in scored_keys, "result contains a move not in scored_moves"


def test_no_duplicates_in_result():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    keys = [_path_key(m) for m in result]
    assert len(keys) == len(set(keys)), "result contains duplicate moves"


def test_result_elements_are_same_objects():
    """Returned dicts must be the exact objects from scored_moves (no copies)."""
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    scored_by_path = {_path_key(m): m for m in scored}
    for r in result:
        assert r is scored_by_path[_path_key(r)], "result element is a copy, not the original"


# ── Minimax top-3 inclusion ───────────────────────────────────────────────────

def test_top3_minimax_included_when_enough_moves():
    """When n >= 5, the three highest-minimax-score moves must appear in result."""
    board = _start_board()
    scored = _scored(board)
    assert len(scored) >= 5, "need >= 5 moves to test mm-pin"

    top3_keys = {_path_key(scored[i]) for i in range(3)}
    result = select_proposal_candidates(scored)
    result_keys = {_path_key(r) for r in result}
    assert top3_keys <= result_keys, (
        f"top-3 minimax moves not all included.\n"
        f"top3_keys={top3_keys}\nresult_keys={result_keys}"
    )


def test_top3_are_highest_minimax_in_result():
    """Verify top-3 by minimax_score are in the result, regardless of symbolic sort order."""
    board = _start_board()
    scored = _scored(board)
    assert len(scored) >= 5

    sorted_by_mm = sorted(scored, key=lambda m: m["facts"]["minimax_score"], reverse=True)
    top3_keys = {_path_key(m) for m in sorted_by_mm[:3]}

    result = select_proposal_candidates(scored)
    result_keys = {_path_key(r) for r in result}
    assert top3_keys <= result_keys


# ── Capture inclusion ─────────────────────────────────────────────────────────

def test_capture_included_when_available():
    """A capture move must appear in the shortlist when legal captures exist."""
    board = _board_with_capture()
    scored = _scored(board)
    jump_in_scored = any(m["type"] == "jump" for m in scored)
    if not jump_in_scored:
        pytest.skip("no jump moves in this position")

    result = select_proposal_candidates(scored)
    assert any(m["type"] == "jump" for m in result), "capture move missing from shortlist"


# ── Determinism ───────────────────────────────────────────────────────────────

def test_deterministic_output():
    """Same inputs must always produce the same output."""
    board = _start_board()
    scored = _scored(board)
    result_a = select_proposal_candidates(scored)
    result_b = select_proposal_candidates(scored)
    assert [_path_key(m) for m in result_a] == [_path_key(m) for m in result_b]


def test_deterministic_with_strategic_context():
    board = _start_board()
    scored = _scored(board)
    ctx = {
        "score_state": "SLIGHTLY_WINNING",
        "game_phase": "MIDGAME",
        "strategic_priorities": ["CONVERT_ADVANTAGE"],
    }
    result_a = select_proposal_candidates(scored, strategic_context=ctx)
    result_b = select_proposal_candidates(scored, strategic_context=ctx)
    assert [_path_key(m) for m in result_a] == [_path_key(m) for m in result_b]


# ── Fewer-than-5 edge cases ───────────────────────────────────────────────────

def test_single_move_board():
    board = _board_with_one_legal_move()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    assert len(result) == 1
    assert _path_key(result[0]) in {_path_key(m) for m in scored}


def test_k_zero_returns_empty():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored, k=0)
    assert result == []


def test_k_one_returns_one_move():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored, k=1)
    assert len(result) == 1
    assert _path_key(result[0]) in {_path_key(m) for m in scored}


def test_facts_preserved_in_result():
    """Returned dicts retain minimax_score and symbolic_rank from scorer_agent."""
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    for r in result:
        assert "facts" in r
        assert "minimax_score" in r["facts"]
        assert "symbolic_rank" in r["facts"]
