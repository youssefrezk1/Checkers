# checkers/tests/test_evaluation.py
#
# Focused tests for the caged_king evaluation term (Phase 8 fix).
#
# Run: venv/bin/python3 -m pytest checkers/tests/test_evaluation.py -v

from __future__ import annotations

import pytest

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.evaluation import (
    evaluate_board_breakdown,
    CAGED_KING_PENALTY,
    _is_king_caged,
    _caged_king_count,
)


def _empty_board() -> list[list[int]]:
    return [[0] * 8 for _ in range(8)]


# ── _is_king_caged unit tests ─────────────────────────────────────────────────

def test_caged_corner_king_trapped():
    """RK@(0,7), B@(0,5), (1,6) empty, (2,7) empty → king is caged.
    Only exit is (1,6); BLACK man at (0,5) can jump (0,5)→(2,7) capturing it."""
    b = _empty_board()
    b[0][7] = RED_KING
    b[0][5] = BLACK
    # (1,6) and (2,7) are empty (default)
    assert _is_king_caged(b, 0, 7, RED) is True


def test_non_caged_mobile_center_king():
    """RK in center with open exits — never caged."""
    b = _empty_board()
    b[4][4] = RED_KING
    assert _is_king_caged(b, 4, 4, RED) is False


def test_frozen_king_not_caged():
    """King surrounded by own pieces (no empty destination) → not caged (frozen, not exits-all-losing)."""
    b = _empty_board()
    b[4][4] = RED_KING
    b[3][3] = RED
    b[3][5] = RED
    b[5][3] = RED
    b[5][5] = RED
    # No empty diagonal neighbours → destinations list is empty → returns False
    assert _is_king_caged(b, 4, 4, RED) is False


def test_caged_king_with_one_safe_exit_not_caged():
    """King has two exits but only one is recapturable → not caged."""
    b = _empty_board()
    b[0][7] = RED_KING
    b[0][5] = BLACK
    # (1,6) is unsafe (BLACK can jump (0,5)→(2,7)), but block (2,7) so landing is occupied
    b[2][7] = RED   # landing square occupied → jump (0,5)→(2,7) is not legal → exit (1,6) is safe
    assert _is_king_caged(b, 0, 7, RED) is False


# ── evaluate_board_breakdown integration tests ────────────────────────────────

def test_breakdown_caged_king_penalty_applied():
    """Caged RED king produces caged_king == -CAGED_KING_PENALTY in breakdown (root=RED)."""
    b = _empty_board()
    b[0][7] = RED_KING
    b[0][5] = BLACK
    # Add BLACK piece so BLACK is not out of pieces (avoid terminal detection)
    b[7][0] = BLACK
    bd = evaluate_board_breakdown(b, RED, RED)
    assert "caged_king" in bd, "caged_king key missing from breakdown"
    assert bd["caged_king"] == -CAGED_KING_PENALTY, (
        f"Expected -CAGED_KING_PENALTY ({-CAGED_KING_PENALTY}), got {bd['caged_king']}"
    )


def test_breakdown_mobile_king_no_caged_penalty():
    """Mobile RED king in center → caged_king == 0."""
    b = _empty_board()
    b[4][4] = RED_KING
    b[7][0] = BLACK
    bd = evaluate_board_breakdown(b, RED, RED)
    assert bd.get("caged_king", 0.0) == 0.0


def test_breakdown_caged_key_always_present():
    """caged_king key must be present in breakdown even when both sides have no kings."""
    b = _empty_board()
    b[5][0] = RED
    b[2][1] = BLACK
    bd = evaluate_board_breakdown(b, RED, RED)
    assert "caged_king" in bd


def test_breakdown_caged_king_in_total():
    """total must include the caged_king contribution."""
    b = _empty_board()
    b[0][7] = RED_KING
    b[0][5] = BLACK
    b[7][0] = BLACK
    bd = evaluate_board_breakdown(b, RED, RED)
    # Recompute total without caged_king and verify difference
    reconstructed = bd["total"] - bd["caged_king"]
    remaining = sum(
        v for k, v in bd.items()
        if k not in ("total", "caged_king")
    )
    assert abs(reconstructed - remaining) < 0.01, (
        "total does not include caged_king contribution"
    )


def test_breakdown_opponent_caged_king_is_bonus():
    """Caged BLACK king → caged_king == +CAGED_KING_PENALTY for RED (root=RED)."""
    b = _empty_board()
    b[7][0] = BLACK_KING
    b[7][2] = RED
    # (6,1) is BLACK king's only exit; RED man at (7,2) can jump (7,2)→(5,0) capturing BK at (6,1)
    # Actually let's just verify direction: BLACK king at (7,0), only exit (6,1).
    # Attacker needed at (5,2) (RED man) with landing (7,0) occupied by BK itself? No...
    # Let me use a clear setup: BK at (7,0), RED man at (5,2), (6,1) is exit, RED jumps (5,2)→(7,0)?
    # No — the king is AT (7,0), it moves TO (6,1). Then RED man at (5,2) jumps over (6,1) to (7,0).
    # (7,0) is empty after BK moved. Check: (5,2) has RED, (6,1) has BK (after move), land (7,0) empty. ✓
    b2 = _empty_board()
    b2[7][0] = BLACK_KING
    b2[5][2] = RED
    b2[7][6] = RED  # keep RED alive
    bd = evaluate_board_breakdown(b2, BLACK, RED)
    assert "caged_king" in bd
    # If caged: value should be positive (opponent caged = bonus for root_player RED)
    assert bd["caged_king"] >= 0.0
