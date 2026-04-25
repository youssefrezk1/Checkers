from __future__ import annotations

from checkers.engine.board import BLACK, EMPTY, RED
from checkers.engine.evaluation import evaluate_board_breakdown


def _empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def test_isolation_penalty_worse_for_isolated_structure() -> None:
    isolated = _empty_board()
    isolated[5][2] = RED
    isolated[0][1] = BLACK
    isolated[1][2] = BLACK

    supported = _empty_board()
    supported[5][2] = RED
    supported[6][1] = RED
    supported[0][1] = BLACK
    supported[1][2] = BLACK

    iso_breakdown = evaluate_board_breakdown(isolated, RED, RED)
    sup_breakdown = evaluate_board_breakdown(supported, RED, RED)
    assert sup_breakdown["isolation"] > iso_breakdown["isolation"]


def test_connectivity_support_prefers_supported_men() -> None:
    weak = _empty_board()
    weak[5][2] = RED
    weak[2][5] = BLACK

    strong = _empty_board()
    strong[5][2] = RED
    strong[6][1] = RED
    strong[2][5] = BLACK

    weak_breakdown = evaluate_board_breakdown(weak, RED, RED)
    strong_breakdown = evaluate_board_breakdown(strong, RED, RED)
    assert strong_breakdown["connectivity_support"] > weak_breakdown["connectivity_support"]


def test_frozen_restriction_rewards_opponent_restriction() -> None:
    free_opp = _empty_board()
    free_opp[5][2] = RED
    free_opp[2][3] = BLACK

    restricted_opp = _empty_board()
    restricted_opp[5][2] = RED
    restricted_opp[2][3] = BLACK

    # Block both forward diagonals for BLACK structurally.
    restricted_opp[3][2] = RED
    restricted_opp[3][4] = RED

    free_breakdown = evaluate_board_breakdown(free_opp, RED, RED)
    restricted_breakdown = evaluate_board_breakdown(restricted_opp, RED, RED)

    assert restricted_breakdown["frozen_restriction"] > free_breakdown["frozen_restriction"]


def test_breakdown_exposes_phase6_terms() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[2][3] = BLACK

    breakdown = evaluate_board_breakdown(board, RED, RED)
    assert "isolation" in breakdown
    assert "connectivity_support" in breakdown
    assert "frozen_restriction" in breakdown
