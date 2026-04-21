from __future__ import annotations

from checkers.engine.board import BLACK, EMPTY, RED
from checkers.engine.evaluation import evaluate_board, evaluate_board_breakdown


def _empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def test_back_row_guard_prefers_preserved_home_row_man() -> None:
    guarded = _empty_board()
    guarded[7][0] = RED
    guarded[5][2] = RED
    guarded[0][7] = BLACK
    guarded[2][5] = BLACK

    exposed = _empty_board()
    exposed[6][1] = RED  # same man stepped off the back row
    exposed[5][2] = RED
    exposed[0][7] = BLACK
    exposed[2][5] = BLACK

    guarded_breakdown = evaluate_board_breakdown(guarded, RED, RED)
    exposed_breakdown = evaluate_board_breakdown(exposed, RED, RED)
    assert guarded_breakdown["back_row_guard"] > exposed_breakdown["back_row_guard"]


def test_promotion_proximity_prefers_closer_man_to_crowning() -> None:
    far_board = _empty_board()
    far_board[5][2] = RED
    far_board[2][5] = BLACK

    near_board = _empty_board()
    near_board[2][3] = RED
    near_board[2][5] = BLACK

    far_score = evaluate_board(far_board, RED, RED)
    near_score = evaluate_board(near_board, RED, RED)
    assert near_score > far_score


def test_simplification_bonus_when_ahead_prefers_fewer_pieces() -> None:
    complex_board = _empty_board()
    complex_board[5][0] = RED
    complex_board[5][2] = RED
    complex_board[7][6] = RED
    complex_board[2][1] = BLACK
    complex_board[0][7] = BLACK

    simple_board = _empty_board()
    simple_board[5][0] = RED
    simple_board[5][2] = RED
    simple_board[2][1] = BLACK

    complex_breakdown = evaluate_board_breakdown(complex_board, RED, RED)
    simple_breakdown = evaluate_board_breakdown(simple_board, RED, RED)
    assert simple_breakdown["simplification_when_ahead"] > complex_breakdown["simplification_when_ahead"]


def test_evaluation_breakdown_exposes_new_phase5_terms() -> None:
    board = _empty_board()
    board[7][0] = RED
    board[5][2] = RED
    board[2][1] = BLACK

    breakdown = evaluate_board_breakdown(board, RED, RED)
    assert "back_row_guard" in breakdown
    assert "promotion_proximity" in breakdown
    assert "simplification_when_ahead" in breakdown
    assert "total" in breakdown
