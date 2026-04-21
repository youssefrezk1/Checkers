from __future__ import annotations

from checkers.engine.board import BLACK, BLACK_KING, EMPTY, RED, RED_KING
from checkers.engine.evaluation import evaluate_board_breakdown


def _empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def test_endgame_king_centralization_prefers_center_king() -> None:
    center_king = _empty_board()
    center_king[3][3] = RED_KING
    center_king[0][1] = BLACK

    edge_king = _empty_board()
    edge_king[0][7] = RED_KING
    edge_king[0][1] = BLACK

    center_breakdown = evaluate_board_breakdown(center_king, RED, RED)
    edge_breakdown = evaluate_board_breakdown(edge_king, RED, RED)
    assert center_breakdown["king_centralization"] > edge_breakdown["king_centralization"]


def test_endgame_king_mobility_prefers_mobile_king() -> None:
    mobile = _empty_board()
    mobile[3][3] = RED_KING
    mobile[0][1] = BLACK

    blocked = _empty_board()
    blocked[0][1] = RED_KING
    blocked[1][0] = RED
    blocked[1][2] = RED
    blocked[7][6] = BLACK

    mobile_breakdown = evaluate_board_breakdown(mobile, RED, RED)
    blocked_breakdown = evaluate_board_breakdown(blocked, RED, RED)
    assert mobile_breakdown["king_mobility"] > blocked_breakdown["king_mobility"]


def test_endgame_king_chase_pressure_prefers_closer_king() -> None:
    close = _empty_board()
    close[3][3] = RED_KING
    close[2][4] = BLACK

    far = _empty_board()
    far[7][0] = RED_KING
    far[0][7] = BLACK

    close_breakdown = evaluate_board_breakdown(close, RED, RED)
    far_breakdown = evaluate_board_breakdown(far, RED, RED)
    assert close_breakdown["king_chase_pressure"] > far_breakdown["king_chase_pressure"]
