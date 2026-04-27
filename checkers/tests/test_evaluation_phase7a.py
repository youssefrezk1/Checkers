from __future__ import annotations

from checkers.engine.board import BLACK, BLACK_KING, EMPTY, RED, RED_KING
from checkers.engine.evaluation import CAGED_KING_PENALTY, evaluate_board_breakdown, _is_king_caged


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


# ── _is_king_caged false-positive prevention tests ────────────────────────────
#
# CAGED_KING_PENALTY = 75 is the highest single-fact penalty in the evaluator.
# A false positive applies −75 pt to every node containing that king for the
# entire endgame.  The tests below lock down the three ways a king must NOT be
# classified as caged, plus one confirmed-caged position (verified by hand
# against the code path in evaluation.py:345-398).


def test_caged_king_center_open_exits_not_caged() -> None:
    # Baseline false-positive prevention: king in center, no opponents anywhere.
    # All four diagonal exits (2,2),(2,4),(4,2),(4,4) are open and unattacked.
    board = _empty_board()
    board[3][3] = RED_KING
    assert not _is_king_caged(board, 3, 3, RED)


def test_caged_king_backward_man_attacker_ignored() -> None:
    # A BLACK man that can only reach an exit square by jumping backward
    # (upward, decreasing row — illegal for BLACK men) must NOT count as a threat.
    # Without this constraint a non-caged king would be falsely penalised.
    #
    # Setup: RED_KING at (3,3); BLACK man at (5,3).
    # Exit (4,2): BLACK at (5,3) would jump from row 5 to row 3 (backward) — illegal.
    # Exit (4,4): BLACK at (5,3) would jump from row 5 to row 3 (backward) — illegal.
    # Exits (2,2),(2,4) have no attacker at all.
    # → King is NOT caged.
    board = _empty_board()
    board[3][3] = RED_KING
    board[5][3] = BLACK   # man; can only jump forward (downward, row increases)
    assert not _is_king_caged(board, 3, 3, RED)


def test_caged_king_frozen_all_exits_blocked_by_own_pieces_not_caged() -> None:
    # A king whose every diagonal neighbour is occupied by a friendly piece has
    # zero destinations.  The function explicitly returns False for that case
    # ("frozen king — not caged in the all-exits-losing sense").
    board = _empty_board()
    board[3][3] = RED_KING
    board[2][2] = RED   # blocks exit (2,2)
    board[2][4] = RED   # blocks exit (2,4)
    board[4][2] = RED   # blocks exit (4,2)
    board[4][4] = RED   # blocks exit (4,4)
    assert not _is_king_caged(board, 3, 3, RED)


def test_caged_king_one_safe_exit_not_caged() -> None:
    # Even when multiple exits are immediately recapturable, a single safe exit
    # is enough for the function to return False.
    #
    # Setup: RED_KING at (3,3); BLACK_KING at (5,3).
    # BLACK_KING at (5,3) threatens exits (4,2) and (4,4):
    #   (4,2): attacker (5,3), jump dir (+1,+1), landing (3,1) — empty → unsafe.
    #   (4,4): attacker (5,3), jump dir (+1,-1), landing (3,5) — empty → unsafe.
    # But exits (2,2) and (2,4) have no forward-jumping attacker → safe.
    # → King is NOT caged.
    board = _empty_board()
    board[3][3] = RED_KING
    board[5][3] = BLACK_KING
    assert not _is_king_caged(board, 3, 3, RED)


def test_caged_king_all_exits_immediately_recapturable_is_caged() -> None:
    # Verified caged position (all four exit squares attacked by forward-jumping
    # BLACK men whose landing squares are empty after the king's simulated move):
    #
    #   RED_KING at (2,2);  exits: (1,1),(1,3),(3,1),(3,3)
    #
    #   Exit (1,1): BLACK at (0,0) jumps (0,0)→(1,1)→(2,2)[vacated] — legal ✓
    #   Exit (1,3): BLACK at (0,4) jumps (0,4)→(1,3)→(2,2)[vacated] — legal ✓
    #   Exit (3,1): BLACK at (2,0) jumps (2,0)→(3,1)→(4,2)[empty]   — legal ✓
    #   Exit (3,3): BLACK at (2,4) jumps (2,4)→(3,3)→(4,2)[empty]   — legal ✓
    #
    # All jumps are forward (positive jump_row_dir) and landing squares are
    # empty → every destination is immediately recapturable → caged.
    board = _empty_board()
    board[2][2] = RED_KING
    board[0][0] = BLACK
    board[0][4] = BLACK
    board[2][0] = BLACK
    board[2][4] = BLACK
    assert _is_king_caged(board, 2, 2, RED)


def test_caged_king_breakdown_negative_for_caged_player() -> None:
    # Evaluator surface check: the 'caged_king' term in evaluate_board_breakdown
    # must be strictly negative when root_player's king is caged and the
    # opponent has no caged kings.
    # Board is the same confirmed-caged position from the test above.
    # Total pieces = 5 (≤ ENDGAME_FEATURE_PIECE_THRESHOLD=14) → phase-7a fires.
    board = _empty_board()
    board[2][2] = RED_KING
    board[0][0] = BLACK
    board[0][4] = BLACK
    board[2][0] = BLACK
    board[2][4] = BLACK

    breakdown = evaluate_board_breakdown(board, RED, RED)
    assert breakdown["caged_king"] < 0, (
        f"Expected negative caged_king contribution (RED king is caged), "
        f"got {breakdown['caged_king']}"
    )
    assert breakdown["caged_king"] == -float(CAGED_KING_PENALTY), (
        f"Expected exactly -{CAGED_KING_PENALTY}, got {breakdown['caged_king']}"
    )
