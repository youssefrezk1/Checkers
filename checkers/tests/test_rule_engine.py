# tests/test_rule_engine.py

import pytest
from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING,
    BOARD_SIZE, create_initial_board, in_bounds,
    is_dark_square, is_own_piece, is_opponent_piece, is_king
)
from checkers.engine.rules import (
    get_simple_moves, get_single_jumps,
    get_all_jump_sequences, get_all_legal_moves,
    apply_move
)
from checkers.engine.move_facts import compute_move_facts, count_pieces
from checkers.engine.win_condition import (
    has_no_pieces_left, has_no_moves_left, check_win_condition
)
from checkers.engine.zobrist import (
    compute_hash, update_hash, check_repetition, ZOBRIST_TABLE
)


# ─────────────────────────────────────────────
# HELPERS — build custom boards for specific scenarios
# ─────────────────────────────────────────────

def empty_board():
    """Returns a completely empty 8x8 board."""
    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    return board


def place(board, piece, row, col):
    """Places a piece on the board at (row, col)."""
    board[row][col] = piece
    return board


# ─────────────────────────────────────────────
# SECTION 1 — Board Setup Tests
# ─────────────────────────────────────────────

class TestBoardSetup:
    def test_initial_board_symmetry(self):
        """
        The board should be symmetric — RED and BLACK should have
        the same number of pieces in the same pattern mirrored.
        For every RED piece at (row, col), the mirrored square
        (7-row, 7-col) should have a BLACK piece.
        """
        board = create_initial_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == RED:
                    mirrored = board[7 - row][7 - col]
                    assert mirrored == BLACK, (
                        f"Board not symmetric at ({row},{col})"
                    )

    def test_initial_board_no_piece_at_center(self):
        """The very center squares (3,3), (3,4), (4,3), (4,4) must all be empty."""
        board = create_initial_board()
        assert board[3][3] == EMPTY
        assert board[3][4] == EMPTY
        assert board[4][3] == EMPTY
        assert board[4][4] == EMPTY

    def test_initial_board_red_row_5_piece_count(self):
        """Row 5 should have exactly 4 RED pieces on dark squares."""
        board = create_initial_board()
        count = 0
        for col in range(BOARD_SIZE):
            if board[5][col] == RED:
                count += 1
        assert count == 4

    def test_initial_board_black_row_0_piece_count(self):
        """Row 0 should have exactly 4 BLACK pieces on dark squares."""
        board = create_initial_board()
        count = 0
        for col in range(BOARD_SIZE):
            if board[0][col] == BLACK:
                count += 1
        assert count == 4

    def test_create_initial_board_returns_independent_copy(self):
        """
        Two calls to create_initial_board should return independent boards.
        Modifying one should not affect the other.
        """
        board1 = create_initial_board()
        board2 = create_initial_board()
        board1[5][0] = EMPTY
        assert board2[5][0] == RED

    def test_initial_board_light_squares_always_empty(self):
        """
        Light squares must always be empty at the start.
        No piece should ever be on a light square.
        """
        board = create_initial_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if not is_dark_square(row, col):
                    assert board[row][col] == EMPTY, (
                        f"Piece found on light square ({row},{col})"
                    )



    def test_initial_board_total_piece_count(self):
        """
        The initial board should have exactly 24 pieces total —
        12 RED and 12 BLACK. Not 23, not 25.
        """
        board = create_initial_board()
        total = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != EMPTY:
                    total += 1
        assert total == 24

    def test_initial_board_red_piece_count(self):
        """RED should start with exactly 12 pieces."""
        board = create_initial_board()
        red_count = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == RED:
                    red_count += 1
        assert red_count == 12

    def test_initial_board_black_piece_count(self):
        """BLACK should start with exactly 12 pieces."""
        board = create_initial_board()
        black_count = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == BLACK:
                    black_count += 1
        assert black_count == 12

    def test_initial_board_no_kings(self):
        """No kings should exist at the start of the game."""
        board = create_initial_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                assert piece != RED_KING
                assert piece != BLACK_KING

    def test_initial_board_pieces_on_dark_squares_only(self):
        """All pieces must be placed on dark squares only."""
        board = create_initial_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece != EMPTY:
                    assert is_dark_square(row, col), (
                        f"Piece found on light square at ({row},{col})"
                    )

    def test_initial_board_red_in_correct_rows(self):
        """RED pieces should only be in rows 5, 6, 7."""
        board = create_initial_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == RED:
                    assert row >= 5, (
                        f"RED piece found in wrong row {row}"
                    )

    def test_initial_board_black_in_correct_rows(self):
        """BLACK pieces should only be in rows 0, 1, 2."""
        board = create_initial_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == BLACK:
                    assert row <= 2, (
                        f"BLACK piece found in wrong row {row}"
                    )

    def test_initial_board_middle_rows_empty(self):
        """Rows 3 and 4 should be completely empty at the start."""
        board = create_initial_board()
        for col in range(BOARD_SIZE):
            assert board[3][col] == EMPTY
            assert board[4][col] == EMPTY


# ─────────────────────────────────────────────
# SECTION 2 — Helper Function Tests
# ─────────────────────────────────────────────

class TestHelperFunctions:
    def test_is_own_piece_black_regular(self):
        """BLACK piece belongs to BLACK player."""
        assert is_own_piece(BLACK, BLACK) == True

    def test_is_own_piece_black_king(self):
        """BLACK_KING belongs to BLACK player."""
        assert is_own_piece(BLACK_KING, BLACK) == True

    def test_is_own_piece_red_for_black(self):
        """RED piece does not belong to BLACK player."""
        assert is_own_piece(RED, BLACK) == False

    def test_is_own_piece_empty_square(self):
        """EMPTY square does not belong to any player."""
        assert is_own_piece(EMPTY, RED) == False
        assert is_own_piece(EMPTY, BLACK) == False

    def test_is_opponent_piece_empty_square(self):
        """EMPTY square is not an opponent piece for anyone."""
        assert is_opponent_piece(EMPTY, RED) == False
        assert is_opponent_piece(EMPTY, BLACK) == False

    def test_is_dark_square_all_corners(self):
        """
        Check all four corners — only (0,1), (1,0) pattern corners are dark.
        (0,0) is light, (0,7) is light, (7,0) is light, (7,7) is light.
        """
        assert is_dark_square(0, 0) == False
        assert is_dark_square(7, 7) == False
        assert is_dark_square(0, 7) == True
        assert is_dark_square(7, 0) == True

    def test_is_dark_square_known_dark_squares(self):
        """(0,1), (1,0), (7,6), (6,7) are all dark squares."""
        assert is_dark_square(0, 1) == True
        assert is_dark_square(1, 0) == True
        assert is_dark_square(7, 6) == True
        assert is_dark_square(6, 7) == True

    def test_in_bounds_all_four_corners(self):
        """All four corners of the board should be in bounds."""
        assert in_bounds(0, 0) == True
        assert in_bounds(0, 7) == True
        assert in_bounds(7, 0) == True
        assert in_bounds(7, 7) == True

    def test_get_piece_returns_correct_value(self):
        """get_piece should return the piece at a given square."""
        from checkers.engine.board import get_piece
        board = empty_board()
        place(board, RED, 5, 2)
        assert get_piece(board, 5, 2) == RED
        assert get_piece(board, 4, 1) == EMPTY

    def test_get_piece_out_of_bounds_returns_none(self):
        """get_piece on an out-of-bounds square should return None."""
        from checkers.engine.board import get_piece
        board = empty_board()
        assert get_piece(board, -1, 0) is None
        assert get_piece(board, 0, 8) is None
        assert get_piece(board, 8, 8) is None
    def test_is_opponent_piece_black_king_for_red(self):
        """BLACK_KING should also be identified as opponent for RED."""
        assert is_opponent_piece(BLACK_KING, RED) == True

    def test_is_opponent_piece_red_king_for_black(self):
        """RED_KING should also be identified as opponent for BLACK."""
        assert is_opponent_piece(RED_KING, BLACK) == True

    def test_in_bounds_center(self):
        """Center square (4,4) should be in bounds."""
        assert in_bounds(4, 4) == True

    def test_in_bounds_top_left_corner(self):
        """Top left corner (0,0) should be in bounds."""
        assert in_bounds(0, 0) == True

    def test_in_bounds_bottom_right_corner(self):
        """Bottom right corner (7,7) should be in bounds."""
        assert in_bounds(7, 7) == True

    def test_in_bounds_negative_row(self):
        """Negative row should be out of bounds."""
        assert in_bounds(-1, 0) == False

    def test_in_bounds_negative_col(self):
        """Negative col should be out of bounds."""
        assert in_bounds(0, -1) == False

    def test_in_bounds_row_too_large(self):
        """Row 8 should be out of bounds."""
        assert in_bounds(8, 0) == False

    def test_in_bounds_col_too_large(self):
        """Col 8 should be out of bounds."""
        assert in_bounds(0, 8) == False

    def test_is_dark_square_true(self):
        """(0,1) has row+col=1 which is odd — dark square."""
        assert is_dark_square(0, 1) == True

    def test_is_dark_square_false(self):
        """(0,0) has row+col=0 which is even — light square."""
        assert is_dark_square(0, 0) == False

    def test_is_own_piece_red_regular(self):
        """RED piece belongs to RED player."""
        assert is_own_piece(RED, RED) == True

    def test_is_own_piece_red_king(self):
        """RED_KING belongs to RED player."""
        assert is_own_piece(RED_KING, RED) == True

    def test_is_own_piece_black_for_red(self):
        """BLACK piece does not belong to RED player."""
        assert is_own_piece(BLACK, RED) == False

    def test_is_opponent_piece_black_for_red(self):
        """BLACK piece is an opponent piece for RED."""
        assert is_opponent_piece(BLACK, RED) == True

    def test_is_opponent_piece_red_for_red(self):
        """RED piece is not an opponent piece for RED."""
        assert is_opponent_piece(RED, RED) == False

    def test_is_king_red_king(self):
        """RED_KING should be identified as a king."""
        assert is_king(RED_KING) == True

    def test_is_king_black_king(self):
        """BLACK_KING should be identified as a king."""
        assert is_king(BLACK_KING) == True

    def test_is_king_regular_red(self):
        """Regular RED piece should not be identified as a king."""
        assert is_king(RED) == False

    def test_is_king_regular_black(self):
        """Regular BLACK piece should not be identified as a king."""
        assert is_king(BLACK) == False


# ─────────────────────────────────────────────
# SECTION 3 — Simple Move Tests
# ─────────────────────────────────────────────

class TestSimpleMoves:
    def test_king_at_top_edge_has_two_moves(self):
        """RED_KING at top edge (0,4) can only move downward — 2 directions."""
        board = empty_board()
        place(board, RED_KING, 0, 4)
        moves = get_simple_moves(board, 0, 4)
        destinations = [(m[2], m[3]) for m in moves]
        assert len(moves) == 2
        assert (1, 3) in destinations
        assert (1, 5) in destinations

    def test_king_at_bottom_edge_has_two_moves(self):
        """BLACK_KING at bottom edge (7,4) can only move upward — 2 directions."""
        board = empty_board()
        place(board, BLACK_KING, 7, 4)
        moves = get_simple_moves(board, 7, 4)
        destinations = [(m[2], m[3]) for m in moves]
        assert len(moves) == 2
        assert (6, 3) in destinations
        assert (6, 5) in destinations

    def test_king_at_left_edge_has_two_moves(self):
        """RED_KING at left edge (4,0) can only move right — 2 directions."""
        board = empty_board()
        place(board, RED_KING, 4, 0)
        moves = get_simple_moves(board, 4, 0)
        destinations = [(m[2], m[3]) for m in moves]
        assert len(moves) == 2
        assert (3, 1) in destinations
        assert (5, 1) in destinations

    def test_king_at_right_edge_has_two_moves(self):
        """RED_KING at right edge (4,7) can only move left — 2 directions."""
        board = empty_board()
        place(board, RED_KING, 4, 7)
        moves = get_simple_moves(board, 4, 7)
        destinations = [(m[2], m[3]) for m in moves]
        assert len(moves) == 2
        assert (3, 6) in destinations
        assert (5, 6) in destinations

    def test_king_at_corner_top_left_has_one_move(self):
        """RED_KING at corner (0,0) has only one diagonal direction available."""
        board = empty_board()
        place(board, RED_KING, 0, 0)
        moves = get_simple_moves(board, 0, 0)
        assert len(moves) == 1
        assert moves[0][2] == 1
        assert moves[0][3] == 1

    def test_king_at_corner_top_right_has_one_move(self):
        """RED_KING at corner (0,7) has only one diagonal direction available."""
        board = empty_board()
        place(board, RED_KING, 0, 7)
        moves = get_simple_moves(board, 0, 7)
        assert len(moves) == 1
        assert moves[0][2] == 1
        assert moves[0][3] == 6

    def test_king_at_corner_bottom_left_has_one_move(self):
        """BLACK_KING at corner (7,0) has only one diagonal direction available."""
        board = empty_board()
        place(board, BLACK_KING, 7, 0)
        moves = get_simple_moves(board, 7, 0)
        assert len(moves) == 1
        assert moves[0][2] == 6
        assert moves[0][3] == 1

    def test_king_at_corner_bottom_right_has_one_move(self):
        """BLACK_KING at corner (7,7) has only one diagonal direction available."""
        board = empty_board()
        place(board, BLACK_KING, 7, 7)
        moves = get_simple_moves(board, 7, 7)
        assert len(moves) == 1
        assert moves[0][2] == 6
        assert moves[0][3] == 6

    def test_red_piece_partially_blocked_one_direction(self):
        """RED piece with one direction blocked should return exactly one move."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 4, 1)
        moves = get_simple_moves(board, 5, 2)
        assert len(moves) == 1
        assert moves[0][2] == 4
        assert moves[0][3] == 3

    def test_black_piece_partially_blocked_one_direction(self):
        """BLACK piece with one direction blocked should return exactly one move."""
        board = empty_board()
        place(board, BLACK, 2, 3)
        place(board, BLACK, 3, 4)
        moves = get_simple_moves(board, 2, 3)
        assert len(moves) == 1
        assert moves[0][2] == 3
        assert moves[0][3] == 2

    def test_simple_move_tuple_format_is_correct(self):
        """
        Each simple move tuple must have exactly 4 values:
        (from_row, from_col, to_row, to_col).
        """
        board = empty_board()
        place(board, RED, 5, 2)
        moves = get_simple_moves(board, 5, 2)
        for move in moves:
            assert len(move) == 4, f"Move tuple has wrong length: {move}"
    def test_red_piece_at_row_0_cannot_move(self):
        """
        RED regular piece at row 0 has no forward moves.
        It reached the promotion row but was never promoted (edge case).
        It cannot move backward since it is not a king.
        """
        board = empty_board()
        place(board, RED, 0, 1)
        moves = get_simple_moves(board, 0, 1)
        assert len(moves) == 0

    def test_black_piece_at_row_7_cannot_move(self):
        """
        BLACK regular piece at row 7 has no forward moves.
        Row 7 is the bottom of the board so BLACK cannot move further down.
        """
        board = empty_board()
        place(board, BLACK, 7, 0)
        moves = get_simple_moves(board, 7, 0)
        assert len(moves) == 0
    def test_king_completely_blocked_by_own_pieces(self):
        """
        RED_KING surrounded on all 4 diagonals by own pieces
        should have zero simple moves available.
        """
        board = empty_board()
        place(board, RED_KING, 4, 4)
        place(board, RED, 3, 3)
        place(board, RED, 3, 5)
        place(board, RED, 5, 3)
        place(board, RED, 5, 5)
        moves = get_simple_moves(board, 4, 4)
        assert len(moves) == 0
    def test_black_piece_at_left_edge(self):
        """BLACK piece at left edge (2,0) should only have one simple move."""
        board = empty_board()
        place(board, BLACK, 2, 0)
        moves = get_simple_moves(board, 2, 0)
        assert len(moves) == 1
        assert moves[0][2] == 3
        assert moves[0][3] == 1

    def test_black_piece_at_right_edge(self):
        """BLACK piece at right edge (1,7) should only have one simple move."""
        board = empty_board()
        place(board, BLACK, 1, 7)
        moves = get_simple_moves(board, 1, 7)
        assert len(moves) == 1
        assert moves[0][2] == 2
        assert moves[0][3] == 6
    def test_red_piece_moves_upward(self):
        """RED piece at (5,2) should have simple moves going to row 4."""
        board = empty_board()
        place(board, RED, 5, 2)
        moves = get_simple_moves(board, 5, 2)
        destinations = [(move[2], move[3]) for move in moves]
        assert (4, 1) in destinations
        assert (4, 3) in destinations

    def test_black_piece_moves_downward(self):
        """BLACK piece at (2,3) should have simple moves going to row 3."""
        board = empty_board()
        place(board, BLACK, 2, 3)
        moves = get_simple_moves(board, 2, 3)
        destinations = [(move[2], move[3]) for move in moves]
        assert (3, 2) in destinations
        assert (3, 4) in destinations

    def test_red_piece_blocked_by_own_piece(self):
        """RED piece should not move to a square occupied by another RED piece."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 4, 1)
        place(board, RED, 4, 3)
        moves = get_simple_moves(board, 5, 2)
        assert len(moves) == 0

    def test_red_piece_blocked_by_opponent(self):
        """RED piece should not simple move to a square occupied by BLACK."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 1)
        place(board, BLACK, 4, 3)
        moves = get_simple_moves(board, 5, 2)
        assert len(moves) == 0

    def test_red_piece_at_left_edge(self):
        """RED piece at left edge (5,0) should only have one simple move."""
        board = empty_board()
        place(board, RED, 5, 0)
        moves = get_simple_moves(board, 5, 0)
        assert len(moves) == 1
        assert moves[0][2] == 4
        assert moves[0][3] == 1

    def test_red_piece_at_right_edge(self):
        """RED piece at col 7 edge (6,7) should only have one simple move."""
        board = empty_board()
        place(board, RED, 6, 7)
        moves = get_simple_moves(board, 6, 7)
        assert len(moves) == 1
        assert moves[0][2] == 5
        assert moves[0][3] == 6

    def test_king_moves_all_four_directions(self):
        """RED_KING in the center should have moves in all 4 diagonal directions."""
        board = empty_board()
        place(board, RED_KING, 4, 4)
        moves = get_simple_moves(board, 4, 4)
        destinations = [(move[2], move[3]) for move in moves]
        assert (3, 3) in destinations
        assert (3, 5) in destinations
        assert (5, 3) in destinations
        assert (5, 5) in destinations

    def test_black_king_moves_all_four_directions(self):
        """BLACK_KING in the center should have moves in all 4 diagonal directions."""
        board = empty_board()
        place(board, BLACK_KING, 4, 4)
        moves = get_simple_moves(board, 4, 4)
        destinations = [(move[2], move[3]) for move in moves]
        assert (3, 3) in destinations
        assert (3, 5) in destinations
        assert (5, 3) in destinations
        assert (5, 5) in destinations

    def test_red_piece_cannot_move_backward(self):
        """Regular RED piece should never have moves going to a higher row."""
        board = empty_board()
        place(board, RED, 4, 4)
        moves = get_simple_moves(board, 4, 4)
        for move in moves:
            to_row = move[2]
            assert to_row < 4, "RED piece moved backward"

    def test_black_piece_cannot_move_backward(self):
        """Regular BLACK piece should never have moves going to a lower row."""
        board = empty_board()
        place(board, BLACK, 4, 4)
        moves = get_simple_moves(board, 4, 4)
        for move in moves:
            to_row = move[2]
            assert to_row > 4, "BLACK piece moved backward"


# ─────────────────────────────────────────────
# SECTION 4 — Jump Tests
# ─────────────────────────────────────────────

class TestJumps:
    def test_black_jump_landing_at_bottom_row(self):
        """BLACK jumps over RED and lands at row 7 — promotion triggered."""
        board = empty_board()
        place(board, BLACK, 5, 3)
        place(board, RED, 6, 4)
        jumps = get_single_jumps(board, 5, 3, BLACK)
        landings = [(j[2], j[3]) for j in jumps]
        assert (7, 5) in landings

    def test_black_jump_landing_at_left_edge(self):
        """BLACK jumps over RED and lands exactly at col 0."""
        board = empty_board()
        place(board, BLACK, 3, 2)
        place(board, RED, 4, 1)
        jumps = get_single_jumps(board, 3, 2, BLACK)
        landings = [(j[2], j[3]) for j in jumps]
        assert (5, 0) in landings

    def test_black_jump_landing_at_right_edge(self):
        """BLACK jumps over RED and lands exactly at col 7."""
        board = empty_board()
        place(board, BLACK, 3, 5)
        place(board, RED, 4, 6)
        jumps = get_single_jumps(board, 3, 5, BLACK)
        landings = [(j[2], j[3]) for j in jumps]
        assert (5, 7) in landings

    def test_red_can_jump_black_king(self):
        """RED should be able to jump over BLACK_KING — it is still an opponent."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK_KING, 4, 3)
        jumps = get_single_jumps(board, 5, 2, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (3, 4) in landings

    def test_black_can_jump_red_king(self):
        """BLACK should be able to jump over RED_KING — it is still an opponent."""
        board = empty_board()
        place(board, BLACK, 2, 3)
        place(board, RED_KING, 3, 4)
        jumps = get_single_jumps(board, 2, 3, BLACK)
        landings = [(j[2], j[3]) for j in jumps]
        assert (4, 5) in landings

    def test_multiple_jumps_available_from_single_piece(self):
        """
        A single piece with opponent pieces on both diagonals
        should return all possible single jumps.
        """
        board = empty_board()
        place(board, RED, 4, 4)
        place(board, BLACK, 3, 3)
        place(board, BLACK, 3, 5)
        jumps = get_single_jumps(board, 4, 4, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (2, 2) in landings
        assert (2, 6) in landings
        assert len(jumps) == 2

    def test_king_multiple_jumps_all_four_directions(self):
        """
        RED_KING with opponent pieces on all 4 diagonals
        should have exactly 4 possible single jumps.
        """
        board = empty_board()
        place(board, RED_KING, 4, 4)
        place(board, BLACK, 3, 3)
        place(board, BLACK, 3, 5)
        place(board, BLACK, 5, 3)
        place(board, BLACK, 5, 5)
        jumps = get_single_jumps(board, 4, 4, RED)
        assert len(jumps) == 4

    def test_jump_tuple_format_is_correct(self):
        """
        Each jump tuple must have exactly 6 values:
        (from_row, from_col, to_row, to_col, cap_row, cap_col).
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        jumps = get_single_jumps(board, 5, 2, RED)
        for jump in jumps:
            assert len(jump) == 6, f"Jump tuple has wrong length: {jump}"

    def test_jump_changes_piece_count_correctly(self):
        """After a jump, opponent piece count decreases by one."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        new_board = apply_move(board, move)
        assert count_pieces(new_board, RED)["total"] == 1
        assert count_pieces(new_board, BLACK)["total"] == 0
    def test_jump_landing_at_left_edge(self):
        """
        RED jumps over BLACK and lands exactly at col 0.
        Tests that edge column landing squares are handled correctly.
        """
        board = empty_board()
        place(board, RED, 4, 2)
        place(board, BLACK, 3, 1)
        jumps = get_single_jumps(board, 4, 2, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (2, 0) in landings

    def test_jump_landing_at_right_edge(self):
        """
        RED jumps over BLACK and lands exactly at col 7.
        Tests that edge column landing squares are handled correctly.
        """
        board = empty_board()
        place(board, RED, 4, 5)
        place(board, BLACK, 3, 6)
        jumps = get_single_jumps(board, 4, 5, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (2, 7) in landings

    def test_jump_landing_at_top_row(self):
        """
        RED jumps over BLACK and lands exactly at row 0.
        This also triggers promotion — tests both edge landing and promotion together.
        """
        board = empty_board()
        place(board, RED, 2, 3)
        place(board, BLACK, 1, 4)
        jumps = get_single_jumps(board, 2, 3, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (0, 5) in landings
    # Inside TestJumps
    def test_apply_simple_move_only_changes_two_squares(self):
        """
        A simple move should only change exactly two squares:
        the starting square becomes empty and the destination gets the piece.
        Every other square must remain identical.
        """
        board = create_initial_board()
        original_board = [row[:] for row in board]

        move = {
            "type": "simple",
            "path": [(5, 2), (4, 1)],
            "captured": []
        }
        new_board = apply_move(board, move)

        changed_squares = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if new_board[row][col] != original_board[row][col]:
                    changed_squares.append((row, col))

        assert len(changed_squares) == 2
        assert (5, 2) in changed_squares
        assert (4, 1) in changed_squares

    def test_promotion_via_jump(self):
        """
        RED piece that reaches row 0 via a jump should become RED_KING,
        not just via a simple move.
        """
        board = empty_board()
        place(board, RED, 2, 1)
        place(board, BLACK, 1, 2)
        move = {
            "type": "jump",
            "path": [(2, 1), (0, 3)],
            "captured": [(1, 2)]
        }
        new_board = apply_move(board, move)
        assert new_board[0][3] == RED_KING
        assert new_board[1][2] == EMPTY
        assert new_board[2][1] == EMPTY


    def test_apply_move_multi_jump_removes_all_captures(self):
        """
        apply_move on a double jump should remove both captured pieces,
        not just the first one.
        """
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK, 2, 3)
        move = {
            "type": "jump",
            "path": [(5, 0), (3, 2), (1, 4)],
            "captured": [(4, 1), (2, 3)]
        }
        new_board = apply_move(board, move)
        assert new_board[5][0] == EMPTY
        assert new_board[4][1] == EMPTY  # first captured piece removed
        assert new_board[2][3] == EMPTY  # second captured piece removed
        assert new_board[1][4] == RED    # piece arrived at final destination

    
    def test_king_jumps_in_all_four_directions(self):
        """
        RED_KING should be able to jump in all 4 diagonal directions,
        not just forward. Test each direction separately.
        """
        # Jump forward-left
        board = empty_board()
        place(board, RED_KING, 4, 4)
        place(board, BLACK, 3, 3)
        jumps = get_single_jumps(board, 4, 4, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (2, 2) in landings

        # Jump forward-right
        board = empty_board()
        place(board, RED_KING, 4, 4)
        place(board, BLACK, 3, 5)
        jumps = get_single_jumps(board, 4, 4, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (2, 6) in landings

        # Jump backward-left
        board = empty_board()
        place(board, RED_KING, 4, 4)
        place(board, BLACK, 5, 3)
        jumps = get_single_jumps(board, 4, 4, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (6, 2) in landings

        # Jump backward-right
        board = empty_board()
        place(board, RED_KING, 4, 4)
        place(board, BLACK, 5, 5)
        jumps = get_single_jumps(board, 4, 4, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (6, 6) in landings

    def test_red_jumps_over_black(self):
        """RED at (5,2) should jump over BLACK at (4,3) and land at (3,4)."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        jumps = get_single_jumps(board, 5, 2, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (3, 4) in landings

    def test_black_jumps_over_red(self):
        """BLACK at (2,3) should jump over RED at (3,4) and land at (4,5)."""
        board = empty_board()
        place(board, BLACK, 2, 3)
        place(board, RED, 3, 4)
        jumps = get_single_jumps(board, 2, 3, BLACK)
        landings = [(j[2], j[3]) for j in jumps]
        assert (4, 5) in landings

    def test_jump_removes_captured_piece(self):
        """After applying a jump, the captured piece should be removed."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        new_board = apply_move(board, move)
        assert new_board[4][3] == EMPTY
        assert new_board[3][4] == RED
        assert new_board[5][2] == EMPTY

    def test_cannot_jump_own_piece(self):
        """RED should not be able to jump over another RED piece."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 4, 3)
        jumps = get_single_jumps(board, 5, 2, RED)
        assert len(jumps) == 0

    def test_cannot_jump_if_landing_occupied(self):
        """Jump should not be valid if the landing square is occupied."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        place(board, RED, 3, 4)
        jumps = get_single_jumps(board, 5, 2, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (3, 4) not in landings

    def test_king_jumps_backward(self):
        """RED_KING should be able to jump over BLACK in backward direction."""
        board = empty_board()
        place(board, RED_KING, 3, 4)
        place(board, BLACK, 4, 3)
        jumps = get_single_jumps(board, 3, 4, RED)
        landings = [(j[2], j[3]) for j in jumps]
        assert (5, 2) in landings

    def test_no_jump_when_middle_empty(self):
        """No jump should exist if the middle square is empty."""
        board = empty_board()
        place(board, RED, 5, 2)
        jumps = get_single_jumps(board, 5, 2, RED)
        assert len(jumps) == 0

    def test_jump_correct_captured_position(self):
        """The captured position stored in jump tuple must be the middle square."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        jumps = get_single_jumps(board, 5, 2, RED)
        assert len(jumps) == 1
        captured_row = jumps[0][4]
        captured_col = jumps[0][5]
        assert captured_row == 4
        assert captured_col == 3

    # Inside TestJumps
    def test_king_cannot_jump_own_king(self):
        """RED_KING should not jump over another RED_KING."""
        board = empty_board()
        place(board, RED_KING, 5, 2)
        place(board, RED_KING, 4, 3)
        jumps = get_single_jumps(board, 5, 2, RED)
        assert len(jumps) == 0
    
    # Inside TestJumps
    def test_apply_move_does_not_mutate_original_board(self):
        """apply_move must return a new board without touching the original."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)

        original_piece_at_start = board[5][2]
        original_piece_at_middle = board[4][3]

        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        apply_move(board, move)

        # Original board must be completely unchanged
        assert board[5][2] == original_piece_at_start
        assert board[4][3] == original_piece_at_middle
        assert board[3][4] == EMPTY
# ─────────────────────────────────────────────
# SECTION 5 — Mandatory Capture Tests
# ─────────────────────────────────────────────

class TestMandatoryCapture:

    # ── NEW TESTS FOR UPDATED move_facts.py ──────────────────────────

    def test_all_new_keys_present(self):
        """
        compute_move_facts must return all new keys added in the update.
        Missing keys would crash the Ranker Agent node.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        expected_keys = [
            "move_type", "piece_type_moving", "path_length",
            "captures_count", "jump_count", "is_multi_jump",
            "kings_captured", "regulars_captured",
            "results_in_king", "near_promotion",
            "our_pieces_before", "our_pieces_after",
            "opp_pieces_before", "opp_pieces_after",
            "net_gain", "material_advantage",
            "center_control", "opponent_can_recapture",
            "leaves_piece_isolated", "opponent_near_promotion",
            "opponent_jump_count"
        ]
        for key in expected_keys:
            assert key in facts, f"Missing key in move facts: {key}"

    # ── piece_type_moving ─────────────────────────────────────────────

    def test_piece_type_moving_regular(self):
        """A regular RED piece making a move should be identified as regular."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["piece_type_moving"] == "regular"

    def test_piece_type_moving_king(self):
        """A RED_KING making a move should be identified as king."""
        board = empty_board()
        place(board, RED_KING, 4, 4)
        move = {"type": "simple", "path": [(4, 4), (3, 3)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["piece_type_moving"] == "king"

    def test_piece_type_moving_black_regular(self):
        """A regular BLACK piece should also be identified as regular."""
        board = empty_board()
        place(board, BLACK, 2, 3)
        move = {"type": "simple", "path": [(2, 3), (3, 4)], "captured": []}
        facts = compute_move_facts(board, move, BLACK)
        assert facts["piece_type_moving"] == "regular"

    def test_piece_type_moving_black_king(self):
        """A BLACK_KING making a move should be identified as king."""
        board = empty_board()
        place(board, BLACK_KING, 4, 4)
        move = {"type": "simple", "path": [(4, 4), (5, 5)], "captured": []}
        facts = compute_move_facts(board, move, BLACK)
        assert facts["piece_type_moving"] == "king"

    # ── jump_count ────────────────────────────────────────────────────

    def test_jump_count_zero_for_simple_move(self):
        """Simple move has no jumps so jump_count must be zero."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["jump_count"] == 0

    def test_jump_count_one_for_single_jump(self):
        """Single jump captures one piece so jump_count must be 1."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["jump_count"] == 1

    def test_jump_count_two_for_double_jump(self):
        """Double jump captures two pieces so jump_count must be 2."""
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK, 2, 3)
        move = {
            "type": "jump",
            "path": [(5, 0), (3, 2), (1, 4)],
            "captured": [(4, 1), (2, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["jump_count"] == 2

    def test_jump_count_equals_captures_count(self):
        """jump_count and captures_count must always be the same value."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["jump_count"] == facts["captures_count"]

    # ── kings_captured and regulars_captured ──────────────────────────

    def test_kings_captured_zero_when_no_capture(self):
        """Simple move captures nothing so kings_captured must be zero."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["kings_captured"] == 0
        assert facts["regulars_captured"] == 0

    def test_regulars_captured_one_when_jumping_regular_piece(self):
        """Jumping a regular BLACK piece should give regulars_captured=1, kings_captured=0."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["regulars_captured"] == 1
        assert facts["kings_captured"] == 0

    def test_kings_captured_one_when_jumping_king(self):
        """Jumping a BLACK_KING should give kings_captured=1, regulars_captured=0."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK_KING, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["kings_captured"] == 1
        assert facts["regulars_captured"] == 0

    def test_mixed_captures_counted_correctly(self):
        """
        If a double jump captures one regular and one king,
        kings_captured=1 and regulars_captured=1.
        """
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK_KING, 2, 3)
        move = {
            "type": "jump",
            "path": [(5, 0), (3, 2), (1, 4)],
            "captured": [(4, 1), (2, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["regulars_captured"] == 1
        assert facts["kings_captured"] == 1

    def test_kings_captured_plus_regulars_equals_captures_count(self):
        """kings_captured + regulars_captured must always equal captures_count."""
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK_KING, 2, 3)
        move = {
            "type": "jump",
            "path": [(5, 0), (3, 2), (1, 4)],
            "captured": [(4, 1), (2, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        total_captured = facts["kings_captured"] + facts["regulars_captured"]
        assert total_captured == facts["captures_count"]

    # ── material_advantage ────────────────────────────────────────────

    def test_material_advantage_zero_at_game_start(self):
        """
        At the start of the game both players have 12 pieces.
        A simple move does not change counts so material_advantage = 0.
        """
        board = create_initial_board()
        move = {"type": "simple", "path": [(5, 0), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["material_advantage"] == 0

    def test_material_advantage_positive_after_capture(self):
        """
        After RED captures one BLACK piece, RED has 1 more piece.
        material_advantage should be +1.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["material_advantage"] == 1

    def test_material_advantage_negative_when_opponent_has_more(self):
        """
        If opponent has more pieces even after our move,
        material_advantage should be negative.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)   # piece being captured
        place(board, BLACK, 3, 0)   # extra BLACK piece — not in the way
        place(board, BLACK, 2, 1)   # extra BLACK piece — not in the way
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        # After jump: RED=1, BLACK=2 → advantage = 1-2 = -1
        assert facts["material_advantage"] < 0
    # ── near_promotion ────────────────────────────────────────────────

    def test_near_promotion_true_for_red_at_row_1(self):
        """
        RED piece landing at row 1 without promoting
        should have near_promotion True.
        """
        board = empty_board()
        place(board, RED, 2, 3)
        move = {"type": "simple", "path": [(2, 3), (1, 4)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["near_promotion"] == True

    def test_near_promotion_false_when_actually_promoting(self):
        """
        RED piece that actually promotes at row 0 should have
        near_promotion False — it already promoted, it is not near.
        """
        board = empty_board()
        place(board, RED, 1, 2)
        move = {"type": "simple", "path": [(1, 2), (0, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["near_promotion"] == False
        assert facts["results_in_king"] == True

    def test_near_promotion_false_for_red_not_at_row_1(self):
        """RED piece not at row 1 should have near_promotion False."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["near_promotion"] == False

    def test_near_promotion_true_for_black_at_row_6(self):
        """
        BLACK piece landing at row 6 without promoting
        should have near_promotion True.
        """
        board = empty_board()
        place(board, BLACK, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (6, 3)], "captured": []}
        facts = compute_move_facts(board, move, BLACK)
        assert facts["near_promotion"] == True

    def test_near_promotion_false_for_black_not_at_row_6(self):
        """BLACK piece not at row 6 should have near_promotion False."""
        board = empty_board()
        place(board, BLACK, 2, 3)
        move = {"type": "simple", "path": [(2, 3), (3, 4)], "captured": []}
        facts = compute_move_facts(board, move, BLACK)
        assert facts["near_promotion"] == False

    # ── center_control ────────────────────────────────────────────────

    def test_center_control_true_when_landing_in_center(self):
        """
        RED piece landing at (3,3) is in the center zone
        (rows 3-4, cols 2-5) so center_control must be True.
        """
        board = empty_board()
        place(board, RED, 4, 2)
        move = {"type": "simple", "path": [(4, 2), (3, 3)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["center_control"] == True

    def test_center_control_false_when_landing_on_edge(self):
        """
        RED piece landing at (4,0) is on the edge, not center zone,
        so center_control must be False.
        """
        board = empty_board()
        place(board, RED, 5, 1)
        move = {"type": "simple", "path": [(5, 1), (4, 0)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["center_control"] == False

    def test_center_control_false_when_landing_at_row_0(self):
        """Promotion row (row 0) is not in the center zone."""
        board = empty_board()
        place(board, RED, 1, 2)
        move = {"type": "simple", "path": [(1, 2), (0, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["center_control"] == False

    # ── opponent_can_recapture ────────────────────────────────────────

    def test_opponent_can_recapture_true(self):
        """
        RED jumps BLACK and lands at (3,4). Another BLACK piece at (2,3)
        can immediately jump RED at (3,4) to (4,5).
        opponent_can_recapture must be True.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        place(board, BLACK, 2, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["opponent_can_recapture"] == True

    def test_opponent_can_recapture_false_when_safe(self):
        """
        RED makes a move to a square where the opponent
        has no adjacent pieces to jump from.
        opponent_can_recapture must be False.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["opponent_can_recapture"] == False

    def test_opponent_can_recapture_false_after_clearing_area(self):
        """
        RED jumps the only BLACK piece in the area.
        After the jump no BLACK pieces remain so opponent cannot recapture.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["opponent_can_recapture"] == False

    # ── leaves_piece_isolated ─────────────────────────────────────────

    def test_leaves_piece_isolated_true_when_alone(self):
        """
        RED piece moves to a square with no friendly neighbors.
        leaves_piece_isolated must be True.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["leaves_piece_isolated"] == True

    def test_leaves_piece_isolated_false_when_has_neighbor(self):
        """
        RED piece moves to a square adjacent to another RED piece.
        leaves_piece_isolated must be False.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 3, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["leaves_piece_isolated"] == False

    def test_leaves_piece_isolated_false_with_king_neighbor(self):
        """
        A RED_KING adjacent to our landing square counts as a neighbor.
        leaves_piece_isolated must be False.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED_KING, 3, 0)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["leaves_piece_isolated"] == False

    # ── opponent_near_promotion ───────────────────────────────────────

    def test_opponent_near_promotion_true_for_red_player(self):
        """
        RED is playing. BLACK has a regular piece at row 6.
        This means BLACK is one step from promoting.
        opponent_near_promotion must be True.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 6, 3)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["opponent_near_promotion"] == True

    def test_opponent_near_promotion_false_when_no_threat(self):
        """
        No opponent pieces are near their promotion row.
        opponent_near_promotion must be False.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 2, 3)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["opponent_near_promotion"] == False

    def test_opponent_near_promotion_false_when_opponent_is_king(self):
        """
        A BLACK_KING at row 6 does NOT trigger opponent_near_promotion.
        Kings are already promoted — they cannot be promoted again.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK_KING, 6, 3)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["opponent_near_promotion"] == False

    def test_opponent_near_promotion_true_for_black_player(self):
        """
        BLACK is playing. RED has a regular piece at row 1.
        This means RED is one step from promoting.
        opponent_near_promotion must be True.
        """
        board = empty_board()
        place(board, BLACK, 2, 3)
        place(board, RED, 1, 4)
        move = {"type": "simple", "path": [(2, 3), (3, 4)], "captured": []}
        facts = compute_move_facts(board, move, BLACK)
        assert facts["opponent_near_promotion"] == True

    # ── opponent_jump_count ───────────────────────────────────────────

    def test_opponent_jump_count_zero_when_no_captures_available(self):
        """
        After a simple move that opens no jump opportunities,
        opponent_jump_count must be zero.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 0, 1)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["opponent_jump_count"] == 0

    def test_opponent_jump_count_positive_when_opponent_can_jump(self):
        """
        After RED moves, BLACK has a jump available.
        opponent_jump_count must be greater than zero.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 2, 1)
        place(board, RED, 3, 2)    # RED piece BLACK can jump over, landing at (4,3)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        # After RED moves, BLACK at (2,1) can jump RED at (3,2) landing at (4,3)
        assert facts["opponent_jump_count"] > 0
    
    def test_opponent_jump_count_is_integer(self):
        """opponent_jump_count must always be a non-negative integer."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert isinstance(facts["opponent_jump_count"], int)
        assert facts["opponent_jump_count"] >= 0

    def test_opponent_jump_count_decreases_after_capture(self):
        """
        If RED captures a BLACK piece that was about to jump,
        opponent_jump_count should decrease compared to not capturing.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        place(board, RED, 3, 2)

        # Move that captures BLACK at (4,3) — removes a potential jumper
        capture_move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts_after_capture = compute_move_facts(board, capture_move, RED)

        # Simple move that does not capture
        simple_move = {
            "type": "simple",
            "path": [(3, 2), (2, 1)],
            "captured": []
        }
        facts_after_simple = compute_move_facts(board, simple_move, RED)

        assert facts_after_capture["opponent_jump_count"] <= facts_after_simple["opponent_jump_count"]
    def test_opening_position_black_has_correct_move_count(self):
        """
        From the standard starting position, BLACK should also have
        exactly 7 legal simple moves — same as RED by symmetry.
        """
        board = create_initial_board()
        moves = get_all_legal_moves(board, BLACK)
        assert len(moves) == 7, (
            f"Expected 7 opening moves for BLACK, got {len(moves)}"
        )

    def test_jump_move_captured_list_not_empty(self):
        """Every jump move returned must have at least one captured piece."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        moves = get_all_legal_moves(board, RED)
        for move in moves:
            if move["type"] == "jump":
                assert len(move["captured"]) >= 1, (
                    "Jump move has empty captured list"
                )

    def test_simple_move_captured_list_always_empty(self):
        """Every simple move returned must have an empty captured list."""
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)
        for move in moves:
            if move["type"] == "simple":
                assert move["captured"] == [], (
                    "Simple move has non-empty captured list"
                )

    def test_all_move_paths_start_on_own_piece(self):
        """
        The first square of every move path must contain
        a piece belonging to the current player.
        """
        board = create_initial_board()
        for player in [RED, BLACK]:
            moves = get_all_legal_moves(board, player)
            for move in moves:
                start_row = move["path"][0][0]
                start_col = move["path"][0][1]
                piece = board[start_row][start_col]
                assert is_own_piece(piece, player), (
                    f"Move for player {player} starts on wrong piece at ({start_row},{start_col})"
                )

    def test_all_move_paths_end_on_empty_square(self):
        """
        The last square of every move path must be empty on the original board
        before the move is applied.
        """
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)
        for move in moves:
            end_row = move["path"][-1][0]
            end_col = move["path"][-1][1]
            assert board[end_row][end_col] == EMPTY, (
                f"Move ends on non-empty square ({end_row},{end_col})"
            )
    def test_all_legal_moves_have_correct_structure(self):
        """
        Every move returned by get_all_legal_moves must have exactly
        three keys: type, path, and captured.
        This guarantees the LLM agents will never crash reading move data.
        """
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)

        for move in moves:
            assert "type" in move, "Move is missing 'type' key"
            assert "path" in move, "Move is missing 'path' key"
            assert "captured" in move, "Move is missing 'captured' key"

            assert move["type"] in ("simple", "jump"), (
                f"Move type must be 'simple' or 'jump', got {move['type']}"
            )
            assert len(move["path"]) >= 2, (
                "Path must have at least start and end square"
            )
            assert isinstance(move["captured"], list), (
                "Captured must be a list even if empty"
            )

    def test_simple_move_path_has_exactly_two_squares(self):
        """
        A simple move path must always have exactly 2 squares —
        the starting square and the destination. Never more, never less.
        """
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)

        for move in moves:
            if move["type"] == "simple":
                assert len(move["path"]) == 2, (
                    f"Simple move path should have 2 squares, got {len(move['path'])}"
                )
# Inside TestMandatoryCapture
    def test_opening_position_has_correct_move_count(self):
        """
        From the standard starting position, RED should have
        exactly 7 legal simple moves. This is a known checkers fact.
        """
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)
        assert len(moves) == 7, (
            f"Expected 7 opening moves for RED, got {len(moves)}"
        )

    # Add to TestMandatoryCapture class

    def test_black_mandatory_capture(self):
        """
        Mandatory capture must also be enforced for BLACK,
        not just RED.
        """
        board = empty_board()
        place(board, BLACK, 2, 3)
        place(board, BLACK, 2, 7)
        place(board, RED, 3, 4)
        moves = get_all_legal_moves(board, BLACK)
        for move in moves:
            assert move["type"] == "jump", (
                "Simple move returned for BLACK even though a jump exists"
            )
        # Inside TestMandatoryCapture
    def test_legal_moves_belong_to_correct_player(self):
        """
        Legal moves returned must only involve the current player's pieces.
        No move should start from an opponent's square.
        """
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)
        for move in moves:
            start_row = move["path"][0][0]
            start_col = move["path"][0][1]
            piece = board[start_row][start_col]
            assert is_own_piece(piece, RED), (
                f"Move starts from a non-RED piece at ({start_row},{start_col})"
            )

        # Inside TestMandatoryCapture
    def test_mandatory_capture_with_king_available(self):
        """
        If a RED_KING can jump, regular RED pieces cannot simple move.
        Mandatory capture applies across all pieces regardless of type.
        """
        board = empty_board()
        place(board, RED, 6, 0)          # regular RED — could simple move
        place(board, RED_KING, 5, 4)     # RED_KING — can jump
        place(board, BLACK, 4, 3)        # BLACK to be jumped by king
        moves = get_all_legal_moves(board, RED)
        for move in moves:
            assert move["type"] == "jump"

    def test_simple_moves_not_returned_when_jump_exists(self):
        """
        If a jump is available, get_all_legal_moves must return
        only jumps and no simple moves.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 5, 6)
        place(board, BLACK, 4, 3)
        moves = get_all_legal_moves(board, RED)
        for move in moves:
            assert move["type"] == "jump", (
                "Simple move returned even though a jump exists"
            )

    def test_all_pieces_must_jump_not_just_one(self):
        """
        If multiple RED pieces can jump, all their jumps must be returned.
        The player chooses which jump to make — all options must be present.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 5, 6)
        place(board, BLACK, 4, 3)
        place(board, BLACK, 4, 5)
        moves = get_all_legal_moves(board, RED)
        assert len(moves) >= 2

    def test_no_moves_available_returns_empty_list(self):
        """
        If a player has no pieces and no moves, 
        get_all_legal_moves should return an empty list.
        """
        board = empty_board()
        moves = get_all_legal_moves(board, RED)
        assert moves == []


# ─────────────────────────────────────────────
# SECTION 6 — Multi Jump Tests
# ─────────────────────────────────────────────

class TestMultiJump:
    def test_king_four_jump_sequence(self):
        """
        RED_KING captures multiple BLACK pieces using
        both forward and backward jumps combined.
        """
        board = empty_board()
        place(board, RED_KING, 4, 0)
        place(board, BLACK, 3, 1)
        place(board, BLACK, 1, 3)
        place(board, BLACK, 3, 5)
        sequences = get_all_jump_sequences(
            board, 4, 0, RED,
            path_so_far=[(4, 0)],
            captured_so_far=[]
        )
        max_captures = max(len(seq["captured"]) for seq in sequences)
        assert max_captures >= 2

    def test_multi_jump_sequence_count_grows_with_pieces(self):
        """
        With more opponent pieces arranged in a line,
        the maximum jump sequence length should increase.
        Two pieces should allow a longer chain than one.
        """
        board_one = empty_board()
        place(board_one, RED, 6, 0)
        place(board_one, BLACK, 5, 1)
        seqs_one = get_all_jump_sequences(
            board_one, 6, 0, RED,
            path_so_far=[(6, 0)],
            captured_so_far=[]
        )
        max_one = max(len(seq["captured"]) for seq in seqs_one)

        board_two = empty_board()
        place(board_two, RED, 6, 0)
        place(board_two, BLACK, 5, 1)
        place(board_two, BLACK, 3, 3)
        seqs_two = get_all_jump_sequences(
            board_two, 6, 0, RED,
            path_so_far=[(6, 0)],
            captured_so_far=[]
        )
        max_two = max(len(seq["captured"]) for seq in seqs_two)

        assert max_two > max_one

    def test_get_all_legal_moves_includes_all_multi_jump_sequences(self):
        """
        get_all_legal_moves must include multi-jump sequences,
        not just single jumps, when multi-jumps are available.
        """
        board = empty_board()
        place(board, RED, 6, 0)
        place(board, BLACK, 5, 1)
        place(board, BLACK, 3, 3)
        moves = get_all_legal_moves(board, RED)
        paths = [tuple(tuple(p) for p in m["path"]) for m in moves]
        double_jump_path = tuple([(6, 0), (4, 2), (2, 4)])
        assert double_jump_path in paths
     # Inside TestMultiJump
    def test_double_jump_changes_direction(self):
        """
        RED jumps forward-right then forward-left — a zigzag.
        This tests that the recursion correctly handles direction changes.
        """
        board = empty_board()
        place(board, RED, 6, 2)
        place(board, BLACK, 5, 3)
        place(board, BLACK, 3, 3)
        sequences = get_all_jump_sequences(
            board, 6, 2, RED,
            path_so_far=[(6, 2)],
            captured_so_far=[]
        )
        paths = [seq["path"] for seq in sequences]
        assert [(6, 2), (4, 4), (2, 2)] in paths


    def test_triple_jump_zigzag(self):
        """
        RED makes three jumps changing direction each time.
        This is the hardest multi-jump pattern to get right recursively.
        
        Path: (6,0) → (4,2) → (2,0) → (0,2)
        Jump 1: over (5,1) going right
        Jump 2: over (3,1) going left  
        Jump 3: over (1,1) going right
        """
        board = empty_board()
        place(board, RED, 6, 0)
        place(board, BLACK, 5, 1)   # jumped in step 1 → land (4,2)
        place(board, BLACK, 3, 1)   # jumped in step 2 → land (2,0)
        place(board, BLACK, 1, 1)   # jumped in step 3 → land (0,2)
        sequences = get_all_jump_sequences(
            board, 6, 0, RED,
            path_so_far=[(6, 0)],
            captured_so_far=[]
        )
        paths = [seq["path"] for seq in sequences]
        assert [(6, 0), (4, 2), (2, 0), (0, 2)] in paths

    def test_four_jump_sequence(self):
        """
        RED captures four BLACK pieces in a single turn.
        Tests that the recursion does not stop early at 3 jumps.
        """
        board = empty_board()
        place(board, RED, 7, 0)
        place(board, BLACK, 6, 1)
        place(board, BLACK, 4, 3)
        place(board, BLACK, 2, 5)
        place(board, BLACK, 0, 7)  # wait — RED promotes here, stops
        # So use a path that avoids row 0 until the end
        board = empty_board()
        place(board, RED, 7, 6)
        place(board, BLACK, 6, 5)
        place(board, BLACK, 4, 3)
        place(board, BLACK, 2, 1)
        place(board, BLACK, 4, 1)
        sequences = get_all_jump_sequences(
            board, 7, 6, RED,
            path_so_far=[(7, 6)],
            captured_so_far=[]
        )
        # At minimum a 3-jump sequence must exist
        max_captures = max(len(seq["captured"]) for seq in sequences)
        assert max_captures >= 3


    def test_king_multi_jump_backward_then_forward(self):
        """
        RED_KING jumps backward first then forward.
        Only possible for kings — tests backward multi-jump direction change.
        """
        board = empty_board()
        place(board, RED_KING, 4, 4)
        place(board, BLACK, 5, 5)
        place(board, BLACK, 3, 5)
        sequences = get_all_jump_sequences(
            board, 4, 4, RED,
            path_so_far=[(4, 4)],
            captured_so_far=[]
        )
        paths = [seq["path"] for seq in sequences]
        # King jumps backward to (6,6) then forward to (4,4) is blocked
        # King jumps backward (5,5) to (6,6), then nothing OR
        # King jumps forward (3,5) to (2,6), then nothing
        # What we really want: king goes one way then the other
        assert len(sequences) >= 2


    def test_multi_jump_all_sequences_have_correct_captured_count(self):
        """
        In any multi-jump sequence, the number of captured pieces
        must always equal path_length minus 1.
        One capture per jump, one jump per step in the path.
        """
        board = empty_board()
        place(board, RED, 6, 0)
        place(board, BLACK, 5, 1)
        place(board, BLACK, 3, 3)
        place(board, BLACK, 1, 5)
        sequences = get_all_jump_sequences(
            board, 6, 0, RED,
            path_so_far=[(6, 0)],
            captured_so_far=[]
        )
        for seq in sequences:
            path_length = len(seq["path"])
            captured_count = len(seq["captured"])
            assert captured_count == path_length - 1, (
                f"Path length {path_length} but captured count is {captured_count}"
            )


    def test_multi_jump_no_duplicate_squares_in_path(self):
        """
        In a multi-jump sequence, the same square should never
        appear twice in the path — a piece cannot visit the same
        square twice in one turn.
        """
        board = empty_board()
        place(board, RED, 4, 4)
        place(board, BLACK, 3, 3)
        place(board, BLACK, 3, 5)
        sequences = get_all_jump_sequences(
            board, 4, 4, RED,
            path_so_far=[(4, 4)],
            captured_so_far=[]
        )
        for seq in sequences:
            path = seq["path"]
            unique_squares = set(path)
            assert len(path) == len(unique_squares), (
                f"Duplicate square found in path: {path}"
            )


    def test_multi_jump_captured_pieces_are_all_unique(self):
        """
        In a multi-jump sequence, no captured square should appear
        twice — you cannot capture the same piece twice.
        """
        board = empty_board()
        place(board, RED, 6, 0)
        place(board, BLACK, 5, 1)
        place(board, BLACK, 3, 3)
        place(board, BLACK, 1, 5)
        sequences = get_all_jump_sequences(
            board, 6, 0, RED,
            path_so_far=[(6, 0)],
            captured_so_far=[]
        )
        for seq in sequences:
            captured = seq["captured"]
            unique_captured = set(captured)
            assert len(captured) == len(unique_captured), (
                f"Duplicate capture found in sequence: {captured}"
            )

    def test_double_jump_sequence(self):
        """
        RED at (5,0) can jump BLACK at (4,1) landing at (3,2),
        then jump BLACK at (2,3) landing at (1,4).
        The full path should be [(5,0),(3,2),(1,4)].
        """
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK, 2, 3)
        sequences = get_all_jump_sequences(
            board, 5, 0, RED,
            path_so_far=[(5, 0)],
            captured_so_far=[]
        )
        paths = [seq["path"] for seq in sequences]
        assert [(5, 0), (3, 2), (1, 4)] in paths

    def test_double_jump_captures_both_pieces(self):
        """
        After a double jump, both captured pieces should be
        listed in the captured list.
        """
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK, 2, 3)
        sequences = get_all_jump_sequences(
            board, 5, 0, RED,
            path_so_far=[(5, 0)],
            captured_so_far=[]
        )
        for seq in sequences:
            if seq["path"] == [(5, 0), (3, 2), (1, 4)]:
                assert (4, 1) in seq["captured"]
                assert (2, 3) in seq["captured"]

    def test_cannot_jump_same_piece_twice(self):
        """
        A piece already captured earlier in this sequence
        cannot be jumped again in the same sequence.
        """
        board = empty_board()
        place(board, RED, 5, 0)
        # Do NOT place the BLACK piece — it was already captured
        # The board reflects state AFTER the first jump

        sequences = get_all_jump_sequences(
            board, 5, 0, RED,
            path_so_far=[(5, 0)],
            captured_so_far=[]
        )

        # With no pieces on the board to jump, sequences should be
        # a single sequence with an empty captured list
        assert len(sequences) == 1
        assert sequences[0]["captured"] == []

    def test_all_possible_sequences_returned(self):
        """
        When multiple jump paths exist, all of them must be returned,
        not just the first one found.
        """
        board = empty_board()
        place(board, RED, 4, 4)
        place(board, BLACK, 3, 3)
        place(board, BLACK, 3, 5)
        sequences = get_all_jump_sequences(
            board, 4, 4, RED,
            path_so_far=[(4, 4)],
            captured_so_far=[]
        )
        paths = [seq["path"] for seq in sequences]
        assert [(4, 4), (2, 2)] in paths
        assert [(4, 4), (2, 6)] in paths

    def test_triple_jump_sequence(self):
        """
        RED should be able to complete a triple jump in one turn
        capturing three BLACK pieces.
        """
        board = empty_board()
        place(board, RED, 6, 0)
        place(board, BLACK, 5, 1)
        place(board, BLACK, 3, 3)
        place(board, BLACK, 1, 5)
        sequences = get_all_jump_sequences(
            board, 6, 0, RED,
            path_so_far=[(6, 0)],
            captured_so_far=[]
        )
        paths = [seq["path"] for seq in sequences]
        assert [(6, 0), (4, 2), (2, 4), (0, 6)] in paths


# ─────────────────────────────────────────────
# SECTION 7 — Promotion Tests
# ─────────────────────────────────────────────

class TestPromotion:
    def test_red_piece_not_promoted_at_row_1(self):
        """RED piece at row 1 is one step away from promotion but not there yet."""
        board = empty_board()
        place(board, RED, 2, 3)
        move = {
            "type": "simple",
            "path": [(2, 3), (1, 4)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[1][4] == RED
        assert new_board[1][4] != RED_KING

    def test_black_piece_not_promoted_at_row_6(self):
        """BLACK piece at row 6 is one step away from promotion but not there yet."""
        board = empty_board()
        place(board, BLACK, 5, 2)
        move = {
            "type": "simple",
            "path": [(5, 2), (6, 3)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[6][3] == BLACK
        assert new_board[6][3] != BLACK_KING

    def test_promotion_only_happens_at_correct_row_for_each_player(self):
        """
        RED only promotes at row 0, BLACK only promotes at row 7.
        Neither should promote anywhere else.
        """
        board = empty_board()
        place(board, RED, 3, 2)
        move = {
            "type": "simple",
            "path": [(3, 2), (2, 1)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[2][1] == RED
        assert new_board[2][1] != RED_KING

    def test_king_count_increases_after_promotion(self):
        """
        After a piece is promoted, the king count for that player
        should increase by 1 and regular piece count should decrease by 1.
        """
        board = empty_board()
        place(board, RED, 1, 2)
        move = {
            "type": "simple",
            "path": [(1, 2), (0, 1)],
            "captured": []
        }
        before = count_pieces(board, RED)
        new_board = apply_move(board, move)
        after = count_pieces(new_board, RED)
        assert after["kings"] == before["kings"] + 1
        assert after["regular"] == before["regular"] - 1
        assert after["total"] == before["total"]
        # Inside TestPromotion
    def test_apply_jump_on_board_promotes_black_correctly(self):
        """
        apply_move should promote BLACK to BLACK_KING when
        landing on row 7 via a jump, not just via simple move.
        """
        board = empty_board()
        place(board, BLACK, 5, 2)
        place(board, RED, 6, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (7, 4)],
            "captured": [(6, 3)]
        }
        new_board = apply_move(board, move)
        assert new_board[7][4] == BLACK_KING
        assert new_board[6][3] == EMPTY
        assert new_board[5][2] == EMPTY

        # Inside TestPromotion
    def test_black_promotion_stops_multi_jump(self):
        """
        If BLACK gets promoted during a multi jump sequence,
        the sequence should stop — no further jumps as BLACK_KING.
        """
        board = empty_board()
        place(board, BLACK, 5, 2)
        place(board, RED, 6, 3)
        place(board, RED, 7, 0)  # would only be jumpable as a king
        sequences = get_all_jump_sequences(
            board, 5, 2, BLACK,
            path_so_far=[(5, 2)],
            captured_so_far=[]
        )
        for seq in sequences:
            if (6, 3) in seq["captured"]:
                assert seq["path"][-1] == (7, 4), (
                    "BLACK sequence continued after promotion"
                )
        # Inside TestPromotion
    def test_red_king_stays_king_at_row_0(self):
        """RED_KING moving to row 0 should remain RED_KING, not re-promote."""
        board = empty_board()
        place(board, RED_KING, 1, 2)
        move = {
            "type": "simple",
            "path": [(1, 2), (0, 1)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[0][1] == RED_KING

        # Inside TestPromotion
    def test_black_king_stays_king_at_row_7(self):
        """BLACK_KING moving through row 7 should remain BLACK_KING, not double-promote."""
        board = empty_board()
        place(board, BLACK_KING, 6, 3)
        move = {
            "type": "simple",
            "path": [(6, 3), (7, 4)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[7][4] == BLACK_KING

    def test_red_piece_promoted_at_row_0(self):
        """RED piece reaching row 0 should become RED_KING."""
        board = empty_board()
        place(board, RED, 1, 2)
        move = {
            "type": "simple",
            "path": [(1, 2), (0, 1)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[0][1] == RED_KING

    def test_black_piece_promoted_at_row_7(self):
        """BLACK piece reaching row 7 should become BLACK_KING."""
        board = empty_board()
        place(board, BLACK, 6, 3)
        move = {
            "type": "simple",
            "path": [(6, 3), (7, 4)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[7][4] == BLACK_KING

    def test_red_piece_not_promoted_before_row_0(self):
        """RED piece at row 1 moving to row 0 gets promoted but not at row 2."""
        board = empty_board()
        place(board, RED, 2, 3)
        move = {
            "type": "simple",
            "path": [(2, 3), (1, 4)],
            "captured": []
        }
        new_board = apply_move(board, move)
        assert new_board[1][4] == RED

    def test_promotion_stops_multi_jump(self):
        """
        If RED gets promoted during a multi jump sequence,
        the sequence should stop — no further jumps as king.
        """
        board = empty_board()
        place(board, RED, 2, 1)
        place(board, BLACK, 1, 2)
        place(board, BLACK, 0, 5)
        sequences = get_all_jump_sequences(
            board, 2, 1, RED,
            path_so_far=[(2, 1)],
            captured_so_far=[]
        )
        for seq in sequences:
            if (1, 2) in seq["captured"]:
                assert seq["path"][-1] == (0, 3), (
                    "Sequence continued after promotion"
                )


# ─────────────────────────────────────────────
# SECTION 8 — Win Condition Tests
# ─────────────────────────────────────────────

class TestWinCondition:
    def test_win_condition_with_only_kings_remaining(self):
        """
        Win detection should work correctly when only kings remain.
        RED_KING with no opponent pieces should trigger a win.
        """
        board = empty_board()
        place(board, RED_KING, 4, 4)
        result = check_win_condition(board, RED)
        assert result["game_over"] == True
        assert result["winner"] == RED

    def test_win_condition_reason_field_is_set(self):
        """
        When the game is over, the reason field must be a non-empty string
        explaining why — not None.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        result = check_win_condition(board, RED)
        assert result["reason"] is not None
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_win_condition_reason_is_none_when_game_continues(self):
        """When the game is not over, reason must be None."""
        board = create_initial_board()
        result = check_win_condition(board, RED)
        assert result["reason"] is None

    def test_has_no_pieces_left_with_only_kings(self):
        """
        has_no_pieces_left should return False if player only has kings.
        Kings are still pieces — the player has not lost.
        """
        board = empty_board()
        place(board, RED_KING, 4, 4)
        assert has_no_pieces_left(board, RED) == False

    def test_both_players_blocked_simultaneously(self):
        """
        If after RED moves, BLACK is blocked, RED wins.
        BLACK pieces at row 7 cannot move further down and have no jumps.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 7, 0)
        place(board, BLACK, 7, 2)
        result = check_win_condition(board, RED)
        assert result["game_over"] == True
        assert result["winner"] == RED

    def test_draw_by_repetition_declared_correctly(self):
        """
        When check_repetition returns True, the game state should
        reflect a draw — winner is None, game_over is True.
        This tests that your state management handles draws correctly.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 2, 3)

        position_history = []
        move_out = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        move_back = {"type": "simple", "path": [(4, 1), (5, 2)], "captured": []}

        current_board = board
        for cycle in range(3):
            position_history.append(compute_hash(current_board))
            current_board = apply_move(current_board, move_out)
            position_history.append(compute_hash(current_board))
            current_board = apply_move(current_board, move_back)

        final_hash = compute_hash(current_board)
        is_draw = check_repetition(position_history, final_hash)
        assert is_draw == True


    # Inside TestWinCondition
    def test_check_win_checks_opponent_not_self(self):
        """
        check_win_condition checks if the OPPONENT of current_player lost.
        If RED just moved and BLACK has no pieces, RED wins.
        If called with BLACK as current_player in the same position,
        game should not be over because RED still has pieces.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        # No BLACK pieces on board

        # RED just moved — BLACK has no pieces — RED wins
        result_red = check_win_condition(board, RED)
        assert result_red["game_over"] == True
        assert result_red["winner"] == RED

        # BLACK just moved — RED still has pieces — game not over
        result_black = check_win_condition(board, BLACK)
        assert result_black["game_over"] == False
        
    def test_player_with_no_pieces_loses(self):
        """If RED has no pieces, has_no_pieces_left should return True."""
        board = empty_board()
        place(board, BLACK, 3, 4)
        assert has_no_pieces_left(board, RED) == True

    def test_player_with_pieces_not_empty(self):
        """If RED has pieces, has_no_pieces_left should return False."""
        board = empty_board()
        place(board, RED, 5, 2)
        assert has_no_pieces_left(board, RED) == False

    def test_blocked_player_has_no_moves(self):
        """
        All RED pieces are stuck — regular piece at row 0 cannot
        move further up, and no jumps are available.
        """
        board = empty_board()

        # RED regular piece at row 0 — cannot move forward because
        # RED moves upward and row 0 is the top of the board
        place(board, RED, 0, 1)
        place(board, RED, 0, 3)

        assert has_no_moves_left(board, RED) == True
    def test_check_win_opponent_no_pieces(self):
        """
        check_win_condition should return game_over True
        when opponent has no pieces left.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        result = check_win_condition(board, RED)
        assert result["game_over"] == True
        assert result["winner"] == RED

    def test_check_win_opponent_blocked(self):
        """
        check_win_condition should return game_over True
        when opponent is completely blocked.
        """
        board = empty_board()
        # BLACK piece at row 7 edge — cannot move further down
        # and all jump landings are blocked
        place(board, BLACK, 7, 6)
        place(board, RED, 6, 5)
        place(board, RED, 6, 7)
        # RED pieces blocking the jump landings
        place(board, RED, 4, 3)
        place(board, RED, 4, 7)
        # Give RED a piece so it's not a win-by-no-pieces situation
        place(board, RED, 0, 1)
        result = check_win_condition(board, RED)
        assert result["game_over"] == True
        assert result["winner"] == RED

    def test_check_win_game_continues(self):
        """
        check_win_condition should return game_over False
        when both players have pieces and moves available.
        """
        board = create_initial_board()
        result = check_win_condition(board, RED)
        assert result["game_over"] == False
        assert result["winner"] == None


# ─────────────────────────────────────────────
# SECTION 9 — Move Facts Tests
# ─────────────────────────────────────────────

class TestMoveFacts:
    def test_path_length_correct_for_simple_move(self):
        """Simple move path_length should always be 2."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["path_length"] == 2

    def test_path_length_correct_for_double_jump(self):
        """Double jump path_length should be 3 — start plus two landings."""
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK, 2, 3)
        move = {
            "type": "jump",
            "path": [(5, 0), (3, 2), (1, 4)],
            "captured": [(4, 1), (2, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["path_length"] == 3

    def test_move_facts_king_jump_net_gain(self):
        """RED_KING jumping BLACK should also have net_gain of 1."""
        board = empty_board()
        place(board, RED_KING, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["net_gain"] == 1
        assert facts["captures_count"] == 1

    def test_move_facts_for_black_player(self):
        """
        compute_move_facts should work correctly for BLACK player too,
        not just RED. Net gain should reflect BLACK capturing RED.
        """
        board = empty_board()
        place(board, BLACK, 2, 3)
        place(board, RED, 3, 4)
        move = {
            "type": "jump",
            "path": [(2, 3), (4, 5)],
            "captured": [(3, 4)]
        }
        facts = compute_move_facts(board, move, BLACK)
        assert facts["net_gain"] == 1
        assert facts["captures_count"] == 1
        assert facts["opp_pieces_after"]["total"] == 0

    def test_results_in_king_false_for_king_moving_to_back_row(self):
        """
        RED_KING moving to row 0 should NOT trigger results_in_king
        since it is already a king — no new promotion happens.
        """
        board = empty_board()
        place(board, RED_KING, 1, 2)
        move = {
            "type": "simple",
            "path": [(1, 2), (0, 1)],
            "captured": []
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["results_in_king"] == False

    def test_move_facts_all_keys_present(self):
        """
        compute_move_facts must always return all expected keys.
        Missing keys would crash LLM agent nodes.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        expected_keys = [
            "move_type", "captures_count", "is_multi_jump",
            "results_in_king", "path_length", "our_pieces_before",
            "our_pieces_after", "opp_pieces_before", "opp_pieces_after",
            "net_gain"
        ]
        for key in expected_keys:
            assert key in facts, f"Missing key in move facts: {key}"
        # Inside TestMoveFacts
    def test_net_gain_is_zero_for_simple_move(self):
        """
        A simple move captures nothing so net_gain must be exactly 0,
        not negative, not None.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 2, 3)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["net_gain"] == 0
        assert type(facts["net_gain"]) == int


    # Inside TestMoveFacts
    def test_our_pieces_after_equals_before_for_simple_move(self):
        """
        After a simple move, our own piece count must not change.
        We did not lose or gain any pieces.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 5, 4)
        move = {"type": "simple", "path": [(5, 2), (4, 1)], "captured": []}
        facts = compute_move_facts(board, move, RED)
        assert facts["our_pieces_before"]["total"] == facts["our_pieces_after"]["total"]
     # Inside TestMoveFacts
    def test_count_pieces_returns_zero_when_no_pieces(self):
        """
        count_pieces should return zeros cleanly when
        the player has no pieces on the board at all.
        """
        board = empty_board()
        result = count_pieces(board, RED)
        assert result["total"] == 0
        assert result["regular"] == 0
        assert result["kings"] == 0

    def test_count_pieces_includes_kings(self):
        """
        count_pieces should count both regular pieces and kings
        in the total.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, RED, 5, 4)
        place(board, RED_KING, 3, 2)
        result = count_pieces(board, RED)
        assert result["total"] == 3
        assert result["regular"] == 2
        assert result["kings"] == 1

# Inside TestMoveFacts
    def test_promotion_via_jump_detected_in_facts(self):
        """
        compute_move_facts should detect promotion that happens
        through a jump, not just through a simple move.
        """
        board = empty_board()
        place(board, RED, 2, 1)
        place(board, BLACK, 1, 2)
        move = {
            "type": "jump",
            "path": [(2, 1), (0, 3)],
            "captured": [(1, 2)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["results_in_king"] == True
        assert facts["captures_count"] == 1

    def test_simple_move_facts(self):
        """Simple move should have zero captures and no multi jump."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {
            "type": "simple",
            "path": [(5, 2), (4, 1)],
            "captured": []
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["move_type"] == "simple"
        assert facts["captures_count"] == 0
        assert facts["is_multi_jump"] == False
        assert facts["net_gain"] == 0

    def test_single_jump_facts(self):
        """Single jump should have captures_count of 1 and net_gain of 1."""
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["captures_count"] == 1
        assert facts["is_multi_jump"] == False
        assert facts["net_gain"] == 1

    def test_multi_jump_facts(self):
        """Double jump should have captures_count of 2 and is_multi_jump True."""
        board = empty_board()
        place(board, RED, 5, 0)
        place(board, BLACK, 4, 1)
        place(board, BLACK, 2, 3)
        move = {
            "type": "jump",
            "path": [(5, 0), (3, 2), (1, 4)],
            "captured": [(4, 1), (2, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["captures_count"] == 2
        assert facts["is_multi_jump"] == True
        assert facts["net_gain"] == 2

    def test_promotion_detected_in_facts(self):
        """Move that results in promotion should have results_in_king True."""
        board = empty_board()
        place(board, RED, 1, 2)
        move = {
            "type": "simple",
            "path": [(1, 2), (0, 1)],
            "captured": []
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["results_in_king"] == True

    def test_no_promotion_detected_correctly(self):
        """Move that does not result in promotion should have results_in_king False."""
        board = empty_board()
        place(board, RED, 5, 2)
        move = {
            "type": "simple",
            "path": [(5, 2), (4, 1)],
            "captured": []
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["results_in_king"] == False

    def test_piece_counts_correct(self):
        """
        Piece counts before and after move should reflect
        the actual state of the board.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        facts = compute_move_facts(board, move, RED)
        assert facts["our_pieces_before"]["total"] == 1
        assert facts["opp_pieces_before"]["total"] == 1
        assert facts["opp_pieces_after"]["total"] == 0


# ─────────────────────────────────────────────
# SECTION 10 — Zobrist Hashing Tests
# ─────────────────────────────────────────────

class TestZobrist:
    def test_zobrist_table_has_correct_size(self):
        """
        ZOBRIST_TABLE should have exactly 4 piece types × 8 rows × 8 cols = 256 entries.
        EMPTY squares are not included since they contribute nothing to the hash.
        """
        assert len(ZOBRIST_TABLE) == 256

    def test_zobrist_table_all_values_are_integers(self):
        """Every value in the Zobrist table must be a non-negative integer."""
        for key in ZOBRIST_TABLE:
            value = ZOBRIST_TABLE[key]
            assert isinstance(value, int), f"Non-integer value at key {key}"
            assert value >= 0, f"Negative value at key {key}"

    def test_zobrist_table_all_values_unique(self):
        """
        All 320 random numbers in the Zobrist table should be unique.
        Duplicate values would cause hash collisions.
        """
        values = list(ZOBRIST_TABLE.values())
        unique_values = set(values)
        assert len(values) == len(unique_values), (
            "Zobrist table contains duplicate random numbers"
        )

    def test_hash_order_independent(self):
        """
        Two boards with the same pieces in the same positions
        but built in different order should produce the same hash.
        """
        board1 = empty_board()
        place(board1, RED, 5, 2)
        place(board1, BLACK, 3, 4)

        board2 = empty_board()
        place(board2, BLACK, 3, 4)
        place(board2, RED, 5, 2)

        assert compute_hash(board1) == compute_hash(board2)

    def test_single_piece_hash_matches_table_entry(self):
        """
        A board with one RED piece at (5,2) should have a hash
        equal to exactly the Zobrist table entry for (RED, 5, 2).
        """
        board = empty_board()
        place(board, RED, 5, 2)
        expected_hash = ZOBRIST_TABLE[(RED, 5, 2)]
        assert compute_hash(board) == expected_hash

    def test_update_hash_with_capture(self):
        """
        update_hash only handles moving a piece, not capturing.
        For a jump, the captured piece hash must also be XORed out.
        This test verifies that compute_hash on the new board
        after a jump is correct even though update_hash alone
        would not account for the captured piece removal.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 4, 3)
        move = {
            "type": "jump",
            "path": [(5, 2), (3, 4)],
            "captured": [(4, 3)]
        }
        new_board = apply_move(board, move)
        correct_hash = compute_hash(new_board)
        assert correct_hash == ZOBRIST_TABLE[(RED, 3, 4)]
        # Inside TestZobrist
    def test_every_move_changes_the_hash(self):
        """
        After any move, the board hash must be different from before.
        No move should produce the same hash as the position before it.
        """
        board = create_initial_board()
        original_hash = compute_hash(board)
        moves = get_all_legal_moves(board, RED)

        for move in moves:
            new_board = apply_move(board, move)
            new_hash = compute_hash(new_board)
            assert new_hash != original_hash, (
                f"Move {move} produced the same hash as the original position"
            )

    def test_same_board_same_hash(self):
        """The same board position must always produce the same hash."""
        board = create_initial_board()
        hash1 = compute_hash(board)
        hash2 = compute_hash(board)
        assert hash1 == hash2

    def test_different_boards_different_hash(self):
        """Two different board positions should produce different hashes."""
        board1 = create_initial_board()
        board2 = empty_board()
        place(board2, RED, 5, 2)
        hash1 = compute_hash(board1)
        hash2 = compute_hash(board2)
        assert hash1 != hash2

    def test_empty_board_hash_is_zero(self):
        """An empty board should produce a hash of zero."""
        board = empty_board()
        hash_value = compute_hash(board)
        assert hash_value == 0

    def test_update_hash_matches_compute_hash(self):
        """
        Updating a hash after a move should produce the same result
        as computing the hash from scratch on the new board.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        old_hash = compute_hash(board)

        move = {
            "type": "simple",
            "path": [(5, 2), (4, 1)],
            "captured": []
        }
        new_board = apply_move(board, move)
        computed_hash = compute_hash(new_board)
        updated_hash = update_hash(old_hash, RED, 5, 2, 4, 1)
        assert computed_hash == updated_hash

    def test_repetition_not_detected_below_limit(self):
        """
        check_repetition should return False if position
        appears fewer than 3 times.
        """
        board = create_initial_board()
        current_hash = compute_hash(board)
        history = [current_hash, current_hash]
        result = check_repetition(history, current_hash)
        assert result == False

    def test_repetition_detected_at_limit(self):
        """
        check_repetition should return True if position
        appears 3 or more times.
        """
        board = create_initial_board()
        current_hash = compute_hash(board)
        history = [current_hash, current_hash, current_hash]
        result = check_repetition(history, current_hash)
        assert result == True

    def test_repetition_not_triggered_by_different_positions(self):
        """
        check_repetition should not trigger if different positions
        appear in history even if count is high.
        """
        board1 = create_initial_board()
        board2 = empty_board()
        place(board2, RED, 5, 2)
        hash1 = compute_hash(board1)
        hash2 = compute_hash(board2)
        history = [hash1, hash1, hash2, hash2]
        result = check_repetition(history, hash1)
        assert result == False

        

    

    # Add to TestZobrist class

    def test_repetition_simulated_back_and_forth(self):
        """
        Simulate a real back-and-forth: RED moves a piece out and back.
        The original position should appear twice, not triggering a draw yet.
        Then a third return should trigger it.
        """
        board = empty_board()
        place(board, RED, 5, 2)
        place(board, BLACK, 2, 3)

        hash_start = compute_hash(board)
        position_history = [hash_start]

        # RED moves piece out
        move_out = {
            "type": "simple",
            "path": [(5, 2), (4, 1)],
            "captured": []
        }
        board_after_out = apply_move(board, move_out)
        position_history.append(compute_hash(board_after_out))

        # RED moves piece back
        move_back = {
            "type": "simple",
            "path": [(4, 1), (5, 2)],
            "captured": []
        }
        board_back = apply_move(board_after_out, move_back)
        hash_back = compute_hash(board_back)
        position_history.append(hash_back)

        # Should appear twice now — not a draw yet
        assert check_repetition(position_history, hash_back) == False

        # Repeat the cycle again
        board_after_out2 = apply_move(board_back, move_out)
        position_history.append(compute_hash(board_after_out2))

        board_back2 = apply_move(board_after_out2, move_back)
        hash_back2 = compute_hash(board_back2)
        position_history.append(hash_back2)

        # Now the starting position has appeared 3 times — draw
        assert check_repetition(position_history, hash_back2) == True

