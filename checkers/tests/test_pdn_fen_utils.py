# checkers/tests/test_pdn_fen_utils.py
"""
Tests for PDN FEN utilities — verifying:
  1. PDN square number → board (row, col) conversion
  2. PDN FEN piece-colour mapping (B→BLACK, W→RED)
  3. Standard starting PDN FEN matches engine create_initial_board()
  4. Side-to-move B/W maps to correct engine player
"""

import pytest
from checkers.data.pdn_importer.fen_utils import (
    square_to_rowcol, rowcol_to_square, parse_fen, side_to_str
)
from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING,
    create_initial_board
)


# ---------------------------------------------------------------------------
# 1. Square number → (row, col) conversion
# ---------------------------------------------------------------------------

class TestSquareToRowcol:
    # Row 0 (even): dark cols are 1, 3, 5, 7
    def test_square_1(self):
        assert square_to_rowcol(1) == (0, 1)

    def test_square_2(self):
        assert square_to_rowcol(2) == (0, 3)

    def test_square_3(self):
        assert square_to_rowcol(3) == (0, 5)

    def test_square_4(self):
        assert square_to_rowcol(4) == (0, 7)

    # Row 1 (odd): dark cols are 0, 2, 4, 6
    def test_square_5(self):
        assert square_to_rowcol(5) == (1, 0)

    def test_square_8(self):
        assert square_to_rowcol(8) == (1, 6)

    # Row 2 (even): dark cols are 1, 3, 5, 7
    def test_square_9(self):
        assert square_to_rowcol(9) == (2, 1)

    def test_square_12(self):
        assert square_to_rowcol(12) == (2, 7)

    # Row 5 (odd): dark cols are 0, 2, 4, 6  → WHITE/RED territory
    def test_square_21(self):
        assert square_to_rowcol(21) == (5, 0)

    def test_square_24(self):
        assert square_to_rowcol(24) == (5, 6)

    # Row 7 (odd): dark cols are 0, 2, 4, 6
    def test_square_29(self):
        assert square_to_rowcol(29) == (7, 0)

    def test_square_32(self):
        assert square_to_rowcol(32) == (7, 6)

    def test_out_of_range_low(self):
        with pytest.raises(ValueError):
            square_to_rowcol(0)

    def test_out_of_range_high(self):
        with pytest.raises(ValueError):
            square_to_rowcol(33)


class TestRowcolToSquare:
    def test_roundtrip_all_squares(self):
        """Verify square → (row,col) → square is identity for all 32 squares."""
        for sq in range(1, 33):
            row, col = square_to_rowcol(sq)
            assert rowcol_to_square(row, col) == sq, \
                f"Roundtrip failed for square {sq}: ({row},{col})"

    def test_light_square_raises(self):
        # (0, 0) is a light square
        with pytest.raises(ValueError):
            rowcol_to_square(0, 0)


# ---------------------------------------------------------------------------
# 2. FEN piece-colour mapping: B→BLACK, W→RED
# ---------------------------------------------------------------------------

class TestFENColourMapping:
    def test_white_pieces_map_to_red(self):
        """PDN FEN 'W' pieces must land as engine RED (1)."""
        # W21 = white piece on square 21 = row5, col0 → should be RED
        board, side = parse_fen("B:W21:B.")
        row, col = square_to_rowcol(21)
        assert board[row][col] == RED, \
            f"Expected RED at sq21 ({row},{col}), got {board[row][col]}"

    def test_black_pieces_map_to_black(self):
        """PDN FEN 'B' pieces must land as engine BLACK (2)."""
        # B9 = black piece on square 9 = row2, col1 → should be BLACK
        board, side = parse_fen("B:W.:B9")
        row, col = square_to_rowcol(9)
        assert board[row][col] == BLACK, \
            f"Expected BLACK at sq9 ({row},{col}), got {board[row][col]}"

    def test_white_king_maps_to_red_king(self):
        """PDN FEN 'W' + 'K' prefix → RED_KING (3)."""
        board, side = parse_fen("W:WK14:B.")
        row, col = square_to_rowcol(14)
        assert board[row][col] == RED_KING

    def test_black_king_maps_to_black_king(self):
        """PDN FEN 'B' + 'K' prefix → BLACK_KING (4)."""
        board, side = parse_fen("B:W.:BK17")
        row, col = square_to_rowcol(17)
        assert board[row][col] == BLACK_KING

    def test_multiple_pieces(self):
        """gem.pdn #1: B:W21,K5:BK17,K14"""
        board, side = parse_fen("B:W21,K5:BK17,K14")
        assert board[square_to_rowcol(21)[0]][square_to_rowcol(21)[1]] == RED
        assert board[square_to_rowcol(5)[0]][square_to_rowcol(5)[1]]   == RED_KING
        assert board[square_to_rowcol(17)[0]][square_to_rowcol(17)[1]] == BLACK_KING
        assert board[square_to_rowcol(14)[0]][square_to_rowcol(14)[1]] == BLACK_KING

    def test_empty_squares_stay_empty(self):
        """All squares not listed in the FEN must be EMPTY."""
        board, side = parse_fen("B:W21:B9")
        listed_squares = {21, 9}
        for sq in range(1, 33):
            row, col = square_to_rowcol(sq)
            if sq in listed_squares:
                continue
            assert board[row][col] == EMPTY, \
                f"Square {sq} should be EMPTY but got {board[row][col]}"


# ---------------------------------------------------------------------------
# 3. Standard starting position FEN matches create_initial_board()
# ---------------------------------------------------------------------------

class TestStartingPositionFEN:
    """
    The standard checkers starting position in PDN FEN:
      Black (top, squares 1-12)  → engine BLACK, rows 0-2
      White (bottom, squares 21-32) → engine RED, rows 5-7
    """
    STANDARD_FEN = (
        "B:"
        "W21,22,23,24,25,26,27,28,29,30,31,32:"
        "B1,2,3,4,5,6,7,8,9,10,11,12"
    )

    def test_matches_engine_initial_board(self):
        parsed_board, side = parse_fen(self.STANDARD_FEN)
        expected = create_initial_board()
        assert parsed_board == expected, (
            "Parsed starting FEN does not match create_initial_board().\n"
            f"Parsed:   {parsed_board}\n"
            f"Expected: {expected}"
        )

    def test_side_to_move_is_black(self):
        _, side = parse_fen(self.STANDARD_FEN)
        assert side == BLACK, f"Starting position: side should be BLACK, got {side}"


# ---------------------------------------------------------------------------
# 4. Side-to-move B / W mapping
# ---------------------------------------------------------------------------

class TestSideToMove:
    def test_B_maps_to_BLACK(self):
        _, side = parse_fen("B:W21:B9")
        assert side == BLACK, f"PDN 'B' should map to engine BLACK (2), got {side}"

    def test_W_maps_to_RED(self):
        _, side = parse_fen("W:W21:B9")
        assert side == RED, f"PDN 'W' should map to engine RED (1), got {side}"

    def test_side_to_str_BLACK(self):
        assert side_to_str(BLACK) == "BLACK"

    def test_side_to_str_RED(self):
        assert side_to_str(RED) == "RED"

    def test_lowercase_b_also_works(self):
        """Parser should handle lowercase side char gracefully."""
        _, side = parse_fen("b:W21:B9")
        assert side == BLACK


# ---------------------------------------------------------------------------
# 5. Sanity: legal moves exist for standard starting position
# ---------------------------------------------------------------------------

class TestLegalMovesStartingPosition:
    def test_black_has_legal_moves_at_start(self):
        """At the start, BLACK (top, moves down) must have exactly 7 simple moves."""
        from checkers.engine.rules import get_all_legal_moves
        board = create_initial_board()
        moves = get_all_legal_moves(board, BLACK)
        # All moves at the start are simple (no captures)
        assert all(m["type"] == "simple" for m in moves)
        assert len(moves) == 7, f"Expected 7 opening moves for BLACK, got {len(moves)}"

    def test_red_has_legal_moves_at_start(self):
        """At the start, RED (bottom, moves up) must also have exactly 7 simple moves."""
        from checkers.engine.rules import get_all_legal_moves
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)
        assert all(m["type"] == "simple" for m in moves)
        assert len(moves) == 7, f"Expected 7 opening moves for RED, got {len(moves)}"

    def test_black_moves_in_correct_direction(self):
        """BLACK pieces should only move to higher row numbers (downward)."""
        from checkers.engine.rules import get_all_legal_moves
        board = create_initial_board()
        moves = get_all_legal_moves(board, BLACK)
        for m in moves:
            from_row = m["path"][0][0]
            to_row   = m["path"][1][0]
            assert to_row > from_row, \
                f"BLACK piece moved upward: {m['path']}"

    def test_red_moves_in_correct_direction(self):
        """RED pieces should only move to lower row numbers (upward)."""
        from checkers.engine.rules import get_all_legal_moves
        board = create_initial_board()
        moves = get_all_legal_moves(board, RED)
        for m in moves:
            from_row = m["path"][0][0]
            to_row   = m["path"][1][0]
            assert to_row < from_row, \
                f"RED piece moved downward: {m['path']}"
