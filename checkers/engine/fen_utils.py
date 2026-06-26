# checkers/engine/fen_utils.py
"""
Utilities to convert between PDN square numbers (1-32) and the engine's
8×8 board representation.

=== VERIFIED COLOUR MAPPING ===

Standard English draughts / PDN convention:
  - Black pieces start at squares 1-12 (rows 0-2, TOP of board)
  - White pieces start at squares 21-32 (rows 5-7, BOTTOM of board)
  - "B" in FEN side-to-move = Black moves next

Engine convention (board.py):
  - BLACK (2) starts at rows 0-2 (top), moves DOWN (row increases)
  - RED   (1) starts at rows 5-7 (bottom), moves UP (row decreases)

Therefore the verified mapping is:
  PDN 'B' (Black to move)    → engine BLACK (2)
  PDN 'W' (White to move)    → engine RED   (1)
  PDN FEN "B<squares>" pieces → engine BLACK (piece code 2 / BLACK_KING 4)
  PDN FEN "W<squares>" pieces → engine RED   (piece code 1 / RED_KING   3)

PDN square → (row, col) layout:
  Row 0 (top): squares  1- 4  → cols 1,3,5,7   (even row: dark at odd cols)
  Row 1:       squares  5- 8  → cols 0,2,4,6   (odd  row: dark at even cols)
  Row 2:       squares  9-12  → cols 1,3,5,7
  Row 3:       squares 13-16  → cols 0,2,4,6
  Row 4:       squares 17-20  → cols 1,3,5,7
  Row 5:       squares 21-24  → cols 0,2,4,6
  Row 6:       squares 25-28  → cols 1,3,5,7
  Row 7 (bot): squares 29-32  → cols 0,2,4,6
"""

from checkers.engine.board import EMPTY, RED, BLACK, RED_KING, BLACK_KING


# ---------------------------------------------------------------------------
# Square number ↔ (row, col) conversion
# ---------------------------------------------------------------------------

def square_to_rowcol(sq: int) -> tuple:
    """
    Convert a PDN square number (1-32) to (row, col) in the engine's 8×8 board.

    Examples:
        square_to_rowcol(1)  → (0, 1)   # top-left dark square
        square_to_rowcol(4)  → (0, 7)   # top-right dark square
        square_to_rowcol(5)  → (1, 0)
        square_to_rowcol(32) → (7, 6)   # bottom-right dark square
    """
    if sq < 1 or sq > 32:
        raise ValueError(f"PDN square out of range: {sq}")
    idx = sq - 1                # 0-based
    row = idx // 4
    pos_in_row = idx % 4        # 0,1,2,3 position within the 4 dark squares of this row
    if row % 2 == 0:
        col = pos_in_row * 2 + 1   # even rows: cols 1,3,5,7
    else:
        col = pos_in_row * 2       # odd  rows: cols 0,2,4,6
    return (row, col)


def rowcol_to_square(row: int, col: int) -> int:
    """
    Convert engine (row, col) to PDN square number (1-32).
    Only valid for dark squares where (row+col)%2 == 1.
    """
    if (row + col) % 2 == 0:
        raise ValueError(f"({row},{col}) is a light square — not a valid checkers square")
    if row % 2 == 0:
        pos_in_row = (col - 1) // 2
    else:
        pos_in_row = col // 2
    return row * 4 + pos_in_row + 1


# ---------------------------------------------------------------------------
# FEN string parser
# ---------------------------------------------------------------------------

def parse_fen(fen: str):
    """
    Parse a PDN FEN string into an engine board and side-to-move.

    FEN format examples:
        "B:W21,K5:BK17,K14."
        "W:WK32,K31,30:BK28,K23,K22,21."

    Returns:
        board (list[list[int]]): 8×8 engine board
        side  (int):             BLACK (2) if 'B', RED (1) if 'W'
    """
    fen = fen.strip().rstrip('.')
    board = [[EMPTY] * 8 for _ in range(8)]

    parts = fen.split(':')
    if len(parts) < 3:
        raise ValueError(f"Invalid FEN (need ≥3 colon-separated parts): {fen!r}")

    side_char = parts[0].strip().upper()
    # PDN 'B' → engine BLACK (2)  |  PDN 'W' → engine RED (1)
    side = BLACK if side_char == 'B' else RED

    # Parse each colour section
    for section in parts[1:]:
        section = section.strip()
        if not section:
            continue
        colour_char = section[0].upper()   # 'W' or 'B'
        piece_list_str = section[1:]       # remainder after the colour letter

        # PDN 'W' pieces → engine RED (bottom)
        # PDN 'B' pieces → engine BLACK (top)
        if colour_char == 'W':
            base_piece = RED
            king_piece = RED_KING
        else:
            base_piece = BLACK
            king_piece = BLACK_KING

        tokens = [t.strip() for t in piece_list_str.split(',') if t.strip()]
        for token in tokens:
            is_king = token.startswith('K')
            sq_str = token[1:] if is_king else token
            if not sq_str.isdigit():
                continue
            sq = int(sq_str)
            row, col = square_to_rowcol(sq)
            board[row][col] = king_piece if is_king else base_piece

    return board, side


# ---------------------------------------------------------------------------
# Board ↔ PDN move helpers
# ---------------------------------------------------------------------------

def pdn_move_to_engine(move_str: str):
    """
    Convert a PDN move string to an engine move dict skeleton.

    Supports:
        "11-15"           → simple move
        "14x23"           → single jump
        "9x14x23"         → multi-jump (parsed as a sequence of captured midpoints)

    Returns:
        {"type": "simple"|"jump", "path": [(r,c),...], "captured": [(r,c),...]}
        or None if the string is not parseable.

    NOTE: captured squares are inferred as midpoints; only valid for single-step
    jumps (2-square diagonal leaps).  The caller should cross-validate against
    get_all_legal_moves when needed.
    """
    move_str = move_str.strip().replace('X', 'x')

    if 'x' in move_str:
        sq_strings = move_str.split('x')
        try:
            squares = [int(s.strip()) for s in sq_strings if s.strip()]
        except ValueError:
            return None
        if len(squares) < 2:
            return None
        path = [square_to_rowcol(sq) for sq in squares]
        captured = []
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            cap_r = (r1 + r2) // 2
            cap_c = (c1 + c2) // 2
            captured.append((cap_r, cap_c))
        return {"type": "jump", "path": path, "captured": captured}

    elif '-' in move_str:
        parts = move_str.split('-')
        if len(parts) != 2:
            return None
        try:
            from_sq = int(parts[0].strip())
            to_sq   = int(parts[1].strip())
        except ValueError:
            return None
        return {
            "type": "simple",
            "path": [square_to_rowcol(from_sq), square_to_rowcol(to_sq)],
            "captured": []
        }
    return None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def board_to_serializable(board):
    """Return a JSON-serialisable copy of the engine board (list of lists of ints)."""
    return [row[:] for row in board]


def side_to_str(side: int) -> str:
    return "BLACK" if side == BLACK else "RED"


def str_to_side(s: str) -> int:
    return BLACK if s.strip().upper() == "BLACK" else RED
