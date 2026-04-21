# engine/board.py

# --- Piece constants ---
EMPTY = 0
RED = 1          # You (player), starts at rows 5-7, moves UP (row decreases)
BLACK = 2        # Opponent, starts at rows 0-2, moves DOWN (row increases)
RED_KING = 3     # Red piece that reached row 0
BLACK_KING = 4   # Black piece that reached row 7

# --- Board dimensions ---
BOARD_SIZE = 8


def create_initial_board():
    """
    Returns 8x8 board as a list of lists.
    0 = empty, 1 = RED, 2 = BLACK, 3 = RED_KING, 4 = BLACK_KING
    
    Black occupies rows 0-2 on dark squares.
    Red occupies rows 5-7 on dark squares.
    Dark squares: (row + col) % 2 == 1
    """
    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:          # dark square only
                if row < 3:
                    board[row][col] = BLACK
                elif row > 4:
                    board[row][col] = RED

    return board

def in_bounds(row, col):
    """Check if (row, col) is inside the 8x8 board."""
    row_is_valid = row >= 0 and row < BOARD_SIZE
    col_is_valid = col >= 0 and col < BOARD_SIZE
    return row_is_valid and col_is_valid


def is_dark_square(row, col):
    """Only dark squares are playable in checkers."""
    total = row + col
    return total % 2 == 1


def get_piece(board, row, col):
    """Safely get a piece from the board."""
    if in_bounds(row, col):
        piece = board[row][col]
        return piece
    return None


def is_own_piece(piece, current_player):
    """Check if a piece belongs to the current player."""
    if current_player == RED:
        if piece == RED or piece == RED_KING:
            return True
        return False

    if current_player == BLACK:
        if piece == BLACK or piece == BLACK_KING:
            return True
        return False

    return False


def is_opponent_piece(piece, current_player):
    """Check if a piece belongs to the opponent."""
    if current_player == RED:
        if piece == BLACK or piece == BLACK_KING:
            return True
        return False

    if current_player == BLACK:
        if piece == RED or piece == RED_KING:
            return True
        return False

    return False


def is_king(piece):
    """Check if a piece is a king."""
    if piece == RED_KING or piece == BLACK_KING:
        return True
    return False


def print_board(board):
    """
    Prints the board in a human-readable format.
    r = RED, R = RED_KING, b = BLACK, B = BLACK_KING, . = empty
    """
        
    symbols = {
        EMPTY: ".",
        RED: "r",
        BLACK: "b",
        RED_KING: "R",
        BLACK_KING: "B"
    }

    column_headers = ""
    for col in range(BOARD_SIZE):
        column_headers += str(col) + " "
    print("  " + column_headers)

    for row in range(BOARD_SIZE):
        row_str = str(row) + " "
        for col in range(BOARD_SIZE):
            current_piece = board[row][col]
            row_str += symbols[current_piece] + " "
        print(row_str)

    print()