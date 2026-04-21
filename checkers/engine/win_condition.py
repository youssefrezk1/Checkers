# engine/win_condition.py

from checkers.engine.board import (
    RED, BLACK, BOARD_SIZE,
    is_own_piece
)
from checkers.engine.rules import get_all_legal_moves


def has_no_pieces_left(board, player):
    """
    Checks if the player has zero pieces remaining on the board.
    Loops through every square and returns True if no piece belongs to player.
    """
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if is_own_piece(piece, player):
                return False

    # If we finished the loop without finding any piece, player has none left
    return True


def has_no_moves_left(board, player):
    """
    Checks if the player has pieces but none of them can move.
    Uses get_all_legal_moves which already handles both simple moves
    and jumps — if it returns an empty list, the player is fully blocked.
    """
    legal_moves = get_all_legal_moves(board, player)

    if len(legal_moves) == 0:
        return True

    return False


def check_win_condition(board, current_player):
    """
    Checks if the game is over after current_player just made their move.
    We check if the OPPONENT is now in a losing state.

    Returns a dict with:
        - game_over  : True or False
        - winner     : RED, BLACK, or None
        - reason     : explanation string for logging and LLM context
    """
    if current_player == RED:
        opponent = BLACK
    else:
        opponent = RED

    # Condition 1 — opponent has no pieces left
    if has_no_pieces_left(board, opponent):
        return {
            "game_over": True,
            "winner": current_player,
            "reason": "opponent has no pieces remaining"
        }

    # Condition 2 — opponent has pieces but cannot move
    if has_no_moves_left(board, opponent):
        return {
            "game_over": True,
            "winner": current_player,
            "reason": "opponent has pieces but is completely blocked"
        }

    # No win condition met — game continues
    return {
        "game_over": False,
        "winner": None,
        "reason": None
    }