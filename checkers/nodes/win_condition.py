# nodes/win_condition.py
#
# The win_condition node runs after state_manager every turn.
# By this point the board has already been updated with the move
# that was just played.
#
# This node checks all three end conditions:
#   1. Draw by repetition — same board position appeared 3 times
#      detected using Zobrist hashing on position_history
#   2. The opponent has no pieces left on the board
#   3. The opponent has pieces but all of them are completely blocked
#
# This node is purely symbolic — no LLM calls, no randomness.

from checkers.state.state import CheckersState
from checkers.engine.win_condition import check_win_condition
from checkers.engine.zobrist import check_repetition, compute_hash
from checkers.engine.board import RED, BLACK


def win_condition(state: CheckersState) -> dict:
    """
    Checks if the game has ended after the most recent move.
    Returns only the fields that changed so LangGraph can merge
    them back into the existing state.
    """

    current_board = state.board
    current_player = state.current_player

    # Case 1 — Draw by repetition.
    # Compute the hash of the current board and check if it has
    # appeared 3 times in position_history.
    current_hash = compute_hash(current_board)
    is_draw = check_repetition(state.position_history, current_hash)

    if is_draw:
        return {
            "game_over": True,
            "winner": None,
            "draw": True,
            "last_completed_node": "win_condition"
        }

    # Case 2 and 3 — Opponent has no pieces or is fully blocked.
    # current_player has already been switched by state_manager to
    # the player whose turn is next. So the player who just moved
    # is the opposite of current_player right now.
    if current_player == RED:
        player_who_just_moved = BLACK
    else:
        player_who_just_moved = RED

    result = check_win_condition(current_board, player_who_just_moved)

    if result["game_over"] == True:
        return {
            "game_over": True,
            "winner": result["winner"],
            "draw": False,
            "last_completed_node": "win_condition"
        }

    # Case 4 — No end condition met. Game continues.
    return {
        "game_over": False,
        "winner": None,
        "draw": False,
        "last_completed_node": "win_condition"
    }
