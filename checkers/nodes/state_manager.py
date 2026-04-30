# nodes/state_manager.py
#
# The state_manager is the transition node between turns.
# It runs after the ranker_agent and before win_condition.
#
# Engine responsibilities (via apply_move): full path, all captures in one ply,
# multi-jump sequences, king promotion — already implemented in rules.apply_move.
#
# This node: verifies chosen_move against the live board (defense in depth),
# applies the move, hashes the new position, appends move_history, switches side,
# clears intra-turn fields (not cumulative thesis counters).
#
# Not duplicated here: full board snapshot per ply (heavy; use move_history +
# initial board for replay if you add inverse apply), or a separate piece registry
# (the board[][] is the registry).

from checkers.state.state import CheckersState
from checkers.engine.rules import apply_move
from checkers.engine.zobrist import compute_hash
from checkers.engine.board import RED, BLACK, BOARD_SIZE
from checkers.nodes.state_manager_verify import (
    _slim_move,
    verify_board_after_move,
    verify_chosen_move_for_state,
)


def state_manager(state: CheckersState) -> dict:
    """
    Applies the chosen move to the board and prepares the state
    for the next turn. Returns only the fields that changed so
    LangGraph can merge them back into the existing state.
    """

    current_board = state.board
    chosen_move = state.chosen_move
    current_player = state.current_player

    verify_chosen_move_for_state(state)

    zobrist_before = compute_hash(current_board)

    # Step 1 — Apply the chosen move to produce the new board.
    # apply_move never modifies the original board — it always
    # returns a new independent copy with the move applied.
    path = chosen_move["path"]
    from_row, from_col = path[0][0], path[0][1]
    to_row = path[-1][0]
    piece_before = current_board[from_row][from_col]
    promotion = (
        (piece_before == RED and to_row == 0)
        or (piece_before == BLACK and to_row == BOARD_SIZE - 1)
    )

    new_board = apply_move(current_board, chosen_move)
    verify_board_after_move(current_board, new_board, chosen_move)

    # Step 2 — Compute the Zobrist hash of the new board position
    # and append it to the position history.
    # This is what win_condition will use to detect draw by repetition.
    # We return the full updated list because LangGraph replaces
    # list fields entirely — it does not append automatically.
    new_hash = compute_hash(new_board)
    updated_position_history = state.position_history + [new_hash]

    # Step 3 — Switch the current player.
    # After RED moves it is BLACK's turn and vice versa.
    if current_player == RED:
        new_player = BLACK
    else:
        new_player = RED

    # Step 4 — Increment the turn counter.
    new_turn_number = state.turn_number + 1

    # Step 5 — Build the move record for this turn.
    # This gets appended to move_history so inter_turn_memory
    # can compute trends and detect patterns across turns.
    move_record = {
        "turn": new_turn_number,
        "player": current_player,
        "move": _slim_move(chosen_move),
        "promotion": promotion,
        "last_move_reasoning": state.last_move_reasoning,
        "zobrist_before": zobrist_before,
        "zobrist_after": new_hash,
    }
    
    # Step 6 — Return all changed fields.
    # Fields not included here remain unchanged in the state.
    # We clear all intra-turn working fields so the next turn
    # starts completely clean with no leftover data.
    # feedback: cleared so the next turn does not reuse validator/format errors.
    # format_error_count: cumulative session metric — not reset here.
    return {
        "board": new_board,
        "current_player": new_player,
        "turn_number": new_turn_number,
        "position_history": updated_position_history,
        "proposed_moves": [],
        "legal_moves": [],
        "chosen_move": None,
        "last_move_reasoning": None,
        "ranker_retry_count": 0,
        "retry_count": 0,
        "insufficient_proposals": False,
        "feedback": None,
        "move_history": state.move_history + [move_record],
        # ── Phase 8: clear per-turn symbolic fields ────────────────────────
        "symbolic_scored_moves": [],
        "symbolic_best_move": None,
        "symbolic_best_score": 0.0,
        "symbolic_second_best_score": None,
        "symbolic_gap": 0.0,
        "symbolic_bypass": False,
        "symbolic_bypass_reason": None,
        "llm_invoked": False,
        "llm_agreed_with_symbolic_best": None,
        "proposal_diagnostics": None,
        "last_completed_node": "state_manager",
    }


