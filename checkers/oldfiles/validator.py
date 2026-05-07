# nodes/validator.py
#
# The validator node runs after the proposal agent proposes moves.
# It is the symbolic gate between the LLM proposal and the ranker.
#
# Responsibilities:
#   1. Compute all truly legal moves from the current board state
#      using the symbolic rule engine — this is the ground truth
#   2. Enrich each legal move with move_facts so the ranker receives
#      fully annotated moves and focuses purely on strategic reasoning
#   3. Validate each proposed move against the legal moves
#   4. If ALL proposed moves are illegal classify why each failed
#      and return feedback to the proposal agent for retry
#   5. If at least one proposed move is legal pass only the valid
#      enriched moves to the ranker — discard illegal ones silently
#
# This node is purely symbolic — no LLM calls, no randomness.
# It is the hard enforcement layer that guarantees the ranker
# never sees an illegal move.
import os
DEBUG_ALL_LEGAL_TO_RANKER = os.environ.get("DEBUG_ALL_LEGAL_TO_RANKER", "false").lower() in (
    "1", "true", "yes", "on"
)
from checkers.state.state import CheckersState
from checkers.engine.rules import get_all_legal_moves, apply_move, _moves_match
from checkers.engine.move_facts import compute_move_facts
from checkers.engine.board import BOARD_SIZE, in_bounds, is_own_piece
from checkers.engine.zobrist import compute_hash

def _classify_error(proposed_move, legal_moves, board, current_player):
    """
    Classifies why a proposed move is illegal.
    Returns one of four error type strings:

    CAPTURE_IGNORED  — jumps exist but a simple move was proposed
    BOUNDARY         — from or to coordinates are off the board
    DIRECTION        — piece moved in a direction it is not allowed to
    INVALID_JUMP     — jump path is wrong or landing square is incorrect
    PIECE_MISMATCH   — no own piece exists at the from square
    """

    from_pos = proposed_move.get("from")
    to_pos = proposed_move.get("to")
    move_type = proposed_move.get("type")

    # Check boundary first — if coordinates are off board nothing else matters
    if from_pos is None or to_pos is None:
        return "BOUNDARY"

    from_row = from_pos[0]
    from_col = from_pos[1]
    to_row = to_pos[0]
    to_col = to_pos[1]

    if not in_bounds(from_row, from_col) or not in_bounds(to_row, to_col):
        return "BOUNDARY"

    # Check if there is actually an own piece at the from square
    piece_at_from = board[from_row][from_col]
    if not is_own_piece(piece_at_from, current_player):
        return "PIECE_MISMATCH"

    # Check if any jump exists in legal moves — if yes and proposal is
    # simple then the player ignored a mandatory capture
    any_jump_exists = any(m["type"] == "jump" for m in legal_moves)
    if any_jump_exists and move_type == "simple":
        return "CAPTURE_IGNORED"

    # Check direction — row must decrease for RED regular, increase for BLACK regular
    # Kings are exempt from direction check
    from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
    if piece_at_from == RED:
        if to_row >= from_row:
            return "DIRECTION"
    elif piece_at_from == BLACK:
        if to_row <= from_row:
            return "DIRECTION"

    # Everything else that does not match a legal move is an invalid jump
    return "INVALID_JUMP"


def _deduplicate_moves(valid_enriched, board):
    """
    Removes duplicate moves from the valid enriched list.
    Uses Zobrist hashing of the resulting board after each move
    to detect moves that lead to identical board states —
    this catches cases that path comparison alone would miss.
    """
    seen_hashes = set()
    deduplicated = []

    for enriched in valid_enriched:
        board_after = apply_move(board, enriched)
        board_hash = compute_hash(board_after)

        if board_hash not in seen_hashes:
            seen_hashes.add(board_hash)
            deduplicated.append(enriched)

    return deduplicated

def validator(state: CheckersState) -> dict:
    """
    Validates proposed moves against the symbolic rule engine.
    Returns enriched legal moves for the ranker or feedback
    for the proposal agent if all proposals were illegal.
    """

    board = state.board
    current_player = state.current_player
    proposed_moves = state.proposed_moves

    if isinstance(proposed_moves, str):
        return {
            "legal_moves": [],
            "feedback": (
                "proposed_moves must be a list of move dicts after format_checker; "
                "received a string."
            ),
            "retry_count": state.retry_count + 1,
            "last_completed_node": "validator",
        }
    if not isinstance(proposed_moves, list):
        return {
            "legal_moves": [],
            "feedback": (
                f"proposed_moves must be a list, got {type(proposed_moves).__name__}."
            ),
            "retry_count": state.retry_count + 1,
            "last_completed_node": "validator",
        }

    if len(proposed_moves) == 0:
        return {
            "legal_moves": [],
            "feedback": (
                "No proposed moves to validate — format_checker produced an empty list "
                "(JSON parse failure or every candidate failed structural checks)."
            ),
            "retry_count": state.retry_count + 1,
            "last_completed_node": "validator",
        }

    # Step 1 — Compute all truly legal moves from the engine.
    # This is the symbolic ground truth — no LLM involved.
    all_legal_moves = get_all_legal_moves(board, current_player)

    # Step 2 — Enrich each legal move with move_facts.
    # The ranker receives fully annotated moves and focuses
    # purely on strategic reasoning — no computation needed there.
    enriched_legal_moves = []
    for move in all_legal_moves:
        facts = compute_move_facts(board, move, current_player)
        enriched_move = {
            "type": move["type"],
            "path": move["path"],
            "captured": move["captured"],
            "facts": facts
        }
        enriched_legal_moves.append(enriched_move)

    # DEBUG MODE:
    # Bypass proposal narrowing and send all engine-legal enriched moves
    # directly to minimax_scorer and ranker_agent.
    if DEBUG_ALL_LEGAL_TO_RANKER:
        # Keep proposal-selected legal moves first, in proposal order,
        # then append the remaining engine-legal moves in engine order.
        debug_moves = []

        # 1) Add legal moves that match the proposal list, preserving proposal order
        for proposed in proposed_moves:
            for enriched in enriched_legal_moves:
                if _moves_match(proposed, enriched):
                    debug_moves.append(enriched)
                    break

        # 2) Append the remaining legal moves not already included
        for enriched in enriched_legal_moves:
            already_added = any(
                _moves_match({"type": m["type"], "path": m["path"]}, enriched)
                for m in debug_moves
            )
            if not already_added:
                debug_moves.append(enriched)

        deduplicated_moves = _deduplicate_moves(debug_moves, board)
        return {
            "legal_moves": deduplicated_moves,
            "feedback": None,
            "last_completed_node": "validator"
        }



    # Step 3 — Validate each proposed move against legal moves.
    valid_enriched = []
    error_classifications = []

    for proposed in proposed_moves:
        matched = False

        for enriched in enriched_legal_moves:
            if _moves_match(proposed, enriched):
                valid_enriched.append(enriched)
                matched = True
                break

        if not matched:
            error = _classify_error(
                proposed, all_legal_moves, board, current_player
            )
            error_classifications.append({
                "proposed": proposed,
                "error": error
            })

    # Step 4 — Deduplicate valid moves using Zobrist hashing.
    # Two different paths that lead to the same board state
    # are strategically identical — the ranker only needs one.
    # This also catches exact path duplicates since identical
    # paths always produce identical board hashes.
    if len(valid_enriched) > 0:
        deduplicated_moves = _deduplicate_moves(valid_enriched, board)
        return {
            "legal_moves": deduplicated_moves,
            "feedback": None,
            "last_completed_node": "validator"
        }

    # Step 5 — ALL proposed moves were illegal.
    # Build a clean feedback string classifying each failure.
    # The proposal agent reads this on retry to self-correct.
    feedback_lines = []
    for entry in error_classifications:
        proposed = entry["proposed"]
        error = entry["error"]
        from_pos = proposed.get("from", "?")
        to_pos = proposed.get("to", "?")
        feedback_lines.append(
            f"{error}: move from {from_pos} to {to_pos}"
        )

    feedback_message = (
        "All proposed moves were illegal. Errors:\n" +
        "\n".join(feedback_lines) +
        "\nPlease choose selected_indices that appear in the engine legal list (see proposal prompt)."
    )

    return {
        "legal_moves": [],
        "feedback": feedback_message,
        "retry_count": state.retry_count + 1,
        "last_completed_node": "validator"
    }