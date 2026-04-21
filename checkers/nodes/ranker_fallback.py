# nodes/ranker_fallback.py
#
# Symbolic last resort when ranker_agent exhausts ranker_retry_budget without
# a valid chosen_move. Does not call an LLM.

from checkers.state.state import CheckersState


def ranker_fallback(state: CheckersState) -> dict:
    """
    Sets chosen_move to legal_moves[0] and records that the ranker chain failed.
    Cumulative ranker_fallback_count is for thesis evaluation.
    """
    legal = state.legal_moves
    if not legal:
        return {
            "chosen_move": None,
            "last_move_reasoning": None,
            "ranker_retry_count": 0,
            "legal_moves": [],
            "last_completed_node": "ranker_fallback",
        }
    return {
        "chosen_move": legal[0],
        "last_move_reasoning": (
            "Symbolic fallback: ranker retries exhausted (ranker_retry_budget); "
            "applied legal_moves[0]."
        ),
        "ranker_fallback_count": state.ranker_fallback_count + 1,
        "last_completed_node": "ranker_fallback",
    }
