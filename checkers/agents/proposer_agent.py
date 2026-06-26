# checkers/agents/deterministic_proposal.py
#
# Deterministic single-move selector for the proposal-authoritative pipeline.
# Accepts the fully-scored legal-move list from scorer_agent and returns
# the rank-1 move as the sole authority for move selection.
#
# Public API:
#   select_best_move(scored_moves, score_state="EQUAL")
#       -> (chosen_move, chosen_score, unchosen_moves, proposal_meta)

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def select_best_move(
    scored_moves: list[dict[str, Any]],
    score_state: str = "EQUAL",
) -> tuple[dict[str, Any], float, list[dict[str, Any]], dict[str, Any]]:
    """
    Deterministically choose the SINGLE BEST move from scored_moves.

    scored_moves must be sorted best-first by minimax_score (scorer_agent
    guarantees this).  The first element is always the chosen move.

    Parameters
    ----------
    scored_moves :
        Output of score_all_legal_moves — list of enriched move dicts, each:
            {"type": str, "path": list, "captured": list, "facts": dict}
        facts must contain "minimax_score" (float) and "symbolic_rank" (int).

    score_state :
        Whole-position balance string from state.score_state (written by
        scorer_node).  Used for proposal_meta diagnostics only — selection
        is purely minimax rank-1.

    Returns
    -------
    chosen_move : dict
        The best move dict (with full facts preserved).
    chosen_score : float
        minimax_score of the chosen move.
    unchosen_moves : list[dict]
        All other moves, preserving full facts and original order.
    proposal_meta : dict
        Diagnostic metadata about the selection.
    """
    if not scored_moves:
        return {}, 0.0, [], {"n_legal": 0, "selection_method": "none"}

    chosen_move = scored_moves[0]
    chosen_score = chosen_move["facts"]["minimax_score"]
    unchosen_moves = scored_moves[1:]

    second_best_score = (
        unchosen_moves[0]["facts"]["minimax_score"]
        if unchosen_moves
        else None
    )
    gap = (
        round(chosen_score - second_best_score, 2)
        if second_best_score is not None
        else None
    )

    proposal_meta = {
        "n_legal": len(scored_moves),
        "chosen_path": chosen_move["path"],
        "chosen_score": chosen_score,
        "second_best_score": second_best_score,
        "gap": gap,
        "score_state": score_state,
        "selection_method": "minimax_rank_1",
    }

    logger.info(
        "[deterministic_proposal] select_best_move: n_legal=%d "
        "chosen_score=%.2f gap=%s score_state=%s path=%s",
        len(scored_moves),
        chosen_score,
        f"{gap:.2f}" if gap is not None else "N/A",
        score_state,
        chosen_move["path"],
    )

    return chosen_move, chosen_score, unchosen_moves, proposal_meta
