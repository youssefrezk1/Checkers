# checkers/nodes/deterministic_proposal_node.py
#
# Simplified pipeline node — replaces proposal_agent + format_checker + validator.
# Only active when USE_SIMPLIFIED_PIPELINE=true; the old nodes are untouched.
#
# Reads state.legal_moves (all enriched moves from scorer_node, sorted best-first)
# and overwrites state.legal_moves with a shortlist of min(5, n) moves.
# ranker_agent reads state.legal_moves directly, so it sees the shortlist immediately.
#
# Never raises: select_proposal_candidates is fully deterministic and handles
# edge cases (empty input, fewer than 5 moves) without retry logic.

from __future__ import annotations

from checkers.state.state import CheckersState
from checkers.agents.deterministic_proposal import select_proposal_candidates


def deterministic_proposal_node(state: CheckersState) -> dict:
    shortlist = select_proposal_candidates(
        state.legal_moves,
        strategic_context=state.strategic_context,
        k=5,
    )

    return {
        "legal_moves": shortlist,
        "last_completed_node": "deterministic_proposal_node",
    }
