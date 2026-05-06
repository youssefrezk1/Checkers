# checkers/agents/deterministic_proposal.py
#
# Deterministic shortlist selector — Python-only alternative to the Groq LLM
# proposal stage.  Accepts scored_moves from scorer_agent.score_all_legal_moves
# and returns move dicts directly (no JSON, no indices, no LLM call).
#
# Reuses from proposal_agent:
#   _proposal_sort_key  — symbolic pre-sort (safety, captures, promotion, role)
#   _role_pin_moves     — strategic diversity guarantee
# Plus a local mirror of the inline minimax top-3 pin block from
# _build_legal_moves_with_facts (same algorithm, extracted as _mm_pin).
#
# Public API:
#   select_proposal_candidates(scored_moves, strategic_context=None, k=5)
#       -> list[dict]
#
#   Each returned dict is an element of scored_moves (no copies, no new fields).
#   Exactly min(k, len(scored_moves)) moves are returned.
#   Selection order: minimax top-3 first (via mm-pin), then role-diversity picks,
#   then fill from the symbolic-sorted list.

from __future__ import annotations

import logging
from typing import Any

from checkers.agents.proposal_agent import _proposal_sort_key, _role_pin_moves

logger = logging.getLogger(__name__)


def _mm_pin(
    moves_with_facts: list[tuple[dict, dict]],
    n_slots: int,
) -> list[tuple[dict, dict]]:
    """
    Guarantee that the three highest-minimax-score moves appear within
    positions 0..n_slots-1 of moves_with_facts.

    Mirrors the inline mm-pin block in _build_legal_moves_with_facts.
    Mutates and returns the list.  No-op when n_slots < 5 or no scores.
    """
    if n_slots < 5 or len(moves_with_facts) < n_slots:
        return moves_with_facts

    def _score(mv: dict, facts: dict) -> float | None:
        sc = facts.get("minimax_score")
        if sc is None or sc == float("-inf"):
            return None
        try:
            return float(sc)
        except (TypeError, ValueError):
            return None

    def _path_key(mv: dict) -> tuple:
        return tuple(tuple(sq) for sq in mv.get("path", []))

    def _top3_indices() -> list[int]:
        scored = []
        for i, (mv, facts) in enumerate(moves_with_facts):
            sc = _score(mv, facts)
            if sc is not None:
                scored.append((sc, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [i for _, i in scored[:3]]

    top3 = _top3_indices()

    for top_i in top3:
        if top_i < n_slots:
            continue

        top3_set = set(top3)
        displace_at: int | None = None
        for j in range(n_slots - 1, -1, -1):
            if j not in top3_set:
                displace_at = j
                break
        if displace_at is None:
            continue

        # Locate the target move by path (position may have shifted after earlier pins).
        target_pk = _path_key(moves_with_facts[top_i][0])
        current_idx = next(
            (ci for ci, (mv, _) in enumerate(moves_with_facts)
             if _path_key(mv) == target_pk),
            top_i,
        )
        pin_item = moves_with_facts.pop(current_idx)
        actual = displace_at if current_idx >= displace_at else displace_at - 1
        actual = max(0, min(actual, n_slots - 1))
        moves_with_facts.insert(actual, pin_item)
        top3 = _top3_indices()

    return moves_with_facts


def select_proposal_candidates(
    scored_moves: list[dict[str, Any]],
    strategic_context: dict[str, Any] | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Deterministically select min(k, len(scored_moves)) candidates.

    Parameters
    ----------
    scored_moves:
        Output of score_all_legal_moves — list of enriched move dicts, each:
            {"type": str, "path": list, "captured": list, "facts": dict}
        facts must contain "minimax_score" (float) and "symbolic_rank" (int).
        The list must be sorted best-first by minimax_score (scorer_agent
        guarantees this).

    strategic_context:
        Optional dict from CheckersState.strategic_context.  Used by
        _proposal_sort_key and _role_pin_moves to weight captures, promotions,
        counterplay, etc. relative to the current game phase and score state.
        Pass None (or omit) to use neutral defaults.

    k:
        Maximum shortlist size (default 5).

    Returns
    -------
    list[dict]
        Ordered list of move dicts, each a direct reference to an element of
        scored_moves (no copies, no new fields added).
        Guarantees:
          - len(result) == min(k, len(scored_moves))
          - every element is in scored_moves
          - when len(scored_moves) >= 5 and k >= 5, the top-3 moves by
            minimax_score are always included
    """
    n = len(scored_moves)
    if n == 0:
        return []

    target = min(k, n)

    ctx = strategic_context or {}
    score_state          = ctx.get("score_state", "EQUAL")
    game_phase           = ctx.get("game_phase", "MIDGAME")
    strategic_priorities = ctx.get("strategic_priorities", [])

    # Build lookup: path_key -> original scored_move dict for final reconstruction.
    path_to_original: dict[tuple, dict] = {
        tuple(tuple(sq) for sq in m["path"]): m
        for m in scored_moves
    }

    # Decompose each scored move into the (move_meta, facts) format that the
    # imported proposal_agent helpers expect.
    moves_with_facts: list[tuple[dict, dict]] = [
        (
            {"type": m["type"], "path": m["path"], "captured": m.get("captured", [])},
            m["facts"],
        )
        for m in scored_moves
    ]

    # ── Step 1: symbolic pre-sort ──────────────────────────────────────────────
    moves_with_facts.sort(
        key=lambda pair: _proposal_sort_key(
            pair[1], score_state, game_phase, strategic_priorities
        )
    )

    # ── Step 2: strategic role-pinning ─────────────────────────────────────────
    n_slots = min(5, n)
    moves_with_facts, _ = _role_pin_moves(moves_with_facts, score_state, n_slots)

    # ── Step 3: minimax top-3 inclusion guarantee ──────────────────────────────
    if target >= 5:
        moves_with_facts = _mm_pin(moves_with_facts, n_slots)

    # ── Step 4: reconstruct and return min(k, n) original move dicts ──────────
    result: list[dict] = []
    for mv, _ in moves_with_facts[:target]:
        pk = tuple(tuple(sq) for sq in mv.get("path", []))
        orig = path_to_original.get(pk)
        if orig is not None:
            result.append(orig)

    logger.info(
        "[deterministic_proposal] n_legal=%d target=%d selected=%d scores=%s",
        n,
        target,
        len(result),
        [round(m["facts"].get("minimax_score", 0.0), 2) for m in result],
    )

    return result
