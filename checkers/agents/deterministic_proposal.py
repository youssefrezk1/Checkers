# checkers/agents/deterministic_proposal.py
#
# Deterministic shortlist selector — Python-only alternative to the Groq LLM
# proposal stage.  Accepts scored_moves from scorer_agent.score_all_legal_moves
# and returns move dicts directly (no JSON, no indices, no LLM call).
#
# Self-contained: all sorting and role-pinning helpers are defined here.
# No imports from proposal_agent.py.
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

logger = logging.getLogger(__name__)


# ── Symbolic pre-sort key ─────────────────────────────────────────────────────
# Copied verbatim from proposal_agent._proposal_sort_key.
# Behavior must remain identical to keep shortlist selection consistent.

def _proposal_sort_key(
    facts: dict,
    score_state: str,
    game_phase: str,
    strategic_priorities: list[str],
) -> tuple:
    """
    Symbolic pre-sort key for legal moves before the shortlist selector runs.
    Lower tuple value = presented earlier (more likely to be shortlisted).

    Sort priority order:
      1. Safety first  — our_pieces_threatened_after (ascending)
      2. Unsafe flag   — unsafe_simple_move=True goes last
      3. Captures      — captures_count (descending)
      4. Promotion     — results_in_king or near_promotion
      5. Score-state   — winning_conversion_score (winning) or counterplay_score (losing)
      6. Role coverage — quiet_move_role ranking
      7. Tiebreakers   — center_control, isolation penalty
    """
    threatened_after = facts.get("our_pieces_threatened_after", 0)
    unsafe_simple = facts.get(
        "unsafe_simple_move",
        facts.get("move_type") == "simple" and threatened_after > 0
    )
    unsafe = 1 if unsafe_simple else 0
    captures         = facts.get("captures_count", 0)
    is_promotion     = 1 if (facts.get("results_in_king", False) or facts.get("near_promotion", False)) else 0
    conversion_score = facts.get("winning_conversion_score", 0)
    counterplay      = facts.get("counterplay_score", 0)
    center           = 1 if facts.get("center_control", False) else 0
    isolated         = 1 if facts.get("leaves_piece_isolated", False) else 0

    losing_states  = ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
    winning_states = ("CLEARLY_WINNING", "SLIGHTLY_WINNING")

    if score_state in winning_states:
        state_score = -conversion_score
    elif score_state in losing_states:
        state_score = -counterplay
    else:
        state_score = -max(conversion_score, counterplay)

    _ROLE_RANK = {
        "TACTICAL":                 0,
        "PROMOTION_PUSH":           1,
        "KING_ACTIVATION":          2,
        "COUNTERPLAY":              3,
        "CONVERSION":               4,
        "DEFENSIVE_STABILIZATION":  5,
        "MOBILITY_IMPROVEMENT":     6,
        "QUIET_DEFAULT":            7,
    }
    role = facts.get("quiet_move_role", "QUIET_DEFAULT")

    if game_phase == "ENDGAME" and role == "KING_ACTIVATION":
        role_rank = 1
    elif "SEEK_COUNTERPLAY" in strategic_priorities and role == "COUNTERPLAY":
        role_rank = 2
    elif "CONVERT_ADVANTAGE" in strategic_priorities and role == "CONVERSION":
        role_rank = 2
    else:
        role_rank = _ROLE_RANK.get(role, 7)

    return (
        threatened_after,
        unsafe,
        -captures,
        -is_promotion,
        state_score,
        role_rank,
        -center,
        isolated,
    )


# ── Strategic role-pinning ────────────────────────────────────────────────────
# Copied verbatim from proposal_agent._role_pin_moves.
# Behavior must remain identical.

def _role_pin_moves(
    sorted_moves: list[tuple[dict, dict]],
    score_state: str,
    n_slots: int,
) -> tuple[list[tuple[dict, dict]], frozenset[int]]:
    """
    Ensures strategic role diversity by pinning up to one protected move per
    critical role into the first n_slots positions after the symbolic pre-sort.

    Only acts when a protected move is currently OUTSIDE positions 0..n_slots-1.
    Position 0 (symbolic best) is never displaced.

    Protected roles (pinned in priority order after position 0):
      1. Best promotion / near-promotion move
      2. Best mobility-reduction move (mobility_reduction > 0)
      3. Best winning-conversion move  (only when score_state is winning)
      4. Best counterplay move         (only when score_state is not winning)

    Returns (reordered_moves, pinned_new_positions) where pinned_new_positions
    is the frozenset of positions in the returned list that were role-pinned.
    """
    n = len(sorted_moves)
    if n <= n_slots:
        return sorted_moves, frozenset()

    _WINNING = ("CLEARLY_WINNING", "SLIGHTLY_WINNING")

    def _find_best(score_fn, filter_fn) -> int | None:
        best_i, best_s = None, float("-inf")
        for i, (_, f) in enumerate(sorted_moves):
            if not filter_fn(f):
                continue
            s = score_fn(f)
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    to_pin_indices: set[int] = set()

    i = _find_best(
        score_fn=lambda f: 2 if f.get("results_in_king") else 1,
        filter_fn=lambda f: f.get("results_in_king") or f.get("near_promotion"),
    )
    if i is not None and i >= n_slots:
        to_pin_indices.add(i)

    i = _find_best(
        score_fn=lambda f: f.get("mobility_reduction", 0),
        filter_fn=lambda f: (
            f.get("mobility_reduction", 0) > 0
            and not f.get("opponent_can_recapture", False)
        ),
    )
    if i is not None and i >= n_slots:
        to_pin_indices.add(i)

    if score_state in _WINNING:
        i = _find_best(
            score_fn=lambda f: f.get("winning_conversion_score", 0),
            filter_fn=lambda f: (
                f.get("winning_conversion_score", 0) > 0
                and not f.get("opponent_can_recapture", False)
            ),
        )
        if i is not None and i >= n_slots:
            to_pin_indices.add(i)

    if score_state not in _WINNING:
        _LOSING = ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
        if score_state in _LOSING:
            i = _find_best(
                score_fn=lambda f: f.get("counterplay_score", 0),
                filter_fn=lambda f: f.get("counterplay_score", 0) > 0,
            )
        else:
            i = _find_best(
                score_fn=lambda f: f.get("counterplay_score", 0),
                filter_fn=lambda f: (
                    f.get("counterplay_score", 0) > 0
                    and not f.get("opponent_can_recapture", False)
                ),
            )
        if i is not None and i >= n_slots:
            to_pin_indices.add(i)

    if not to_pin_indices:
        return sorted_moves, frozenset()

    to_pin = [sorted_moves[i] for i in sorted(to_pin_indices)]
    rest   = [
        item
        for i, item in enumerate(sorted_moves)
        if i != 0 and i not in to_pin_indices
    ]
    result = [sorted_moves[0]] + to_pin + rest
    assert len(result) == n, "_role_pin_moves must not change the number of moves"

    pinned_new_positions = frozenset(range(1, 1 + len(to_pin)))
    return result, pinned_new_positions

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
