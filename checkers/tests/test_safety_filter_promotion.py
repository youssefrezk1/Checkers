#!/usr/bin/env python3
"""
Regression test for the T37 safety-filter promotion bug.

Root cause:
  At Turn 37, RED had a legal promotion move (1,6) -> (0,7) with
  results_in_king=True. The move also had opponent_can_recapture=True
  (corner King is theoretically reachable by BLACK's King). The safety
  filter classified it as "unsafe" and excluded it. Only 1 safe move
  survived (filtered_menu_size=1). Minimax, ranker, and override never
  compared against the promotion. RED chose a defensive shuffle instead.

Fix:
  Promotion moves (results_in_king=True) must always survive the safety
  filter. They are added back unconditionally before the candidate set is
  returned. The ranker and minimax override remain free to reject them.

This test verifies:
1. A promotion move that is "unsafe" (opponent_can_recapture=True) is NOT
   removed by _apply_safety_filter.
2. The filtered candidate set has size >= 2 (promotion + at least one safe move).
3. The promotion move's path appears in the returned candidate set.
4. The safe defensive move also survives (filter is additive, not replacing).
"""

from __future__ import annotations

from checkers.agents.ranker_agent import _apply_safety_filter


def _make_move(path, minimax_score, results_in_king=False, opponent_can_recapture=False):
    return {
        "type": "simple",
        "path": path,
        "captured": [],
        "facts": {
            "minimax_score": minimax_score,
            "results_in_king": results_in_king,
            "opponent_can_recapture": opponent_can_recapture,
            "our_pieces_threatened_after": 1 if opponent_can_recapture else 0,
            "net_gain": 0,
            "counterplay_score": 2,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
        },
    }


def test_promotion_survives_safety_filter_when_unsafe():
    """
    Mirrors T37: one safe low-score move, one promotion move tagged unsafe.
    The promotion MUST appear in the filtered output.
    """
    safe_shuffle = _make_move(
        path=[(6, 5), (5, 4)],
        minimax_score=20.0,
        results_in_king=False,
        opponent_can_recapture=False,
    )
    promotion = _make_move(
        path=[(1, 6), (0, 7)],
        minimax_score=-25.0,
        results_in_king=True,
        opponent_can_recapture=True,  # corner King is theoretically threatened
    )
    legal = [safe_shuffle, promotion]

    filtered, index_map = _apply_safety_filter(
        legal,
        strategic_priorities=["PROMOTE", "TRADE_WHEN_AHEAD"],
        score_state="EQUAL",
    )

    filtered_paths = [m.get("path") for m in filtered]

    # Promotion must NOT be excluded.
    assert [(1, 6), (0, 7)] in filtered_paths, (
        f"Promotion move was wrongly excluded. Filtered set: {filtered_paths}"
    )
    # Safe move must also survive.
    assert [(6, 5), (5, 4)] in filtered_paths, (
        f"Safe move was wrongly excluded. Filtered set: {filtered_paths}"
    )
    # Filter must not have collapsed to 1 candidate.
    assert len(filtered) >= 2, (
        f"Filter collapsed to {len(filtered)} candidate(s); promotion was excluded. "
        f"filtered_paths={filtered_paths}"
    )
    # index_map must be consistent with the legal list.
    for filtered_idx, legal_idx in enumerate(index_map):
        assert filtered[filtered_idx] is legal[legal_idx], (
            f"index_map[{filtered_idx}]={legal_idx} does not point to the right move."
        )
    print("PASS: promotion move survived safety filter.")


def test_promotion_survives_when_safe_moves_outnumber_it():
    """
    Multiple safe moves exist. The promotion is still unsafe but must be kept.
    """
    safe1 = _make_move([(5, 2), (4, 3)], minimax_score=10.0)
    safe2 = _make_move([(5, 4), (4, 5)], minimax_score=5.0)
    promotion = _make_move(
        [(1, 6), (0, 7)],
        minimax_score=-10.0,
        results_in_king=True,
        opponent_can_recapture=True,
    )
    legal = [safe1, safe2, promotion]

    filtered, index_map = _apply_safety_filter(legal, score_state="EQUAL")
    filtered_paths = [m.get("path") for m in filtered]

    assert [(1, 6), (0, 7)] in filtered_paths, (
        f"Promotion was excluded from multi-safe-move set. Filtered: {filtered_paths}"
    )
    assert len(filtered) >= 3, (
        f"Expected >= 3 candidates, got {len(filtered)}."
    )
    print("PASS: promotion survives when multiple safe moves exist.")


def test_promotion_that_is_also_safe_is_not_duplicated():
    """
    If a promotion move has opponent_can_recapture=False, it is already in the
    safe set. It must appear exactly once in the output.
    """
    safe_promo = _make_move(
        [(1, 6), (0, 7)],
        minimax_score=30.0,
        results_in_king=True,
        opponent_can_recapture=False,
    )
    safe_other = _make_move([(5, 2), (4, 3)], minimax_score=10.0)
    legal = [safe_other, safe_promo]

    filtered, index_map = _apply_safety_filter(legal, score_state="EQUAL")
    filtered_paths = [m.get("path") for m in filtered]

    assert filtered_paths.count([(1, 6), (0, 7)]) == 1, (
        f"Promotion appeared {filtered_paths.count([(1, 6), (0, 7)])} times; expected exactly 1."
    )
    print("PASS: safe promotion is not duplicated.")


if __name__ == "__main__":
    test_promotion_survives_safety_filter_when_unsafe()
    test_promotion_survives_when_safe_moves_outnumber_it()
    test_promotion_that_is_also_safe_is_not_duplicated()
    print("\nAll 3 promotion safety-filter tests passed.")
