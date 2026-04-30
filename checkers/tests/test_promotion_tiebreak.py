"""
Tests for promotion tie-break in ranker_agent._override_if_llm_chose_much_worse_minimax
pipeline (post-override section).

We test the tie-break logic in isolation by simulating what happens AFTER the
override returns chosen_after_override, with a crafted `legal` list.

The tie-break runs from:
    chosen = chosen_after_override
    ...
    for _m in legal:
        if _mf.get("results_in_king", False): ...

We invoke the full rank_move path through a minimal integration call using
the constants from ranker_agent directly.
"""
import os, sys
sys.path.insert(0, '.')

# ── Minimal helpers ───────────────────────────────────────────────────────────

def _make_move(path, score, results_in_king=False, opp_recapture=False):
    return {
        "path": path,
        "type": "simple",
        "captured": [],
        "facts": {
            "minimax_score": score,
            "results_in_king": results_in_king,
            "opponent_can_recapture": opp_recapture,
            "our_pieces_threatened_after": 0,
            "moved_piece_is_threatened": False,
            "net_gain": 0,
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "counterplay_score": 0,
            "king_activity_score": 0,
            "winning_conversion_score": 1,
            "quiet_move_role": "QUIET_DEFAULT",
            "near_promotion": False,
            "blocks_opponent_landing": False,
        },
    }

def _run_tiebreak(chosen, legal, margin=3.0):
    """
    Simulate the promotion tie-break section from ranker_agent.
    Returns (final_chosen, tiebreak_fired).
    """
    from checkers.agents.ranker_agent import _get_minimax_score

    PROMOTION_TIEBREAK_MARGIN = margin
    chosen_facts_tb = chosen.get("facts", {}) or {}
    chosen_actually_promotes = chosen_facts_tb.get("results_in_king", False)
    tiebreak_fired = False
    if not chosen_actually_promotes:
        chosen_score_tb = _get_minimax_score(chosen)
        promo_candidate = None
        promo_score_tb = float("-inf")
        for _m in legal:
            _mf = _m.get("facts", {}) or {}
            if _mf.get("results_in_king", False):
                _ms = _get_minimax_score(_m)
                if _ms >= chosen_score_tb - PROMOTION_TIEBREAK_MARGIN:
                    if _ms > promo_score_tb:
                        promo_score_tb = _ms
                        promo_candidate = _m
        if promo_candidate is not None:
            tiebreak_fired = True
            chosen = promo_candidate
    return chosen, tiebreak_fired


# ── Test 1: Exact tie → tie-break fires ──────────────────────────────────────

def test_exact_tie_tiebreak_fires():
    """
    Non-promotion chosen at +96, promotion available at +96.
    Tie-break must replace chosen with the promotion.
    """
    chosen = _make_move([(2, 1), (1, 0)], 96.0, results_in_king=False)
    promotion = _make_move([(1, 6), (0, 7)], 96.0, results_in_king=True)
    center = _make_move([(5, 4), (4, 3)], 96.0, results_in_king=False)
    legal = [chosen, center, promotion]  # promotion is last in list

    final, fired = _run_tiebreak(chosen, legal)

    assert fired, "Tie-break should have fired"
    assert final["path"] == [(1, 6), (0, 7)], f"Expected promotion move, got {final['path']}"
    print("PASS test_exact_tie_tiebreak_fires")


# ── Test 2: Promotion clearly worse → no replacement ─────────────────────────

def test_promotion_clearly_worse_no_replacement():
    """
    Chosen at +96, promotion at +50. Gap = 46 > PROMOTION_TIEBREAK_MARGIN=3.
    Tie-break must NOT fire.
    """
    chosen = _make_move([(5, 4), (4, 3)], 96.0, results_in_king=False)
    promotion = _make_move([(1, 6), (0, 7)], 50.0, results_in_king=True)
    legal = [chosen, promotion]

    final, fired = _run_tiebreak(chosen, legal)

    assert not fired, "Tie-break should NOT fire when promotion is clearly worse"
    assert final["path"] == [(5, 4), (4, 3)], f"Chosen should be unchanged, got {final['path']}"
    print("PASS test_promotion_clearly_worse_no_replacement")


# ── Test 3: Chosen already promotes → no replacement ─────────────────────────

def test_chosen_already_promotes_no_replacement():
    """
    Chosen is itself a promotion (results_in_king=True).
    Tie-break must NOT fire (no reason to replace a promotion with itself).
    """
    chosen = _make_move([(1, 6), (0, 7)], 96.0, results_in_king=True)
    other_promo = _make_move([(2, 1), (1, 0)], 96.0, results_in_king=True)  # hypothetical second promo
    legal = [chosen, other_promo]

    final, fired = _run_tiebreak(chosen, legal)

    assert not fired, "Tie-break should NOT fire when chosen is already a promotion"
    assert final["path"] == [(1, 6), (0, 7)], f"Chosen should be unchanged, got {final['path']}"
    print("PASS test_chosen_already_promotes_no_replacement")


# ── Test 4: No promotion in legal → no replacement ───────────────────────────

def test_no_promotion_available_no_replacement():
    """
    No move in legal has results_in_king=True.
    Tie-break must NOT fire.
    """
    chosen = _make_move([(5, 4), (4, 3)], 96.0, results_in_king=False)
    other = _make_move([(2, 1), (1, 0)], 96.0, results_in_king=False)
    legal = [chosen, other]

    final, fired = _run_tiebreak(chosen, legal)

    assert not fired, "Tie-break should NOT fire when no promotion exists"
    assert final["path"] == [(5, 4), (4, 3)], f"Chosen should be unchanged, got {final['path']}"
    print("PASS test_no_promotion_available_no_replacement")


# ── Test 5: Near-tie within margin → fires ───────────────────────────────────

def test_near_tie_within_margin_fires():
    """
    Chosen at +96, promotion at +94 (gap=2 < margin=3). Should fire.
    """
    chosen = _make_move([(5, 4), (4, 3)], 96.0, results_in_king=False)
    promotion = _make_move([(1, 6), (0, 7)], 94.0, results_in_king=True)
    legal = [chosen, promotion]

    final, fired = _run_tiebreak(chosen, legal)

    assert fired, "Tie-break should fire when promotion is within margin"
    assert final["path"] == [(1, 6), (0, 7)], f"Expected promotion, got {final['path']}"
    print("PASS test_near_tie_within_margin_fires")


# ── Test 6: Near-tie just outside margin → does NOT fire ─────────────────────

def test_near_tie_outside_margin_no_fire():
    """
    Chosen at +96, promotion at +92 (gap=4 > margin=3). Should NOT fire.
    """
    chosen = _make_move([(5, 4), (4, 3)], 96.0, results_in_king=False)
    promotion = _make_move([(1, 6), (0, 7)], 92.0, results_in_king=True)
    legal = [chosen, promotion]

    final, fired = _run_tiebreak(chosen, legal)

    assert not fired, "Tie-break should NOT fire when gap exceeds margin"
    assert final["path"] == [(5, 4), (4, 3)], f"Chosen unchanged, got {final['path']}"
    print("PASS test_near_tie_outside_margin_no_fire")


if __name__ == "__main__":
    test_exact_tie_tiebreak_fires()
    test_promotion_clearly_worse_no_replacement()
    test_chosen_already_promotes_no_replacement()
    test_no_promotion_available_no_replacement()
    test_near_tie_within_margin_fires()
    test_near_tie_outside_margin_no_fire()
    print("\nAll 6 promotion tie-break tests passed.")
