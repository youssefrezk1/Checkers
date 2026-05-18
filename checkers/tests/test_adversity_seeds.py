# checkers/tests/test_adversity_seeds.py
#
# Phase 3.0: focused tests for _build_adversity_context_seeds and its
# integration into _build_grounded_reasoning_seeds.
#
# What IS tested:
#   - adversity seeds fire only when minimax_score < -20
#   - each of the five seed conditions (A-E) fires/suppresses correctly
#   - adversity seeds appear before the minimax-confirmation seed
#   - standard seeds (mobility, minimax) still present in adversity branch
#   - no forbidden vague words in adversity seeds
#   - non-losing positions produce identical seed lists (no regression)
#
# What is NOT tested:
#   - LLM call mechanics
#   - override, safety filter, retry logic
#   - evaluator / logger / state fields
#   - claim_extractor / claim_verifier

from __future__ import annotations

import re
from typing import Any

from checkers.agents.ranker_agent import (
    _build_adversity_context_seeds,
    _build_grounded_reasoning_seeds,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FORBIDDEN_WORDS = [
    "counterplay", "pressure", "activity", "balance",
    "trap", "initiative",
]


def _make_move(
    path=None,
    *,
    minimax_score: float = -50.0,
    material_advantage: int = 0,
    our_pieces_threatened_before: int = 0,
    our_pieces_threatened_after: int = 0,
    opponent_near_promotion: bool = False,
    opponent_mobility_before: int = 9,
    our_mobility_before: int = 9,
    opponent_can_recapture: bool = False,
    captures_count: int = 0,
    net_gain: int = 0,
    leaves_piece_isolated: bool = False,
    opponent_mobility_after: int = 9,
    our_mobility_after: int = 9,
    **extra_facts: Any,
) -> dict:
    if path is None:
        path = [[5, 2], [4, 3]]
    facts: dict = {
        "minimax_score": minimax_score,
        "material_advantage": material_advantage,
        "our_pieces_threatened_before": our_pieces_threatened_before,
        "our_pieces_threatened_after": our_pieces_threatened_after,
        "opponent_near_promotion": opponent_near_promotion,
        "opponent_mobility_before": opponent_mobility_before,
        "our_mobility_before": our_mobility_before,
        "opponent_can_recapture": opponent_can_recapture,
        "captures_count": captures_count,
        "net_gain": net_gain,
        "leaves_piece_isolated": leaves_piece_isolated,
        "opponent_mobility_after": opponent_mobility_after,
        "our_mobility_after": our_mobility_after,
        "moved_piece_is_threatened": False,
        "our_pieces_threatened_before": our_pieces_threatened_before,
        **extra_facts,
    }
    return {"path": path, "type": "simple", "facts": facts}


def _alt(path=None, minimax_score: float = -100.0) -> dict:
    """Minimal alternative move for all_candidates lists."""
    if path is None:
        path = [[6, 1], [5, 0]]
    return {"path": path, "type": "simple", "facts": {"minimax_score": minimax_score}}


# ---------------------------------------------------------------------------
# 1. Activation gate: fires below -20, silent at -20 and above
# ---------------------------------------------------------------------------

class TestAdversitySeedsActivationGate:

    def test_fires_below_threshold(self):
        move = _make_move(minimax_score=-25.0)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        # No special conditional seeds fire here (all suppressed by their own
        # guards), but the function must run without error and return a list.
        assert isinstance(seeds, list)

    def test_fires_at_clearly_losing(self):
        """Clearly losing (mm < -100) must also produce adversity seeds."""
        move = _make_move(minimax_score=-200.0, material_advantage=-2)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("material_advantage" in s for s in seeds)

    def test_integrated_fires_below_threshold(self):
        move = _make_move(minimax_score=-25.0, material_advantage=-1)
        all_c = [move, _alt(minimax_score=-80.0)]
        seeds = _build_grounded_reasoning_seeds(move, all_c)
        assert any("material_advantage" in s for s in seeds)

    def test_integrated_silent_at_minus_nineteen(self):
        move = _make_move(minimax_score=-19.0, material_advantage=-1)
        all_c = [move, _alt(minimax_score=-80.0)]
        seeds = _build_grounded_reasoning_seeds(move, all_c)
        assert not any("material_advantage" in s for s in seeds)

    def test_integrated_silent_at_zero(self):
        move = _make_move(minimax_score=0.0, material_advantage=-1)
        all_c = [move, _alt(minimax_score=-80.0)]
        seeds = _build_grounded_reasoning_seeds(move, all_c)
        assert not any("material_advantage" in s for s in seeds)

    def test_integrated_silent_at_positive(self):
        move = _make_move(minimax_score=30.0, material_advantage=-1)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        assert not any("material_advantage" in s for s in seeds)

    def test_no_crash_when_minimax_score_absent(self):
        move = {"path": [[5, 2], [4, 3]], "type": "simple", "facts": {}}
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert isinstance(seeds, list)

    def test_no_crash_when_facts_none(self):
        move = {"path": [[5, 2], [4, 3]], "type": "simple", "facts": None}
        seeds = _build_grounded_reasoning_seeds(move, [])
        assert isinstance(seeds, list)


# ---------------------------------------------------------------------------
# 2. Seed A — score gap to next-best alternative
# ---------------------------------------------------------------------------

class TestScoreGapSeed:

    def _seeds(self, chosen_mm: float, alt_mm: float) -> list[str]:
        move = _make_move(path=[[5, 2], [4, 3]], minimax_score=chosen_mm)
        alt  = _alt(path=[[6, 1], [5, 0]], minimax_score=alt_mm)
        return _build_adversity_context_seeds(move["facts"], [move, alt], move["path"])

    def test_fires_when_gap_exceeds_20(self):
        seeds = self._seeds(-50.0, -100.0)
        assert any("points better" in s for s in seeds)

    def test_contains_exact_chosen_score(self):
        seeds = self._seeds(-50.0, -100.0)
        gap_seed = next(s for s in seeds if "points better" in s)
        assert "-50.0" in gap_seed

    def test_contains_exact_alt_score(self):
        seeds = self._seeds(-50.0, -100.0)
        gap_seed = next(s for s in seeds if "points better" in s)
        assert "-100.0" in gap_seed

    def test_contains_gap_value(self):
        seeds = self._seeds(-50.0, -100.0)
        gap_seed = next(s for s in seeds if "points better" in s)
        assert "50.0" in gap_seed

    def test_absent_when_gap_exactly_20(self):
        seeds = self._seeds(-50.0, -70.0)  # gap == 20.0 — not > 20
        assert not any("points better" in s for s in seeds)

    def test_absent_when_gap_below_20(self):
        seeds = self._seeds(-50.0, -65.0)  # gap == 15
        assert not any("points better" in s for s in seeds)

    def test_absent_when_no_alternatives(self):
        move = _make_move(minimax_score=-50.0)
        seeds = _build_adversity_context_seeds(move["facts"], [move], move["path"])
        assert not any("points better" in s for s in seeds)

    def test_absent_when_alt_mm_is_none(self):
        move = _make_move(minimax_score=-50.0)
        alt = {"path": [[6, 1], [5, 0]], "type": "simple", "facts": {}}
        seeds = _build_adversity_context_seeds(move["facts"], [move, alt], move["path"])
        assert not any("points better" in s for s in seeds)


# ---------------------------------------------------------------------------
# 3. Seed B — material deficit
# ---------------------------------------------------------------------------

class TestMaterialDeficitSeed:

    def test_fires_when_behind_by_one(self):
        move = _make_move(minimax_score=-30.0, material_advantage=-1)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("material_advantage=-1" in s for s in seeds)

    def test_fires_when_behind_by_three(self):
        move = _make_move(minimax_score=-30.0, material_advantage=-3)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("behind by 3 piece(s)" in s for s in seeds)

    def test_uses_exact_deficit_value(self):
        move = _make_move(minimax_score=-30.0, material_advantage=-2)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        mat_seed = next(s for s in seeds if "material_advantage" in s)
        assert "material_advantage=-2" in mat_seed
        assert "behind by 2 piece(s)" in mat_seed

    def test_absent_when_equal(self):
        move = _make_move(minimax_score=-30.0, material_advantage=0)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("material_advantage" in s for s in seeds)

    def test_absent_when_ahead(self):
        move = _make_move(minimax_score=-30.0, material_advantage=2)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("material_advantage" in s for s in seeds)

    def test_absent_when_key_missing(self):
        facts = {"minimax_score": -30.0}
        seeds = _build_adversity_context_seeds(facts, [], [[5, 2], [4, 3]])
        assert not any("material_advantage" in s for s in seeds)


# ---------------------------------------------------------------------------
# 4. Seed C — threat reduction
# ---------------------------------------------------------------------------

class TestThreatReductionSeed:

    def test_fires_when_threat_reduced(self):
        move = _make_move(
            minimax_score=-30.0,
            our_pieces_threatened_before=2,
            our_pieces_threatened_after=1,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("reduces threatened pieces from 2 to 1" in s for s in seeds)

    def test_fires_when_fully_resolved(self):
        move = _make_move(
            minimax_score=-30.0,
            our_pieces_threatened_before=2,
            our_pieces_threatened_after=0,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("reduces threatened pieces from 2 to 0" in s for s in seeds)

    def test_contains_exact_numbers(self):
        move = _make_move(
            minimax_score=-30.0,
            our_pieces_threatened_before=3,
            our_pieces_threatened_after=1,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        seed = next(s for s in seeds if "reduces threatened pieces" in s)
        assert "3" in seed and "1" in seed

    def test_absent_when_no_prior_threat(self):
        move = _make_move(
            minimax_score=-30.0,
            our_pieces_threatened_before=0,
            our_pieces_threatened_after=0,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("reduces threatened pieces" in s for s in seeds)

    def test_absent_when_no_improvement(self):
        move = _make_move(
            minimax_score=-30.0,
            our_pieces_threatened_before=1,
            our_pieces_threatened_after=1,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("reduces threatened pieces" in s for s in seeds)

    def test_absent_when_threat_increases(self):
        move = _make_move(
            minimax_score=-30.0,
            our_pieces_threatened_before=1,
            our_pieces_threatened_after=2,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("reduces threatened pieces" in s for s in seeds)


# ---------------------------------------------------------------------------
# 5. Seed D — opponent near promotion
# ---------------------------------------------------------------------------

class TestOpponentNearPromotionSeed:

    def test_fires_when_true(self):
        move = _make_move(minimax_score=-30.0, opponent_near_promotion=True)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("opponent_near_promotion=true" in s for s in seeds)

    def test_uses_board_state_wording_only(self):
        move = _make_move(minimax_score=-30.0, opponent_near_promotion=True)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        promo_seed = next(s for s in seeds if "opponent_near_promotion" in s)
        # Must describe board state, not claim move blocks it
        assert "one step from promotion" in promo_seed
        assert "block" not in promo_seed.lower()
        assert "prevent" not in promo_seed.lower()

    def test_absent_when_false(self):
        move = _make_move(minimax_score=-30.0, opponent_near_promotion=False)
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("opponent_near_promotion" in s for s in seeds)

    def test_absent_when_key_missing(self):
        facts = {"minimax_score": -30.0}
        seeds = _build_adversity_context_seeds(facts, [], [[5, 2], [4, 3]])
        assert not any("opponent_near_promotion" in s for s in seeds)


# ---------------------------------------------------------------------------
# 6. Seed E — mobility asymmetry
# ---------------------------------------------------------------------------

class TestMobilityAsymmetrySeed:

    def test_fires_when_gap_exactly_4(self):
        move = _make_move(
            minimax_score=-30.0,
            opponent_mobility_before=13,
            our_mobility_before=9,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("structural disadvantage" in s for s in seeds)

    def test_fires_when_gap_greater_than_4(self):
        move = _make_move(
            minimax_score=-30.0,
            opponent_mobility_before=15,
            our_mobility_before=8,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("structural disadvantage" in s for s in seeds)

    def test_contains_exact_mobility_values(self):
        move = _make_move(
            minimax_score=-30.0,
            opponent_mobility_before=14,
            our_mobility_before=9,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        mob_seed = next(s for s in seeds if "structural disadvantage" in s)
        assert "14" in mob_seed and "9" in mob_seed

    def test_fires_when_gap_exactly_3(self):
        move = _make_move(
            minimax_score=-30.0,
            opponent_mobility_before=12,
            our_mobility_before=9,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert any("structural disadvantage" in s for s in seeds)

    def test_absent_when_gap_is_2(self):
        move = _make_move(
            minimax_score=-30.0,
            opponent_mobility_before=11,
            our_mobility_before=9,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("structural disadvantage" in s for s in seeds)

    def test_absent_when_mobility_facts_absent(self):
        facts = {"minimax_score": -30.0}
        seeds = _build_adversity_context_seeds(facts, [], [[5, 2], [4, 3]])
        assert not any("structural disadvantage" in s for s in seeds)

    def test_absent_when_not_losing(self):
        """Gate: Seed E must not appear when mm >= -20 (gate in _build_grounded_reasoning_seeds)."""
        move = _make_move(
            minimax_score=-19.0,
            opponent_mobility_before=12,
            our_mobility_before=9,
        )
        seeds = _build_grounded_reasoning_seeds(move, [move])
        assert not any("structural disadvantage" in s for s in seeds)

    def test_absent_when_equal(self):
        move = _make_move(
            minimax_score=-30.0,
            opponent_mobility_before=9,
            our_mobility_before=9,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("structural disadvantage" in s for s in seeds)

    def test_absent_when_we_have_more(self):
        move = _make_move(
            minimax_score=-30.0,
            opponent_mobility_before=7,
            our_mobility_before=11,
        )
        seeds = _build_adversity_context_seeds(move["facts"], [], move["path"])
        assert not any("structural disadvantage" in s for s in seeds)


# ---------------------------------------------------------------------------
# 7. Ordering and structure
# ---------------------------------------------------------------------------

class TestAdversitySeedOrdering:

    def _full_seeds(self, minimax_score: float = -60.0, **kw) -> list[str]:
        move = _make_move(minimax_score=minimax_score, **kw)
        alt  = _alt(minimax_score=minimax_score - 50.0)
        return _build_grounded_reasoning_seeds(move, [move, alt])

    def test_adversity_seeds_before_minimax_wording(self):
        seeds = self._full_seeds(material_advantage=-1)
        mat_idx = next(i for i, s in enumerate(seeds) if "material_advantage" in s)
        mm_idx  = next(i for i, s in enumerate(seeds) if "minimax_score=" in s)
        assert mat_idx < mm_idx

    def test_minimax_seed_is_last(self):
        seeds = self._full_seeds(material_advantage=-1)
        assert "minimax_score=" in seeds[-1]

    def test_standard_mobility_seeds_still_present(self):
        """Phase-6 dedup: mobility seeds now use natural-language wording
        ('opponent mobility remains at N' / 'our mobility changes from X to Y').
        The legacy 'key=value' structured form was removed to prevent the
        LLM from mechanically restating the same fact twice."""
        seeds = self._full_seeds(
            opponent_mobility_before=9,
            our_mobility_before=9,
            opponent_mobility_after=9,
            our_mobility_after=9,
        )
        assert any("opponent mobility remains at 9" in s for s in seeds), seeds
        assert any("our mobility remains at 9" in s for s in seeds), seeds

    def test_adversity_block_before_recapture_seed(self):
        """Material deficit seed precedes opponent_can_recapture seed."""
        seeds = self._full_seeds(
            material_advantage=-2,
            opponent_can_recapture=True,
        )
        mat_idx  = next((i for i, s in enumerate(seeds) if "material_advantage" in s), None)
        recap_idx = next((i for i, s in enumerate(seeds) if "opponent_can_recapture" in s), None)
        if mat_idx is not None and recap_idx is not None:
            assert mat_idx < recap_idx

    def test_multiple_adversity_seeds_all_present(self):
        seeds = self._full_seeds(
            material_advantage=-1,
            our_pieces_threatened_before=2,
            our_pieces_threatened_after=1,
            opponent_near_promotion=True,
            opponent_mobility_before=15,
            our_mobility_before=9,
        )
        assert any("material_advantage" in s for s in seeds)
        assert any("reduces threatened pieces" in s for s in seeds)
        assert any("opponent_near_promotion" in s for s in seeds)
        assert any("structural disadvantage" in s for s in seeds)


# ---------------------------------------------------------------------------
# 8. No forbidden vague words
# ---------------------------------------------------------------------------

class TestNoForbiddenWords:

    def test_no_forbidden_words_in_adversity_seeds(self):
        move = _make_move(
            minimax_score=-60.0,
            material_advantage=-2,
            our_pieces_threatened_before=1,
            our_pieces_threatened_after=0,
            opponent_near_promotion=True,
            opponent_mobility_before=14,
            our_mobility_before=9,
        )
        alt = _alt(minimax_score=-130.0)
        seeds = _build_adversity_context_seeds(move["facts"], [move, alt], move["path"])
        combined = " ".join(seeds).lower()
        for word in _FORBIDDEN_WORDS:
            assert word not in combined, f"Forbidden word '{word}' found in adversity seeds"

    def test_gap_seed_no_forbidden_words(self):
        move = _make_move(minimax_score=-50.0)
        alt  = _alt(minimax_score=-120.0)
        seeds = _build_adversity_context_seeds(move["facts"], [move, alt], move["path"])
        gap_seeds = [s for s in seeds if "points better" in s]
        combined = " ".join(gap_seeds).lower()
        for word in _FORBIDDEN_WORDS:
            assert word not in combined


# ---------------------------------------------------------------------------
# 9. Non-losing regression: seed output unchanged for mm >= -20
# ---------------------------------------------------------------------------

class TestNonLosingRegression:

    def _seeds_at_score(self, score: float) -> list[str]:
        move = _make_move(
            minimax_score=score,
            material_advantage=-1,          # would fire deficit seed if losing
            our_pieces_threatened_before=2,  # would fire threat seed if losing
            our_pieces_threatened_after=1,
            opponent_near_promotion=True,    # would fire promo seed if losing
            opponent_mobility_before=14,     # would fire mobility seed if losing
            our_mobility_before=9,
        )
        alt = _alt(minimax_score=score - 60.0)
        return _build_grounded_reasoning_seeds(move, [move, alt])

    def test_no_adversity_seeds_at_minus_nineteen(self):
        seeds = self._seeds_at_score(-19.0)
        assert not any("material_advantage" in s for s in seeds)
        assert not any("reduces threatened pieces" in s for s in seeds)
        assert not any("opponent_near_promotion=true" in s for s in seeds)
        assert not any("structural disadvantage" in s for s in seeds)
        assert not any("points better" in s for s in seeds)

    def test_no_adversity_seeds_at_zero(self):
        seeds = self._seeds_at_score(0.0)
        assert not any("material_advantage" in s for s in seeds)
        assert not any("structural disadvantage" in s for s in seeds)

    def test_standard_seeds_still_present_at_minus_nineteen(self):
        seeds = self._seeds_at_score(-19.0)
        assert any("opponent_can_recapture" in s for s in seeds)
        assert any("minimax_score=" in s for s in seeds)

    def test_minimax_last_in_non_losing(self):
        seeds = self._seeds_at_score(-19.0)
        assert "minimax_score=" in seeds[-1]

    def test_minimax_last_in_losing(self):
        seeds = self._seeds_at_score(-50.0)
        assert "minimax_score=" in seeds[-1]
