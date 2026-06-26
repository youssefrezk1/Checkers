# checkers/tests/test_semantic_hallucination_guards.py
#
# Regression tests for semantic hallucination patterns found in the human
# annotation audit (63/96 paragraphs contained at least one factual error).
#
# Patterns covered:
#   SH-1  Fabricated comparison values ("N points better")
#   SH-2  False "only legal move" / forced-move claim when multiple moves exist
#   SH-3  Reverse recapture (opponent_can_recapture=False but claims recapture)
#   SH-4  Near-promotion false claim (near_promotion=False)
#   SH-5  Wrong capture count (claims 1 capture when captures_count=2)
#   SH-6  Adversity seeds use natural language, not field=value format
#   SH-7  Near-promotion negative seed emitted when near_promotion=False
#
# Run:
#     pytest checkers/tests/test_semantic_hallucination_guards.py -v

from __future__ import annotations

from checkers.agents.explainer_agent import (
    _build_adversity_context_seeds,
    _build_grounded_reasoning_seeds,
    _check_reasoning_truthfulness,
    _negative_grounding_seeds,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _warns(reasoning: str, facts: dict, seeds=None) -> bool:
    return len(_check_reasoning_truthfulness(reasoning, facts, seeds=seeds)) > 0


def _clean(reasoning: str, facts: dict, seeds=None) -> bool:
    return not _warns(reasoning, facts, seeds=seeds)


def _base_facts(**overrides) -> dict:
    base = {
        "captures_count":           0,
        "net_gain":                 0,
        "opponent_can_recapture":   False,
        "leaves_piece_isolated":    False,
        "creates_immediate_threat": False,
        "forced_opponent_jump_reply": False,
        "center_control":           False,
        "results_in_king":          False,
        "near_promotion":           False,
        "opponent_mobility_before": 8,
        "opponent_mobility_after":  8,
        "our_mobility_before":      8,
        "our_mobility_after":       8,
        "minimax_score":            2.0,
    }
    base.update(overrides)
    return base


def _multi_move_seeds() -> list[str]:
    """Seeds that include the 'multiple legal moves' negative."""
    return [
        "The moved piece cannot be immediately recaptured.",
        "This move does not create an immediate threat.",
        "The opponent is not forced to respond with a jump.",
        "Multiple legal moves were available; the engine selected this option.",
        "The engine scores this move 2.0 — highest-evaluated option.",
    ]


# ── SH-1: Fabricated comparison values ───────────────────────────────────────

class TestFabricatedComparisonValues:

    def test_fabricated_comparison_flagged_no_seeds(self):
        facts = _base_facts()
        r = "The chosen move is 33.0 points better than all alternatives."
        assert _warns(r, facts, seeds=[])

    def test_fabricated_comparison_flagged_with_unrelated_seeds(self):
        facts = _base_facts()
        seeds = ["The moved piece cannot be immediately recaptured."]
        r = "This move scores 27 points better than the next option."
        assert _warns(r, facts, seeds=seeds)

    def test_grounded_comparison_passes_when_number_in_seeds(self):
        facts = _base_facts()
        seeds = [
            "The chosen move scores 33.0 points better than the next-best option "
            "[move 1] (engine scores: 45.0 vs 12.0)."
        ]
        r = "The move scores 33.0 points better than the alternative."
        assert _clean(r, facts, seeds=seeds)

    def test_different_number_still_flagged_even_if_seed_has_a_number(self):
        facts = _base_facts()
        seeds = [
            "The chosen move scores 33.0 points better than the next-best option "
            "[move 1] (engine scores: 45.0 vs 12.0)."
        ]
        # LLM uses wrong number (22) not present in seeds
        r = "This path is 22 points better than any alternative."
        assert _warns(r, facts, seeds=seeds)

    def test_qualitative_comparison_allowed(self):
        facts = _base_facts()
        seeds = ["The engine scores this move 5.0 — highest-evaluated option."]
        r = "The engine evaluated this path more highly than the alternatives."
        assert _clean(r, facts, seeds=seeds)


# ── SH-2: False forced-move / only-legal-move claims ─────────────────────────

class TestFalseForcedMoveClaim:

    def test_only_legal_move_flagged_when_multiple_exist(self):
        facts = _base_facts()
        seeds = _multi_move_seeds()
        r = "The move from (5,4) to (4,3) is the only legal move available."
        assert _warns(r, facts, seeds=seeds)

    def test_only_viable_option_flagged(self):
        facts = _base_facts()
        seeds = _multi_move_seeds()
        r = "This is the only viable option given the current constraints."
        assert _warns(r, facts, seeds=seeds)

    def test_no_alternative_flagged(self):
        facts = _base_facts()
        seeds = _multi_move_seeds()
        r = "With no alternative, the engine plays to (4,3)."
        assert _warns(r, facts, seeds=seeds)

    def test_forced_move_phrase_flagged(self):
        facts = _base_facts()
        seeds = _multi_move_seeds()
        r = "This forced move advances the piece to a central square."
        assert _warns(r, facts, seeds=seeds)

    def test_normal_reasoning_passes(self):
        facts = _base_facts()
        seeds = _multi_move_seeds()
        r = ("The move from (5,4) to (4,3) advances the piece without creating an "
             "immediate threat. The engine scores this move 2.0 — highest-evaluated option.")
        assert _clean(r, facts, seeds=seeds)

    def test_no_warning_when_seeds_absent(self):
        # Without seeds the check cannot fire (no "multiple legal moves" context)
        facts = _base_facts()
        r = "This is the only viable option given the board."
        assert _clean(r, facts, seeds=None)

    def test_no_warning_for_opponent_forced(self):
        # "opponent is forced" is about the opponent's reply, not our move — should pass
        facts = _base_facts(forced_opponent_jump_reply=True, creates_immediate_threat=True)
        seeds = [
            "This move forces the opponent to respond to an immediate threat.",
            "Multiple legal moves were available; the engine selected this option.",
        ]
        r = ("The move from (5,4) to (4,3) forces the opponent to respond to an "
             "immediate threat. The engine scores this move 2.0.")
        assert _clean(r, facts, seeds=seeds)


# ── SH-3: Reverse recapture direction ────────────────────────────────────────

class TestReverseRecaptureClaim:

    def test_opponent_can_recapture_claim_flagged_when_false(self):
        facts = _base_facts(opponent_can_recapture=False)
        r = "The opponent can recapture the piece on the next turn."
        assert _warns(r, facts)

    def test_vulnerable_to_recapture_flagged_when_false(self):
        facts = _base_facts(opponent_can_recapture=False)
        r = "The piece is vulnerable to recapture after this advance."
        assert _warns(r, facts)

    def test_allows_recapture_flagged(self):
        facts = _base_facts(opponent_can_recapture=False)
        r = "The move allows the opponent to recapture at will."
        assert _warns(r, facts)

    def test_safe_claim_allowed_when_recapture_false(self):
        facts = _base_facts(opponent_can_recapture=False)
        r = "The moved piece cannot be immediately recaptured."
        assert _clean(r, facts)

    def test_no_false_positive_when_recapture_true(self):
        # When recapture=True, claiming the opponent can recapture is correct
        facts = _base_facts(opponent_can_recapture=True)
        r = "The opponent can recapture the moved piece next turn."
        assert _clean(r, facts)


# ── SH-4: Near-promotion false claim ─────────────────────────────────────────

class TestNearPromotionFalseClaim:

    def test_near_promotion_phrase_flagged_when_false(self):
        facts = _base_facts(near_promotion=False)
        r = "The piece is near promotion after this advance."
        assert _warns(r, facts)

    def test_one_step_from_promotion_flagged(self):
        facts = _base_facts(near_promotion=False)
        r = "The piece is now one step from promotion."
        assert _warns(r, facts)

    def test_advancing_toward_promotion_flagged(self):
        facts = _base_facts(near_promotion=False)
        r = "The piece advances toward promotion, closing in on a king."
        assert _warns(r, facts)

    def test_near_promotion_allowed_when_true(self):
        facts = _base_facts(near_promotion=True)
        r = "The piece is now one step from promotion."
        assert _clean(r, facts)

    def test_no_false_positive_neutral_language(self):
        facts = _base_facts(near_promotion=False)
        r = "The piece advances forward without capturing."
        assert _clean(r, facts)


# ── SH-5: Wrong capture count ─────────────────────────────────────────────────

class TestWrongCaptureCount:

    def test_claims_one_capture_when_count_is_two(self):
        facts = _base_facts(captures_count=2, net_gain=2)
        r = "The move captures 1 piece, gaining a net advantage."
        assert _warns(r, facts)

    def test_correct_count_passes(self):
        facts = _base_facts(captures_count=2, net_gain=2)
        r = "The move captures 2 pieces, gaining a net advantage of 2."
        assert _clean(r, facts)

    def test_claims_two_when_count_is_one(self):
        facts = _base_facts(captures_count=1, net_gain=1)
        r = "Capturing 2 pieces gives an immediate material advantage."
        assert _warns(r, facts)

    def test_no_capture_check_still_works(self):
        # captures_count=0 with a capture claim — covered by existing check
        facts = _base_facts(captures_count=0)
        r = "The move captures a piece."
        assert _warns(r, facts)


# ── SH-6: Adversity seeds use natural language (no field=value) ───────────────

class TestAdversitySeedFormat:

    def _make_adversity_candidates(self, chosen_mm: float, alt_mm: float) -> list[dict]:
        return [
            {"path": [[6, 1], [5, 2]], "facts": {"minimax_score": chosen_mm}},
            {"path": [[6, 3], [5, 4]], "facts": {"minimax_score": alt_mm}},
        ]

    def test_material_deficit_seed_natural_language(self):
        facts = {"minimax_score": -30.0, "material_advantage": -2}
        candidates = self._make_adversity_candidates(-30.0, -45.0)
        seeds = _build_adversity_context_seeds(facts, candidates, [[6, 1], [5, 2]])
        all_text = " ".join(seeds)
        assert "material_advantage" not in all_text
        assert "behind by 2 piece" in all_text

    def test_opponent_near_promotion_seed_natural_language(self):
        facts = {"minimax_score": -30.0, "opponent_near_promotion": True}
        candidates = self._make_adversity_candidates(-30.0, -45.0)
        seeds = _build_adversity_context_seeds(facts, candidates, [[6, 1], [5, 2]])
        all_text = " ".join(seeds)
        assert "opponent_near_promotion" not in all_text
        assert "one step from promotion" in all_text

    def test_mobility_asymmetry_seed_natural_language(self):
        facts = {
            "minimax_score": -30.0,
            "opponent_mobility_before": 12,
            "our_mobility_before": 6,
        }
        candidates = self._make_adversity_candidates(-30.0, -45.0)
        seeds = _build_adversity_context_seeds(facts, candidates, [[6, 1], [5, 2]])
        all_text = " ".join(seeds)
        assert "opponent_mobility_before" not in all_text
        assert "our_mobility_before" not in all_text
        assert "12 available moves" in all_text

    def test_score_gap_seed_natural_language(self):
        facts = {"minimax_score": -10.0}
        candidates = self._make_adversity_candidates(-10.0, -35.0)
        seeds = _build_adversity_context_seeds(facts, candidates, [[6, 1], [5, 2]])
        all_text = " ".join(seeds)
        # Old format started with bare "chosen move scores" (no leading "The").
        assert not any(s.startswith("chosen move scores") for s in seeds), \
            "seed must start with 'The chosen move scores', not bare 'chosen move scores'"
        assert any(s.startswith("The chosen move scores") for s in seeds)
        assert "points better" in all_text

    def test_threat_reduction_seed_natural_language(self):
        facts = {
            "minimax_score": -30.0,
            "our_pieces_threatened_before": 3,
            "our_pieces_threatened_after": 1,
        }
        candidates = self._make_adversity_candidates(-30.0, -45.0)
        seeds = _build_adversity_context_seeds(facts, candidates, [[6, 1], [5, 2]])
        all_text = " ".join(seeds)
        # Old format started bare "reduces threatened pieces..." (no leading "This move").
        assert not any(s.startswith("reduces threatened") for s in seeds), \
            "seed must start with 'This move reduces...', not bare 'reduces threatened'"
        assert "reduces threatened allied pieces" in all_text


# ── SH-7: Near-promotion negative seed ───────────────────────────────────────

class TestNearPromotionNegativeSeed:

    def _facts_near_promo_false(self) -> dict:
        return {
            "captures_count": 0,
            "net_gain": 0,
            "opponent_can_recapture": False,
            "creates_immediate_threat": False,
            "forced_opponent_jump_reply": False,
            "near_promotion": False,
            "results_in_king": False,
            "leaves_piece_isolated": False,
        }

    def test_negative_seed_emitted_when_near_promotion_false(self):
        facts = self._facts_near_promo_false()
        seeds = _negative_grounding_seeds(facts, n_candidates=3, existing_seeds=[])
        assert any("not near promotion" in s.lower() for s in seeds)

    def test_negative_seed_suppressed_when_near_promotion_true(self):
        facts = self._facts_near_promo_false()
        facts["near_promotion"] = True
        existing = ["The piece is now one step from promotion."]
        seeds = _negative_grounding_seeds(facts, n_candidates=3, existing_seeds=existing)
        assert not any("not near promotion" in s.lower() for s in seeds)

    def test_negative_seed_suppressed_when_results_in_king(self):
        facts = self._facts_near_promo_false()
        facts["results_in_king"] = True
        # "promoted" contains the "promot" prefix — suppresses the negative near-promo seed
        existing = ["The piece is immediately promoted to king."]
        seeds = _negative_grounding_seeds(facts, n_candidates=3, existing_seeds=existing)
        assert not any("not near promotion" in s.lower() for s in seeds)

    def test_negative_seed_suppressed_when_promotion_in_existing_seeds(self):
        facts = self._facts_near_promo_false()
        existing = ["The move advances the piece toward promotion."]
        seeds = _negative_grounding_seeds(facts, n_candidates=3, existing_seeds=existing)
        assert not any("not near promotion" in s.lower() for s in seeds)

    def test_negative_seed_not_duplicated_in_full_seed_build(self):
        # When near_promotion=True, _build_grounded_reasoning_seeds emits the positive
        # seed via the promotion block; _negative_grounding_seeds must not add "not near".
        chosen_move = {
            "type": "simple",
            "path": [[3, 2], [2, 1]],
            "facts": {
                "captures_count": 0,
                "near_promotion": True,
                "results_in_king": False,
                "opponent_can_recapture": False,
                "creates_immediate_threat": False,
                "forced_opponent_jump_reply": False,
                "minimax_score": 3.0,
            },
        }
        seeds = _build_grounded_reasoning_seeds(chosen_move, [chosen_move])
        neg_promo = [s for s in seeds if "not near promotion" in s.lower()]
        pos_promo = [s for s in seeds if "one step from promotion" in s.lower()]
        assert not neg_promo, "negative seed must not appear when near_promotion=True"
        assert pos_promo, "positive seed must appear when near_promotion=True"
