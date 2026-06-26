# checkers/tests/test_phase_b1_fixes.py
#
# Regression tests for Phase B1 fixes:
#   B1.1  Comparative recapture fabrication check
#   B1.2  "Outweighs" / tradeoff language requires numeric grounding
#   B1.3  Negative-score absolute advantage protection
#
# Each class verifies both the runtime inline check (_check_reasoning_truthfulness)
# and the evaluator mirror (_check_* + verify_all + contradiction_strings), plus
# E.1 invariant parity.

from __future__ import annotations

import pytest

from checkers.agents.explainer_agent import _check_reasoning_truthfulness
from checkers.evaluation.unified_verifier import (
    _check_comparative_recapture_fabrication,
    _check_outweighs_numeric_grounding,
    _check_negative_score_advantage,
    contradiction_strings,
    verify_all,
    assert_runtime_evaluator_agreement,
)
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _runtime_warns(text: str, facts: dict) -> list[str]:
    return _check_reasoning_truthfulness(text, facts, seeds=[])


def _runtime_clean(text: str, facts: dict) -> bool:
    return not any(
        "COMPARATIVE_CONTRADICTION" in w
        for w in _runtime_warns(text, facts)
    )


def _evaluator_warns(text: str, facts: dict) -> list[str]:
    return contradiction_strings(text, facts=facts)


def _evaluator_clean(text: str, facts: dict) -> bool:
    return not any(
        "COMPARATIVE_CONTRADICTION" in w
        for w in _evaluator_warns(text, facts)
    )


def _contains_b1_contradiction(warnings: list[str], substring: str) -> bool:
    return any(substring in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# B1.1 — Comparative recapture fabrication
# ─────────────────────────────────────────────────────────────────────────────

_RECAP_TRUE_FACTS = {"opponent_can_recapture": True, "minimax_score": 12.0}
_RECAP_FALSE_FACTS = {"opponent_can_recapture": False, "minimax_score": 12.0}


class TestComparativeRecaptureFabrication:

    def test_evaluator_fires_recapture_safety_when_recap_true(self):
        records = _check_comparative_recapture_fabrication(
            "this move provides recapture safety over the alternative.",
            _RECAP_TRUE_FACTS,
        )
        assert len(records) == 1
        assert records[0].claim_status == ClaimStatus.CONTRADICTED
        assert records[0].matched_phrase == "recapture safety"

    def test_evaluator_fires_avoiding_recapture_when_recap_true(self):
        records = _check_comparative_recapture_fabrication(
            "the chosen move excels at avoiding recapture compared to [1].",
            _RECAP_TRUE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "avoiding recapture"

    def test_evaluator_fires_avoid_recapture_risk_when_recap_true(self):
        records = _check_comparative_recapture_fabrication(
            "this path helps avoid recapture risk unlike the alternative.",
            _RECAP_TRUE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "avoid recapture risk"

    def test_evaluator_no_fire_when_recap_false(self):
        records = _check_comparative_recapture_fabrication(
            "this move provides recapture safety over the alternative.",
            _RECAP_FALSE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_when_phrase_absent(self):
        records = _check_comparative_recapture_fabrication(
            "the chosen move captures two pieces while [1] captures one.",
            _RECAP_TRUE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_when_facts_missing(self):
        records = _check_comparative_recapture_fabrication(
            "recapture safety is claimed here.",
            {},
        )
        assert records == []

    def test_runtime_fires_recapture_safety_when_recap_true(self):
        warns = _runtime_warns(
            "The chosen move offers recapture safety over alternative [1].",
            _RECAP_TRUE_FACTS,
        )
        assert _contains_b1_contradiction(warns, "recapture safety")
        assert _contains_b1_contradiction(warns, "fabricated_claim")

    def test_runtime_clean_when_recap_false(self):
        warns = _runtime_warns(
            "The chosen move offers recapture safety over alternative [1].",
            _RECAP_FALSE_FACTS,
        )
        assert not _contains_b1_contradiction(
            warns, "recapture safety"
        ) or not any("fabricated_claim" in w and "recapture safety" in w for w in warns)

    def test_e1_parity_fires(self):
        text = "The chosen move provides recapture safety over [1]."
        facts = _RECAP_TRUE_FACTS
        rt = _runtime_warns(text, facts)
        assert_runtime_evaluator_agreement(rt, text, facts=facts)

    def test_e1_parity_clean(self):
        text = "The chosen move captures two pieces; [1] captures only one."
        facts = _RECAP_TRUE_FACTS
        rt = _runtime_warns(text, facts)
        assert_runtime_evaluator_agreement(rt, text, facts=facts)


# ─────────────────────────────────────────────────────────────────────────────
# B1.2 — Tradeoff language without numeric grounding
# ─────────────────────────────────────────────────────────────────────────────

_NEUTRAL_FACTS: dict = {"minimax_score": 14.0}


class TestOutweighsNumericGrounding:

    def test_evaluator_fires_outweighs_without_number(self):
        records = _check_outweighs_numeric_grounding(
            "the material gain outweighs the recapture risk."
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "outweighs"

    def test_evaluator_clean_outweighs_with_integer(self):
        records = _check_outweighs_numeric_grounding(
            "capturing 2 pieces outweighs the recapture risk."
        )
        assert records == []

    def test_evaluator_clean_outweighs_with_float(self):
        records = _check_outweighs_numeric_grounding(
            "the 28.0 point margin outweighs the alternative."
        )
        assert records == []

    def test_evaluator_fires_compensates_for_without_number(self):
        records = _check_outweighs_numeric_grounding(
            "the positional gain compensates for the isolation risk."
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "compensates for"

    def test_evaluator_clean_compensates_for_with_number(self):
        records = _check_outweighs_numeric_grounding(
            "capturing one piece compensates for the recapture exposure."
        )
        assert records == []

    def test_evaluator_fires_offsets_the_without_number(self):
        records = _check_outweighs_numeric_grounding(
            "the center placement offsets the isolation penalty."
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "offsets the"

    def test_evaluator_clean_offsets_the_with_number(self):
        records = _check_outweighs_numeric_grounding(
            "the 56.0 point engine edge offsets the isolation penalty."
        )
        assert records == []

    def test_evaluator_only_fires_once_for_multiple_sentences(self):
        text = (
            "Capturing 2 pieces outweighs the risk. "
            "The positional gain compensates for the weakness."
        )
        records = _check_outweighs_numeric_grounding(text)
        # First sentence has a number → clean; second has no number → fires once
        assert len(records) == 1
        assert records[0].matched_phrase == "compensates for"

    def test_runtime_fires_outweighs_without_number(self):
        warns = _runtime_warns(
            "The mobility gain outweighs the recapture risk.",
            _NEUTRAL_FACTS,
        )
        assert _contains_b1_contradiction(warns, "outweighs")
        assert _contains_b1_contradiction(warns, "tradeoff_without_evidence")

    def test_runtime_clean_outweighs_with_number(self):
        warns = _runtime_warns(
            "Capturing 2 pieces outweighs the recapture risk.",
            _NEUTRAL_FACTS,
        )
        assert not any(
            "outweighs" in w and "tradeoff_without_evidence" in w for w in warns
        )

    def test_e1_parity_fires(self):
        text = "The mobility gain outweighs the recapture risk."
        facts = _NEUTRAL_FACTS
        rt = _runtime_warns(text, facts)
        assert_runtime_evaluator_agreement(rt, text, facts=facts)

    def test_e1_parity_clean(self):
        text = "Capturing 2 pieces outweighs the recapture risk."
        facts = _NEUTRAL_FACTS
        rt = _runtime_warns(text, facts)
        assert_runtime_evaluator_agreement(rt, text, facts=facts)


# ─────────────────────────────────────────────────────────────────────────────
# B1.3 — Negative-score absolute advantage protection
# ─────────────────────────────────────────────────────────────────────────────

_NEG_SCORE_FACTS = {"minimax_score": -28.0, "opponent_can_recapture": True}
_POS_SCORE_FACTS = {"minimax_score": 28.0, "opponent_can_recapture": False}
_ZERO_SCORE_FACTS = {"minimax_score": 0.0}


class TestNegativeScoreAdvantage:

    def test_evaluator_fires_positional_advantage_neg_score(self):
        records = _check_negative_score_advantage(
            "this move secures a positional advantage over the opponent.",
            _NEG_SCORE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "positional advantage"

    def test_evaluator_fires_strongest_option_neg_score(self):
        records = _check_negative_score_advantage(
            "despite material loss, this remains the strongest option.",
            _NEG_SCORE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "strongest option"

    def test_evaluator_fires_decisive_advantage_neg_score(self):
        records = _check_negative_score_advantage(
            "this captures 2 pieces, creating a decisive advantage.",
            _NEG_SCORE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "decisive advantage"

    def test_evaluator_fires_advantage_gained_neg_score(self):
        records = _check_negative_score_advantage(
            "advantage gained by this move outpaces the opponent reply.",
            _NEG_SCORE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "advantage gained"

    def test_evaluator_no_fire_positive_score(self):
        records = _check_negative_score_advantage(
            "this move secures a positional advantage.",
            _POS_SCORE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_zero_score(self):
        records = _check_negative_score_advantage(
            "this is the strongest option.",
            _ZERO_SCORE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_best_available_framing(self):
        records = _check_negative_score_advantage(
            "this is the strongest option and best available in the position.",
            _NEG_SCORE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_least_unfavorable_framing(self):
        records = _check_negative_score_advantage(
            "this is the positional advantage least unfavorable among the candidates.",
            _NEG_SCORE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_least_harmful_framing(self):
        records = _check_negative_score_advantage(
            "the strongest option is the least harmful continuation.",
            _NEG_SCORE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_only_option_framing(self):
        records = _check_negative_score_advantage(
            "the strongest option is the only option left.",
            _NEG_SCORE_FACTS,
        )
        assert records == []

    def test_runtime_fires_strongest_option_neg_score(self):
        warns = _runtime_warns(
            "This remains the strongest option for the position.",
            _NEG_SCORE_FACTS,
        )
        assert _contains_b1_contradiction(warns, "strongest option")
        assert _contains_b1_contradiction(warns, "misleading_advantage_claim")

    def test_runtime_clean_positive_score(self):
        warns = _runtime_warns(
            "This remains the strongest option for the position.",
            _POS_SCORE_FACTS,
        )
        assert not any(
            "strongest option" in w and "misleading_advantage_claim" in w
            for w in warns
        )

    def test_t01_scenario_negative_score_with_strongest_option(self):
        # Reproduces T01 trace: minimax_score=-28.0, text claims "strongest option"
        text = (
            "Moving from 5,4 to 4,3 avoids the opponent's jump threat "
            "and represents the strongest option in this difficult position."
        )
        facts = {"minimax_score": -28.0, "opponent_can_recapture": False}
        records = _check_negative_score_advantage(text.lower(), facts)
        assert len(records) == 1
        assert records[0].matched_phrase == "strongest option"

    def test_e1_parity_fires(self):
        text = "This remains the strongest option for the position."
        facts = _NEG_SCORE_FACTS
        rt = _runtime_warns(text, facts)
        assert_runtime_evaluator_agreement(rt, text, facts=facts)

    def test_e1_parity_clean_with_relative_framing(self):
        text = "This is the best available option given the material deficit."
        facts = _NEG_SCORE_FACTS
        rt = _runtime_warns(text, facts)
        assert_runtime_evaluator_agreement(rt, text, facts=facts)


# ─────────────────────────────────────────────────────────────────────────────
# E.1 invariant — cross-check all three Phase B1 fixes
# ─────────────────────────────────────────────────────────────────────────────

class TestE1InvariantPhaseB1:

    def test_b11_e1_all_phrases(self):
        phrases = [
            "recapture safety", "avoiding recapture", "avoid recapture risk",
        ]
        facts = _RECAP_TRUE_FACTS
        for phrase in phrases:
            text = f"The chosen path provides {phrase} over move [1]."
            rt = _runtime_warns(text, facts)
            assert_runtime_evaluator_agreement(rt, text, facts=facts)

    def test_b12_e1_multiple_tradeoff_phrases(self):
        for phrase in ("outweighs", "compensates for"):
            text = f"The positional gain {phrase} the recapture risk."
            facts = _NEUTRAL_FACTS
            rt = _runtime_warns(text, facts)
            assert_runtime_evaluator_agreement(rt, text, facts=facts)

    def test_b13_e1_all_forbidden_phrases(self):
        phrases = [
            "positional advantage", "advantage gained",
            "strongest option", "decisive advantage",
        ]
        facts = _NEG_SCORE_FACTS
        for phrase in phrases:
            text = f"This move achieves {phrase} for the player."
            rt = _runtime_warns(text, facts)
            assert_runtime_evaluator_agreement(rt, text, facts=facts)

    def test_combined_clean_text_parity(self):
        text = (
            "Moving from 4,1 to 3,2 captures 2 pieces (opponent_can_recapture=true). "
            "The 28.0 point margin outweighs the recapture risk. "
            "Among all available options this is the best available continuation "
            "with a minimax score of -17.0."
        )
        facts = {"minimax_score": -17.0, "opponent_can_recapture": True}
        rt = _runtime_warns(text, facts)
        assert_runtime_evaluator_agreement(rt, text, facts=facts)
