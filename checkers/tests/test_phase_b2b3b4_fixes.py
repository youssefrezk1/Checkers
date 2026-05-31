# checkers/tests/test_phase_b2b3b4_fixes.py
#
# Regression tests for Phase B2+B3+B4 fixes:
#   B2.1b  Deliberate-choice framing forbidden in forced-move context
#   B2.3   Geometric impossibility phrases
#   B2.5   Our-mobility directional consistency
#   B2.6   Tactical move defensive framing
#
# Each class verifies both the runtime inline check (_check_reasoning_truthfulness)
# and the evaluator mirror (contradiction_strings / verify_all), plus E.1 parity.
# Note: B2.1a (first-sentence forced-move acknowledgment) was not implemented
# because it produces false positives when single-candidate seed lists are used
# in other test contexts.

from __future__ import annotations

import pytest

from checkers.agents.ranker_agent import _check_reasoning_truthfulness
from checkers.evaluation.unified_verifier import (
    _check_forced_move_framing,
    _check_geometric_impossibility,
    _check_our_mobility_direction,
    _check_tactical_move_framing,
    contradiction_strings,
    verify_all,
    assert_runtime_evaluator_agreement,
)
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_FORCED_SEEDS = ["This is the only legal move available; the engine assigns it a minimax score of 12.0."]
_NO_SEEDS: list = []


def _runtime_warns(text: str, facts: dict, seeds: list | None = None) -> list[str]:
    return _check_reasoning_truthfulness(text, facts, seeds=seeds or [])


def _evaluator_warns(text: str, facts: dict, seeds: list | None = None) -> list[str]:
    return contradiction_strings(text, reasoning_seeds=seeds or [], facts=facts)


def _contains(warnings: list[str], substring: str) -> bool:
    return any(substring in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# B2.1b — Deliberate-choice framing forbidden in forced-move context
# ─────────────────────────────────────────────────────────────────────────────

_FORCED_FACTS: dict = {"minimax_score": 12.0}

class TestDeliberateChoiceFraming:

    def test_evaluator_fires_drives_the_decision(self):
        records = _check_forced_move_framing(
            "this is the only legal move. the recapture risk drives the decision.",
            _FORCED_SEEDS,
        )
        deliberate = [r for r in records if r.claim_type == "forced_move_deliberate_framing"]
        assert len(deliberate) == 1
        assert deliberate[0].matched_phrase == "drives the decision"

    def test_evaluator_fires_chosen_for_its(self):
        records = _check_forced_move_framing(
            "this is the only legal move. the piece was chosen for its mobility.",
            _FORCED_SEEDS,
        )
        deliberate = [r for r in records if r.claim_type == "forced_move_deliberate_framing"]
        assert len(deliberate) == 1
        assert deliberate[0].matched_phrase == "chosen for its"

    def test_evaluator_fires_selected_for_its(self):
        records = _check_forced_move_framing(
            "this is the only legal move. the move was selected for its safety.",
            _FORCED_SEEDS,
        )
        deliberate = [r for r in records if r.claim_type == "forced_move_deliberate_framing"]
        assert len(deliberate) == 1
        assert deliberate[0].matched_phrase == "selected for its"

    def test_evaluator_fires_was_preferred_because(self):
        records = _check_forced_move_framing(
            "this is the only legal move. it was preferred because of its safety.",
            _FORCED_SEEDS,
        )
        deliberate = [r for r in records if r.claim_type == "forced_move_deliberate_framing"]
        assert len(deliberate) == 1
        assert deliberate[0].matched_phrase == "was preferred because"

    def test_evaluator_no_fire_without_forced_seed(self):
        records = _check_forced_move_framing(
            "the recapture risk drives the decision to move forward.",
            _NO_SEEDS,
        )
        assert records == []

    def test_runtime_fires_drives_the_decision_in_forced_context(self):
        warns = _runtime_warns(
            "This is the only legal move. The recapture risk drives the decision.",
            _FORCED_FACTS,
            seeds=_FORCED_SEEDS,
        )
        assert _contains(warns, "drives the decision")
        assert _contains(warns, "forced_move_deliberate_framing")

    def test_runtime_clean_deliberate_phrase_without_forced_seed(self):
        warns = _runtime_warns(
            "The recapture risk drives the decision to move forward.",
            _FORCED_FACTS,
            seeds=_NO_SEEDS,
        )
        assert not _contains(warns, "forced_move_deliberate_framing")

    def test_e1_parity_fires(self):
        text = "This is the only legal move. The recapture risk drives the decision."
        rt = _runtime_warns(text, _FORCED_FACTS, seeds=_FORCED_SEEDS)
        assert_runtime_evaluator_agreement(rt, text, reasoning_seeds=_FORCED_SEEDS, facts=_FORCED_FACTS)

    def test_e1_parity_clean(self):
        text = "This is the only legal move. The piece advances to row 4."
        rt = _runtime_warns(text, _FORCED_FACTS, seeds=_FORCED_SEEDS)
        assert_runtime_evaluator_agreement(rt, text, reasoning_seeds=_FORCED_SEEDS, facts=_FORCED_FACTS)


# ─────────────────────────────────────────────────────────────────────────────
# B2.3 — Geometric impossibility
# ─────────────────────────────────────────────────────────────────────────────

_ANY_FACTS: dict = {"minimax_score": 14.0}


class TestGeometricImpossibility:

    def test_evaluator_fires_piece_remains_stationary(self):
        records = _check_geometric_impossibility(
            "after this move, the piece remains stationary at row 3."
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "piece remains stationary"

    def test_evaluator_fires_no_piece_movement_occurred(self):
        records = _check_geometric_impossibility(
            "no piece movement occurred in this turn."
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "no piece movement occurred"

    def test_evaluator_fires_piece_did_not_move(self):
        records = _check_geometric_impossibility(
            "the piece did not move; it stayed in its original square."
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "piece did not move"

    def test_evaluator_no_fire_on_clean_text(self):
        records = _check_geometric_impossibility(
            "the piece advances forward from row 3 to row 4."
        )
        assert records == []

    def test_evaluator_no_fire_on_empty_text(self):
        assert _check_geometric_impossibility("") == []

    def test_runtime_fires_piece_remains_stationary(self):
        warns = _runtime_warns(
            "After this move, the piece remains stationary at row 3.",
            _ANY_FACTS,
        )
        assert _contains(warns, "piece remains stationary")
        assert _contains(warns, "geometric_impossibility")

    def test_runtime_fires_piece_did_not_move(self):
        warns = _runtime_warns(
            "The piece did not move; it stayed in its original square.",
            _ANY_FACTS,
        )
        assert _contains(warns, "piece did not move")
        assert _contains(warns, "geometric_impossibility")

    def test_runtime_clean_on_normal_move_text(self):
        warns = _runtime_warns(
            "The piece advances from row 3 to row 4 without capturing.",
            _ANY_FACTS,
        )
        assert not _contains(warns, "geometric_impossibility")

    def test_e1_parity_fires(self):
        text = "After this move, the piece remains stationary at row 3."
        rt = _runtime_warns(text, _ANY_FACTS)
        assert_runtime_evaluator_agreement(rt, text, facts=_ANY_FACTS)

    def test_e1_parity_clean(self):
        text = "The piece advances forward from row 3 to row 4."
        rt = _runtime_warns(text, _ANY_FACTS)
        assert_runtime_evaluator_agreement(rt, text, facts=_ANY_FACTS)


# ─────────────────────────────────────────────────────────────────────────────
# B2.5 — Our-mobility directional consistency
# ─────────────────────────────────────────────────────────────────────────────

_MOB_DECREASE_FACTS = {"our_mobility_before": 8, "our_mobility_after": 6, "minimax_score": 14.0}
_MOB_INCREASE_FACTS = {"our_mobility_before": 6, "our_mobility_after": 8, "minimax_score": 14.0}
_MOB_EQUAL_FACTS    = {"our_mobility_before": 7, "our_mobility_after": 7, "minimax_score": 14.0}


class TestOurMobilityDirection:

    def test_evaluator_fires_increases_our_mobility_when_decrease(self):
        records = _check_our_mobility_direction(
            "this move increases our mobility significantly.",
            _MOB_DECREASE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "increases our mobility"

    def test_evaluator_fires_our_mobility_improves_when_equal(self):
        records = _check_our_mobility_direction(
            "the move shows our mobility improves after the piece advances.",
            _MOB_EQUAL_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "our mobility improves"

    def test_evaluator_fires_our_mobility_grows_when_decrease(self):
        records = _check_our_mobility_direction(
            "our mobility grows from this positional move.",
            _MOB_DECREASE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "our mobility grows"

    def test_evaluator_clean_increases_our_mobility_when_actual_increase(self):
        records = _check_our_mobility_direction(
            "this move increases our mobility from 6 to 8.",
            _MOB_INCREASE_FACTS,
        )
        assert records == []

    def test_evaluator_clean_when_facts_missing(self):
        records = _check_our_mobility_direction(
            "this move increases our mobility.",
            {},
        )
        assert records == []

    def test_evaluator_clean_when_only_one_fact_present(self):
        records = _check_our_mobility_direction(
            "our mobility improves after this move.",
            {"our_mobility_before": 8},
        )
        assert records == []

    def test_runtime_fires_improves_our_mobility_when_decrease(self):
        warns = _runtime_warns(
            "This improves our mobility and strengthens our position.",
            _MOB_DECREASE_FACTS,
        )
        assert _contains(warns, "our_mobility_direction")

    def test_runtime_clean_when_actual_increase(self):
        warns = _runtime_warns(
            "This move increases our mobility from 6 to 8.",
            _MOB_INCREASE_FACTS,
        )
        assert not _contains(warns, "our_mobility_direction")

    def test_e1_parity_fires(self):
        text = "This move increases our mobility significantly."
        rt = _runtime_warns(text, _MOB_DECREASE_FACTS)
        assert_runtime_evaluator_agreement(rt, text, facts=_MOB_DECREASE_FACTS)

    def test_e1_parity_clean_actual_increase(self):
        text = "This move increases our mobility from 6 to 8."
        rt = _runtime_warns(text, _MOB_INCREASE_FACTS)
        assert_runtime_evaluator_agreement(rt, text, facts=_MOB_INCREASE_FACTS)

    def test_e1_parity_clean_no_facts(self):
        text = "This move increases our mobility significantly."
        rt = _runtime_warns(text, {})
        assert_runtime_evaluator_agreement(rt, text, facts={})


# ─────────────────────────────────────────────────────────────────────────────
# B2.6 — Tactical move defensive framing
# ─────────────────────────────────────────────────────────────────────────────

_THREAT_TRUE_FACTS  = {"creates_immediate_threat": True,  "minimax_score": 18.0}
_THREAT_FALSE_FACTS = {"creates_immediate_threat": False, "minimax_score": 18.0}


class TestTacticalMoveDefensiveFraming:

    def test_evaluator_fires_no_tactical_pressure_when_threat_true(self):
        records = _check_tactical_move_framing(
            "this move advances the piece with no tactical pressure on the opponent.",
            _THREAT_TRUE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "no tactical pressure"

    def test_evaluator_fires_applies_no_pressure_when_threat_true(self):
        records = _check_tactical_move_framing(
            "the chosen move applies no pressure and simply repositions the piece.",
            _THREAT_TRUE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "applies no pressure"

    def test_evaluator_fires_creates_no_pressure_when_threat_true(self):
        records = _check_tactical_move_framing(
            "this creates no pressure while improving piece placement.",
            _THREAT_TRUE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "creates no pressure"

    def test_evaluator_fires_no_immediate_pressure_when_threat_true(self):
        records = _check_tactical_move_framing(
            "the move offers no immediate pressure on the opponent.",
            _THREAT_TRUE_FACTS,
        )
        assert len(records) == 1
        assert records[0].matched_phrase == "no immediate pressure"

    def test_evaluator_no_fire_when_threat_false(self):
        records = _check_tactical_move_framing(
            "this move applies no pressure and simply repositions.",
            _THREAT_FALSE_FACTS,
        )
        assert records == []

    def test_evaluator_no_fire_when_creates_immediate_threat_absent(self):
        records = _check_tactical_move_framing(
            "this move applies no pressure.",
            {},
        )
        assert records == []

    def test_evaluator_no_fire_on_clean_tactical_text(self):
        records = _check_tactical_move_framing(
            "this move forces the opponent to respond to an immediate threat.",
            _THREAT_TRUE_FACTS,
        )
        assert records == []

    def test_runtime_fires_no_tactical_pressure_when_threat_true(self):
        warns = _runtime_warns(
            "The move advances the piece with no tactical pressure.",
            _THREAT_TRUE_FACTS,
        )
        assert _contains(warns, "no tactical pressure")
        assert _contains(warns, "tactical_move_defensive_framing")

    def test_runtime_clean_when_threat_false(self):
        warns = _runtime_warns(
            "The move advances the piece with no tactical pressure.",
            _THREAT_FALSE_FACTS,
        )
        assert not _contains(warns, "tactical_move_defensive_framing")

    def test_e1_parity_fires(self):
        text = "The move advances the piece with no tactical pressure."
        rt = _runtime_warns(text, _THREAT_TRUE_FACTS)
        assert_runtime_evaluator_agreement(rt, text, facts=_THREAT_TRUE_FACTS)

    def test_e1_parity_clean(self):
        text = "This move forces the opponent to respond to an immediate threat."
        rt = _runtime_warns(text, _THREAT_TRUE_FACTS)
        assert_runtime_evaluator_agreement(rt, text, facts=_THREAT_TRUE_FACTS)


# ─────────────────────────────────────────────────────────────────────────────
# E.1 invariant — cross-check all Phase B2+B3+B4 fixes
# ─────────────────────────────────────────────────────────────────────────────

class TestE1InvariantPhaseB2B3B4:

    def test_b21b_e1_all_deliberate_phrases(self):
        for phrase in ("drives the decision", "chosen for its", "was chosen for"):
            text = f"This is the only legal move. The move {phrase} safety value."
            rt = _runtime_warns(text, _FORCED_FACTS, seeds=_FORCED_SEEDS)
            assert_runtime_evaluator_agreement(rt, text, reasoning_seeds=_FORCED_SEEDS, facts=_FORCED_FACTS)

    def test_b23_e1_all_geo_phrases(self):
        for phrase in ("piece remains stationary", "no piece movement occurred", "piece did not move"):
            text = f"After this turn, the {phrase} on the board."
            rt = _runtime_warns(text, _ANY_FACTS)
            assert_runtime_evaluator_agreement(rt, text, facts=_ANY_FACTS)

    def test_b25_e1_all_increase_phrases(self):
        for phrase in ("increases our mobility", "improves our mobility", "our mobility grows"):
            text = f"This move {phrase} going forward."
            rt = _runtime_warns(text, _MOB_DECREASE_FACTS)
            assert_runtime_evaluator_agreement(rt, text, facts=_MOB_DECREASE_FACTS)

    def test_b26_e1_all_defensive_phrases(self):
        for phrase in ("no tactical pressure", "applies no pressure", "creates no pressure"):
            text = f"The move offers {phrase} on the opponent."
            rt = _runtime_warns(text, _THREAT_TRUE_FACTS)
            assert_runtime_evaluator_agreement(rt, text, facts=_THREAT_TRUE_FACTS)

    def test_combined_clean_text_parity(self):
        text = (
            "This is the only legal move available. "
            "The piece advances from row 3 to row 4, "
            "which forces the opponent to respond to an immediate threat. "
            "Our mobility changes from 7 to 7 — no change in our mobility. "
            "The engine scores this move 12.0 — slightly ahead."
        )
        facts = {
            "minimax_score": 12.0,
            "creates_immediate_threat": True,
            "our_mobility_before": 7,
            "our_mobility_after": 7,
        }
        seeds = _FORCED_SEEDS
        rt = _runtime_warns(text, facts, seeds=seeds)
        assert_runtime_evaluator_agreement(rt, text, reasoning_seeds=seeds, facts=facts)
