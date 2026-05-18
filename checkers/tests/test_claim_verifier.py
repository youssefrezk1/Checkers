# checkers/tests/test_claim_verifier.py
#
# Tests for checkers/evaluation/claim_verifier.py
#
# PURPOSE
# -------
# Verify that verify_claims() correctly applies deterministic symbolic
# fact-checking rules to extracted ClaimRecord objects.
#
# Coverage:
#   1. Isolation — no runtime pipeline imports.
#   2. Correct SUPPORTED verdict for each verifiable claim type.
#   3. Correct CONTRADICTED verdict (contradiction examples from spec).
#   4. Correct UNSUPPORTED verdict when fact absent or neutral.
#   5. VAGUE assignment for always-unverifiable strategic claims.
#   6. Hallucination annotation (FACTUAL_CONTRADICTION on CONTRADICTED;
#      OVERCLAIM on VAGUE strategic claims).
#   7. Input immutability — original list and records are not mutated.
#   8. Empty inputs handled gracefully.
#   9. Unknown claim type is returned unchanged.
#  10. No-facts (empty dict / None) leaves claims unchanged.

import sys
from dataclasses import replace
from typing import Optional

import pytest

# ── Isolation guard ───────────────────────────────────────────────────────────
_modules_before = set(sys.modules.keys())

from checkers.evaluation.claim_verifier import verify_claims, _VERIFICATION_RULES
from checkers.evaluation.claim_extractor import ClaimRecord, extract_claims
from checkers.evaluation.reasoning_taxonomy import (
    ClaimStatus,
    ClaimVerifiability,
    HallucinationType,
    SeedRiskType,
)

_modules_after = set(sys.modules.keys())

_FORBIDDEN_RUNTIME_PREFIXES = (
    "checkers.engine",
    "checkers.agents",
    "checkers.graph",
    "checkers.state",
    "checkers.nodes",
    "checkers.search",
)


def test_no_runtime_pipeline_imports():
    new_modules = _modules_after - _modules_before
    for mod in new_modules:
        for prefix in _FORBIDDEN_RUNTIME_PREFIXES:
            assert not mod.startswith(prefix), (
                f"claim_verifier pulled in runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(
    claim_type: str,
    status: ClaimStatus = ClaimStatus.NOT_CHECKED,
    verifiability: ClaimVerifiability = ClaimVerifiability.FULLY_VERIFIABLE,
    hallucination: Optional[HallucinationType] = None,
) -> ClaimRecord:
    return ClaimRecord(
        claim_type=claim_type,
        claim_status=status,
        claim_verifiability=verifiability,
        hallucination_type=hallucination,
        source="unsupported_phrase",
    )


def _verify_one(claim_type: str, facts: dict) -> ClaimRecord:
    """Convenience: verify a single claim and return the updated record."""
    claim = _make_claim(claim_type)
    result = verify_claims([claim], facts)
    assert len(result) == 1
    return result[0]


# ---------------------------------------------------------------------------
# avoids_recapture
# ---------------------------------------------------------------------------

class TestAvoidsRecapture:

    def test_supported_when_cannot_recapture(self):
        rec = _verify_one("avoids_recapture", {"opponent_can_recapture": False})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_can_recapture(self):
        """avoids_recapture + opponent_can_recapture=True → CONTRADICTED"""
        rec = _verify_one("avoids_recapture", {"opponent_can_recapture": True})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("avoids_recapture", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# can_be_recaptured
# ---------------------------------------------------------------------------

class TestCanBeRecaptured:

    def test_supported_when_can_recapture(self):
        rec = _verify_one("can_be_recaptured", {"opponent_can_recapture": True})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_cannot_recapture(self):
        """can_be_recaptured + opponent_can_recapture=False → CONTRADICTED"""
        rec = _verify_one("can_be_recaptured", {"opponent_can_recapture": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("can_be_recaptured", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# gains_material
# ---------------------------------------------------------------------------

class TestGainsMaterial:

    def test_supported_when_net_gain_positive(self):
        rec = _verify_one("gains_material", {"net_gain": 2})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_supported_when_net_gain_one(self):
        rec = _verify_one("gains_material", {"net_gain": 1})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_net_gain_zero(self):
        """gains_material + net_gain=0 → CONTRADICTED"""
        rec = _verify_one("gains_material", {"net_gain": 0})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_contradicted_when_net_gain_negative(self):
        """gains_material + net_gain<0 → CONTRADICTED (material loss)"""
        rec = _verify_one("gains_material", {"net_gain": -1})
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("gains_material", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# promotes_to_king
# ---------------------------------------------------------------------------

class TestPromotesToKing:

    def test_supported_when_promotes(self):
        rec = _verify_one("promotes_to_king", {"results_in_king": True})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_no_promotion(self):
        """promotes_to_king + results_in_king=False → CONTRADICTED"""
        rec = _verify_one("promotes_to_king", {"results_in_king": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("promotes_to_king", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# near_promotion
# ---------------------------------------------------------------------------

class TestNearPromotion:

    def test_supported_when_near_and_not_promoted(self):
        rec = _verify_one("near_promotion", {
            "near_promotion": True,
            "results_in_king": False,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_neither_near_nor_promoted(self):
        rec = _verify_one("near_promotion", {
            "near_promotion": False,
            "results_in_king": False,
        })
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_unsupported_when_promoted_instead(self):
        """Piece promoted outright, not near — claim is unsupported not contradicted."""
        rec = _verify_one("near_promotion", {
            "near_promotion": False,
            "results_in_king": True,
        })
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("near_promotion", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# opponent_near_promotion  (entity-aware verifier)
# ---------------------------------------------------------------------------

class TestOpponentNearPromotion:

    def test_supported_when_opponent_near_promotion_true(self):
        rec = _verify_one("opponent_near_promotion", {"opponent_near_promotion": True})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_opponent_near_promotion_false(self):
        rec = _verify_one("opponent_near_promotion", {"opponent_near_promotion": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("opponent_near_promotion", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_our_near_promotion_false_does_not_block_support(self):
        """near_promotion=False on our piece must not affect opponent_near_promotion."""
        rec = _verify_one("opponent_near_promotion", {
            "opponent_near_promotion": True,
            "near_promotion": False,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_results_in_king_irrelevant(self):
        """results_in_king has no bearing on opponent_near_promotion verification."""
        rec = _verify_one("opponent_near_promotion", {
            "opponent_near_promotion": True,
            "results_in_king": True,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED


# ---------------------------------------------------------------------------
# near_promotion + opponent_near_promotion — entity-aware end-to-end
# ---------------------------------------------------------------------------

class TestNearPromotionEntityAwareness:
    """
    End-to-end tests: extract_claims → verify_claims.
    Verifies entity-context routing for near_promotion phrases.
    """

    @staticmethod
    def _run(reasoning: str, seeds: list, facts: dict) -> list:
        from checkers.evaluation.claim_extractor import extract_claims
        from checkers.evaluation.claim_verifier  import verify_claims
        return verify_claims(extract_claims(reasoning, reasoning_seeds=seeds, facts=facts), facts)

    _OPP_SEED = (
        "opponent_near_promotion=true — at least one opponent piece "
        "is one step from promotion"
    )
    _OUR_SEED = "near_promotion=true — piece is one step from the back rank"

    def test_opponent_context_opp_true_is_supported(self):
        """'opponent has a piece one step from promotion' + opponent_near_promotion=True → SUPPORTED."""
        claims = self._run(
            "Although the opponent has a piece one step from promotion, we advance safely.",
            seeds=[self._OPP_SEED],
            facts={"near_promotion": False, "opponent_near_promotion": True, "results_in_king": False},
        )
        opp = [c for c in claims if c.claim_type == "opponent_near_promotion"]
        assert len(opp) == 1
        assert opp[0].claim_status == ClaimStatus.SUPPORTED

    def test_opponent_context_opp_false_is_not_supported(self):
        """'opponent has a piece one step from promotion' + opponent_near_promotion=False → CONTRADICTED."""
        claims = self._run(
            "Although the opponent has a piece one step from promotion, we advance.",
            seeds=[],
            facts={"near_promotion": False, "opponent_near_promotion": False, "results_in_king": False},
        )
        opp = [c for c in claims if c.claim_type == "opponent_near_promotion"]
        assert len(opp) == 1
        assert opp[0].claim_status != ClaimStatus.SUPPORTED

    def test_our_piece_context_uses_near_promotion_fact(self):
        """'our piece is one step from promotion' + near_promotion=True → SUPPORTED (near_promotion type)."""
        claims = self._run(
            "Our piece is one step from promotion on the back rank.",
            seeds=[self._OUR_SEED],
            facts={"near_promotion": True, "opponent_near_promotion": False, "results_in_king": False},
        )
        ours = [c for c in claims if c.claim_type == "near_promotion"]
        assert len(ours) == 1
        assert ours[0].claim_status == ClaimStatus.SUPPORTED

    def test_our_piece_false_plus_opponent_true_not_rescued(self):
        """Claim about OUR piece near promotion + near_promotion=False → CONTRADICTED.
        opponent_near_promotion=True must NOT rescue it."""
        claims = self._run(
            "Our piece is near promotion.",
            seeds=[],
            facts={"near_promotion": False, "opponent_near_promotion": True, "results_in_king": False},
        )
        ours = [c for c in claims if c.claim_type == "near_promotion"]
        assert len(ours) == 1
        assert ours[0].claim_status != ClaimStatus.SUPPORTED

    def test_ambiguous_phrase_not_auto_supported_from_opponent_seed(self):
        """Ambiguous 'one step from promotion' with no entity context → stays near_promotion.
        opponent_near_promotion=True must NOT auto-support it."""
        claims = self._run(
            "The piece is one step from promotion.",
            seeds=[self._OPP_SEED],
            facts={"near_promotion": False, "opponent_near_promotion": True, "results_in_king": False},
        )
        ours = [c for c in claims if c.claim_type == "near_promotion"]
        assert len(ours) == 1
        assert ours[0].claim_status != ClaimStatus.SUPPORTED

    def test_exact_t23_four_supported_zero_contradicted(self):
        """Exact t23 turn: 4 supported, 0 contradicted."""
        t23_reasoning = (
            "The move advances a piece forward to column 4 in the center while "
            "increasing our mobility from 7 to 9, directly addressing the opponent’s "
            "structural disadvantage where they hold 11 mobility options. Although the "
            "opponent has a piece one step from promotion, they cannot recapture this "
            "piece next turn, and our pieces remain unthreatened after the move. The "
            "move does not isolate the advanced piece and leaves our piece coordination "
            "intact, despite the opponent’s mobility remaining unchanged at 11. "
            "With no captures and no immediate tactical threats created, the move "
            "focuses on improving positional placement. The minimax_score confirms "
            "this as the best available option at -54.00."
        )
        t23_seeds = [
            "opponent_near_promotion=true — at least one opponent piece is one step from promotion",
            "opponent_can_recapture=false — immediate tactical safety: opponent cannot recapture this piece next turn",
            "leaves_piece_isolated=false — preserves piece coordination by keeping the moved piece connected",
            "minimax_score=-54.00 — best available option in a difficult position",
        ]
        t23_facts = {
            "near_promotion": False,
            "opponent_near_promotion": True,
            "results_in_king": False,
            "opponent_can_recapture": False,
            "leaves_piece_isolated": False,
            "minimax_score": -54.0,
        }
        claims = self._run(t23_reasoning, t23_seeds, t23_facts)
        contradicted = [c for c in claims if c.claim_status == ClaimStatus.CONTRADICTED]
        supported    = [c for c in claims if c.claim_status == ClaimStatus.SUPPORTED]
        assert contradicted == [], (
            f"Expected 0 contradictions, got: "
            f"{[(c.claim_type, c.matched_phrase) for c in contradicted]}"
        )
        assert len(supported) == 4, f"Expected 4 supported, got {len(supported)}"


# ---------------------------------------------------------------------------
# mobility_decrease
# ---------------------------------------------------------------------------

class TestMobilityDecrease:

    def test_supported_when_reduction_positive(self):
        rec = _verify_one("mobility_decrease", {"mobility_reduction": 3})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_reduction_negative(self):
        rec = _verify_one("mobility_decrease", {"mobility_reduction": -2})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_reduction_zero(self):
        """Zero reduction: claim overstated but not directly contradicted."""
        rec = _verify_one("mobility_decrease", {"mobility_reduction": 0})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("mobility_decrease", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    # Self-mobility path (our_mobility_before/after)

    def test_supported_when_our_mobility_drops(self):
        rec = _verify_one("mobility_decrease", {
            "our_mobility_before": 8,
            "our_mobility_after": 5,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_our_mobility_rises(self):
        rec = _verify_one("mobility_decrease", {
            "our_mobility_before": 5,
            "our_mobility_after": 8,
        })
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_unsupported_when_our_mobility_unchanged(self):
        rec = _verify_one("mobility_decrease", {
            "our_mobility_before": 6,
            "our_mobility_after": 6,
        })
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_opponent_path_takes_priority_over_self(self):
        # mobility_reduction present — opponent path wins regardless of self facts.
        rec = _verify_one("mobility_decrease", {
            "mobility_reduction": 3,
            "our_mobility_before": 8,
            "our_mobility_after": 9,  # self mobility rises (contradicts self-path)
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED  # opponent path dominates


# ---------------------------------------------------------------------------
# mobility_decrease — end-to-end (extract + verify)
# ---------------------------------------------------------------------------

class TestMobilityDecreaseEndToEnd:
    """
    Integration tests: text → extract_claims → verify_claims.
    Verifies that self-mobility phrasings round-trip correctly.
    """

    @staticmethod
    def _pipeline(text: str, facts: dict, seeds: Optional[list] = None) -> list:
        records = extract_claims(text, reasoning_seeds=seeds or [])
        decrease = [r for r in records if r.claim_type == "mobility_decrease"]
        return verify_claims(decrease, facts)

    def test_reduces_our_mobility_by_2_is_supported(self):
        # Scenario 1: explicit numeric reduction, facts confirm drop 9→7.
        verified = self._pipeline(
            "This move reduces our mobility by 2, narrowing our options.",
            facts={"our_mobility_before": 9, "our_mobility_after": 7},
        )
        assert len(verified) >= 1
        assert verified[0].claim_status == ClaimStatus.SUPPORTED

    def test_our_mobility_decreases_from_9_to_7_is_supported(self):
        # Scenario 2: "mobility decreases from X to Y" phrasing, facts confirm.
        verified = self._pipeline(
            "Our mobility decreases from 9 to 7 after this move.",
            facts={"our_mobility_before": 9, "our_mobility_after": 7},
        )
        assert len(verified) >= 1
        assert verified[0].claim_status == ClaimStatus.SUPPORTED

    def test_reduces_opponent_mobility_opponent_path_still_supported(self):
        # Scenario 3: opponent path unaffected — mobility_reduction=1 drives it.
        verified = self._pipeline(
            "This reduces opponent mobility, limiting their replies.",
            facts={"mobility_reduction": 1},
        )
        assert len(verified) >= 1
        assert verified[0].claim_status == ClaimStatus.SUPPORTED

    def test_reduces_our_mobility_but_facts_show_rise_is_contradicted(self):
        # Scenario 4: claim says decrease, facts show our mobility rose 7→9.
        verified = self._pipeline(
            "This reduces our mobility by 2.",
            facts={"our_mobility_before": 7, "our_mobility_after": 9},
        )
        assert len(verified) >= 1
        assert verified[0].claim_status == ClaimStatus.CONTRADICTED

    def test_mobility_increase_verifier_unaffected(self):
        # Scenario 5: mobility_increase verification untouched by these changes.
        rec = _verify_one("mobility_increase", {
            "our_mobility_before": 5,
            "our_mobility_after": 8,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED


# ---------------------------------------------------------------------------
# mobility_decrease — production-realistic (full engine facts, mobility_reduction=0)
# ---------------------------------------------------------------------------

class TestMobilityDecreaseProductionFacts:
    """
    Tests using complete engine fact dicts (always include mobility_reduction).
    These reproduce the exact conditions seen in manual trace T15 and T27
    where mobility_reduction=0 (opponent mobility unchanged) previously
    caused an early-return UNSUPPORTED before reaching our_mobility_before/after.
    """

    @staticmethod
    def _full_pipeline(text: str, facts: dict, seeds: Optional[list] = None):
        """Run full extract + verify, return all claims."""
        records = extract_claims(text, reasoning_seeds=seeds or [])
        return verify_claims(records, facts)

    def test_t15_scenario_our_mobility_decreases_with_mobility_reduction_zero(self):
        # T15: our_mobility_before=11, our_mobility_after=9, mobility_reduction=0.
        # Old code: mobility_reduction=0 → early UNSUPPORTED. Fixed: fall-through → SUPPORTED.
        rec = _verify_one("mobility_decrease", {
            "our_mobility_before": 11,
            "our_mobility_after": 9,
            "mobility_reduction": 0,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_t15_scenario_reducing_our_mobility_by_no_spurious_increase(self):
        # T15 reasoning: "reducing our mobility by two squares"
        # Bug 2: "reducing" (not "reduces") must now redirect to mobility_decrease.
        # Expect: mobility_decrease SUPPORTED, no mobility_increase CONTRADICTED.
        t15_facts = {
            "our_mobility_before": 11,
            "our_mobility_after": 9,
            "mobility_reduction": 0,
        }
        t15_seeds = [
            "our_mobility_before=11, our_mobility_after=9 — decreases our mobility by 2",
        ]
        claims = self._full_pipeline(
            "The move advances a piece to a structurally restricted position, "
            "reducing our mobility by two squares.",
            facts=t15_facts,
            seeds=t15_seeds,
        )
        types_statuses = {c.claim_type: c.claim_status for c in claims}
        assert "mobility_decrease" in types_statuses
        assert types_statuses["mobility_decrease"] == ClaimStatus.SUPPORTED
        assert "mobility_increase" not in types_statuses

    def test_t27_scenario_our_mobility_decreases_slightly(self):
        # T27: our_mobility_before=10, our_mobility_after=9, mobility_reduction=0.
        rec = _verify_one("mobility_decrease", {
            "our_mobility_before": 10,
            "our_mobility_after": 9,
            "mobility_reduction": 0,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_opponent_path_still_supported_when_mobility_reduction_positive(self):
        # Opponent mobility path must not be broken by the val==0 fall-through change.
        rec = _verify_one("mobility_decrease", {
            "our_mobility_before": 8,
            "our_mobility_after": 9,   # our mobility rises — self-path would CONTRADICT
            "mobility_reduction": 3,   # opponent path dominates → SUPPORTED
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_our_mobility_rises_with_mobility_reduction_zero_is_contradicted(self):
        # Claim says decrease; facts show our mobility rose; opponent unchanged.
        rec = _verify_one("mobility_decrease", {
            "our_mobility_before": 7,
            "our_mobility_after": 9,
            "mobility_reduction": 0,
        })
        assert rec.claim_status == ClaimStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# mobility_increase
# ---------------------------------------------------------------------------

class TestMobilityIncrease:

    def test_supported_when_after_greater_than_before(self):
        rec = _verify_one("mobility_increase", {
            "our_mobility_before": 5,
            "our_mobility_after": 8,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_after_less_than_before(self):
        rec = _verify_one("mobility_increase", {
            "our_mobility_before": 8,
            "our_mobility_after": 5,
        })
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_unsupported_when_equal(self):
        rec = _verify_one("mobility_increase", {
            "our_mobility_before": 6,
            "our_mobility_after": 6,
        })
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_before_absent(self):
        rec = _verify_one("mobility_increase", {"our_mobility_after": 7})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_both_absent(self):
        rec = _verify_one("mobility_increase", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# piece_isolated
# ---------------------------------------------------------------------------

class TestPieceIsolated:

    def test_supported_when_isolated(self):
        rec = _verify_one("piece_isolated", {"leaves_piece_isolated": True})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_not_isolated(self):
        rec = _verify_one("piece_isolated", {"leaves_piece_isolated": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_absent(self):
        rec = _verify_one("piece_isolated", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# piece_connected
# ---------------------------------------------------------------------------

class TestPieceConnected:

    def test_supported_when_not_isolated(self):
        rec = _verify_one("piece_connected", {"leaves_piece_isolated": False})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_isolated(self):
        rec = _verify_one("piece_connected", {"leaves_piece_isolated": True})
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_absent(self):
        rec = _verify_one("piece_connected", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# creates_immediate_threat
# ---------------------------------------------------------------------------

class TestCreatesImmediateThreat:

    def test_supported_via_creates_immediate_threat(self):
        rec = _verify_one("creates_immediate_threat", {
            "creates_immediate_threat": True,
            "shot_sequence_available": False,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_supported_via_shot_sequence_available(self):
        rec = _verify_one("creates_immediate_threat", {
            "creates_immediate_threat": False,
            "shot_sequence_available": True,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_supported_when_both_true(self):
        rec = _verify_one("creates_immediate_threat", {
            "creates_immediate_threat": True,
            "shot_sequence_available": True,
        })
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_both_false(self):
        rec = _verify_one("creates_immediate_threat", {
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
        })
        assert rec.claim_status == ClaimStatus.CONTRADICTED
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_unsupported_when_both_absent(self):
        rec = _verify_one("creates_immediate_threat", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_only_one_present_and_false(self):
        """One fact present and False, other absent → conservative UNSUPPORTED."""
        rec = _verify_one("creates_immediate_threat", {
            "creates_immediate_threat": False,
            # shot_sequence_available absent
        })
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# minimax_confirmation
# ---------------------------------------------------------------------------

class TestMinimaxConfirmation:

    def test_supported_when_score_present(self):
        rec = _verify_one("minimax_confirmation", {"minimax_score": 4.5})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_supported_when_score_is_zero(self):
        rec = _verify_one("minimax_confirmation", {"minimax_score": 0.0})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_supported_when_score_is_integer(self):
        rec = _verify_one("minimax_confirmation", {"minimax_score": 3})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_unsupported_when_absent(self):
        rec = _verify_one("minimax_confirmation", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_score_is_none(self):
        rec = _verify_one("minimax_confirmation", {"minimax_score": None})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# Always-VAGUE strategic claims
# ---------------------------------------------------------------------------

class TestAlwaysVagueStrategicClaims:

    @pytest.mark.parametrize("claim_type", [
        "positional_pressure",
        "strategic_initiative",
        "long_term_compensation",
    ])
    def test_always_vague_regardless_of_facts(self, claim_type):
        """Strategic claims must NEVER become SUPPORTED."""
        # Even with a rich facts dict — no facts can support these.
        facts = {
            "opponent_can_recapture": False,
            "net_gain": 2,
            "results_in_king": True,
            "creates_immediate_threat": True,
            "mobility_reduction": 5,
            "minimax_score": 9.9,
        }
        rec = _verify_one(claim_type, facts)
        assert rec.claim_status == ClaimStatus.VAGUE, (
            f"{claim_type} must be VAGUE, got {rec.claim_status}"
        )
        assert rec.claim_status != ClaimStatus.SUPPORTED

    @pytest.mark.parametrize("claim_type", [
        "positional_pressure",
        "strategic_initiative",
        "long_term_compensation",
    ])
    def test_always_vague_with_empty_facts(self, claim_type):
        rec = _verify_one(claim_type, {})
        assert rec.claim_status == ClaimStatus.VAGUE

    @pytest.mark.parametrize("claim_type", [
        "positional_pressure",
        "strategic_initiative",
        "long_term_compensation",
    ])
    def test_overclaim_hallucination_on_vague(self, claim_type):
        """VAGUE strategic claims should receive OVERCLAIM hallucination type."""
        claim = _make_claim(claim_type, hallucination=None)
        result = verify_claims([claim], {"minimax_score": 5.0})
        assert result[0].hallucination_type == HallucinationType.OVERCLAIM


# ---------------------------------------------------------------------------
# Hallucination annotation rules
# ---------------------------------------------------------------------------

class TestHallucinationAnnotation:

    def test_contradicted_receives_factual_contradiction(self):
        """A claim moving to CONTRADICTED must have FACTUAL_CONTRADICTION."""
        rec = _verify_one("avoids_recapture", {"opponent_can_recapture": True})
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_supported_clears_fabricated_claim_annotation(self):
        """If a claim was pre-labelled FABRICATED_CLAIM but facts now SUPPORT it,
        the new status is SUPPORTED but hallucination annotation should remain as
        set by verify (i.e. _upgrade_hallucination returns None for non-CONTRADICTED
        non-VAGUE statuses)."""
        claim = _make_claim(
            "avoids_recapture",
            hallucination=HallucinationType.FABRICATED_CLAIM,
        )
        result = verify_claims([claim], {"opponent_can_recapture": False})
        rec = result[0]
        assert rec.claim_status == ClaimStatus.SUPPORTED
        # hallucination_type is None for supported claims
        assert rec.hallucination_type is None

    def test_unsupported_preserves_existing_hallucination(self):
        """UNSUPPORTED outcome preserves existing hallucination annotation."""
        claim = _make_claim(
            "avoids_recapture",
            hallucination=HallucinationType.FABRICATED_CLAIM,
        )
        result = verify_claims([claim], {})  # no facts → UNSUPPORTED
        rec = result[0]
        assert rec.claim_status == ClaimStatus.UNSUPPORTED
        # _upgrade_hallucination returns existing annotation for UNSUPPORTED
        assert rec.hallucination_type == HallucinationType.FABRICATED_CLAIM


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_original_list_not_mutated(self):
        claim = _make_claim("avoids_recapture")
        original_list = [claim]
        original_status = claim.claim_status
        _ = verify_claims(original_list, {"opponent_can_recapture": False})
        # Original list still contains original object
        assert original_list[0] is claim
        assert original_list[0].claim_status == original_status

    def test_original_claim_record_not_mutated(self):
        claim = _make_claim("gains_material")
        _ = verify_claims([claim], {"net_gain": 2})
        assert claim.claim_status == ClaimStatus.NOT_CHECKED

    def test_original_facts_dict_not_mutated(self):
        claim = _make_claim("avoids_recapture")
        facts = {"opponent_can_recapture": False}
        _ = verify_claims([claim], facts)
        assert facts == {"opponent_can_recapture": False}

    def test_output_is_independent_list(self):
        claim = _make_claim("avoids_recapture")
        result = verify_claims([claim], {"opponent_can_recapture": False})
        assert result is not [claim]
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_claims_returns_empty(self):
        result = verify_claims([], {"opponent_can_recapture": False})
        assert result == []

    def test_empty_facts_leaves_claims_unchanged(self):
        """Empty facts dict → conservative: return same statuses unchanged."""
        claim = _make_claim("avoids_recapture", status=ClaimStatus.SUPPORTED)
        result = verify_claims([claim], {})
        # UNSUPPORTED because fact absent (conservative rule)
        assert result[0].claim_status == ClaimStatus.UNSUPPORTED

    def test_none_facts_returns_unchanged(self):
        claim = _make_claim("avoids_recapture", status=ClaimStatus.SUPPORTED)
        result = verify_claims([claim], None)
        assert result[0].claim_status == ClaimStatus.SUPPORTED

    def test_unknown_claim_type_returned_unchanged(self):
        """claim_type not in the dispatch table → return unchanged."""
        claim = _make_claim("completely_unknown_claim_type")
        original_status = claim.claim_status
        result = verify_claims([claim], {"opponent_can_recapture": False})
        assert result[0].claim_status == original_status
        assert result[0].claim_type == "completely_unknown_claim_type"

    def test_multiple_claims_independently_verified(self):
        """Each claim in a list is verified independently."""
        c1 = _make_claim("avoids_recapture")
        c2 = _make_claim("gains_material")
        c3 = _make_claim("promotes_to_king")
        facts = {
            "opponent_can_recapture": False,  # avoids_recapture → SUPPORTED
            "net_gain": 0,                    # gains_material → CONTRADICTED
            # results_in_king absent          # promotes_to_king → UNSUPPORTED
        }
        result = verify_claims([c1, c2, c3], facts)
        assert result[0].claim_status == ClaimStatus.SUPPORTED      # avoids_recapture
        assert result[1].claim_status == ClaimStatus.CONTRADICTED   # gains_material
        assert result[2].claim_status == ClaimStatus.UNSUPPORTED    # promotes_to_king

    def test_order_preserved(self):
        claims = [
            _make_claim("avoids_recapture"),
            _make_claim("gains_material"),
            _make_claim("positional_pressure"),
        ]
        facts = {"opponent_can_recapture": False, "net_gain": 1}
        result = verify_claims(claims, facts)
        assert len(result) == 3
        assert result[0].claim_type == "avoids_recapture"
        assert result[1].claim_type == "gains_material"
        assert result[2].claim_type == "positional_pressure"


# ---------------------------------------------------------------------------
# Dispatch table completeness
# ---------------------------------------------------------------------------

class TestDispatchTableCompleteness:

    def test_all_required_claim_types_have_rules(self):
        required = {
            "avoids_recapture",
            "can_be_recaptured",
            "gains_material",
            "promotes_to_king",
            "near_promotion",
            "opponent_near_promotion",
            "mobility_increase",
            "mobility_decrease",
            "piece_isolated",
            "piece_connected",
            "creates_immediate_threat",
            "minimax_confirmation",
            "positional_pressure",
            "strategic_initiative",
            "long_term_compensation",
            # Phase 4.1 additions:
            "shot_sequence_or_multi_jump",
            "blocks_landing_square",
            "forced_opponent_jump",
        }
        missing = required - set(_VERIFICATION_RULES.keys())
        assert not missing, f"Missing verification rules for: {missing}"

    def test_all_rules_are_callable(self):
        for claim_type, rule in _VERIFICATION_RULES.items():
            assert callable(rule), f"Rule for {claim_type!r} is not callable"

    def test_all_rules_return_claim_status(self):
        """Every rule returns a ClaimStatus when called with empty facts."""
        for claim_type, rule in _VERIFICATION_RULES.items():
            result = rule({})
            assert isinstance(result, ClaimStatus), (
                f"Rule for {claim_type!r} returned {type(result).__name__}, "
                f"expected ClaimStatus"
            )


# ---------------------------------------------------------------------------
# Spec contradiction examples (from the implementation brief)
# ---------------------------------------------------------------------------

class TestSpecContradictionExamples:
    """
    Explicitly named tests for the three contradiction examples from the spec.
    """

    def test_spec_example_avoids_recapture_contradicted(self):
        """avoids_recapture + opponent_can_recapture=True → CONTRADICTED"""
        rec = _verify_one("avoids_recapture", {"opponent_can_recapture": True})
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_spec_example_gains_material_contradicted(self):
        """gains_material + net_gain<=0 → CONTRADICTED"""
        for val in [0, -1, -5]:
            rec = _verify_one("gains_material", {"net_gain": val})
            assert rec.claim_status == ClaimStatus.CONTRADICTED, (
                f"Expected CONTRADICTED for net_gain={val}"
            )

    def test_spec_example_promotes_to_king_contradicted(self):
        """promotes_to_king + results_in_king=False → CONTRADICTED"""
        rec = _verify_one("promotes_to_king", {"results_in_king": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# Phase 4.1 — new verifier rules
# ---------------------------------------------------------------------------

class TestShotSequenceOrMultiJump:
    """Verifier for shot_sequence_or_multi_jump (Phase 4.1)."""

    def test_supported_when_shot_sequence_true(self):
        rec = _verify_one("shot_sequence_or_multi_jump", {"shot_sequence_available": True})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_shot_sequence_false(self):
        rec = _verify_one("shot_sequence_or_multi_jump", {"shot_sequence_available": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("shot_sequence_or_multi_jump", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_unrelated_facts_only(self):
        rec = _verify_one("shot_sequence_or_multi_jump", {"net_gain": 1, "captures_count": 1})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_contradicted_annotates_factual_contradiction(self):
        rec = _verify_one("shot_sequence_or_multi_jump", {"shot_sequence_available": False})
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_supported_clears_hallucination_annotation(self):
        claim = _make_claim(
            "shot_sequence_or_multi_jump",
            hallucination=HallucinationType.FABRICATED_CLAIM,
        )
        result = verify_claims([claim], {"shot_sequence_available": True})
        assert result[0].claim_status == ClaimStatus.SUPPORTED
        assert result[0].hallucination_type is None

    def test_end_to_end_extract_then_verify_supported(self):
        """Full pipeline: extract → verify with True fact."""
        from checkers.evaluation.claim_extractor import extract_claims
        claims = extract_claims(
            "The move extends the attack with a multi-jump sequence.",
            reasoning_seeds=["shot_sequence_available=true"],
            facts={"shot_sequence_available": True},
        )
        assert any(c.claim_type == "shot_sequence_or_multi_jump" for c in claims)
        verified = verify_claims(claims, {"shot_sequence_available": True})
        match = next(c for c in verified if c.claim_type == "shot_sequence_or_multi_jump")
        assert match.claim_status == ClaimStatus.SUPPORTED

    def test_end_to_end_extract_then_verify_contradicted(self):
        """Full pipeline: extract → verify with False fact."""
        from checkers.evaluation.claim_extractor import extract_claims
        claims = extract_claims(
            "The move enables a multi-jump sequence.",
            reasoning_seeds=[],
            facts={"shot_sequence_available": False},
        )
        assert any(c.claim_type == "shot_sequence_or_multi_jump" for c in claims)
        verified = verify_claims(claims, {"shot_sequence_available": False})
        match = next(c for c in verified if c.claim_type == "shot_sequence_or_multi_jump")
        assert match.claim_status == ClaimStatus.CONTRADICTED


class TestBlocksLandingSquare:
    """Verifier for blocks_landing_square (Phase 4.1)."""

    def test_supported_when_blocks_opponent_landing_true(self):
        rec = _verify_one("blocks_landing_square", {"blocks_opponent_landing": True})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_blocks_opponent_landing_false(self):
        rec = _verify_one("blocks_landing_square", {"blocks_opponent_landing": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("blocks_landing_square", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_unrelated_facts_only(self):
        rec = _verify_one("blocks_landing_square", {"opponent_can_recapture": False})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_contradicted_annotates_factual_contradiction(self):
        rec = _verify_one("blocks_landing_square", {"blocks_opponent_landing": False})
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_supported_clears_hallucination_annotation(self):
        claim = _make_claim(
            "blocks_landing_square",
            hallucination=HallucinationType.FABRICATED_CLAIM,
        )
        result = verify_claims([claim], {"blocks_opponent_landing": True})
        assert result[0].claim_status == ClaimStatus.SUPPORTED
        assert result[0].hallucination_type is None

    def test_end_to_end_extract_then_verify_supported(self):
        """Full pipeline: extract → verify with True fact."""
        from checkers.evaluation.claim_extractor import extract_claims
        claims = extract_claims(
            "This move blocks the opponent from landing on a key square.",
            reasoning_seeds=["blocks_opponent_landing=true"],
            facts={"blocks_opponent_landing": True},
        )
        assert any(c.claim_type == "blocks_landing_square" for c in claims)
        verified = verify_claims(claims, {"blocks_opponent_landing": True})
        match = next(c for c in verified if c.claim_type == "blocks_landing_square")
        assert match.claim_status == ClaimStatus.SUPPORTED

    def test_end_to_end_extract_then_verify_contradicted(self):
        """Full pipeline: extract → verify with False fact."""
        from checkers.evaluation.claim_extractor import extract_claims
        claims = extract_claims(
            "The move denies a key landing square to the opponent.",
            reasoning_seeds=[],
            facts={"blocks_opponent_landing": False},
        )
        assert any(c.claim_type == "blocks_landing_square" for c in claims)
        verified = verify_claims(claims, {"blocks_opponent_landing": False})
        match = next(c for c in verified if c.claim_type == "blocks_landing_square")
        assert match.claim_status == ClaimStatus.CONTRADICTED


class TestForcedOpponentJump:
    """Verifier for forced_opponent_jump (Phase 4.1)."""

    def test_supported_when_forced_jump_true(self):
        rec = _verify_one("forced_opponent_jump", {"forced_opponent_jump_reply": True})
        assert rec.claim_status == ClaimStatus.SUPPORTED

    def test_contradicted_when_forced_jump_false(self):
        rec = _verify_one("forced_opponent_jump", {"forced_opponent_jump_reply": False})
        assert rec.claim_status == ClaimStatus.CONTRADICTED

    def test_unsupported_when_fact_absent(self):
        rec = _verify_one("forced_opponent_jump", {})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_unsupported_when_unrelated_facts_only(self):
        rec = _verify_one("forced_opponent_jump", {"net_gain": 2, "results_in_king": False})
        assert rec.claim_status == ClaimStatus.UNSUPPORTED

    def test_contradicted_annotates_factual_contradiction(self):
        rec = _verify_one("forced_opponent_jump", {"forced_opponent_jump_reply": False})
        assert rec.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION

    def test_supported_clears_hallucination_annotation(self):
        claim = _make_claim(
            "forced_opponent_jump",
            hallucination=HallucinationType.FABRICATED_CLAIM,
        )
        result = verify_claims([claim], {"forced_opponent_jump_reply": True})
        assert result[0].claim_status == ClaimStatus.SUPPORTED
        assert result[0].hallucination_type is None

    def test_end_to_end_extract_then_verify_supported(self):
        """Full pipeline: extract → verify with True fact."""
        from checkers.evaluation.claim_extractor import extract_claims
        claims = extract_claims(
            "The opponent is constrained to a jump in reply.",
            reasoning_seeds=["forced_opponent_jump_reply=true"],
            facts={"forced_opponent_jump_reply": True},
        )
        assert any(c.claim_type == "forced_opponent_jump" for c in claims)
        verified = verify_claims(claims, {"forced_opponent_jump_reply": True})
        match = next(c for c in verified if c.claim_type == "forced_opponent_jump")
        assert match.claim_status == ClaimStatus.SUPPORTED

    def test_end_to_end_extract_then_verify_contradicted(self):
        """Full pipeline: extract → verify with False fact."""
        from checkers.evaluation.claim_extractor import extract_claims
        claims = extract_claims(
            "The opponent is limited to a jump in response.",
            reasoning_seeds=[],
            facts={"forced_opponent_jump_reply": False},
        )
        assert any(c.claim_type == "forced_opponent_jump" for c in claims)
        verified = verify_claims(claims, {"forced_opponent_jump_reply": False})
        match = next(c for c in verified if c.claim_type == "forced_opponent_jump")
        assert match.claim_status == ClaimStatus.CONTRADICTED


class TestPhase41EvaluateTurnIntegration:
    """evaluate_turn no longer leaves the three new types at dispatch-unknown status."""

    def test_shot_sequence_claim_verified_through_evaluate_turn(self):
        from checkers.evaluation.turn_evaluator import evaluate_turn
        result = evaluate_turn(
            reasoning_text="The move enables a multi-jump sequence to extend the attack.",
            reasoning_seeds=["shot_sequence_available=true"],
            facts={"shot_sequence_available": True},
            ranker_diagnostics={},
            turn_id="phase41_shot_seq",
        )
        match = next(
            (c for c in result.claims if c.claim_type == "shot_sequence_or_multi_jump"),
            None,
        )
        assert match is not None, "shot_sequence_or_multi_jump not extracted"
        assert match.claim_status == ClaimStatus.SUPPORTED

    def test_blocks_landing_claim_verified_through_evaluate_turn(self):
        from checkers.evaluation.turn_evaluator import evaluate_turn
        result = evaluate_turn(
            reasoning_text="The move denies a key landing square to the opponent.",
            reasoning_seeds=["blocks_opponent_landing=true"],
            facts={"blocks_opponent_landing": True},
            ranker_diagnostics={},
            turn_id="phase41_blocks_landing",
        )
        match = next(
            (c for c in result.claims if c.claim_type == "blocks_landing_square"),
            None,
        )
        assert match is not None, "blocks_landing_square not extracted"
        assert match.claim_status == ClaimStatus.SUPPORTED

    def test_forced_jump_claim_verified_through_evaluate_turn(self):
        from checkers.evaluation.turn_evaluator import evaluate_turn
        result = evaluate_turn(
            reasoning_text="The opponent is constrained to a jump in reply.",
            reasoning_seeds=["forced_opponent_jump_reply=true"],
            facts={"forced_opponent_jump_reply": True},
            ranker_diagnostics={},
            turn_id="phase41_forced_jump",
        )
        match = next(
            (c for c in result.claims if c.claim_type == "forced_opponent_jump"),
            None,
        )
        assert match is not None, "forced_opponent_jump not extracted"
        assert match.claim_status == ClaimStatus.SUPPORTED

    def test_total_claims_count_consistency_with_new_types(self):
        """supported + contradicted + unsupported + vague still equals total_claims."""
        from checkers.evaluation.turn_evaluator import evaluate_turn
        result = evaluate_turn(
            reasoning_text=(
                "The move extends the attack with a multi-jump sequence. "
                "It also blocks the opponent from landing on a key square. "
                "The opponent is constrained to a jump in reply."
            ),
            reasoning_seeds=[
                "shot_sequence_available=true",
                "blocks_opponent_landing=true",
                "forced_opponent_jump_reply=true",
            ],
            facts={
                "shot_sequence_available": True,
                "blocks_opponent_landing": True,
                "forced_opponent_jump_reply": True,
            },
            ranker_diagnostics={},
            turn_id="phase41_all_three",
        )
        total = (
            result.supported_count
            + result.contradicted_count
            + result.unsupported_count
            + result.vague_count
        )
        assert total == result.total_claims
