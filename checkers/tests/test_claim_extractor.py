# checkers/tests/test_claim_extractor.py
#
# Tests for checkers/evaluation/claim_extractor.py
#
# PURPOSE
# -------
# Verify that the deterministic claim extractor:
#   1. Extracts multiple claims from compound sentences.
#   2. Correctly labels seed-supported claims.
#   3. Correctly labels unsupported strategic claims.
#   4. Detects minimax confirmation sentences.
#   5. Marks shot_sequence/multi-jump as OVERCLAIM_RISK.
#   6. Requires no runtime pipeline imports.
#   7. Produces JSON-serialisable ClaimRecord objects.
#
# These tests have NO side effects and require NO external services.

import json
import sys
from typing import Any

import pytest

# ── Guard: record modules before import ──────────────────────────────────────
_modules_before = set(sys.modules.keys())

from checkers.evaluation.claim_extractor import (
    ClaimRecord,
    extract_claims,
    _PHRASE_TABLE,
    _find_matching_seed,
    _check_fact_support,
    _phrase_in_negated_context,
    _MATERIAL_NEGATABLE_PHRASES,
    _MATERIAL_NEGATION_SENTINELS,
    _MOBILITY_DIRECTION_NEGATIVE,
    _MOBILITY_INCREASE_AMBIGUOUS,
)
from checkers.evaluation.reasoning_taxonomy import (
    ClaimStatus,
    ClaimVerifiability,
    HallucinationType,
    SeedRiskType,
)

_modules_after = set(sys.modules.keys())


# ---------------------------------------------------------------------------
# Isolation: no runtime pipeline imports
# ---------------------------------------------------------------------------

_FORBIDDEN_RUNTIME_PREFIXES = (
    "checkers.engine",
    "checkers.agents",
    "checkers.graph",
    "checkers.state",
    "checkers.nodes",
    "checkers.search",
)


def test_no_runtime_pipeline_imports():
    """
    Importing claim_extractor must not pull in any runtime pipeline module.
    """
    new_modules = _modules_after - _modules_before
    for mod in new_modules:
        for prefix in _FORBIDDEN_RUNTIME_PREFIXES:
            assert not mod.startswith(prefix), (
                f"Importing claim_extractor pulled in runtime module: {mod!r}."
            )


# ---------------------------------------------------------------------------
# Phrase table integrity
# ---------------------------------------------------------------------------

class TestPhraseTable:

    def test_all_entries_have_nonempty_phrases(self):
        for entry in _PHRASE_TABLE:
            assert len(entry.phrases) > 0, (
                f"{entry.claim_type} has no phrases"
            )

    def test_all_entries_have_valid_verifiability(self):
        for entry in _PHRASE_TABLE:
            assert isinstance(entry.verifiability, ClaimVerifiability), (
                f"{entry.claim_type} verifiability is not ClaimVerifiability"
            )

    def test_all_entries_have_valid_seed_risk(self):
        for entry in _PHRASE_TABLE:
            assert isinstance(entry.seed_risk, SeedRiskType), (
                f"{entry.claim_type} seed_risk is not SeedRiskType"
            )

    def test_no_duplicate_claim_types(self):
        types = [e.claim_type for e in _PHRASE_TABLE]
        assert len(types) == len(set(types)), (
            f"Duplicate claim_type entries: "
            f"{[t for t in types if types.count(t) > 1]}"
        )

    def test_expected_claim_types_present(self):
        expected = {
            "avoids_recapture",
            "can_be_recaptured",
            "gains_material",
            "promotes_to_king",
            "near_promotion",
            "creates_immediate_threat",
            "shot_sequence_or_multi_jump",
            "blocks_landing_square",
            "forced_opponent_jump",
            "piece_isolated",
            "piece_connected",
            "weakens_king_row",
            "center_control",
            "mobility_decrease",
            "mobility_increase",
            "minimax_confirmation",
            "positional_pressure",
            "strategic_initiative",
            "long_term_compensation",
        }
        actual = {e.claim_type for e in _PHRASE_TABLE}
        assert expected.issubset(actual), (
            f"Missing claim types: {expected - actual}"
        )


# ---------------------------------------------------------------------------
# Seed matching
# ---------------------------------------------------------------------------

class TestSeedMatching:

    def test_find_matching_seed_positive(self):
        seeds = [
            "opponent_can_recapture=false — immediate tactical safety",
            "captures_count=2, net_gain=2 — wins material",
        ]
        result = _find_matching_seed(["opponent_can_recapture=false"], seeds)
        assert result is not None
        assert "opponent_can_recapture=false" in result

    def test_find_matching_seed_negative(self):
        seeds = [
            "creates_immediate_threat=true — puts opponent on defensive",
        ]
        result = _find_matching_seed(["opponent_can_recapture=false"], seeds)
        assert result is None

    def test_find_matching_seed_empty_seeds(self):
        result = _find_matching_seed(["opponent_can_recapture=false"], [])
        assert result is None

    def test_find_matching_seed_case_insensitive(self):
        seeds = ["Opponent_Can_Recapture=False — safety"]
        result = _find_matching_seed(["opponent_can_recapture=false"], seeds)
        assert result is not None


# ---------------------------------------------------------------------------
# Fact support
# ---------------------------------------------------------------------------

class TestFactSupport:

    def test_recapture_false_supported(self):
        from checkers.evaluation.claim_extractor import _ENTRY_BY_TYPE
        entry = _ENTRY_BY_TYPE["avoids_recapture"]
        assert _check_fact_support(entry, {"opponent_can_recapture": False})

    def test_recapture_true_not_supported_for_avoids(self):
        from checkers.evaluation.claim_extractor import _ENTRY_BY_TYPE
        entry = _ENTRY_BY_TYPE["avoids_recapture"]
        assert not _check_fact_support(entry, {"opponent_can_recapture": True})

    def test_captures_count_positive_supported(self):
        from checkers.evaluation.claim_extractor import _ENTRY_BY_TYPE
        entry = _ENTRY_BY_TYPE["gains_material"]
        assert _check_fact_support(entry, {"captures_count": 2})

    def test_captures_count_zero_not_supported(self):
        from checkers.evaluation.claim_extractor import _ENTRY_BY_TYPE
        entry = _ENTRY_BY_TYPE["gains_material"]
        assert not _check_fact_support(entry, {"captures_count": 0})

    def test_no_fact_field_returns_false(self):
        from checkers.evaluation.claim_extractor import _ENTRY_BY_TYPE
        entry = _ENTRY_BY_TYPE["positional_pressure"]
        assert not _check_fact_support(entry, {"some_field": 42})


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

class TestExtractClaims:

    def test_empty_text_returns_empty(self):
        assert extract_claims("") == []

    def test_single_recapture_claim(self):
        text = "This move avoids recapture, keeping pieces safe."
        records = extract_claims(
            text,
            reasoning_seeds=["opponent_can_recapture=false — safety"],
        )
        recapture = [r for r in records if r.claim_type == "avoids_recapture"]
        assert len(recapture) == 1
        assert recapture[0].claim_status == ClaimStatus.SUPPORTED
        assert recapture[0].source == "seed"
        assert recapture[0].matched_seed is not None

    def test_multiple_claims_from_one_sentence(self):
        """Compound sentence should yield multiple independent claims."""
        text = (
            "This move avoids recapture and reduces opponent mobility, "
            "denying the opponent a key landing square."
        )
        seeds = [
            "opponent_can_recapture=false — safety",
            "opponent_mobility_before=8, opponent_mobility_after=5 — reduces by 3",
            "blocks_opponent_landing=true — denies landing",
        ]
        records = extract_claims(text, reasoning_seeds=seeds)
        types = {r.claim_type for r in records}
        assert "avoids_recapture" in types
        assert "mobility_decrease" in types
        assert "blocks_landing_square" in types
        # All should be seed-supported
        for r in records:
            if r.claim_type in ("avoids_recapture", "mobility_decrease",
                                "blocks_landing_square"):
                assert r.claim_status == ClaimStatus.SUPPORTED
                assert r.source == "seed"

    def test_seed_supported_gains_material(self):
        text = "Captures 2 pieces for a net gain of +2, winning material."
        seeds = ["captures_count=2, net_gain=2 — wins material"]
        records = extract_claims(text, reasoning_seeds=seeds)
        material = [r for r in records if r.claim_type == "gains_material"]
        assert len(material) == 1
        assert material[0].claim_status == ClaimStatus.SUPPORTED
        assert material[0].source == "seed"

    def test_fact_supported_without_seed(self):
        """A claim backed by a fact but not by an explicit seed."""
        text = "This move avoids recapture."
        records = extract_claims(
            text,
            reasoning_seeds=[],  # no seeds
            facts={"opponent_can_recapture": False},
        )
        recapture = [r for r in records if r.claim_type == "avoids_recapture"]
        assert len(recapture) == 1
        assert recapture[0].claim_status == ClaimStatus.SUPPORTED
        assert recapture[0].source == "fact_phrase"

    def test_unsupported_strategic_claim(self):
        """An unverifiable claim with no seed or fact backing."""
        text = "This move creates positional pressure on the opponent."
        records = extract_claims(text, reasoning_seeds=[], facts={})
        pressure = [r for r in records if r.claim_type == "positional_pressure"]
        assert len(pressure) == 1
        assert pressure[0].claim_status == ClaimStatus.UNSUPPORTED
        assert pressure[0].claim_verifiability == ClaimVerifiability.UNVERIFIABLE
        assert pressure[0].hallucination_type == HallucinationType.FABRICATED_CLAIM
        assert pressure[0].source == "unsupported_phrase"

    def test_minimax_confirmation(self):
        text = "The engine confirms this as the highest-evaluated option with minimax_score=4.00."
        seeds = ["minimax_score=4.00 — highest-evaluated option"]
        records = extract_claims(text, reasoning_seeds=seeds)
        minimax = [r for r in records if r.claim_type == "minimax_confirmation"]
        assert len(minimax) == 1
        assert minimax[0].claim_status == ClaimStatus.SUPPORTED
        assert minimax[0].claim_verifiability == ClaimVerifiability.FULLY_VERIFIABLE

    def test_multi_jump_phrase_marked_overclaim_risk(self):
        """shot_sequence/multi-jump claims should carry OVERCLAIM_RISK."""
        text = "A multi-jump sequence is available to extend the attack."
        seeds = ["shot_sequence_available=true — multi-jump sequence available"]
        records = extract_claims(text, reasoning_seeds=seeds)
        shot = [r for r in records if r.claim_type == "shot_sequence_or_multi_jump"]
        assert len(shot) == 1
        assert shot[0].seed_risk_type == SeedRiskType.OVERCLAIM_RISK
        assert shot[0].claim_verifiability == ClaimVerifiability.PARTIALLY_VERIFIABLE
        assert shot[0].claim_status == ClaimStatus.SUPPORTED

    def test_promotion_claim(self):
        text = "This move immediately converts the piece into a king."
        seeds = ["results_in_king=true — immediately converts the piece"]
        records = extract_claims(text, reasoning_seeds=seeds)
        promo = [r for r in records if r.claim_type == "promotes_to_king"]
        assert len(promo) == 1
        assert promo[0].claim_status == ClaimStatus.SUPPORTED
        assert promo[0].claim_verifiability == ClaimVerifiability.FULLY_VERIFIABLE

    def test_center_control_interpretive(self):
        text = "This move improves influence over central lanes."
        seeds = ["center_control=true — improves influence over central lanes"]
        records = extract_claims(text, reasoning_seeds=seeds)
        center = [r for r in records if r.claim_type == "center_control"]
        assert len(center) == 1
        assert center[0].seed_risk_type == SeedRiskType.INTERPRETIVE

    def test_weakens_king_row(self):
        text = "Back-row defense is weakened by this move."
        seeds = ["weakens_king_row=true — back-row defense is weakened"]
        records = extract_claims(text, reasoning_seeds=seeds)
        kr = [r for r in records if r.claim_type == "weakens_king_row"]
        assert len(kr) == 1
        assert kr[0].claim_status == ClaimStatus.SUPPORTED

    def test_can_be_recaptured(self):
        text = "Although the opponent can recapture this piece, the gain compensates."
        seeds = ["opponent_can_recapture=true — tactical drawback"]
        records = extract_claims(text, reasoning_seeds=seeds)
        recap = [r for r in records if r.claim_type == "can_be_recaptured"]
        assert len(recap) == 1
        assert recap[0].claim_status == ClaimStatus.SUPPORTED

    def test_isolation_claim(self):
        text = "The moved piece is isolated from allies."
        seeds = ["leaves_piece_isolated=true — piece is not supported"]
        records = extract_claims(text, reasoning_seeds=seeds)
        iso = [r for r in records if r.claim_type == "piece_isolated"]
        assert len(iso) == 1
        assert iso[0].claim_status == ClaimStatus.SUPPORTED

    def test_no_false_positives_on_clean_text(self):
        """Text with no claim-like phrases should produce no records."""
        text = "This is just a regular move with nothing special about it."
        records = extract_claims(text, reasoning_seeds=[], facts={})
        assert records == []

    def test_deterministic_output(self):
        """Same inputs must always produce the same output."""
        text = (
            "This move avoids recapture and gains material. "
            "The engine confirms minimax_score=5.00."
        )
        seeds = [
            "opponent_can_recapture=false — safety",
            "captures_count=1, net_gain=1 — wins material",
            "minimax_score=5.00 — highest-evaluated option",
        ]
        r1 = extract_claims(text, reasoning_seeds=seeds)
        r2 = extract_claims(text, reasoning_seeds=seeds)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.claim_type == b.claim_type
            assert a.claim_status == b.claim_status
            assert a.source == b.source
            assert a.matched_phrase == b.matched_phrase

    def test_inputs_not_mutated(self):
        """extract_claims must not mutate its input arguments."""
        text = "This move avoids recapture."
        seeds = ["opponent_can_recapture=false — safety"]
        facts = {"opponent_can_recapture": False}
        seeds_copy = list(seeds)
        facts_copy = dict(facts)
        extract_claims(text, reasoning_seeds=seeds, facts=facts)
        assert seeds == seeds_copy
        assert facts == facts_copy

    def test_strategic_initiative_unsupported(self):
        text = "This move seizes the initiative and dominates the board."
        records = extract_claims(text, reasoning_seeds=[], facts={})
        init = [r for r in records if r.claim_type == "strategic_initiative"]
        assert len(init) == 1
        assert init[0].claim_status == ClaimStatus.UNSUPPORTED
        assert init[0].hallucination_type == HallucinationType.FABRICATED_CLAIM

    def test_long_term_compensation_unsupported(self):
        text = "The sacrifice offers long-term compensation for the material loss."
        records = extract_claims(text, reasoning_seeds=[], facts={})
        comp = [r for r in records if r.claim_type == "long_term_compensation"]
        assert len(comp) == 1
        assert comp[0].claim_status == ClaimStatus.UNSUPPORTED

    def test_near_promotion_partially_verifiable(self):
        text = "Creates a future promotion threat."
        seeds = ["near_promotion=true — creates a future promotion threat"]
        records = extract_claims(text, reasoning_seeds=seeds)
        np_claims = [r for r in records if r.claim_type == "near_promotion"]
        assert len(np_claims) == 1
        assert np_claims[0].claim_verifiability == ClaimVerifiability.PARTIALLY_VERIFIABLE
        assert np_claims[0].seed_risk_type == SeedRiskType.INTERPRETIVE

    def test_forced_opponent_jump(self):
        text = "The opponent response is constrained to a jump."
        seeds = ["forced_opponent_jump_reply=true, max_opponent_jump_captures=1 — constrained"]
        records = extract_claims(text, reasoning_seeds=seeds)
        forced = [r for r in records if r.claim_type == "forced_opponent_jump"]
        assert len(forced) == 1
        assert forced[0].claim_status == ClaimStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Recapture phrase boundary (Step 2 fix)
# ---------------------------------------------------------------------------

class TestRecapturePhraseBoundary:
    """
    Verify that negative/safety recapture phrases do NOT trigger
    can_be_recaptured and that positive-risk phrases DO trigger it.

    This class directly tests the fix for the false-positive where the
    former phrase "recapture risk" matched inside "no recapture risk" and
    "without recapture risk".
    """

    # ── Negative phrases: must map to avoids_recapture, not can_be_recaptured ──

    def test_no_recapture_risk_does_not_fire_can_be_recaptured(self):
        text = "This move carries no recapture risk for our pieces."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" not in types, (
            "\"no recapture risk\" must not trigger can_be_recaptured"
        )

    def test_no_recapture_risk_fires_avoids_recapture(self):
        text = "This move carries no recapture risk for our pieces."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "avoids_recapture" in types

    def test_without_recapture_risk_does_not_fire_can_be_recaptured(self):
        text = "The piece moves forward without recapture risk."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" not in types

    def test_cannot_recapture_does_not_fire_can_be_recaptured(self):
        text = "The opponent cannot recapture the piece next turn."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" not in types

    def test_opponent_cannot_recapture_does_not_fire_can_be_recaptured(self):
        text = "The opponent cannot recapture after this move."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" not in types

    def test_avoids_recapture_does_not_fire_can_be_recaptured(self):
        text = "This move avoids recapture entirely."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" not in types

    # ── Positive phrases: must fire can_be_recaptured ─────────────────────────

    def test_can_be_recaptured_phrase_fires(self):
        text = "The piece can be recaptured by the opponent on the next move."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" in types

    def test_opponent_can_recapture_phrase_fires(self):
        text = "The opponent can recapture this piece, but the material gain outweighs it."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" in types

    def test_exposed_to_recapture_phrase_fires(self):
        text = "After this move the piece is exposed to recapture."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" in types

    def test_vulnerable_to_recapture_phrase_fires(self):
        text = "The moved piece is vulnerable to recapture next turn."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" in types

    def test_recapture_risk_remains_phrase_fires(self):
        text = "Although material is gained, recapture risk remains significant."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" in types

    def test_tactically_exposed_phrase_fires(self):
        text = "The piece is tactically exposed after this advance."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "can_be_recaptured" in types

    # ── Mutual exclusivity in safety-positive sentences ───────────────────────

    def test_safety_sentence_does_not_fire_both(self):
        """A sentence confirming safety must not simultaneously claim risk."""
        text = "This move leaves the piece with no recapture risk and full safety."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        # avoids_recapture is expected; can_be_recaptured must not appear
        assert "avoids_recapture" in types
        assert "can_be_recaptured" not in types

    def test_compound_sentence_with_positive_risk_fires_both(self):
        """A compound sentence acknowledging drawback fires both types."""
        text = (
            "This move avoids recapture for most pieces, "
            "but the advanced piece can be recaptured."
        )
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        # Both should fire since both facts appear
        assert "avoids_recapture" in types
        assert "can_be_recaptured" in types


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class TestClaimRecordSerialisation:


    def test_to_dict_basic(self):
        record = ClaimRecord(
            claim_type="avoids_recapture",
            claim_status=ClaimStatus.SUPPORTED,
            claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
            seed_risk_type=SeedRiskType.STRICT_FACT,
            hallucination_type=None,
            matched_phrase="avoids recapture",
            matched_seed="opponent_can_recapture=false — safety",
            source="seed",
        )
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["claim_type"] == "avoids_recapture"
        assert d["claim_status"] == "supported"
        assert d["claim_verifiability"] == "fully_verifiable"
        assert d["seed_risk_type"] == "strict_fact"
        assert d["hallucination_type"] is None
        assert d["source"] == "seed"

    def test_to_dict_json_serialisable(self):
        record = ClaimRecord(
            claim_type="positional_pressure",
            claim_status=ClaimStatus.UNSUPPORTED,
            claim_verifiability=ClaimVerifiability.UNVERIFIABLE,
            seed_risk_type=None,
            hallucination_type=HallucinationType.FABRICATED_CLAIM,
            matched_phrase="positional pressure",
            matched_seed=None,
            source="unsupported_phrase",
        )
        d = record.to_dict()
        # Must not raise
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        assert loaded["claim_type"] == "positional_pressure"
        assert loaded["claim_status"] == "unsupported"
        assert loaded["hallucination_type"] == "fabricated_claim"
        assert loaded["seed_risk_type"] is None

    def test_all_claim_records_from_extraction_are_serialisable(self):
        """Full integration: extract → serialise → round-trip."""
        text = (
            "This move avoids recapture and creates an immediate threat. "
            "A multi-jump sequence is available. "
            "The engine confirms minimax_score=12.50."
        )
        seeds = [
            "opponent_can_recapture=false — safety",
            "creates_immediate_threat=true — puts opponent on defensive",
            "shot_sequence_available=true — multi-jump sequence",
            "minimax_score=12.50 — highest-evaluated option",
        ]
        records = extract_claims(text, reasoning_seeds=seeds)
        assert len(records) >= 3
        for r in records:
            d = r.to_dict()
            dumped = json.dumps(d)
            loaded = json.loads(dumped)
            assert loaded["claim_type"] == r.claim_type
            assert loaded["source"] == r.source


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_none_seeds_and_facts(self):
        """Gracefully handle None seeds and facts."""
        text = "This move avoids recapture."
        records = extract_claims(text, reasoning_seeds=None, facts=None)
        assert len(records) >= 1
        recapture = [r for r in records if r.claim_type == "avoids_recapture"]
        assert recapture[0].source == "unsupported_phrase"

    def test_case_insensitive_matching(self):
        text = "This move AVOIDS RECAPTURE and GAINS MATERIAL."
        seeds = [
            "opponent_can_recapture=false — safety",
            "captures_count=1, net_gain=1 — wins material",
        ]
        records = extract_claims(text, reasoning_seeds=seeds)
        types = {r.claim_type for r in records}
        assert "avoids_recapture" in types
        assert "gains_material" in types

    def test_claim_appears_at_most_once(self):
        """Even if multiple phrases for the same claim match, only one record."""
        text = (
            "This move avoids recapture. The opponent cannot recapture. "
            "No recapture risk exists."
        )
        seeds = ["opponent_can_recapture=false — safety"]
        records = extract_claims(text, reasoning_seeds=seeds)
        recapture = [r for r in records if r.claim_type == "avoids_recapture"]
        assert len(recapture) == 1

    def test_realistic_compound_paragraph(self):
        """
        A realistic seeded reasoning paragraph should produce multiple
        well-labelled claims without errors.
        """
        text = (
            "This move avoids recapture, keeping all allied pieces safe with "
            "no pieces remaining under threat. It reduces opponent mobility by 3, "
            "restricting available replies and denying the opponent a key landing "
            "square. The moved piece stays connected with adjacent allies. "
            "Move [2] leaves the moved piece threatened while this move does not. "
            "The engine confirms this as the highest-evaluated option with "
            "minimax_score=4.00."
        )
        seeds = [
            "opponent_can_recapture=false — immediate tactical safety: "
            "opponent cannot recapture this piece next turn",
            "our_pieces_threatened_after=0 — no defensive burden remains",
            "opponent_mobility_before=8, opponent_mobility_after=5 — "
            "reduces opponent mobility by 3, restricting available replies",
            "blocks_opponent_landing=true — denies the opponent a key landing square",
            "leaves_piece_isolated=false — preserves piece coordination",
            "Move [2] leaves the moved piece threatened "
            "(moved_piece_is_threatened=true vs false here).",
            "minimax_score=4.00 — highest-evaluated option",
        ]
        facts = {
            "opponent_can_recapture": False,
            "our_pieces_threatened_after": 0,
            "mobility_reduction": 3,
            "blocks_opponent_landing": True,
            "leaves_piece_isolated": False,
            "minimax_score": 4.0,
        }
        records = extract_claims(text, reasoning_seeds=seeds, facts=facts)
        types = {r.claim_type for r in records}

        # Must detect these core claims
        assert "avoids_recapture" in types
        assert "mobility_decrease" in types
        assert "blocks_landing_square" in types
        assert "piece_connected" in types
        assert "minimax_confirmation" in types

        # All detected claims should be supported
        for r in records:
            assert r.claim_status == ClaimStatus.SUPPORTED, (
                f"{r.claim_type} should be SUPPORTED but is {r.claim_status}"
            )

        # No hallucinations in this clean paragraph
        for r in records:
            assert r.hallucination_type is None, (
                f"{r.claim_type} incorrectly flagged as hallucination"
            )


# ---------------------------------------------------------------------------
# Polarity-aware extraction (Phase 1b)
# ---------------------------------------------------------------------------

class TestPolarityHelpers:
    """Direct unit tests for _phrase_in_negated_context."""

    def test_detects_negation_before_phrase(self):
        assert _phrase_in_negated_context(
            "despite no material gain here",
            "material gain",
            _MATERIAL_NEGATION_SENTINELS,
        )

    def test_no_negation_returns_false(self):
        assert not _phrase_in_negated_context(
            "the move results in material gain",
            "material gain",
            _MATERIAL_NEGATION_SENTINELS,
        )

    def test_phrase_absent_returns_false(self):
        assert not _phrase_in_negated_context(
            "the move avoids recapture",
            "material gain",
            _MATERIAL_NEGATION_SENTINELS,
        )

    def test_reduces_before_mobility_phrase(self):
        assert _phrase_in_negated_context(
            "reduces our mobility by one",
            "our mobility by",
            _MOBILITY_DIRECTION_NEGATIVE,
        )

    def test_increases_before_mobility_phrase_not_detected(self):
        assert not _phrase_in_negated_context(
            "increases our mobility by two",
            "our mobility by",
            _MOBILITY_DIRECTION_NEGATIVE,
        )

    def test_negatable_phrases_set_contains_key_phrases(self):
        assert "material gain" in _MATERIAL_NEGATABLE_PHRASES
        assert "net gain" in _MATERIAL_NEGATABLE_PHRASES
        assert "gains material" in _MATERIAL_NEGATABLE_PHRASES

    def test_ambiguous_mobility_set_contains_our_mobility_by(self):
        assert "our mobility by" in _MOBILITY_INCREASE_AMBIGUOUS

    def test_direction_negative_set_contains_reduces(self):
        assert "reduces" in _MOBILITY_DIRECTION_NEGATIVE
        assert "decreases" in _MOBILITY_DIRECTION_NEGATIVE


class TestMaterialGainPolarity:
    """Polarity suppression for gains_material claims."""

    # ── Negated — must NOT produce gains_material ─────────────────────────────

    def test_despite_no_material_gain_suppressed(self):
        text = "Despite no material gain, the move improves piece positioning."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "gains_material" not in types, (
            "\"despite no material gain\" must not produce gains_material"
        )

    def test_no_material_gain_suppressed(self):
        text = "There is no material gain from this quiet move."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "gains_material" not in types

    def test_lack_of_material_gain_suppressed(self):
        text = "Despite the lack of material gain, the position improves structurally."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "gains_material" not in types

    def test_without_gains_material_suppressed(self):
        text = "The move advances a piece without gains material."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "gains_material" not in types

    def test_no_net_gain_suppressed(self):
        text = "Although captures_count=0, there is no net gain here."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "gains_material" not in types

    def test_lack_of_gains_a_piece_suppressed(self):
        text = "The lack of gains a piece makes this a purely positional choice."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "gains_material" not in types

    # ── Positive — must STILL produce gains_material ──────────────────────────

    def test_genuine_gains_material_still_extracted(self):
        text = "This move captures an opponent piece and wins material decisively."
        seeds = ["captures_count=1, net_gain=1 — wins material"]
        records = extract_claims(text, reasoning_seeds=seeds)
        types = [r.claim_type for r in records]
        assert "gains_material" in types

    def test_gains_material_phrase_positive_context(self):
        text = "The move gains material — opponent_can_recapture=false."
        seeds = ["captures_count=1, net_gain=1 — wins material"]
        records = extract_claims(text, reasoning_seeds=seeds)
        material = [r for r in records if r.claim_type == "gains_material"]
        assert len(material) == 1
        assert material[0].claim_status == ClaimStatus.SUPPORTED

    def test_net_gain_positive_still_extracted(self):
        text = "The net gain from this sequence is +2 pieces."
        seeds = ["captures_count=2, net_gain=2 — wins material"]
        records = extract_claims(text, reasoning_seeds=seeds)
        types = [r.claim_type for r in records]
        assert "gains_material" in types

    def test_captures_phrase_not_affected_by_polarity(self):
        # "captures a piece" is an action verb — polarity guard does not apply to it.
        text = "This move captures a piece at [4,3]."
        seeds = ["captures_count=1, net_gain=1 — wins material"]
        records = extract_claims(text, reasoning_seeds=seeds)
        types = [r.claim_type for r in records]
        assert "gains_material" in types


class TestMobilityPolarity:
    """Polarity and direction handling for mobility claims."""

    # ── "reduces our mobility by N" — must produce mobility_decrease ─────────

    def test_reduces_our_mobility_by_one_gives_decrease(self):
        text = "This move reduces our mobility by one, limiting our options."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "mobility_decrease" in types, (
            "\"reduces our mobility by one\" must produce mobility_decrease"
        )
        assert "mobility_increase" not in types, (
            "\"reduces our mobility by one\" must NOT produce mobility_increase"
        )

    def test_decreases_our_mobility_by_two_gives_decrease(self):
        text = "The move decreases our mobility by two."
        records = extract_claims(text)
        types = [r.claim_type for r in records]
        assert "mobility_decrease" in types
        assert "mobility_increase" not in types

    def test_mobility_decrease_redirect_with_opponent_seed_supported(self):
        # mobility_decrease seed_markers check opponent_mobility fields.
        # Redirect picks up seed support when the seed is opponent-mobility based.
        text = "It reduces our mobility by one move."
        seeds = [
            "opponent_mobility_before=8, opponent_mobility_after=5 — reduces opponent mobility by 3"
        ]
        records = extract_claims(text, reasoning_seeds=seeds)
        decrease = [r for r in records if r.claim_type == "mobility_decrease"]
        assert len(decrease) == 1
        assert decrease[0].claim_status == ClaimStatus.SUPPORTED
        assert decrease[0].source == "seed"

    def test_mobility_decrease_with_our_mobility_seed_is_supported(self):
        # Seed containing "decreases our mobility" matches the seed_marker added
        # in Fix 1 — status is SUPPORTED (seed-grounded).
        text = "It reduces our mobility by one move."
        seeds = [
            "our_mobility_before=8, our_mobility_after=7 — decreases our mobility by 1"
        ]
        records = extract_claims(text, reasoning_seeds=seeds)
        decrease = [r for r in records if r.claim_type == "mobility_decrease"]
        assert len(decrease) == 1
        assert decrease[0].claim_status == ClaimStatus.SUPPORTED
        assert "mobility_increase" not in {r.claim_type for r in records}

    def test_redirect_does_not_duplicate_when_decrease_already_present(self):
        # If mobility_decrease was already extracted from an explicit phrase,
        # the redirect must not add a second record.
        text = (
            "Reducing opponent mobility by 3 is a key benefit. "
            "It also reduces our mobility by one."
        )
        records = extract_claims(text)
        decrease_records = [r for r in records if r.claim_type == "mobility_decrease"]
        # Exactly one mobility_decrease record (not two)
        assert len(decrease_records) == 1

    # ── Positive — genuine mobility_increase must still extract ───────────────

    def test_increases_our_mobility_still_extracted(self):
        text = "This move increases our mobility significantly."
        seeds = ["our_mobility_before=6, our_mobility_after=9 — increases our mobility by 3"]
        records = extract_claims(text, reasoning_seeds=seeds)
        types = [r.claim_type for r in records]
        assert "mobility_increase" in types

    def test_improves_our_mobility_still_extracted(self):
        text = "The move improves our mobility, opening new paths."
        seeds = ["our_mobility_before=5, our_mobility_after=8 — increases our mobility"]
        records = extract_claims(text, reasoning_seeds=seeds)
        types = [r.claim_type for r in records]
        assert "mobility_increase" in types

    def test_increases_our_mobility_by_positive_context(self):
        # "increases our mobility by 2" — direction-positive context.
        text = "This advance increases our mobility by 2, opening the diagonal."
        seeds = ["our_mobility_before=6, our_mobility_after=8 — increases our mobility by 2"]
        records = extract_claims(text, reasoning_seeds=seeds)
        increase = [r for r in records if r.claim_type == "mobility_increase"]
        assert len(increase) == 1
        assert increase[0].claim_status == ClaimStatus.SUPPORTED
