"""
checkers/tests/test_adversity_claims.py

Phase 6 — adversity / losing-position claim taxonomy, extraction, and
verifier rules.

Covers the four claim types added in Phase 6:
    - material_deficit
    - threat_reduction
    - score_gap_advantage
    - mobility_asymmetry

Each claim type is exercised at three layers:
    1. taxonomy spec correctness (claim_taxonomy)
    2. extraction from reasoning text (claim_extractor)
    3. symbolic verification against facts (claim_verifier)

No runtime pipeline imports.  No LLM calls.  Deterministic.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any

import pytest

# Guard — record modules before importing evaluation packages.
_modules_before = set(sys.modules.keys())

from checkers.evaluation.claim_extractor import (
    ClaimRecord,
    extract_claims,
    _PHRASE_TABLE,
    _ENTRY_BY_TYPE,
)
from checkers.evaluation.claim_verifier import (
    _VERIFICATION_RULES,
    verify_claims,
)
from checkers.evaluation.claim_taxonomy import (
    TaxonomyCategory,
    get_claim_spec,
    is_verifiable_claim_type,
    claim_type_has_verifier,
)
from checkers.evaluation.reasoning_taxonomy import (
    ClaimStatus,
    ClaimVerifiability,
    HallucinationType,
)

_modules_after = set(sys.modules.keys())


# ---------------------------------------------------------------------------
# Runtime isolation guard
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
    new_modules = _modules_after - _modules_before
    for mod in new_modules:
        for prefix in _FORBIDDEN_RUNTIME_PREFIXES:
            assert not mod.startswith(prefix), (
                f"importing evaluation pulled in runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ADVERSITY_CLAIM_TYPES = (
    "material_deficit",
    "threat_reduction",
    "score_gap_advantage",
    "mobility_asymmetry",
)


def _verify_one(claim_type: str, facts: dict[str, Any]) -> ClaimStatus:
    """Run a single claim through the verifier and return its final status."""
    record = ClaimRecord(
        claim_type=claim_type,
        claim_status=ClaimStatus.UNSUPPORTED,
        claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
    )
    out = verify_claims([record], facts)
    return out[0].claim_status


# ---------------------------------------------------------------------------
# Taxonomy registration
# ---------------------------------------------------------------------------

class TestTaxonomyRegistration:
    """All Phase 6 adversity claim types must be registered in the taxonomy
    AND in the verifier dispatch table AND in the phrase table.
    """

    @pytest.mark.parametrize("claim_type", _ADVERSITY_CLAIM_TYPES)
    def test_claim_type_has_taxonomy_spec(self, claim_type):
        spec = get_claim_spec(claim_type)
        assert spec is not None, f"{claim_type} missing from taxonomy registry"

    @pytest.mark.parametrize("claim_type", _ADVERSITY_CLAIM_TYPES)
    def test_claim_type_has_verifier(self, claim_type):
        assert claim_type in _VERIFICATION_RULES, (
            f"{claim_type} missing from verifier dispatch table"
        )

    @pytest.mark.parametrize("claim_type", _ADVERSITY_CLAIM_TYPES)
    def test_claim_type_has_phrase_entry(self, claim_type):
        assert claim_type in _ENTRY_BY_TYPE, (
            f"{claim_type} missing from claim_extractor phrase table"
        )

    @pytest.mark.parametrize("claim_type", _ADVERSITY_CLAIM_TYPES)
    def test_claim_type_verifier_flag(self, claim_type):
        assert claim_type_has_verifier(claim_type)

    def test_material_deficit_is_verifiable(self):
        assert is_verifiable_claim_type("material_deficit")
        assert get_claim_spec("material_deficit").required_fact_fields == ("material_advantage",)

    def test_threat_reduction_requires_before_and_after(self):
        spec = get_claim_spec("threat_reduction")
        assert "our_pieces_threatened_before" in spec.required_fact_fields
        assert "our_pieces_threatened_after" in spec.required_fact_fields

    def test_score_gap_is_ambiguous_context_required(self):
        spec = get_claim_spec("score_gap_advantage")
        assert spec.category == TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED

    def test_mobility_asymmetry_requires_both_mobilities(self):
        spec = get_claim_spec("mobility_asymmetry")
        assert "opponent_mobility_before" in spec.required_fact_fields
        assert "our_mobility_before" in spec.required_fact_fields


# ---------------------------------------------------------------------------
# Material deficit
# ---------------------------------------------------------------------------

class TestMaterialDeficitVerifier:
    def test_supported_when_material_advantage_negative(self):
        assert _verify_one("material_deficit", {"material_advantage": -2}) == ClaimStatus.SUPPORTED

    def test_contradicted_when_material_advantage_zero(self):
        assert _verify_one("material_deficit", {"material_advantage": 0}) == ClaimStatus.CONTRADICTED

    def test_contradicted_when_material_advantage_positive(self):
        assert _verify_one("material_deficit", {"material_advantage": 3}) == ClaimStatus.CONTRADICTED

    def test_unsupported_when_field_absent(self):
        assert _verify_one("material_deficit", {}) == ClaimStatus.UNSUPPORTED


class TestMaterialDeficitExtraction:
    def test_behind_by_phrase_detected(self):
        records = extract_claims(
            "We are behind by 2 pieces; this move is the least harmful continuation.",
            reasoning_seeds=["material_advantage=-2 — behind by 2 piece(s)"],
            facts={"material_advantage": -2},
        )
        types = [r.claim_type for r in records]
        assert "material_deficit" in types

    def test_seed_path_status_supported(self):
        records = extract_claims(
            "We are behind by 2 pieces.",
            reasoning_seeds=["material_advantage=-2 — behind by 2 piece(s)"],
            facts={"material_advantage": -2},
        )
        rec = next(r for r in records if r.claim_type == "material_deficit")
        assert rec.claim_status == ClaimStatus.SUPPORTED
        assert rec.source == "seed"


# ---------------------------------------------------------------------------
# Threat reduction
# ---------------------------------------------------------------------------

class TestThreatReductionVerifier:
    def test_supported_when_threats_drop(self):
        facts = {
            "our_pieces_threatened_before": 2,
            "our_pieces_threatened_after": 1,
        }
        assert _verify_one("threat_reduction", facts) == ClaimStatus.SUPPORTED

    def test_contradicted_when_threats_unchanged(self):
        facts = {
            "our_pieces_threatened_before": 1,
            "our_pieces_threatened_after": 1,
        }
        assert _verify_one("threat_reduction", facts) == ClaimStatus.CONTRADICTED

    def test_contradicted_when_threats_increase(self):
        facts = {
            "our_pieces_threatened_before": 1,
            "our_pieces_threatened_after": 2,
        }
        assert _verify_one("threat_reduction", facts) == ClaimStatus.CONTRADICTED

    def test_unsupported_when_field_missing(self):
        assert _verify_one("threat_reduction", {}) == ClaimStatus.UNSUPPORTED
        assert _verify_one(
            "threat_reduction", {"our_pieces_threatened_before": 1},
        ) == ClaimStatus.UNSUPPORTED


class TestThreatReductionExtraction:
    def test_reduces_threatened_pieces_phrase(self):
        records = extract_claims(
            "This move reduces threatened pieces from 2 to 1, improving immediate safety.",
            reasoning_seeds=[
                "reduces threatened pieces from 2 to 1 — move improves immediate safety"
            ],
            facts={
                "our_pieces_threatened_before": 2,
                "our_pieces_threatened_after": 1,
            },
        )
        assert any(r.claim_type == "threat_reduction" for r in records)
        rec = next(r for r in records if r.claim_type == "threat_reduction")
        assert rec.claim_status == ClaimStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Score gap advantage
# ---------------------------------------------------------------------------

class TestScoreGapVerifier:
    def test_supported_when_minimax_present(self):
        assert _verify_one("score_gap_advantage", {"minimax_score": 5.0}) == ClaimStatus.SUPPORTED

    def test_supported_when_minimax_zero(self):
        # Zero is a valid numeric score.
        assert _verify_one("score_gap_advantage", {"minimax_score": 0.0}) == ClaimStatus.SUPPORTED

    def test_unsupported_when_minimax_missing(self):
        assert _verify_one("score_gap_advantage", {}) == ClaimStatus.UNSUPPORTED

    def test_never_contradicted(self):
        """Verifier deliberately never returns CONTRADICTED — a single move's
        facts cannot reproduce the gap-vs-best-alternative figure."""
        for facts in (
            {"minimax_score": 5.0},
            {"minimax_score": -100.0},
            {"minimax_score": 0.0},
            {},
        ):
            status = _verify_one("score_gap_advantage", facts)
            assert status != ClaimStatus.CONTRADICTED


class TestScoreGapExtraction:
    def test_points_better_than_phrase(self):
        records = extract_claims(
            "Chosen move scores 25.0 points better than next-best option [3].",
            reasoning_seeds=[
                "chosen move scores 25.0 points better than next-best option [3] "
                "(minimax: -10.0 vs -35.0)"
            ],
            facts={"minimax_score": -10.0},
        )
        assert any(r.claim_type == "score_gap_advantage" for r in records)


# ---------------------------------------------------------------------------
# Mobility asymmetry
# ---------------------------------------------------------------------------

class TestMobilityAsymmetryVerifier:
    def test_supported_when_gap_at_least_three(self):
        facts = {"opponent_mobility_before": 10, "our_mobility_before": 7}
        assert _verify_one("mobility_asymmetry", facts) == ClaimStatus.SUPPORTED

    def test_supported_at_exact_threshold(self):
        facts = {"opponent_mobility_before": 8, "our_mobility_before": 5}
        assert _verify_one("mobility_asymmetry", facts) == ClaimStatus.SUPPORTED

    def test_unsupported_when_gap_below_threshold_but_nonnegative(self):
        facts = {"opponent_mobility_before": 8, "our_mobility_before": 6}
        assert _verify_one("mobility_asymmetry", facts) == ClaimStatus.UNSUPPORTED

    def test_contradicted_when_we_have_more_mobility(self):
        facts = {"opponent_mobility_before": 5, "our_mobility_before": 8}
        assert _verify_one("mobility_asymmetry", facts) == ClaimStatus.CONTRADICTED

    def test_unsupported_when_field_missing(self):
        assert _verify_one(
            "mobility_asymmetry", {"opponent_mobility_before": 10},
        ) == ClaimStatus.UNSUPPORTED


class TestMobilityAsymmetryExtraction:
    def test_structural_disadvantage_phrase_extracted(self):
        records = extract_claims(
            "Our position carries a structural disadvantage in available options.",
            reasoning_seeds=[
                "opponent_mobility_before=10 vs our_mobility_before=6 — "
                "structural disadvantage in available options"
            ],
            facts={"opponent_mobility_before": 10, "our_mobility_before": 6},
        )
        assert any(r.claim_type == "mobility_asymmetry" for r in records)
        rec = next(r for r in records if r.claim_type == "mobility_asymmetry")
        assert rec.claim_status == ClaimStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Integration: extractor + verifier together on a realistic adversity paragraph
# ---------------------------------------------------------------------------

class TestEndToEndAdversityVerification:
    """A full adversity-context paragraph should produce SUPPORTED records
    for each of the four claim types when seeds and facts both agree.
    """

    REASONING = (
        "We are behind by 2 pieces and the opponent has more options. "
        "This move reduces threatened pieces from 2 to 1, improving "
        "immediate safety, and scores 25.0 points better than the "
        "next-best option."
    )
    SEEDS = [
        "material_advantage=-2 — behind by 2 piece(s)",
        "opponent_mobility_before=10 vs our_mobility_before=6 — "
        "structural disadvantage in available options",
        "reduces threatened pieces from 2 to 1 — move improves immediate safety",
        "chosen move scores 25.0 points better than next-best option [3] "
        "(minimax: -10.0 vs -35.0)",
    ]
    FACTS = {
        "material_advantage": -2,
        "opponent_mobility_before": 10,
        "our_mobility_before": 6,
        "our_pieces_threatened_before": 2,
        "our_pieces_threatened_after": 1,
        "minimax_score": -10.0,
    }

    def test_all_four_claim_types_supported(self):
        records = extract_claims(self.REASONING, self.SEEDS, self.FACTS)
        verified = verify_claims(records, self.FACTS)
        by_type = {r.claim_type: r for r in verified}
        for claim_type in _ADVERSITY_CLAIM_TYPES:
            assert claim_type in by_type, f"{claim_type} not extracted"
            assert by_type[claim_type].claim_status == ClaimStatus.SUPPORTED, (
                f"{claim_type} expected SUPPORTED, got {by_type[claim_type].claim_status}"
            )

    def test_no_unsupported_or_contradicted_adversity_claims(self):
        records = extract_claims(self.REASONING, self.SEEDS, self.FACTS)
        verified = verify_claims(records, self.FACTS)
        for r in verified:
            if r.claim_type in _ADVERSITY_CLAIM_TYPES:
                assert r.claim_status == ClaimStatus.SUPPORTED, (
                    f"{r.claim_type} unexpectedly {r.claim_status}"
                )

    def test_contradicted_when_paragraph_lies_about_material(self):
        """If the paragraph claims a deficit but material_advantage >= 0,
        material_deficit must be marked CONTRADICTED."""
        facts = {**self.FACTS, "material_advantage": 1}
        records = extract_claims(self.REASONING, self.SEEDS, facts)
        verified = verify_claims(records, facts)
        material = next(
            (r for r in verified if r.claim_type == "material_deficit"), None,
        )
        assert material is not None
        assert material.claim_status == ClaimStatus.CONTRADICTED
        assert material.hallucination_type == HallucinationType.FACTUAL_CONTRADICTION
