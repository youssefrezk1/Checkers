# checkers/tests/test_claim_taxonomy.py
#
# Tests for checkers/evaluation/claim_taxonomy.py and
#          checkers/evaluation/claim_recall_audit.py
#
# Coverage
# --------
#  1. Isolation — no runtime pipeline imports.
#  2. Registry completeness — taxonomy contains all extractor claim types.
#  3. VERIFIABLE invariant — every VERIFIABLE spec has non-empty fact fields.
#  4. Verifier dispatch coverage — every spec with verifier_exists=True
#     is present in claim_verifier._VERIFICATION_RULES.
#  5. Category exclusion — NON_VERIFIABLE_VAGUE, FORBIDDEN_UNGROUNDED, and
#     SCHEMA_LEAK types are NOT categorized as VERIFIABLE.
#  6. Audit immutability — audit_claim_recall does not alter claim statuses.
#  7. Backward compatibility — evaluate_turn results are structurally
#     unchanged after importing claim_taxonomy.

import sys
from dataclasses import replace
from typing import Optional

import pytest

# ── Isolation guard ───────────────────────────────────────────────────────────
_modules_before = set(sys.modules.keys())

from checkers.evaluation.claim_taxonomy import (
    TaxonomyCategory,
    ClaimSpec,
    _CLAIM_REGISTRY,
    get_claim_spec,
    is_verifiable_claim_type,
    required_fields_for_claim,
    claim_type_has_verifier,
)
from checkers.evaluation.claim_recall_audit import audit_claim_recall
from checkers.evaluation.claim_extractor import (
    ClaimRecord,
    extract_claims,
    _PHRASE_TABLE,
)
from checkers.evaluation.claim_verifier import verify_claims, _VERIFICATION_RULES
from checkers.evaluation.turn_evaluator import evaluate_turn, TurnEvaluationRecord
from checkers.evaluation.reasoning_taxonomy import (
    ClaimStatus,
    ClaimVerifiability,
    HallucinationType,
)

_modules_after = set(sys.modules.keys())

_FORBIDDEN_RUNTIME_PREFIXES = (
    "checkers.engine",
    "checkers.agents",
    "checkers.graph",
    "checkers.nodes",
    "checkers.search",
)


# ---------------------------------------------------------------------------
# 1. Isolation
# ---------------------------------------------------------------------------

def test_no_runtime_pipeline_imports():
    """claim_taxonomy and claim_recall_audit must not pull in runtime modules."""
    new_mods = _modules_after - _modules_before
    for mod in new_mods:
        for prefix in _FORBIDDEN_RUNTIME_PREFIXES:
            assert not mod.startswith(prefix), (
                f"claim_taxonomy/audit pulled in runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# 2. Registry completeness
# ---------------------------------------------------------------------------

def test_registry_contains_all_phrase_table_claim_types():
    """Every claim type in the extractor phrase table must appear in _CLAIM_REGISTRY."""
    extractor_types = {e.claim_type for e in _PHRASE_TABLE}
    registry_types  = set(_CLAIM_REGISTRY.keys())

    missing = extractor_types - registry_types
    assert not missing, (
        f"Claim types in _PHRASE_TABLE but missing from _CLAIM_REGISTRY: {missing}"
    )


def test_registry_has_no_phantom_types():
    """_CLAIM_REGISTRY must not define types absent from the extractor phrase table."""
    extractor_types = {e.claim_type for e in _PHRASE_TABLE}
    registry_types  = set(_CLAIM_REGISTRY.keys())

    phantom = registry_types - extractor_types
    assert not phantom, (
        f"Claim types in _CLAIM_REGISTRY but absent from _PHRASE_TABLE: {phantom}"
    )


def test_registry_has_expected_count():
    """Registry must have exactly as many entries as the phrase table."""
    assert len(_CLAIM_REGISTRY) == len(_PHRASE_TABLE), (
        f"Registry size {len(_CLAIM_REGISTRY)} != phrase table size {len(_PHRASE_TABLE)}"
    )


# ---------------------------------------------------------------------------
# 3. VERIFIABLE invariant
# ---------------------------------------------------------------------------

def test_every_verifiable_spec_has_required_fact_fields():
    """Every VERIFIABLE claim type must declare at least one required fact field."""
    for ct, spec in _CLAIM_REGISTRY.items():
        if spec.category == TaxonomyCategory.VERIFIABLE:
            assert len(spec.required_fact_fields) > 0, (
                f"VERIFIABLE claim type {ct!r} has empty required_fact_fields"
            )


def test_non_verifiable_types_have_empty_required_fields():
    """NON_VERIFIABLE_VAGUE and FORBIDDEN_UNGROUNDED types must have no required fields."""
    no_fields_categories = {
        TaxonomyCategory.NON_VERIFIABLE_VAGUE,
        TaxonomyCategory.FORBIDDEN_UNGROUNDED,
        TaxonomyCategory.SCHEMA_LEAK,
    }
    for ct, spec in _CLAIM_REGISTRY.items():
        if spec.category in no_fields_categories:
            assert len(spec.required_fact_fields) == 0, (
                f"Category {spec.category.value} claim type {ct!r} "
                f"unexpectedly has required_fact_fields: {spec.required_fact_fields}"
            )


# ---------------------------------------------------------------------------
# 4. Verifier dispatch coverage
# ---------------------------------------------------------------------------

def test_specs_with_verifier_exists_are_in_dispatch():
    """
    Every spec with verifier_exists=True must have an entry in
    claim_verifier._VERIFICATION_RULES.
    """
    for ct, spec in _CLAIM_REGISTRY.items():
        if spec.verifier_exists:
            assert ct in _VERIFICATION_RULES, (
                f"Spec {ct!r} has verifier_exists=True but is absent from "
                f"_VERIFICATION_RULES"
            )


def test_specs_without_verifier_are_absent_from_dispatch():
    """Claim types marked verifier_exists=False must not appear in _VERIFICATION_RULES."""
    for ct, spec in _CLAIM_REGISTRY.items():
        if not spec.verifier_exists:
            assert ct not in _VERIFICATION_RULES, (
                f"Spec {ct!r} has verifier_exists=False but IS in _VERIFICATION_RULES"
            )


# ---------------------------------------------------------------------------
# 5. Category exclusion (vague / forbidden / schema-leak not VERIFIABLE)
# ---------------------------------------------------------------------------

def test_non_verifiable_vague_not_marked_verifiable():
    non_verifiable = [
        ct for ct, spec in _CLAIM_REGISTRY.items()
        if spec.category == TaxonomyCategory.NON_VERIFIABLE_VAGUE
    ]
    assert len(non_verifiable) > 0, "Expected at least one NON_VERIFIABLE_VAGUE type"
    for ct in non_verifiable:
        assert not is_verifiable_claim_type(ct), (
            f"NON_VERIFIABLE_VAGUE type {ct!r} incorrectly marked as VERIFIABLE"
        )


def test_forbidden_ungrounded_not_marked_verifiable():
    forbidden = [
        ct for ct, spec in _CLAIM_REGISTRY.items()
        if spec.category == TaxonomyCategory.FORBIDDEN_UNGROUNDED
    ]
    assert len(forbidden) > 0, "Expected at least one FORBIDDEN_UNGROUNDED type"
    for ct in forbidden:
        assert not is_verifiable_claim_type(ct), (
            f"FORBIDDEN_UNGROUNDED type {ct!r} incorrectly marked as VERIFIABLE"
        )


def test_known_unverifiable_types_are_not_verifiable():
    """Spot-check: the three always-VAGUE strategic types must not be VERIFIABLE."""
    for ct in ("positional_pressure", "strategic_initiative", "long_term_compensation"):
        spec = get_claim_spec(ct)
        assert spec is not None, f"Missing registry entry for {ct!r}"
        assert spec.category != TaxonomyCategory.VERIFIABLE, (
            f"{ct!r} must not be VERIFIABLE"
        )


def test_positional_pressure_is_forbidden_ungrounded():
    """positional_pressure carries 'structural pressure' — explicitly forbidden vocab."""
    spec = get_claim_spec("positional_pressure")
    assert spec is not None
    assert spec.category == TaxonomyCategory.FORBIDDEN_UNGROUNDED


# ---------------------------------------------------------------------------
# 6. Audit immutability
# ---------------------------------------------------------------------------

def _make_claim(
    claim_type: str,
    status: ClaimStatus = ClaimStatus.SUPPORTED,
    verifiability: ClaimVerifiability = ClaimVerifiability.FULLY_VERIFIABLE,
    hallucination: Optional[HallucinationType] = None,
) -> ClaimRecord:
    return ClaimRecord(
        claim_type=claim_type,
        claim_status=status,
        claim_verifiability=verifiability,
        hallucination_type=hallucination,
        source="seed",
    )


def test_audit_does_not_alter_claim_status():
    """audit_claim_recall must leave every claim_status field unchanged."""
    records = [
        _make_claim("avoids_recapture", ClaimStatus.SUPPORTED),
        _make_claim("gains_material",   ClaimStatus.CONTRADICTED),
        _make_claim("positional_pressure", ClaimStatus.VAGUE,
                    ClaimVerifiability.UNVERIFIABLE, HallucinationType.OVERCLAIM),
    ]
    statuses_before = [r.claim_status for r in records]

    audit_claim_recall(records, seeds=[], facts={})

    statuses_after = [r.claim_status for r in records]
    assert statuses_before == statuses_after, (
        "audit_claim_recall altered claim_status values"
    )


def test_audit_does_not_alter_hallucination_type():
    """audit_claim_recall must leave hallucination_type fields unchanged."""
    records = [
        _make_claim("gains_material", ClaimStatus.CONTRADICTED,
                    hallucination=HallucinationType.FACTUAL_CONTRADICTION),
    ]
    hallucinations_before = [r.hallucination_type for r in records]

    audit_claim_recall(records)

    hallucinations_after = [r.hallucination_type for r in records]
    assert hallucinations_before == hallucinations_after


def test_audit_does_not_mutate_input_list():
    """audit_claim_recall must not append, remove, or reorder the input list."""
    records = [
        _make_claim("avoids_recapture"),
        _make_claim("piece_isolated"),
    ]
    original_ids = [id(r) for r in records]
    original_len = len(records)

    audit_claim_recall(records)

    assert len(records) == original_len
    assert [id(r) for r in records] == original_ids


def test_audit_empty_claims_returns_empty_lists():
    report = audit_claim_recall([])
    assert report["extracted_claim_types"] == []
    assert report["verifiable_claim_types_present"] == []
    assert report["ambiguous_or_nonverifiable_types"] == []


def test_audit_report_keys():
    """Report always contains exactly the four documented keys."""
    report = audit_claim_recall([])
    expected_keys = {
        "extracted_claim_types",
        "verifiable_claim_types_present",
        "missing_verifier_types",
        "ambiguous_or_nonverifiable_types",
    }
    assert set(report.keys()) == expected_keys


def test_audit_verifiable_present_subset_of_extracted():
    """verifiable_claim_types_present is always a subset of extracted_claim_types."""
    records = [
        _make_claim("avoids_recapture"),
        _make_claim("near_promotion", ClaimVerifiability.PARTIALLY_VERIFIABLE),
        _make_claim("positional_pressure", ClaimVerifiability.UNVERIFIABLE),
    ]
    report = audit_claim_recall(records)
    extracted_set = set(report["extracted_claim_types"])
    for ct in report["verifiable_claim_types_present"]:
        assert ct in extracted_set, (
            f"{ct!r} in verifiable_present but not in extracted_claim_types"
        )


def test_audit_missing_verifier_types_not_in_extracted():
    """missing_verifier_types contains only types absent from extracted_claims."""
    records = [_make_claim("avoids_recapture")]
    report = audit_claim_recall(records)
    extracted_set = set(report["extracted_claim_types"])
    for ct in report["missing_verifier_types"]:
        assert ct not in extracted_set, (
            f"{ct!r} in missing_verifier_types but also in extracted_claim_types"
        )


def test_audit_ambiguous_not_verifiable():
    """ambiguous_or_nonverifiable_types must not include VERIFIABLE claim types."""
    records = [
        _make_claim("avoids_recapture"),           # VERIFIABLE
        _make_claim("near_promotion"),             # AMBIGUOUS_CONTEXT_REQUIRED
        _make_claim("strategic_initiative"),       # NON_VERIFIABLE_VAGUE
        _make_claim("positional_pressure"),        # FORBIDDEN_UNGROUNDED
    ]
    report = audit_claim_recall(records)
    for ct in report["ambiguous_or_nonverifiable_types"]:
        assert not is_verifiable_claim_type(ct), (
            f"{ct!r} appears in ambiguous_or_nonverifiable but is VERIFIABLE"
        )


def test_audit_unknown_type_in_extracted_not_elsewhere():
    """Unknown claim types appear in extracted_claim_types but not in other lists."""
    unknown_record = ClaimRecord(
        claim_type="__unknown_future_type__",
        claim_status=ClaimStatus.UNSUPPORTED,
        claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        source="unsupported_phrase",
    )
    report = audit_claim_recall([unknown_record])
    assert "__unknown_future_type__" in report["extracted_claim_types"]
    assert "__unknown_future_type__" not in report["verifiable_claim_types_present"]
    assert "__unknown_future_type__" not in report["missing_verifier_types"]
    assert "__unknown_future_type__" not in report["ambiguous_or_nonverifiable_types"]


# ---------------------------------------------------------------------------
# 7. Backward compatibility — evaluate_turn results unchanged
# ---------------------------------------------------------------------------

def test_evaluate_turn_returns_turn_evaluation_record():
    """evaluate_turn still returns a TurnEvaluationRecord after taxonomy import."""
    result = evaluate_turn(
        reasoning_text="The move avoids recapture and gains material.",
        reasoning_seeds=[
            "opponent_can_recapture=false",
            "captures_count=1, net_gain=1",
        ],
        facts={
            "opponent_can_recapture": False,
            "net_gain": 1,
            "captures_count": 1,
        },
        ranker_diagnostics={},
        turn_id="test_compat_turn",
    )
    assert isinstance(result, TurnEvaluationRecord)


def test_evaluate_turn_count_consistency():
    """supported + contradicted + unsupported + vague == total_claims (unchanged)."""
    result = evaluate_turn(
        reasoning_text="The move avoids recapture and gains material.",
        reasoning_seeds=[
            "opponent_can_recapture=false",
            "captures_count=1, net_gain=1",
        ],
        facts={
            "opponent_can_recapture": False,
            "net_gain": 1,
            "captures_count": 1,
        },
        ranker_diagnostics={},
        turn_id="test_count_consistency",
    )
    total_from_counts = (
        result.supported_count
        + result.contradicted_count
        + result.unsupported_count
        + result.vague_count
    )
    assert total_from_counts == result.total_claims


def test_evaluate_turn_claims_fire_correctly():
    """avoids_recapture and gains_material should both fire as SUPPORTED."""
    result = evaluate_turn(
        reasoning_text="The move avoids recapture and gains material.",
        reasoning_seeds=[
            "opponent_can_recapture=false",
            "captures_count=1, net_gain=1",
        ],
        facts={
            "opponent_can_recapture": False,
            "net_gain": 1,
            "captures_count": 1,
        },
        ranker_diagnostics={},
        turn_id="test_fire",
    )
    fired_types = {c.claim_type for c in result.claims}
    assert "avoids_recapture" in fired_types
    assert "gains_material" in fired_types
    assert result.supported_count >= 2


def test_evaluate_turn_with_strategic_claim_gives_vague():
    """Strategic claims still produce vague_count > 0 (taxonomy has no side effects)."""
    result = evaluate_turn(
        reasoning_text="This applies strategic pressure on the opponent.",
        reasoning_seeds=[],
        facts={},
        ranker_diagnostics={},
        turn_id="test_strategic_vague",
    )
    assert result.vague_count > 0
    assert result.supported_count == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_get_claim_spec_returns_none_for_unknown():
    assert get_claim_spec("not_a_real_claim") is None


def test_get_claim_spec_returns_correct_spec():
    spec = get_claim_spec("avoids_recapture")
    assert spec is not None
    assert spec.claim_type == "avoids_recapture"
    assert spec.category == TaxonomyCategory.VERIFIABLE
    assert "opponent_can_recapture" in spec.required_fact_fields
    assert spec.verifier_exists is True
    assert spec.polarity_sensitive is True
    assert spec.entity_requirement == "opponent"


def test_required_fields_for_unknown_returns_empty_tuple():
    assert required_fields_for_claim("not_a_real_claim") == ()


def test_required_fields_for_verifiable_claim():
    fields = required_fields_for_claim("gains_material")
    assert "net_gain" in fields
    assert "captures_count" in fields


def test_claim_type_has_verifier_for_unknown():
    assert claim_type_has_verifier("not_a_real_claim") is False


def test_claim_type_has_verifier_true_for_verifiable():
    assert claim_type_has_verifier("avoids_recapture") is True
    assert claim_type_has_verifier("gains_material") is True
    assert claim_type_has_verifier("minimax_confirmation") is True


def test_claim_type_has_verifier_true_after_phase41():
    """Phase 4.1 added verifiers for the three previously-unverified tactical types."""
    assert claim_type_has_verifier("shot_sequence_or_multi_jump") is True
    assert claim_type_has_verifier("blocks_landing_square") is True
    assert claim_type_has_verifier("forced_opponent_jump") is True


def test_claim_type_has_verifier_true_for_always_vague():
    """Always-VAGUE strategic types still have a rule (returns VAGUE)."""
    assert claim_type_has_verifier("positional_pressure") is True
    assert claim_type_has_verifier("strategic_initiative") is True
    assert claim_type_has_verifier("long_term_compensation") is True
