# checkers/tests/test_phase2_verifier_completeness.py
#
# Phase 2 fix — unit tests for verifier output completeness and
# schema-leak → contradiction pipeline routing.
#
# Coverage:
#   1. ClaimRecord attribute names (matched_phrase / claim_status, not
#      claim_text / status) — guards against trace-tool regressions.
#   2. Schema-leak (agree) is now CONTRADICTED, not UNSUPPORTED.
#   3. Schema-leak (disagree) stays CONTRADICTED (pre-existing, no regression).
#   4. All schema leaks carry hallucination_type != None.
#   5. Agreeing schema leak appears in contradictions_only().
#   6. Agreeing schema leak appears in contradiction_strings().
#   7. Refinement invariant: runtime and evaluator agree on agreeing schema leak.
#   8. matched_phrase is populated (not None) for schema leaks.
#   9. claim_type is populated for schema leaks (schema_leak_<field>).
#  10. No false positives: text without schema pattern returns no schema leak.

from __future__ import annotations

import pytest

from checkers.evaluation.unified_verifier import (
    verify_all,
    contradictions_only,
    contradiction_strings,
    assert_runtime_evaluator_agreement,
)
from checkers.evaluation.reasoning_taxonomy import ClaimStatus, HallucinationType
from checkers.agents.ranker_agent import _check_reasoning_truthfulness


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Text with a schema-leak whose value AGREES with the fact.
_TEXT_AGREE = "The fact opponent_can_recapture=false implies the position is safe."
_FACTS_AGREE = {"opponent_can_recapture": False}

# Text with a schema-leak whose value DISAGREES with the fact.
_TEXT_DISAGREE = "The position creates_immediate_threat=true after the move."
_FACTS_DISAGREE = {"creates_immediate_threat": False}

# Text with a schema-leak where no matching fact exists.
_TEXT_NO_FACT = "The engine reports net_gain=1 for this line."
_FACTS_EMPTY: dict = {}

# Clean text — no schema-leak pattern.
_TEXT_CLEAN = "The chosen move advances safely and improves piece activity."
_FACTS_CLEAN = {"opponent_can_recapture": False, "net_gain": 0}


# ═════════════════════════════════════════════════════════════════════════════
# 1. ClaimRecord attribute-name contract
# ═════════════════════════════════════════════════════════════════════════════

class TestClaimRecordAttributeContract:
    """Guards against trace-tool using wrong attribute names."""

    def test_has_matched_phrase_not_claim_text(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema, "Expected at least one schema_leak record"
        assert hasattr(schema[0], "matched_phrase"), "ClaimRecord has no matched_phrase"
        assert not hasattr(schema[0], "claim_text"), "ClaimRecord should NOT have claim_text"

    def test_has_claim_status_not_status(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert hasattr(schema[0], "claim_status"), "ClaimRecord has no claim_status"
        assert not hasattr(schema[0], "status"), "ClaimRecord should NOT have status"

    def test_has_claim_verifiability_not_verifiable_type(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert hasattr(schema[0], "claim_verifiability")
        assert not hasattr(schema[0], "verifiable_type"), \
            "ClaimRecord should NOT have verifiable_type"

    def test_matched_phrase_is_non_empty_string(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert isinstance(schema[0].matched_phrase, str)
        assert len(schema[0].matched_phrase) > 0


# ═════════════════════════════════════════════════════════════════════════════
# 2. Schema-leak status: agreeing value is CONTRADICTED
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaLeakAgreeCONTRADICTED:
    def test_agree_is_contradicted(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema, "Expected schema_leak record"
        assert schema[0].claim_status == ClaimStatus.CONTRADICTED

    def test_agree_no_fact_is_contradicted(self):
        # Even when no fact is available, the schema leak is still a violation.
        records = verify_all(_TEXT_NO_FACT, facts=_FACTS_EMPTY)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert schema[0].claim_status == ClaimStatus.CONTRADICTED


# ═════════════════════════════════════════════════════════════════════════════
# 3. Schema-leak status: disagreeing value stays CONTRADICTED (no regression)
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaLeakDisagreeCONTRADICTED:
    def test_disagree_is_still_contradicted(self):
        records = verify_all(_TEXT_DISAGREE, facts=_FACTS_DISAGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert schema[0].claim_status == ClaimStatus.CONTRADICTED


# ═════════════════════════════════════════════════════════════════════════════
# 4. hallucination_type is always set for schema leaks
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaLeakHallucinationType:
    def test_agree_has_hallucination_type(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert schema[0].hallucination_type is not None

    def test_agree_hallucination_type_is_instruction_inconsistency(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert schema[0].hallucination_type == HallucinationType.INSTRUCTION_INCONSISTENCY

    def test_disagree_has_hallucination_type(self):
        records = verify_all(_TEXT_DISAGREE, facts=_FACTS_DISAGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert schema[0].hallucination_type is not None


# ═════════════════════════════════════════════════════════════════════════════
# 5. Agreeing schema leaks enter contradictions_only()
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaLeakEntersContradictionsOnly:
    def test_agree_in_contradictions_only(self):
        result = contradictions_only(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in result if r.claim_type.startswith("schema_leak_")]
        assert schema, "Schema-leak (agree) must appear in contradictions_only()"

    def test_no_fact_in_contradictions_only(self):
        result = contradictions_only(_TEXT_NO_FACT, facts=_FACTS_EMPTY)
        schema = [r for r in result if r.claim_type.startswith("schema_leak_")]
        assert schema

    def test_clean_text_not_in_contradictions_only(self):
        result = contradictions_only(_TEXT_CLEAN, facts=_FACTS_CLEAN)
        schema = [r for r in result if r.claim_type.startswith("schema_leak_")]
        assert schema == []


# ═════════════════════════════════════════════════════════════════════════════
# 6. Agreeing schema leaks enter contradiction_strings()
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaLeakEntersContradictionStrings:
    def test_agree_produces_contradiction_string(self):
        strings = contradiction_strings(_TEXT_AGREE, facts=_FACTS_AGREE)
        assert strings, "Schema-leak (agree) must produce at least one string"
        assert any("opponent_can_recapture" in s for s in strings)

    def test_agree_string_contains_reasoning_contradiction_prefix(self):
        strings = contradiction_strings(_TEXT_AGREE, facts=_FACTS_AGREE)
        assert any(s.startswith("REASONING_CONTRADICTION:") for s in strings)

    def test_no_fact_produces_contradiction_string(self):
        strings = contradiction_strings(_TEXT_NO_FACT, facts=_FACTS_EMPTY)
        assert strings
        assert any("net_gain" in s for s in strings)

    def test_clean_text_produces_no_schema_string(self):
        strings = contradiction_strings(_TEXT_CLEAN, facts=_FACTS_CLEAN)
        schema_strings = [s for s in strings if "schema" in s.lower()]
        assert schema_strings == []


# ═════════════════════════════════════════════════════════════════════════════
# 7. Runtime–evaluator invariant holds for agreeing schema leak
# ═════════════════════════════════════════════════════════════════════════════

class TestRuntimeEvaluatorInvariantSchemaLeak:
    def test_both_flag_agree_case(self):
        runtime = _check_reasoning_truthfulness(_TEXT_AGREE, _FACTS_AGREE)
        # Must not raise RuntimeEvaluatorDisagreement.
        assert_runtime_evaluator_agreement(
            runtime, _TEXT_AGREE, facts=_FACTS_AGREE,
        )

    def test_both_flag_no_fact_case(self):
        runtime = _check_reasoning_truthfulness(_TEXT_NO_FACT, _FACTS_EMPTY)
        assert_runtime_evaluator_agreement(
            runtime, _TEXT_NO_FACT, facts=_FACTS_EMPTY,
        )

    def test_both_clean_for_clean_text(self):
        runtime = _check_reasoning_truthfulness(_TEXT_CLEAN, _FACTS_CLEAN)
        assert_runtime_evaluator_agreement(
            runtime, _TEXT_CLEAN, facts=_FACTS_CLEAN,
        )


# ═════════════════════════════════════════════════════════════════════════════
# 8 & 9. Populated fields: matched_phrase and claim_type
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaLeakPopulatedFields:
    def test_matched_phrase_contains_field_name(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert "opponent_can_recapture" in schema[0].matched_phrase

    def test_claim_type_is_schema_leak_field(self):
        records = verify_all(_TEXT_AGREE, facts=_FACTS_AGREE)
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema
        assert schema[0].claim_type == "schema_leak_opponent_can_recapture"


# ═════════════════════════════════════════════════════════════════════════════
# 10. No false positives on clean prose
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaLeakNoFalsePositives:
    def test_plain_prose_no_schema_leak(self):
        text = "The chosen move secures a safe capture and improves piece activity."
        records = verify_all(text, facts={"captures_count": 1, "net_gain": 1})
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema == []

    def test_value_equals_sign_in_narrative_not_flagged(self):
        # "=" inside a quoted phrase or normal prose must not trigger schema-leak.
        text = "Mobility equals the number of available moves."
        records = verify_all(text, facts={})
        schema = [r for r in records if r.claim_type.startswith("schema_leak_")]
        assert schema == []
