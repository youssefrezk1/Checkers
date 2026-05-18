# checkers/tests/test_reasoning_taxonomy.py
#
# Tests for checkers/evaluation/reasoning_taxonomy.py
#
# PURPOSE
# -------
# Verify that:
#   1. All expected enum labels exist with their exact lowercase string values.
#   2. No runtime pipeline modules are imported (evaluation is offline-only).
#   3. Each enum is a subclass of both str and Enum (JSON-serialisable).
#
# These tests have NO side effects and require NO external services.

import sys
from enum import Enum

import pytest

# ── Guard: ensure no runtime pipeline modules leak in ────────────────────────
# Record the module set before import so we can diff after.
_modules_before = set(sys.modules.keys())

from checkers.evaluation.reasoning_taxonomy import (
    ClaimVerifiability,
    ClaimStatus,
    HallucinationType,
    SeedRiskType,
    TrajectoryEventType,
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
    Importing reasoning_taxonomy must not pull in any runtime pipeline module.
    This guarantees the evaluation package can be used in offline environments
    without API keys, LLM backends, or board state.
    """
    new_modules = _modules_after - _modules_before
    for mod in new_modules:
        for prefix in _FORBIDDEN_RUNTIME_PREFIXES:
            assert not mod.startswith(prefix), (
                f"Importing reasoning_taxonomy pulled in runtime module: {mod!r}. "
                "Evaluation taxonomy must remain isolated from the pipeline."
            )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _assert_enum_is_str_enum(enum_cls):
    """Every taxonomy enum must be both str and Enum for JSON serialisability."""
    assert issubclass(enum_cls, str), f"{enum_cls.__name__} must subclass str"
    assert issubclass(enum_cls, Enum), f"{enum_cls.__name__} must subclass Enum"


def _assert_value_is_lowercase(member):
    """All enum values must be lowercase strings (no uppercase, no spaces except within)."""
    assert isinstance(member.value, str), (
        f"{member} value must be a str, got {type(member.value)}"
    )
    assert member.value == member.value.lower(), (
        f"{member} has non-lowercase value: {member.value!r}"
    )


# ---------------------------------------------------------------------------
# 1. ClaimVerifiability
# ---------------------------------------------------------------------------

class TestClaimVerifiability:

    def test_is_str_enum(self):
        _assert_enum_is_str_enum(ClaimVerifiability)

    def test_expected_labels_exist(self):
        expected = {
            "FULLY_VERIFIABLE",
            "PARTIALLY_VERIFIABLE",
            "UNVERIFIABLE",
        }
        actual = {m.name for m in ClaimVerifiability}
        assert actual == expected, (
            f"ClaimVerifiability labels mismatch.\n"
            f"  Missing : {expected - actual}\n"
            f"  Extra   : {actual - expected}"
        )

    def test_values_are_lowercase_strings(self):
        for member in ClaimVerifiability:
            _assert_value_is_lowercase(member)

    def test_exact_values(self):
        assert ClaimVerifiability.FULLY_VERIFIABLE   == "fully_verifiable"
        assert ClaimVerifiability.PARTIALLY_VERIFIABLE == "partially_verifiable"
        assert ClaimVerifiability.UNVERIFIABLE        == "unverifiable"

    def test_stable_count(self):
        assert len(ClaimVerifiability) == 3


# ---------------------------------------------------------------------------
# 2. ClaimStatus
# ---------------------------------------------------------------------------

class TestClaimStatus:

    def test_is_str_enum(self):
        _assert_enum_is_str_enum(ClaimStatus)

    def test_expected_labels_exist(self):
        expected = {
            "SUPPORTED",
            "CONTRADICTED",
            "UNSUPPORTED",
            "VAGUE",
            "NOT_CHECKED",
        }
        actual = {m.name for m in ClaimStatus}
        assert actual == expected, (
            f"ClaimStatus labels mismatch.\n"
            f"  Missing : {expected - actual}\n"
            f"  Extra   : {actual - expected}"
        )

    def test_values_are_lowercase_strings(self):
        for member in ClaimStatus:
            _assert_value_is_lowercase(member)

    def test_exact_values(self):
        assert ClaimStatus.SUPPORTED    == "supported"
        assert ClaimStatus.CONTRADICTED == "contradicted"
        assert ClaimStatus.UNSUPPORTED  == "unsupported"
        assert ClaimStatus.VAGUE        == "vague"
        assert ClaimStatus.NOT_CHECKED  == "not_checked"

    def test_stable_count(self):
        assert len(ClaimStatus) == 5


# ---------------------------------------------------------------------------
# 3. HallucinationType
# ---------------------------------------------------------------------------

class TestHallucinationType:

    def test_is_str_enum(self):
        _assert_enum_is_str_enum(HallucinationType)

    def test_expected_labels_exist(self):
        expected = {
            "FACTUAL_CONTRADICTION",
            "CONTEXT_INCONSISTENCY",
            "LOGICAL_INCONSISTENCY",
            "INSTRUCTION_INCONSISTENCY",
            "FABRICATED_CLAIM",
            "OVERCLAIM",
            "WRONG_MOVE_REFERENCE",
        }
        actual = {m.name for m in HallucinationType}
        assert actual == expected, (
            f"HallucinationType labels mismatch.\n"
            f"  Missing : {expected - actual}\n"
            f"  Extra   : {actual - expected}"
        )

    def test_values_are_lowercase_strings(self):
        for member in HallucinationType:
            _assert_value_is_lowercase(member)

    def test_exact_values(self):
        assert HallucinationType.FACTUAL_CONTRADICTION    == "factual_contradiction"
        assert HallucinationType.CONTEXT_INCONSISTENCY    == "context_inconsistency"
        assert HallucinationType.LOGICAL_INCONSISTENCY    == "logical_inconsistency"
        assert HallucinationType.INSTRUCTION_INCONSISTENCY == "instruction_inconsistency"
        assert HallucinationType.FABRICATED_CLAIM         == "fabricated_claim"
        assert HallucinationType.OVERCLAIM                == "overclaim"
        assert HallucinationType.WRONG_MOVE_REFERENCE     == "wrong_move_reference"

    def test_stable_count(self):
        assert len(HallucinationType) == 7


# ---------------------------------------------------------------------------
# 4. SeedRiskType
# ---------------------------------------------------------------------------

class TestSeedRiskType:

    def test_is_str_enum(self):
        _assert_enum_is_str_enum(SeedRiskType)

    def test_expected_labels_exist(self):
        expected = {
            "STRICT_FACT",
            "INTERPRETIVE",
            "OVERCLAIM_RISK",
            "REDUNDANT",
            "MISLEADING",
        }
        actual = {m.name for m in SeedRiskType}
        assert actual == expected, (
            f"SeedRiskType labels mismatch.\n"
            f"  Missing : {expected - actual}\n"
            f"  Extra   : {actual - expected}"
        )

    def test_values_are_lowercase_strings(self):
        for member in SeedRiskType:
            _assert_value_is_lowercase(member)

    def test_exact_values(self):
        assert SeedRiskType.STRICT_FACT    == "strict_fact"
        assert SeedRiskType.INTERPRETIVE   == "interpretive"
        assert SeedRiskType.OVERCLAIM_RISK == "overclaim_risk"
        assert SeedRiskType.REDUNDANT      == "redundant"
        assert SeedRiskType.MISLEADING     == "misleading"

    def test_stable_count(self):
        assert len(SeedRiskType) == 5


# ---------------------------------------------------------------------------
# 5. TrajectoryEventType
# ---------------------------------------------------------------------------

class TestTrajectoryEventType:

    def test_is_str_enum(self):
        _assert_enum_is_str_enum(TrajectoryEventType)

    def test_expected_labels_exist(self):
        expected = {
            "RAW_LLM_SUCCESS",
            "PARSE_FAILURE",
            "API_FAILURE",
            "RETRY_USED",
            "RETRY_REPAIRED",
            "RETRY_FAILED",
            "OVERRIDE_USED",
            "SEED_FALLBACK_USED",
            "PYTHON_RESCUE_USED",
            "FINAL_MOVE_LEGAL",
            "FINAL_MOVE_ILLEGAL",
        }
        actual = {m.name for m in TrajectoryEventType}
        assert actual == expected, (
            f"TrajectoryEventType labels mismatch.\n"
            f"  Missing : {expected - actual}\n"
            f"  Extra   : {actual - expected}"
        )

    def test_values_are_lowercase_strings(self):
        for member in TrajectoryEventType:
            _assert_value_is_lowercase(member)

    def test_exact_values(self):
        assert TrajectoryEventType.RAW_LLM_SUCCESS    == "raw_llm_success"
        assert TrajectoryEventType.PARSE_FAILURE       == "parse_failure"
        assert TrajectoryEventType.API_FAILURE         == "api_failure"
        assert TrajectoryEventType.RETRY_USED          == "retry_used"
        assert TrajectoryEventType.RETRY_REPAIRED      == "retry_repaired"
        assert TrajectoryEventType.RETRY_FAILED        == "retry_failed"
        assert TrajectoryEventType.OVERRIDE_USED       == "override_used"
        assert TrajectoryEventType.SEED_FALLBACK_USED  == "seed_fallback_used"
        assert TrajectoryEventType.PYTHON_RESCUE_USED  == "python_rescue_used"
        assert TrajectoryEventType.FINAL_MOVE_LEGAL    == "final_move_legal"
        assert TrajectoryEventType.FINAL_MOVE_ILLEGAL  == "final_move_illegal"

    def test_stable_count(self):
        assert len(TrajectoryEventType) == 11


# ---------------------------------------------------------------------------
# Cross-enum: no value collision across different enums
# ---------------------------------------------------------------------------

def test_no_cross_enum_value_collision():
    """
    No two different enums should share the same string value.
    This prevents accidental label confusion when values are stored as raw
    strings in JSON logs.
    """
    all_enums = [
        ClaimVerifiability,
        ClaimStatus,
        HallucinationType,
        SeedRiskType,
        TrajectoryEventType,
    ]
    seen: dict[str, str] = {}
    for enum_cls in all_enums:
        for member in enum_cls:
            v = member.value
            assert v not in seen, (
                f"Value {v!r} appears in both {seen[v]} and {enum_cls.__name__}. "
                "All taxonomy values must be globally unique."
            )
            seen[v] = enum_cls.__name__


# ---------------------------------------------------------------------------
# Serialisation: values survive a round-trip through str/json
# ---------------------------------------------------------------------------

def test_enum_values_are_json_serialisable():
    """
    All enum values must be plain Python str instances so they can be
    serialised with json.dumps without a custom encoder.
    """
    import json

    all_enums = [
        ClaimVerifiability,
        ClaimStatus,
        HallucinationType,
        SeedRiskType,
        TrajectoryEventType,
    ]
    for enum_cls in all_enums:
        for member in enum_cls:
            # Must be serialisable
            dumped = json.dumps(member.value)
            loaded = json.loads(dumped)
            assert loaded == member.value, (
                f"{enum_cls.__name__}.{member.name}: JSON round-trip failed"
            )
            # Value must reconstruct the enum by value lookup
            reconstructed = enum_cls(loaded)
            assert reconstructed is member, (
                f"{enum_cls.__name__}.{member.name}: enum reconstruction from "
                f"JSON string failed"
            )
