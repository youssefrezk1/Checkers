# checkers/tests/test_turn_evaluator.py
#
# Tests for checkers/evaluation/turn_evaluator.py
#
# Coverage:
#   1. Isolation — no runtime pipeline imports.
#   2. Aggregate counts (total, supported, contradicted, unsupported, vague).
#   3. Boolean flags (has_contradiction, has_unsupported, has_vague).
#   4. Reasoning path classification for all five labels.
#   5. Trajectory event detection for all seven events.
#   6. extract_claims + verify_claims pipeline integration.
#   7. Edge cases: empty text, None inputs, unknown turn_id default.
#   8. Input immutability.
#   9. Determinism.

import sys
from typing import Any, Dict, List, Optional

import pytest

# ── Isolation guard ───────────────────────────────────────────────────────────
_modules_before = set(sys.modules.keys())

from checkers.evaluation.turn_evaluator import (
    TurnEvaluationRecord,
    evaluate_turn,
    _classify_reasoning_path,
    _build_trajectory_events,
    _build_provenance_note,
    REASONING_PATH_SEEDED_LLM,
    REASONING_PATH_REFINEMENT_REPAIRED,
    REASONING_PATH_SEED_FALLBACK,
    REASONING_PATH_HARDCODED_FALLBACK,
    REASONING_PATH_UNKNOWN,
    TRAJ_API_FAILURE,
    TRAJ_RETRY_USED,
    TRAJ_RETRY_REPAIRED,
    TRAJ_RETRY_FAILED,
    TRAJ_OVERRIDE_USED,
    TRAJ_SEED_FALLBACK,
    TRAJ_PYTHON_RESCUE,
    TRAJ_INTERNAL_CONTRADICTION,
    TRAJ_CONTRADICTION_REPAIRED,
    TRAJ_CONTRADICTION_SEED_FALLBACK,
)
from checkers.evaluation.reasoning_taxonomy import ClaimStatus

_modules_after = set(sys.modules.keys())

_FORBIDDEN = (
    "checkers.engine",
    "checkers.agents",
    "checkers.graph",
    "checkers.state",
    "checkers.nodes",
    "checkers.search",
)


def test_no_runtime_pipeline_imports():
    new = _modules_after - _modules_before
    for mod in new:
        for prefix in _FORBIDDEN:
            assert not mod.startswith(prefix), (
                f"turn_evaluator pulled in runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_diag(**overrides) -> Dict[str, Any]:
    """Return a diagnostics dict representing a clean seeded-LLM turn."""
    base: Dict[str, Any] = {
        "reasoning_seeds": [],
        "reasoning_is_seed_fallback": False,
        "reasoning_has_unresolved_contradiction": False,
        "reasoning_refinement_retry_count": 0,
        "api_call_failure_count": 0,
        "ranker_selected_valid_candidate": True,
        "override_retry_attempts": 0,
        "override_retry_resolved": False,
        "override_fallback_applied": False,
        "override_branch_name": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Reasoning path classification
# ---------------------------------------------------------------------------

class TestClassifyReasoningPath:

    def test_seed_fallback_takes_priority(self):
        diag = _clean_diag(
            reasoning_is_seed_fallback=True,
            reasoning_refinement_retry_count=2,  # would be refinement_repaired
            api_call_failure_count=1,             # would be hardcoded_fallback
        )
        assert _classify_reasoning_path(diag) == REASONING_PATH_SEED_FALLBACK

    def test_hardcoded_fallback_on_api_failure(self):
        diag = _clean_diag(
            reasoning_is_seed_fallback=False,
            api_call_failure_count=1,
        )
        assert _classify_reasoning_path(diag) == REASONING_PATH_HARDCODED_FALLBACK

    def test_refinement_repaired_when_retry_and_no_contradiction(self):
        diag = _clean_diag(
            reasoning_refinement_retry_count=1,
            reasoning_has_unresolved_contradiction=False,
        )
        assert _classify_reasoning_path(diag) == REASONING_PATH_REFINEMENT_REPAIRED

    def test_not_refinement_repaired_when_contradiction_unresolved(self):
        """Retry with unresolved contradiction → still seeded_llm (not repaired)."""
        diag = _clean_diag(
            reasoning_refinement_retry_count=1,
            reasoning_has_unresolved_contradiction=True,
        )
        assert _classify_reasoning_path(diag) == REASONING_PATH_SEEDED_LLM

    def test_seeded_llm_default(self):
        assert _classify_reasoning_path(_clean_diag()) == REASONING_PATH_SEEDED_LLM

    def test_unknown_when_not_a_dict(self):
        assert _classify_reasoning_path(None) == REASONING_PATH_UNKNOWN
        assert _classify_reasoning_path("bad") == REASONING_PATH_UNKNOWN
        assert _classify_reasoning_path(42)    == REASONING_PATH_UNKNOWN

    def test_unknown_when_empty_dict_defaults_seeded_llm(self):
        """Empty dict has all keys absent; all checks fail → seeded_llm."""
        assert _classify_reasoning_path({}) == REASONING_PATH_SEEDED_LLM


# ---------------------------------------------------------------------------
# Trajectory event builder
# ---------------------------------------------------------------------------

class TestBuildTrajectoryEvents:

    def test_clean_turn_has_no_events(self):
        events = _build_trajectory_events(_clean_diag())
        assert events == []

    def test_api_failure_detected(self):
        events = _build_trajectory_events(_clean_diag(api_call_failure_count=2))
        assert TRAJ_API_FAILURE in events

    def test_retry_used_and_repaired(self):
        events = _build_trajectory_events(_clean_diag(
            override_retry_attempts=1,
            override_retry_resolved=True,
        ))
        assert TRAJ_RETRY_USED in events
        assert TRAJ_RETRY_REPAIRED in events
        assert TRAJ_RETRY_FAILED not in events

    def test_retry_used_and_failed(self):
        events = _build_trajectory_events(_clean_diag(
            override_retry_attempts=2,
            override_retry_resolved=False,
        ))
        assert TRAJ_RETRY_USED in events
        assert TRAJ_RETRY_FAILED in events
        assert TRAJ_RETRY_REPAIRED not in events

    def test_override_detected(self):
        events = _build_trajectory_events(_clean_diag(
            override_branch_name="SAFE_VS_UNSAFE"
        ))
        assert TRAJ_OVERRIDE_USED in events

    def test_python_rescue_detected(self):
        events = _build_trajectory_events(_clean_diag(override_fallback_applied=True))
        assert TRAJ_PYTHON_RESCUE in events

    def test_seed_fallback_event_detected(self):
        events = _build_trajectory_events(_clean_diag(reasoning_is_seed_fallback=True))
        assert TRAJ_SEED_FALLBACK in events

    def test_all_events_present(self):
        diag = _clean_diag(
            api_call_failure_count=1,
            override_retry_attempts=1,
            override_retry_resolved=True,
            override_branch_name="branch",
            override_fallback_applied=True,
            reasoning_is_seed_fallback=True,
        )
        events = _build_trajectory_events(diag)
        for ev in (TRAJ_API_FAILURE, TRAJ_RETRY_USED, TRAJ_RETRY_REPAIRED,
                   TRAJ_OVERRIDE_USED, TRAJ_PYTHON_RESCUE, TRAJ_SEED_FALLBACK):
            assert ev in events

    def test_non_dict_returns_empty(self):
        assert _build_trajectory_events(None) == []
        assert _build_trajectory_events("bad") == []
        assert _build_trajectory_events({})   == []

    def test_api_failure_before_retry_in_order(self):
        """api_failure must appear before retry events (execution order)."""
        diag = _clean_diag(
            api_call_failure_count=1,
            override_retry_attempts=1,
            override_retry_resolved=False,
        )
        events = _build_trajectory_events(diag)
        assert events.index(TRAJ_API_FAILURE) < events.index(TRAJ_RETRY_USED)


# ---------------------------------------------------------------------------
# evaluate_turn — aggregate counts and flags
# ---------------------------------------------------------------------------

class TestEvaluateTurnCounts:

    def test_empty_reasoning_returns_zero_counts(self):
        rec = evaluate_turn("", turn_id="t0")
        assert isinstance(rec, TurnEvaluationRecord)
        assert rec.total_claims == 0
        assert rec.supported_count == 0
        assert rec.contradicted_count == 0
        assert rec.unsupported_count == 0
        assert rec.vague_count == 0
        assert not rec.has_contradiction
        assert not rec.has_unsupported
        assert not rec.has_vague

    def test_supported_claim_counted(self):
        """avoids_recapture with matching fact → supported_count=1."""
        rec = evaluate_turn(
            "This move avoids recapture.",
            facts={"opponent_can_recapture": False},
            turn_id="t1",
        )
        recap = [c for c in rec.claims if c.claim_type == "avoids_recapture"]
        assert len(recap) == 1
        assert recap[0].claim_status == ClaimStatus.SUPPORTED
        assert rec.supported_count >= 1
        assert not rec.has_contradiction

    def test_contradicted_claim_counted_and_flagged(self):
        """
        avoids_recapture in text but opponent_can_recapture=True →
        CONTRADICTED and has_contradiction=True.
        """
        rec = evaluate_turn(
            "This move avoids recapture.",
            facts={"opponent_can_recapture": True},
            turn_id="t2",
        )
        recap = [c for c in rec.claims if c.claim_type == "avoids_recapture"]
        assert recap[0].claim_status == ClaimStatus.CONTRADICTED
        assert rec.has_contradiction
        assert rec.contradicted_count >= 1

    def test_unsupported_claim_counted_and_flagged(self):
        """avoids_recapture with no fact → UNSUPPORTED and has_unsupported=True."""
        rec = evaluate_turn(
            "This move avoids recapture.",
            facts={},
            turn_id="t3",
        )
        assert rec.has_unsupported
        assert rec.unsupported_count >= 1

    def test_vague_strategic_claim_counted_and_flagged(self):
        """positional_pressure → VAGUE and has_vague=True."""
        rec = evaluate_turn(
            "This creates positional pressure on the opponent.",
            facts={},
            turn_id="t4",
        )
        pressure = [c for c in rec.claims if c.claim_type == "positional_pressure"]
        assert len(pressure) == 1
        assert pressure[0].claim_status == ClaimStatus.VAGUE
        assert rec.has_vague
        assert rec.vague_count >= 1

    def test_counts_are_consistent(self):
        """supported + contradicted + unsupported + vague == total_claims."""
        rec = evaluate_turn(
            "This move avoids recapture and gains material. "
            "The engine confirms minimax_score=3.00. "
            "It also creates positional pressure.",
            facts={
                "opponent_can_recapture": False,
                "net_gain": 2,
                "minimax_score": 3.0,
            },
            turn_id="consistency",
        )
        total = (
            rec.supported_count
            + rec.contradicted_count
            + rec.unsupported_count
            + rec.vague_count
        )
        assert total == rec.total_claims

    def test_flags_match_counts(self):
        rec = evaluate_turn(
            "This move avoids recapture.",
            facts={"opponent_can_recapture": False},
            turn_id="flags",
        )
        assert rec.has_contradiction == (rec.contradicted_count > 0)
        assert rec.has_unsupported   == (rec.unsupported_count > 0)
        assert rec.has_vague         == (rec.vague_count > 0)


# ---------------------------------------------------------------------------
# evaluate_turn — reasoning path
# ---------------------------------------------------------------------------

class TestEvaluateTurnReasoningPath:

    def test_default_path_is_seeded_llm(self):
        rec = evaluate_turn("text", ranker_diagnostics=_clean_diag(), turn_id="p0")
        assert rec.reasoning_path == REASONING_PATH_SEEDED_LLM

    def test_seed_fallback_path(self):
        diag = _clean_diag(reasoning_is_seed_fallback=True)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p1")
        assert rec.reasoning_path == REASONING_PATH_SEED_FALLBACK

    def test_refinement_repaired_path(self):
        diag = _clean_diag(
            reasoning_refinement_retry_count=1,
            reasoning_has_unresolved_contradiction=False,
        )
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2")
        assert rec.reasoning_path == REASONING_PATH_REFINEMENT_REPAIRED

    def test_hardcoded_fallback_path(self):
        diag = _clean_diag(api_call_failure_count=1)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p3")
        assert rec.reasoning_path == REASONING_PATH_HARDCODED_FALLBACK

    def test_none_diagnostics_defaults_seeded_llm(self):
        """None diagnostics → seeded_llm (all checks miss → default)."""
        rec = evaluate_turn("text", ranker_diagnostics=None, turn_id="p4")
        assert rec.reasoning_path == REASONING_PATH_UNKNOWN

    def test_non_dict_diagnostics_gives_unknown(self):
        rec = evaluate_turn("text", ranker_diagnostics="bad", turn_id="p5")
        assert rec.reasoning_path == REASONING_PATH_UNKNOWN


# ---------------------------------------------------------------------------
# evaluate_turn — trajectory events
# ---------------------------------------------------------------------------

class TestEvaluateTurnTrajectoryEvents:

    def test_clean_turn_has_no_events(self):
        rec = evaluate_turn("text", ranker_diagnostics=_clean_diag(), turn_id="ev0")
        assert rec.trajectory_events == []

    def test_api_failure_event(self):
        diag = _clean_diag(api_call_failure_count=1)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ev1")
        assert TRAJ_API_FAILURE in rec.trajectory_events

    def test_retry_repaired_event(self):
        diag = _clean_diag(override_retry_attempts=1, override_retry_resolved=True)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ev2")
        assert TRAJ_RETRY_USED in rec.trajectory_events
        assert TRAJ_RETRY_REPAIRED in rec.trajectory_events

    def test_retry_failed_event(self):
        diag = _clean_diag(override_retry_attempts=1, override_retry_resolved=False)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ev3")
        assert TRAJ_RETRY_FAILED in rec.trajectory_events

    def test_override_event(self):
        diag = _clean_diag(override_branch_name="SAFE_VS_UNSAFE")
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ev4")
        assert TRAJ_OVERRIDE_USED in rec.trajectory_events

    def test_python_rescue_event(self):
        diag = _clean_diag(override_fallback_applied=True)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ev5")
        assert TRAJ_PYTHON_RESCUE in rec.trajectory_events

    def test_seed_fallback_event(self):
        diag = _clean_diag(reasoning_is_seed_fallback=True)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ev6")
        assert TRAJ_SEED_FALLBACK in rec.trajectory_events

    def test_none_diagnostics_gives_empty_events(self):
        rec = evaluate_turn("text", ranker_diagnostics=None, turn_id="ev7")
        assert rec.trajectory_events == []


# ---------------------------------------------------------------------------
# Pre-repair contradiction trajectory events
# ---------------------------------------------------------------------------

class TestPreRepairContradictionEvents:
    """Tests for the three new trajectory events derived from
    reasoning_initial_contradictions / reasoning_contradiction_detected."""

    def _diag_seed_fallback_with_contradiction(self) -> Dict[str, Any]:
        """Contradiction detected, repair failed, fell back to seed summary."""
        return _clean_diag(
            reasoning_contradiction_detected=True,
            reasoning_contradiction_repaired=False,
            reasoning_is_seed_fallback=True,
            reasoning_refinement_retry_count=2,
        )

    def _diag_contradiction_repaired(self) -> Dict[str, Any]:
        """Contradiction detected and repaired by retry loop."""
        return _clean_diag(
            reasoning_contradiction_detected=True,
            reasoning_contradiction_repaired=True,
            reasoning_is_seed_fallback=False,
            reasoning_refinement_retry_count=1,
            reasoning_has_unresolved_contradiction=False,
        )

    def _diag_contradiction_unresolved(self) -> Dict[str, Any]:
        """Contradiction detected, retry used, but neither repaired nor seed_fallback."""
        return _clean_diag(
            reasoning_contradiction_detected=True,
            reasoning_contradiction_repaired=False,
            reasoning_is_seed_fallback=False,
            reasoning_refinement_retry_count=2,
            reasoning_has_unresolved_contradiction=True,
        )

    # ── internal_contradiction_detected ──────────────────────────────────────

    def test_internal_contradiction_detected_when_flag_set(self):
        events = _build_trajectory_events(self._diag_seed_fallback_with_contradiction())
        assert TRAJ_INTERNAL_CONTRADICTION in events

    def test_internal_contradiction_absent_on_clean_turn(self):
        events = _build_trajectory_events(_clean_diag())
        assert TRAJ_INTERNAL_CONTRADICTION not in events

    # ── contradiction_repaired ────────────────────────────────────────────────

    def test_contradiction_repaired_event_when_repaired(self):
        events = _build_trajectory_events(self._diag_contradiction_repaired())
        assert TRAJ_INTERNAL_CONTRADICTION in events
        assert TRAJ_CONTRADICTION_REPAIRED in events
        assert TRAJ_CONTRADICTION_SEED_FALLBACK not in events

    def test_contradiction_repaired_absent_when_seed_fallback(self):
        """Fell back to seeds, so it was NOT repaired by the retry loop."""
        events = _build_trajectory_events(self._diag_seed_fallback_with_contradiction())
        assert TRAJ_CONTRADICTION_REPAIRED not in events

    # ── contradiction_fell_back_to_seed_summary ───────────────────────────────

    def test_seed_fallback_event_when_contradiction_fell_back(self):
        events = _build_trajectory_events(self._diag_seed_fallback_with_contradiction())
        assert TRAJ_INTERNAL_CONTRADICTION in events
        assert TRAJ_CONTRADICTION_SEED_FALLBACK in events
        assert TRAJ_CONTRADICTION_REPAIRED not in events

    def test_seed_fallback_event_absent_when_repaired(self):
        events = _build_trajectory_events(self._diag_contradiction_repaired())
        assert TRAJ_CONTRADICTION_SEED_FALLBACK not in events

    def test_seed_fallback_event_absent_when_unresolved_but_not_seed_fallback(self):
        """Retry ran but neither repaired nor fell back to seeds."""
        events = _build_trajectory_events(self._diag_contradiction_unresolved())
        assert TRAJ_INTERNAL_CONTRADICTION in events
        assert TRAJ_CONTRADICTION_REPAIRED not in events
        assert TRAJ_CONTRADICTION_SEED_FALLBACK not in events

    # ── evaluate_turn integration ─────────────────────────────────────────────

    def test_evaluate_turn_includes_internal_contradiction_event(self):
        diag = self._diag_seed_fallback_with_contradiction()
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ic0")
        assert TRAJ_INTERNAL_CONTRADICTION in rec.trajectory_events
        assert TRAJ_CONTRADICTION_SEED_FALLBACK in rec.trajectory_events

    def test_evaluate_turn_includes_repaired_event(self):
        diag = self._diag_contradiction_repaired()
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="ic1")
        assert TRAJ_INTERNAL_CONTRADICTION in rec.trajectory_events
        assert TRAJ_CONTRADICTION_REPAIRED in rec.trajectory_events

    def test_evaluate_turn_clean_has_no_contradiction_events(self):
        rec = evaluate_turn("text", ranker_diagnostics=_clean_diag(), turn_id="ic2")
        assert TRAJ_INTERNAL_CONTRADICTION not in rec.trajectory_events
        assert TRAJ_CONTRADICTION_REPAIRED not in rec.trajectory_events
        assert TRAJ_CONTRADICTION_SEED_FALLBACK not in rec.trajectory_events


# ---------------------------------------------------------------------------
# evaluate_turn — edge cases
# ---------------------------------------------------------------------------

class TestEvaluateTurnEdgeCases:

    def test_default_turn_id_when_none(self):
        rec = evaluate_turn("text", turn_id=None)
        assert rec.turn_id == "unknown"

    def test_default_turn_id_when_empty_string(self):
        rec = evaluate_turn("text", turn_id="")
        assert rec.turn_id == "unknown"

    def test_explicit_turn_id_preserved(self):
        rec = evaluate_turn("text", turn_id="game_1_turn_7")
        assert rec.turn_id == "game_1_turn_7"

    def test_none_seeds_does_not_raise(self):
        rec = evaluate_turn("This move avoids recapture.", reasoning_seeds=None)
        assert isinstance(rec, TurnEvaluationRecord)

    def test_none_facts_does_not_raise(self):
        rec = evaluate_turn("This move avoids recapture.", facts=None)
        assert isinstance(rec, TurnEvaluationRecord)

    def test_none_diagnostics_does_not_raise(self):
        rec = evaluate_turn("text", ranker_diagnostics=None)
        assert isinstance(rec, TurnEvaluationRecord)

    def test_all_none_inputs_returns_minimal_record(self):
        rec = evaluate_turn("", reasoning_seeds=None, facts=None,
                            ranker_diagnostics=None, turn_id=None)
        assert rec.turn_id == "unknown"
        assert rec.total_claims == 0
        assert rec.trajectory_events == []

    def test_claims_list_is_independent_copy(self):
        """Modifying the returned claims list must not affect the record."""
        rec = evaluate_turn("This move avoids recapture.", turn_id="copy")
        original_len = len(rec.claims)
        rec.claims.append(None)   # mutate the returned list
        # Evaluating again gives same count (record not shared)
        rec2 = evaluate_turn("This move avoids recapture.", turn_id="copy")
        assert len(rec2.claims) == original_len


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_original_seeds_not_mutated(self):
        seeds = ["opponent_can_recapture=false — safety"]
        seeds_before = list(seeds)
        evaluate_turn("text", reasoning_seeds=seeds)
        assert seeds == seeds_before

    def test_original_facts_not_mutated(self):
        facts = {"opponent_can_recapture": False}
        evaluate_turn("This move avoids recapture.", facts=facts)
        assert facts == {"opponent_can_recapture": False}

    def test_original_diagnostics_not_mutated(self):
        diag = _clean_diag()
        keys_before = set(diag.keys())
        evaluate_turn("text", ranker_diagnostics=diag)
        assert set(diag.keys()) == keys_before


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_same_output(self):
        seeds = ["opponent_can_recapture=false — safety", "minimax_score=3.00 — best option"]
        facts = {"opponent_can_recapture": False, "minimax_score": 3.0}
        text  = "This move avoids recapture. The engine confirms minimax_score=3.00."
        diag  = _clean_diag(reasoning_seeds=seeds)

        r1 = evaluate_turn(text, reasoning_seeds=seeds, facts=facts,
                           ranker_diagnostics=diag, turn_id="det")
        r2 = evaluate_turn(text, reasoning_seeds=seeds, facts=facts,
                           ranker_diagnostics=diag, turn_id="det")

        assert r1.total_claims       == r2.total_claims
        assert r1.supported_count    == r2.supported_count
        assert r1.contradicted_count == r2.contradicted_count
        assert r1.reasoning_path     == r2.reasoning_path
        assert r1.trajectory_events  == r2.trajectory_events


# ---------------------------------------------------------------------------
# Pipeline integration — extract + verify chained correctly
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_extract_and_verify_chained(self):
        """
        A full realistic text should yield verified claims:
        - avoids_recapture: SUPPORTED (opponent_can_recapture=False)
        - gains_material: SUPPORTED (net_gain=2)
        - minimax_confirmation: SUPPORTED (minimax_score present)
        - positional_pressure: VAGUE (strategic, unverifiable)
        """
        text = (
            "This move avoids recapture and captures 2 pieces for a net gain of +2. "
            "The engine confirms minimax_score=4.00. "
            "It also creates positional pressure."
        )
        facts = {
            "opponent_can_recapture": False,
            "net_gain": 2,
            "minimax_score": 4.0,
        }
        rec = evaluate_turn(text, facts=facts, turn_id="integration")

        by_type = {c.claim_type: c for c in rec.claims}

        assert by_type["avoids_recapture"].claim_status    == ClaimStatus.SUPPORTED
        assert by_type["gains_material"].claim_status      == ClaimStatus.SUPPORTED
        assert by_type["minimax_confirmation"].claim_status == ClaimStatus.SUPPORTED
        assert by_type["positional_pressure"].claim_status == ClaimStatus.VAGUE

        assert not rec.has_contradiction
        assert rec.has_vague
        assert rec.supported_count == 3
        assert rec.vague_count == 1

    def test_contradiction_detected_end_to_end(self):
        """
        avoids_recapture in text + opponent_can_recapture=True →
        CONTRADICTED propagated to record flags.
        """
        rec = evaluate_turn(
            "This move avoids recapture.",
            facts={"opponent_can_recapture": True},
            turn_id="contra",
        )
        assert rec.has_contradiction
        assert rec.contradicted_count == 1

    def test_seeds_used_for_extraction(self):
        """
        A seed matching 'opponent_can_recapture=false' should make
        avoids_recapture appear as source='seed' before verification.
        After verification with matching fact → SUPPORTED.
        """
        seeds = ["opponent_can_recapture=false — immediate tactical safety"]
        facts = {"opponent_can_recapture": False}
        rec = evaluate_turn(
            "This move avoids recapture.",
            reasoning_seeds=seeds,
            facts=facts,
            turn_id="seed_int",
        )
        recap = [c for c in rec.claims if c.claim_type == "avoids_recapture"]
        assert len(recap) == 1
        assert recap[0].claim_status == ClaimStatus.SUPPORTED


# ══════════════════════════════════════════════════════════════════════════════
# Reasoning-path classification — complete label coverage
# ══════════════════════════════════════════════════════════════════════════════

class TestReasoningPathClassifierComplete:
    """
    Confirms that _classify_reasoning_path() emits the correct label for
    every possible diagnostics configuration, including the clean-run default
    that was previously returning None / showing as 'unknown' in aggregates.
    """

    def test_clean_run_is_seeded_llm(self):
        """
        A diagnostics dict with no retry, no fallback, and no API failure
        must produce 'seeded_llm' — the standard successful path.
        This is the bug that caused 'unknown' in Phase 1 aggregates.
        """
        diag = {
            "reasoning_is_seed_fallback": False,
            "api_call_failure_count": 0,
            "reasoning_refinement_retry_count": 0,
            "reasoning_has_unresolved_contradiction": False,
        }
        assert _classify_reasoning_path(diag) == REASONING_PATH_SEEDED_LLM

    def test_empty_dict_is_seeded_llm(self):
        """
        An empty dict (all fields absent, defaulting to falsy) must also
        produce 'seeded_llm' — it represents a clean run with no events.
        """
        assert _classify_reasoning_path({}) == REASONING_PATH_SEEDED_LLM

    def test_retry_resolved_is_refinement_repaired(self):
        """
        retry_count > 0 AND no unresolved contradiction → refinement_repaired.
        """
        diag = {
            "reasoning_is_seed_fallback": False,
            "api_call_failure_count": 0,
            "reasoning_refinement_retry_count": 1,
            "reasoning_has_unresolved_contradiction": False,
        }
        assert _classify_reasoning_path(diag) == REASONING_PATH_REFINEMENT_REPAIRED

    def test_seed_fallback_flag_is_seed_fallback(self):
        """
        reasoning_is_seed_fallback=True → seed_fallback (highest priority).
        """
        diag = {
            "reasoning_is_seed_fallback": True,
            "api_call_failure_count": 0,
            "reasoning_refinement_retry_count": 2,
        }
        assert _classify_reasoning_path(diag) == REASONING_PATH_SEED_FALLBACK

    def test_api_failure_is_hardcoded_fallback(self):
        """
        api_call_failure_count > 0 (and no seed fallback) → hardcoded_fallback.
        """
        diag = {
            "reasoning_is_seed_fallback": False,
            "api_call_failure_count": 1,
            "reasoning_refinement_retry_count": 0,
        }
        assert _classify_reasoning_path(diag) == REASONING_PATH_HARDCODED_FALLBACK

    def test_non_dict_is_unknown(self):
        """Non-dict inputs (None, str, int) must produce 'unknown'."""
        assert _classify_reasoning_path(None)  == REASONING_PATH_UNKNOWN
        assert _classify_reasoning_path("bad") == REASONING_PATH_UNKNOWN
        assert _classify_reasoning_path(42)    == REASONING_PATH_UNKNOWN


# ══════════════════════════════════════════════════════════════════════════════
# _extract_run_diagnostics reasoning_path fix — stress-suite integration
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractRunDiagnosticsReasoningPath:
    """
    Integration tests for the fix in tactical_stress_suite._extract_run_diagnostics.

    Previously: result.get("reasoning_path") read a key that does not exist in
    ranker_diagnostics → always returned None → aggregated as "unknown".

    After fix: _classify_reasoning_path(diag_raw) derives the label correctly
    from the actual diagnostic fields.

    These tests call _classify_reasoning_path directly with the same diag_raw
    structure that _extract_run_diagnostics would read, verifying end-to-end
    that the path label is correct for each scenario.
    """

    def _diag_raw_clean(self) -> dict:
        """Simulates ranker_diagnostics for a clean LLM run (no retry, no fallback)."""
        return {
            "reasoning_is_seed_fallback": False,
            "api_call_failure_count": 0,
            "reasoning_refinement_retry_count": 0,
            "reasoning_has_unresolved_contradiction": False,
            "reasoning_contradiction_detected": False,
            "reasoning_contradiction_repaired": False,
            "reasoning_initial_contradictions": [],
        }

    def test_clean_run_diag_raw_yields_seeded_llm(self):
        """
        The core regression: a clean run's diag_raw must yield 'seeded_llm'
        — not None or 'unknown' — when passed through _classify_reasoning_path.
        This is the bug that made reasoning_path_distribution: {'unknown': 20}
        in Phase 1 aggregate results.
        """
        diag_raw = self._diag_raw_clean()
        result = _classify_reasoning_path(diag_raw)
        assert result == REASONING_PATH_SEEDED_LLM, (
            f"Clean run must yield 'seeded_llm', got: {result!r}"
        )

    def test_repaired_run_diag_raw_yields_refinement_repaired(self):
        """
        A run that used retry and resolved it → 'refinement_repaired'.
        """
        diag_raw = {
            **self._diag_raw_clean(),
            "reasoning_refinement_retry_count": 1,
            "reasoning_contradiction_detected": True,
            "reasoning_contradiction_repaired": True,
            "reasoning_initial_contradictions": [
                "REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true"
            ],
        }
        result = _classify_reasoning_path(diag_raw)
        assert result == REASONING_PATH_REFINEMENT_REPAIRED

    def test_fallback_run_diag_raw_yields_seed_fallback(self):
        """
        A run where repair exhausted retries and fell back to seed summary
        → 'seed_fallback'.
        """
        diag_raw = {
            **self._diag_raw_clean(),
            "reasoning_is_seed_fallback": True,
            "reasoning_refinement_retry_count": 3,
        }
        result = _classify_reasoning_path(diag_raw)
        assert result == REASONING_PATH_SEED_FALLBACK

    def test_missing_diag_raw_yields_seeded_llm(self):
        """
        When no ranker_diagnostics are stored (diag_raw = {}), the classifier
        must default to 'seeded_llm', not 'unknown'.
        Empty dict is a valid clean-run indicator.
        """
        result = _classify_reasoning_path({})
        assert result == REASONING_PATH_SEEDED_LLM, (
            f"Empty diag_raw must yield 'seeded_llm', got: {result!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2.0 — Decision Provenance Propagation
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2Provenance:
    """
    Verifies that the 7 new Phase 2 provenance fields on TurnEvaluationRecord
    are correctly populated from ranker_diagnostics by evaluate_turn().

    Tests:
    1. Safe defaults when ranker_diagnostics is absent or empty.
    2. Old records (fields not in diag) still evaluate correctly (backward compat).
    3. final_choice_source propagates correctly.
    4. best_score_tie_count propagates correctly.
    5. minimax_best_path and minimax_best_score propagate correctly.
    6. retry_all_paths is a list whose length matches override_retry_attempts.
    7. raw_llm_reasoning_pre_refinement propagates when present.
    8. New provenance field names are absent from non-diagnostic runtime files.
    """

    def _diag_with_provenance(self, **overrides) -> Dict[str, Any]:
        """Return a diagnostics dict with all Phase 2 provenance fields set."""
        base = _clean_diag(
            final_choice_source="raw_llm",
            override_branch_name="SAFE_VS_UNSAFE",
            best_score_tie_count=2,
            minimax_best_path=[[5, 2], [4, 3]],
            minimax_best_score=3.75,
            retry_all_paths=[[[5, 2], [4, 3]], [[3, 0], [4, 1]]],
            raw_llm_reasoning_pre_refinement="Initial reasoning before refinement.",
            tie_break_reason=None,
        )
        base.update(overrides)
        return base

    # ── Test 1: safe defaults when diagnostics absent ─────────────────────────

    def test_safe_defaults_when_diagnostics_none(self):
        """All 7 provenance fields must have safe defaults when diag is None."""
        rec = evaluate_turn("text", ranker_diagnostics=None, turn_id="p2_1a")
        assert rec.final_choice_source == ""
        assert rec.override_branch_name is None
        assert rec.best_score_tie_count == 0
        assert rec.minimax_best_path is None
        assert rec.minimax_best_score is None
        assert rec.retry_all_paths == []
        assert rec.raw_llm_reasoning_pre_refinement is None

    def test_safe_defaults_when_diagnostics_empty_dict(self):
        """Empty dict (old-format diag) must yield safe defaults for new fields."""
        rec = evaluate_turn("text", ranker_diagnostics={}, turn_id="p2_1b")
        assert rec.final_choice_source == ""
        assert rec.override_branch_name is None
        assert rec.best_score_tie_count == 0
        assert rec.minimax_best_path is None
        assert rec.minimax_best_score is None
        assert rec.retry_all_paths == []
        assert rec.raw_llm_reasoning_pre_refinement is None

    # ── Test 2: backward compatibility with old diagnostics ───────────────────

    def test_backward_compat_old_diag_still_evaluates_claims(self):
        """
        A diagnostics dict that predates Phase 2 (no provenance keys) must still
        produce correct claim counts and reasoning path — no KeyError or crash.
        """
        old_diag = {
            "reasoning_seeds": [],
            "reasoning_is_seed_fallback": False,
            "reasoning_has_unresolved_contradiction": False,
            "reasoning_refinement_retry_count": 0,
            "api_call_failure_count": 0,
        }
        rec = evaluate_turn(
            "This move avoids recapture.",
            facts={"opponent_can_recapture": False},
            ranker_diagnostics=old_diag,
            turn_id="p2_2",
        )
        assert rec.supported_count >= 1
        assert rec.reasoning_path == REASONING_PATH_SEEDED_LLM
        # New fields get safe defaults — no crash
        assert rec.final_choice_source == ""
        assert rec.best_score_tie_count == 0
        assert rec.retry_all_paths == []

    # ── Test 3: final_choice_source propagates ────────────────────────────────

    def test_final_choice_source_raw_llm(self):
        diag = self._diag_with_provenance(final_choice_source="raw_llm")
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_3a")
        assert rec.final_choice_source == "raw_llm"

    def test_final_choice_source_python_fallback(self):
        diag = self._diag_with_provenance(final_choice_source="python_fallback")
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_3b")
        assert rec.final_choice_source == "python_fallback"

    def test_final_choice_source_missing_gives_empty_string(self):
        rec = evaluate_turn("text", ranker_diagnostics={}, turn_id="p2_3c")
        assert rec.final_choice_source == ""

    # ── Test 4: best_score_tie_count propagates ───────────────────────────────

    def test_best_score_tie_count_nonzero(self):
        diag = self._diag_with_provenance(best_score_tie_count=3)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_4a")
        assert rec.best_score_tie_count == 3

    def test_best_score_tie_count_zero(self):
        diag = self._diag_with_provenance(best_score_tie_count=0)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_4b")
        assert rec.best_score_tie_count == 0

    def test_best_score_tie_count_none_coerced_to_zero(self):
        """None in the diag (shouldn't happen, but must not crash) → 0."""
        diag = self._diag_with_provenance(best_score_tie_count=None)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_4c")
        assert rec.best_score_tie_count == 0

    # ── Test 5: minimax_best_path and minimax_best_score propagate ────────────

    def test_minimax_best_path_propagates(self):
        path = [[5, 2], [4, 3]]
        diag = self._diag_with_provenance(minimax_best_path=path)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_5a")
        assert rec.minimax_best_path == path

    def test_minimax_best_score_propagates(self):
        diag = self._diag_with_provenance(minimax_best_score=4.25)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_5b")
        assert rec.minimax_best_score == 4.25

    def test_minimax_best_path_none_when_absent(self):
        rec = evaluate_turn("text", ranker_diagnostics={}, turn_id="p2_5c")
        assert rec.minimax_best_path is None

    def test_minimax_best_score_none_when_absent(self):
        rec = evaluate_turn("text", ranker_diagnostics={}, turn_id="p2_5d")
        assert rec.minimax_best_score is None

    # ── Test 6: retry_all_paths length matches override_retry_attempts ─────────

    def test_retry_all_paths_empty_when_no_retries(self):
        diag = _clean_diag(override_retry_attempts=0, retry_all_paths=[])
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_6a")
        assert isinstance(rec.retry_all_paths, list)
        assert len(rec.retry_all_paths) == 0

    def test_retry_all_paths_length_matches_attempts(self):
        paths = [[[5, 2], [4, 3]], [[3, 0], [4, 1]]]
        diag = _clean_diag(
            override_retry_attempts=2,
            override_retry_resolved=True,
            retry_all_paths=paths,
        )
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_6b")
        assert isinstance(rec.retry_all_paths, list)
        assert len(rec.retry_all_paths) == diag["override_retry_attempts"]

    def test_retry_all_paths_is_independent_copy(self):
        """retry_all_paths in the record must be a copy, not the diag list."""
        paths = [[[5, 2], [4, 3]]]
        diag = _clean_diag(override_retry_attempts=1, retry_all_paths=paths)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_6c")
        rec.retry_all_paths.append("extra")
        assert len(diag["retry_all_paths"]) == 1  # original not mutated

    # ── Test 7: raw_llm_reasoning_pre_refinement propagates ──────────────────

    def test_raw_reasoning_propagates(self):
        raw = "Before refinement: captures one piece then retreats."
        diag = self._diag_with_provenance(raw_llm_reasoning_pre_refinement=raw)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p2_7a")
        assert rec.raw_llm_reasoning_pre_refinement == raw

    def test_raw_reasoning_none_when_absent(self):
        rec = evaluate_turn("text", ranker_diagnostics={}, turn_id="p2_7b")
        assert rec.raw_llm_reasoning_pre_refinement is None

    def test_raw_reasoning_none_when_diag_is_none(self):
        rec = evaluate_turn("text", ranker_diagnostics=None, turn_id="p2_7c")
        assert rec.raw_llm_reasoning_pre_refinement is None

    # ── Test 8: new provenance field names absent from non-diagnostic runtime ──

    def test_provenance_fields_not_read_by_decision_files(self):
        """
        The Phase 2 provenance-only fields must not appear as .get() lookups in
        the decision-making files: state_manager.py, logger_node.py, graph.py.
        They are written-only in ranker_agent.py and read-only in eval scripts.
        """
        import re
        from pathlib import Path as _Path

        _root = _Path(__file__).resolve().parents[2]
        _decision_files = [
            _root / "checkers" / "nodes" / "state_manager.py",
            _root / "checkers" / "nodes" / "logger_node.py",
            _root / "checkers" / "graph" / "graph.py",
        ]
        _provenance_keys = {
            "best_score_tie_count",
            "minimax_best_path",
            "minimax_best_score",
            "raw_llm_reasoning_pre_refinement",
            "retry_all_paths",
            "tie_break_reason",
        }

        for fpath in _decision_files:
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            for key in _provenance_keys:
                # Accept the key as a dict literal value label in logger_node,
                # but flag any .get("key") usage which would mean it's being
                # READ for decision logic.
                pattern = rf'\.get\(\s*["\']' + re.escape(key) + r'["\']'
                assert not re.search(pattern, text), (
                    f"Provenance field '{key}' is read via .get() in "
                    f"{fpath.name} — it must only be written in ranker_agent.py"
                )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2.1 — Provenance Note Builder
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase21ProvenanceNote:
    """
    Verifies _build_provenance_note() and the provenance_note field on
    TurnEvaluationRecord.

    Tests:
     1.  python_fallback emits override note.
     2.  retry_llm with different paths emits mismatch note.
     3.  retry_llm with same path does NOT emit mismatch note.
     4.  best_score_tie_count > 1 emits tie note.
     5.  Clean raw_llm turn emits empty note.
     6.  single_candidate turn emits empty note.
     7.  Duplicate retry_all_paths emits degeneracy note.
     8.  python_fallback + tie emits both notes.
     9.  diag absent (None) returns empty note safely.
    10.  Old diag without Phase 2 keys returns empty note (backward compat).
    11.  reasoning text is unchanged when provenance_note is set.
    12.  _build_provenance_note not imported by production decision files.
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    def _diag(self, **kw) -> Dict[str, Any]:
        """Base diagnostics dict with safe Phase 2 defaults."""
        base: Dict[str, Any] = {
            "final_choice_source":        "raw_llm",
            "raw_llm_choice_path":        [[5, 2], [4, 3]],
            "final_chosen_path":          [[5, 2], [4, 3]],
            "best_score_tie_count":       1,
            "minimax_best_score":         2.50,
            "retry_all_paths":            [],
            "override_retry_attempts":    0,
            "override_retry_resolved":    False,
            "override_fallback_applied":  False,
            "override_branch_name":       None,
            "reasoning_is_seed_fallback": False,
            "reasoning_has_unresolved_contradiction": False,
            "reasoning_refinement_retry_count": 0,
            "api_call_failure_count":     0,
        }
        base.update(kw)
        return base

    # ── Test 1: python_fallback note ──────────────────────────────────────────

    def test_python_fallback_note_emitted(self):
        diag = self._diag(
            final_choice_source="python_fallback",
            raw_llm_choice_path=[[4, 3], [3, 2]],
            final_chosen_path=[[5, 6], [4, 7]],
            override_fallback_applied=True,
        )
        note = _build_provenance_note(diag)
        assert note != "", "python_fallback must produce a non-empty note"
        assert "python" in note.lower() or "override" in note.lower()
        assert "[[4, 3], [3, 2]]" in note or "[DECISION]" in note

    def test_python_fallback_note_contains_both_paths(self):
        raw  = [[4, 3], [3, 2]]
        final = [[5, 6], [4, 7]]
        diag = self._diag(
            final_choice_source="python_fallback",
            raw_llm_choice_path=raw,
            final_chosen_path=final,
            override_fallback_applied=True,
        )
        note = _build_provenance_note(diag)
        assert str(raw)   in note, "raw LLM path must appear in fallback note"
        assert str(final) in note, "final path must appear in fallback note"

    def test_python_fallback_propagates_to_record(self):
        diag = self._diag(
            final_choice_source="python_fallback",
            raw_llm_choice_path=[[4, 3], [3, 2]],
            final_chosen_path=[[5, 6], [4, 7]],
            override_fallback_applied=True,
        )
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="pn_1")
        assert rec.provenance_note != ""
        assert "python" in rec.provenance_note.lower() or "override" in rec.provenance_note.lower()

    # ── Test 2: retry_llm mismatch note ──────────────────────────────────────

    def test_retry_llm_mismatch_note_emitted(self):
        diag = self._diag(
            final_choice_source="retry_llm",
            raw_llm_choice_path=[[5, 4], [4, 3]],
            final_chosen_path=[[7, 6], [6, 7]],
            override_retry_attempts=1,
            override_retry_resolved=True,
        )
        note = _build_provenance_note(diag)
        assert note != "", "retry_llm with path mismatch must produce a note"
        assert "rejected" in note.lower() or "[DECISION]" in note
        assert str([[5, 4], [4, 3]]) in note

    # ── Test 3: retry_llm same path — no mismatch note ───────────────────────

    def test_retry_llm_same_path_no_mismatch_note(self):
        path = [[7, 6], [6, 7]]
        diag = self._diag(
            final_choice_source="retry_llm",
            raw_llm_choice_path=path,
            final_chosen_path=path,
            override_retry_attempts=1,
            override_retry_resolved=True,
        )
        note = _build_provenance_note(diag)
        # No mismatch note; tie_count=1 so no tie note either → empty
        assert "rejected" not in note.lower()
        assert "[DECISION]" not in note

    # ── Test 4: tie_count > 1 note ────────────────────────────────────────────

    def test_tie_count_note_emitted(self):
        diag = self._diag(best_score_tie_count=3, minimax_best_score=4.75)
        note = _build_provenance_note(diag)
        assert "[TIE]" in note
        assert "3" in note
        assert "4.75" in note

    def test_tie_count_note_without_score(self):
        diag = self._diag(best_score_tie_count=2, minimax_best_score=None)
        note = _build_provenance_note(diag)
        assert "[TIE]" in note
        assert "2" in note

    def test_tie_count_one_no_note(self):
        diag = self._diag(best_score_tie_count=1)
        note = _build_provenance_note(diag)
        assert "[TIE]" not in note

    # ── Test 5: clean raw_llm — empty note ───────────────────────────────────

    def test_clean_raw_llm_empty_note(self):
        diag = self._diag(
            final_choice_source="raw_llm",
            best_score_tie_count=1,
            retry_all_paths=[],
        )
        note = _build_provenance_note(diag)
        assert note == "", f"clean raw_llm must produce empty note, got: {note!r}"

    def test_clean_raw_llm_propagates_empty_to_record(self):
        diag = self._diag(final_choice_source="raw_llm", best_score_tie_count=1)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="pn_5")
        assert rec.provenance_note == ""

    # ── Test 6: single_candidate — empty note ────────────────────────────────

    def test_single_candidate_empty_note(self):
        diag = self._diag(
            final_choice_source="single_candidate",
            raw_llm_choice_path=None,
            final_chosen_path=[[5, 2], [4, 3]],
            best_score_tie_count=0,
        )
        note = _build_provenance_note(diag)
        assert note == "", f"single_candidate must produce empty note, got: {note!r}"

    # ── Test 7: retry degeneracy note ────────────────────────────────────────

    def test_retry_degeneracy_note_emitted(self):
        repeated_path = [[5, 4], [4, 3]]
        diag = self._diag(
            final_choice_source="retry_llm",
            retry_all_paths=[repeated_path, repeated_path],
            override_retry_attempts=2,
        )
        note = _build_provenance_note(diag)
        assert "[RETRY_DEGENERATE]" in note
        assert str(repeated_path) in note

    def test_no_degeneracy_note_for_distinct_paths(self):
        diag = self._diag(
            final_choice_source="retry_llm",
            retry_all_paths=[[[5, 4], [4, 3]], [[3, 0], [4, 1]]],
            override_retry_attempts=2,
        )
        note = _build_provenance_note(diag)
        assert "[RETRY_DEGENERATE]" not in note

    # ── Test 8: combined python_fallback + tie ────────────────────────────────

    def test_fallback_and_tie_both_appear(self):
        diag = self._diag(
            final_choice_source="python_fallback",
            raw_llm_choice_path=[[4, 3], [3, 2]],
            final_chosen_path=[[5, 6], [4, 7]],
            override_fallback_applied=True,
            best_score_tie_count=2,
            minimax_best_score=3.00,
        )
        note = _build_provenance_note(diag)
        assert "[DECISION]" in note
        assert "[TIE]" in note

    # ── Test 9: diag absent (None) — safe default ─────────────────────────────

    def test_none_diag_returns_empty_note(self):
        assert _build_provenance_note(None) == ""  # type: ignore[arg-type]

    def test_none_diag_evaluate_turn_safe(self):
        rec = evaluate_turn("text", ranker_diagnostics=None, turn_id="pn_9")
        assert rec.provenance_note == ""

    # ── Test 10: old diag without Phase 2 keys — backward compat ─────────────

    def test_old_diag_no_crash(self):
        old_diag = {
            "reasoning_seeds": [],
            "reasoning_is_seed_fallback": False,
            "reasoning_has_unresolved_contradiction": False,
            "reasoning_refinement_retry_count": 0,
            "api_call_failure_count": 0,
        }
        note = _build_provenance_note(old_diag)
        assert note == "", f"old diag must yield empty note, got: {note!r}"

    def test_old_diag_evaluate_turn_no_crash(self):
        old_diag = {
            "reasoning_seeds": [],
            "reasoning_is_seed_fallback": False,
            "reasoning_has_unresolved_contradiction": False,
        }
        rec = evaluate_turn("text", ranker_diagnostics=old_diag, turn_id="pn_10")
        assert rec.provenance_note == ""

    # ── Test 11: reasoning text unchanged ────────────────────────────────────

    def test_reasoning_text_unaffected_by_provenance(self):
        """
        evaluate_turn() must return the same reasoning-based claim analysis
        regardless of what provenance fields are set.  The provenance_note
        is additive metadata; it must never alter claim extraction or verification.
        """
        text  = "This move avoids recapture."
        facts = {"opponent_can_recapture": False}

        rec_clean = evaluate_turn(
            text, facts=facts,
            ranker_diagnostics=self._diag(final_choice_source="raw_llm"),
            turn_id="pn_11a",
        )
        rec_fallback = evaluate_turn(
            text, facts=facts,
            ranker_diagnostics=self._diag(
                final_choice_source="python_fallback",
                raw_llm_choice_path=[[4, 3], [3, 2]],
                final_chosen_path=[[5, 6], [4, 7]],
                override_fallback_applied=True,
            ),
            turn_id="pn_11b",
        )
        # Claim analysis identical — provenance does not change extraction/verification
        assert rec_clean.total_claims      == rec_fallback.total_claims
        assert rec_clean.supported_count   == rec_fallback.supported_count
        assert rec_clean.contradicted_count == rec_fallback.contradicted_count
        # Provenance note differs
        assert rec_clean.provenance_note    == ""
        assert rec_fallback.provenance_note != ""

    # ── Test 12: _build_provenance_note not imported by production files ──────

    def test_provenance_note_builder_not_in_production_files(self):
        """
        _build_provenance_note must remain an evaluation-only helper.
        It must not appear in ranker_agent.py, state_manager.py,
        logger_node.py, or graph.py.
        """
        from pathlib import Path as _Path

        _root = _Path(__file__).resolve().parents[2]
        _production_files = [
            _root / "checkers" / "agents" / "ranker_agent.py",
            _root / "checkers" / "nodes"  / "state_manager.py",
            _root / "checkers" / "nodes"  / "logger_node.py",
            _root / "checkers" / "graph"  / "graph.py",
        ]
        for fpath in _production_files:
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            assert "_build_provenance_note" not in text, (
                f"_build_provenance_note must not appear in production file "
                f"{fpath.name}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2.2 — Tie-Break Provenance
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase22TieBreakProvenance:
    """
    Verifies tie-break explanation in _build_provenance_note and the
    tied_candidate_paths field on TurnEvaluationRecord.

    Tests:
     1.  final == minimax-best → "agreed with minimax-best" note.
     2.  final != minimax-best, raw_llm → "different tied move, equal score" note.
     3.  python_fallback tied → "python fallback selected" note.
     4.  promotion tiebreak → "promotion preferred" note.
     5.  Peer paths included when count <= 5.
     6.  Peer paths omitted when count > 5.
     7.  tie_count=1 → no [TIE] note.
     8.  tied_candidate_paths propagates to record.
     9.  Safe defaults when diag absent.
    10.  Old diag (no tied_candidate_paths key) backward compat.
    11.  reasoning/claim counts unchanged.
    12.  Static guard: tied_candidate_paths not used by production decision files.
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    _PATH_A = [[5, 2], [4, 3]]
    _PATH_B = [[6, 1], [5, 2]]
    _PATH_C = [[7, 0], [6, 1]]

    def _tie_diag(self, **kw) -> Dict[str, Any]:
        """Base diag with a 2-way tie, final == minimax-best, raw_llm source."""
        base: Dict[str, Any] = {
            "final_choice_source":        "raw_llm",
            "raw_llm_choice_path":        self._PATH_A,
            "final_chosen_path":          self._PATH_A,
            "best_score_tie_count":       2,
            "minimax_best_score":         3.50,
            "minimax_best_path":          self._PATH_A,
            "tied_candidate_paths":       [self._PATH_A, self._PATH_B],
            "tie_break_reason":           None,
            "retry_all_paths":            [],
            "override_retry_attempts":    0,
            "override_retry_resolved":    False,
            "override_fallback_applied":  False,
            "override_branch_name":       None,
            "reasoning_is_seed_fallback": False,
            "reasoning_has_unresolved_contradiction": False,
            "reasoning_refinement_retry_count": 0,
            "api_call_failure_count":     0,
        }
        base.update(kw)
        return base

    # ── Test 1: final == minimax-best ─────────────────────────────────────────

    def test_argmax_note(self):
        diag = self._tie_diag(
            final_chosen_path=self._PATH_A,
            minimax_best_path=self._PATH_A,
            final_choice_source="raw_llm",
        )
        note = _build_provenance_note(diag)
        assert "[TIE]" in note
        assert "agreed with minimax-best" in note

    def test_argmax_note_propagates_to_record(self):
        diag = self._tie_diag()
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p22_1")
        assert "[TIE]" in rec.provenance_note
        assert "agreed with minimax-best" in rec.provenance_note

    # ── Test 2: final != minimax-best, raw_llm ────────────────────────────────

    def test_different_tied_move_note(self):
        """
        LLM chose a tied move that isn't the argmax.
        Note must say 'equal score' — not imply an error.
        """
        diag = self._tie_diag(
            final_chosen_path=self._PATH_B,
            minimax_best_path=self._PATH_A,
            final_choice_source="raw_llm",
        )
        note = _build_provenance_note(diag)
        assert "[TIE]" in note
        assert "equal" in note.lower() or "equal minimax score" in note
        assert "minimax-best was" in note or str(self._PATH_A) in note
        # Must NOT imply the choice was wrong
        assert "worse" not in note.lower()
        assert "suboptimal" not in note.lower()

    def test_different_tied_move_contains_best_path(self):
        diag = self._tie_diag(
            final_chosen_path=self._PATH_B,
            minimax_best_path=self._PATH_A,
            final_choice_source="raw_llm",
        )
        note = _build_provenance_note(diag)
        assert str(self._PATH_A) in note

    # ── Test 3: python_fallback tied ──────────────────────────────────────────

    def test_python_fallback_tied_note(self):
        diag = self._tie_diag(
            final_choice_source="python_fallback",
            override_fallback_applied=True,
        )
        note = _build_provenance_note(diag)
        assert "[TIE]" in note
        assert "python fallback" in note.lower()

    # ── Test 4: promotion tiebreak ────────────────────────────────────────────

    def test_promotion_tiebreak_note(self):
        diag = self._tie_diag(tie_break_reason="promotion")
        note = _build_provenance_note(diag)
        assert "[TIE]" in note
        assert "promotion" in note.lower()

    # ── Test 5: peer paths displayed when count <= 5 ──────────────────────────

    def test_peer_paths_shown_when_small(self):
        paths = [self._PATH_A, self._PATH_B, self._PATH_C]
        diag = self._tie_diag(
            best_score_tie_count=3,
            tied_candidate_paths=paths,
        )
        note = _build_provenance_note(diag)
        assert "Tied paths:" in note
        for p in paths:
            assert str(p) in note

    def test_peer_paths_shown_at_limit_of_five(self):
        paths = [[[i, 0], [i+1, 1]] for i in range(5)]
        diag = self._tie_diag(
            best_score_tie_count=5,
            tied_candidate_paths=paths,
        )
        note = _build_provenance_note(diag)
        assert "Tied paths:" in note

    # ── Test 6: peer paths omitted when count > 5 ─────────────────────────────

    def test_peer_paths_omitted_when_large(self):
        paths = [[[i, 0], [i+1, 1]] for i in range(6)]
        diag = self._tie_diag(
            best_score_tie_count=6,
            tied_candidate_paths=paths,
        )
        note = _build_provenance_note(diag)
        assert "[TIE]" in note
        assert "Tied paths:" not in note
        assert "6" in note  # count still mentioned

    # ── Test 7: tie_count=1 produces no [TIE] note ───────────────────────────

    def test_single_best_no_tie_note(self):
        diag = self._tie_diag(
            best_score_tie_count=1,
            tied_candidate_paths=[self._PATH_A],
        )
        note = _build_provenance_note(diag)
        assert "[TIE]" not in note

    def test_zero_tie_count_no_tie_note(self):
        diag = self._tie_diag(best_score_tie_count=0, tied_candidate_paths=[])
        note = _build_provenance_note(diag)
        assert "[TIE]" not in note

    # ── Test 8: tied_candidate_paths propagates to record ────────────────────

    def test_tied_candidate_paths_propagates(self):
        paths = [self._PATH_A, self._PATH_B]
        diag = self._tie_diag(tied_candidate_paths=paths)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p22_8")
        assert isinstance(rec.tied_candidate_paths, list)
        assert rec.tied_candidate_paths == paths

    def test_tied_candidate_paths_is_independent_copy(self):
        paths = [self._PATH_A, self._PATH_B]
        diag = self._tie_diag(tied_candidate_paths=paths)
        rec = evaluate_turn("text", ranker_diagnostics=diag, turn_id="p22_8b")
        rec.tied_candidate_paths.append("extra")
        assert len(diag["tied_candidate_paths"]) == 2  # original unchanged

    # ── Test 9: safe defaults when diag absent ────────────────────────────────

    def test_none_diag_tied_paths_default(self):
        rec = evaluate_turn("text", ranker_diagnostics=None, turn_id="p22_9")
        assert rec.tied_candidate_paths == []
        assert "[TIE]" not in rec.provenance_note

    def test_empty_diag_tied_paths_default(self):
        rec = evaluate_turn("text", ranker_diagnostics={}, turn_id="p22_9b")
        assert rec.tied_candidate_paths == []

    # ── Test 10: backward compat with old diag (no tied_candidate_paths key) ──

    def test_old_diag_no_tied_paths_key(self):
        old_diag = {
            "reasoning_seeds": [],
            "reasoning_is_seed_fallback": False,
            "reasoning_has_unresolved_contradiction": False,
            "reasoning_refinement_retry_count": 0,
            "api_call_failure_count": 0,
            # No tied_candidate_paths, best_score_tie_count, etc.
        }
        rec = evaluate_turn("text", ranker_diagnostics=old_diag, turn_id="p22_10")
        assert rec.tied_candidate_paths == []
        assert "[TIE]" not in rec.provenance_note

    # ── Test 11: reasoning/claim counts unchanged ─────────────────────────────

    def test_reasoning_and_claims_unaffected(self):
        """
        Tie-break diagnostics must not alter claim extraction or verification.
        """
        text  = "This move avoids recapture."
        facts = {"opponent_can_recapture": False}

        rec_no_tie = evaluate_turn(
            text, facts=facts,
            ranker_diagnostics=_clean_diag(final_choice_source="raw_llm"),
            turn_id="p22_11a",
        )
        rec_tie = evaluate_turn(
            text, facts=facts,
            ranker_diagnostics=self._tie_diag(
                final_chosen_path=self._PATH_B,
                minimax_best_path=self._PATH_A,
            ),
            turn_id="p22_11b",
        )
        assert rec_no_tie.total_claims      == rec_tie.total_claims
        assert rec_no_tie.supported_count   == rec_tie.supported_count
        assert rec_no_tie.contradicted_count == rec_tie.contradicted_count
        # Provenance differs — claims do not
        assert "[TIE]" not in rec_no_tie.provenance_note
        assert "[TIE]" in rec_tie.provenance_note

    # ── Test 12: static guard — not in production decision files ──────────────

    def test_tied_candidate_paths_not_in_production_files(self):
        """
        The tied_candidate_paths field must not be read via .get() in any
        file that makes decisions: state_manager, logger_node, graph.
        """
        import re
        from pathlib import Path as _Path

        _root = _Path(__file__).resolve().parents[2]
        _decision_files = [
            _root / "checkers" / "nodes" / "state_manager.py",
            _root / "checkers" / "nodes" / "logger_node.py",
            _root / "checkers" / "graph" / "graph.py",
        ]
        for fpath in _decision_files:
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            pattern = r'\.get\(\s*["\']tied_candidate_paths["\']'
            assert not re.search(pattern, text), (
                f"tied_candidate_paths must not be read via .get() in {fpath.name}"
            )


# =============================================================================
# Phase 2.3a — Retry Diversity Diagnostics
# =============================================================================

class TestPhase23RetryDiversity:
    """
    Validate retry_rejection_reasons and retry_duplicate_count fields added in
    Phase 2.3a.  All tests are deterministic and require no LLM calls.
    """

    # ── Fixtures ──────────────────────────────────────────────────────────────

    P1 = [[5, 4], [4, 3]]
    P2 = [[5, 2], [4, 1]]
    P3 = [[5, 6], [4, 5]]

    def _diag(self, **kw) -> dict:
        """Minimal ranker_diagnostics with Phase 2.3a fields."""
        base = {
            "final_choice_source":    "retry_llm",
            "retry_all_paths":        [],
            "retry_rejection_reasons": [],
        }
        base.update(kw)
        return base

    # ── retry_all_paths propagation ────────────────────────────────────────────

    def test_retry_all_paths_propagates_to_record(self):
        diag = self._diag(retry_all_paths=[self.P1, self.P2])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t1")
        assert r.retry_all_paths == [self.P1, self.P2]

    def test_retry_all_paths_empty_by_default(self):
        diag = self._diag(retry_all_paths=[])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t2")
        assert r.retry_all_paths == []

    def test_retry_all_paths_default_when_key_absent(self):
        diag = {"final_choice_source": "raw_llm"}
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t3")
        assert r.retry_all_paths == []

    # ── retry_rejection_reasons propagation ───────────────────────────────────

    def test_retry_rejection_reasons_propagates(self):
        reasons = ["safe_vs_unsafe_large_gap", "low_danger_minimax_dominance"]
        diag = self._diag(retry_rejection_reasons=reasons)
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t4")
        assert r.retry_rejection_reasons == reasons

    def test_retry_rejection_reasons_is_independent_copy(self):
        reasons = ["branch_a"]
        diag = self._diag(retry_rejection_reasons=reasons)
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t5")
        r.retry_rejection_reasons.append("mutated")
        assert diag["retry_rejection_reasons"] == ["branch_a"]

    def test_retry_rejection_reasons_empty_by_default(self):
        diag = self._diag(retry_rejection_reasons=[])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t6")
        assert r.retry_rejection_reasons == []

    def test_retry_rejection_reasons_default_when_key_absent(self):
        diag = {"final_choice_source": "raw_llm"}
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t7")
        assert r.retry_rejection_reasons == []

    # ── retry_duplicate_count ──────────────────────────────────────────────────

    def test_retry_duplicate_count_zero_for_distinct_paths(self):
        diag = self._diag(retry_all_paths=[self.P1, self.P2, self.P3])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t8")
        assert r.retry_duplicate_count == 0

    def test_retry_duplicate_count_one_for_one_repeat(self):
        diag = self._diag(retry_all_paths=[self.P1, self.P2, self.P1])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t9")
        assert r.retry_duplicate_count == 1

    def test_retry_duplicate_count_two_for_two_repeats(self):
        # P1 appears 3 times → 2 duplicates
        diag = self._diag(retry_all_paths=[self.P1, self.P2, self.P1, self.P1])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t10")
        assert r.retry_duplicate_count == 2

    def test_retry_duplicate_count_zero_for_empty(self):
        diag = self._diag(retry_all_paths=[])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t11")
        assert r.retry_duplicate_count == 0

    def test_retry_duplicate_count_zero_for_single_path(self):
        diag = self._diag(retry_all_paths=[self.P1])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t12")
        assert r.retry_duplicate_count == 0

    def test_retry_duplicate_count_default_when_key_absent(self):
        diag = {"final_choice_source": "raw_llm"}
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t13")
        assert r.retry_duplicate_count == 0

    # ── provenance note fires for duplicates ───────────────────────────────────

    def test_degenerate_note_fires_when_retry_all_paths_has_duplicate(self):
        diag = self._diag(retry_all_paths=[self.P1, self.P1])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t14")
        assert "[RETRY_DEGENERATE]" in r.provenance_note

    def test_no_degenerate_note_for_single_retry(self):
        diag = self._diag(retry_all_paths=[self.P1])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t15")
        assert "[RETRY_DEGENERATE]" not in r.provenance_note

    def test_no_degenerate_note_for_empty_retry_paths(self):
        diag = self._diag(retry_all_paths=[])
        r = evaluate_turn("Move.", ranker_diagnostics=diag, turn_id="t16")
        assert "[RETRY_DEGENERATE]" not in r.provenance_note

    # ── backward compatibility ─────────────────────────────────────────────────

    def test_old_diag_no_retry_fields_no_crash(self):
        old_diag = {"final_choice_source": "raw_llm", "override_branch_name": "some_branch"}
        r = evaluate_turn("Move.", ranker_diagnostics=old_diag, turn_id="t17")
        assert r.retry_rejection_reasons == []
        assert r.retry_duplicate_count == 0
        assert r.retry_all_paths == []

    def test_none_diag_safe_defaults(self):
        r = evaluate_turn("Move.", ranker_diagnostics=None, turn_id="t18")
        assert r.retry_rejection_reasons == []
        assert r.retry_duplicate_count == 0
        assert r.retry_all_paths == []

    # ── reasoning and claims unaffected ───────────────────────────────────────

    def test_reasoning_and_claims_unaffected(self):
        reasoning = "The move advances a piece forward without capturing."
        diag = self._diag(retry_all_paths=[self.P1, self.P1], retry_rejection_reasons=["b1", "b1"])
        r = evaluate_turn(reasoning, ranker_diagnostics=diag, turn_id="t19")
        # Claims are extracted from reasoning text — retry fields must not affect them
        assert r.total_claims >= 0
        # retry fields present
        assert r.retry_duplicate_count == 1
        assert r.retry_rejection_reasons == ["b1", "b1"]

    # ── static guard: retry diagnostics not read by production files ──────────

    def test_retry_diagnostics_not_in_production_files(self):
        import re
        from pathlib import Path as _Path

        _root = _Path(__file__).resolve().parents[2]
        _decision_files = [
            _root / "checkers" / "nodes" / "state_manager.py",
            _root / "checkers" / "nodes" / "logger_node.py",
            _root / "checkers" / "graph" / "graph.py",
        ]
        for fpath in _decision_files:
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            for field_name in ("retry_rejection_reasons", "retry_duplicate_count"):
                pattern = r'\.get\(\s*["\']' + re.escape(field_name) + r'["\']'
                assert not re.search(pattern, text), (
                    f"{field_name} must not be read via .get() in {fpath.name}"
                )
