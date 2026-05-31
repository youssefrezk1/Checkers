# checkers/tests/test_comparative_reasoner_step4.py
#
# Step 4 of the Comparative Reasoning v2 roadmap: unit tests for
# refine_comparative_reasoning and its supporting primitives in
# checkers/agents/comparative_reasoner.py.
#
# Coverage matrix (locked by the roadmap):
#   - RefinementCandidate dataclass
#   - _evaluate_refinement_candidate monotonic gate
#   - better candidate accepted
#   - worse candidate rejected
#   - API failure preservation
#   - malformed JSON preservation
#   - diagnostics correctness (retry_count, resolved)
#   - max_attempts hard cap (= 1)
#   - chosen-move not mutated (refinement untouched)
#   - no-improvement preservation
#   - existing isolation invariant still holds

from __future__ import annotations

import copy
import inspect
import json as _json

import pytest

from checkers.agents.comparative_reasoner import (
    RefinementCandidate,
    _evaluate_refinement_candidate,
    refine_comparative_reasoning,
    verify_comparative_reasoning,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _move(path, **facts):
    return {"path": path, "facts": dict(facts)}


# Chosen move: safe, non-aggressive, no captures.
CHOSEN_SAFE = _move(
    [(5, 4), (4, 3)],
    opponent_can_recapture=False,
    creates_immediate_threat=False,
    shot_sequence_available=False,
    captures_count=0,
    net_gain=0,
    leaves_piece_isolated=False,
    weakens_king_row=False,
    results_in_king=False,
    near_promotion=False,
    our_pieces_threatened_after=0,
    opponent_mobility_before=8,
    opponent_mobility_after=8,
)


def _alt_aggressive(path, opp_recap=True):
    """Alternative that creates a threat and (by default) allows recapture."""
    return _move(
        path,
        creates_immediate_threat=True,
        shot_sequence_available=False,
        opponent_can_recapture=opp_recap,
        captures_count=0,
        net_gain=0,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        our_pieces_threatened_after=0,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
    )


def _alt_quiet(path):
    return _move(
        path,
        creates_immediate_threat=False,
        shot_sequence_available=False,
        opponent_can_recapture=False,
        captures_count=0,
        net_gain=0,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        our_pieces_threatened_after=0,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
    )


# Three-candidate list: index 0 = chosen, 1 = aggressive alt, 2 = quiet alt.
CANDIDATES = [
    CHOSEN_SAFE,
    _alt_aggressive([(5, 2), (4, 3)]),
    _alt_quiet([(5, 0), (4, 1)]),
]

# Clean comparative paragraph (0 contradictions with CANDIDATES + CHOSEN_SAFE).
# Alt [1] creates a threat and allows recapture — both facts are True.
# Tradeoff: chosen forfeits aggressive options for recapture safety — valid
# because chosen has opponent_can_recapture=False.
CLEAN_TEXT = (
    "Aggressive alternative [1] creates an immediate threat but allows "
    "recapture. Chosen move forfeits aggressive options in favour of "
    "recapture safety."
)

# Dirty paragraph: claims alt [1] avoids recapture, but alt [1] allows it.
# Produces exactly 1 contradiction (per_alt_mismatch on opponent_can_recapture).
DIRTY_TEXT = (
    "Aggressive alternative [1] avoids recapture. "
    "Chosen move forfeits aggressive options in favour of recapture safety."
)


def _verify_counts(text: str) -> int:
    """Return contradiction count for text against CANDIDATES + CHOSEN_SAFE."""
    return len(verify_comparative_reasoning(text, CANDIDATES, CHOSEN_SAFE))


# Sanity-check fixtures before running any test.
assert _verify_counts(CLEAN_TEXT) == 0, "CLEAN_TEXT must have 0 contradictions"
assert _verify_counts(DIRTY_TEXT) >= 1, "DIRTY_TEXT must have >=1 contradiction"


# ═══════════════════════════════════════════════════════════════════════════
# 1. RefinementCandidate dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementCandidate:
    def test_fields_accessible(self):
        rc = RefinementCandidate(raw="r", text="t", n_contradictions=0)
        assert rc.raw == "r"
        assert rc.text == "t"
        assert rc.n_contradictions == 0

    def test_none_text_allowed(self):
        rc = RefinementCandidate(raw="r", text=None, n_contradictions=5)
        assert rc.text is None

    def test_zero_contradictions_valid(self):
        rc = RefinementCandidate(raw="raw", text="para", n_contradictions=0)
        assert rc.n_contradictions == 0

    def test_dataclass_identity(self):
        rc1 = RefinementCandidate(raw="r", text="t", n_contradictions=1)
        rc2 = RefinementCandidate(raw="r", text="t", n_contradictions=1)
        assert rc1 == rc2


# ═══════════════════════════════════════════════════════════════════════════
# 2. _evaluate_refinement_candidate — monotonic gate
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateRefinementCandidate:
    def test_strict_improvement_accepted(self):
        c = RefinementCandidate(raw="r", text="clean text", n_contradictions=0)
        assert _evaluate_refinement_candidate(c, baseline_count=2) is True

    def test_partial_improvement_accepted(self):
        c = RefinementCandidate(raw="r", text="better", n_contradictions=1)
        assert _evaluate_refinement_candidate(c, baseline_count=2) is True

    def test_equal_count_rejected(self):
        c = RefinementCandidate(raw="r", text="same level", n_contradictions=2)
        assert _evaluate_refinement_candidate(c, baseline_count=2) is False

    def test_worse_count_rejected(self):
        c = RefinementCandidate(raw="r", text="worse", n_contradictions=3)
        assert _evaluate_refinement_candidate(c, baseline_count=2) is False

    def test_none_text_rejected_regardless_of_count(self):
        c = RefinementCandidate(raw="r", text=None, n_contradictions=0)
        assert _evaluate_refinement_candidate(c, baseline_count=5) is False

    def test_baseline_zero_never_improves(self):
        # Nothing can be strictly less than 0 contradictions.
        c = RefinementCandidate(raw="r", text="t", n_contradictions=0)
        assert _evaluate_refinement_candidate(c, baseline_count=0) is False

    def test_gate_is_strict_not_lte(self):
        # Strictly < is required; equal must fail.
        c = RefinementCandidate(raw="r", text="t", n_contradictions=1)
        assert _evaluate_refinement_candidate(c, baseline_count=1) is False
        c2 = RefinementCandidate(raw="r", text="t", n_contradictions=0)
        assert _evaluate_refinement_candidate(c2, baseline_count=1) is True


# ═══════════════════════════════════════════════════════════════════════════
# 3. refine_comparative_reasoning — core behaviours
# ═══════════════════════════════════════════════════════════════════════════


class TestRefineComparativeReasoning:

    # ── 3a. Clean input (no refinement needed) ───────────────────────────────

    def test_clean_text_returns_immediately_no_api_call(self):
        call_count = [0]

        def _counted_api(system, user):
            call_count[0] += 1
            return '{"comparative_reasoning": "should not be called"}'

        result, retry_count, resolved = refine_comparative_reasoning(
            CLEAN_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_counted_api,
        )
        assert result == CLEAN_TEXT
        assert retry_count == 0
        assert resolved is True
        assert call_count[0] == 0  # no API call made

    def test_clean_text_retry_count_is_zero(self):
        _, retry_count, _ = refine_comparative_reasoning(
            CLEAN_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
        )
        assert retry_count == 0

    def test_clean_text_resolved_is_true(self):
        _, _, resolved = refine_comparative_reasoning(
            CLEAN_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
        )
        assert resolved is True

    # ── 3b. Better candidate accepted ───────────────────────────────────────

    def test_better_candidate_accepted(self):
        """API returns CLEAN_TEXT (0 contradictions) for DIRTY_TEXT (1+)."""
        def _good_api(system, user):
            return _json.dumps({"comparative_reasoning": CLEAN_TEXT})

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_good_api,
        )
        assert result == CLEAN_TEXT
        assert retry_count == 1
        assert resolved is True

    def test_better_candidate_result_is_refined_text(self):
        improved = (
            "Aggressive alternative [1] creates an immediate threat but "
            "allows recapture; chosen move forfeits initiative for safety."
        )
        assert _verify_counts(improved) == 0  # must be clean

        def _api(system, user):
            return _json.dumps({"comparative_reasoning": improved})

        result, _, _ = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_api,
        )
        assert result == improved

    # ── 3c. Worse candidate rejected ────────────────────────────────────────

    def test_worse_candidate_preserves_original(self):
        """API returns text with same contradiction count — gate rejects it."""
        def _same_api(system, user):
            # DIRTY_TEXT has same contradiction count as the input.
            return _json.dumps({"comparative_reasoning": DIRTY_TEXT})

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_same_api,
        )
        assert result == DIRTY_TEXT  # original preserved
        assert retry_count == 1
        assert resolved is False

    def test_higher_contradiction_count_preserves_original(self):
        # Alt [1] has creates_immediate_threat=True and opponent_can_recapture=True.
        # Both "avoids recapture" AND "does not create threat" are wrong for it.
        worse_text = (
            "Aggressive alternative [1] avoids recapture and does not "
            "create an immediate threat. "
            "Chosen move forfeits aggressive options in favour of recapture safety."
        )
        assert _verify_counts(worse_text) >= 2  # at least as bad as DIRTY_TEXT

        def _worse_api(system, user):
            return _json.dumps({"comparative_reasoning": worse_text})

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_worse_api,
        )
        assert result == DIRTY_TEXT
        assert retry_count == 1
        assert resolved is False

    # ── 3d. API failure preservation ────────────────────────────────────────

    def test_api_exception_preserves_original(self):
        def _failing_api(system, user):
            raise RuntimeError("Network failure")

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_failing_api,
        )
        assert result == DIRTY_TEXT
        assert retry_count == 1
        assert resolved is False

    def test_api_value_error_preserves_original(self):
        def _key_missing_api(system, user):
            raise ValueError("MISTRAL_API_KEY is not set")

        result, _, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_key_missing_api,
        )
        assert result == DIRTY_TEXT
        assert resolved is False

    def test_api_os_error_preserves_original(self):
        def _net_api(system, user):
            raise OSError("connection refused")

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_net_api,
        )
        assert result == DIRTY_TEXT
        assert retry_count == 1
        assert resolved is False

    # ── 3e. Malformed JSON preservation ─────────────────────────────────────

    def test_plain_text_response_preserves_original(self):
        def _text_api(system, user):
            return "just some prose, not JSON"

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_text_api,
        )
        assert result == DIRTY_TEXT
        assert retry_count == 1
        assert resolved is False

    def test_wrong_json_key_preserves_original(self):
        def _wrong_key_api(system, user):
            return '{"reasoning": "some text"}'  # wrong key name

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_wrong_key_api,
        )
        assert result == DIRTY_TEXT
        assert retry_count == 1
        assert resolved is False

    def test_empty_string_response_preserves_original(self):
        def _empty_api(system, user):
            return ""

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_empty_api,
        )
        assert result == DIRTY_TEXT
        assert retry_count == 1
        assert resolved is False

    def test_json_with_empty_value_preserves_original(self):
        def _blank_api(system, user):
            return '{"comparative_reasoning": ""}'

        result, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_blank_api,
        )
        assert result == DIRTY_TEXT
        assert retry_count == 1
        assert resolved is False

    # ── 3f. Diagnostics correctness ──────────────────────────────────────────

    def test_diagnostics_clean_input(self):
        result, retry_count, resolved = refine_comparative_reasoning(
            CLEAN_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
        )
        assert retry_count == 0
        assert resolved is True
        assert result == CLEAN_TEXT

    def test_diagnostics_after_api_failure(self):
        def _bad(system, user):
            raise RuntimeError("down")

        _, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_bad,
        )
        assert retry_count == 1
        assert resolved is False

    def test_diagnostics_after_successful_refinement(self):
        def _good(system, user):
            return _json.dumps({"comparative_reasoning": CLEAN_TEXT})

        _, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_good,
        )
        assert retry_count == 1
        assert resolved is True

    def test_diagnostics_after_rejected_candidate(self):
        def _rejected(system, user):
            return _json.dumps({"comparative_reasoning": DIRTY_TEXT})

        _, retry_count, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_rejected,
        )
        assert retry_count == 1
        assert resolved is False

    # ── 3g. max_attempts hard cap = 1 ───────────────────────────────────────

    def test_max_attempts_exactly_one_call(self):
        call_count = [0]

        def _counting(system, user):
            call_count[0] += 1
            # Return the same dirty text (rejected by gate) — but still returns.
            return _json.dumps({"comparative_reasoning": DIRTY_TEXT})

        refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_counting,
        )
        assert call_count[0] == 1

    def test_max_attempts_one_call_even_on_success(self):
        call_count = [0]

        def _good(system, user):
            call_count[0] += 1
            return _json.dumps({"comparative_reasoning": CLEAN_TEXT})

        refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_good,
        )
        assert call_count[0] == 1

    def test_max_attempts_one_call_on_api_failure(self):
        call_count = [0]

        def _fail(system, user):
            call_count[0] += 1
            raise ValueError("fail")

        refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_fail,
        )
        assert call_count[0] == 1

    # ── 3h. Chosen-move refinement untouched ─────────────────────────────────

    def test_chosen_move_not_mutated(self):
        chosen_before = copy.deepcopy(CHOSEN_SAFE)

        def _api(system, user):
            return _json.dumps({"comparative_reasoning": CLEAN_TEXT})

        refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_api,
        )
        assert CHOSEN_SAFE == chosen_before

    def test_all_candidates_not_mutated(self):
        candidates_copy = copy.deepcopy(CANDIDATES)

        def _api(system, user):
            return _json.dumps({"comparative_reasoning": CLEAN_TEXT})

        refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_api,
        )
        assert CANDIDATES == candidates_copy

    def test_no_refine_reasoning_call(self):
        """refine_comparative_reasoning must not invoke _refine_reasoning."""
        import checkers.agents.comparative_reasoner as cr
        src = inspect.getsource(cr.refine_comparative_reasoning)
        # Check that _refine_reasoning is never CALLED (not just mentioned in docs).
        assert "_refine_reasoning(" not in src

    def test_no_ranker_agent_reference_in_function(self):
        import checkers.agents.comparative_reasoner as cr
        src = inspect.getsource(cr.refine_comparative_reasoning)
        assert "ranker_agent" not in src
        assert "_check_reasoning_truthfulness" not in src

    # ── 3i. No-improvement preservation ─────────────────────────────────────

    def test_no_improvement_same_text_preserves_original(self):
        """Returning the same contradicting text is not an improvement."""
        def _noop(system, user):
            return _json.dumps({"comparative_reasoning": DIRTY_TEXT})

        result, _, resolved = refine_comparative_reasoning(
            DIRTY_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
            _api_caller=_noop,
        )
        assert result == DIRTY_TEXT
        assert resolved is False

    # ── 3j. Return type contract ─────────────────────────────────────────────

    def test_return_is_tuple_of_three(self):
        result = refine_comparative_reasoning(
            CLEAN_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_return_str_int_bool_types(self):
        text, retry, resolved = refine_comparative_reasoning(
            CLEAN_TEXT, CANDIDATES, CHOSEN_SAFE, seeds=[],
        )
        assert isinstance(text, str)
        assert isinstance(retry, int)
        assert isinstance(resolved, bool)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Existing isolation invariant still holds after Step 4
# ═══════════════════════════════════════════════════════════════════════════


class TestIsolationInvariantUnchanged:
    """The Step 4 additions must not break the Step 3 isolation guarantee."""

    def test_module_does_not_import_ranker_agent(self):
        import checkers.agents.comparative_reasoner as cr
        src = inspect.getsource(cr)
        assert "from checkers.agents.ranker_agent" not in src
        assert "import checkers.agents.ranker_agent" not in src

    def test_module_does_not_reference_chosen_move_verifier(self):
        import checkers.agents.comparative_reasoner as cr
        src = inspect.getsource(cr)
        for forbidden in (
            "_check_reasoning_truthfulness",
            "contradiction_strings(",
            "verify_all(",
            "extract_claims(",
            "verify_claims(",
            "from checkers.evaluation.unified_verifier",
            "from checkers.evaluation.claim_extractor",
            "from checkers.evaluation.claim_verifier",
        ):
            assert forbidden not in src, (
                f"isolation invariant violated: '{forbidden}' found in module"
            )

    def test_refine_function_does_not_reference_ranker_agent(self):
        import checkers.agents.comparative_reasoner as cr
        src = inspect.getsource(cr.refine_comparative_reasoning)
        assert "ranker_agent" not in src

    def test_step4_exports_in_all(self):
        from checkers.agents.comparative_reasoner import __all__
        assert "RefinementCandidate" in __all__
        assert "_evaluate_refinement_candidate" in __all__
        assert "refine_comparative_reasoning" in __all__
