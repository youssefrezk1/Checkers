# checkers/tests/test_comparative_orchestrator_step5.py
#
# Step 5 of the Comparative Reasoning v2 roadmap: orchestrator tests for
# generate_comparative_reasoning in checkers/agents/comparative_reasoner.py.
#
# Coverage matrix:
#   - fixture sanity: CLEAN_TEXT has 0 contradictions, DIRTY_TEXT has ≥1
#   - end-to-end mocked orchestrator flow (short-circuit path)
#   - end-to-end mocked orchestrator flow (refinement path)
#   - skip paths: no groups, no valid samples, API failure, malformed JSON
#   - refinement failure returns None
#   - diagnostics shape: exactly 12 keys, names match _COMPARATIVE_DIAGNOSTICS_KEYS
#   - diagnostics populated on every exit path
#   - isolation invariant: module source has no ranker_agent import strings
#   - chosen_move / all_candidates / chosen_facts not mutated
#   - deterministic behaviour: same inputs → same outputs

from __future__ import annotations

import copy
import inspect
import json

from checkers.agents.comparative_reasoner import (
    EXPLAINER_COMPARATIVE_SYSTEM as RANKER_COMPARATIVE_SYSTEM,
    _COMPARATIVE_DIAGNOSTICS_KEYS,
    generate_comparative_reasoning,
    verify_comparative_reasoning,
)
import checkers.agents.comparative_reasoner as cr


# ── Shared fixtures ───────────────────────────────────────────────────────────

_CHOSEN_FACTS = {
    "opponent_can_recapture": False,
    "creates_immediate_threat": False,
    "shot_sequence_available": False,
    "captures_count": 0,
    "leaves_piece_isolated": False,
    "weakens_king_row": False,
    "our_pieces_threatened_after": 1,
    "results_in_king": False,
    "near_promotion": False,
    "opponent_mobility_before": 5,
    "opponent_mobility_after": 5,
}

_CHOSEN_SAFE = {"path": [[5, 4], [4, 3]], "facts": _CHOSEN_FACTS}

_ALT_AGGRESSIVE = {
    "path": [[5, 2], [4, 3]],
    "facts": {
        "opponent_can_recapture": True,
        "creates_immediate_threat": True,
        "shot_sequence_available": False,
        "captures_count": 1,
        "leaves_piece_isolated": False,
        "weakens_king_row": False,
        "our_pieces_threatened_after": 1,
        "results_in_king": False,
        "near_promotion": False,
        "opponent_mobility_before": 5,
        "opponent_mobility_after": 4,
    },
}

_CANDIDATES = [_CHOSEN_SAFE, _ALT_AGGRESSIVE]

# Zero-contradiction text: all claims about alt [1] match ALT_AGGRESSIVE facts;
# tradeoff "recapture safety" matches CHOSEN_SAFE.opponent_can_recapture=False.
_CLEAN_TEXT = (
    "Aggressive alternative [1] creates an immediate threat but allows "
    "recapture; chosen move forfeits initiative for recapture safety."
)

# One-contradiction text: "avoids recapture" asserts
# ALT_AGGRESSIVE.opponent_can_recapture=False, but the fact is True.
_DIRTY_TEXT = (
    "Aggressive alternative [1] avoids recapture; "
    "chosen move forfeits initiative for recapture safety."
)


# ── Mock callers ──────────────────────────────────────────────────────────────

def _json_resp(text: str) -> str:
    return json.dumps({"comparative_reasoning": text})


def _clean_caller(system: str, user: str) -> str:
    return _json_resp(_CLEAN_TEXT)


def _dirty_then_clean_caller(system: str, user: str) -> str:
    """Generation (RANKER_COMPARATIVE_SYSTEM) → dirty; refinement → clean."""
    if system == RANKER_COMPARATIVE_SYSTEM:
        return _json_resp(_DIRTY_TEXT)
    return _json_resp(_CLEAN_TEXT)


def _always_dirty_caller(system: str, user: str) -> str:
    return _json_resp(_DIRTY_TEXT)


def _always_failing_caller(system: str, user: str) -> str:
    raise RuntimeError("mock API failure")


def _malformed_caller(system: str, user: str) -> str:
    return "not valid json at all"


# ═══════════════════════════════════════════════════════════════════════════
# 0. Fixture sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestFixtureSanity:
    def test_clean_text_has_zero_contradictions(self):
        n = len(verify_comparative_reasoning(_CLEAN_TEXT, _CANDIDATES, _CHOSEN_SAFE))
        assert n == 0, (
            f"_CLEAN_TEXT must have 0 contradictions against the fixtures; got {n}"
        )

    def test_dirty_text_has_at_least_one_contradiction(self):
        n = len(verify_comparative_reasoning(_DIRTY_TEXT, _CANDIDATES, _CHOSEN_SAFE))
        assert n >= 1, (
            f"_DIRTY_TEXT must have ≥1 contradiction against the fixtures; got {n}"
        )

    def test_comparative_diagnostics_keys_constant_has_12_entries(self):
        assert len(_COMPARATIVE_DIAGNOSTICS_KEYS) == 12


# ═══════════════════════════════════════════════════════════════════════════
# 1. Short-circuit path (perfect first sample — no refinement)
# ═══════════════════════════════════════════════════════════════════════════


class TestShortCircuitPath:
    def _run(self, **kw):
        diag: dict = {}
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            diagnostics_out=diag,
            _api_caller=_clean_caller,
            **kw,
        )
        return result, diag

    def test_returns_clean_text(self):
        result, _ = self._run()
        assert result == _CLEAN_TEXT

    def test_short_circuit_flag_set(self):
        _, diag = self._run()
        assert diag["comparative_generation_short_circuited"] is True

    def test_samples_used_is_one(self):
        _, diag = self._run()
        assert diag["comparative_generation_samples_used"] == 1

    def test_contradiction_counts_is_zero_list(self):
        _, diag = self._run()
        assert diag["comparative_sample_contradiction_counts"] == [0]

    def test_no_refinement_attempt(self):
        _, diag = self._run()
        assert diag["comparative_refinement_attempts"] == 0

    def test_paragraph_text_in_diagnostics(self):
        _, diag = self._run()
        assert diag["comparative_paragraph_text"] == _CLEAN_TEXT

    def test_initial_contradictions_zero(self):
        _, diag = self._run()
        assert diag["comparative_initial_contradictions"] == 0

    def test_final_contradictions_zero(self):
        _, diag = self._run()
        assert diag["comparative_final_contradictions"] == 0

    def test_was_not_skipped(self):
        _, diag = self._run()
        assert diag["comparative_was_skipped"] is False

    def test_skip_reason_is_none(self):
        _, diag = self._run()
        assert diag["comparative_skip_reason"] is None

    def test_provider_is_mistral(self):
        _, diag = self._run()
        assert diag["comparative_provider"] == "mistral"

    def test_seeds_non_empty(self):
        _, diag = self._run()
        seeds = diag["comparative_seeds"]
        assert isinstance(seeds, list) and len(seeds) > 0

    def test_groups_non_empty(self):
        _, diag = self._run()
        groups = diag["comparative_groups"]
        assert isinstance(groups, dict) and len(groups) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Refinement path (dirty generation sample, refinement resolves to clean)
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementPath:
    def _run(self):
        diag: dict = {}
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=1,
            diagnostics_out=diag,
            _api_caller=_dirty_then_clean_caller,
        )
        return result, diag

    def test_returns_clean_refined_text(self):
        result, _ = self._run()
        assert result == _CLEAN_TEXT

    def test_short_circuit_flag_not_set(self):
        _, diag = self._run()
        assert diag["comparative_generation_short_circuited"] is False

    def test_refinement_attempts_is_one(self):
        _, diag = self._run()
        assert diag["comparative_refinement_attempts"] == 1

    def test_initial_contradictions_positive(self):
        _, diag = self._run()
        assert diag["comparative_initial_contradictions"] >= 1

    def test_final_contradictions_zero(self):
        _, diag = self._run()
        assert diag["comparative_final_contradictions"] == 0

    def test_paragraph_text_is_clean(self):
        _, diag = self._run()
        assert diag["comparative_paragraph_text"] == _CLEAN_TEXT

    def test_was_not_skipped(self):
        _, diag = self._run()
        assert diag["comparative_was_skipped"] is False

    def test_samples_used_is_one(self):
        _, diag = self._run()
        assert diag["comparative_generation_samples_used"] == 1

    def test_sample_contradiction_counts_all_positive(self):
        _, diag = self._run()
        counts = diag["comparative_sample_contradiction_counts"]
        assert all(n >= 1 for n in counts)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Skip paths
# ═══════════════════════════════════════════════════════════════════════════


class TestSkipPaths:

    # 3a. No groups (only chosen move in candidate list)
    def test_no_groups_returns_none(self):
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, [_CHOSEN_SAFE], _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        assert result is None

    def test_no_groups_skip_reason(self):
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, [_CHOSEN_SAFE], _CHOSEN_FACTS,
            diagnostics_out=diag, _api_caller=_clean_caller,
        )
        assert diag["comparative_skip_reason"] == "no_informative_groups"

    def test_no_groups_was_skipped(self):
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, [_CHOSEN_SAFE], _CHOSEN_FACTS,
            diagnostics_out=diag, _api_caller=_clean_caller,
        )
        assert diag["comparative_was_skipped"] is True

    def test_empty_candidates_returns_none(self):
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, [], _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        assert result is None

    # 3b. No valid samples — API always fails
    def test_api_failure_all_samples_returns_none(self):
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=2, _api_caller=_always_failing_caller,
        )
        assert result is None

    def test_api_failure_skip_reason(self):
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=2, diagnostics_out=diag,
            _api_caller=_always_failing_caller,
        )
        assert diag["comparative_skip_reason"] == "api_failure"

    def test_api_failure_was_skipped(self):
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=2, diagnostics_out=diag,
            _api_caller=_always_failing_caller,
        )
        assert diag["comparative_was_skipped"] is True

    # 3c. No valid samples — malformed JSON on every call
    def test_malformed_json_returns_none(self):
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=2, _api_caller=_malformed_caller,
        )
        assert result is None

    def test_malformed_json_skip_reason(self):
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=2, diagnostics_out=diag,
            _api_caller=_malformed_caller,
        )
        assert diag["comparative_skip_reason"] == "all_samples_rejected"

    # 3d. max_samples=0 → empty loop → no valid samples
    def test_max_samples_zero_returns_none(self):
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=0, _api_caller=_clean_caller,
        )
        assert result is None

    # 3e. Refinement failure (generation → dirty, refinement also → dirty)
    def test_refinement_failure_returns_none(self):
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=1, _api_caller=_always_dirty_caller,
        )
        assert result is None

    def test_refinement_failure_final_contradictions_nonzero(self):
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=1, diagnostics_out=diag,
            _api_caller=_always_dirty_caller,
        )
        assert diag["comparative_final_contradictions"] >= 1

    def test_refinement_failure_paragraph_text_none(self):
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=1, diagnostics_out=diag,
            _api_caller=_always_dirty_caller,
        )
        assert diag["comparative_paragraph_text"] is None

    # 3f. Refinement failure: API failure during refinement
    def _count_calls(self):
        """Counter helper to limit failures to refinement only."""
        calls = []

        def caller(system: str, user: str) -> str:
            calls.append(system)
            if system == RANKER_COMPARATIVE_SYSTEM:
                return _json_resp(_DIRTY_TEXT)
            raise RuntimeError("refinement API failure")

        return caller, calls

    def test_refinement_api_failure_returns_none(self):
        caller, _ = self._count_calls()
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=1, _api_caller=caller,
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. Diagnostics shape and content
# ═══════════════════════════════════════════════════════════════════════════


class TestDiagnosticsShape:

    def _get_diag(self, caller=_clean_caller, candidates=_CANDIDATES, **kw) -> dict:
        diag: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, candidates, _CHOSEN_FACTS,
            diagnostics_out=diag, _api_caller=caller,
            **kw,
        )
        return diag

    def test_exactly_12_keys_on_success(self):
        assert len(self._get_diag()) == 12

    def test_key_names_match_constant_on_success(self):
        assert set(self._get_diag().keys()) == set(_COMPARATIVE_DIAGNOSTICS_KEYS)

    def test_exactly_12_keys_on_no_groups_skip(self):
        diag = self._get_diag(candidates=[_CHOSEN_SAFE])
        assert len(diag) == 12

    def test_key_names_match_constant_on_no_groups_skip(self):
        diag = self._get_diag(candidates=[_CHOSEN_SAFE])
        assert set(diag.keys()) == set(_COMPARATIVE_DIAGNOSTICS_KEYS)

    def test_exactly_12_keys_on_no_valid_samples(self):
        diag = self._get_diag(caller=_always_failing_caller, max_samples=1)
        assert len(diag) == 12

    def test_key_names_match_constant_on_no_valid_samples(self):
        diag = self._get_diag(caller=_always_failing_caller, max_samples=1)
        assert set(diag.keys()) == set(_COMPARATIVE_DIAGNOSTICS_KEYS)

    def test_exactly_12_keys_on_refinement_failure(self):
        diag = self._get_diag(caller=_always_dirty_caller, max_samples=1)
        assert len(diag) == 12

    def test_no_ranker_prefixed_keys(self):
        diag = self._get_diag()
        bad = [k for k in diag if k.startswith("ranker_")]
        assert bad == []

    def test_diagnostics_none_does_not_raise(self):
        result = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            diagnostics_out=None, _api_caller=_clean_caller,
        )
        assert result == _CLEAN_TEXT

    def test_preexisting_keys_not_removed(self):
        diag = {"existing_key": "untouched"}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            diagnostics_out=diag, _api_caller=_clean_caller,
        )
        assert diag["existing_key"] == "untouched"

    def test_only_comparative_keys_added(self):
        diag = {"existing_key": "untouched"}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            diagnostics_out=diag, _api_caller=_clean_caller,
        )
        added = {k for k in diag if k != "existing_key"}
        assert added == set(_COMPARATIVE_DIAGNOSTICS_KEYS)

    def test_groups_stored_as_index_lists(self):
        diag = self._get_diag()
        groups = diag["comparative_groups"]
        assert isinstance(groups, dict)
        for tag, idxs in groups.items():
            assert isinstance(idxs, list), f"group {tag!r} should be a list"
            assert all(isinstance(i, int) for i in idxs)

    def test_seeds_is_list_of_strings(self):
        diag = self._get_diag()
        seeds = diag["comparative_seeds"]
        assert isinstance(seeds, list)
        assert all(isinstance(s, str) for s in seeds)

    def test_sample_contradiction_counts_is_list(self):
        diag = self._get_diag()
        assert isinstance(diag["comparative_sample_contradiction_counts"], list)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Isolation and runtime-unchanged invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestIsolationAndRuntimeUnchanged:

    def test_module_source_no_from_ranker_agent_import(self):
        src = inspect.getsource(cr)
        assert "from checkers.agents.ranker_agent" not in src

    def test_module_source_no_import_ranker_agent(self):
        src = inspect.getsource(cr)
        assert "import checkers.agents.ranker_agent" not in src

    def test_generate_wired_into_ranker_agent_in_step6(self):
        # Step 6 legitimately calls generate_comparative_reasoning from
        # ranker_agent.  The isolation invariant that matters is the
        # one-way dependency: comparative_reasoner must NOT import from
        # ranker_agent (checked above), not the reverse.
        import checkers.agents.explainer_agent as ra
        ra_src = inspect.getsource(ra)
        assert "generate_comparative_reasoning(" in ra_src

    def test_chosen_move_not_mutated(self):
        original = copy.deepcopy(_CHOSEN_SAFE)
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        assert _CHOSEN_SAFE == original

    def test_all_candidates_not_mutated(self):
        original = copy.deepcopy(_CANDIDATES)
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        assert _CANDIDATES == original

    def test_chosen_facts_not_mutated(self):
        original = copy.deepcopy(_CHOSEN_FACTS)
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        assert _CHOSEN_FACTS == original

    def test_none_inputs_do_not_raise(self):
        # Defensive: None chosen_move / all_candidates must not crash.
        result = generate_comparative_reasoning(
            None, None, None,
            _api_caller=_clean_caller,
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 6. Deterministic behaviour
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminism:

    def test_same_inputs_same_output_success_path(self):
        r1 = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        r2 = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        assert r1 == r2

    def test_same_inputs_same_diagnostics_success_path(self):
        d1: dict = {}
        d2: dict = {}
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            diagnostics_out=d1, _api_caller=_clean_caller,
        )
        generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            diagnostics_out=d2, _api_caller=_clean_caller,
        )
        assert d1 == d2

    def test_same_inputs_same_output_refinement_path(self):
        r1 = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=1, _api_caller=_dirty_then_clean_caller,
        )
        r2 = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=1, _api_caller=_dirty_then_clean_caller,
        )
        assert r1 == r2

    def test_same_inputs_same_output_skip_path(self):
        r1 = generate_comparative_reasoning(
            _CHOSEN_SAFE, [_CHOSEN_SAFE], _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        r2 = generate_comparative_reasoning(
            _CHOSEN_SAFE, [_CHOSEN_SAFE], _CHOSEN_FACTS,
            _api_caller=_clean_caller,
        )
        assert r1 is None
        assert r2 is None

    def test_same_inputs_same_output_failure_path(self):
        r1 = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=2, _api_caller=_always_failing_caller,
        )
        r2 = generate_comparative_reasoning(
            _CHOSEN_SAFE, _CANDIDATES, _CHOSEN_FACTS,
            max_samples=2, _api_caller=_always_failing_caller,
        )
        assert r1 is None
        assert r2 is None
