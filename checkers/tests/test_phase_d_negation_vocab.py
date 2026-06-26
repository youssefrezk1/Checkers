# checkers/tests/test_phase_d_negation_vocab.py
#
# Phase D regression tests: expanded negation-aware context-forbidden-vocab check.
#
#   D1 — _ctx_phrase_negated helper returns True for all recognised negation
#        patterns and False for positive assertions or missing phrase.
#
#   D2 — Runtime _check_reasoning_truthfulness no longer fires
#        forbidden_vocab_ctx:new vulnerabilities for negated forms.
#
#   D3 — Evaluator _check_forbidden_vocab mirrors the same suppression (E.1 parity).
#
#   D4 — Positive assertions of "new vulnerabilities" still produce a warning /
#        CONTRADICTED record on both sides.
#
#   D5 — Mixed text (one negated + one positive occurrence) still fires.

from __future__ import annotations

import pytest

from checkers.agents.explainer_agent import (
    _ctx_phrase_negated,
    _check_reasoning_truthfulness,
)
from checkers.evaluation.unified_verifier import (
    _ctx_phrase_negated as _uv_ctx_phrase_negated,
    contradictions_only,
    assert_runtime_evaluator_agreement,
)

# ── Minimal facts dict that doesn't trigger any other checks ─────────────────

_QUIET_FACTS = {
    "captures_count": 0,
    "net_gain": 0,
    "opponent_can_recapture": False,
    "our_pieces_threatened_after": 0,
    "creates_immediate_threat": False,
    "leaves_piece_isolated": False,
    "opponent_mobility_before": 7,
    "opponent_mobility_after": 7,
    "minimax_score": -2.0,
}

_PHRASE = "new vulnerabilities"


# ═══════════════════════════════════════════════════════════════════════════
# D1 — _ctx_phrase_negated unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCtxPhraseNegatedHelper:
    """Direct tests of the helper function in both modules (must be identical)."""

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_no_prefix(self, helper):
        assert helper(f"this move avoids no {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_without_creating(self, helper):
        assert helper(f"without creating {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_without_exposing(self, helper):
        assert helper(f"without exposing {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_without_introducing(self, helper):
        assert helper(f"without introducing {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_avoids_introducing(self, helper):
        assert helper(f"avoids introducing {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_avoid_creating(self, helper):
        assert helper(f"designed to avoid creating {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_prevents(self, helper):
        assert helper(f"prevents {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_negated_never_introduces(self, helper):
        assert helper(f"never introduces {_PHRASE}", _PHRASE) is True

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_positive_creates(self, helper):
        assert helper(f"this move creates {_PHRASE}", _PHRASE) is False

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_positive_introduces(self, helper):
        assert helper(f"this move introduces {_PHRASE}", _PHRASE) is False

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_positive_opens(self, helper):
        assert helper(f"this move opens {_PHRASE}", _PHRASE) is False

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_phrase_not_in_text(self, helper):
        assert helper("completely unrelated text", _PHRASE) is False

    @pytest.mark.parametrize("helper", [_ctx_phrase_negated, _uv_ctx_phrase_negated])
    def test_mixed_negated_and_positive_returns_false(self, helper):
        # One un-negated occurrence means the overall check should fire.
        mixed = f"without creating {_PHRASE} but later creates {_PHRASE}"
        assert helper(mixed, _PHRASE) is False


# ═══════════════════════════════════════════════════════════════════════════
# D2 — Runtime check: negated forms no longer produce a warning
# ═══════════════════════════════════════════════════════════════════════════

class TestRuntimeNegatedFormsAllowed:

    def _warnings(self, reasoning: str) -> list[str]:
        return _check_reasoning_truthfulness(reasoning, _QUIET_FACTS, seeds=[])

    def _vuln_warnings(self, reasoning: str) -> list[str]:
        return [w for w in self._warnings(reasoning) if "new vulnerabilities" in w]

    def test_no_warning_for_no_prefix(self):
        assert not self._vuln_warnings(
            "The move is solid and introduces no new vulnerabilities."
        )

    def test_no_warning_for_without_creating(self):
        assert not self._vuln_warnings(
            "The piece advances without creating new vulnerabilities in the position."
        )

    def test_no_warning_for_without_exposing(self):
        assert not self._vuln_warnings(
            "This advance is safe without exposing new vulnerabilities."
        )

    def test_no_warning_for_avoids_introducing(self):
        assert not self._vuln_warnings(
            "The chosen move avoids introducing new vulnerabilities while improving position."
        )

    def test_no_warning_for_prevents(self):
        assert not self._vuln_warnings(
            "This consolidation prevents new vulnerabilities from forming."
        )


# ═══════════════════════════════════════════════════════════════════════════
# D3 — Evaluator: same suppression (E.1 parity)
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluatorNegatedFormsAllowed:

    def _vuln_records(self, reasoning: str) -> list:
        records = contradictions_only(
            reasoning, facts=_QUIET_FACTS, reasoning_seeds=[]
        )
        return [r for r in records if "new vulnerabilities" in r.claim_type]

    def test_no_record_for_no_prefix(self):
        assert not self._vuln_records(
            "The move is solid and introduces no new vulnerabilities."
        )

    def test_no_record_for_without_creating(self):
        assert not self._vuln_records(
            "The piece advances without creating new vulnerabilities in the position."
        )

    def test_no_record_for_without_exposing(self):
        assert not self._vuln_records(
            "This advance is safe without exposing new vulnerabilities."
        )

    def test_no_record_for_avoids_introducing(self):
        assert not self._vuln_records(
            "The chosen move avoids introducing new vulnerabilities while improving position."
        )

    def test_no_record_for_prevents(self):
        assert not self._vuln_records(
            "This consolidation prevents new vulnerabilities from forming."
        )


# ═══════════════════════════════════════════════════════════════════════════
# D4 — Positive assertions still fire on both sides
# ═══════════════════════════════════════════════════════════════════════════

class TestPositiveAssertionsStillFire:

    def _runtime_has_vuln_warning(self, reasoning: str) -> bool:
        warnings = _check_reasoning_truthfulness(reasoning, _QUIET_FACTS, [])
        return any("new vulnerabilities" in w for w in warnings)

    def _evaluator_has_vuln_record(self, reasoning: str) -> bool:
        records = contradictions_only(
            reasoning, facts=_QUIET_FACTS, reasoning_seeds=[]
        )
        return any("new vulnerabilities" in r.claim_type for r in records)

    def test_runtime_fires_for_creates(self):
        assert self._runtime_has_vuln_warning(
            "This move creates new vulnerabilities but wins a piece."
        )

    def test_evaluator_fires_for_creates(self):
        assert self._evaluator_has_vuln_record(
            "This move creates new vulnerabilities but wins a piece."
        )

    def test_runtime_fires_for_introduces(self):
        assert self._runtime_has_vuln_warning(
            "The sacrifice introduces new vulnerabilities on the king side."
        )

    def test_evaluator_fires_for_introduces(self):
        assert self._evaluator_has_vuln_record(
            "The sacrifice introduces new vulnerabilities on the king side."
        )

    def test_runtime_fires_for_opens(self):
        assert self._runtime_has_vuln_warning(
            "Advancing here opens new vulnerabilities the opponent can exploit."
        )

    def test_evaluator_fires_for_opens(self):
        assert self._evaluator_has_vuln_record(
            "Advancing here opens new vulnerabilities the opponent can exploit."
        )


# ═══════════════════════════════════════════════════════════════════════════
# D5 — E.1 parity: runtime and evaluator agree on all cases
# ═══════════════════════════════════════════════════════════════════════════

_PARITY_CASES = [
    ("negated_no",        "The advance introduces no new vulnerabilities.",       False),
    ("negated_without",   "The advance is safe without creating new vulnerabilities.", False),
    ("negated_avoids",    "The move avoids introducing new vulnerabilities.",     False),
    ("positive_creates",  "The move creates new vulnerabilities.",                True),
    ("positive_opens",    "This opens new vulnerabilities for the opponent.",     True),
    ("mixed",             "without creating new vulnerabilities, this also creates new vulnerabilities", True),
]


class TestE1ParityNegationVocab:

    @pytest.mark.parametrize("label,reasoning,expect_contradiction", _PARITY_CASES)
    def test_runtime_evaluator_agree(self, label, reasoning, expect_contradiction):
        runtime_warnings = _check_reasoning_truthfulness(reasoning, _QUIET_FACTS, [])
        assert_runtime_evaluator_agreement(
            runtime_warnings, reasoning, facts=_QUIET_FACTS, reasoning_seeds=[]
        )

    @pytest.mark.parametrize("label,reasoning,expect_contradiction", _PARITY_CASES)
    def test_runtime_produces_expected_result(self, label, reasoning, expect_contradiction):
        warnings = _check_reasoning_truthfulness(reasoning, _QUIET_FACTS, [])
        has_vuln = any("new vulnerabilities" in w for w in warnings)
        assert has_vuln == expect_contradiction, (
            f"[{label}] runtime expected contradiction={expect_contradiction}, "
            f"got warnings={[w for w in warnings if 'new vulnerabilities' in w]}"
        )

    @pytest.mark.parametrize("label,reasoning,expect_contradiction", _PARITY_CASES)
    def test_evaluator_produces_expected_result(self, label, reasoning, expect_contradiction):
        records = contradictions_only(
            reasoning, facts=_QUIET_FACTS, reasoning_seeds=[]
        )
        has_vuln = any("new vulnerabilities" in r.claim_type for r in records)
        assert has_vuln == expect_contradiction, (
            f"[{label}] evaluator expected contradiction={expect_contradiction}, "
            f"got records={[r.claim_type for r in records if 'new vulnerabilities' in r.claim_type]}"
        )
