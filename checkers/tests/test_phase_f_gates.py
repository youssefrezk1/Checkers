# checkers/tests/test_phase_f_gates.py
#
# Phase F regression tests for the deterministic post-repair gates added to
# ranker_agent._refine_reasoning:
#
#   (1) monotonicity         — strict-improvement commit; never accept equal
#                              or worse contradiction counts.
#   (2) forced-framing       — forced wording in output IFF a forced-move seed
#                              is present.  Rejects fabricated forced framing
#                              (turn 47 / turn 55 class) and silently-dropped
#                              forced disclosures (turn 3 class).
#   (3) king-promotion       — when facts["results_in_king"]=True, the output
#                              must contain promotion/king/crown wording.
#
# All three gates live in checkers/agents/ranker_agent._validate_and_select.
# They are deterministic: regex over output bytes + a numeric comparison on
# verifier counts.  No prompt instruction, no LLM call, no schema migration.

from __future__ import annotations

from checkers.agents.ranker_agent import (
    _FORCED_FRAMING_RE,
    _PROMOTION_RE,
    _has_forced_seed,
    _validate_and_select,
)


_FORCED_SEED = "This is the only legal move available; the engine assigns it a minimax score of 20.0."
_QUIET_SEED  = "The engine scores this move -29.0 — best available option in a difficult position."

_FACTS_NO_KING   = {"results_in_king": False}
_FACTS_KING_TRUE = {"results_in_king": True}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Monotonicity
# ─────────────────────────────────────────────────────────────────────────────

class TestMonotonicity:
    """Strict improvement only — never accept equal or worse counts."""

    def test_strict_improvement_accepted(self):
        out, count, decision = _validate_and_select(
            prior_text="prior", prior_count=3,
            repair_text="repaired", repair_count=1,
            seeds=[_QUIET_SEED], facts=_FACTS_NO_KING,
        )
        assert decision == "accepted"
        assert out == "repaired" and count == 1

    def test_equal_count_rejected(self):
        out, count, decision = _validate_and_select(
            prior_text="prior", prior_count=2,
            repair_text="repaired", repair_count=2,
            seeds=[_QUIET_SEED], facts=_FACTS_NO_KING,
        )
        assert decision == "rejected_no_improvement"
        assert out == "prior" and count == 2

    def test_worse_count_rejected(self):
        # Reproduces turn 61: init=1 -> repair=3.  Must keep prior.
        out, count, decision = _validate_and_select(
            prior_text="prior", prior_count=1,
            repair_text="repaired_with_more_contradictions", repair_count=3,
            seeds=[_QUIET_SEED], facts=_FACTS_NO_KING,
        )
        assert decision == "rejected_no_improvement"
        assert out == "prior" and count == 1

    def test_empty_repair_text_rejected(self):
        out, count, decision = _validate_and_select(
            prior_text="prior", prior_count=2,
            repair_text=None, repair_count=0,
            seeds=[_QUIET_SEED], facts=_FACTS_NO_KING,
        )
        assert decision == "rejected_parse"
        assert out == "prior" and count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. Forced-framing gate (bidirectional)
# ─────────────────────────────────────────────────────────────────────────────

class TestForcedFramingDetector:
    """The output-side regex must catch the canonical forced-framing surface
    forms observed across audits, including the synonym-bypass class."""

    POSITIVE_CASES = [
        "this is the only legal move available",
        "this move is the only legal positional option available",   # turn 55 bypass
        "this move is the only available choice",
        "there is no alternative for the engine to consider",
        "there is no other option",
        "we have no other choice",
        "we have no choice but to advance",
        "this is a forced jump",
        "this is a forced capture",
        "this is a forced move",
        "this is a forced sequence",
        "the rules require us; we must capture",
        "we must jump",
        "we must take the opponent piece",
        "this is a mandatory advance",
        "this is compulsory",
        "the position cannot avoid this exchange",
    ]

    NEGATIVE_CASES = [
        "the move improves piece placement without capturing",
        "the engine favours this advance for tactical reasons",
        "the opponent is forced to respond with a jump",            # opponent-side
        "we restrict the opponent's mobility",
        "we secure central control",
        "the minimax score confirms this as the highest-evaluated option",
    ]

    def test_positive_cases_all_match(self):
        for text in self.POSITIVE_CASES:
            assert _FORCED_FRAMING_RE.search(text), \
                f"forced-framing regex missed: {text!r}"

    def test_negative_cases_none_match(self):
        for text in self.NEGATIVE_CASES:
            assert not _FORCED_FRAMING_RE.search(text), \
                f"forced-framing regex false-positive on: {text!r}"


class TestForcedSeedDetection:
    """_has_forced_seed must recognise the three canonical seed markers."""

    def test_only_legal_move_marker(self):
        assert _has_forced_seed([_FORCED_SEED])

    def test_mandatory_jump_marker(self):
        assert _has_forced_seed(["A mandatory jump is on the board."])

    def test_must_capture_marker(self):
        assert _has_forced_seed(["The player must capture this turn."])

    def test_opponent_forced_seed_does_not_count(self):
        # The audit's most common confounder.
        seed = "The opponent is forced to respond with a jump (at most 1 piece(s) captured)."
        assert not _has_forced_seed([seed])

    def test_empty_seeds(self):
        assert not _has_forced_seed([])
        assert not _has_forced_seed(None)


class TestForcedFramingGateBidirectional:
    """Output must contain forced wording IFF seeds contain a forced marker."""

    def test_fabricated_forced_in_output_no_seed_rejected(self):
        # Reproduces turn 47 / turn 55 class.
        out, _, decision = _validate_and_select(
            prior_text="prior",
            prior_count=2,
            repair_text="This move is the only legal positional option available.",
            repair_count=1,
            seeds=[_QUIET_SEED],
            facts=_FACTS_NO_KING,
        )
        assert decision == "rejected_forced_fabricated"
        assert out == "prior"

    def test_dropped_forced_disclosure_seed_present_rejected(self):
        # Reproduces turn 3 class — seeds say forced, repair output does not.
        out, _, decision = _validate_and_select(
            prior_text="prior",
            prior_count=2,
            repair_text="The jump captures one piece and improves placement.",
            repair_count=1,
            seeds=[_FORCED_SEED],
            facts=_FACTS_NO_KING,
        )
        assert decision == "rejected_forced_dropped"
        assert out == "prior"

    def test_forced_seed_and_forced_output_accepted(self):
        out, count, decision = _validate_and_select(
            prior_text="prior",
            prior_count=2,
            repair_text="This is the only legal move available; the engine scores it 20.0.",
            repair_count=1,
            seeds=[_FORCED_SEED],
            facts=_FACTS_NO_KING,
        )
        assert decision == "accepted"
        assert "only legal" in out and count == 1

    def test_no_seed_no_forced_output_accepted(self):
        out, count, decision = _validate_and_select(
            prior_text="prior",
            prior_count=2,
            repair_text="The move advances forward without capturing.",
            repair_count=1,
            seeds=[_QUIET_SEED],
            facts=_FACTS_NO_KING,
        )
        assert decision == "accepted"
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. King-promotion preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestPromotionDetector:
    """The output-side regex must catch the canonical promotion vocabulary."""

    POSITIVE_CASES = [
        "the piece is promoted to king",
        "this move promotes the piece",
        "the move secures immediate promotion",
        "the new king occupies the center",
        "the piece crowns at the back rank",
        "this advance crowns the moved piece immediately",
    ]

    NEGATIVE_CASES = [
        "the move improves piece placement without capturing",
        "the piece advances toward the center",
        "the engine scores this move 20.0",
        "our mobility increases from 7 to 9",
    ]

    def test_positive_cases_all_match(self):
        for text in self.POSITIVE_CASES:
            assert _PROMOTION_RE.search(text), \
                f"promotion regex missed: {text!r}"

    def test_negative_cases_none_match(self):
        for text in self.NEGATIVE_CASES:
            assert not _PROMOTION_RE.search(text), \
                f"promotion regex false-positive on: {text!r}"


class TestKingPromotionGate:
    """results_in_king=True ⇒ output must contain promotion token."""

    def test_king_true_without_promotion_token_rejected(self):
        # Reproduces turn 55 audit (Phase E run) — king promotion silently dropped.
        out, _, decision = _validate_and_select(
            prior_text="prior",
            prior_count=2,
            repair_text="The move advances without capturing and improves placement.",
            repair_count=1,
            seeds=[_QUIET_SEED],
            facts=_FACTS_KING_TRUE,
        )
        assert decision == "rejected_promotion_dropped"
        assert out == "prior"

    def test_king_true_with_promotion_token_accepted(self):
        out, count, decision = _validate_and_select(
            prior_text="prior",
            prior_count=2,
            repair_text="The piece crowns immediately upon reaching the back rank.",
            repair_count=1,
            seeds=[_QUIET_SEED],
            facts=_FACTS_KING_TRUE,
        )
        assert decision == "accepted"
        assert count == 1

    def test_king_false_gate_inert(self):
        # When results_in_king is False, the gate must not fire on the absence
        # of promotion vocabulary.
        out, count, decision = _validate_and_select(
            prior_text="prior",
            prior_count=2,
            repair_text="The move advances without capturing.",
            repair_count=1,
            seeds=[_QUIET_SEED],
            facts=_FACTS_NO_KING,
        )
        assert decision == "accepted"
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Combined behaviour — preserve prior text on every rejection path
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedGatesPreservePriorOnRejection:
    """Every rejection path must return the prior text/count unchanged."""

    PRIOR_TEXT  = "the prior accepted paragraph"
    PRIOR_COUNT = 2

    def _check_preserved(self, repair_text, repair_count, seeds, facts, expected_decision):
        out, count, decision = _validate_and_select(
            prior_text=self.PRIOR_TEXT,
            prior_count=self.PRIOR_COUNT,
            repair_text=repair_text,
            repair_count=repair_count,
            seeds=seeds,
            facts=facts,
        )
        assert decision == expected_decision
        assert out == self.PRIOR_TEXT
        assert count == self.PRIOR_COUNT

    def test_parse_failure_preserves_prior(self):
        self._check_preserved(
            repair_text=None,
            repair_count=0,
            seeds=[_QUIET_SEED],
            facts=_FACTS_NO_KING,
            expected_decision="rejected_parse",
        )

    def test_no_improvement_preserves_prior(self):
        self._check_preserved(
            repair_text="some equal-count candidate",
            repair_count=self.PRIOR_COUNT,
            seeds=[_QUIET_SEED],
            facts=_FACTS_NO_KING,
            expected_decision="rejected_no_improvement",
        )

    def test_forced_fabricated_preserves_prior(self):
        self._check_preserved(
            repair_text="this move is the only legal move available",
            repair_count=1,
            seeds=[_QUIET_SEED],
            facts=_FACTS_NO_KING,
            expected_decision="rejected_forced_fabricated",
        )

    def test_forced_dropped_preserves_prior(self):
        self._check_preserved(
            repair_text="the jump captures one piece quietly",
            repair_count=1,
            seeds=[_FORCED_SEED],
            facts=_FACTS_NO_KING,
            expected_decision="rejected_forced_dropped",
        )

    def test_promotion_dropped_preserves_prior(self):
        self._check_preserved(
            repair_text="the piece advances forward without capturing",
            repair_count=1,
            seeds=[_QUIET_SEED],
            facts=_FACTS_KING_TRUE,
            expected_decision="rejected_promotion_dropped",
        )
