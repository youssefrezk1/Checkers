# checkers/tests/test_phase_e_f1_over_fix.py
#
# F1-OVER regression tests.  After the Phase E forced-move front-loading fix
# (see test_phase_e_fixes.py), an over-application regression appeared: the
# LLM began asserting "the only legal move available" on regular simple moves
# whose seeds contained no such marker (5/21 simple moves in
# logs/manual_reasoning_trace/20260526_184553 — turns 29, 31, 37, 43, 49).
#
# Root cause: the Phase E instruction was unconditional; the LLM treated it as
# preferred narrative style instead of a strict conditional.  It also confused
# 'the opponent is forced to respond' seeds with 'our move was forced'.
#
# Fix: tighten the instruction to be IF-AND-ONLY-IF gated on the actual seed
# list, and explicitly disclaim the opponent-forced-response case.
#
# These tests verify the prompt text carries:
#   1. an "if and only if" / strict gating phrasing
#   2. a disclaimer that "opponent is forced to respond" does NOT make our
#      move forced
#   3. the original front-loading requirement for the true-forced case

from __future__ import annotations

from checkers.agents.ranker_agent import _build_seed_reasoning_prompt


_CHOSEN_MOVE = {
    "path": [[5, 6], [4, 5]],
    "type": "simple",
    "facts": {
        "captures_count": 0,
        "net_gain": 0,
        "opponent_can_recapture": False,
        "our_pieces_threatened_after": 0,
        "minimax_score": -2.0,
    },
}

_NON_FORCED_SEEDS = [
    "The moved piece cannot be immediately recaptured.",
    "The opponent is forced to respond with a jump (at most 1 piece(s) captured).",
    "The engine scores this move -30.0 — best available option in a difficult position.",
]

_FORCED_SEEDS = [
    "The move captures 1 piece(s), gaining a net advantage of 1.",
    "This is the only legal move available; the engine assigns it a minimax score of 20.0.",
]


class TestF1OverFixStrictGating:
    """The prompt must carry an 'if and only if' (or equivalent strict)
    gating clause so the forced-move instruction does not fire on every turn."""

    def test_if_and_only_if_gating_present(self):
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _NON_FORCED_SEEDS).lower()
        assert "if and only if" in p, (
            "forced-move framing must be gated by 'if and only if' so the LLM "
            "does not apply it opportunistically"
        )

    def test_gating_keyed_to_seed_presence(self):
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _NON_FORCED_SEEDS).lower()
        # The condition must mention the seed list as the trigger.
        assert "seed" in p
        # Must enumerate the qualifying markers.
        assert "only legal move" in p
        assert "mandatory jump" in p
        assert "must capture" in p

    def test_negative_assertion_when_not_forced(self):
        # Prompt must explicitly prohibit asserting forced/only-legal when no
        # such seed is present.
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _NON_FORCED_SEEDS).lower()
        assert "do not assert" in p
        # And spell out the words it must not assert.
        assert "forced" in p and "only-legal" in p
        assert "no alternative" in p


class TestF1OverFixOpponentForcedDisclaimer:
    """The prompt must explicitly state that 'opponent is forced to respond'
    does NOT make our move forced — this was the most common misread."""

    def test_disclaimer_present(self):
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _NON_FORCED_SEEDS).lower()
        assert "opponent is forced to respond" in p

    def test_disclaimer_clarifies_about_their_reply(self):
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _NON_FORCED_SEEDS).lower()
        # The disclaimer must make clear this describes the opponent, not us.
        assert "opponent" in p and "reply" in p
        # And must explicitly say it does NOT make our move forced.
        assert "does not make our" in p or "does not mean our" in p


class TestF1OverFixTrueForcedStillFrontLoaded:
    """Regression guard: when the seeds DO contain a forced-move marker, the
    front-loading requirement must still be present in the prompt."""

    def test_must_open_with_disclosure_clause_retained(self):
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _FORCED_SEEDS).lower()
        assert "must open with" in p

    def test_closing_sentence_burial_still_prohibited(self):
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _FORCED_SEEDS).lower()
        assert "closing sentence" in p

    def test_forced_markers_enumerated_for_gating(self):
        p = _build_seed_reasoning_prompt(_CHOSEN_MOVE, _FORCED_SEEDS).lower()
        # The qualifying-seed markers must remain enumerated.
        for marker in ("only legal move", "mandatory jump", "must capture"):
            assert marker in p, f"missing forced-move marker: {marker!r}"
