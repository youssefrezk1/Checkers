# checkers/tests/test_phase_e_f1_over_repair_mirror.py
#
# F1-OVER repair-mirror regression tests.  After the seed-reasoning-side
# F1-OVER fix (see test_phase_e_f1_over_fix.py), the audit of
# logs/manual_reasoning_trace/20260526_190654 surfaced a remaining blind spot:
# the repair prompts had no strict gating, so the repair LLM could invent a
# forced-move framing (e.g., turn 47: "This move is the only legal option
# available, as the rules require capturing when possible and no captures
# exist here.") even though no forced-move seed was present.
#
# Fix: mirror the same IF-AND-ONLY-IF strict-gating clause and opponent-forced
# disclaimer into BOTH _build_refinement_prompt and
# _build_targeted_refinement_prompt.  Preserve E.1 symmetry.
#
# These tests verify:
#   1. both repair prompts contain the strict IF-AND-ONLY-IF gating
#   2. both contain the opponent-forced-response disclaimer
#   3. true forced-move preservation wording (R1 clause) still present
#   4. E.1 symmetry: both prompts carry the same gating shape

from __future__ import annotations

from checkers.agents.explainer_agent import (
    _build_refinement_prompt,
    _build_targeted_refinement_prompt,
)


_CHOSEN_MOVE = {
    "path": [[5, 2], [4, 3]],
    "type": "simple",
    "facts": {
        "captures_count": 0,
        "net_gain": 0,
        "opponent_can_recapture": False,
        "our_pieces_threatened_after": 0,
        "our_mobility_before": 8,
        "our_mobility_after": 9,
        "opponent_mobility_before": 8,
        "opponent_mobility_after": 8,
        "minimax_score": -29.0,
    },
}

_CONTRADICTION = (
    "REASONING_CONTRADICTION: claims center_control but center_control=false"
)


def _normalize(text: str) -> str:
    """Collapse all whitespace so substring checks are insensitive to the
    prompt's line-wrap formatting (the prompts wrap long instructions across
    multiple lines with leading indentation)."""
    import re as _re
    return _re.sub(r"\s+", " ", text).lower()


def _full() -> str:
    return _normalize(_build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION]))


def _targeted() -> str:
    bad = ["The piece occupies the center of the board."]
    return _normalize(_build_targeted_refinement_prompt(
        _CHOSEN_MOVE, bad, [_CONTRADICTION]
    ))


class TestRepairMirrorStrictGating:
    """Both repair prompts must carry the strict 'if and only if' gating
    so the repair LLM cannot invent forced-move framing."""

    def test_full_repair_has_if_and_only_if(self):
        assert "if and only if" in _full().lower()

    def test_targeted_repair_has_if_and_only_if(self):
        assert "if and only if" in _targeted().lower()

    def test_full_repair_lists_forced_markers(self):
        lower = _full().lower()
        for marker in ("only legal move", "mandatory jump", "must capture"):
            assert marker in lower, f"missing forced-move marker: {marker!r}"

    def test_targeted_repair_lists_forced_markers(self):
        lower = _targeted().lower()
        for marker in ("only legal move", "mandatory jump", "must capture"):
            assert marker in lower, f"missing forced-move marker: {marker!r}"

    def test_full_repair_negative_assertion(self):
        lower = _full().lower()
        # Must explicitly prohibit introducing/inventing forced framing.
        assert "do not introduce" in lower or "do not invent" in lower
        # And enumerate the prohibited words.
        assert "forced" in lower and "only-legal" in lower
        assert "no alternative" in lower

    def test_targeted_repair_negative_assertion(self):
        lower = _targeted().lower()
        assert "do not introduce" in lower or "do not invent" in lower
        assert "forced" in lower and "only-legal" in lower
        assert "no alternative" in lower


class TestRepairMirrorOpponentForcedDisclaimer:
    """Both repair prompts must explicitly state that
    'opponent is forced to respond' does NOT make our move forced."""

    def test_full_repair_disclaimer(self):
        lower = _full().lower()
        assert "opponent is forced to respond" in lower
        assert "does not make our move forced" in lower

    def test_targeted_repair_disclaimer(self):
        lower = _targeted().lower()
        assert "opponent is forced to respond" in lower
        assert "does not make our move forced" in lower


class TestRepairMirrorPreservesR1Clause:
    """Regression guard: the existing R1 preservation clause (protected
    grounded-fact categories) must still be present in both prompts."""

    PROTECTED = [
        "only-legal-move",
        "must-capture",
        "mobility transitions",
        "king-promotion",
        "comparative anchors",
    ]

    def test_full_repair_keeps_r1_protected_list(self):
        lower = _full().lower()
        for phrase in self.PROTECTED:
            assert phrase in lower, f"R1 protected category missing: {phrase!r}"
        assert "silently drop" in lower

    def test_targeted_repair_keeps_r1_protected_list(self):
        lower = _targeted().lower()
        for phrase in self.PROTECTED:
            assert phrase in lower, f"R1 protected category missing: {phrase!r}"
        assert "silently drop" in lower


class TestRepairMirrorE1Symmetry:
    """E.1 parity: both repair prompts must share the same forced-move
    gating shape so the runtime and evaluator sides repair identically."""

    def test_full_and_targeted_share_gating_signature(self):
        full = _full().lower()
        tgt = _targeted().lower()
        signature_phrases = [
            "if and only if",
            "only legal move",
            "mandatory jump",
            "must capture",
            "opponent is forced to respond",
            "does not make our move forced",
        ]
        for phrase in signature_phrases:
            assert phrase in full, f"full repair missing: {phrase!r}"
            assert phrase in tgt, f"targeted repair missing: {phrase!r}"
