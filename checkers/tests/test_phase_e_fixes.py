# checkers/tests/test_phase_e_fixes.py
#
# Phase E regression tests.  Two targeted prompt-only fixes from the
# 20260526_182158 audit:
#
#   E-F1 — Forced-move disclosure must be front-loaded
#          The seed-reasoning prompt must instruct the LLM to OPEN the paragraph
#          with the forced-move disclosure when the move is forced (only legal,
#          mandatory jump, or equivalent), not bury it in the closing sentence.
#
#   E-R1 — Repair prompts must protect high-salience grounded facts
#          Both repair-prompt builders must list specific protected categories
#          (forced-move disclosures, must-capture, mobility transitions, king
#          promotion, comparative anchors) that may be paraphrased but never
#          silently dropped during repair.

from __future__ import annotations

from checkers.agents.explainer_agent import (
    _build_refinement_prompt,
    _build_seed_reasoning_prompt,
    _build_targeted_refinement_prompt,
)


_CHOSEN_MOVE = {
    "path": [[4, 5], [2, 3]],
    "type": "jump",
    "facts": {
        "captures_count": 1,
        "net_gain": 1,
        "opponent_can_recapture": True,
        "our_pieces_threatened_after": 1,
        "our_mobility_before": 8,
        "our_mobility_after": 7,
        "opponent_mobility_before": 8,
        "opponent_mobility_after": 7,
        "minimax_score": 20.0,
    },
}

_FORCED_SEEDS = [
    "The move captures 1 piece(s), gaining a net advantage of 1.",
    "opponent mobility changes from 8 to 7 — reduces opponent mobility by 1",
    "our mobility changes from 8 to 7 — decreases our mobility by 1",
    "This is the only legal move available; the engine assigns it a minimax score of 20.0.",
]

_CONTRADICTION = (
    "REASONING_CONTRADICTION: our-mobility decrease seeded but "
    "omitted from reasoning (negative_fact_omission)"
)


# ═══════════════════════════════════════════════════════════════════════════
# E-F1 — Forced-move front-loading instruction in seed reasoning prompt
# ═══════════════════════════════════════════════════════════════════════════

class TestEF1ForcedMoveFrontLoadingInstruction:
    """_build_seed_reasoning_prompt must instruct the LLM to OPEN the
    paragraph with the forced-move fact when applicable."""

    def _prompt(self) -> str:
        return _build_seed_reasoning_prompt(_CHOSEN_MOVE, _FORCED_SEEDS)

    def test_mentions_forced_move_condition(self):
        lower = self._prompt().lower()
        assert "forced" in lower
        assert "only legal" in lower or "only legal move" in lower

    def test_requires_opening_disclosure(self):
        lower = self._prompt().lower()
        # Instruction must require opening the paragraph with the fact.
        assert "open" in lower or "opens" in lower

    def test_prohibits_burying_in_closing_sentence(self):
        lower = self._prompt().lower()
        assert "closing sentence" in lower

    def test_instruction_present_even_without_forced_seed(self):
        # The instruction is global to the prompt; it does not require any
        # specific seed to be activated. (Conditional on the seed list at
        # generation time, not on the prompt itself.)
        prompt = _build_seed_reasoning_prompt(
            _CHOSEN_MOVE,
            ["The move captures 1 piece(s), gaining a net advantage of 1."],
        )
        assert "forced" in prompt.lower()


# ═══════════════════════════════════════════════════════════════════════════
# E-R1 — Repair prompts must enumerate protected grounded categories
# ═══════════════════════════════════════════════════════════════════════════

_PROTECTED_PHRASES = [
    "only-legal-move",
    "must-capture",
    "mobility transitions",
    "king-promotion",
    "comparative anchors",
]


class TestER1RefinementPromptProtectsGroundedFacts:
    """_build_refinement_prompt must list each protected category and
    explicitly forbid silently dropping them."""

    def _prompt(self) -> str:
        return _build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION])

    def test_lists_all_protected_categories(self):
        lower = self._prompt().lower()
        for phrase in _PROTECTED_PHRASES:
            assert phrase in lower, f"missing protected category: {phrase!r}"

    def test_forbids_silent_drop(self):
        lower = self._prompt().lower()
        assert "silently drop" in lower or "may not silently drop" in lower

    def test_allows_paraphrase(self):
        lower = self._prompt().lower()
        assert "paraphrase" in lower


class TestER1TargetedPromptProtectsGroundedFacts:
    """_build_targeted_refinement_prompt must carry the same protections."""

    def _prompt(self) -> str:
        bad = ["Some false sentence."]
        return _build_targeted_refinement_prompt(
            _CHOSEN_MOVE, bad, [_CONTRADICTION]
        )

    def test_lists_all_protected_categories(self):
        lower = self._prompt().lower()
        for phrase in _PROTECTED_PHRASES:
            assert phrase in lower, f"missing protected category: {phrase!r}"

    def test_forbids_silent_drop(self):
        lower = self._prompt().lower()
        assert "silently drop" in lower or "may not silently drop" in lower

    def test_allows_paraphrase(self):
        lower = self._prompt().lower()
        assert "paraphrase" in lower
