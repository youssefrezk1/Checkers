# checkers/tests/test_phase_d_fixes.py
#
# Phase D regression tests.  Two targeted fixes from the 20260526_174201 audit:
#
#   D1 — mobility_decrease omission false positives
#        The verifier (and its mirrored runtime check in explainer_agent) missed
#        valid synonym / gerund acknowledgements of our-mobility decrease
#        ("reducing our mobility", "our mobility narrows from X to Y", etc.),
#        causing destructive repair cascades on semantically correct reasoning.
#
#   D2 — Repair must preserve grounded facts
#        Both repair-prompt builders must instruct the LLM to retain factual
#        claims already grounded in Key Facts / seeds while fixing only the
#        flagged contradiction (prevents collateral loss like dropping a king
#        promotion claim during an unrelated repair).

from __future__ import annotations

from checkers.agents.explainer_agent import (
    _build_refinement_prompt,
    _build_targeted_refinement_prompt,
    _check_reasoning_truthfulness,
)
from checkers.evaluation.unified_verifier import (
    _check_mobility_decrease_omission,
)


# ── Shared fixtures ─────────────────────────────────────────────────────────

_DECREASE_SEED = "our mobility changes from 8 to 7 — decreases our mobility by 1"
_SEEDS_WITH_DECREASE = [
    "The piece advances forward without capturing.",
    _DECREASE_SEED,
]

_CHOSEN_MOVE = {
    "path": [[5, 6], [4, 5]],
    "type": "simple",
    "facts": {
        "captures_count": 0,
        "net_gain": 0,
        "opponent_can_recapture": False,
        "our_pieces_threatened_after": 0,
        "creates_immediate_threat": False,
        "leaves_piece_isolated": False,
        "opponent_mobility_before": 7,
        "opponent_mobility_after": 7,
        "our_mobility_before": 8,
        "our_mobility_after": 7,
        "minimax_score": -2.0,
    },
}

_CONTRADICTION = (
    "REASONING_CONTRADICTION: 'narrowing the gap' is wrong when "
    "our_mobility_after=7 >= opponent_mobility_after=7"
)


# ═══════════════════════════════════════════════════════════════════════════
# D1 — Synonym/gerund acknowledgements must clear the omission check
# ═══════════════════════════════════════════════════════════════════════════

class TestD1UnifiedVerifierAcceptsSynonyms:
    """unified_verifier._check_mobility_decrease_omission must accept the
    extended set of acknowledgement phrasings observed in real reasoning."""

    def _check(self, text: str) -> list:
        # Function signature expects lowercased text.
        return _check_mobility_decrease_omission(text.lower(), _SEEDS_WITH_DECREASE)

    def test_reducing_our_mobility_clears_omission(self):
        text = "The move advances safely while reducing our mobility by one square."
        assert self._check(text) == []

    def test_our_mobility_narrows_clears_omission(self):
        text = "After the advance our mobility narrows from 8 to 7."
        assert self._check(text) == []

    def test_our_mobility_shrinks_clears_omission(self):
        text = "Our mobility shrinks to 7 after the move."
        assert self._check(text) == []

    def test_our_mobility_contracts_clears_omission(self):
        text = "Our mobility contracts from 8 to 7 after the advance."
        assert self._check(text) == []

    def test_our_mobility_falls_from_clears_omission(self):
        text = "Our mobility falls from 8 to 7 after the move."
        assert self._check(text) == []

    def test_our_mobility_drops_from_clears_omission(self):
        text = "Our mobility drops from 8 to 7 after the move."
        assert self._check(text) == []

    def test_silent_omission_still_flagged(self):
        # No acknowledgement at all — must still produce the contradiction.
        text = "The move advances safely and improves piece placement."
        out = self._check(text)
        assert len(out) == 1
        assert out[0].claim_type == "mobility_decrease_omission"

    def test_no_seed_means_no_check(self):
        # When no decrease seed is present, the check must be a no-op.
        text = "The move advances safely."
        assert _check_mobility_decrease_omission(text.lower(), []) == []


class TestD1RankerRuntimeAcceptsSynonyms:
    """The runtime mirror in ranker_agent._validate_reasoning_against_seeds
    must accept the same extended phrasings (E.1 parity)."""

    @staticmethod
    def _warnings(reasoning: str) -> list[str]:
        return _check_reasoning_truthfulness(
            reasoning, _CHOSEN_MOVE["facts"], _SEEDS_WITH_DECREASE
        )

    @staticmethod
    def _has_omission(warnings: list[str]) -> bool:
        return any("negative_fact_omission" in w for w in warnings)

    def test_reducing_our_mobility_clears(self):
        text = "The move advances safely while reducing our mobility by one square."
        assert not self._has_omission(self._warnings(text))

    def test_narrows_clears(self):
        text = "After the advance our mobility narrows from 8 to 7."
        assert not self._has_omission(self._warnings(text))

    def test_shrinks_clears(self):
        text = "Our mobility shrinks to 7 after the move."
        assert not self._has_omission(self._warnings(text))

    def test_contracts_clears(self):
        text = "Our mobility contracts from 8 to 7 after the advance."
        assert not self._has_omission(self._warnings(text))

    def test_silent_omission_still_flagged(self):
        text = "The move advances safely and improves piece placement."
        assert self._has_omission(self._warnings(text))


class TestD1ParityVerifierAndRuntime:
    """For each phrasing, the unified verifier and the runtime mirror must
    agree on whether the omission is present (E.1 parity)."""

    PHRASINGS_CLEARING = [
        "The move advances safely while reducing our mobility by one square.",
        "After the advance our mobility narrows from 8 to 7.",
        "Our mobility shrinks to 7 after the move.",
        "Our mobility contracts from 8 to 7 after the advance.",
        "Our mobility falls from 8 to 7 after the move.",
        "Our mobility drops from 8 to 7 after the move.",
        "This reduces our mobility but improves placement.",
        "Our mobility decreases by 1 after the advance.",
    ]

    PHRASINGS_OMITTING = [
        "The move advances safely and improves piece placement.",
        "The piece moves toward the center and remains supported.",
    ]

    def test_clearing_phrasings_parity(self):
        for text in self.PHRASINGS_CLEARING:
            v_flag = bool(_check_mobility_decrease_omission(
                text.lower(), _SEEDS_WITH_DECREASE
            ))
            r_flag = any(
                "negative_fact_omission" in w
                for w in _check_reasoning_truthfulness(
                    text, _CHOSEN_MOVE["facts"], _SEEDS_WITH_DECREASE
                )
            )
            assert v_flag == r_flag == False, (
                f"parity broken on clearing phrase: {text!r} "
                f"(verifier={v_flag}, runtime={r_flag})"
            )

    def test_omitting_phrasings_parity(self):
        for text in self.PHRASINGS_OMITTING:
            v_flag = bool(_check_mobility_decrease_omission(
                text.lower(), _SEEDS_WITH_DECREASE
            ))
            r_flag = any(
                "negative_fact_omission" in w
                for w in _check_reasoning_truthfulness(
                    text, _CHOSEN_MOVE["facts"], _SEEDS_WITH_DECREASE
                )
            )
            assert v_flag == r_flag == True, (
                f"parity broken on omitting phrase: {text!r} "
                f"(verifier={v_flag}, runtime={r_flag})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# D2 — Repair prompts must instruct preservation of grounded facts
# ═══════════════════════════════════════════════════════════════════════════

class TestD2RefinementPromptPreservesGroundedFacts:
    """_build_refinement_prompt must include a preservation clause."""

    def _prompt(self) -> str:
        return _build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION])

    def test_preserve_keyword_present(self):
        lower = self._prompt().lower()
        assert "preserve" in lower, "must instruct preservation of grounded facts"

    def test_references_key_facts_or_seeds(self):
        lower = self._prompt().lower()
        assert "key facts" in lower or "reasoning seeds" in lower or "seeds" in lower

    def test_scopes_change_to_flagged_contradiction(self):
        lower = self._prompt().lower()
        assert "flagged" in lower or "specifically flagged" in lower


class TestD2TargetedPromptPreservesGroundedFacts:
    """_build_targeted_refinement_prompt must include a preservation clause."""

    def _prompt(self) -> str:
        bad = ["Some false sentence."]
        return _build_targeted_refinement_prompt(
            _CHOSEN_MOVE, bad, [_CONTRADICTION]
        )

    def test_preserve_keyword_present(self):
        lower = self._prompt().lower()
        assert "preserve" in lower

    def test_references_key_facts_or_seeds(self):
        lower = self._prompt().lower()
        assert "key facts" in lower or "reasoning seeds" in lower or "seeds" in lower

    def test_scopes_change_to_flagged_contradiction(self):
        lower = self._prompt().lower()
        assert "flagged" in lower or "specifically flagged" in lower
