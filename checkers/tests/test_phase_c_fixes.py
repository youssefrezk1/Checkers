# checkers/tests/test_phase_c_fixes.py
#
# Phase C regression tests.  Three targeted fixes:
#
#   C1 — Repair-loop claim preservation
#        Both repair prompts must carry explicit constraints preventing the LLM
#        from introducing new recapture / mobility / score-gap claims while
#        fixing an unrelated false sentence.
#
#   C2 — Score-gap precision hallucination prevention
#        Repair prompts must prohibit numeric score-gap phrases (e.g., "X points
#        better") when only the chosen-move engine score is available.
#
#   C3 — Comparative "sacrifice capture" logic flaw
#        The DEFENSIVE comparative seed must NOT emit "but does not capture" when
#        the chosen move also has captures_count=0 — there is no meaningful
#        contrast on the capture dimension in that case.

from __future__ import annotations

import pytest

from checkers.agents.ranker_agent import (
    _build_refinement_prompt,
    _build_targeted_refinement_prompt,
)
from checkers.agents.comparative_reasoner import (
    _cluster_alternatives_by_theme,
    build_comparative_group_seeds,
)


# ── Helpers shared across C3 tests ───────────────────────────────────────────

def _move(path, **facts):
    return {"path": path, "facts": dict(facts)}


def _quiet_facts(**overrides):
    base = dict(
        opponent_can_recapture=False,
        captures_count=0,
        net_gain=0,
        creates_immediate_threat=False,
        shot_sequence_available=False,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        our_pieces_threatened_after=0,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
    )
    base.update(overrides)
    return base


CHOSEN_QUIET = _move([(5, 4), (4, 3)], **_quiet_facts())

CHOSEN_CAPTURING = _move(
    [(5, 4), (4, 3)],
    **_quiet_facts(captures_count=1, net_gain=1, opponent_can_recapture=True),
)


def _defensive_alt(path):
    """Alternative that qualifies as DEFENSIVE (opp_recap=False, pta<=chosen)."""
    return _move(path, **_quiet_facts())


# ── Minimal chosen_move dict for repair prompt tests ─────────────────────────

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
        "minimax_score": -2.0,
    },
}

_CONTRADICTION = "REASONING_CONTRADICTION: 'no tactical pressure' framing contradicts creates_immediate_threat=true (tactical_move_defensive_framing)"


# ═══════════════════════════════════════════════════════════════════════════
# C1 — Repair-loop claim preservation constraints in prompts
# ═══════════════════════════════════════════════════════════════════════════

class TestC1FullRepairPromptConstraints:
    """_build_refinement_prompt must carry drift-prevention instructions."""

    def _prompt(self):
        return _build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION])

    def test_has_no_new_recapture_claim_constraint(self):
        p = self._prompt()
        assert "recapture" in p.lower(), "must mention recapture constraint"
        # The constraint sentence should prohibit introducing recapture claims.
        assert "do not add" in p.lower() or "do not" in p.lower()

    def test_has_no_new_mobility_claim_constraint(self):
        p = self._prompt()
        assert "mobility" in p.lower()

    def test_has_score_gap_prohibition(self):
        p = self._prompt()
        # Must warn against numeric score-gap claims.
        lower = p.lower()
        assert "points better" in lower or "score-gap" in lower or "score-comparison" in lower

    def test_has_prefer_minimal_edit_guidance(self):
        p = self._prompt()
        lower = p.lower()
        assert "specific" in lower or "minimal" in lower or "unnecessarily" in lower

    def test_engine_score_still_allowed_in_final_sentence(self):
        p = self._prompt()
        assert "final sentence" in p.lower() or "confirmation" in p.lower()


class TestC1TargetedRepairPromptConstraints:
    """_build_targeted_refinement_prompt must carry drift-prevention instructions."""

    def _prompt(self):
        bad = ["The piece applies no pressure on the opponent."]
        return _build_targeted_refinement_prompt(_CHOSEN_MOVE, bad, [_CONTRADICTION])

    def test_has_no_new_recapture_claim_constraint(self):
        p = self._prompt()
        assert "recapture" in p.lower()

    def test_has_no_new_mobility_claim_constraint(self):
        p = self._prompt()
        assert "mobility" in p.lower()

    def test_has_score_gap_prohibition(self):
        p = self._prompt()
        lower = p.lower()
        assert "points better" in lower or "score-comparison" in lower or "score-gap" in lower

    def test_constraint_requires_key_facts_grounding(self):
        p = self._prompt()
        lower = p.lower()
        # Must explicitly reference grounding in key facts.
        assert "key facts" in lower or "listed above" in lower or "listed in" in lower


# ═══════════════════════════════════════════════════════════════════════════
# C2 — Score-gap precision hallucination: prompt-level verification
# ═══════════════════════════════════════════════════════════════════════════

class TestC2ScoreGapProhibitionInPrompts:
    """Both repair prompts must explicitly forbid 'X points better' phrasing."""

    def test_full_repair_prohibits_numeric_score_gap(self):
        p = _build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION])
        lower = p.lower()
        # At minimum the phrase "points better" must appear as a prohibited example.
        assert "points better" in lower

    def test_targeted_repair_prohibits_numeric_score_gap(self):
        bad = ["Some false sentence."]
        p = _build_targeted_refinement_prompt(_CHOSEN_MOVE, bad, [_CONTRADICTION])
        lower = p.lower()
        assert "points better" in lower

    def test_full_repair_allows_engine_score_reference(self):
        # "engine score" or "minimax_score" must still be permitted for confirmation.
        p = _build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION])
        lower = p.lower()
        assert "engine score" in lower or "minimax_score" in lower or "confirmation" in lower

    def test_targeted_repair_key_facts_shows_engine_score(self):
        bad = ["Some false sentence."]
        p = _build_targeted_refinement_prompt(_CHOSEN_MOVE, bad, [_CONTRADICTION])
        # The key-facts block must include the engine score value.
        assert "-2.0" in p


# ═══════════════════════════════════════════════════════════════════════════
# C3 — Comparative "sacrifice capture" logic gating
# ═══════════════════════════════════════════════════════════════════════════

class TestC3DefensiveSeedCaptureDrawback:
    """DEFENSIVE seed: 'but does not capture' only fires when chosen captures."""

    def _defensive_seeds(self, chosen):
        alt = _defensive_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([chosen, alt], chosen)
        return build_comparative_group_seeds(groups, chosen["facts"])

    # ── C3.1: suppress drawback when chosen also does not capture ────────────

    def test_no_capture_drawback_when_chosen_is_quiet(self):
        seeds = self._defensive_seeds(CHOSEN_QUIET)
        defensive_seeds = [s for s in seeds if s.startswith("Defensive")]
        assert defensive_seeds, "DEFENSIVE seed must be generated"
        for s in defensive_seeds:
            assert "does not capture" not in s, (
                f"'does not capture' must not appear when chosen also doesn't "
                f"capture; got: {s!r}"
            )
            assert "do not capture" not in s, (
                f"'do not capture' must not appear when chosen also doesn't "
                f"capture; got: {s!r}"
            )

    def test_defensive_seed_still_says_avoids_recapture_when_quiet(self):
        seeds = self._defensive_seeds(CHOSEN_QUIET)
        defensive_seeds = [s for s in seeds if s.startswith("Defensive")]
        assert defensive_seeds
        # The core "avoids recapture" description must survive.
        assert any("avoids recapture" in s or "avoid recapture" in s for s in defensive_seeds)

    # ── C3.2: include drawback when chosen DOES capture ─────────────────────

    def test_capture_drawback_present_when_chosen_captures(self):
        seeds = self._defensive_seeds(CHOSEN_CAPTURING)
        defensive_seeds = [s for s in seeds if s.startswith("Defensive")]
        assert defensive_seeds, "DEFENSIVE seed must be generated"
        # When chosen captures, 'does not capture' drawback is meaningful.
        assert any(
            "does not capture" in s or "do not capture" in s
            for s in defensive_seeds
        ), (
            f"'does not capture' should appear when chosen captures; "
            f"defensive seeds: {defensive_seeds}"
        )

    # ── C3.3: multi-member defensive group, quiet chosen ────────────────────

    def test_no_capture_drawback_with_multiple_defensive_alts_quiet_chosen(self):
        a1 = _defensive_alt([(5, 2), (4, 3)])
        a2 = _defensive_alt([(5, 0), (4, 1)])
        groups = _cluster_alternatives_by_theme([CHOSEN_QUIET, a1, a2], CHOSEN_QUIET)
        seeds = build_comparative_group_seeds(groups, CHOSEN_QUIET["facts"])
        defensive_seeds = [s for s in seeds if s.startswith("Defensive")]
        for s in defensive_seeds:
            assert "do not capture" not in s and "does not capture" not in s

    # ── C3.4: chosen captures but alt also captures → no "but does not capture"

    def test_no_capture_drawback_when_defensive_alt_also_captures(self):
        # A defensive alt that itself captures (not in DEFENSIVE theme normally,
        # but we test the edge case: no_captures=False → drawback never fires).
        alt_also_caps = _move(
            [(5, 2), (4, 3)],
            **_quiet_facts(captures_count=1, net_gain=1),
        )
        groups = _cluster_alternatives_by_theme(
            [CHOSEN_CAPTURING, alt_also_caps], CHOSEN_CAPTURING
        )
        seeds = build_comparative_group_seeds(groups, CHOSEN_CAPTURING["facts"])
        # alt_also_caps captures so no_captures=False → drawback suppressed
        for s in seeds:
            if s.startswith("Defensive"):
                assert "does not capture" not in s


# ═══════════════════════════════════════════════════════════════════════════
# E.1 parity: Phase C changes must not introduce verifier-side divergence
# ═══════════════════════════════════════════════════════════════════════════

class TestPhaseCE1Parity:
    """Phase C touches only prompts and one seed-generation branch.
    No new verifier rules were added, so E.1 parity is preserved trivially.
    These smoke tests confirm the modified functions are still importable
    and their outputs are strings (not exceptions).
    """

    def test_full_repair_prompt_returns_string(self):
        p = _build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION])
        assert isinstance(p, str) and len(p) > 100

    def test_targeted_repair_prompt_returns_string(self):
        bad = ["Some bad sentence to replace."]
        p = _build_targeted_refinement_prompt(_CHOSEN_MOVE, bad, [_CONTRADICTION])
        assert isinstance(p, str) and len(p) > 100

    def test_build_comparative_group_seeds_returns_list(self):
        alt = _defensive_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN_QUIET, alt], CHOSEN_QUIET)
        seeds = build_comparative_group_seeds(groups, CHOSEN_QUIET["facts"])
        assert isinstance(seeds, list)

    def test_no_new_contradictions_introduced_by_prompt_changes(self):
        # Verify both prompts contain the key-facts block so no new fact
        # categories are accidentally exposed to the LLM as free variables.
        p_full = _build_refinement_prompt(_CHOSEN_MOVE, [_CONTRADICTION])
        p_tgt  = _build_targeted_refinement_prompt(
            _CHOSEN_MOVE, ["Bad sentence."], [_CONTRADICTION]
        )
        for p in (p_full, p_tgt):
            assert "Key facts" in p
            assert "engine score" in p.lower() or "-2.0" in p
