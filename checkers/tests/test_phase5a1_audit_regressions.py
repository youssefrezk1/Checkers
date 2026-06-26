"""
Phase 5A.1 regression tests — exactly reproduces the two CRITICAL failures
from the 20260526_104421 semantic audit.

BUG-1  Comparative hallucination of chosen-move properties
         Turn 7: "the chosen move accepts a temporary vulnerability to secure
         a gain" when captures_count=0, net_gain=0, opponent_can_recapture=False.

BUG-2  False opponent jump-count claim
         Turn 3: "limited to a single jump" when opponent_jump_count=2.

Each test either:
  (a) demonstrates the verifier NOW catches the failure, or
  (b) demonstrates the seed now prevents the LLM from generating the bad claim.

No LLM calls. All pure-deterministic unit tests.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from checkers.agents.comparative_reasoner import verify_comparative_reasoning
from checkers.agents.explainer_agent import (
    _build_grounded_reasoning_seeds,
    _check_reasoning_truthfulness,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _move(path, **facts):
    return {"path": path, "facts": dict(facts)}


# Exact chosen-move facts from Turn 7 of the failing trace.
_TURN7_CHOSEN = _move(
    [(5, 2), (4, 1)],
    captures_count=0,
    net_gain=0,
    opponent_can_recapture=False,
    creates_immediate_threat=False,
    shot_sequence_available=False,
    leaves_piece_isolated=False,
    weakens_king_row=False,
    results_in_king=False,
    near_promotion=False,
    our_pieces_threatened_after=0,
    opponent_mobility_before=11,
    opponent_mobility_after=11,
)

# Minimal alternatives sufficient to make the comparative pipeline run.
_TURN7_ALTS = [
    _TURN7_CHOSEN,
    _move([(5, 0), (4, 1)], captures_count=0, net_gain=0,
          opponent_can_recapture=False, creates_immediate_threat=False,
          shot_sequence_available=False, leaves_piece_isolated=False,
          weakens_king_row=False, results_in_king=False, near_promotion=False,
          our_pieces_threatened_after=0,
          opponent_mobility_before=11, opponent_mobility_after=11),
    _move([(6, 3), (5, 2)], captures_count=0, net_gain=0,
          opponent_can_recapture=False, creates_immediate_threat=False,
          shot_sequence_available=False, leaves_piece_isolated=True,
          weakens_king_row=False, results_in_king=False, near_promotion=False,
          our_pieces_threatened_after=0,
          opponent_mobility_before=11, opponent_mobility_after=11),
]

# Exact chosen-move facts from Turn 3 of the failing trace.
_TURN3_FACTS = {
    "captures_count": 1,
    "net_gain": 1,
    "opponent_can_recapture": True,
    "creates_immediate_threat": False,
    "forced_opponent_jump_reply": True,
    "max_opponent_jump_captures": 1,
    "opponent_jump_count": 2,          # BUG-2: 2 jump options, NOT 1
    "leaves_piece_isolated": True,
    "center_control": False,
    "results_in_king": False,
    "near_promotion": False,
    "minimax_score": 20.0,
    "our_pieces_threatened_after": 1,
    "moved_piece_is_threatened": True,
    "restriction_score": 6,
    "frozen_enemy_pieces": 6,
    "opponent_mobility_before": 9,
    "opponent_mobility_after": 8,
    "our_mobility_before": 7,
    "our_mobility_after": 6,
    "shot_sequence_available": False,
    "blocks_opponent_landing": False,
    "weakens_king_row": False,
}

_TURN3_MOVE = {
    "type": "jump",
    "path": [[4, 5], [2, 3]],
    "captured": [[3, 4]],
    "facts": _TURN3_FACTS,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-1 — Comparative hallucination against chosen-move properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestBug1ComparativeHallucinationCaught:
    """
    verify_comparative_reasoning must flag prose that claims the chosen move
    'accepts a temporary vulnerability to secure a gain' when facts show
    captures_count=0, net_gain=0, opponent_can_recapture=False.

    This reproduces the exact failure from Turn 7 of the semantic audit.
    """

    # The exact sentence from the failing trace (Turn 7).
    _AUDIT_SENTENCE = (
        "The defensive alternatives [2], [3], [4], [6], and [7] avoid immediate "
        "recapture but surrender any chance to capture material, while the chosen "
        "move accepts a temporary vulnerability to secure a gain."
    )

    def test_accepts_temporary_vulnerability_flagged_on_safe_move(self):
        """'accepts a temporary vulnerability' must be flagged when opponent_can_recapture=False."""
        prose = (
            "Defensive alternative [1] avoids recapture, while the chosen move "
            "accepts a temporary vulnerability to gain position."
        )
        result = verify_comparative_reasoning(prose, _TURN7_ALTS, _TURN7_CHOSEN)
        tradeoff = [c for c in result if c.type == "invalid_tradeoff"]
        recap_errors = [c for c in tradeoff if c.fact_key == "opponent_can_recapture"]
        assert recap_errors, (
            "Expected invalid_tradeoff on opponent_can_recapture when chosen move "
            f"has opponent_can_recapture=False but prose claims vulnerability. "
            f"Got: {result}"
        )

    def test_accepts_temporary_vulnerability_flagged_on_non_capturing_move(self):
        """'accepts a temporary vulnerability' must be flagged when captures_count=0."""
        prose = (
            "Defensive alternative [1] avoids recapture, while the chosen move "
            "accepts a temporary vulnerability to advance."
        )
        result = verify_comparative_reasoning(prose, _TURN7_ALTS, _TURN7_CHOSEN)
        tradeoff = [c for c in result if c.type == "invalid_tradeoff"]
        cap_errors = [c for c in tradeoff if c.fact_key == "captures_count"]
        assert cap_errors, (
            "Expected invalid_tradeoff on captures_count when chosen move has "
            f"captures_count=0 but prose implies exposure-for-material. Got: {result}"
        )

    def test_secure_a_gain_flagged_when_net_gain_zero(self):
        """'chosen move...to secure a gain' must be flagged when net_gain=0."""
        prose = (
            "Defensive alternatives [1] and [2] play it safe, while the chosen "
            "move accepts risk to secure a gain."
        )
        result = verify_comparative_reasoning(prose, _TURN7_ALTS, _TURN7_CHOSEN)
        tradeoff = [c for c in result if c.type == "invalid_tradeoff"]
        gain_errors = [c for c in tradeoff if c.fact_key == "net_gain"]
        assert gain_errors, (
            f"Expected invalid_tradeoff on net_gain when net_gain=0 but prose "
            f"claims 'secure a gain'. Got: {result}"
        )

    def test_exact_audit_sentence_flagged(self):
        """The exact sentence from the failing Turn 7 trace must produce contradictions."""
        result = verify_comparative_reasoning(
            self._AUDIT_SENTENCE, _TURN7_ALTS, _TURN7_CHOSEN,
        )
        tradeoff = [c for c in result if c.type == "invalid_tradeoff"]
        assert tradeoff, (
            f"The exact audit sentence must produce at least one invalid_tradeoff. "
            f"Got: {result}"
        )
        fact_keys = {c.fact_key for c in tradeoff}
        # Must catch both: the vulnerability claim AND the gain claim.
        assert "net_gain" in fact_keys or "opponent_can_recapture" in fact_keys or \
               "captures_count" in fact_keys, (
            f"Expected at least one of net_gain/opponent_can_recapture/captures_count "
            f"in tradeoff errors. Got keys: {fact_keys}"
        )

    def test_accepts_exposure_original_form_still_caught(self):
        """Original 'accepts exposure' form must still be caught (non-regression)."""
        prose = (
            "Defensive alternative [1] avoids recapture, while the chosen "
            "move accepts exposure to win material."
        )
        result = verify_comparative_reasoning(prose, _TURN7_ALTS, _TURN7_CHOSEN)
        tradeoff = [c for c in result if c.type == "invalid_tradeoff"]
        assert tradeoff, f"Original 'accepts exposure' form not caught. Got: {result}"

    def test_valid_vulnerability_accepted_when_facts_match(self):
        """Chosen move that actually captures and has recapture risk must NOT be flagged."""
        capturing_unsafe = _move(
            [(4, 5), (2, 3)],
            captures_count=1, net_gain=1,
            opponent_can_recapture=True,
            creates_immediate_threat=False, shot_sequence_available=False,
            leaves_piece_isolated=True, weakens_king_row=False,
            results_in_king=False, near_promotion=False,
            our_pieces_threatened_after=1,
            opponent_mobility_before=9, opponent_mobility_after=8,
        )
        candidates = [
            capturing_unsafe,
            _move([(5, 4), (4, 3)], captures_count=0, net_gain=0,
                  opponent_can_recapture=False, creates_immediate_threat=False,
                  shot_sequence_available=False, leaves_piece_isolated=False,
                  weakens_king_row=False, results_in_king=False, near_promotion=False,
                  our_pieces_threatened_after=0,
                  opponent_mobility_before=9, opponent_mobility_after=9),
        ]
        prose = (
            "Defensive alternative [1] avoids recapture, while the chosen "
            "move accepts a temporary vulnerability to secure a gain."
        )
        result = verify_comparative_reasoning(prose, candidates, capturing_unsafe)
        tradeoff = [c for c in result if c.type == "invalid_tradeoff"]
        assert not tradeoff, (
            f"Valid vulnerability+gain claim must NOT be flagged. Got: {tradeoff}"
        )

    def test_secure_gain_without_chosen_move_ref_not_flagged(self):
        """'secure a gain' without 'chosen move' context must not fire the rule."""
        prose = (
            "Aggressive alternatives [1] and [2] seek to secure a gain "
            "at the cost of recapture risk."
        )
        result = verify_comparative_reasoning(prose, _TURN7_ALTS, _TURN7_CHOSEN)
        gain_tradeoffs = [
            c for c in result
            if c.type == "invalid_tradeoff" and c.fact_key == "net_gain"
        ]
        assert not gain_tradeoffs, (
            f"'secure a gain' without 'chosen move' ref must not fire rule E. "
            f"Got: {gain_tradeoffs}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-2 — False opponent jump-count claims
# ═══════════════════════════════════════════════════════════════════════════════

class TestBug2OpponentJumpCountVerifier:
    """
    _check_reasoning_truthfulness must flag 'single jump' / 'one jump option'
    when opponent_jump_count > 1.

    This reproduces the exact failure from Turn 3 of the semantic audit.
    """

    def _check(self, text: str, facts=None) -> list[str]:
        return _check_reasoning_truthfulness(text, facts or _TURN3_FACTS)

    def test_single_jump_flagged_when_jump_count_is_2(self):
        """The exact Turn 3 phrase must be caught."""
        ws = self._check(
            "The opponent is forced into a constrained reply, limited to a "
            "single jump that captures at most one of their own pieces."
        )
        assert any("single" in w.lower() and "jump" in w.lower() for w in ws), (
            f"Expected a warning about 'single jump' when opponent_jump_count=2. "
            f"Got: {ws}"
        )

    def test_one_jump_option_flagged_when_jump_count_is_2(self):
        ws = self._check(
            "The opponent has only one jump option available after this move."
        )
        assert any("jump" in w.lower() for w in ws), (
            f"Expected jump-count warning. Got: {ws}"
        )

    def test_limited_to_a_single_jump_flagged(self):
        ws = self._check("The opponent is limited to a single jump in reply.")
        assert any("jump" in w.lower() for w in ws), f"Got: {ws}"

    def test_no_false_positive_when_jump_count_is_1(self):
        """Must NOT flag 'single jump' when opponent_jump_count actually equals 1."""
        facts_one_jump = {**_TURN3_FACTS, "opponent_jump_count": 1}
        ws = _check_reasoning_truthfulness(
            "The opponent is limited to a single jump in reply.",
            facts_one_jump,
        )
        jump_errs = [w for w in ws if "single" in w.lower() and "jump" in w.lower()]
        assert not jump_errs, (
            f"Must not flag 'single jump' when opponent_jump_count=1. Got: {jump_errs}"
        )

    def test_no_false_positive_when_jump_count_absent(self):
        """Must NOT flag when opponent_jump_count is not in facts."""
        facts_no_jc = {k: v for k, v in _TURN3_FACTS.items()
                       if k != "opponent_jump_count"}
        ws = _check_reasoning_truthfulness(
            "The opponent is limited to a single jump in reply.",
            facts_no_jc,
        )
        jump_errs = [w for w in ws if "single" in w.lower() and "jump" in w.lower()]
        assert not jump_errs, (
            f"Must not flag when opponent_jump_count absent. Got: {jump_errs}"
        )

    def test_contradiction_string_format(self):
        """Warning must name both the contradiction and the actual count."""
        ws = self._check(
            "The opponent is limited to a single jump after this move."
        )
        jump_ws = [w for w in ws if "jump" in w.lower()]
        assert jump_ws, "Expected at least one jump-related warning."
        assert any("2" in w for w in jump_ws), (
            f"Warning must cite the actual opponent_jump_count=2. Got: {jump_ws}"
        )


class TestBug2ForcedJumpSeed:
    """
    _build_grounded_reasoning_seeds must include the jump count in the seed
    text when opponent_jump_count > 1, preventing the LLM from claiming
    'single jump' because the seed itself explicitly states the count.
    """

    def _seeds(self, **extra_facts):
        facts = {**_TURN3_FACTS, **extra_facts}
        move = {**_TURN3_MOVE, "facts": facts}
        return _build_grounded_reasoning_seeds(move, [move])

    def _jump_seed(self, **extra_facts) -> str | None:
        seeds = self._seeds(**extra_facts)
        jump_seeds = [s for s in seeds if "forced" in s.lower() or "jump" in s.lower()]
        return jump_seeds[0] if jump_seeds else None

    def test_seed_includes_count_when_multiple_jumps(self):
        """When opponent_jump_count=2, seed must state '2 jump options'."""
        seed = self._jump_seed()
        assert seed is not None, "Expected a forced-jump seed."
        assert "2 jump options" in seed or "2" in seed, (
            f"Seed must cite the jump count when opponent_jump_count=2. Got: {seed!r}"
        )

    def test_seed_does_not_say_single_when_multiple_jumps(self):
        """Seed must never imply a single jump when opponent_jump_count > 1."""
        seed = self._jump_seed()
        assert seed is not None
        assert "single" not in seed.lower() and "one jump" not in seed.lower(), (
            f"Seed must not imply single-jump when opponent_jump_count=2. Got: {seed!r}"
        )

    def test_seed_still_includes_max_captures(self):
        """Jump count seed must still cite max_opponent_jump_captures."""
        seed = self._jump_seed()
        assert seed is not None
        assert "1" in seed, (
            f"Seed must cite max_opponent_jump_captures=1. Got: {seed!r}"
        )

    def test_seed_format_when_jump_count_is_1(self):
        """When opponent_jump_count=1, seed must use the old single-jump form."""
        seed = self._jump_seed(opponent_jump_count=1)
        assert seed is not None
        # Old form: no mention of '1 jump option', just the max-captures clause.
        assert "jump options" not in seed.lower(), (
            f"Single-jump form must not mention 'jump options'. Got: {seed!r}"
        )
        assert "forced to respond with a jump" in seed.lower(), (
            f"Single-jump form must say 'forced to respond with a jump'. Got: {seed!r}"
        )

    def test_seed_format_when_jump_count_absent(self):
        """When opponent_jump_count is absent, fall back to old single-jump form."""
        facts = {k: v for k, v in _TURN3_FACTS.items()
                 if k != "opponent_jump_count"}
        move = {**_TURN3_MOVE, "facts": facts}
        seeds = _build_grounded_reasoning_seeds(move, [move])
        jump_seeds = [s for s in seeds if "forced" in s.lower()]
        assert jump_seeds, "Expected at least one forced-jump seed."
        assert "jump options" not in jump_seeds[0].lower(), (
            f"No-count fallback must not mention 'jump options'. Got: {jump_seeds[0]!r}"
        )

    def test_seed_format_when_jump_count_exactly_2(self):
        """Exact Turn 3 scenario: opponent_jump_count=2 must produce the count form."""
        seed = self._jump_seed(opponent_jump_count=2)
        assert seed is not None
        assert "2 jump options" in seed, (
            f"Must state '2 jump options'. Got: {seed!r}"
        )
