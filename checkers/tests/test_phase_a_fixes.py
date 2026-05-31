# checkers/tests/test_phase_a_fixes.py
#
# Phase A regression tests — pure-deterministic, no LLM calls.
#
# Covers the four targeted fixes from Phase A:
#   Fix 1 — 2-candidate binary comparative fast-path
#   Fix 2 — Must-mention check for our-mobility decrease
#   Fix 3 — any_piece_isolated contradicts "no vulnerabilities"
#   Fix 4 — "narrowing the gap" mobility-direction check
#
# All tests use only deterministic functions; no network calls.

from __future__ import annotations

from typing import Optional

import pytest

from checkers.agents.ranker_agent import (
    _check_reasoning_truthfulness,
    _generate_binary_comparative,
)
from checkers.evaluation.unified_verifier import (
    contradiction_strings,
    verify_all,
)
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


# ── Shared helpers ────────────────────────────────────────────────────────────

def _rt(text: str, facts: dict, seeds: Optional[list] = None) -> list[str]:
    return _check_reasoning_truthfulness(text, facts, seeds=seeds or [])


def _ev(text: str, facts: dict, seeds: Optional[list] = None) -> list[str]:
    return contradiction_strings(text, reasoning_seeds=seeds or [], facts=facts)


def _has(warnings: list[str], fragment: str) -> bool:
    return any(fragment in w for w in warnings)


def _clean(warnings: list[str]) -> bool:
    return not any("REASONING_CONTRADICTION" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — Binary comparative fast-path
# ─────────────────────────────────────────────────────────────────────────────

_CHOSEN_MULTI = {
    "type": "jump",
    "path": [[6, 5], [4, 3], [2, 5]],
    "facts": {
        "captures_count": 2,
        "minimax_score": 28.0,
        "moved_piece_is_threatened": True,
        "opponent_can_recapture": True,
        "leaves_piece_isolated": True,
    },
}

_ALT_SINGLE = {
    "type": "jump",
    "path": [[6, 5], [4, 3]],
    "facts": {
        "captures_count": 1,
        "minimax_score": -28.0,
        "moved_piece_is_threatened": True,
        "opponent_can_recapture": True,
        "leaves_piece_isolated": False,
    },
}

_CANDIDATES_2 = [_CHOSEN_MULTI, _ALT_SINGLE]


class TestBinaryComparativeFastPath:
    """Fix 1: _generate_binary_comparative for the 2-candidate case."""

    def test_returns_string_for_two_candidates(self):
        result = _generate_binary_comparative(
            _CHOSEN_MULTI, _CANDIDATES_2, _CHOSEN_MULTI["facts"]
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_score_margin_in_output(self):
        result = _generate_binary_comparative(
            _CHOSEN_MULTI, _CANDIDATES_2, _CHOSEN_MULTI["facts"]
        )
        assert result is not None
        # Should mention the alt score, chosen score, and the gap
        assert "-28.0" in result
        assert "28.0" in result

    def test_captures_difference_in_output(self):
        result = _generate_binary_comparative(
            _CHOSEN_MULTI, _CANDIDATES_2, _CHOSEN_MULTI["facts"]
        )
        assert result is not None
        # Chosen captures 2 vs alt captures 1 — should mention this
        assert "2" in result
        assert "1" in result

    def test_references_alt_as_index_one(self):
        result = _generate_binary_comparative(
            _CHOSEN_MULTI, _CANDIDATES_2, _CHOSEN_MULTI["facts"]
        )
        assert result is not None
        assert "[1]" in result

    def test_returns_none_when_no_alternative_found(self):
        # Only one candidate — the chosen move itself — so no alt.
        result = _generate_binary_comparative(
            _CHOSEN_MULTI, [_CHOSEN_MULTI], _CHOSEN_MULTI["facts"]
        )
        assert result is None

    def test_returns_none_for_empty_candidates(self):
        result = _generate_binary_comparative(_CHOSEN_MULTI, [], _CHOSEN_MULTI["facts"])
        assert result is None

    def test_safety_edge_sentence_when_no_capture_diff(self):
        # When captures are equal but recapture differs.
        _safe_chosen = {
            "path": [[5, 2], [4, 1]],
            "facts": {
                "captures_count": 0,
                "minimax_score": 10.0,
                "moved_piece_is_threatened": False,
                "opponent_can_recapture": False,
                "leaves_piece_isolated": False,
            },
        }
        _risky_alt = {
            "path": [[5, 4], [4, 3]],
            "facts": {
                "captures_count": 0,
                "minimax_score": 3.0,
                "moved_piece_is_threatened": False,
                "opponent_can_recapture": True,
                "leaves_piece_isolated": False,
            },
        }
        result = _generate_binary_comparative(
            _safe_chosen, [_safe_chosen, _risky_alt], _safe_chosen["facts"]
        )
        assert result is not None
        # Should mention recapture safety edge
        assert "recapture" in result.lower()

    def test_isolation_sentence_when_no_other_diff(self):
        # When captures and recapture are equal but isolation differs.
        _conn_chosen = {
            "path": [[5, 2], [4, 1]],
            "facts": {
                "captures_count": 0,
                "minimax_score": 10.0,
                "moved_piece_is_threatened": False,
                "opponent_can_recapture": False,
                "leaves_piece_isolated": False,
            },
        }
        _iso_alt = {
            "path": [[5, 4], [4, 3]],
            "facts": {
                "captures_count": 0,
                "minimax_score": 3.0,
                "moved_piece_is_threatened": False,
                "opponent_can_recapture": False,
                "leaves_piece_isolated": True,
            },
        }
        result = _generate_binary_comparative(
            _conn_chosen, [_conn_chosen, _iso_alt], _conn_chosen["facts"]
        )
        assert result is not None
        assert "support" in result.lower() or "connect" in result.lower()

    def test_no_llm_call_for_two_candidates(self):
        # Verifies the binary path is fully deterministic — calling twice
        # returns identical output.
        r1 = _generate_binary_comparative(
            _CHOSEN_MULTI, _CANDIDATES_2, _CHOSEN_MULTI["facts"]
        )
        r2 = _generate_binary_comparative(
            _CHOSEN_MULTI, _CANDIDATES_2, _CHOSEN_MULTI["facts"]
        )
        assert r1 == r2

    def test_path_notation_included(self):
        result = _generate_binary_comparative(
            _CHOSEN_MULTI, _CANDIDATES_2, _CHOSEN_MULTI["facts"]
        )
        assert result is not None
        # Alternative path [6,5]→[4,3] should appear in notation
        assert "6,5→4,3" in result


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — Our-mobility decrease must-mention
# ─────────────────────────────────────────────────────────────────────────────

_MOB_DEC_FACTS = {
    "our_mobility_before": 11,
    "our_mobility_after": 9,
    "opponent_mobility_before": 12,
    "opponent_mobility_after": 12,
}

_MOB_DEC_SEEDS = [
    "our mobility changes from 11 to 9 — decreases our mobility by 2",
    "opponent mobility remains at 12 — no change in opponent mobility",
]

_MOB_DEC_GOOD = (
    "Advancing the piece from (7, 6) to (6, 7) restricts one opponent piece, "
    "though this decreases our mobility from 11 to 9."
)

_MOB_DEC_BAD = (
    "Advancing the piece from (7, 6) to (6, 7) restricts one opponent piece "
    "while maintaining the opponent's current mobility at 12."
)


class TestMobilityDecreaseOmission:
    """Fix 2: must mention our-mobility decrease when seeded."""

    def test_runtime_fires_when_decrease_omitted(self):
        warnings = _rt(_MOB_DEC_BAD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        assert _has(warnings, "our-mobility decrease seeded but omitted")

    def test_runtime_clean_when_decrease_mentioned(self):
        warnings = _rt(_MOB_DEC_GOOD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        assert not _has(warnings, "our-mobility decrease seeded but omitted")

    def test_evaluator_fires_when_decrease_omitted(self):
        warnings = _ev(_MOB_DEC_BAD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        assert _has(warnings, "our-mobility decrease seeded but omitted")

    def test_evaluator_clean_when_decrease_mentioned(self):
        warnings = _ev(_MOB_DEC_GOOD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        assert not _has(warnings, "our-mobility decrease seeded but omitted")

    def test_no_fire_when_no_decrease_seed(self):
        # No decrease seed — must not fire even if text omits mobility.
        neutral_seeds = ["opponent mobility remains at 12 — no change in opponent mobility"]
        warnings = _rt(_MOB_DEC_BAD, _MOB_DEC_FACTS, seeds=neutral_seeds)
        assert not _has(warnings, "our-mobility decrease seeded but omitted")

    def test_no_fire_when_mobility_increases(self):
        # Increase seed: "increases our mobility by 1" — different pattern.
        inc_seeds = ["our mobility changes from 7 to 8 — increases our mobility by 1"]
        text = "Advancing the piece without capturing improves piece placement."
        facts = {"our_mobility_before": 7, "our_mobility_after": 8}
        warnings = _rt(text, facts, seeds=inc_seeds)
        assert not _has(warnings, "our-mobility decrease seeded but omitted")

    def test_runtime_evaluator_agree_on_fire(self):
        rt = _rt(_MOB_DEC_BAD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        ev = _ev(_MOB_DEC_BAD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        rt_has = _has(rt, "our-mobility decrease seeded but omitted")
        ev_has = _has(ev, "our-mobility decrease seeded but omitted")
        assert rt_has == ev_has, f"E.1 violation: runtime={rt_has}, evaluator={ev_has}"

    def test_runtime_evaluator_agree_on_clean(self):
        rt = _rt(_MOB_DEC_GOOD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        ev = _ev(_MOB_DEC_GOOD, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
        rt_has = _has(rt, "our-mobility decrease seeded but omitted")
        ev_has = _has(ev, "our-mobility decrease seeded but omitted")
        assert rt_has == ev_has, f"E.1 violation: runtime={rt_has}, evaluator={ev_has}"

    def test_various_mention_words_accepted(self):
        # Each of these surface forms should satisfy the must-mention check.
        mention_forms = [
            "our mobility decreases from 11 to 9",
            "our mobility drops from 11 to 9",
            "our mobility falls after this move",
            "reduces our mobility by 2",
            "losing mobility as a result",
        ]
        for form in mention_forms:
            text = f"Advancing the piece. {form}. Opponent unchanged at 12."
            warnings = _rt(text, _MOB_DEC_FACTS, seeds=_MOB_DEC_SEEDS)
            assert not _has(warnings, "our-mobility decrease seeded but omitted"), (
                f"False positive for mention form: '{form}'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3 — any_piece_isolated contradicts "no vulnerabilities"
# ─────────────────────────────────────────────────────────────────────────────

_ISO_FACTS_TRUE = {
    "any_piece_isolated": True,
    "leaves_piece_isolated": False,  # moved piece itself is NOT isolated
}

_ISO_FACTS_FALSE = {
    "any_piece_isolated": False,
    "leaves_piece_isolated": False,
}

_ISO_BAD_TEXT = (
    "Advancing from (6, 1) to (5, 2) avoids exposing the piece to immediate "
    "recapture, ensuring no tactical vulnerabilities are created."
)

_ISO_GOOD_TEXT = (
    "Advancing from (6, 1) to (5, 2) avoids exposing the piece to immediate "
    "recapture. Note that another allied piece remains isolated after this move."
)


class TestAnyPieceIsolatedVulnerability:
    """Fix 3: any_piece_isolated=true contradicts 'no tactical vulnerabilities'."""

    def test_runtime_fires_when_isolated_and_no_vuln_claimed(self):
        warnings = _rt(_ISO_BAD_TEXT, _ISO_FACTS_TRUE)
        assert _has(warnings, "any_piece_isolated=true")

    def test_runtime_clean_when_not_isolated(self):
        warnings = _rt(_ISO_BAD_TEXT, _ISO_FACTS_FALSE)
        assert not _has(warnings, "any_piece_isolated=true")

    def test_runtime_clean_when_isolated_but_no_vuln_claim(self):
        warnings = _rt(_ISO_GOOD_TEXT, _ISO_FACTS_TRUE)
        assert not _has(warnings, "any_piece_isolated=true")

    def test_evaluator_fires_when_isolated_and_no_vuln_claimed(self):
        warnings = _ev(_ISO_BAD_TEXT, _ISO_FACTS_TRUE)
        assert _has(warnings, "any_piece_isolated=true")

    def test_evaluator_clean_when_not_isolated(self):
        warnings = _ev(_ISO_BAD_TEXT, _ISO_FACTS_FALSE)
        assert not _has(warnings, "any_piece_isolated=true")

    def test_all_trigger_phrases_caught(self):
        phrases = [
            "no tactical vulnerabilities",
            "ensuring no tactical",
            "no vulnerabilities are created",
            "no tactical vulnerabilities are created",
        ]
        for phrase in phrases:
            text = f"The piece moves forward. {phrase}. Safe outcome."
            rt = _rt(text, _ISO_FACTS_TRUE)
            ev = _ev(text, _ISO_FACTS_TRUE)
            assert _has(rt, "any_piece_isolated=true"), (
                f"Runtime missed trigger phrase: '{phrase}'"
            )
            assert _has(ev, "any_piece_isolated=true"), (
                f"Evaluator missed trigger phrase: '{phrase}'"
            )

    def test_runtime_evaluator_agree_on_fire(self):
        rt = _rt(_ISO_BAD_TEXT, _ISO_FACTS_TRUE)
        ev = _ev(_ISO_BAD_TEXT, _ISO_FACTS_TRUE)
        rt_has = _has(rt, "any_piece_isolated=true")
        ev_has = _has(ev, "any_piece_isolated=true")
        assert rt_has == ev_has, f"E.1 violation: runtime={rt_has}, evaluator={ev_has}"

    def test_runtime_evaluator_agree_on_clean(self):
        rt = _rt(_ISO_GOOD_TEXT, _ISO_FACTS_TRUE)
        ev = _ev(_ISO_GOOD_TEXT, _ISO_FACTS_TRUE)
        rt_has = _has(rt, "any_piece_isolated=true")
        ev_has = _has(ev, "any_piece_isolated=true")
        assert rt_has == ev_has, f"E.1 violation: runtime={rt_has}, evaluator={ev_has}"

    def test_leaves_piece_isolated_true_does_not_double_fire(self):
        # When leaves_piece_isolated=True also — only Fix3 fires; no double-count.
        facts_both = {"any_piece_isolated": True, "leaves_piece_isolated": True}
        rt = _rt(_ISO_BAD_TEXT, facts_both)
        fire_count = sum(1 for w in rt if "any_piece_isolated=true" in w)
        assert fire_count == 1, "Should fire exactly once even when both flags are true"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4 — "narrowing the gap" mobility-direction
# ─────────────────────────────────────────────────────────────────────────────

_NARROW_FACTS_REVERSED = {
    # our_after=12 > opp_after=10 → gap REVERSED (we lead)
    "our_mobility_after": 12,
    "opponent_mobility_after": 10,
    "our_mobility_before": 10,
    "opponent_mobility_before": 11,
}

_NARROW_FACTS_MATCHED = {
    # our_after=10 == opp_after=10 → gap MATCHED (not narrowed)
    "our_mobility_after": 10,
    "opponent_mobility_after": 10,
    "our_mobility_before": 8,
    "opponent_mobility_before": 11,
}

_NARROW_FACTS_STILL_BEHIND = {
    # our_after=9 < opp_after=11 → gap truly narrowed (legal)
    "our_mobility_after": 9,
    "opponent_mobility_after": 11,
    "our_mobility_before": 8,
    "opponent_mobility_before": 11,
}

_NARROW_BAD_TEXT = (
    "The move increases our mobility while narrowing the gap in available replies."
)

_NARROW_GOOD_TEXT = (
    "The move increases our mobility from 10 to 12, exceeding the opponent's 10."
)


class TestNarrowingGapDirection:
    """Fix 4: 'narrowing the gap' forbidden when our_mobility_after >= opp_mobility_after."""

    def test_runtime_fires_when_gap_reversed(self):
        warnings = _rt(_NARROW_BAD_TEXT, _NARROW_FACTS_REVERSED)
        assert _has(warnings, "narrowing the gap")

    def test_runtime_fires_when_gap_matched(self):
        warnings = _rt(_NARROW_BAD_TEXT, _NARROW_FACTS_MATCHED)
        assert _has(warnings, "narrowing the gap")

    def test_runtime_clean_when_still_behind(self):
        # Gap genuinely narrowed — should not fire.
        warnings = _rt(_NARROW_BAD_TEXT, _NARROW_FACTS_STILL_BEHIND)
        assert not _has(warnings, "narrowing the gap")

    def test_runtime_clean_when_phrase_absent(self):
        warnings = _rt(_NARROW_GOOD_TEXT, _NARROW_FACTS_REVERSED)
        assert not _has(warnings, "narrowing the gap")

    def test_evaluator_fires_when_gap_reversed(self):
        warnings = _ev(_NARROW_BAD_TEXT, _NARROW_FACTS_REVERSED)
        assert _has(warnings, "narrowing the gap")

    def test_evaluator_fires_when_gap_matched(self):
        warnings = _ev(_NARROW_BAD_TEXT, _NARROW_FACTS_MATCHED)
        assert _has(warnings, "narrowing the gap")

    def test_evaluator_clean_when_still_behind(self):
        warnings = _ev(_NARROW_BAD_TEXT, _NARROW_FACTS_STILL_BEHIND)
        assert not _has(warnings, "narrowing the gap")

    def test_runtime_evaluator_agree_on_fire_reversed(self):
        rt = _rt(_NARROW_BAD_TEXT, _NARROW_FACTS_REVERSED)
        ev = _ev(_NARROW_BAD_TEXT, _NARROW_FACTS_REVERSED)
        rt_has = _has(rt, "narrowing the gap")
        ev_has = _has(ev, "narrowing the gap")
        assert rt_has == ev_has, f"E.1 violation reversed: runtime={rt_has}, evaluator={ev_has}"

    def test_runtime_evaluator_agree_on_fire_matched(self):
        rt = _rt(_NARROW_BAD_TEXT, _NARROW_FACTS_MATCHED)
        ev = _ev(_NARROW_BAD_TEXT, _NARROW_FACTS_MATCHED)
        rt_has = _has(rt, "narrowing the gap")
        ev_has = _has(ev, "narrowing the gap")
        assert rt_has == ev_has, f"E.1 violation matched: runtime={rt_has}, evaluator={ev_has}"

    def test_runtime_evaluator_agree_on_clean_behind(self):
        rt = _rt(_NARROW_BAD_TEXT, _NARROW_FACTS_STILL_BEHIND)
        ev = _ev(_NARROW_BAD_TEXT, _NARROW_FACTS_STILL_BEHIND)
        rt_has = _has(rt, "narrowing the gap")
        ev_has = _has(ev, "narrowing the gap")
        assert rt_has == ev_has, f"E.1 violation clean: runtime={rt_has}, evaluator={ev_has}"

    def test_warning_contains_fact_values(self):
        # Warning string should include the actual fact values.
        warnings = _rt(_NARROW_BAD_TEXT, _NARROW_FACTS_REVERSED)
        matching = [w for w in warnings if "narrowing the gap" in w]
        assert matching
        w = matching[0]
        assert "12" in w  # our_mobility_after
        assert "10" in w  # opponent_mobility_after

    def test_turn_0013_scenario(self):
        # Reproduces the trace bug: our_after=12 > opp_after=10.
        facts = {
            "our_mobility_after": 12,
            "opponent_mobility_after": 10,
            "our_mobility_before": 10,
            "opponent_mobility_before": 11,
        }
        text = (
            "The move reduces the opponent's mobility from 11 to 10 while "
            "increasing ours from 10 to 12, narrowing the gap in available replies."
        )
        assert _has(_rt(text, facts), "narrowing the gap")
        assert _has(_ev(text, facts), "narrowing the gap")

    def test_turn_0007_narrowing_is_correct(self):
        # Turn 7: our_after=9, opp_after=11 — gap genuinely narrows.
        facts = {
            "our_mobility_after": 9,
            "opponent_mobility_after": 11,
            "our_mobility_before": 8,
            "opponent_mobility_before": 11,
        }
        text = (
            "The move increases our mobility from 8 to 9, narrowing the gap "
            "in available options though the opponent's mobility remains at 11."
        )
        assert not _has(_rt(text, facts), "narrowing the gap")
        assert not _has(_ev(text, facts), "narrowing the gap")


# ─────────────────────────────────────────────────────────────────────────────
# E.1 invariant — all four fixes must produce parity
# ─────────────────────────────────────────────────────────────────────────────

class TestE1InvariantPhaseA:
    """All new checks must show runtime ↔ evaluator parity."""

    def test_fix2_parity_on_five_random_texts(self):
        seeds = ["our mobility changes from 11 to 9 — decreases our mobility by 2"]
        facts = {"our_mobility_before": 11, "our_mobility_after": 9}
        texts = [
            "The piece moves forward to (5, 2).",
            "The piece advances, decreasing our mobility.",
            "Maintaining safety while opponent stays at 9.",
            "Our mobility drops from 11 to 9 after the move.",
            "No captures; the engine scores this move 5.0.",
        ]
        for t in texts:
            rt = _rt(t, facts, seeds=seeds)
            ev = _ev(t, facts, seeds=seeds)
            rt_f = _has(rt, "our-mobility decrease seeded but omitted")
            ev_f = _has(ev, "our-mobility decrease seeded but omitted")
            assert rt_f == ev_f, (
                f"E.1 Fix2 parity failure on: '{t[:50]}' "
                f"(runtime={rt_f}, evaluator={ev_f})"
            )

    def test_fix3_parity_on_four_texts(self):
        facts = {"any_piece_isolated": True}
        texts = [
            "Move forward ensuring no tactical vulnerabilities are created.",
            "The piece is safe and no vulnerabilities are created.",
            "The piece advances with one ally piece remaining isolated.",
            "Safe advance from (6, 1) to (5, 2).",
        ]
        for t in texts:
            rt = _rt(t, facts)
            ev = _ev(t, facts)
            rt_f = _has(rt, "any_piece_isolated=true")
            ev_f = _has(ev, "any_piece_isolated=true")
            assert rt_f == ev_f, (
                f"E.1 Fix3 parity failure on: '{t[:50]}' "
                f"(runtime={rt_f}, evaluator={ev_f})"
            )

    def test_fix4_parity_on_four_texts(self):
        facts = {"our_mobility_after": 12, "opponent_mobility_after": 10}
        texts = [
            "Narrowing the gap between our options and the opponent's.",
            "This increases our flexibility while narrowing the gap.",
            "Our mobility now exceeds the opponent's.",
            "The gap is reversed, not merely narrowed.",
        ]
        for t in texts:
            rt = _rt(t, facts)
            ev = _ev(t, facts)
            rt_f = _has(rt, "narrowing the gap")
            ev_f = _has(ev, "narrowing the gap")
            assert rt_f == ev_f, (
                f"E.1 Fix4 parity failure on: '{t[:50]}' "
                f"(runtime={rt_f}, evaluator={ev_f})"
            )
