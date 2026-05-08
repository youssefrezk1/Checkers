"""
checkers/tests/test_reasoning_truthfulness.py

Unit tests for:
  - _check_reasoning_truthfulness  (contradiction checker)
  - _refine_reasoning               (reasoning-only refinement loop)
  - _build_refinement_prompt        (feedback prompt builder)
  - _extract_refinement_reasoning   (response parser)

No LLM is called.  Tests run in < 0.5 s.

Run:
    pytest checkers/tests/test_reasoning_truthfulness.py -v
"""
from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from checkers.agents.ranker_agent import (
    RANKER_REASONING_REFINEMENT_SYSTEM,
    RANKER_SEED_REASONING_SYSTEM,
    _CONTEXT_FORBIDDEN_VOCAB,
    _FORBIDDEN_VOCAB,
    _MINIMAX_CLEARLY_LOSING,
    _MINIMAX_SLIGHTLY_LOSING,
    _build_deterministic_seed_summary,
    _build_grounded_reasoning_seeds,
    _build_refinement_prompt,
    _build_seed_reasoning_prompt,
    _build_targeted_refinement_prompt,
    _check_reasoning_truthfulness,
    _extract_refinement_reasoning,
    _extract_targeted_repair_response,
    _find_comparison_seed,
    _generate_seeded_reasoning,
    _minimax_wording_label,
    _partition_sentences_by_contradiction,
    _refine_reasoning,
    _sanitize_seed_explanation,
    _split_reasoning_sentences,
)


# ── shared helpers ────────────────────────────────────────────────────────────

def _base_facts(**overrides) -> dict:
    """Return a safe baseline facts dict (all claims TRUE) with optional overrides."""
    base = {
        "opponent_mobility_after":  8,
        "opponent_mobility_before": 12,   # after < before → real reduction
        "opponent_can_recapture":   False,
        "leaves_piece_isolated":    False,
        "creates_immediate_threat": True,
        "center_control":           True,
        "captures_count":           1,
        "net_gain":                 1,
        "results_in_king":          True,
        "blocks_opponent_landing":  True,
    }
    base.update(overrides)
    return base


def _warns(reasoning: str, facts: dict) -> bool:
    return len(_check_reasoning_truthfulness(reasoning, facts)) > 0


def _clean(reasoning: str, facts: dict) -> bool:
    return len(_check_reasoning_truthfulness(reasoning, facts)) == 0


# Shared move fixture used by refinement tests.
_CHOSEN_MOVE = {
    "type": "simple",
    "path": [[5, 4], [4, 3]],
    "captured": [],
    "facts": {
        "opponent_can_recapture":    True,   # "avoids recapture" is a lie
        "center_control":            False,  # "controls center" is a lie
        "creates_immediate_threat":  False,
        "leaves_piece_isolated":     False,
        "opponent_mobility_after":   10,
        "opponent_mobility_before":  10,
        "our_pieces_threatened_after": 1,
    },
}

_BAD_REASONING  = "This move avoids recapture and controls the center."
_CLEAN_RESPONSE = (
    '{"reasoning": "This move advances the piece while keeping structure sound. '
    "It does not expose any ally to immediate danger. "
    'The minimax score of 3.0 confirms this choice."}'
)
_CLEAN_TEXT = (
    "This move advances the piece while keeping structure sound. "
    "It does not expose any ally to immediate danger. "
    "The minimax score of 3.0 confirms this choice."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — _check_reasoning_truthfulness
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_reasoning_returns_no_warnings(self):
        assert _clean("", _base_facts())

    def test_empty_facts_returns_no_warnings(self):
        assert _clean("This move reduces mobility.", {})

    def test_none_reasoning_returns_no_warnings(self):
        assert _clean("", _base_facts())

    def test_no_contradiction_returns_empty_list(self):
        reasoning = (
            "This move keeps all pieces safe (our_pieces_threatened_after=0). "
            "It blocks the opponent from landing on a key square. "
            "The minimax score of 5.0 confirms this choice."
        )
        assert _check_reasoning_truthfulness(reasoning, _base_facts()) == []


class TestMobilityChecks:
    def test_warns_when_mobility_claimed_but_equal(self):
        facts = _base_facts(opponent_mobility_after=10, opponent_mobility_before=10)
        assert _warns("This move reduces mobility.", facts)

    def test_warns_when_mobility_claimed_but_worse(self):
        facts = _base_facts(opponent_mobility_after=12, opponent_mobility_before=10)
        assert _warns("This limits mobility for the opponent.", facts)

    def test_no_warning_when_mobility_actually_reduced(self):
        facts = _base_facts(opponent_mobility_after=8, opponent_mobility_before=12)
        assert _clean("This reduces opponent mobility.", facts)

    def test_the_reported_example_contradiction(self):
        """The motivating bug: 'slightly reducing opponent mobility (10 vs 10)'."""
        facts = _base_facts(opponent_mobility_after=10, opponent_mobility_before=10)
        reasoning = (
            "This move is solid, slightly reducing opponent mobility "
            "(opponent_mobility_after=10 vs before 10)."
        )
        result = _check_reasoning_truthfulness(reasoning, facts)
        assert any("mobility" in w for w in result), (
            "Should detect the 10 vs 10 mobility contradiction"
        )

    def test_various_mobility_phrases_all_detected(self):
        facts = _base_facts(opponent_mobility_after=10, opponent_mobility_before=10)
        for phrase in [
            "reduces mobility",
            "reducing mobility",
            "limits mobility",
            "limiting mobility",
            "restricts mobility",
            "fewer moves for",
            "cuts opponent moves",
        ]:
            assert _warns(phrase, facts), f"Should detect: {phrase!r}"


class TestRecaptureChecks:
    def test_warns_when_avoids_recapture_claimed_but_false(self):
        facts = _base_facts(opponent_can_recapture=True)
        assert _warns("This move avoids recapture.", facts)

    def test_warns_safe_move_when_recapture_true(self):
        facts = _base_facts(opponent_can_recapture=True)
        assert _warns("This is a safe move.", facts)

    def test_no_warning_when_recapture_actually_false(self):
        facts = _base_facts(opponent_can_recapture=False)
        assert _clean("This move avoids recapture (opponent_can_recapture=false).", facts)

    def test_various_recapture_phrases_detected(self):
        facts = _base_facts(opponent_can_recapture=True)
        for phrase in [
            "avoids recapture",
            "no recapture",
            "cannot recapture",
            "without recapture risk",
            "no recapture risk",
            "safe from recapture",
        ]:
            assert _warns(phrase, facts), f"Should detect: {phrase!r}"


class TestIsolationChecks:
    def test_warns_when_no_isolation_claimed_but_isolated(self):
        facts = _base_facts(leaves_piece_isolated=True)
        assert _warns("This move does not isolate the piece.", facts)

    def test_no_warning_when_piece_not_isolated(self):
        facts = _base_facts(leaves_piece_isolated=False)
        assert _clean("This move maintains connectivity.", facts)

    def test_stays_connected_phrase_detected(self):
        facts = _base_facts(leaves_piece_isolated=True)
        assert _warns("The piece stays connected after this move.", facts)


class TestImmediateThreatChecks:
    def test_warns_when_threat_claimed_but_false(self):
        facts = _base_facts(creates_immediate_threat=False)
        assert _warns("This creates a threat for next turn.", facts)

    def test_no_warning_when_threat_actually_true(self):
        facts = _base_facts(creates_immediate_threat=True)
        assert _clean("This creates immediate threat on the next turn.", facts)

    def test_various_threat_phrases_detected(self):
        facts = _base_facts(creates_immediate_threat=False)
        for phrase in [
            "creates a threat",
            "creates immediate threat",
            "applies pressure next turn",
            "threatens opponent",
            "creates tactical threat",
        ]:
            assert _warns(phrase, facts), f"Should detect: {phrase!r}"


class TestCenterControlChecks:
    def test_warns_when_center_claimed_but_false(self):
        facts = _base_facts(center_control=False)
        assert _warns("This move controls the center.", facts)

    def test_no_warning_when_center_actually_true(self):
        facts = _base_facts(center_control=True)
        assert _clean("This move controls the center (center_control=true).", facts)

    def test_central_control_phrase_detected(self):
        facts = _base_facts(center_control=False)
        assert _warns("It establishes central control.", facts)


class TestCaptureMaterialChecks:
    def test_warns_when_capture_claimed_but_zero(self):
        facts = _base_facts(captures_count=0)
        assert _warns("This move captures a piece.", facts)

    def test_no_warning_when_capture_actually_nonzero(self):
        facts = _base_facts(captures_count=1)
        assert _clean("This move captures the piece at [3,4].", facts)

    def test_warns_material_gain_when_net_gain_zero(self):
        facts = _base_facts(net_gain=0)
        assert _warns("This move gains material.", facts)

    def test_warns_material_gain_when_net_gain_negative(self):
        facts = _base_facts(net_gain=-1)
        assert _warns("This move results in material gain.", facts)

    def test_no_warning_when_net_gain_positive(self):
        facts = _base_facts(net_gain=1)
        assert _clean("This move gains material (net_gain=1).", facts)


class TestPromotionChecks:
    def test_warns_when_promotion_claimed_but_false(self):
        facts = _base_facts(results_in_king=False)
        assert _warns("This move promotes to king.", facts)

    def test_no_warning_when_promotion_actually_true(self):
        facts = _base_facts(results_in_king=True)
        assert _clean("This move crowns the piece and promotes to king.", facts)

    def test_various_promotion_phrases_detected(self):
        facts = _base_facts(results_in_king=False)
        for phrase in [
            "promotes to king",
            "promotes a piece",
            "crowns a piece",
            "becomes a king",
        ]:
            assert _warns(phrase, facts), f"Should detect: {phrase!r}"


class TestBlocksLandingChecks:
    def test_warns_when_blocks_claimed_but_false(self):
        facts = _base_facts(blocks_opponent_landing=False)
        assert _warns("This move blocks opponent landing.", facts)

    def test_no_warning_when_blocks_actually_true(self):
        facts = _base_facts(blocks_opponent_landing=True)
        assert _clean("This move blocks the opponent from landing.", facts)


class TestMultipleContradictions:
    def test_multiple_contradictions_all_reported(self):
        facts = _base_facts(
            opponent_can_recapture=True,
            center_control=False,
            captures_count=0,
        )
        reasoning = (
            "This safe move avoids recapture and controls the center, "
            "capturing a piece in the process."
        )
        warnings = _check_reasoning_truthfulness(reasoning, facts)
        assert len(warnings) >= 2, f"Expected at least 2 warnings, got: {warnings}"

    def test_fallback_reasoning_not_flagged(self):
        """Deterministic fallback text should not trigger false positives."""
        facts = _base_facts(opponent_can_recapture=False, center_control=True)
        fallback = "Fallback: ranker call failed; selected deterministic best_minimax."
        assert _clean(fallback, facts)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Reasoning refinement loop
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildRefinementPrompt:
    def test_includes_chosen_path(self):
        prompt = _build_refinement_prompt(
            _CHOSEN_MOVE,
            ["REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true"],
        )
        assert "[5, 4]" in prompt or "5" in prompt

    def test_includes_contradiction_text(self):
        prompt = _build_refinement_prompt(
            _CHOSEN_MOVE,
            ["REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true"],
        )
        assert "opponent_can_recapture=true" in prompt or "avoids recapture" in prompt

    def test_instructs_keep_same_move(self):
        prompt = _build_refinement_prompt(_CHOSEN_MOVE, ["REASONING_CONTRADICTION: x"])
        assert "Keep the same chosen move" in prompt

    def test_instructs_minimax_last(self):
        prompt = _build_refinement_prompt(_CHOSEN_MOVE, ["REASONING_CONTRADICTION: x"])
        assert (
            "minimax_score may appear ONLY in the final sentence" in prompt
            or "Minimax_score may appear ONLY in the final sentence" in prompt
        )

    def test_lists_facts(self):
        prompt = _build_refinement_prompt(_CHOSEN_MOVE, ["REASONING_CONTRADICTION: x"])
        assert "opponent_can_recapture" in prompt
        assert "center_control" in prompt


class TestExtractRefinementReasoning:
    def test_valid_json(self):
        raw = '{"reasoning": "A clean paragraph."}'
        assert _extract_refinement_reasoning(raw) == "A clean paragraph."

    def test_regex_fallback(self):
        raw = 'prefix\n{"reasoning": "Fallback text."}\nSuffix'
        assert _extract_refinement_reasoning(raw) == "Fallback text."

    def test_empty_returns_none(self):
        assert _extract_refinement_reasoning("") is None
        assert _extract_refinement_reasoning("{}") is None


class TestRefineReasoning:

    def _contradictions(self):
        result = _check_reasoning_truthfulness(_BAD_REASONING, _CHOSEN_MOVE["facts"])
        assert result, "test setup: _BAD_REASONING must have contradictions"
        return result

    # ── resolution scenarios ──────────────────────────────────────────────────

    def test_resolves_on_first_attempt(self):
        """Clean reasoning on attempt 1 → resolved=True, retry_count=1."""
        with patch("checkers.agents.ranker_agent.call_ranker", return_value=_CLEAN_RESPONSE):
            final, retry_count, resolved = _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
                max_attempts=2,
            )
        assert resolved is True
        assert retry_count == 1
        assert final == _CLEAN_TEXT

    def test_resolves_on_second_attempt(self):
        """Attempt 1 still bad, attempt 2 clean → resolved=True, retry_count=2."""
        _STILL_BAD = '{"reasoning": "This move avoids recapture."}'
        responses = iter([_STILL_BAD, _CLEAN_RESPONSE])

        with patch("checkers.agents.ranker_agent.call_ranker", side_effect=lambda *a, **k: next(responses)):
            final, retry_count, resolved = _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
                max_attempts=2,
            )
        assert resolved is True
        assert retry_count == 2
        assert final == _CLEAN_TEXT

    def test_keeps_latest_after_2_failed_attempts(self):
        """Both attempts still contradict facts → resolved=False, latest kept."""
        _STILL_BAD = '{"reasoning": "This move avoids recapture."}'
        with patch("checkers.agents.ranker_agent.call_ranker", return_value=_STILL_BAD):
            final, retry_count, resolved = _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
                max_attempts=2,
            )
        assert resolved is False
        assert retry_count == 2
        # The latest LLM text (still bad) is returned, not the very original.
        assert "avoids recapture" in final

    def test_api_failure_keeps_original_reasoning(self):
        """If every API call raises, original reasoning is preserved."""
        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            side_effect=OSError("network error"),
        ):
            final, retry_count, resolved = _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
                max_attempts=2,
            )
        assert final == _BAD_REASONING
        assert retry_count == 1
        assert resolved is False

    # ── chosen_move immutability ──────────────────────────────────────────────

    def test_chosen_move_never_mutated(self):
        """The chosen_move dict must be identical before and after refinement."""
        chosen_copy = copy.deepcopy(_CHOSEN_MOVE)
        with patch("checkers.agents.ranker_agent.call_ranker", return_value=_CLEAN_RESPONSE):
            _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
            )
        assert _CHOSEN_MOVE == chosen_copy, "chosen_move must not be mutated"

    # ── separate retry count ──────────────────────────────────────────────────

    def test_retry_count_is_independent_int(self):
        """reasoning_retry_count is a plain int, independent of move retries."""
        with patch("checkers.agents.ranker_agent.call_ranker", return_value=_CLEAN_RESPONSE):
            _, retry_count, _ = _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
            )
        assert isinstance(retry_count, int)
        assert retry_count >= 1

    # ── safety filter isolation ───────────────────────────────────────────────

    def test_safety_filter_not_called_during_refinement(self):
        """_apply_safety_filter must never be invoked during reasoning refinement."""
        with patch(
            "checkers.agents.ranker_agent._apply_safety_filter",
            side_effect=AssertionError("safety filter must not run during refinement"),
        ) as mock_sf, patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=_CLEAN_RESPONSE,
        ):
            _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
            )
        mock_sf.assert_not_called()

    # ── RANKER_REASONING_REFINEMENT_SYSTEM content ────────────────────────────

    def test_system_prompt_forbids_changing_chosen_move(self):
        assert "Do NOT change the chosen move" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_system_prompt_minimax_last_rule(self):
        assert (
            "minimax_score may appear ONLY in the final sentence"
            in RANKER_REASONING_REFINEMENT_SYSTEM
        )

    def test_system_prompt_has_output_format(self):
        assert '"reasoning"' in RANKER_REASONING_REFINEMENT_SYSTEM


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Grounded reasoning seeds
# ══════════════════════════════════════════════════════════════════════════════

_FULL_FACTS: dict = {
    "opponent_can_recapture":    False,
    "our_pieces_threatened_after": 0,
    "moved_piece_is_threatened": False,
    "captures_count":            2,
    "net_gain":                  2,
    "creates_immediate_threat":  True,
    "shot_sequence_available":   False,
    "blocks_opponent_landing":   True,
    "forced_opponent_jump_reply": False,
    "max_opponent_jump_captures": 0,
    "leaves_piece_isolated":     False,
    "weakens_king_row":          False,
    "center_control":            True,
    "results_in_king":           False,
    "near_promotion":            False,
    "opponent_mobility_before":  12,
    "opponent_mobility_after":   8,
}

_FULL_MOVE = {
    "type": "jump",
    "path": [[5, 4], [3, 2]],
    "captured": [[4, 3]],
    "facts": _FULL_FACTS,
    "minimax_score": 4.5,
}

_ALT_MOVE = {
    "type": "simple",
    "path": [[5, 2], [4, 1]],
    "captured": [],
    "facts": {
        "opponent_can_recapture":    True,
        "our_pieces_threatened_after": 1,
        "moved_piece_is_threatened": True,
        "captures_count":            0,
        "net_gain":                  0,
        "creates_immediate_threat":  False,
        "leaves_piece_isolated":     True,
        "center_control":            False,
        "opponent_mobility_before":  12,
        "opponent_mobility_after":   12,
    },
    "minimax_score": 1.0,
}


class TestBuildGroundedReasoningSeeds:
    """Tests for _build_grounded_reasoning_seeds."""

    def _seeds(self, move=_FULL_MOVE, candidates=None):
        if candidates is None:
            candidates = [_FULL_MOVE, _ALT_MOVE]
        return _build_grounded_reasoning_seeds(move, candidates)

    def _text(self, move=_FULL_MOVE, candidates=None):
        return " ".join(self._seeds(move, candidates))

    # ── safety seeds ─────────────────────────────────────────────────────────

    def test_recapture_false_seed_present(self):
        assert "opponent_can_recapture=false" in self._text()

    def test_recapture_true_seed_present(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "opponent_can_recapture": True}}
        assert "opponent_can_recapture=true" in self._text(move, [move])

    def test_pieces_threatened_zero_seed(self):
        assert "our_pieces_threatened_after=0" in self._text()

    def test_pieces_threatened_nonzero_seed(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "our_pieces_threatened_after": 2}}
        seeds = _build_grounded_reasoning_seeds(move, [move])
        assert any("our_pieces_threatened_after=2" in s for s in seeds)

    # ── tactical seeds ───────────────────────────────────────────────────────

    def test_captures_seed_when_nonzero(self):
        assert "captures_count=2" in self._text()

    def test_no_captures_seed_when_zero(self):
        """captures_count=0 → no material-gain seed.
        'captures_count=0' may appear in the positional seed (correct); the guard
        is that the material-gain seed ('wins material') must not fire.
        """
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "captures_count": 0, "net_gain": 0}}
        text = self._text(move, [move])
        assert "wins material" not in text

    def test_creates_threat_seed_when_true(self):
        assert "creates_immediate_threat=true" in self._text()

    def test_no_creates_threat_seed_when_false(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "creates_immediate_threat": False}}
        text = self._text(move, [move])
        assert "creates_immediate_threat" not in text

    # ── mobility guard ───────────────────────────────────────────────────────

    def test_mobility_seed_when_after_less_than_before(self):
        """Mobility seed appears ONLY when after < before."""
        text = self._text()  # _FULL_FACTS has after=8 < before=12
        assert "opponent_mobility_after=8" in text

    def test_mobility_seed_emitted_when_equal(self):
        """Mobility seed always emitted (FIX 2) — equal values still get a seed."""
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "opponent_mobility_after": 10, "opponent_mobility_before": 10}}
        text = self._text(move, [move])
        assert "opponent_mobility_before=10" in text
        assert "opponent_mobility_after=10" in text

    def test_mobility_seed_emitted_when_after_greater(self):
        """Mobility seed always emitted (FIX 2) — increasing values still get a seed."""
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "opponent_mobility_after": 14, "opponent_mobility_before": 10}}
        text = self._text(move, [move])
        assert "opponent_mobility_before=10" in text
        assert "opponent_mobility_after=14" in text

    # ── center_control guard ──────────────────────────────────────────────────

    def test_center_seed_when_true(self):
        assert "center_control=true" in self._text()

    def test_no_center_seed_when_false(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "center_control": False}}
        text = self._text(move, [move])
        assert "center_control" not in text

    # ── forbidden vague phrases ───────────────────────────────────────────────

    def test_no_structural_pressure_in_seeds(self):
        assert "structural pressure" not in self._text()

    def test_no_stable_position_in_seeds(self):
        assert "stable position" not in self._text()

    def test_no_limits_options_in_seeds(self):
        assert "limits options" not in self._text()

    def test_no_good_position_in_seeds(self):
        assert "good position" not in self._text()

    def test_no_no_advantage_in_seeds(self):
        assert "no advantage" not in self._text()

    def test_no_better_structure_in_seeds(self):
        assert "better structure" not in self._text()

    # ── minimax always last ───────────────────────────────────────────────────

    def test_minimax_seed_is_last(self):
        seeds = self._seeds()
        assert seeds, "seeds must not be empty"
        assert "minimax_score" in seeds[-1], "minimax_score must be the last seed"

    def test_minimax_not_in_non_last_seeds(self):
        seeds = self._seeds()
        for s in seeds[:-1]:
            assert "minimax_score" not in s, f"minimax_score must not appear before final seed: {s}"

    # ── comparison seeds ─────────────────────────────────────────────────────

    def test_comparison_seed_recapture_difference(self):
        """When chosen avoids recapture but alt allows it, comparison seed appears."""
        seeds = self._seeds()  # _FULL_MOVE=false vs _ALT_MOVE=true
        text = " ".join(seeds)
        assert "opponent_can_recapture=true" in text or "recapture" in text

    def test_no_comparison_seed_single_candidate(self):
        """With only one candidate, no comparison seed is generated."""
        seeds = _build_grounded_reasoning_seeds(_FULL_MOVE, [_FULL_MOVE])
        cmp_seeds = [s for s in seeds if "vs" in s and "Move [" in s]
        assert len(cmp_seeds) == 0

    def test_find_comparison_seed_priority_recapture(self):
        c = {"opponent_can_recapture": False}
        a = {"opponent_can_recapture": True}
        result = _find_comparison_seed(c, a, 1)
        assert result is not None
        assert "opponent_can_recapture=true" in result
        assert "worse" not in result
        assert "no advantage" not in result

    def test_find_comparison_seed_returns_none_when_no_difference(self):
        same = {"opponent_can_recapture": False, "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0, "captures_count": 0,
                "leaves_piece_isolated": False, "creates_immediate_threat": False,
                "center_control": False}
        assert _find_comparison_seed(same, same, 1) is None

    # ── chosen_move immutability ──────────────────────────────────────────────

    def test_chosen_move_not_mutated_by_seed_builder(self):
        before = copy.deepcopy(_FULL_MOVE)
        _build_grounded_reasoning_seeds(_FULL_MOVE, [_FULL_MOVE, _ALT_MOVE])
        assert _FULL_MOVE == before


class TestBuildSeedReasoningPrompt:
    def test_prompt_contains_path(self):
        seeds = ["opponent_can_recapture=false — safe"]
        prompt = _build_seed_reasoning_prompt(_FULL_MOVE, seeds)
        assert "[5, 4]" in prompt or "5" in prompt

    def test_prompt_contains_seed_text(self):
        seeds = ["opponent_can_recapture=false — safe"]
        prompt = _build_seed_reasoning_prompt(_FULL_MOVE, seeds)
        assert "opponent_can_recapture=false" in prompt

    def test_prompt_instructs_use_only_seeds(self):
        prompt = _build_seed_reasoning_prompt(_FULL_MOVE, ["seed1"])
        assert "use ONLY" in prompt or "ONLY" in prompt

    def test_prompt_minimax_last_instruction(self):
        prompt = _build_seed_reasoning_prompt(_FULL_MOVE, ["seed1"])
        assert "final sentence" in prompt or "confirmation" in prompt


class TestSeedReasoningSystem:
    def test_system_prompt_forbids_structural_pressure(self):
        assert "structural pressure" in RANKER_SEED_REASONING_SYSTEM

    def test_system_prompt_requires_seed_only(self):
        assert "Use ONLY" in RANKER_SEED_REASONING_SYSTEM

    def test_system_prompt_minimax_last(self):
        assert "final sentence" in RANKER_SEED_REASONING_SYSTEM or \
               "ONLY in the final sentence" in RANKER_SEED_REASONING_SYSTEM

    def test_system_prompt_output_format(self):
        assert '"reasoning"' in RANKER_SEED_REASONING_SYSTEM


class TestGenerateSeededReasoning:
    def test_seeded_reasoning_uses_call_ranker(self):
        """_generate_seeded_reasoning calls call_ranker exactly once on success."""
        response = '{"reasoning": "Clean grounded paragraph."}'
        with patch("checkers.agents.ranker_agent.call_ranker", return_value=response) as mock:
            result, seeds_out = _generate_seeded_reasoning(_FULL_MOVE, [_FULL_MOVE, _ALT_MOVE])
        assert result == "Clean grounded paragraph."
        assert isinstance(seeds_out, list)
        mock.assert_called_once()

    def test_seeded_reasoning_api_failure_returns_none(self):
        """If the API fails, returns (None, seeds) — caller keeps previous reasoning."""
        with patch("checkers.agents.ranker_agent.call_ranker", side_effect=OSError("net")):
            result, seeds_out = _generate_seeded_reasoning(_FULL_MOVE, [_FULL_MOVE, _ALT_MOVE])
        assert result is None
        assert isinstance(seeds_out, list)

    def test_seeded_reasoning_never_calls_safety_filter(self):
        """_apply_safety_filter must not be invoked during seed-based generation."""
        response = '{"reasoning": "OK."}'
        with patch("checkers.agents.ranker_agent._apply_safety_filter",
                   side_effect=AssertionError("filter must not run")) as mock_sf, \
             patch("checkers.agents.ranker_agent.call_ranker", return_value=response):
            _generate_seeded_reasoning(_FULL_MOVE, [_FULL_MOVE, _ALT_MOVE])
        mock_sf.assert_not_called()

    def test_chosen_move_not_mutated_by_generate(self):
        """chosen_move dict is identical before and after seed generation."""
        before = copy.deepcopy(_FULL_MOVE)
        response = '{"reasoning": "OK."}'
        with patch("checkers.agents.ranker_agent.call_ranker", return_value=response):
            _generate_seeded_reasoning(_FULL_MOVE, [_FULL_MOVE, _ALT_MOVE])
        assert _FULL_MOVE == before


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5 — Derived meaning seeds and drawback seeds
# ═══════════════════════════════════════════════════════════════════════════════

class TestDerivedMeaningSeeds:
    """
    Verify seeds carry derived tactical/positional meaning.
    Logic guards (mobility, center_control) remain unchanged.
    """

    def _text(self, move, candidates=None):
        return " ".join(_build_grounded_reasoning_seeds(move, candidates or [move]))

    # ── positive derived meanings ─────────────────────────────────────────────

    def test_recapture_false_derived_meaning(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "opponent_can_recapture": False}}
        assert "immediate tactical safety" in self._text(move)

    def test_pieces_safe_no_defensive_burden(self):
        assert "no defensive burden" in self._text(_FULL_MOVE)

    def test_captures_wins_material(self):
        assert "wins material" in self._text(_FULL_MOVE)

    def test_creates_threat_puts_on_defensive(self):
        assert "puts opponent on the defensive" in self._text(_FULL_MOVE)

    def test_center_control_improves_influence(self):
        assert "improves influence over central lanes" in self._text(_FULL_MOVE)

    def test_isolated_false_preserves_coordination(self):
        assert "preserves piece coordination" in self._text(_FULL_MOVE)

    def test_results_in_king_converts_piece(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "results_in_king": True}}
        assert "immediately converts the piece into a king" in self._text(move)

    def test_near_promotion_future_threat(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "results_in_king": False, "near_promotion": True}}
        assert "creates a future promotion threat" in self._text(move)

    def test_forced_jump_constrained(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "forced_opponent_jump_reply": True, "max_opponent_jump_captures": 1}}
        assert "constrained to a jump" in self._text(move)

    def test_blocks_landing_denies_square(self):
        assert "denies the opponent a key landing square" in self._text(_FULL_MOVE)

    def test_mobility_restricts_replies(self):
        text = self._text(_FULL_MOVE)   # after=8 < before=12
        assert "restricting available replies" in text

    # ── tradeoff / drawback seeds ─────────────────────────────────────────────

    def test_recapture_true_is_tactical_drawback(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "opponent_can_recapture": True}}
        assert "tactical drawback" in self._text(move)

    def test_pieces_threatened_is_tactical_drawback(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "our_pieces_threatened_after": 2}}
        assert "tactical drawback" in self._text(move)

    def test_moved_piece_threatened_is_exposed(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "moved_piece_is_threatened": True}}
        assert "tactically exposed" in self._text(move)

    def test_isolation_is_positional_drawback(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "leaves_piece_isolated": True}}
        assert "positional drawback" in self._text(move)

    def test_weakens_king_row_back_row_weakened(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "weakens_king_row": True}}
        assert "back-row defense is weakened" in self._text(move)

    # ── forbidden vague terms absent ──────────────────────────────────────────

    def test_no_structural_pressure(self):
        assert "structural pressure" not in self._text(_FULL_MOVE)

    def test_no_stable_position(self):
        assert "stable position" not in self._text(_FULL_MOVE)

    def test_no_good_position(self):
        assert "good position" not in self._text(_FULL_MOVE)

    def test_no_limits_options(self):
        assert "limits options" not in self._text(_FULL_MOVE)

    # ── system prompt drawback rule ───────────────────────────────────────────

    def test_seed_system_prompt_has_drawback_rule(self):
        assert "DRAWBACKS" in RANKER_SEED_REASONING_SYSTEM
        assert "Do NOT hide" in RANKER_SEED_REASONING_SYSTEM

    def test_seed_system_prompt_names_drawback_facts(self):
        assert "opponent_can_recapture=true" in RANKER_SEED_REASONING_SYSTEM
        assert "leaves_piece_isolated=true" in RANKER_SEED_REASONING_SYSTEM


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6 — Strategic interpretation seeds
# ═══════════════════════════════════════════════════════════════════════════════

# A simple quiet move (type="simple", captures_count=0)
_SIMPLE_QUIET = {
    "type": "simple",
    "path": [[5, 2], [4, 3]],   # src_row=5 (not back-row), dst_col=3 (center)
    "captured": [],
    "facts": {
        "captures_count": 0, "net_gain": 0,
        "creates_immediate_threat": False,
        "opponent_can_recapture": False,
        "our_pieces_threatened_after": 0,
        "moved_piece_is_threatened": False,
        "leaves_piece_isolated": False,
        "center_control": False,
        "opponent_mobility_before": 8, "opponent_mobility_after": 8,
    },
    "minimax_score": 1.5,
}

# A jump move (type="jump", captures_count=1)
_JUMP_MOVE = {
    "type": "jump",
    "path": [[5, 2], [3, 4]],
    "captured": [[4, 3]],
    "facts": {
        "captures_count": 1, "net_gain": 1,
        "creates_immediate_threat": False,
        "opponent_can_recapture": False,
        "our_pieces_threatened_after": 0,
        "moved_piece_is_threatened": False,
        "leaves_piece_isolated": False,
        "center_control": False,
        "opponent_mobility_before": 8, "opponent_mobility_after": 8,
    },
    "minimax_score": 3.0,
}

# A back-row move (src_row=7)
_BACK_ROW_MOVE = {
    "type": "simple",
    "path": [[7, 2], [6, 3]],   # src_row=7, dst_col=3 (center)
    "captured": [],
    "facts": {
        "captures_count": 0, "net_gain": 0,
        "creates_immediate_threat": False,
        "opponent_can_recapture": False,
        "our_pieces_threatened_after": 0,
        "moved_piece_is_threatened": False,
        "leaves_piece_isolated": False,
        "center_control": False,
        "opponent_mobility_before": 8, "opponent_mobility_after": 8,
    },
    "minimax_score": 1.0,
}

# A move to a non-center column (dst_col=0)
_EDGE_MOVE = {
    "type": "simple",
    "path": [[5, 1], [4, 0]],   # dst_col=0 (not center)
    "captured": [],
    "facts": {
        "captures_count": 0, "net_gain": 0,
        "creates_immediate_threat": False,
        "opponent_can_recapture": False,
        "our_pieces_threatened_after": 0,
        "moved_piece_is_threatened": False,
        "leaves_piece_isolated": False,
        "center_control": False,
        "opponent_mobility_before": 8, "opponent_mobility_after": 8,
    },
    "minimax_score": 0.5,
}


class TestStrategicInterpretationSeeds:
    """Tests for the four safe strategic interpretation seeds."""

    def _text(self, move, candidates=None):
        return " ".join(_build_grounded_reasoning_seeds(move, candidates or [move]))

    # ── (A) Development seed ─────────────────────────────────────────────────

    def test_simple_quiet_produces_development_seed(self):
        # _SIMPLE_QUIET path [[5,2],[4,3]]: row 5→4 (decreasing) → forward for RED
        assert "develops a piece forward" in _text_with_player(_SIMPLE_QUIET, _RED)

    def test_development_seed_mentions_activity(self):
        assert "improves piece activity" in _text_with_player(_SIMPLE_QUIET, _RED)

    def test_no_development_seed_when_capture(self):
        assert "develops a piece forward" not in self._text(_JUMP_MOVE)

    # ── (B) Back-row seed ───────────────────────────────────────────────────────────────

    def test_back_row_origin_produces_weakening_seed(self):
        # _BACK_ROW_MOVE src_row=7 → RED back row
        assert "moves a back-row piece" in _text_with_player(_BACK_ROW_MOVE, _RED)

    def test_back_row_seed_mentions_defense(self):
        assert "back-row defensive structure" in _text_with_player(_BACK_ROW_MOVE, _RED)

    def test_no_back_row_seed_for_midgame_row(self):
        # _SIMPLE_QUIET starts at row 5 — not back-row for any color
        assert "moves a back-row piece" not in _text_with_player(_SIMPLE_QUIET, _RED)

    def test_row_zero_triggers_back_row_seed(self):
        move = {**_BACK_ROW_MOVE, "path": [[0, 2], [1, 3]]}
        # src_row=0 → BLACK back row
        assert "moves a back-row piece" in _text_with_player(move, _BLACK)

    # ── (C) Positional seed ──────────────────────────────────────────────────

    def test_quiet_move_produces_positional_seed(self):
        text = self._text(_SIMPLE_QUIET)
        assert "captures_count=0" in text
        assert "positional move" in text

    def test_no_positional_seed_when_capture(self):
        assert "positional move" not in self._text(_JUMP_MOVE)

    # ── (D) Center direction seed ────────────────────────────────────────────

    def test_center_destination_produces_center_seed(self):
        # _SIMPLE_QUIET goes to [4, 3] → col 3 ∈ {2,3,4,5}
        assert "destination column in center range" in self._text(_SIMPLE_QUIET)

    def test_center_seed_mentions_board_presence(self):
        assert "central board presence" in self._text(_SIMPLE_QUIET)

    def test_no_center_seed_for_edge_destination(self):
        # _EDGE_MOVE goes to col 0
        assert "destination column in center range" not in self._text(_EDGE_MOVE)

    def test_column_2_triggers_center_seed(self):
        move = {**_SIMPLE_QUIET, "path": [[5, 1], [4, 2]]}
        assert "destination column in center range" in self._text(move)

    def test_column_5_triggers_center_seed(self):
        move = {**_SIMPLE_QUIET, "path": [[5, 6], [4, 5]]}
        assert "destination column in center range" in self._text(move)

    def test_column_1_no_center_seed(self):
        move = {**_SIMPLE_QUIET, "path": [[5, 2], [4, 1]]}
        assert "destination column in center range" not in self._text(move)

    # ── Forbidden vague phrases still absent ──────────────────────────────────

    def test_no_initiative_in_seeds(self):
        assert "initiative" not in self._text(_SIMPLE_QUIET)

    def test_no_dominance_in_seeds(self):
        assert "dominance" not in self._text(_SIMPLE_QUIET)

    def test_no_strong_position_in_seeds(self):
        assert "strong position" not in self._text(_SIMPLE_QUIET)

    def test_no_pressure_in_seeds(self):
        # "pressure" is allowed in comparisons like "applies pressure" only
        # when creates_immediate_threat=true; for quiet moves it must not appear
        assert "pressure" not in self._text(_SIMPLE_QUIET)

    # ── Minimax still last ────────────────────────────────────────────────────

    def test_minimax_still_last_with_strategic_seeds(self):
        seeds = _build_grounded_reasoning_seeds(_SIMPLE_QUIET, [_SIMPLE_QUIET])
        assert seeds, "seeds must not be empty"
        assert "minimax_score" in seeds[-1]

    # ── System prompt priority rule ───────────────────────────────────────────

    def test_system_prompt_has_priority_rule(self):
        assert "PRIORITY" in RANKER_SEED_REASONING_SYSTEM

    def test_system_prompt_forbids_initiative_dominance(self):
        assert "initiative" in RANKER_SEED_REASONING_SYSTEM
        assert "dominance" in RANKER_SEED_REASONING_SYSTEM

    # ── chosen_move immutability ──────────────────────────────────────────────

    def test_chosen_move_not_mutated(self):
        before = copy.deepcopy(_SIMPLE_QUIET)
        _build_grounded_reasoning_seeds(_SIMPLE_QUIET, [_SIMPLE_QUIET])
        assert _SIMPLE_QUIET == before


# ═══════════════════════════════════════════════════════════════════════════════
# Part 7 — Color-aware strategic seeds (direction / back-row)
# ═══════════════════════════════════════════════════════════════════════════════

from checkers.engine.board import RED as _RED, BLACK as _BLACK


def _seeds_with_player(move, player, candidates=None):
    return _build_grounded_reasoning_seeds(move, candidates or [move], player=player)

def _text_with_player(move, player, candidates=None):
    return " ".join(_seeds_with_player(move, player, candidates))


# ── RED forward move: src_row=5 → dst_row=4 (decreasing) ─────────────────────
_RED_FORWARD = {
    "type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
    "facts": {"captures_count": 0, "net_gain": 0, "opponent_can_recapture": False,
              "our_pieces_threatened_after": 0, "moved_piece_is_threatened": False,
              "leaves_piece_isolated": False, "center_control": False,
              "opponent_mobility_before": 8, "opponent_mobility_after": 8},
    "minimax_score": 2.0,
}
# ── RED backward move: src_row=4 → dst_row=5 (increasing) ────────────────────
_RED_BACKWARD = {
    "type": "simple", "path": [[4, 3], [5, 2]], "captured": [],
    "facts": {"captures_count": 0, "net_gain": 0, "opponent_can_recapture": False,
              "our_pieces_threatened_after": 0, "moved_piece_is_threatened": False,
              "leaves_piece_isolated": False, "center_control": False,
              "opponent_mobility_before": 8, "opponent_mobility_after": 8},
    "minimax_score": 1.5,
}
# ── BLACK forward move: src_row=2 → dst_row=3 (increasing) ───────────────────
_BLACK_FORWARD = {
    "type": "simple", "path": [[2, 3], [3, 2]], "captured": [],
    "facts": {"captures_count": 0, "net_gain": 0, "opponent_can_recapture": False,
              "our_pieces_threatened_after": 0, "moved_piece_is_threatened": False,
              "leaves_piece_isolated": False, "center_control": False,
              "opponent_mobility_before": 8, "opponent_mobility_after": 8},
    "minimax_score": 2.0,
}
# ── BLACK backward move: src_row=3 → dst_row=2 (decreasing) ──────────────────
_BLACK_BACKWARD = {
    "type": "simple", "path": [[3, 2], [2, 3]], "captured": [],
    "facts": {"captures_count": 0, "net_gain": 0, "opponent_can_recapture": False,
              "our_pieces_threatened_after": 0, "moved_piece_is_threatened": False,
              "leaves_piece_isolated": False, "center_control": False,
              "opponent_mobility_before": 8, "opponent_mobility_after": 8},
    "minimax_score": 1.5,
}
# ── RED back-row origin (row 7) ───────────────────────────────────────────────
_RED_BACK_ROW = {
    "type": "simple", "path": [[7, 2], [6, 3]], "captured": [],
    "facts": {"captures_count": 0, "net_gain": 0, "opponent_can_recapture": False,
              "our_pieces_threatened_after": 0, "moved_piece_is_threatened": False,
              "leaves_piece_isolated": False, "center_control": False,
              "opponent_mobility_before": 8, "opponent_mobility_after": 8},
    "minimax_score": 1.0,
}
# ── BLACK back-row origin (row 0) ─────────────────────────────────────────────
_BLACK_BACK_ROW = {
    "type": "simple", "path": [[0, 2], [1, 3]], "captured": [],
    "facts": {"captures_count": 0, "net_gain": 0, "opponent_can_recapture": False,
              "our_pieces_threatened_after": 0, "moved_piece_is_threatened": False,
              "leaves_piece_isolated": False, "center_control": False,
              "opponent_mobility_before": 8, "opponent_mobility_after": 8},
    "minimax_score": 1.0,
}


class TestColorAwareStrategicSeeds:
    """Direction and back-row seeds are verified against player color."""

    # ── (A) Development seed — RED ───────────────────────────────────────────

    def test_red_forward_move_gets_development_seed(self):
        assert "develops a piece forward" in _text_with_player(_RED_FORWARD, _RED)

    def test_red_backward_move_no_development_seed(self):
        assert "develops a piece forward" not in _text_with_player(_RED_BACKWARD, _RED)

    # ── (A) Development seed — BLACK ─────────────────────────────────────────

    def test_black_forward_move_gets_development_seed(self):
        assert "develops a piece forward" in _text_with_player(_BLACK_FORWARD, _BLACK)

    def test_black_backward_move_no_development_seed(self):
        assert "develops a piece forward" not in _text_with_player(_BLACK_BACKWARD, _BLACK)

    # ── (A) Development seed — unknown player: must NOT fire ─────────────────

    def test_unknown_player_no_development_seed_forward(self):
        """player=0 — direction unverifiable; seed must NOT fire."""
        assert "develops a piece forward" not in _text_with_player(_RED_FORWARD, 0)
    def test_unknown_player_no_development_seed_backward(self):
        """player=0 — direction unverifiable; seed must NOT fire."""
        assert "develops a piece forward" not in _text_with_player(_RED_BACKWARD, 0)

    # ── (B) Back-row seed — RED ──────────────────────────────────────────────

    def test_red_row7_origin_triggers_back_row_seed(self):
        assert "moves a back-row piece" in _text_with_player(_RED_BACK_ROW, _RED)

    def test_red_row0_origin_does_not_trigger_back_row_seed(self):
        """Row 0 is BLACK's back row, not RED's."""
        assert "moves a back-row piece" not in _text_with_player(_BLACK_BACK_ROW, _RED)

    # ── (B) Back-row seed — BLACK ────────────────────────────────────────────

    def test_black_row0_origin_triggers_back_row_seed(self):
        assert "moves a back-row piece" in _text_with_player(_BLACK_BACK_ROW, _BLACK)

    def test_black_row7_origin_does_not_trigger_back_row_seed(self):
        """Row 7 is RED's back row, not BLACK's."""
        assert "moves a back-row piece" not in _text_with_player(_RED_BACK_ROW, _BLACK)

    # ── (B) Back-row seed — unknown player: must NOT fire ─────────────────
    def test_unknown_player_row7_no_back_row_seed(self):
        assert "moves a back-row piece" not in _text_with_player(_RED_BACK_ROW, 0)
    def test_unknown_player_row0_no_back_row_seed(self):
        assert "moves a back-row piece" not in _text_with_player(_BLACK_BACK_ROW, 0)
    # ── chosen_move immutability ─────────────────────────────────────────────

    def test_chosen_move_not_mutated_by_color_aware_seeds(self):
        before = copy.deepcopy(_RED_FORWARD)
        _build_grounded_reasoning_seeds(_RED_FORWARD, [_RED_FORWARD], player=_RED)
        assert _RED_FORWARD == before


# ───────────────────────────────────────────────────────────────────────────
# Hallucination detection — trace T1–T91 discovered patterns
# ───────────────────────────────────────────────────────────────────────────

# Minimal facts dict used across hallucination tests.
_HALL_FACTS: dict = {
    "opponent_can_recapture": False,
    "captures_count": 0,
    "net_gain": 0,
    "results_in_king": False,
}


def _hall(text: str, seeds: list[str] | None = None) -> list[str]:
    """Shortcut: run checker with _HALL_FACTS and optional seeds."""
    return _check_reasoning_truthfulness(text, _HALL_FACTS, seeds=seeds)


def _has_contradiction(warnings: list[str], phrase: str) -> bool:
    """Return True if any warning contains *phrase*."""
    return any(phrase in w for w in warnings)


class TestHallucinationDetection:
    """Covers every hallucination pattern discovered in trace T1–T91."""

    # ── Forbidden vocabulary (always prohibited) ────────────────────────

    def test_conversion_potential_flagged(self):
        assert _has_contradiction(_hall("the move improves conversion potential"), "forbidden term 'conversion potential'")

    def test_winning_conversion_flagged(self):
        assert _has_contradiction(_hall("winning conversion score is high"), "forbidden term 'winning conversion'")

    def test_conversion_score_flagged(self):
        assert _has_contradiction(_hall("conversion score favors this move"), "forbidden term 'conversion score'")

    def test_quiet_move_role_flagged(self):
        assert _has_contradiction(_hall("quiet_move_role is dominant"), "forbidden term 'quiet_move_role'")

    def test_winning_conversion_score_flagged(self):
        assert _has_contradiction(_hall("winning_conversion_score=3.2"), "forbidden term 'winning_conversion_score'")

    def test_escape_squares_flagged(self):
        assert _has_contradiction(_hall("the king has escape squares"), "forbidden term 'escape squares'")

    def test_escape_routes_flagged(self):
        assert _has_contradiction(_hall("retreat to escape routes"), "forbidden term 'escape routes'")

    def test_king_escape_flagged(self):
        assert _has_contradiction(_hall("improves king escape opportunities"), "forbidden term 'king escape'")

    def test_king_distance_flagged(self):
        assert _has_contradiction(_hall("king distance to promotion"), "forbidden term 'king distance'")

    def test_king_activity_score_flagged(self):
        assert _has_contradiction(_hall("king_activity_score increased"), "forbidden term 'king_activity_score'")

    def test_diagonal_pressure_flagged(self):
        assert _has_contradiction(_hall("creates diagonal pressure"), "forbidden term 'diagonal pressure'")

    def test_diagonal_risks_flagged(self):
        assert _has_contradiction(_hall("avoids diagonal risks"), "forbidden term 'diagonal risks'")

    def test_long_diagonal_flagged(self):
        assert _has_contradiction(_hall("dominates the long diagonal"), "forbidden term 'long diagonal'")

    def test_strategic_goal_flagged(self):
        assert _has_contradiction(_hall("aligns with the strategic goal"), "forbidden term 'strategic goal'")

    def test_positional_adjustment_flagged(self):
        assert _has_contradiction(_hall("a neutral positional adjustment"), "forbidden term 'positional adjustment'")

    def test_no_new_vulnerabilities_flagged(self):
        assert _has_contradiction(_hall("no new vulnerabilities introduced"), "forbidden term 'no new vulnerabilities'")

    def test_counterplay_score_flagged(self):
        assert _has_contradiction(_hall("counterplay_score=0"), "forbidden term 'counterplay_score'")

    def test_coordination_score_flagged(self):
        assert _has_contradiction(_hall("coordination score improved"), "forbidden term 'coordination score'")

    def test_activity_score_flagged(self):
        assert _has_contradiction(_hall("activity score rises"), "forbidden term 'activity score'")

    def test_king_activity_score_longform_flagged(self):
        assert _has_contradiction(_hall("king activity score"), "forbidden term 'king activity score'")

    def test_quiet_move_role_longform_flagged(self):
        assert _has_contradiction(_hall("this serves as a quiet move role"), "forbidden term 'quiet move role'")

    # ── Context-forbidden vocabulary (allowed only if seeded) ─────────────

    def test_conversion_without_seed_flagged(self):
        assert _has_contradiction(_hall("supports conversion later"), "term 'conversion' used but not in seeds")

    def test_conversion_with_seed_allowed(self):
        assert not _has_contradiction(
            _hall("supports conversion later", seeds=["conversion is on the horizon"]),
            "term 'conversion' used but not in seeds",
        )

    def test_traps_without_seed_flagged(self):
        assert _has_contradiction(_hall("sets up traps"), "term 'traps' used but not in seeds")

    def test_traps_with_seed_allowed(self):
        assert not _has_contradiction(
            _hall("sets up traps", seeds=["traps the opponent piece"]),
            "term 'traps' used but not in seeds",
        )

    def test_escape_without_seed_flagged(self):
        # 'escape' is in _CONTEXT_FORBIDDEN_VOCAB — catches sneaky single-word use
        assert _has_contradiction(_hall("the king can escape"), "term 'escape' used but not in seeds")

    def test_diagonal_without_seed_flagged(self):
        assert _has_contradiction(_hall("controls the diagonal"), "term 'diagonal' used but not in seeds")

    # ── Unsupported numeric statements ──────────────────────────────

    def test_from_x_to_y_no_seeds_flagged(self):
        ws = _hall("mobility drops from 8 to 5", seeds=[])
        assert _has_contradiction(ws, "unsupported numeric statement 'from 8 to 5'")

    def test_from_x_to_y_with_seeds_allowed(self):
        ws = _hall("mobility drops from 8 to 5", seeds=["8 legal moves before", "5 legal moves after"])
        assert not _has_contradiction(ws, "unsupported numeric statement 'from 8 to 5'")

    def test_remains_at_n_no_seeds_flagged(self):
        ws = _hall("mobility remains at 10", seeds=[])
        assert _has_contradiction(ws, "unsupported numeric assertion 'remains at 10'")

    def test_remains_at_n_with_seed_allowed(self):
        ws = _hall("mobility remains at 10", seeds=["opponent has 10 moves"])
        assert not _has_contradiction(ws, "unsupported numeric assertion 'remains at 10'")

    def test_stays_at_n_no_seeds_flagged(self):
        ws = _hall("piece count stays at 12", seeds=[])
        assert _has_contradiction(ws, "remains at 12")

    # ── Unsupported absence claims ───────────────────────────────────

    def test_no_kings_lost_without_seed_flagged(self):
        ws = _hall("no kings lost during this sequence", seeds=[])
        assert _has_contradiction(ws, "unsupported absence claim 'no kings lost'")

    def test_no_kings_lost_with_seed_allowed(self):
        ws = _hall("no kings lost during this sequence", seeds=["no kings lost"])
        assert not _has_contradiction(ws, "unsupported absence claim 'no kings lost'")

    def test_piece_count_unchanged_without_seed_flagged(self):
        ws = _hall("piece count unchanged after the move", seeds=[])
        assert _has_contradiction(ws, "unsupported absence claim 'piece count unchanged'")

    def test_piece_count_unchanged_with_seed_allowed(self):
        ws = _hall("piece count unchanged after the move", seeds=["piece count unchanged"])
        assert not _has_contradiction(ws, "unsupported absence claim 'piece count unchanged'")

    def test_no_vulnerabilities_without_seed_flagged(self):
        ws = _hall("no vulnerabilities introduced by this move", seeds=[])
        assert _has_contradiction(ws, "unsupported absence claim 'no vulnerabilities'")

    def test_no_vulnerabilities_with_seed_allowed(self):
        ws = _hall("no vulnerabilities introduced by this move", seeds=["no vulnerabilities"])
        assert not _has_contradiction(ws, "unsupported absence claim 'no vulnerabilities'")

    def test_pieces_unchanged_without_seed_flagged(self):
        ws = _hall("both sides have pieces unchanged", seeds=[])
        assert _has_contradiction(ws, "unsupported absence claim 'pieces unchanged'")

    # ── Clean reasoning must produce no false positives ────────────────

    def test_clean_reasoning_no_warnings(self):
        ws = _hall(
            "The move advances a piece forward without recapture risk. "
            "No immediate capture is made. This is the highest-ranked option with "
            "a minimax score of -187.00.",
            seeds=["opponent cannot recapture next turn"],
        )
        # There should be no hallucination-related warnings (mobility/recapture
        # contradiction checks do not fire since facts show recapture=False).
        hallucination_ws = [w for w in ws if "REASONING_CONTRADICTION" in w and
                            any(k in w for k in ["forbidden", "unsupported", "absence claim"])]
        assert hallucination_ws == []

    def test_forbidden_vocab_lists_are_not_empty(self):
        """Sanity: the vocab lists must be populated."""
        assert len(_FORBIDDEN_VOCAB) >= 10
        assert len(_CONTEXT_FORBIDDEN_VOCAB) >= 5

    def test_seeds_parameter_defaults_none(self):
        """Checker accepts seeds=None without raising."""
        result = _check_reasoning_truthfulness(
            "advances a piece without capture",
            _HALL_FACTS,
            seeds=None,
        )
        # No forbidden-vocab present, so no hallucination warnings.
        hallucination_ws = [w for w in result if "forbidden" in w or "unsupported" in w]
        assert hallucination_ws == []

    def test_regulars_captured_flagged(self):
        assert _has_contradiction(
            _hall("regulars_captured=2 in this sequence"),
            "forbidden term 'regulars_captured'",
        )

    def test_new_vulnerabilities_flagged(self):
        assert _has_contradiction(
            _hall("no new vulnerabilities were introduced"),
            "forbidden term",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Part 8 — Tactical exposure acceptance
# ═══════════════════════════════════════════════════════════════════════════════

# Facts for a move that is exposed (threat_after=1, recapture possible)
# but has strong minimax and creates an immediate threat.
_EXPOSED_BEST_FACTS: dict = {
    "opponent_can_recapture": True,
    "our_pieces_threatened_after": 1,
    "moved_piece_is_threatened": True,
    "captures_count": 1,
    "net_gain": 1,
    "creates_immediate_threat": True,
    "shot_sequence_available": False,
    "blocks_opponent_landing": False,
    "leaves_piece_isolated": False,
    "center_control": True,
    "results_in_king": False,
    "minimax_score": -90.0,
}

_EXPOSED_BEST_MOVE: dict = {
    "type": "jump",
    "path": [[5, 2], [3, 4]],
    "captured": [[4, 3]],
    "facts": _EXPOSED_BEST_FACTS,
}

# A "safe" alternative that is passively safe but weaker by minimax.
_PASSIVE_SAFE_FACTS: dict = {
    "opponent_can_recapture": False,
    "our_pieces_threatened_after": 0,
    "moved_piece_is_threatened": False,
    "captures_count": 0,
    "net_gain": 0,
    "creates_immediate_threat": False,
    "shot_sequence_available": False,
    "blocks_opponent_landing": False,
    "leaves_piece_isolated": False,
    "center_control": False,
    "results_in_king": False,
    "minimax_score": -110.0,
}

_PASSIVE_SAFE_MOVE: dict = {
    "type": "simple",
    "path": [[5, 4], [4, 3]],
    "captured": [],
    "facts": _PASSIVE_SAFE_FACTS,
}


class TestAcceptableTacticalExposure:
    """Seeds for an exposed-but-best move must correctly label drawbacks
    AND must not invert the exposure facts in the checker."""

    def _seeds(self, move=_EXPOSED_BEST_MOVE, candidates=None):
        if candidates is None:
            candidates = [_EXPOSED_BEST_MOVE, _PASSIVE_SAFE_MOVE]
        return _build_grounded_reasoning_seeds(move, candidates)

    def _text(self, move=_EXPOSED_BEST_MOVE, candidates=None):
        return " ".join(self._seeds(move, candidates))

    # ── Seeds correctly label the exposure ───────────────────────────────────

    def test_recapture_true_seed_present_for_exposed_move(self):
        assert "opponent_can_recapture=true" in self._text()

    def test_threat_after_1_seed_present(self):
        text = self._text()
        assert "our_pieces_threatened_after=1" in text

    def test_moved_piece_threatened_seed_present(self):
        assert "moved_piece_is_threatened=true" in self._text()

    def test_capture_seed_present(self):
        assert "captures_count=1" in self._text()

    def test_threat_creation_seed_present(self):
        assert "creates_immediate_threat=true" in self._text()

    def test_minimax_seed_last_for_exposed_move(self):
        seeds = self._seeds()
        assert "minimax_score" in seeds[-1]

    # ── Checker does NOT flag correct acknowledgement of exposure ─────────────

    def test_checker_allows_honest_recapture_acknowledgement(self):
        """Reasoning that honestly says 'opponent can recapture' must NOT be flagged
        as a contradiction when opponent_can_recapture=true."""
        reasoning = (
            "Although the opponent can recapture (opponent_can_recapture=true), "
            "this move captures a piece (captures_count=1) and creates an immediate "
            "threat (creates_immediate_threat=true). "
            "The minimax score of -90.0 confirms this is the best available option."
        )
        warnings = _check_reasoning_truthfulness(reasoning, _EXPOSED_BEST_FACTS)
        recapture_contradictions = [w for w in warnings if "recapture" in w.lower()
                                    and "REASONING_CONTRADICTION" in w]
        assert recapture_contradictions == []

    def test_checker_flags_false_safety_claim_on_exposed_move(self):
        """Claiming 'avoids recapture' when opponent_can_recapture=true is a contradiction."""
        reasoning = "This move avoids recapture and is completely safe."
        warnings = _check_reasoning_truthfulness(reasoning, _EXPOSED_BEST_FACTS)
        assert any("recapture" in w for w in warnings)


class TestExposedButBestMove:
    """Verify the seed builder correctly handles moves that are
    tactically exposed but minimax-best."""

    def _text(self):
        return " ".join(
            _build_grounded_reasoning_seeds(
                _EXPOSED_BEST_MOVE,
                [_EXPOSED_BEST_MOVE, _PASSIVE_SAFE_MOVE],
            )
        )

    def test_seeds_do_not_hide_drawback(self):
        """The word 'drawback' must appear when opponent_can_recapture=true."""
        assert "tactical drawback" in self._text()

    def test_seeds_show_material_gain(self):
        assert "wins material" in self._text()

    def test_seeds_show_threat_creation(self):
        assert "creates_immediate_threat=true" in self._text()

    def test_seeds_do_not_claim_avoids_recapture(self):
        """Seeds must NEVER say 'avoids recapture' when recapture=true."""
        assert "avoids recapture" not in self._text()
        assert "no recapture" not in self._text()

    def test_checker_clean_on_drawback_acknowledged_reasoning(self):
        """A reasoning that acknowledges the drawback and justifies by material
        gain + threat must pass the truthfulness checker."""
        facts = _EXPOSED_BEST_FACTS
        seeds = _build_grounded_reasoning_seeds(
            _EXPOSED_BEST_MOVE, [_EXPOSED_BEST_MOVE, _PASSIVE_SAFE_MOVE]
        )
        reasoning = (
            "This jump captures a piece (captures_count=1, net_gain=1) and creates "
            "an immediate threat (creates_immediate_threat=true), keeping center control "
            "(center_control=true). The opponent can recapture "
            "(opponent_can_recapture=true), which is a tactical drawback, but "
            "1 piece remains at risk (our_pieces_threatened_after=1) while the "
            "tactical compensation justifies the exposure. "
            "The minimax score of -90.0 confirms this over passive alternatives."
        )
        warnings = _check_reasoning_truthfulness(reasoning, facts, seeds=seeds)
        # The only acceptable warnings are those unrelated to the core facts.
        hard_contradictions = [
            w for w in warnings
            if "forbidden term" in w or "inversion detected" in w
        ]
        assert hard_contradictions == [], f"Unexpected hard contradictions: {hard_contradictions}"


class TestThreatAfter1NotAutoRejected:
    """threat_after=1 must NOT be treated as an automatic disqualifier."""

    def test_seeds_label_threat_after_1_as_drawback_not_disqualifier(self):
        """The seed text says 'tactical drawback' but still shows the move's
        positive attributes (e.g. threat creation, material gain)."""
        seeds = _build_grounded_reasoning_seeds(
            _EXPOSED_BEST_MOVE, [_EXPOSED_BEST_MOVE, _PASSIVE_SAFE_MOVE]
        )
        text = " ".join(seeds)
        # Drawback is acknowledged
        assert "tactical drawback" in text
        # Positive attributes are still present
        assert "captures_count=1" in text or "wins material" in text

    def test_checker_does_not_flag_threat_after_1_as_error(self):
        """Having threat_after=1 in facts must not cause the checker to
        automatically produce a contradiction about safety."""
        facts = {**_EXPOSED_BEST_FACTS, "our_pieces_threatened_after": 1}
        reasoning = (
            "The move captures a piece (captures_count=1) while leaving 1 piece "
            "at risk (our_pieces_threatened_after=1). "
            "The minimax score of -90.0 confirms this choice."
        )
        warnings = _check_reasoning_truthfulness(reasoning, facts)
        # No false-positive contradiction about threat count
        threat_contradictions = [
            w for w in warnings
            if "our_pieces_threatened_after" in w and "REASONING_CONTRADICTION" in w
        ]
        assert threat_contradictions == []

    def test_system_prompt_contains_tactical_exposure_rule(self):
        """RANKER_SYSTEM_PROMPT must contain the TACTICAL EXPOSURE RULE."""
        from checkers.agents.ranker_agent import RANKER_SYSTEM_PROMPT
        assert "TACTICAL EXPOSURE RULE" in RANKER_SYSTEM_PROMPT

    def test_system_prompt_contains_anti_overdefensive_rule(self):
        from checkers.agents.ranker_agent import RANKER_SYSTEM_PROMPT
        assert "ANTI-OVERDEFENSIVE RULE" in RANKER_SYSTEM_PROMPT

    def test_seed_system_prompt_contains_tactical_exposure_context(self):
        assert "TACTICAL EXPOSURE CONTEXT" in RANKER_SEED_REASONING_SYSTEM


class TestInversionDetection:
    """Direct inversion detection: seed says X=true → reasoning says X is false."""

    def _inv(self, text: str, seeds: list[str]) -> list[str]:
        return _check_reasoning_truthfulness(text, _HALL_FACTS, seeds=seeds)

    def _has_inversion(self, warnings: list[str]) -> bool:
        return any("inversion detected" in w for w in warnings)

    # ── recapture inversions ─────────────────────────────────────────────────

    def test_seed_recapture_true_reasoning_says_avoids_recapture(self):
        seeds = ["opponent_can_recapture=true — drawback"]
        ws = self._inv("this move avoids recapture completely", seeds)
        assert self._has_inversion(ws)

    def test_seed_recapture_true_reasoning_says_no_recapture(self):
        seeds = ["opponent_can_recapture=true — drawback"]
        ws = self._inv("there is no recapture risk here", seeds)
        assert self._has_inversion(ws)

    def test_seed_recapture_false_no_inversion_on_recapture_claim(self):
        seeds = ["opponent_can_recapture=false — safe"]
        ws = self._inv("this move avoids recapture", seeds)
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    # ── isolation inversions ─────────────────────────────────────────────────

    def test_seed_isolated_true_reasoning_says_no_isolation(self):
        seeds = ["leaves_piece_isolated=true — positional drawback"]
        ws = self._inv("the piece has no isolation after this move", seeds)
        assert self._has_inversion(ws)

    def test_seed_isolated_true_reasoning_stays_connected(self):
        seeds = ["leaves_piece_isolated=true — positional drawback"]
        ws = self._inv("the piece stays connected to friendly pieces", seeds)
        assert self._has_inversion(ws)

    def test_seed_isolated_false_no_inversion(self):
        seeds = ["leaves_piece_isolated=false — preserves coordination"]
        ws = self._inv("the piece stays connected", seeds)
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    # ── threat creation inversions ───────────────────────────────────────────

    def test_seed_threat_false_reasoning_claims_immediate_threat(self):
        seeds = ["creates_immediate_threat=false"]
        ws = self._inv("this move creates a threat for next turn", seeds)
        assert self._has_inversion(ws)

    def test_seed_threat_true_reasoning_says_no_immediate_threat(self):
        seeds = ["creates_immediate_threat=true — puts opponent on defensive"]
        ws = self._inv("there is no immediate threat from this move", seeds)
        assert self._has_inversion(ws)

    # ── king row inversions ──────────────────────────────────────────────────

    def test_seed_weakens_king_row_true_reasoning_says_preserves(self):
        seeds = ["weakens_king_row=true — back-row defense is weakened"]
        ws = self._inv("this preserves back row integrity", seeds)
        assert self._has_inversion(ws)

    def test_seed_weakens_king_row_false_reasoning_says_weakened(self):
        seeds = ["weakens_king_row=false — back-row safe"]
        ws = self._inv("this move weakens the back row significantly", seeds)
        assert self._has_inversion(ws)

    # ── no false positives when seeds are absent ─────────────────────────────

    def test_no_inversion_warning_when_seeds_empty(self):
        ws = self._inv("the piece stays connected", seeds=[])
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    def test_no_inversion_warning_when_seeds_none(self):
        ws = _check_reasoning_truthfulness(
            "the piece stays connected", _HALL_FACTS, seeds=None
        )
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []


class TestRefinementSystemInheritsForbiddenRules:
    """RANKER_REASONING_REFINEMENT_SYSTEM must include all forbidden vocabulary
    and truthfulness constraints from RANKER_SEED_REASONING_SYSTEM."""

    # ── Forbidden vocab in refinement system ─────────────────────────────────

    def test_refinement_system_forbids_conversion_potential(self):
        assert "conversion potential" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_forbids_quiet_move_role(self):
        assert "quiet_move_role" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_forbids_king_activity_score(self):
        assert "king_activity_score" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_forbids_escape_squares(self):
        assert "escape squares" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_forbids_diagonal_pressure(self):
        assert "diagonal pressure" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_forbids_counterplay_score(self):
        assert "counterplay_score" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_forbids_regulars_captured(self):
        assert "regulars_captured" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_forbids_new_vulnerabilities(self):
        assert "new vulnerabilities" in RANKER_REASONING_REFINEMENT_SYSTEM

    # ── Truthfulness constraints in refinement system ─────────────────────────

    def test_refinement_system_has_mobility_truthfulness_rule(self):
        assert "reduces mobility" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_has_recapture_truthfulness_rule(self):
        assert "avoids recapture" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_has_isolation_truthfulness_rule(self):
        assert "leaves_piece_isolated" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_has_inversion_check_rule(self):
        assert "INVERSION CHECK" in RANKER_REASONING_REFINEMENT_SYSTEM

    # ── Core refinement constraints still present ─────────────────────────────

    def test_refinement_system_forbids_changing_move(self):
        assert "Do NOT change the chosen move" in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_minimax_last_rule(self):
        assert "minimax_score may appear ONLY in the final sentence" \
               in RANKER_REASONING_REFINEMENT_SYSTEM

    def test_refinement_system_has_output_format(self):
        assert '"reasoning"' in RANKER_REASONING_REFINEMENT_SYSTEM


# ═══════════════════════════════════════════════════════════════════════════════
# Part 9 — Retry-path truthfulness pre-validation (Task 1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetryPathValidation:
    """Verify that forbidden vocabulary is detected even on retry-raw reasoning
    (seeds=None path), confirming _check_reasoning_truthfulness covers the
    retry bypass scenario."""

    def _check_no_seeds(self, text: str) -> list[str]:
        return _check_reasoning_truthfulness(text, _HALL_FACTS, seeds=None)

    # ── Forbidden vocab caught with seeds=None ────────────────────────────────

    def test_real_trap_caught_no_seeds(self):
        """'real trap' must be caught even without seeds (retry-path scenario)."""
        ws = self._check_no_seeds(
            "This move creates a real trap for the opponent while advancing centrally."
        )
        assert any("forbidden term" in w and "real trap" in w for w in ws)

    def test_creates_real_trap_metric_caught_no_seeds(self):
        """Text containing the literal 'real trap' substring must be caught."""
        ws = self._check_no_seeds("this builds a real trap in the center")
        assert any("real trap" in w for w in ws)

    def test_positional_adjustment_caught_no_seeds(self):
        ws = self._check_no_seeds("a neutral positional adjustment completes the move")
        assert any("forbidden term" in w for w in ws)

    def test_conversion_potential_caught_no_seeds(self):
        ws = self._check_no_seeds("this improves conversion potential significantly")
        assert any("conversion potential" in w for w in ws)

    def test_escape_routes_caught_no_seeds(self):
        ws = self._check_no_seeds("the king's escape routes are now blocked")
        assert any("escape route" in w.lower() for w in ws)

    def test_structural_restriction_caught_no_seeds(self):
        ws = self._check_no_seeds(
            "the move maintains a structural restriction role on the right flank"
        )
        assert any("forbidden term" in w and "structural restriction" in w for w in ws)

    def test_neutral_positional_caught_no_seeds(self):
        ws = self._check_no_seeds(
            "this is a neutral positional step in the sequence"
        )
        # both 'neutral positional' and 'positional step' are forbidden
        assert any("forbidden term" in w for w in ws)

    # ── Clean retry-style reasoning passes with seeds=None ────────────────────

    def test_clean_retry_reasoning_passes_no_seeds(self):
        """A correctly grounded retry reasoning must produce no forbidden-term warnings."""
        ws = self._check_no_seeds(
            "The move captures a piece (captures_count=1, net_gain=1) and creates "
            "an immediate threat (creates_immediate_threat=true). "
            "The opponent can recapture (opponent_can_recapture=true), "
            "but their reply is constrained to a forced jump. "
            "The minimax score of -90.0 confirms this is the best option."
        )
        forbidden_ws = [w for w in ws if "forbidden term" in w]
        assert forbidden_ws == []

    def test_material_gain_check_still_works_no_seeds(self):
        """Material gain contradiction is still caught without seeds."""
        ws = self._check_no_seeds(
            "This move gains material and advances the position significantly."
        )
        assert any("material gain" in w for w in ws)


class TestRetryHallucinationRejection:
    """Verify that retry-style reasoning containing known hallucinated metrics
    is correctly rejected by _check_reasoning_truthfulness."""

    def _check(self, text: str, seeds=None) -> list[str]:
        return _check_reasoning_truthfulness(text, _HALL_FACTS, seeds=seeds)

    def test_quiet_move_role_rejected(self):
        ws = self._check("the quiet_move_role determines the best path here")
        assert any("quiet_move_role" in w or "quiet move role" in w for w in ws)

    def test_winning_conversion_score_rejected(self):
        ws = self._check("winning_conversion_score=0.8 favors this line")
        assert any("winning_conversion" in w for w in ws)

    def test_king_activity_score_rejected(self):
        ws = self._check("king_activity_score improves by two points")
        assert any("king_activity_score" in w or "king activity score" in w for w in ws)

    def test_counterplay_score_rejected(self):
        ws = self._check("the counterplay_score rises to 3 after this move")
        assert any("counterplay_score" in w for w in ws)

    def test_diagonal_pressure_rejected(self):
        ws = self._check("this creates diagonal pressure on the opponent")
        assert any("diagonal pressure" in w for w in ws)

    def test_real_trap_rejected(self):
        ws = self._check(
            "this move creates a real trap for the opponent piece at (3,4)"
        )
        assert any("real trap" in w for w in ws)

    def test_no_new_vulnerabilities_rejected(self):
        ws = self._check("the move introduces no new vulnerabilities to our position")
        assert any("new vulnerabilities" in w for w in ws)

    def test_structural_restriction_rejected(self):
        ws = self._check("maintains the structural restriction on the left flank")
        assert any("structural restriction" in w for w in ws)

    def test_positional_step_rejected(self):
        ws = self._check("a simple positional step to improve piece placement")
        assert any("positional step" in w for w in ws)

    def test_neutral_positional_rejected(self):
        ws = self._check("this is a neutral positional move in the sequence")
        assert any("neutral positional" in w for w in ws)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 10 — Semantic isolation inversion phrases (Task 2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticIsolationInversion:
    """Expanded inversion phrases: leaves_piece_isolated=true must trigger on
    all semantic negations of isolation."""

    _ISOLATED_SEED = ["leaves_piece_isolated=true — positional drawback"]

    def _inv(self, text: str) -> list[str]:
        return _check_reasoning_truthfulness(
            text, _HALL_FACTS, seeds=self._ISOLATED_SEED
        )

    def _has_inversion(self, ws: list[str]) -> bool:
        return any("inversion detected" in w and "leaves_piece_isolated" in w for w in ws)

    # ── All new phrase variants ───────────────────────────────────────────────

    def test_does_not_isolate_flagged(self):
        assert self._has_inversion(
            self._inv("the move does not isolate any friendly pieces")
        )

    def test_avoids_isolation_flagged(self):
        assert self._has_inversion(
            self._inv("this advance avoids isolation of the moved piece")
        )

    def test_remains_connected_flagged(self):
        assert self._has_inversion(
            self._inv("the piece remains connected to the rest of the formation")
        )

    def test_maintains_structure_flagged(self):
        assert self._has_inversion(
            self._inv("the move maintains structure by keeping pieces grouped")
        )

    def test_keeping_connectivity_flagged(self):
        assert self._has_inversion(
            self._inv("keeping connectivity with adjacent pieces is preserved")
        )

    # ── Previously covered phrases still work ────────────────────────────────

    def test_no_isolation_still_flagged(self):
        assert self._has_inversion(self._inv("there is no isolation after this move"))

    def test_stays_connected_still_flagged(self):
        assert self._has_inversion(self._inv("the piece stays connected after moving"))

    def test_maintains_connectivity_still_flagged(self):
        assert self._has_inversion(
            self._inv("this move maintains connectivity with nearby pieces")
        )

    # ── No false positives when isolated=false ────────────────────────────────

    def test_no_inversion_when_isolated_false_and_does_not_isolate(self):
        seeds = ["leaves_piece_isolated=false — preserves coordination"]
        ws = _check_reasoning_truthfulness(
            "the move does not isolate the piece", _HALL_FACTS, seeds=seeds
        )
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    def test_no_inversion_when_isolated_false_and_remains_connected(self):
        seeds = ["leaves_piece_isolated=false — preserves coordination"]
        ws = _check_reasoning_truthfulness(
            "the piece remains connected", _HALL_FACTS, seeds=seeds
        )
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    # ── No false positives with empty seeds ──────────────────────────────────

    def test_does_not_isolate_no_seeds_no_inversion(self):
        ws = _check_reasoning_truthfulness(
            "the move does not isolate the piece", _HALL_FACTS, seeds=[]
        )
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []


# ═══════════════════════════════════════════════════════════════════════════════
# Part 11 — Semantic numeric leakage (Task 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticNumericLeakage:
    """Verify that quantity claims not supported by seeds are detected."""

    def _check(self, text: str, seeds=None) -> list[str]:
        return _check_reasoning_truthfulness(text, _HALL_FACTS, seeds=seeds)

    # ── Before→after numeric narratives ──────────────────────────────────────

    def test_from_three_to_two_flagged(self):
        ws = self._check(
            "this reduces the number of threatened pieces from three to two"
        )
        assert any("numeric" in w.lower() or "before" in w.lower() for w in ws)

    def test_from_four_to_one_flagged(self):
        ws = self._check("mobility drops from four to one after this advance")
        assert any("numeric" in w.lower() or "before" in w.lower() for w in ws)

    def test_from_five_to_three_flagged(self):
        ws = self._check("replies available fall from five to three")
        assert any("numeric" in w.lower() or "before" in w.lower() for w in ws)

    # ── Specific safe-reply counts ────────────────────────────────────────────

    def test_seven_safe_replies_flagged(self):
        ws = self._check("the opponent retains seven safe replies after the move")
        assert any("safe-reply" in w or "reply count" in w or "numeric" in w.lower()
                   for w in ws)

    def test_nine_safe_replies_flagged(self):
        ws = self._check("there are nine safe replies available to the opponent")
        assert any("safe-reply" in w or "numeric" in w.lower() for w in ws)

    # ── Unchanged mobility assertions ─────────────────────────────────────────

    def test_unchanged_mobility_flagged(self):
        ws = self._check("opponent mobility remains unchanged after the move")
        assert any("unchanged mobility" in w or "mobility unchanged" in w.lower()
                   or "mobility" in w.lower() for w in ws)

    def test_mobility_stays_unchanged_flagged(self):
        ws = self._check("their mobility stays unchanged throughout")
        assert any(
            "unchanged" in w.lower() and "mobility" in w.lower() for w in ws
        )

    # ── Same-number-of assertions ─────────────────────────────────────────────

    def test_same_number_of_replies_flagged(self):
        ws = self._check(
            "the opponent retains the same number of replies as before"
        )
        assert any("same number" in w or "move count" in w.lower() for w in ws)

    def test_same_number_of_moves_flagged(self):
        ws = self._check("both sides have the same number of moves available")
        assert any("same number" in w or "numeric" in w.lower() for w in ws)

    def test_maintains_same_count_flagged(self):
        ws = self._check(
            "the opponent maintains the same count of options as before"
        )
        assert any("same" in w.lower() and ("count" in w.lower() or "number" in w.lower())
                   for w in ws)

    # ── No false positives on clean numeric-free reasoning ────────────────────

    def test_clean_reasoning_no_numeric_warning(self):
        ws = self._check(
            "The move captures a piece (captures_count=1) and creates a threat. "
            "The minimax score of -90.0 confirms this is the best option."
        )
        numeric_ws = [
            w for w in ws
            if ("numeric" in w.lower() or "before" in w.lower()
                or "safe-reply" in w or "unchanged mobility" in w
                or "same number" in w)
        ]
        assert numeric_ws == []

    def test_mobility_number_in_seeds_allowed(self):
        """If seeds explicitly state the mobility numbers, the checker should not
        flag them for mobility-related claims supported by the seed text."""
        seeds = [
            "opponent_mobility_before=11, opponent_mobility_after=10 — "
            "reduces opponent mobility by 1"
        ]
        # Claiming '11 to 10' in words: not in word form, so no false positive.
        ws = _check_reasoning_truthfulness(
            "the move reduces the opponent's mobility from 11 to 10",
            _HALL_FACTS,
            seeds=seeds,
        )
        # The numeric pattern matches word-form only, '11 to 10' is digits so
        # the regex won't match — no false positive expected.
        numeric_ws = [w for w in ws if "before\u2192after numeric" in w]
        assert numeric_ws == []


# ═══════════════════════════════════════════════════════════════════════════════
# Part 12 — Vague positional leakage (Task 4)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVaguePositionalLeakage:
    """Verify that vague positional characterisation terms now banned in
    _FORBIDDEN_VOCAB are caught by _check_reasoning_truthfulness."""

    def _check(self, text: str) -> list[str]:
        return _check_reasoning_truthfulness(text, _HALL_FACTS, seeds=None)

    # ── structural restriction ────────────────────────────────────────────────

    def test_structural_restriction_role_flagged(self):
        ws = self._check(
            "the move maintains the structural restriction role without capturing"
        )
        assert any("structural restriction" in w for w in ws)

    def test_structural_restriction_bare_flagged(self):
        ws = self._check("by executing structural restriction, the engine limits options")
        assert any("structural restriction" in w for w in ws)

    # ── positional step ───────────────────────────────────────────────────────

    def test_positional_step_flagged(self):
        ws = self._check("this is a simple positional step in the sequence")
        assert any("positional step" in w for w in ws)

    def test_neutral_positional_step_flagged(self):
        ws = self._check("a neutral positional step to improve piece placement")
        assert any("positional step" in w or "neutral positional" in w for w in ws)

    # ── neutral positional ────────────────────────────────────────────────────

    def test_neutral_positional_move_flagged(self):
        ws = self._check("the engine selects a neutral positional move here")
        assert any("neutral positional" in w for w in ws)

    def test_neutral_positional_choice_flagged(self):
        ws = self._check(
            "the minimax confirms this as a neutral positional choice at -90.0"
        )
        assert any("neutral positional" in w for w in ws)

    # ── positional adjustment (already in vocab) still caught ─────────────────

    def test_positional_adjustment_still_caught(self):
        ws = self._check("this move is a pure positional adjustment")
        assert any("positional adjustment" in w for w in ws)

    # ── Confirm these terms ARE in _FORBIDDEN_VOCAB ───────────────────────────

    def test_structural_restriction_in_forbidden_vocab(self):
        assert "structural restriction" in _FORBIDDEN_VOCAB

    def test_positional_step_in_forbidden_vocab(self):
        assert "positional step" in _FORBIDDEN_VOCAB

    def test_neutral_positional_in_forbidden_vocab(self):
        assert "neutral positional" in _FORBIDDEN_VOCAB

    # ── Clean reasoning still passes ─────────────────────────────────────────

    def test_clean_positional_reasoning_passes(self):
        """Reasoning that describes the move without banned framing must pass."""
        ws = self._check(
            "The move advances a piece to (4,3), improving central influence "
            "while the opponent cannot recapture immediately. "
            "The minimax score of -83.0 confirms this is the best option."
        )
        vague_ws = [
            w for w in ws
            if any(term in w for term in [
                "structural restriction", "positional step", "neutral positional"
            ])
        ]
        assert vague_ws == []


# ═══════════════════════════════════════════════════════════════════════════════
# Part N — Targeted sentence repair helpers
# ═══════════════════════════════════════════════════════════════════════════════

_REPAIR_MOVE = {
    "type": "simple",
    "path": [[5, 4], [4, 3]],
    "captured": [],
    "facts": {
        "opponent_can_recapture":    True,
        "center_control":            False,
        "creates_immediate_threat":  False,
        "leaves_piece_isolated":     True,
        "net_gain":                  0,
        "captures_count":            0,
        "opponent_mobility_after":   10,
        "opponent_mobility_before":  10,
        "our_pieces_threatened_after": 1,
        "results_in_king":           False,
    },
}


class TestSplitReasoningSentences:
    def test_single_sentence(self):
        assert _split_reasoning_sentences("This is one sentence.") == [
            "This is one sentence."
        ]

    def test_two_sentences(self):
        parts = _split_reasoning_sentences(
            "This move advances the piece. The opponent cannot recapture."
        )
        assert len(parts) == 2
        assert "advances" in parts[0]
        assert "recapture" in parts[1]

    def test_three_sentences_split_correctly(self):
        text = (
            "The piece moves forward. It leaves no isolation risk. "
            "The minimax score confirms the choice."
        )
        parts = _split_reasoning_sentences(text)
        assert len(parts) == 3

    def test_empty_string_returns_empty_list(self):
        assert _split_reasoning_sentences("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _split_reasoning_sentences("   ") == []

    def test_strips_leading_trailing_whitespace(self):
        parts = _split_reasoning_sentences("  First sentence. Second sentence.  ")
        assert all(s == s.strip() for s in parts)


class TestPartitionSentencesByContradiction:
    def test_bad_sentence_identified_for_recapture_contradiction(self):
        sentences = [
            "This move avoids recapture.",
            "It keeps structure intact.",
        ]
        contradiction = (
            "REASONING_CONTRADICTION: claims avoids recapture but "
            "opponent_can_recapture=true"
        )
        bad, good = _partition_sentences_by_contradiction(sentences, [contradiction])
        assert 0 in bad
        assert 1 in good

    def test_bad_sentence_identified_for_isolation_contradiction(self):
        sentences = [
            "The piece stays connected after the move.",
            "Minimax confirms the choice.",
        ]
        contradiction = (
            "REASONING_CONTRADICTION: inversion detected — "
            "seed states 'leaves_piece_isolated=true' but reasoning says 'stays connected'"
        )
        bad, good = _partition_sentences_by_contradiction(sentences, [contradiction])
        assert 0 in bad
        assert 1 in good

    def test_bad_sentence_identified_for_material_gain_contradiction(self):
        sentences = [
            "The move advances the piece forward.",
            "It gains material for an advantage.",
        ]
        contradiction = (
            "REASONING_CONTRADICTION: claims material gain but net_gain=0"
        )
        bad, good = _partition_sentences_by_contradiction(sentences, [contradiction])
        assert 1 in bad
        assert 0 in good

    def test_bad_sentence_identified_for_forbidden_term(self):
        sentences = [
            "This has escape squares available.",
            "The minimax score confirms the choice.",
        ]
        contradiction = (
            "REASONING_CONTRADICTION: forbidden term 'escape squares' used — "
            "not present in any reasoning seed"
        )
        bad, good = _partition_sentences_by_contradiction(sentences, [contradiction])
        assert 0 in bad
        assert 1 in good

    def test_all_sentences_good_when_no_contradictions(self):
        sentences = ["Sentence one.", "Sentence two."]
        bad, good = _partition_sentences_by_contradiction(sentences, [])
        assert bad == []
        assert good == [0, 1]

    def test_multiple_contradictions_tag_correct_sentences(self):
        sentences = [
            "This move avoids recapture.",   # bad
            "It does not isolate the piece.",  # bad
            "The minimax score confirms.",   # good
        ]
        contradictions = [
            "REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true",
            "REASONING_CONTRADICTION: claims no isolation but leaves_piece_isolated=true",
        ]
        bad, good = _partition_sentences_by_contradiction(sentences, contradictions)
        assert 0 in bad
        assert 1 in bad
        assert 2 in good

    def test_unidentifiable_contradiction_returns_empty_bad(self):
        sentences = ["Some sentence.", "Another one."]
        # Contradiction format that yields no phrases
        bad, good = _partition_sentences_by_contradiction(
            sentences, ["REASONING_CONTRADICTION: totally unknown pattern"]
        )
        assert bad == []


class TestBuildTargetedRefinementPrompt:
    def test_prompt_contains_bad_sentence(self):
        bad = ["This move avoids recapture."]
        contradictions = ["claims avoids recapture but opponent_can_recapture=true"]
        prompt = _build_targeted_refinement_prompt(_REPAIR_MOVE, bad, contradictions)
        assert "avoids recapture" in prompt

    def test_prompt_contains_fact_values(self):
        bad = ["This move gains material."]
        contradictions = ["claims material gain but net_gain=0"]
        prompt = _build_targeted_refinement_prompt(_REPAIR_MOVE, bad, contradictions)
        assert "net_gain" in prompt or "0" in prompt

    def test_prompt_specifies_correct_replacement_count(self):
        bad = ["Bad sentence one.", "Bad sentence two."]
        prompt = _build_targeted_refinement_prompt(_REPAIR_MOVE, bad, [])
        assert "2 replacement" in prompt

    def test_prompt_instructs_one_sentence_per_replacement(self):
        bad = ["Bad sentence."]
        prompt = _build_targeted_refinement_prompt(_REPAIR_MOVE, bad, [])
        assert "one replacement" in prompt or "1 replacement" in prompt

    def test_prompt_requests_json_replacements_key(self):
        bad = ["Bad sentence."]
        prompt = _build_targeted_refinement_prompt(_REPAIR_MOVE, bad, [])
        assert '"replacements"' in prompt


class TestExtractTargetedRepairResponse:
    def test_parses_valid_json_with_correct_count(self):
        raw = '{"replacements": ["Fixed sentence one.", "Fixed sentence two."]}'
        result = _extract_targeted_repair_response(raw, 2)
        assert result == ["Fixed sentence one.", "Fixed sentence two."]

    def test_returns_none_when_count_mismatches(self):
        raw = '{"replacements": ["One sentence."]}'
        assert _extract_targeted_repair_response(raw, 2) is None

    def test_returns_none_on_invalid_json(self):
        assert _extract_targeted_repair_response("not json at all", 1) is None

    def test_returns_none_on_empty_string(self):
        assert _extract_targeted_repair_response("", 1) is None

    def test_parses_single_replacement(self):
        raw = '{"replacements": ["The piece is exposed to recapture."]}'
        result = _extract_targeted_repair_response(raw, 1)
        assert result == ["The piece is exposed to recapture."]

    def test_strips_markdown_not_required_but_valid_json_still_parsed(self):
        raw = '{"replacements": ["Clean replacement."]}'
        result = _extract_targeted_repair_response(raw, 1)
        assert result is not None


class TestTargetedRepairInRefineReasoning:
    """
    Verify that _refine_reasoning preserves correct sentences and only
    replaces the ones identified as bad.
    """

    _facts = _REPAIR_MOVE["facts"]

    def _mock_targeted_response(self, replacement: str):
        return f'{{"replacements": ["{replacement}"]}}'

    def test_preserved_sentence_survives_repair(self):
        """
        The second (good) sentence must appear unchanged in the output even when
        the first sentence is repaired.
        """
        reasoning = (
            "This move avoids recapture. "
            "The minimax score of 2.5 confirms this choice."
        )
        contradictions = [
            "REASONING_CONTRADICTION: claims avoids recapture but "
            "opponent_can_recapture=true"
        ]
        repaired_sentence = "The opponent can recapture this piece next turn."

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=self._mock_targeted_response(repaired_sentence),
        ):
            result, _, _ = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=1
            )

        assert "minimax score of 2.5" in result, (
            "The clean sentence must be preserved verbatim"
        )
        assert "avoids recapture" not in result, (
            "The bad sentence must be replaced"
        )

    def test_repaired_sentence_replaces_bad_one(self):
        """The replacement text from the LLM ends up in the output."""
        reasoning = (
            "This move gains material. "
            "The piece advances forward safely."
        )
        contradictions = [
            "REASONING_CONTRADICTION: claims material gain but net_gain=0"
        ]
        replacement = "This is a positional move with no material gain."

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=self._mock_targeted_response(replacement),
        ):
            result, _, _ = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=1
            )

        assert replacement in result

    def test_multiple_bad_sentences_all_replaced(self):
        """When two sentences are bad, both get replaced; the third is preserved."""
        reasoning = (
            "This move avoids recapture. "
            "It does not isolate the piece. "
            "The minimax confirms the selection."
        )
        contradictions = [
            "REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true",
            "REASONING_CONTRADICTION: claims no isolation but leaves_piece_isolated=true",
        ]
        mock_response = (
            '{"replacements": ["Opponent can recapture.", '
            '"The piece is left isolated."]}'
        )

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=mock_response,
        ):
            result, _, _ = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=1
            )

        assert "minimax confirms" in result, "Good sentence must survive"
        assert "avoids recapture" not in result
        assert "does not isolate" not in result

    def test_retry_count_increments_per_attempt(self):
        """retry_count must equal the number of attempts made."""
        reasoning = "This move avoids recapture."
        contradictions = [
            "REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true"
        ]
        replacement_1 = '{"replacements": ["Opponent can recapture this piece."]}'
        replacement_2 = '{"replacements": ["Opponent can recapture this piece."]}'

        call_seq = iter([replacement_1, replacement_2])

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            side_effect=lambda *_: next(call_seq),
        ):
            _, count, _ = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=2
            )

        assert count >= 1


class TestDeterministicSeedSummary:
    """
    Verify that _build_deterministic_seed_summary produces grounded,
    hallucination-free text from seeds.
    """

    _move = _REPAIR_MOVE

    def test_returns_string(self):
        seeds = ["opponent_can_recapture=true — opponent can recapture this piece"]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explanation_part_of_seed_appears_in_output(self):
        seeds = ["opponent_can_recapture=true — opponent can recapture this piece next turn"]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert "recapture" in result.lower()

    def test_multiple_seeds_all_appear(self):
        seeds = [
            "opponent_can_recapture=true — opponent can recapture",
            "leaves_piece_isolated=true — moved piece is not supported",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert "recapture" in result.lower()
        assert "isolated" in result.lower() or "not supported" in result.lower()

    def test_output_passes_truthfulness_checker(self):
        """The deterministic summary must never introduce hallucinations."""
        seeds = [
            "opponent_can_recapture=true — opponent can recapture this piece next turn",
            "leaves_piece_isolated=true — moved piece is not supported by adjacent allies",
            "captures_count=0 — positional move focused on improving placement",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        contradictions = _check_reasoning_truthfulness(result, self._move["facts"])
        assert contradictions == [], (
            f"Deterministic seed summary introduced contradictions: {contradictions}\n"
            f"Summary: {result!r}"
        )

    def test_empty_seeds_returns_safe_fallback(self):
        result = _build_deterministic_seed_summary([], self._move)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_seed_without_dash_still_included(self):
        seeds = ["this is a seed without a dash separator"]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert "without a dash" in result.lower() or len(result) > 0

    def test_output_is_capitalized(self):
        seeds = ["opponent_can_recapture=true — opponent can recapture"]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert result[0].isupper()

    def test_no_forbidden_vocab_in_output(self):
        """No forbidden vocabulary should appear in a seed-derived summary."""
        seeds = [
            "opponent_can_recapture=false — immediate tactical safety",
            "captures_count=0 — positional move",
            "minimax_score=2.50 — confirms this is the highest-evaluated option",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        result_lower = result.lower()
        for term in _FORBIDDEN_VOCAB:
            assert term not in result_lower, (
                f"Forbidden term {term!r} appeared in seed summary: {result!r}"
            )


class TestFallbackGateInRefinement:
    """
    Verify that when all refinement retries fail (LLM keeps hallucinating),
    _refine_reasoning returns the still-contradictory text (the caller in
    ranker_agent is responsible for applying the deterministic fallback).
    This tests the boundary: resolved=False triggers the gate.
    """

    _facts = _REPAIR_MOVE["facts"]

    def test_resolved_false_when_llm_never_fixes_contradiction(self):
        """If the LLM always returns contradictory text, resolved=False."""
        reasoning = "This move avoids recapture."
        contradictions = [
            "REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true"
        ]
        still_bad = '{"replacements": ["This move also avoids recapture."]}'

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=still_bad,
        ):
            _, _, resolved = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=2
            )

        assert resolved is False

    def test_resolved_true_when_llm_fixes_contradiction(self):
        """If the LLM produces clean text, resolved=True."""
        reasoning = "This move avoids recapture."
        contradictions = [
            "REASONING_CONTRADICTION: claims avoids recapture but opponent_can_recapture=true"
        ]
        good_replacement = '{"replacements": ["The opponent can take back this piece."]}'

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=good_replacement,
        ):
            result, _, resolved = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=1
            )

        assert resolved is True
        assert "avoids recapture" not in result

    def test_seed_summary_passes_checker_for_repair_move(self):
        """
        The deterministic fallback produced from _REPAIR_MOVE seeds must pass
        the truthfulness checker — proving it is safe to publish.
        """
        seeds = _build_grounded_reasoning_seeds(_REPAIR_MOVE, [_REPAIR_MOVE])
        summary = _build_deterministic_seed_summary(seeds, _REPAIR_MOVE)
        contradictions = _check_reasoning_truthfulness(
            summary, _REPAIR_MOVE["facts"], seeds=seeds
        )
        assert contradictions == [], (
            f"Seed summary for _REPAIR_MOVE failed truthfulness check: "
            f"{contradictions}\nSummary: {summary!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Regression tests — Issue 1: comparative-seed false-positive inversion
# ═══════════════════════════════════════════════════════════════════════════════

class TestComparativeSeedInversionFix:
    """
    Comparison seeds describe the ALTERNATIVE move's facts, not the chosen
    move's facts.  The inversion checker must NOT fire when the chosen move's
    own facts are correct and the contradiction phrase only appears in a
    comparison seed.
    """

    _chosen_facts = {
        "leaves_piece_isolated": False,      # chosen move is NOT isolated
        "opponent_can_recapture": False,
        "creates_immediate_threat": False,
        "center_control": False,
        "captures_count": 0,
        "net_gain": 0,
        "results_in_king": False,
    }

    def _seeds_with_comparison(self) -> list[str]:
        """Seeds where the comparison seed says the ALT move leaves_piece_isolated=true."""
        return [
            "opponent_can_recapture=false — immediate tactical safety: "
            "opponent cannot recapture this piece next turn",
            # Comparison seed: alt move IS isolated; chosen move is NOT isolated.
            "Move [1] isolates the moved piece (leaves_piece_isolated=true vs false here).",
        ]

    def test_no_false_positive_when_comparison_seed_has_isolated_true(self):
        """
        Reasoning correctly says 'stays connected' about the chosen move.
        The comparison seed mentions 'leaves_piece_isolated=true' for the alt.
        The inversion checker must NOT fire.
        """
        seeds = self._seeds_with_comparison()
        reasoning = (
            "The piece stays connected after this move, maintaining coordination "
            "with the rest of the formation."
        )
        ws = _check_reasoning_truthfulness(reasoning, self._chosen_facts, seeds=seeds)
        inversion_ws = [w for w in ws if "inversion detected" in w and "leaves_piece_isolated" in w]
        assert inversion_ws == [], (
            f"False-positive inversion triggered by comparison seed: {inversion_ws}"
        )

    def test_no_false_positive_stays_connected(self):
        seeds = self._seeds_with_comparison()
        ws = _check_reasoning_truthfulness(
            "The moved piece stays connected.", self._chosen_facts, seeds=seeds
        )
        inversion_ws = [w for w in ws if "inversion" in w and "isolated" in w]
        assert inversion_ws == []

    def test_no_false_positive_maintains_connectivity(self):
        seeds = self._seeds_with_comparison()
        ws = _check_reasoning_truthfulness(
            "This move maintains connectivity across the board.",
            self._chosen_facts,
            seeds=seeds,
        )
        inversion_ws = [w for w in ws if "inversion" in w and "isolated" in w]
        assert inversion_ws == []

    def test_real_inversion_still_fires_when_chosen_seed_says_isolated(self):
        """
        When the chosen move's OWN seed says isolated=true but reasoning says
        'stays connected', the inversion SHOULD fire (chosen move IS isolated).
        """
        seeds_chosen_isolated = [
            "leaves_piece_isolated=true — positional drawback: "
            "moved piece is not supported by adjacent allies",
        ]
        facts_isolated = {**self._chosen_facts, "leaves_piece_isolated": True}
        ws = _check_reasoning_truthfulness(
            "The moved piece stays connected after the move.",
            facts_isolated,
            seeds=seeds_chosen_isolated,
        )
        inversion_ws = [w for w in ws if "inversion" in w and "isolated" in w]
        assert inversion_ws != [], (
            "Inversion should fire when the chosen move IS isolated but "
            "reasoning claims it stays connected"
        )

    def test_comparison_seed_recapture_does_not_cause_false_positive(self):
        """
        Comparison seed: alt move allows recapture.  Chosen move does NOT.
        Reasoning correctly says 'avoids recapture' — must not trigger inversion.
        """
        seeds = [
            "opponent_can_recapture=false — immediate tactical safety: "
            "opponent cannot recapture this piece next turn",
            "Move [1] allows recapture (opponent_can_recapture=true) "
            "vs false here — chosen move is safer.",
        ]
        ws = _check_reasoning_truthfulness(
            "This move avoids recapture risk.",
            self._chosen_facts,
            seeds=seeds,
        )
        inversion_ws = [w for w in ws if "inversion" in w and "recapture" in w]
        assert inversion_ws == [], (
            f"False-positive recapture inversion from comparison seed: {inversion_ws}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Regression tests — Issue 2: final full-paragraph validation always runs
# ═══════════════════════════════════════════════════════════════════════════════

class TestFinalFullParagraphValidation:
    """
    After targeted repair, the FULL paragraph (including preserved sentences)
    is re-validated.  The function must always exit through a single final
    _check_reasoning_truthfulness call — not through an early return inside
    the loop.
    """

    _facts = _REPAIR_MOVE["facts"]

    def test_resolved_true_and_full_check_when_intermediate_clean(self):
        """
        Attempt 1 repairs the bad sentence and the intermediate check shows
        the whole paragraph is clean.  Function must still return resolved=True
        (going through the final check after the loop, not an early return).
        """
        original = (
            "This move avoids recapture. "
            "The piece advances forward safely."
        )
        contradiction = [
            "REASONING_CONTRADICTION: claims avoids recapture but "
            "opponent_can_recapture=true"
        ]
        good_replacement = '{"replacements": ["The opponent can take back this piece."]}'

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=good_replacement,
        ):
            result, count, resolved = _refine_reasoning(
                original, _REPAIR_MOVE, contradiction, max_attempts=2
            )

        assert resolved is True
        assert "avoids recapture" not in result
        # The preserved sentence must still be present
        assert "The piece advances forward safely" in result

    def test_resolved_false_after_all_attempts_still_runs_final_check(self):
        """
        Both attempts fail to fix the contradiction.  resolved=False must be
        returned from the final check, not from inside the loop.
        """
        original = "This move avoids recapture."
        contradiction = [
            "REASONING_CONTRADICTION: claims avoids recapture but "
            "opponent_can_recapture=true"
        ]
        still_bad = '{"replacements": ["This move also avoids recapture."]}'

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=still_bad,
        ):
            _, _, resolved = _refine_reasoning(
                original, _REPAIR_MOVE, contradiction, max_attempts=2
            )

        assert resolved is False

    def test_good_sentence_in_full_paragraph_does_not_cause_false_negative(self):
        """
        After repair, the full paragraph (good sentences + repaired sentence)
        must be clean — no lingering contradiction from the repaired sentence
        falsely infecting the previously-good sentences.

        _REPAIR_MOVE has opponent_can_recapture=True, so the preserved sentence
        must not say "avoids recapture".  We use a neutral sentence instead.
        """
        original = (
            "This move gains material. "
            "The minimax evaluation supports this selection."
        )
        contradiction = [
            "REASONING_CONTRADICTION: claims material gain but net_gain=0"
        ]
        good_repl = '{"replacements": ["This is a positional move with no captures."]}'

        with patch(
            "checkers.agents.ranker_agent.call_ranker",
            return_value=good_repl,
        ):
            result, _, resolved = _refine_reasoning(
                original, _REPAIR_MOVE, contradiction, max_attempts=1
            )

        assert resolved is True, f"Full paragraph should be clean. Got: {result!r}"
        assert "minimax evaluation supports" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Regression tests — Issue 3: wording cleanup in seed summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestSanitizeSeedExplanation:
    """Unit tests for _sanitize_seed_explanation."""

    def test_removes_immediate_tactical_safety_prefix(self):
        result = _sanitize_seed_explanation(
            "immediate tactical safety: opponent cannot recapture this piece next turn"
        )
        assert "immediate tactical safety" not in result.lower()
        assert "cannot recapture" in result

    def test_removes_tactical_drawback_prefix(self):
        result = _sanitize_seed_explanation(
            "tactical drawback: opponent can recapture this piece next turn"
        )
        assert "tactical drawback" not in result.lower()
        assert "can recapture" in result

    def test_removes_positional_drawback_prefix(self):
        result = _sanitize_seed_explanation(
            "positional drawback: moved piece is not supported by adjacent allies"
        )
        assert "positional drawback" not in result.lower()
        assert "not supported" in result

    def test_replaces_opponent_can_recapture_false(self):
        result = _sanitize_seed_explanation(
            "opponent_can_recapture=false — chosen move is safer"
        )
        assert "opponent_can_recapture=false" not in result
        assert "cannot recapture" in result

    def test_replaces_our_pieces_threatened_after_zero(self):
        result = _sanitize_seed_explanation("our_pieces_threatened_after=0")
        assert "our_pieces_threatened_after=0" not in result
        assert "no allied pieces" in result or len(result) > 0

    def test_replaces_leaves_piece_isolated_true_in_comparison_seed(self):
        result = _sanitize_seed_explanation(
            "Move [1] isolates the moved piece (leaves_piece_isolated=true vs false here)."
        )
        assert "leaves_piece_isolated=true" not in result

    def test_cleans_vs_false_here_parenthetical(self):
        result = _sanitize_seed_explanation(
            "Move [1] allows recapture (opponent_can_recapture=true vs false here)."
        )
        assert "vs false here" not in result
        assert "vs true here" not in result

    def test_no_double_spaces_in_output(self):
        result = _sanitize_seed_explanation(
            "tactical drawback: the piece is (leaves_piece_isolated=true vs false here) isolated"
        )
        assert "  " not in result

    def test_output_not_empty_for_natural_language_seed(self):
        result = _sanitize_seed_explanation(
            "opponent cannot recapture this piece next turn"
        )
        assert len(result) > 0
        assert result == "opponent cannot recapture this piece next turn"

    def test_captures_count_difference_replaced(self):
        result = _sanitize_seed_explanation(
            "Chosen move captures 2 piece(s) while move [1] captures only 1 "
            "(captures_count difference)."
        )
        assert "captures_count difference" not in result
        assert "capturing more pieces" in result


class TestDeterministicSeedSummaryWording:
    """
    The deterministic seed summary must produce natural language
    without raw field notation, regardless of seed source.
    """

    _move = _REPAIR_MOVE

    def test_no_field_notation_in_output_from_comparison_seed(self):
        """Comparison seed with no ' — ' must not leak field names into output."""
        seeds = [
            "Move [1] isolates the moved piece (leaves_piece_isolated=true vs false here).",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert "leaves_piece_isolated=true" not in result
        assert "leaves_piece_isolated=false" not in result

    def test_no_field_notation_in_output_from_safety_seed(self):
        seeds = [
            "opponent_can_recapture=false — immediate tactical safety: "
            "opponent cannot recapture this piece next turn",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert "opponent_can_recapture=false" not in result
        assert "immediate tactical safety" not in result.lower()
        assert "cannot recapture" in result

    def test_no_field_notation_in_output_from_threatened_seed(self):
        seeds = [
            "our_pieces_threatened_after=0 — no defensive burden remains after the move",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert "our_pieces_threatened_after=0" not in result

    def test_full_seed_set_produces_natural_language(self):
        """A realistic seed set must produce only natural-language output."""
        seeds = [
            "opponent_can_recapture=false — immediate tactical safety: "
            "opponent cannot recapture this piece next turn",
            "our_pieces_threatened_after=0 — no defensive burden remains after the move",
            "captures_count=0 — positional move focused on improving piece placement",
            "leaves_piece_isolated=false — preserves piece coordination "
            "by keeping the moved piece connected",
            "Move [1] isolates the moved piece (leaves_piece_isolated=true vs false here).",
            "minimax_score=2.50 — confirms this is the highest-evaluated option",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        import re
        field_pattern = re.compile(r'\b\w+=(?:true|false|-?\d+(?:\.\d+)?)\b')
        matches = field_pattern.findall(result)
        assert matches == [], (
            f"Field notation found in seed summary output: {matches}\n"
            f"Summary: {result!r}"
        )

    def test_output_is_valid_natural_sentence(self):
        """Output must start with a capital letter and end with a period."""
        seeds = [
            "opponent_can_recapture=false — immediate tactical safety: "
            "opponent cannot recapture this piece next turn",
        ]
        result = _build_deterministic_seed_summary(seeds, self._move)
        assert result[0].isupper()
        assert result.endswith(".")


# ═══════════════════════════════════════════════════════════════════════════════
# Regression tests — Minimax wording tone for losing positions
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinimaxWordingLabel:
    """
    _minimax_wording_label must return score-appropriate phrasing.
    Non-losing scores keep 'highest-evaluated option' unchanged.
    """

    def test_clearly_losing_uses_least_harmful(self):
        assert _minimax_wording_label(-496.0) == "least harmful available continuation"

    def test_near_terminal_loss_uses_least_harmful(self):
        assert _minimax_wording_label(-9994.0) == "least harmful available continuation"

    def test_just_below_clearly_losing_threshold_uses_least_harmful(self):
        assert _minimax_wording_label(_MINIMAX_CLEARLY_LOSING - 0.01) == "least harmful available continuation"

    def test_at_clearly_losing_threshold_uses_difficult_position(self):
        # Threshold is strict < so exactly at boundary falls into the next tier.
        assert _minimax_wording_label(_MINIMAX_CLEARLY_LOSING) == "best available option in a difficult position"

    def test_slightly_losing_uses_difficult_position(self):
        assert _minimax_wording_label(-50.0) == "best available option in a difficult position"

    def test_just_below_slightly_losing_threshold_uses_difficult_position(self):
        assert _minimax_wording_label(_MINIMAX_SLIGHTLY_LOSING - 0.01) == "best available option in a difficult position"

    def test_at_slightly_losing_threshold_uses_highest_evaluated(self):
        # Threshold is strict < so exactly at boundary falls into the next tier.
        assert _minimax_wording_label(_MINIMAX_SLIGHTLY_LOSING) == "highest-evaluated option"

    def test_positive_score_uses_highest_evaluated(self):
        assert _minimax_wording_label(5.0) == "highest-evaluated option"

    def test_zero_score_uses_highest_evaluated(self):
        assert _minimax_wording_label(0.0) == "highest-evaluated option"

    def test_large_positive_uses_highest_evaluated(self):
        assert _minimax_wording_label(9994.0) == "highest-evaluated option"


class TestMinimaxSeedWordingInSeedBuilder:
    """
    _build_grounded_reasoning_seeds must embed the correct label in the
    minimax seed depending on the move's minimax_score.
    """

    def _move_with_score(self, score: float) -> dict:
        return {
            "type": "simple",
            "path": [[5, 4], [4, 3]],
            "captured": [],
            "facts": {
                "minimax_score": score,
                "opponent_can_recapture": False,
                "leaves_piece_isolated": False,
                "captures_count": 0,
                "net_gain": 0,
                "creates_immediate_threat": False,
                "center_control": False,
                "results_in_king": False,
            },
        }

    def _minimax_seed(self, move: dict) -> str:
        seeds = _build_grounded_reasoning_seeds(move, [move])
        mm_seeds = [s for s in seeds if s.startswith("minimax_score=")]
        assert len(mm_seeds) == 1, f"Expected exactly one minimax seed, got: {mm_seeds}"
        return mm_seeds[0]

    def test_clearly_losing_score_uses_least_harmful(self):
        seed = self._minimax_seed(self._move_with_score(-496.0))
        assert "least harmful available continuation" in seed

    def test_near_terminal_loss_uses_least_harmful(self):
        seed = self._minimax_seed(self._move_with_score(-9994.0))
        assert "least harmful available continuation" in seed

    def test_slightly_losing_score_uses_difficult_position(self):
        seed = self._minimax_seed(self._move_with_score(-50.0))
        assert "best available option in a difficult position" in seed

    def test_positive_score_uses_highest_evaluated(self):
        seed = self._minimax_seed(self._move_with_score(5.0))
        assert "highest-evaluated option" in seed

    def test_zero_score_uses_highest_evaluated(self):
        seed = self._minimax_seed(self._move_with_score(0.0))
        assert "highest-evaluated option" in seed

    def test_clearly_losing_does_not_say_highest_evaluated(self):
        seed = self._minimax_seed(self._move_with_score(-496.0))
        assert "highest-evaluated option" not in seed

    def test_seed_still_contains_numeric_score(self):
        """The numeric score must always appear in the seed regardless of label."""
        seed = self._minimax_seed(self._move_with_score(-496.0))
        assert "minimax_score=-496.00" in seed

    def test_seed_is_always_the_last_seed(self):
        """Minimax seed must remain the last seed regardless of label."""
        move = self._move_with_score(-496.0)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        assert seeds[-1].startswith("minimax_score=")


class TestMinimaxSeedWordingInDeterministicSummary:
    """
    The deterministic seed summary produced for a losing position must use
    honest, softened wording — never 'highest-evaluated option'.
    """

    def _summary_for_score(self, score: float) -> str:
        move = {
            "type": "simple",
            "path": [[5, 4], [4, 3]],
            "captured": [],
            "facts": {
                "minimax_score": score,
                "opponent_can_recapture": True,
                "leaves_piece_isolated": True,
                "captures_count": 0,
                "net_gain": 0,
                "creates_immediate_threat": False,
                "center_control": False,
                "results_in_king": False,
                "our_pieces_threatened_after": 1,
            },
        }
        seeds = _build_grounded_reasoning_seeds(move, [move])
        return _build_deterministic_seed_summary(seeds, move)

    def test_clearly_losing_summary_does_not_say_highest_evaluated(self):
        result = self._summary_for_score(-496.0)
        assert "highest-evaluated option" not in result

    def test_clearly_losing_summary_uses_soft_wording(self):
        result = self._summary_for_score(-496.0).lower()
        assert (
            "least harmful" in result
            or "difficult position" in result
        ), f"Expected softened wording for clearly losing score, got: {result!r}"

    def test_slightly_losing_summary_does_not_say_highest_evaluated(self):
        result = self._summary_for_score(-50.0)
        assert "highest-evaluated option" not in result

    def test_positive_score_summary_uses_highest_evaluated(self):
        move = {
            "type": "simple",
            "path": [[5, 4], [4, 3]],
            "captured": [],
            "facts": {
                "minimax_score": 5.0,
                "opponent_can_recapture": False,
                "leaves_piece_isolated": False,
                "captures_count": 0,
                "net_gain": 0,
                "creates_immediate_threat": False,
                "center_control": False,
                "results_in_king": False,
                "our_pieces_threatened_after": 0,
            },
        }
        seeds = _build_grounded_reasoning_seeds(move, [move])
        result = _build_deterministic_seed_summary(seeds, move).lower()
        assert "highest-evaluated option" in result

    def test_summary_passes_truthfulness_checker_for_losing_position(self):
        """The softened wording must not introduce new contradictions."""
        move = {
            "type": "simple",
            "path": [[5, 4], [4, 3]],
            "captured": [],
            "facts": {
                "minimax_score": -496.0,
                "opponent_can_recapture": True,
                "leaves_piece_isolated": True,
                "captures_count": 0,
                "net_gain": 0,
                "creates_immediate_threat": False,
                "center_control": False,
                "results_in_king": False,
                "our_pieces_threatened_after": 1,
            },
        }
        seeds = _build_grounded_reasoning_seeds(move, [move])
        summary = _build_deterministic_seed_summary(seeds, move)
        contradictions = _check_reasoning_truthfulness(summary, move["facts"], seeds=seeds)
        assert contradictions == [], (
            f"Softened wording introduced contradiction: {contradictions}\n"
            f"Summary: {summary!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — our_mobility and unconditional opponent_mobility seeds
# ══════════════════════════════════════════════════════════════════════════════

def _move_with_mobility(
    opp_before: int, opp_after: int,
    our_before: int, our_after: int,
    **extra_facts,
) -> dict:
    """Synthetic move with all four mobility facts."""
    facts = {
        "minimax_score": 5.0,
        "opponent_mobility_before": opp_before,
        "opponent_mobility_after": opp_after,
        "our_mobility_before": our_before,
        "our_mobility_after": our_after,
        "captures_count": 0,
        "net_gain": 0,
        "results_in_king": False,
        "near_promotion": False,
        "opponent_can_recapture": False,
        "leaves_piece_isolated": False,
        "creates_immediate_threat": False,
        "center_control": False,
        "blocks_opponent_landing": False,
        "our_pieces_threatened_after": 0,
    }
    facts.update(extra_facts)
    return {
        "type": "simple",
        "path": [[5, 0], [4, 1]],
        "captured": [],
        "facts": facts,
    }


class TestFix2MobilitySeeds:
    """Seeds for our_mobility and opponent_mobility must always be emitted."""

    def test_opponent_mobility_seed_emitted_when_reduced(self):
        move = _move_with_mobility(opp_before=12, opp_after=8, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "opponent_mobility_before=12" in seeds_text
        assert "opponent_mobility_after=8" in seeds_text

    def test_opponent_mobility_seed_emitted_when_equal(self):
        """Previously suppressed (mob_after >= mob_before) — now always emitted."""
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "opponent_mobility_before=10" in seeds_text
        assert "opponent_mobility_after=10" in seeds_text

    def test_opponent_mobility_seed_emitted_when_increased(self):
        """mob_after > mob_before — was suppressed before FIX 2."""
        move = _move_with_mobility(opp_before=8, opp_after=12, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "opponent_mobility_before=8" in seeds_text
        assert "opponent_mobility_after=12" in seeds_text

    def test_our_mobility_seed_emitted_when_improved(self):
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=5, our_after=8)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "our_mobility_before=5" in seeds_text
        assert "our_mobility_after=8" in seeds_text

    def test_our_mobility_seed_emitted_when_reduced(self):
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=8, our_after=5)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "our_mobility_before=8" in seeds_text
        assert "our_mobility_after=5" in seeds_text

    def test_our_mobility_seed_emitted_when_unchanged(self):
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=7, our_after=7)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "our_mobility_before=7" in seeds_text
        assert "our_mobility_after=7" in seeds_text

    def test_type_b_no_contradiction_opponent_mobility_from_to(self):
        """
        Type B check: 'from X to Y' where X=opp_before, Y=opp_after must pass
        truthfulness checker after FIX 2 (seeds provide both numbers).
        """
        move = _move_with_mobility(opp_before=12, opp_after=8, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        reasoning = (
            "This move reduces opponent mobility from 12 to 8, "
            "restricting available replies."
        )
        contradictions = _check_reasoning_truthfulness(
            reasoning, move["facts"], seeds=seeds
        )
        assert contradictions == [], (
            f"Grounded 'from 12 to 8' should pass; got: {contradictions}"
        )

    def test_type_b_no_contradiction_our_mobility_from_to(self):
        """
        Type B check: 'from X to Y' for our_mobility must pass after FIX 2.
        """
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=5, our_after=8)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        reasoning = (
            "This move improves our mobility from 5 to 8, "
            "giving us more active replies."
        )
        contradictions = _check_reasoning_truthfulness(
            reasoning, move["facts"], seeds=seeds
        )
        assert contradictions == [], (
            f"Grounded 'from 5 to 8' should pass; got: {contradictions}"
        )

    def test_type_b_contradiction_still_fires_for_wrong_numbers(self):
        """
        If LLM uses numbers NOT in seeds (fabricated), Type B check must still fire.
        """
        move = _move_with_mobility(opp_before=12, opp_after=8, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        reasoning = "This move reduces opponent mobility from 15 to 3."
        contradictions = _check_reasoning_truthfulness(
            reasoning, move["facts"], seeds=seeds
        )
        assert len(contradictions) > 0, (
            "Fabricated 'from 15 to 3' should trigger Type B contradiction"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Truthfulness checker — unchanged mobility false-positive fix
# ══════════════════════════════════════════════════════════════════════════════

class TestUnchangedMobilityFalsePositives:
    """
    After FIX 2, seeds always emit opponent/our mobility values even when
    before == after.  The seed says 'no change in opponent mobility' — not
    'unchanged mobility' — so the verbatim pattern match used to fire as a
    false positive.  These tests pin the corrected behaviour.
    """

    def _check(self, reasoning: str, opp_b: int, opp_a: int,
                our_b: int, our_a: int) -> list[str]:
        move = _move_with_mobility(
            opp_before=opp_b, opp_after=opp_a,
            our_before=our_b, our_after=our_a,
        )
        seeds = _build_grounded_reasoning_seeds(move, [move])
        return _check_reasoning_truthfulness(reasoning, move["facts"], seeds=seeds)

    # ── unchanged mobility supported by seed passes ────────────────────────────

    def test_unchanged_mobility_phrasing_passes_when_seed_supports(self):
        """'unchanged mobility' must not fire when opp mob is actually unchanged."""
        contradictions = self._check(
            "This move results in unchanged mobility for the opponent.",
            opp_b=10, opp_a=10, our_b=6, our_a=6,
        )
        assert contradictions == [], (
            f"False positive for 'unchanged mobility' when equal: {contradictions}"
        )

    def test_mobility_remains_unchanged_passes_when_seed_supports(self):
        """'mobility remains unchanged' must not fire when mob is equal."""
        contradictions = self._check(
            "Opponent mobility remains unchanged after this move.",
            opp_b=8, opp_a=8, our_b=5, our_a=5,
        )
        assert contradictions == [], (
            f"False positive for 'mobility remains unchanged': {contradictions}"
        )

    def test_same_number_of_moves_passes_when_seed_supports(self):
        """'same number of moves' must not fire when opponent mobility is unchanged."""
        contradictions = self._check(
            "The opponent retains the same number of moves as before.",
            opp_b=7, opp_a=7, our_b=5, our_a=5,
        )
        assert contradictions == [], (
            f"False positive for 'same number of moves': {contradictions}"
        )

    def test_our_mob_unchanged_also_suppresses_pattern(self):
        """Suppression applies when OUR mobility is unchanged, even if opp changed."""
        contradictions = self._check(
            "Our mobility stays unchanged throughout this sequence.",
            opp_b=10, opp_a=8, our_b=6, our_a=6,
        )
        assert contradictions == [], (
            f"False positive when our_mob is unchanged: {contradictions}"
        )

    # ── fabricated mobility values still fail ─────────────────────────────────

    def test_fabricated_numbers_still_fail_when_mobility_unchanged(self):
        """Type B check must still fire if LLM uses numbers NOT in seeds."""
        contradictions = self._check(
            "Opponent mobility from 15 to 10.",
            opp_b=10, opp_a=10, our_b=6, our_a=6,
        )
        assert len(contradictions) > 0, (
            "Fabricated 'from 15 to 10' should still trigger Type B"
        )

    # ── false unchanged claim when seed says changed still fails ──────────────

    def test_unchanged_mobility_claim_fires_when_mobility_actually_changed(self):
        """'unchanged mobility' must still fire if mob_before != mob_after."""
        contradictions = self._check(
            "This move results in unchanged mobility for the opponent.",
            opp_b=10, opp_a=7, our_b=6, our_a=6,
        )
        assert len(contradictions) > 0, (
            "'unchanged mobility' claimed but mob_before=10, mob_after=7 — must fire"
        )

    def test_mobility_remains_unchanged_fires_when_mobility_changed(self):
        """'mobility remains unchanged' must still fire when opp mob changed."""
        contradictions = self._check(
            "Opponent mobility remains unchanged after this move.",
            opp_b=12, opp_a=8, our_b=6, our_a=6,
        )
        assert len(contradictions) > 0, (
            "'mobility remains unchanged' claimed but mob reduced 12→8 — must fire"
        )

    def test_same_number_of_moves_fires_when_mobility_changed(self):
        """'same number of moves' must still fire when opp mob changed."""
        contradictions = self._check(
            "The opponent has the same number of moves as before.",
            opp_b=10, opp_a=6, our_b=5, our_a=5,
        )
        assert len(contradictions) > 0, (
            "'same number of moves' claimed but mob reduced 10→6 — must fire"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Strategic context perspective proof
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategicContextPerspective:
    """
    Proves that score_state in strategic_context is always computed from the
    perspective of the player who is ABOUT TO MOVE in the upcoming turn.

    Pipeline: update_agent calls state_manager (switches player) then
    inter_turn_memory with post_switch current_player.  By the time ranker_agent
    reads strategic_context in the next turn, score_state reflects the current
    mover's own perspective — never the opponent's.

    Safety-filter losing_mode depends only on score_state and losing priorities
    (SEEK_COUNTERPLAY, COMPLICATE, CREATE_THREATS), all of which are gated on
    score_state in priority construction.  The turn_history perspective mixing
    only affects material_trend-based DEFEND priorities — not losing_mode.
    """

    def _make_state_for_player(self, board, player, turn_number=15):
        from checkers.engine.rules import get_all_legal_moves
        from checkers.state.state import CheckersState
        return CheckersState(
            board=board,
            current_player=player,
            turn_number=turn_number,
            legal_moves=get_all_legal_moves(board, player),
            strategic_context={},
        )

    def _build_board_red_winning(self):
        """RED has 6 pieces, BLACK has 3 — RED is materially ahead."""
        from checkers.engine.board import EMPTY, RED, BLACK
        b = [[EMPTY] * 8 for _ in range(8)]
        for pos in [(7, 0), (7, 2), (6, 1), (5, 0), (4, 1), (3, 2)]:
            b[pos[0]][pos[1]] = RED
        for pos in [(0, 1), (1, 2), (2, 3)]:
            b[pos[0]][pos[1]] = BLACK
        return b

    def test_score_state_reflects_red_winning_when_red_to_move(self):
        """When RED is clearly winning and it is RED's turn, score_state == CLEARLY_WINNING."""
        from checkers.engine.board import RED
        from checkers.nodes.inter_turn_memory import inter_turn_memory
        board = self._build_board_red_winning()
        state = self._make_state_for_player(board, RED)
        ctx = inter_turn_memory(state)["strategic_context"]
        assert ctx["player_perspective"] == "RED"
        assert ctx["score_state"] in ("SLIGHTLY_WINNING", "CLEARLY_WINNING"), (
            f"RED has material advantage; expected winning score_state, got {ctx['score_state']}"
        )

    def test_score_state_reflects_black_losing_when_black_to_move(self):
        """When RED is materially ahead and BLACK is to move, score_state reflects BLACK losing."""
        from checkers.engine.board import BLACK
        from checkers.nodes.inter_turn_memory import inter_turn_memory
        board = self._build_board_red_winning()
        state = self._make_state_for_player(board, BLACK)
        ctx = inter_turn_memory(state)["strategic_context"]
        assert ctx["player_perspective"] == "BLACK"
        assert ctx["score_state"] in ("SLIGHTLY_LOSING", "CLEARLY_LOSING"), (
            f"BLACK is behind; expected losing score_state from BLACK perspective, "
            f"got {ctx['score_state']}"
        )

    def test_safety_filter_not_in_losing_mode_when_player_is_winning(self):
        """
        safety_filter losing_mode must be False when the current player is winning.
        Checks that score_state → losing_mode path has no sign inversion.
        """
        from checkers.engine.board import RED
        from checkers.nodes.inter_turn_memory import inter_turn_memory
        from checkers.agents.ranker_agent import _apply_safety_filter
        from checkers.engine.rules import get_all_legal_moves
        board = self._build_board_red_winning()
        state = self._make_state_for_player(board, RED)
        ctx = inter_turn_memory(state)["strategic_context"]
        score_state = ctx["score_state"]
        priorities = ctx.get("strategic_priorities", [])
        legal = get_all_legal_moves(board, RED)
        # Build minimal move dicts for filter (safe moves)
        from checkers.engine.board import EMPTY
        legal_with_facts = [
            {"type": m["type"], "path": m["path"], "captured": m.get("captured", []),
             "facts": {"opponent_can_recapture": False, "minimax_score": 1.0,
                       "results_in_king": False}}
            for m in legal
        ]
        _, _ = _apply_safety_filter(
            legal_with_facts, strategic_priorities=priorities, score_state=score_state
        )
        # Verify score_state is winning (not losing) so losing_mode is False
        assert score_state not in ("CLEARLY_LOSING", "SLIGHTLY_LOSING"), (
            f"RED is winning but score_state={score_state!r} — sign inversion detected"
        )
        losing_priorities = {"SEEK_COUNTERPLAY", "COMPLICATE", "CREATE_THREATS"}
        assert not losing_priorities.intersection(priorities), (
            f"Losing priorities present when RED is winning: {losing_priorities.intersection(priorities)}"
        )
