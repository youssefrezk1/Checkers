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

from checkers.agents.explainer_agent import (
    EXPLAINER_REASONING_REFINEMENT_SYSTEM as RANKER_REASONING_REFINEMENT_SYSTEM,
    EXPLAINER_SEED_REASONING_SYSTEM as RANKER_SEED_REASONING_SYSTEM,
    _CONTEXT_FORBIDDEN_VOCAB,
    _FORBIDDEN_VOCAB,
    _MINIMAX_CLEARLY_LOSING,
    _MINIMAX_SLIGHTLY_LOSING,
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
            "This move keeps all pieces safe. "
            "It blocks the opponent from landing on a key square. "
            "The engine score of 5.0 confirms this choice."
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
        assert _clean("This move avoids recapture.", facts)

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
        assert _clean("This move controls the center.", facts)

    def test_central_control_phrase_detected(self):
        facts = _base_facts(center_control=False)
        assert _warns("It establishes central control.", facts)

    def test_central_board_presence_blocked_when_false(self):
        # "central board presence" conflates geometric and tactical center.
        # Must be rejected when center_control=False.
        facts = _base_facts(center_control=False)
        assert _warns("This move contributes to central board presence.", facts)

    def test_influence_over_central_blocked_when_false(self):
        facts = _base_facts(center_control=False)
        assert _warns("The piece improves influence over central squares.", facts)

    def test_central_board_presence_always_forbidden(self):
        # "central board presence" is in _CONTEXT_FORBIDDEN_VOCAB and no seed
        # ever emits it — so it is blocked regardless of center_control value.
        facts = _base_facts(center_control=True)
        assert _warns("This move contributes to central board presence.", facts)

    def test_no_warning_for_geometric_center_phrasing_when_false(self):
        # The new geometric-center seed uses natural language without key=value syntax.
        facts = _base_facts(center_control=False)
        assert _clean(
            "The destination is in the center of the board (column 3).",
            facts,
        )


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
        assert _clean("This move gains material.", facts)


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
        # _CHOSEN_MOVE has opponent_can_recapture=True → expect human-readable line
        assert "opponent CAN recapture" in prompt
        # must not expose raw schema key names
        assert "opponent_can_recapture=" not in prompt


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
        with patch("checkers.agents.explainer_agent.call_explainer", return_value=_CLEAN_RESPONSE):
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

        with patch("checkers.agents.explainer_agent.call_explainer", side_effect=lambda *a, **k: next(responses)):
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
        with patch("checkers.agents.explainer_agent.call_explainer", return_value=_STILL_BAD):
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
            "checkers.agents.explainer_agent.call_explainer",
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
        with patch("checkers.agents.explainer_agent.call_explainer", return_value=_CLEAN_RESPONSE):
            _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
            )
        assert _CHOSEN_MOVE == chosen_copy, "chosen_move must not be mutated"

    # ── separate retry count ──────────────────────────────────────────────────

    def test_retry_count_is_independent_int(self):
        """reasoning_retry_count is a plain int, independent of move retries."""
        with patch("checkers.agents.explainer_agent.call_explainer", return_value=_CLEAN_RESPONSE):
            _, retry_count, _ = _refine_reasoning(
                reasoning=_BAD_REASONING,
                chosen_move=_CHOSEN_MOVE,
                initial_contradictions=self._contradictions(),
            )
        assert isinstance(retry_count, int)
        assert retry_count >= 1

    # ── safety filter isolation ───────────────────────────────────────────────
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
        assert "cannot be immediately recaptured" in self._text()

    def test_recapture_true_seed_present(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "opponent_can_recapture": True}}
        assert "opponent can recapture" in self._text(move, [move]).lower()

    def test_pieces_threatened_zero_seed(self):
        assert "No allied pieces remain under attack" in self._text()

    def test_pieces_threatened_nonzero_seed(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "our_pieces_threatened_after": 2}}
        seeds = _build_grounded_reasoning_seeds(move, [move])
        assert any("2 allied piece(s) remain under threat" in s for s in seeds)

    # ── tactical seeds ───────────────────────────────────────────────────────

    def test_captures_seed_when_nonzero(self):
        assert "The move captures 2 piece(s)" in self._text()

    def test_no_captures_seed_when_zero(self):
        """captures_count=0 → no material-gain seed fires."""
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "captures_count": 0, "net_gain": 0}}
        text = self._text(move, [move])
        assert "The move captures" not in text

    def test_creates_threat_seed_when_true(self):
        assert "immediate threat" in self._text()

    def test_no_creates_threat_seed_when_false(self):
        # Phase G, Step 1: when creates_immediate_threat=False the seed builder
        # now intentionally emits an explicit NEGATIVE seed
        # ("This move does not create an immediate threat.") to suppress the
        # LLM's tendency to fabricate the positive.  The previous behaviour
        # (silent absence) was the cause of the audit's T2 failure class.
        # The semantic guarantee we still want to hold here is that NO
        # POSITIVE threat assertion is emitted.
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "creates_immediate_threat": False}}
        text = self._text(move, [move])
        # No positive threat-creation seed
        assert "This move forces the opponent to respond to an immediate threat" not in text
        # Explicit negative seed IS emitted
        assert "does not create an immediate threat" in text

    # ── mobility guard ───────────────────────────────────────────────────────

    def test_mobility_seed_when_after_less_than_before(self):
        """Mobility seed uses natural-language wording when after < before."""
        text = self._text()  # _FULL_FACTS has after=8 < before=12
        assert "opponent mobility changes from 12 to 8" in text

    def test_mobility_seed_emitted_when_equal(self):
        """Mobility seed uses 'remains at N' wording when before == after."""
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "opponent_mobility_after": 10, "opponent_mobility_before": 10}}
        text = self._text(move, [move])
        assert "opponent mobility remains at 10" in text

    def test_mobility_seed_emitted_when_after_greater(self):
        """Mobility seed uses 'changes from N to M' wording when after > before."""
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "opponent_mobility_after": 14, "opponent_mobility_before": 10}}
        text = self._text(move, [move])
        assert "opponent mobility changes from 10 to 14" in text

    def test_mobility_seed_does_not_duplicate_key_value_form(self):
        """Phase-6 dedup: only the natural-language form is emitted.
        The legacy 'opponent_mobility_before=N, opponent_mobility_after=M'
        structured form must NOT appear alongside it."""
        text = self._text()  # natural form is "from 12 to 8"
        assert "opponent_mobility_before=12, opponent_mobility_after=8" not in text
        assert "our_mobility_before=" not in text  # _FULL_FACTS sets our mob too

    # ── center_control guard ──────────────────────────────────────────────────

    def test_center_seed_when_true(self):
        assert "central board control" in self._text()

    def test_no_center_seed_when_false(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "center_control": False}}
        text = self._text(move, [move])
        # Positive center seed must not appear; an explicit negative
        # ("does not gain central board control") is allowed and desired
        # to push back against center-fabrication.
        assert "gains central board control" not in text
        assert "claims central control" not in text

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
        assert "engine scores this move" in seeds[-1], "minimax seed must be the last seed"

    def test_minimax_not_in_non_last_seeds(self):
        seeds = self._seeds()
        for s in seeds[:-1]:
            assert "engine scores this move" not in s, (
                f"minimax seed must not appear before final seed: {s}"
            )

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
        assert "recaptur" in result.lower()  # "recaptured" or "recapture"
        assert "worse" not in result
        assert "no advantage" not in result
        assert "opponent_can_recapture=" not in result  # no schema syntax

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


class TestDecisionRelevantFactsBlock:
    """
    Phase 4.3b: the seed-reasoning prompt must instruct the LLM to surface the
    most decision-relevant verifiable facts present in the seeds, in a
    fixed priority order, without mechanically restating every seed.
    """

    PROMPT = RANKER_SEED_REASONING_SYSTEM

    def test_block_header_present(self):
        assert "DECISION-RELEVANT FACTS" in self.PROMPT

    def test_uses_grounded_facts_only(self):
        assert "grounded facts" in self.PROMPT

    def test_priority_material_change(self):
        assert "material change" in self.PROMPT
        assert "captures_count" in self.PROMPT
        assert "net_gain" in self.PROMPT

    def test_priority_mobility_change(self):
        assert "mobility change" in self.PROMPT
        assert "opponent_mobility_before/after" in self.PROMPT \
            or "opponent_mobility_after" in self.PROMPT
        assert "mobility_reduction" in self.PROMPT

    def test_priority_immediate_threat(self):
        assert "immediate threat" in self.PROMPT
        assert "creates_immediate_threat" in self.PROMPT

    def test_priority_recapture_safety_or_risk(self):
        assert "recapture safety or risk" in self.PROMPT
        assert "opponent_can_recapture" in self.PROMPT

    def test_priority_forced_reply(self):
        assert "forced opponent reply" in self.PROMPT
        assert "forced_opponent_jump_reply" in self.PROMPT

    def test_priority_isolation_connectivity(self):
        assert ("isolation or connectivity" in self.PROMPT
                or "isolation/connectivity" in self.PROMPT)
        assert "leaves_piece_isolated" in self.PROMPT

    def test_priority_adversity_losing_context(self):
        # Mention adversity / losing-position context — at least one cue.
        assert "losing-position" in self.PROMPT or "adversity" in self.PROMPT
        # And at least one adversity seed name.
        assert ("slightly_losing" in self.PROMPT
                or "clearly_losing" in self.PROMPT
                or "least_harmful" in self.PROMPT
                or "forced_choice" in self.PROMPT)

    def test_do_not_mechanically_restate_every_seed(self):
        assert "Do not mechanically restate every seed" in self.PROMPT \
            or "mechanically restate" in self.PROMPT

    def test_do_not_invent_strategic_terms(self):
        assert "Do not invent unsupported strategic terms" in self.PROMPT \
            or "Do not invent" in self.PROMPT

    def test_no_forbidden_vocabulary_introduced(self):
        """The new DECISION-RELEVANT FACTS block must not introduce any of
        the strategic phrases the prompt elsewhere forbids the LLM from using."""
        prompt = self.PROMPT
        start = prompt.find("DECISION-RELEVANT FACTS")
        assert start != -1, "block header missing"
        # Block ends at its own closing line.
        end = prompt.find("mechanically restate every seed", start)
        assert end != -1, "block terminator missing"
        # Walk to end of that line.
        end = prompt.find("\n", end)
        block = prompt[start:end if end != -1 else len(prompt)]

        forbidden_in_block = (
            "structural pressure",
            "stable position",
            "good position",
            "limits options",
            "initiative",
            "dominance",
            "control the game",
            "strong position",
            "winning conversion",
            "no advantage",
        )
        for banned in forbidden_in_block:
            assert banned not in block.lower(), (
                f"new block must not contain forbidden phrase {banned!r}"
            )


class TestGenerateSeededReasoning:
    def test_seeded_reasoning_uses_call_ranker(self):
        """_generate_seeded_reasoning calls call_ranker exactly once on success."""
        response = '{"reasoning": "Clean grounded paragraph."}'
        with patch("checkers.agents.explainer_agent.call_explainer", return_value=response) as mock:
            result, seeds_out = _generate_seeded_reasoning(_FULL_MOVE, [_FULL_MOVE, _ALT_MOVE])
        assert result == "Clean grounded paragraph."
        assert isinstance(seeds_out, list)
        mock.assert_called_once()

    def test_seeded_reasoning_api_failure_returns_none(self):
        """If the API fails, returns (None, seeds) — caller keeps previous reasoning."""
        with patch("checkers.agents.explainer_agent.call_explainer", side_effect=OSError("net")):
            result, seeds_out = _generate_seeded_reasoning(_FULL_MOVE, [_FULL_MOVE, _ALT_MOVE])
        assert result is None
        assert isinstance(seeds_out, list)
    def test_chosen_move_not_mutated_by_generate(self):
        """chosen_move dict is identical before and after seed generation."""
        before = copy.deepcopy(_FULL_MOVE)
        response = '{"reasoning": "OK."}'
        with patch("checkers.agents.explainer_agent.call_explainer", return_value=response):
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
        assert "cannot be immediately recaptured" in self._text(move)

    def test_pieces_safe_no_defensive_burden(self):
        assert "No allied pieces remain under attack" in self._text(_FULL_MOVE)

    def test_captures_wins_material(self):
        assert "The move captures" in self._text(_FULL_MOVE)

    def test_creates_threat_puts_on_defensive(self):
        assert "forces the opponent to respond" in self._text(_FULL_MOVE)

    def test_center_control_improves_influence(self):
        assert "central board control" in self._text(_FULL_MOVE)

    def test_isolated_false_preserves_coordination(self):
        assert "not left isolated" in self._text(_FULL_MOVE)

    def test_results_in_king_converts_piece(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "results_in_king": True}}
        assert "immediately promoted to king" in self._text(move)

    def test_near_promotion_future_threat(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "results_in_king": False, "near_promotion": True}}
        assert "one step from promotion" in self._text(move)

    def test_forced_jump_constrained(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS,
            "forced_opponent_jump_reply": True, "max_opponent_jump_captures": 1}}
        assert "forced to respond with a jump" in self._text(move)

    def test_blocks_landing_denies_square(self):
        assert "denies the opponent a key landing square" in self._text(_FULL_MOVE)

    def test_mobility_restricts_replies(self):
        text = self._text(_FULL_MOVE)   # after=8 < before=12
        assert "restricting available replies" in text

    # ── tradeoff / drawback seeds ─────────────────────────────────────────────

    def test_recapture_true_is_tactical_drawback(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "opponent_can_recapture": True}}
        assert "opponent can recapture" in self._text(move).lower()

    def test_pieces_threatened_is_tactical_drawback(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "our_pieces_threatened_after": 2}}
        assert "2 allied piece(s) remain under threat" in self._text(move)

    def test_moved_piece_threatened_is_exposed(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "moved_piece_is_threatened": True}}
        assert "remains under immediate threat" in self._text(move)

    def test_isolation_is_positional_drawback(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "leaves_piece_isolated": True}}
        assert "left without adjacent support" in self._text(move)

    def test_weakens_king_row_back_row_weakened(self):
        move = {**_FULL_MOVE, "facts": {**_FULL_FACTS, "weakens_king_row": True}}
        assert "weakens the back-row defensive structure" in self._text(move)

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
        # Prompt uses natural language to describe drawbacks, not schema keys
        prompt_lower = RANKER_SEED_REASONING_SYSTEM.lower()
        assert "opponent can recapture" in prompt_lower
        assert "isolated" in prompt_lower


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
        assert "advances forward without capturing" in _text_with_player(_SIMPLE_QUIET, _RED)

    def test_development_seed_is_factual_only(self):
        # Seed (A) must state the geometric fact without filler phrases.
        text = _text_with_player(_SIMPLE_QUIET, _RED)
        assert "advances forward without capturing" in text
        assert "improves piece activity" not in text

    def test_no_development_seed_when_capture(self):
        assert "advances forward without capturing" not in self._text(_JUMP_MOVE)

    # ── (B) Back-row seed ───────────────────────────────────────────────────────────────

    def test_back_row_origin_produces_weakening_seed(self):
        # _BACK_ROW_MOVE src_row=7 → RED back row → seed mentions "A back-row piece is moved"
        assert "A back-row piece is moved" in _text_with_player(_BACK_ROW_MOVE, _RED)

    def test_back_row_seed_intact_when_weakens_false(self):
        # weakens_king_row absent/False → seed says structure remains intact
        text = _text_with_player(_BACK_ROW_MOVE, _RED)
        assert "defensive structure remains intact" in text
        assert "weakening" not in text

    def test_back_row_seed_weakens_when_flag_true(self):
        # When weakens_king_row=True the seed must say it weakens.
        move = {**_BACK_ROW_MOVE, "facts": {**_BACK_ROW_MOVE["facts"], "weakens_king_row": True}}
        text = _text_with_player(move, _RED)
        assert "weakening the defensive structure" in text
        assert "remains intact" not in text

    def test_no_back_row_seed_for_midgame_row(self):
        # _SIMPLE_QUIET starts at row 5 — not back-row for any color
        assert "A back-row piece is moved" not in _text_with_player(_SIMPLE_QUIET, _RED)

    def test_row_zero_triggers_back_row_seed(self):
        move = {**_BACK_ROW_MOVE, "path": [[0, 2], [1, 3]]}
        # src_row=0 → BLACK back row
        assert "A back-row piece is moved" in _text_with_player(move, _BLACK)

    # ── (C) Positional seed ──────────────────────────────────────────────────

    def test_quiet_move_produces_positional_seed(self):
        text = self._text(_SIMPLE_QUIET)
        assert "improves piece placement without capturing" in text

    def test_no_positional_seed_when_capture(self):
        assert "improves piece placement" not in self._text(_JUMP_MOVE)

    # ── (D) Center direction seed ────────────────────────────────────────────

    def test_center_destination_produces_center_seed(self):
        # _SIMPLE_QUIET has center_control=False; no center seed should be emitted
        # even though dst_col=3 is geometrically central.  The gate is center_control.
        assert "center of the board" not in self._text(_SIMPLE_QUIET)

    def test_center_seed_mentions_column(self):
        # Center seed is emitted only when center_control=True; includes column number.
        move = {**_SIMPLE_QUIET, "facts": {**_SIMPLE_QUIET["facts"], "center_control": True}}
        assert "center of the board" in self._text(move)
        assert "column 3" in self._text(move)

    def test_no_center_seed_for_edge_destination(self):
        # _EDGE_MOVE goes to col 0 — no center seed regardless of center_control.
        assert "center of the board" not in self._text(_EDGE_MOVE)

    def test_column_2_triggers_center_seed(self):
        # col 2 only produces a center seed when center_control=True.
        move_false = {**_SIMPLE_QUIET, "path": [[5, 1], [4, 2]]}
        move_true  = {**move_false, "facts": {**_SIMPLE_QUIET["facts"], "center_control": True}}
        assert "center of the board" not in self._text(move_false)
        assert "center of the board" in self._text(move_true)

    def test_column_5_triggers_center_seed(self):
        # col 5 only produces a center seed when center_control=True.
        move_false = {**_SIMPLE_QUIET, "path": [[5, 6], [4, 5]]}
        move_true  = {**move_false, "facts": {**_SIMPLE_QUIET["facts"], "center_control": True}}
        assert "center of the board" not in self._text(move_false)
        assert "center of the board" in self._text(move_true)

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
        _SOME_ALT = {
            "type": "simple", "path": [[5, 2], [4, 1]],
            "facts": {**_SIMPLE_QUIET["facts"]}, "minimax_score": -99.0,
        }
        seeds = _build_grounded_reasoning_seeds(_SIMPLE_QUIET, [_SIMPLE_QUIET, _SOME_ALT])
        assert seeds, "seeds must not be empty"
        assert "engine scores this move" in seeds[-1]

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
        assert "advances forward without capturing" in _text_with_player(_RED_FORWARD, _RED)

    def test_red_backward_move_no_development_seed(self):
        assert "advances forward without capturing" not in _text_with_player(_RED_BACKWARD, _RED)

    # ── (A) Development seed — BLACK ─────────────────────────────────────────

    def test_black_forward_move_gets_development_seed(self):
        assert "advances forward without capturing" in _text_with_player(_BLACK_FORWARD, _BLACK)

    def test_black_backward_move_no_development_seed(self):
        assert "advances forward without capturing" not in _text_with_player(_BLACK_BACKWARD, _BLACK)

    # ── (A) Development seed — unknown player: must NOT fire ─────────────────

    def test_unknown_player_no_development_seed_forward(self):
        """player=0 — direction unverifiable; seed must NOT fire."""
        assert "advances forward without capturing" not in _text_with_player(_RED_FORWARD, 0)
    def test_unknown_player_no_development_seed_backward(self):
        """player=0 — direction unverifiable; seed must NOT fire."""
        assert "advances forward without capturing" not in _text_with_player(_RED_BACKWARD, 0)

    # ── (B) Back-row seed — RED ──────────────────────────────────────────────

    def test_red_row7_origin_triggers_back_row_seed(self):
        assert "A back-row piece is moved" in _text_with_player(_RED_BACK_ROW, _RED)

    def test_red_row0_origin_does_not_trigger_back_row_seed(self):
        """Row 0 is BLACK's back row, not RED's."""
        assert "A back-row piece is moved" not in _text_with_player(_BLACK_BACK_ROW, _RED)

    # ── (B) Back-row seed — BLACK ────────────────────────────────────────────

    def test_black_row0_origin_triggers_back_row_seed(self):
        assert "A back-row piece is moved" in _text_with_player(_BLACK_BACK_ROW, _BLACK)

    def test_black_row7_origin_does_not_trigger_back_row_seed(self):
        """Row 7 is RED's back row, not BLACK's."""
        assert "A back-row piece is moved" not in _text_with_player(_RED_BACK_ROW, _BLACK)

    # ── (B) Back-row seed — unknown player: must NOT fire ─────────────────
    def test_unknown_player_row7_no_back_row_seed(self):
        assert "A back-row piece is moved" not in _text_with_player(_RED_BACK_ROW, 0)
    def test_unknown_player_row0_no_back_row_seed(self):
        assert "A back-row piece is moved" not in _text_with_player(_BLACK_BACK_ROW, 0)
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

    def test_no_new_vulnerabilities_allowed(self):
        # Negated safety claim — "no new vulnerabilities" must not be flagged.
        assert not _has_contradiction(_hall("no new vulnerabilities introduced"), "new vulnerabilities")

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

    def test_escape_compound_phrases_in_forbidden_vocab(self):
        # Bare "escape" was intentionally removed from _CONTEXT_FORBIDDEN_VOCAB
        # (too short — fires on "cannot escape the capture").
        # Compound forms are covered by _FORBIDDEN_VOCAB instead.
        assert "escape squares" in _FORBIDDEN_VOCAB
        assert "escape routes" in _FORBIDDEN_VOCAB
        assert "king escape" in _FORBIDDEN_VOCAB
        # Bare "escape" alone must NOT fire (see comment in _CONTEXT_FORBIDDEN_VOCAB)
        assert not _has_contradiction(_hall("the king can escape"), "term 'escape' used but not in seeds")

    def test_diagonal_valid_usage_allowed(self):
        # Bare "diagonal" is valid checkers vocabulary — must not be context-forbidden.
        assert not _has_contradiction(_hall("controls the diagonal"), "term 'diagonal' used but not in seeds")

    def test_diagonal_compound_forms_still_forbidden(self):
        # Compound invented forms remain in the absolute _FORBIDDEN_VOCAB.
        assert _has_contradiction(_hall("creates diagonal pressure on the left"), "forbidden term 'diagonal pressure'")
        assert _has_contradiction(_hall("exposes diagonal risks"), "forbidden term 'diagonal risks'")
        assert _has_contradiction(_hall("dominates the long diagonal"), "forbidden term 'long diagonal'")

    def test_new_vulnerabilities_without_seed_flagged(self):
        # Positive form without seed — context-forbidden fires.
        assert _has_contradiction(
            _hall("this creates new vulnerabilities in our structure"),
            "term 'new vulnerabilities' used but not in seeds",
        )

    def test_new_vulnerabilities_with_seed_allowed(self):
        # Positive form allowed when seed introduces it.
        assert not _has_contradiction(
            _hall("this creates new vulnerabilities", seeds=["new vulnerabilities=true"]),
            "term 'new vulnerabilities' used but not in seeds",
        )

    def test_no_new_vulnerabilities_negation_skipped(self):
        # Negated form must be skipped even though substring matches.
        result = _hall("introduces no new vulnerabilities to our formation")
        assert not _has_contradiction(result, "new vulnerabilities")

    # ── Domain vocabulary — must never trigger ──────────────────────

    def test_diagonal_move_allowed(self):
        # "diagonal" is valid checkers vocabulary.
        assert not _has_contradiction(_hall("a diagonal move to (3,4)"), "diagonal")

    def test_moves_diagonally_allowed(self):
        # "diagonally" must not fire any forbidden-vocab check.
        assert not _has_contradiction(_hall("the piece moves diagonally forward"), "diagonal")

    def test_no_new_vulnerabilities_not_absolute_forbidden(self):
        # Negated safety statement must not trigger the absolute _FORBIDDEN_VOCAB check.
        assert not _has_contradiction(_hall("no new vulnerabilities are introduced"), "forbidden")

    # ── Strict schema-leak terms — must always trigger ───────────────

    def test_creates_real_trap_field_name_flagged(self):
        # Underscore field name in reasoning text is a schema leak.
        assert _has_contradiction(
            _hall("creates_real_trap=true confirms this is sound"),
            "forbidden term 'creates_real_trap'",
        )

    def test_restriction_score_field_name_flagged(self):
        assert _has_contradiction(
            _hall("restriction_score rises after this move"),
            "forbidden term 'restriction_score'",
        )

    def test_real_trap_english_phrase_flagged(self):
        # English prose form also remains forbidden.
        assert _has_contradiction(_hall("this creates a real trap"), "forbidden term 'real trap'")

    def test_counterplay_score_still_flagged(self):
        assert _has_contradiction(_hall("counterplay_score=0"), "forbidden term 'counterplay_score'")

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

    def test_new_vulnerabilities_positive_without_seed_flagged(self):
        # Positive form (not negated) — flagged when not seeded.
        assert _has_contradiction(
            _hall("this move creates new vulnerabilities in our position"),
            "term 'new vulnerabilities' used but not in seeds",
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
        assert "opponent can recapture" in self._text().lower()

    def test_threat_after_1_seed_present(self):
        text = self._text()
        assert "1 allied piece(s) remain under threat" in text

    def test_moved_piece_threatened_seed_present(self):
        assert "moved piece remains under immediate threat" in self._text()

    def test_capture_seed_present(self):
        assert "captures 1 piece(s)" in self._text()

    def test_threat_creation_seed_present(self):
        assert "immediate threat" in self._text()

    def test_minimax_seed_last_for_exposed_move(self):
        seeds = self._seeds()
        assert seeds[-1].startswith("The engine scores this move")

    # ── Checker does NOT flag correct acknowledgement of exposure ─────────────

    def test_checker_allows_honest_recapture_acknowledgement(self):
        """Reasoning that honestly says 'opponent can recapture' must NOT be flagged
        as a contradiction when opponent_can_recapture=true."""
        reasoning = (
            "Although the opponent can recapture the moved piece, "
            "this move captures a piece and creates an immediate threat. "
            "The engine scores this move -90.0 — best available option in a difficult position."
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
        """Seed text must acknowledge the recapture risk when opponent_can_recapture=true."""
        assert "opponent can recapture" in self._text().lower()

    def test_seeds_show_material_gain(self):
        assert "The move captures" in self._text()

    def test_seeds_show_threat_creation(self):
        assert "forces the opponent to respond" in self._text()

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
            "This jump captures a piece and creates an immediate threat, "
            "maintaining central control. The opponent can recapture, which is a "
            "risk, but one piece remains at risk while the tactical compensation "
            "justifies the exposure. "
            "The engine score of -90.0 confirms this over passive alternatives."
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
        """Seeds must acknowledge recapture risk AND show positive attributes."""
        seeds = _build_grounded_reasoning_seeds(
            _EXPOSED_BEST_MOVE, [_EXPOSED_BEST_MOVE, _PASSIVE_SAFE_MOVE]
        )
        text = " ".join(seeds)
        # Recapture risk is acknowledged
        assert "opponent can recapture" in text.lower()
        # Positive attributes are still present
        assert "The move captures" in text

    def test_checker_does_not_flag_threat_after_1_as_error(self):
        """Having threat_after=1 in facts must not cause the checker to
        automatically produce a contradiction about safety."""
        facts = {**_EXPOSED_BEST_FACTS, "our_pieces_threatened_after": 1}
        reasoning = (
            "The move captures a piece while leaving one ally at risk. "
            "The engine score of -90.0 confirms this choice."
        )
        warnings = _check_reasoning_truthfulness(reasoning, facts)
        # No false-positive contradiction about threat count
        threat_contradictions = [
            w for w in warnings
            if "our_pieces_threatened_after" in w and "REASONING_CONTRADICTION" in w
        ]
        assert threat_contradictions == []

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
        seeds = ["The opponent can recapture the moved piece next turn."]
        ws = self._inv("this move avoids recapture completely", seeds)
        assert self._has_inversion(ws)

    def test_seed_recapture_true_reasoning_says_no_recapture(self):
        seeds = ["The opponent can recapture the moved piece next turn."]
        ws = self._inv("there is no recapture risk here", seeds)
        assert self._has_inversion(ws)

    def test_seed_recapture_false_no_inversion_on_recapture_claim(self):
        seeds = ["The moved piece cannot be immediately recaptured."]
        ws = self._inv("this move avoids recapture", seeds)
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    # ── isolation inversions ─────────────────────────────────────────────────

    def test_seed_isolated_true_reasoning_says_no_isolation(self):
        seeds = ["The moved piece is left without adjacent support."]
        ws = self._inv("the piece has no isolation after this move", seeds)
        assert self._has_inversion(ws)

    def test_seed_isolated_true_reasoning_stays_connected(self):
        seeds = ["The moved piece is left without adjacent support."]
        ws = self._inv("the piece stays connected to friendly pieces", seeds)
        assert self._has_inversion(ws)

    def test_seed_isolated_false_no_inversion(self):
        seeds = ["The moved piece is not left isolated."]
        ws = self._inv("the piece stays connected", seeds)
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    # ── threat creation inversions ───────────────────────────────────────────

    def test_seed_threat_false_reasoning_claims_immediate_threat(self):
        # No seed is emitted when creates_immediate_threat=False; use facts dict
        # via the factual checker instead. The seed-inversion path is only for True.
        from checkers.agents.explainer_agent import _check_reasoning_truthfulness
        facts = {**_HALL_FACTS, "creates_immediate_threat": False}
        ws = _check_reasoning_truthfulness("this move creates a threat for next turn", facts)
        assert any("REASONING_CONTRADICTION" in w for w in ws)

    def test_seed_threat_true_reasoning_says_no_immediate_threat(self):
        seeds = ["This move forces the opponent to respond to an immediate threat."]
        ws = self._inv("there is no immediate threat from this move", seeds)
        assert self._has_inversion(ws)

    # ── king row inversions ──────────────────────────────────────────────────

    def test_seed_weakens_king_row_true_reasoning_says_preserves(self):
        seeds = ["The move weakens the back-row defensive structure."]
        ws = self._inv("this preserves back row integrity", seeds)
        assert self._has_inversion(ws)

    def test_seed_weakens_king_row_false_reasoning_says_weakened(self):
        seeds = ["A back-row piece is moved; the defensive structure remains intact."]
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
        assert any("gains material" in w or "material gain" in w for w in ws)


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

    def test_no_new_vulnerabilities_allowed(self):
        # Negated form — must pass the vocabulary check cleanly.
        ws = self._check("the move introduces no new vulnerabilities to our position")
        assert not any("new vulnerabilities" in w for w in ws)

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

    _ISOLATED_SEED = ["The moved piece is left without adjacent support."]

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
        seeds = ["The moved piece is not left isolated."]
        ws = _check_reasoning_truthfulness(
            "the move does not isolate the piece", _HALL_FACTS, seeds=seeds
        )
        inversion_ws = [w for w in ws if "inversion detected" in w]
        assert inversion_ws == []

    def test_no_inversion_when_isolated_false_and_remains_connected(self):
        seeds = ["The moved piece is not left isolated."]
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

    # Phase 1 — generic filler additions
    def test_improves_activity_in_forbidden_vocab(self):
        assert "improves activity" in _FORBIDDEN_VOCAB

    def test_piece_activity_in_forbidden_vocab(self):
        assert "piece activity" in _FORBIDDEN_VOCAB

    def test_more_active_position_in_forbidden_vocab(self):
        assert "more active position" in _FORBIDDEN_VOCAB

    def test_maintains_pressure_in_forbidden_vocab(self):
        assert "maintains pressure" in _FORBIDDEN_VOCAB

    def test_central_board_presence_in_context_forbidden_vocab(self):
        assert "central board presence" in _CONTEXT_FORBIDDEN_VOCAB

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
            "checkers.agents.explainer_agent.call_explainer",
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
            "checkers.agents.explainer_agent.call_explainer",
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
            "checkers.agents.explainer_agent.call_explainer",
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
            "checkers.agents.explainer_agent.call_explainer",
            side_effect=lambda *_: next(call_seq),
        ):
            _, count, _ = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=2
            )

        assert count >= 1
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
            "checkers.agents.explainer_agent.call_explainer",
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
            "checkers.agents.explainer_agent.call_explainer",
            return_value=good_replacement,
        ):
            result, _, resolved = _refine_reasoning(
                reasoning, _REPAIR_MOVE, contradictions, max_attempts=1
            )

        assert resolved is True
        assert "avoids recapture" not in result



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
        """Seeds where the comparison seed describes the chosen move (not isolated)."""
        return [
            "The moved piece cannot be immediately recaptured.",
            # Comparison seed: unlike the alt, the chosen move stays supported.
            "Unlike move [1], the moved piece remains supported by adjacent allies.",
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
            "The moved piece is left without adjacent support.",
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
        Comparison seed: unlike the alt, chosen move is safe from recapture.
        Reasoning correctly says 'avoids recapture' — must not trigger inversion.
        """
        seeds = [
            "The moved piece cannot be immediately recaptured.",
            "Unlike move [1], the chosen piece cannot be immediately recaptured.",
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
            "checkers.agents.explainer_agent.call_explainer",
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
            "checkers.agents.explainer_agent.call_explainer",
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
            "checkers.agents.explainer_agent.call_explainer",
            return_value=good_repl,
        ):
            result, _, resolved = _refine_reasoning(
                original, _REPAIR_MOVE, contradiction, max_attempts=1
            )

        assert resolved is True, f"Full paragraph should be clean. Got: {result!r}"
        assert "minimax evaluation supports" in result


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

    _ALT = {
        "type": "simple", "path": [[5, 2], [4, 1]], "captured": [],
        "facts": {"minimax_score": -99.0, "opponent_can_recapture": True,
                  "leaves_piece_isolated": True, "captures_count": 0, "net_gain": 0},
    }

    def _minimax_seed(self, move: dict) -> str:
        seeds = _build_grounded_reasoning_seeds(move, [move, self._ALT])
        mm_seeds = [s for s in seeds if s.startswith("The engine scores this move")]
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
        assert "-496.0" in seed

    def test_seed_is_always_the_last_seed(self):
        """Minimax seed must remain the last seed regardless of label."""
        move = self._move_with_score(-496.0)
        seeds = _build_grounded_reasoning_seeds(move, [move, self._ALT])
        assert seeds[-1].startswith("The engine scores this move")
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
    """Seeds for our_mobility and opponent_mobility must always be emitted.

    Phase-6 dedup: seeds now use the natural-language form only
    ('opponent mobility changes from X to Y' / 'remains at N').  The
    structured 'opponent_mobility_before=N, opponent_mobility_after=M' form
    is no longer emitted alongside it.
    """

    def test_opponent_mobility_seed_emitted_when_reduced(self):
        move = _move_with_mobility(opp_before=12, opp_after=8, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "opponent mobility changes from 12 to 8" in seeds_text

    def test_opponent_mobility_seed_emitted_when_equal(self):
        """Always emitted; equal values get the 'remains at N' form."""
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "opponent mobility remains at 10" in seeds_text

    def test_opponent_mobility_seed_emitted_when_increased(self):
        move = _move_with_mobility(opp_before=8, opp_after=12, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "opponent mobility changes from 8 to 12" in seeds_text

    def test_our_mobility_seed_emitted_when_improved(self):
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=5, our_after=8)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "our mobility changes from 5 to 8" in seeds_text

    def test_our_mobility_seed_emitted_when_reduced(self):
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=8, our_after=5)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "our mobility changes from 8 to 5" in seeds_text

    def test_our_mobility_seed_emitted_when_unchanged(self):
        move = _move_with_mobility(opp_before=10, opp_after=10, our_before=7, our_after=7)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "our mobility remains at 7" in seeds_text

    def test_no_duplicate_structured_form_emitted(self):
        """Regression guard: legacy 'key=N, key=M' form must NOT coexist with the
        natural-language form."""
        move = _move_with_mobility(opp_before=12, opp_after=8, our_before=6, our_after=6)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        seeds_text = " ".join(seeds)
        assert "opponent_mobility_before=12, opponent_mobility_after=8" not in seeds_text
        assert "our_mobility_before=6, our_mobility_after=6" not in seeds_text

    def test_one_mobility_seed_per_pair(self):
        """Exactly one seed contains each mobility direction's key phrase."""
        move = _move_with_mobility(opp_before=12, opp_after=8, our_before=6, our_after=5)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        opp_count = sum(1 for s in seeds if "opponent mobility" in s)
        our_count = sum(1 for s in seeds if "our mobility" in s)
        assert opp_count == 1, f"expected 1 opponent mobility seed, got {opp_count}: {seeds!r}"
        assert our_count == 1, f"expected 1 our mobility seed, got {our_count}: {seeds!r}"

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
        from checkers.agents.scorer_agent import compute_score_state
        board = self._build_board_red_winning()
        score_state = compute_score_state(board, RED)
        assert score_state in ("SLIGHTLY_WINNING", "CLEARLY_WINNING"), (
            f"RED has material advantage; expected winning score_state, got {score_state}"
        )

    def test_score_state_reflects_black_losing_when_black_to_move(self):
        """When RED is materially ahead and BLACK is to move, score_state reflects BLACK losing."""
        from checkers.engine.board import BLACK
        from checkers.agents.scorer_agent import compute_score_state
        board = self._build_board_red_winning()
        score_state = compute_score_state(board, BLACK)
        assert score_state in ("SLIGHTLY_LOSING", "CLEARLY_LOSING"), (
            f"BLACK is behind; expected losing score_state from BLACK perspective, "
            f"got {score_state}"
        )
# ══════════════════════════════════════════════════════════════════════════════
# Number-word before→after fact-grounded bypass
# ══════════════════════════════════════════════════════════════════════════════

class TestNumberWordBeforeAfterBypass:
    """
    Tests for the fact-grounded bypass in _check_reasoning_truthfulness that
    suppresses 'unsupported before→after numeric claim' when the LLM expresses
    a mobility or threatened-pieces transition in number-word form (e.g.
    'from six to four') but the fact dict contains the matching digit values.

    Coverage:
      1. opponent mobility — grounded word form → no warning
      2. opponent mobility — wrong numbers (mismatch) → warning fires
      3. our_mobility — grounded word form → no warning
      4. our_pieces_threatened — grounded word form → no warning
      5. no matching fact at all → warning fires
      6. digit form is still handled by existing rule (not affected)
    """

    def _facts(self, **kwargs) -> dict:
        base = {
            "opponent_can_recapture": False,
            "opponent_mobility_before": None,
            "opponent_mobility_after": None,
            "our_mobility_before": None,
            "our_mobility_after": None,
            "our_pieces_threatened_before": None,
            "our_pieces_threatened_after": None,
            "minimax_score": 5.0,
            "captures_count": 0,
            "net_gain": 0,
            "leaves_piece_isolated": False,
            "creates_immediate_threat": False,
            "results_in_king": False,
            "near_promotion": False,
            "center_control": False,
        }
        base.update(kwargs)
        return base

    # ── 1. opponent mobility word-form grounded → no warning ──────────────────

    def test_from_six_to_four_grounded_by_opponent_mobility(self):
        """
        'from six to four' with opponent_mobility_before=6, after=4 must NOT fire.
        This is the exact surface form the LLM uses that previously triggered
        the false-positive contradiction.
        """
        facts = self._facts(opponent_mobility_before=6, opponent_mobility_after=4)
        ws = _check_reasoning_truthfulness(
            "The move reduces opponent mobility from six to four, limiting replies.",
            facts,
            seeds=[
                "opponent_mobility_before=6, opponent_mobility_after=4 — "
                "reduces opponent mobility by 2, restricting available replies"
            ],
        )
        word_ws = [w for w in ws if "before→after numeric claim" in w]
        assert word_ws == [], (
            f"'from six to four' grounded by facts should NOT fire; got: {word_ws}"
        )

    def test_from_twelve_to_eight_grounded_by_opponent_mobility(self):
        """Larger mobility values — word form grounded → no warning."""
        facts = self._facts(opponent_mobility_before=8, opponent_mobility_after=6)
        ws = _check_reasoning_truthfulness(
            "Opponent mobility drops from eight to six after this move.",
            facts,
            seeds=[
                "opponent_mobility_before=8, opponent_mobility_after=6 — "
                "reduces opponent mobility by 2"
            ],
        )
        word_ws = [w for w in ws if "before→after numeric claim" in w]
        assert word_ws == [], (
            f"'from eight to six' grounded by facts should NOT fire; got: {word_ws}"
        )

    # ── 2. wrong numbers (mismatch) → warning must fire ───────────────────────

    def test_from_six_to_four_fires_when_facts_say_six_to_five(self):
        """
        Facts say 6→5 but LLM says 'from six to four' — mismatch, must fire.
        """
        facts = self._facts(opponent_mobility_before=6, opponent_mobility_after=5)
        ws = _check_reasoning_truthfulness(
            "The move reduces opponent mobility from six to four.",
            facts,
            seeds=[
                "opponent_mobility_before=6, opponent_mobility_after=5 — "
                "reduces opponent mobility by 1"
            ],
        )
        word_ws = [w for w in ws if "before→after numeric claim" in w]
        assert len(word_ws) > 0, (
            "'from six to four' with facts 6→5 should fire as contradiction"
        )

    def test_completely_fabricated_numbers_fire(self):
        """No fact pair matches 'from nine to three' when facts are 6→4."""
        facts = self._facts(opponent_mobility_before=6, opponent_mobility_after=4)
        ws = _check_reasoning_truthfulness(
            "Opponent moves shrink from nine to three.",
            facts,
            seeds=[
                "opponent_mobility_before=6, opponent_mobility_after=4 — "
                "reduces opponent mobility by 2"
            ],
        )
        word_ws = [w for w in ws if "before→after numeric claim" in w]
        assert len(word_ws) > 0, (
            "Fabricated 'from nine to three' must fire when facts are 6→4"
        )

    # ── 3. our_mobility — grounded word form → no warning ─────────────────────

    def test_from_three_to_two_grounded_by_our_mobility(self):
        """
        'from three to two' with our_mobility_before=3, after=2 → no warning.
        """
        facts = self._facts(our_mobility_before=3, our_mobility_after=2)
        ws = _check_reasoning_truthfulness(
            "Our mobility decreases from three to two after the capture.",
            facts,
            seeds=[
                "our_mobility_before=3, our_mobility_after=2 — "
                "decreases our mobility by 1"
            ],
        )
        word_ws = [w for w in ws if "before→after numeric claim" in w]
        assert word_ws == [], (
            f"'from three to two' grounded by our_mobility facts should NOT fire; "
            f"got: {word_ws}"
        )

    # ── 4. our_pieces_threatened — grounded word form → no warning ────────────

    def test_from_two_to_one_grounded_by_threatened_pieces(self):
        """
        'from two to one' with our_pieces_threatened_before=2, after=1 → no warning.
        """
        facts = self._facts(
            our_pieces_threatened_before=2,
            our_pieces_threatened_after=1,
        )
        ws = _check_reasoning_truthfulness(
            "The number of threatened pieces drops from two to one.",
            facts,
            seeds=[
                "our_pieces_threatened_after=1 — tactical drawback: "
                "1 allied piece(s) remain under attack after the move"
            ],
        )
        word_ws = [w for w in ws if "before→after numeric claim" in w]
        assert word_ws == [], (
            f"'from two to one' grounded by threatened_pieces facts should NOT fire; "
            f"got: {word_ws}"
        )

    # ── 5. no matching fact at all → warning fires ────────────────────────────

    def test_fires_when_no_fact_pair_present(self):
        """
        Number words in reasoning but none of the 3 fact pairs are set.
        Warning must fire (no grounding available).
        """
        facts = self._facts(
            opponent_mobility_before=None,
            opponent_mobility_after=None,
            our_mobility_before=None,
            our_mobility_after=None,
        )
        ws = _check_reasoning_truthfulness(
            "Opponent moves drop from six to four.",
            facts,
            seeds=["minimax_score=5.00 — highest-evaluated option"],
        )
        word_ws = [w for w in ws if "before→after numeric claim" in w]
        assert len(word_ws) > 0, (
            "No fact pair set — 'from six to four' must still fire as contradiction"
        )

    # ── 6. digit form still handled by existing rule (not broken) ─────────────

    def test_digit_form_from_6_to_4_still_passes_when_in_seeds(self):
        """
        Existing digit-form check: 'from 6 to 4' with both digits in seeds → passes.
        Confirms the new bypass does not interfere with the existing digit handler.
        """
        facts = self._facts(opponent_mobility_before=6, opponent_mobility_after=4)
        ws = _check_reasoning_truthfulness(
            "Opponent mobility falls from 6 to 4.",
            facts,
            seeds=[
                "opponent_mobility_before=6, opponent_mobility_after=4 — "
                "reduces opponent mobility by 2"
            ],
        )
        word_ws = [w for w in ws if "unsupported numeric statement" in w]
        assert word_ws == [], (
            f"Digit-form 'from 6 to 4' with digits in seeds must pass; got: {word_ws}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Bug-fix regression tests — false-positive fixes
# ══════════════════════════════════════════════════════════════════════════════

class TestFalsePositiveBugFixes:
    """
    Regression tests for the two confirmed false-positive bugs removed from
    _check_reasoning_truthfulness():

      Fix 1 — "escape" removed from _CONTEXT_FORBIDDEN_VOCAB.
        Valid tactical reasoning that uses "escape" in a natural context must no
        longer trigger a contradiction.  Specific compound phrases
        ("escape squares", "escape routes", "king escape") remain banned via
        _FORBIDDEN_VOCAB.

      Fix 2 — bare "immediate threat" removed from the creates_immediate_threat
        phrase list and from the creates_immediate_threat=false inversion pair.
        Correct negations such as "does not create an immediate threat" must no
        longer fire.  Positive-claim phrases ("creates a threat",
        "creates immediate threat", "creates an immediate threat") remain.
    """

    # ── Shared minimal fact dict ───────────────────────────────────────────────

    def _base_facts(self, **overrides) -> dict:
        base = {
            "opponent_can_recapture": False,
            "leaves_piece_isolated": False,
            "creates_immediate_threat": False,
            "center_control": False,
            "captures_count": 0,
            "net_gain": 0,
            "results_in_king": False,
            "near_promotion": False,
            "blocks_opponent_landing": False,
            "opponent_mobility_before": 6,
            "opponent_mobility_after": 4,
            "our_mobility_before": 3,
            "our_mobility_after": 3,
            "minimax_score": 5.0,
        }
        base.update(overrides)
        return base

    # =========================================================================
    # Fix 1 — "escape" in natural tactical context must NOT fire
    # =========================================================================

    def test_escape_in_valid_tactical_sentence_no_warning(self):
        """
        'The opponent cannot escape the capture' is valid tactical reasoning.
        Removing 'escape' from _CONTEXT_FORBIDDEN_VOCAB must suppress the warning.
        """
        facts = self._base_facts()
        seeds = [
            "opponent_mobility_before=6, opponent_mobility_after=4 — "
            "reduces opponent mobility by 2, restricting available replies",
        ]
        ws = _check_reasoning_truthfulness(
            "The opponent cannot escape the capture after this jump.",
            facts,
            seeds=seeds,
        )
        escape_ws = [w for w in ws if "escape" in w]
        assert escape_ws == [], (
            f"'escape' in valid tactical context must not fire; got: {escape_ws}"
        )

    def test_escape_as_verb_in_opponent_position_no_warning(self):
        """
        Variation: 'no route to escape' — natural English, no seed needed.
        """
        facts = self._base_facts()
        seeds = ["minimax_score=5.00 — highest-evaluated option"]
        ws = _check_reasoning_truthfulness(
            "With this capture, the opponent has no route to escape.",
            facts,
            seeds=seeds,
        )
        escape_ws = [w for w in ws if "escape" in w]
        assert escape_ws == [], (
            f"'escape' as plain verb must not fire; got: {escape_ws}"
        )

    def test_king_escape_still_banned_absolutely(self):
        """
        'king escape' remains in _FORBIDDEN_VOCAB (absolute ban) and must still fire.
        """
        facts = self._base_facts()
        seeds = ["minimax_score=5.00 — highest-evaluated option"]
        ws = _check_reasoning_truthfulness(
            "This move improves our king escape potential significantly.",
            facts,
            seeds=seeds,
        )
        escape_ws = [w for w in ws if "king escape" in w]
        assert len(escape_ws) > 0, (
            "'king escape' is still absolutely forbidden and must fire"
        )

    def test_escape_routes_still_banned_absolutely(self):
        """
        'escape routes' remains in _FORBIDDEN_VOCAB and must still fire.
        """
        facts = self._base_facts()
        seeds = ["minimax_score=5.00 — highest-evaluated option"]
        ws = _check_reasoning_truthfulness(
            "This limits the opponent's escape routes going forward.",
            facts,
            seeds=seeds,
        )
        escape_ws = [w for w in ws if "escape routes" in w]
        assert len(escape_ws) > 0, (
            "'escape routes' is still absolutely forbidden and must fire"
        )

    # =========================================================================
    # Fix 2 — "immediate threat" in negation context must NOT fire
    # =========================================================================

    def test_does_not_create_immediate_threat_no_warning(self):
        """
        'does not create an immediate threat' with creates_immediate_threat=False
        is CORRECT reasoning — must NOT trigger a contradiction.
        """
        facts = self._base_facts(creates_immediate_threat=False)
        seeds = [
            "opponent_mobility_before=6, opponent_mobility_after=4 — "
            "reduces opponent mobility by 2, restricting available replies",
            "minimax_score=5.00 — highest-evaluated option",
        ]
        ws = _check_reasoning_truthfulness(
            "Although this move does not create an immediate threat, "
            "it restricts available replies.",
            facts,
            seeds=seeds,
        )
        threat_ws = [w for w in ws if "creates_immediate_threat" in w]
        assert threat_ws == [], (
            f"Correct negation 'does not create an immediate threat' must NOT fire; "
            f"got: {threat_ws}"
        )

    def test_no_immediate_threat_phrasing_no_warning(self):
        """
        'no immediate threat' with creates_immediate_threat=False is also correct.
        This appeared in the inversion pair — removing it must suppress the warning.
        """
        facts = self._base_facts(creates_immediate_threat=False)
        seeds = ["minimax_score=5.00 — highest-evaluated option"]
        ws = _check_reasoning_truthfulness(
            "This positional move creates no immediate threat but improves structure.",
            facts,
            seeds=seeds,
        )
        # The inversion pair "creates_immediate_threat=true" → "no immediate threat"
        # only fires when seed says creates_immediate_threat=true, which it doesn't here.
        # But previously the bare phrase list also caught it via Rule 4.
        threat_ws = [w for w in ws if "creates_immediate_threat" in w]
        assert threat_ws == [], (
            f"'no immediate threat' when fact=false must NOT fire; got: {threat_ws}"
        )

    def test_creates_an_immediate_threat_still_fires(self):
        """
        'creates an immediate threat' with creates_immediate_threat=False
        IS a contradiction — the new specific phrase must still trigger it.
        """
        facts = self._base_facts(creates_immediate_threat=False)
        seeds = ["minimax_score=5.00 — highest-evaluated option"]
        ws = _check_reasoning_truthfulness(
            "This move creates an immediate threat against the opponent's back row.",
            facts,
            seeds=seeds,
        )
        threat_ws = [w for w in ws if "creates_immediate_threat" in w]
        assert len(threat_ws) > 0, (
            "'creates an immediate threat' with fact=False must still fire"
        )

    def test_creates_a_threat_still_fires(self):
        """
        'creates a threat' with creates_immediate_threat=False must still fire.
        This phrase was already in the list and must remain.
        """
        facts = self._base_facts(creates_immediate_threat=False)
        seeds = ["minimax_score=5.00 — highest-evaluated option"]
        ws = _check_reasoning_truthfulness(
            "This move creates a threat the opponent must respond to.",
            facts,
            seeds=seeds,
        )
        threat_ws = [w for w in ws if "creates_immediate_threat" in w]
        assert len(threat_ws) > 0, (
            "'creates a threat' with fact=False must still fire"
        )

    # =========================================================================
    # Strict rules still enforced — regression guard
    # =========================================================================

    def test_recapture_strict_still_fires(self):
        """Recapture rule unchanged — avoids-recapture claim with fact=True fires."""
        facts = self._base_facts(opponent_can_recapture=True)
        ws = _check_reasoning_truthfulness(
            "This move avoids recapture risk entirely.",
            facts,
            seeds=["opponent_can_recapture=true — opponent can recapture"],
        )
        recap_ws = [w for w in ws if "recapture" in w]
        assert len(recap_ws) > 0, "Recapture contradiction must still fire"

    def test_promotion_strict_still_fires(self):
        """Promotion rule unchanged — 'becomes a king' with fact=False fires."""
        facts = self._base_facts(results_in_king=False)
        ws = _check_reasoning_truthfulness(
            "The piece becomes a king after this move.",
            facts,
        )
        promo_ws = [w for w in ws if "promotion" in w or "results_in_king" in w]
        assert len(promo_ws) > 0, "Promotion contradiction must still fire"

    def test_material_gain_strict_still_fires(self):
        """Material gain rule unchanged — 'gains material' with net_gain=0 fires."""
        facts = self._base_facts(net_gain=0)
        ws = _check_reasoning_truthfulness(
            "This move gains material and improves our count.",
            facts,
        )
        mat_ws = [w for w in ws if "material" in w]
        assert len(mat_ws) > 0, "Material gain contradiction must still fire"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5.1 — Explicit mobility before→after seed wording + checker support
# ══════════════════════════════════════════════════════════════════════════════

class TestExplicitMobilityBeforeAfterSeeds:
    """
    Phase 5.1: the grounded reasoning seeds must include explicit natural
    'changes from X to Y' / 'remains at X' wording for both our mobility and
    opponent mobility, alongside the existing structured form.  The truthfulness
    checker must accept correct mobility numbers and still flag wrong ones.
    """

    def _move(self, our_before=None, our_after=None,
              opp_before=12, opp_after=8) -> dict:
        facts = {**_FULL_FACTS,
                 "opponent_mobility_before": opp_before,
                 "opponent_mobility_after":  opp_after}
        if our_before is not None:
            facts["our_mobility_before"] = our_before
        if our_after is not None:
            facts["our_mobility_after"] = our_after
        return {**_FULL_MOVE, "facts": facts}

    # ── 1. our-mobility changes-from wording ─────────────────────────────────

    def test_seeds_include_our_mobility_changes_from_when_different(self):
        move = self._move(our_before=7, our_after=8)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        joined = " ".join(seeds)
        assert "our mobility changes from 7 to 8" in joined

    # ── 2. our-mobility remains-at wording ───────────────────────────────────

    def test_seeds_include_our_mobility_remains_at_when_equal(self):
        move = self._move(our_before=7, our_after=7)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        joined = " ".join(seeds)
        assert "our mobility remains at 7" in joined

    # ── 3. opponent-mobility wording (both forms) ────────────────────────────

    def test_seeds_include_opponent_mobility_changes_from(self):
        move = self._move(opp_before=12, opp_after=8)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        joined = " ".join(seeds)
        assert "opponent mobility changes from 12 to 8" in joined

    def test_seeds_include_opponent_mobility_remains_at(self):
        move = self._move(opp_before=11, opp_after=11)
        seeds = _build_grounded_reasoning_seeds(move, [move])
        joined = " ".join(seeds)
        assert "opponent mobility remains at 11" in joined

    # ── 4. checker accepts correct mobility "from X to Y" ────────────────────

    def test_checker_accepts_correct_our_mobility_from_X_to_Y(self):
        """Checker must NOT flag 'from 7 to 8' when our_mobility_before=7,
        our_mobility_after=8, even with seeds=None."""
        facts = {**_FULL_FACTS,
                 "our_mobility_before": 7,
                 "our_mobility_after":  8}
        ws = _check_reasoning_truthfulness(
            "Our mobility changes from 7 to 8 after this move.",
            facts,
            seeds=None,
        )
        bad = [w for w in ws if "from 7 to 8" in w]
        assert bad == [], f"correct mobility phrasing should not flag: {ws}"

    def test_checker_accepts_correct_opponent_mobility_from_X_to_Y(self):
        facts = {**_FULL_FACTS,
                 "opponent_mobility_before": 12,
                 "opponent_mobility_after":  8}
        ws = _check_reasoning_truthfulness(
            "Opponent mobility drops from 12 to 8.",
            facts,
            seeds=None,
        )
        bad = [w for w in ws if "from 12 to 8" in w]
        assert bad == [], f"correct mobility phrasing should not flag: {ws}"

    def test_checker_accepts_correct_remains_at_when_equal(self):
        facts = {**_FULL_FACTS,
                 "our_mobility_before": 11,
                 "our_mobility_after":  11}
        ws = _check_reasoning_truthfulness(
            "Our mobility remains at 11.",
            facts,
            seeds=None,
        )
        bad = [w for w in ws if "remains at 11" in w]
        assert bad == [], f"correct stable mobility should not flag: {ws}"

    # ── 5. checker still flags wrong mobility numbers ────────────────────────

    def test_checker_flags_wrong_from_X_to_Y(self):
        """Wrong digits must still trip the numeric check (precision intact)."""
        facts = {**_FULL_FACTS,
                 "our_mobility_before": 7,
                 "our_mobility_after":  8,
                 "opponent_mobility_before": 12,
                 "opponent_mobility_after":  8}
        ws = _check_reasoning_truthfulness(
            "Our mobility changes from 99 to 1 after this move.",
            facts,
            seeds=None,
        )
        bad = [w for w in ws if "from 99 to 1" in w]
        assert bad, f"wrong digits must flag, got warnings: {ws}"

    def test_checker_flags_wrong_remains_at(self):
        facts = {**_FULL_FACTS,
                 "our_mobility_before": 7,
                 "our_mobility_after":  8}  # NOT equal — 'remains at' invalid
        ws = _check_reasoning_truthfulness(
            "Our mobility remains at 5.",
            facts,
            seeds=None,
        )
        bad = [w for w in ws if "remains at 5" in w]
        assert bad, f"wrong stable claim must flag, got warnings: {ws}"

    # ── 6. seeds path: checker accepts when digits come from the new seed ───

    def test_checker_accepts_via_seeds_path(self):
        """With the new natural-language seeds passed in, the seeds_text path
        in the checker should already accept 'from 7 to 8' without needing the
        fact-aware fallback (regression-locks the existing seed-based path)."""
        seeds = ["our mobility changes from 7 to 8"]
        # No facts dict — rely purely on seeds_text matching.
        ws = _check_reasoning_truthfulness(
            "Our mobility changes from 7 to 8.",
            facts={},
            seeds=seeds,
        )
        bad = [w for w in ws if "from 7 to 8" in w]
        assert bad == [], f"seed-path acceptance broken: {ws}"

    # ── 7. immutability ──────────────────────────────────────────────────────

    def test_facts_dict_not_mutated_by_seed_generation(self):
        move = self._move(our_before=7, our_after=8)
        before = copy.deepcopy(move)
        _build_grounded_reasoning_seeds(move, [move])
        assert move == before


# ══════════════════════════════════════════════════════════════════════════════
# Phase-6 Fix 1 — Prompt no longer teaches conflated/forbidden phrases
# ══════════════════════════════════════════════════════════════════════════════

class TestRankerSeedPromptDoesNotTeachForbiddenPhrases:
    """The seed-reasoning system prompt must not mention any phrase that the
    runtime truthfulness checker rejects.  Teaching a phrase the checker bans
    creates a repair-loop trap: the LLM uses what it sees, then the paragraph
    is rejected and refined.
    """

    PROMPT = RANKER_SEED_REASONING_SYSTEM

    def test_prompt_does_not_use_central_board_presence_as_example(self):
        # 'central board presence' must not appear in any PRIORITY example list
        # or anywhere outside the explicit Do-NOT-use vocabulary block.
        # Strategy: occurrences only appear inside the "Do NOT use" block now.
        # Find the Do-NOT-use block boundaries and assert no occurrences exist
        # before that block.
        idx_block = self.PROMPT.find("Do NOT use any of the following")
        assert idx_block != -1, "expected forbidden-vocab block in prompt"
        head = self.PROMPT[:idx_block]
        assert "central board presence" not in head, (
            "prompt teaches the conflated phrase 'central board presence' "
            "as an example — must use 'geometric center position' instead"
        )

    def test_prompt_introduces_geometric_center_position_phrase(self):
        # The safe center phrase is now emitted directly by the seed builder
        # ("The destination is in the center of the board (column X).").
        # The prompt must at least explicitly ban the conflated alternative.
        idx_block = self.PROMPT.find("Do NOT use any of the following")
        assert idx_block != -1, "expected forbidden-vocab block in prompt"
        tail = self.PROMPT[idx_block:]
        assert "central board presence" in tail, (
            "prompt must ban 'central board presence' to prevent LLM from using it"
        )

    def test_prompt_explicitly_bans_central_board_presence(self):
        # The phrase must still be explicitly listed in the Do-NOT-use block.
        idx_block = self.PROMPT.find("Do NOT use any of the following")
        tail = self.PROMPT[idx_block:]
        assert "central board presence" in tail

    def test_prompt_explicitly_bans_generic_filler_phrases(self):
        idx_block = self.PROMPT.find("Do NOT use any of the following")
        tail = self.PROMPT[idx_block:]
        for phrase in ("improves activity", "piece activity",
                       "more active position", "maintains pressure"):
            assert phrase in tail, f"prompt should ban {phrase!r} in Do-NOT-use block"


# ══════════════════════════════════════════════════════════════════════════════
# Phase-6 Fix 2 — semantic_ontology is the single source of truth
# ══════════════════════════════════════════════════════════════════════════════

class TestSemanticOntologyParity:
    """Every forbidden-conflation phrase declared in semantic_ontology must be
    enforced by the runtime checker.  Every generic-filler phrase declared in
    semantic_ontology must be in the absolute _FORBIDDEN_VOCAB.

    Editing the ontology must propagate to the runtime checker through the
    merge step in ranker_agent.py; this test guards that contract.
    """

    def test_all_forbidden_conflation_phrases_are_runtime_forbidden(self):
        from checkers.evaluation.semantic_ontology import FORBIDDEN_CONFLATION_PHRASES
        # Each phrase must be context-forbidden (i.e. allowed only if seeded)
        # or fully forbidden — never silently ignored.
        all_runtime = set(_FORBIDDEN_VOCAB) | set(_CONTEXT_FORBIDDEN_VOCAB)
        for phrase in FORBIDDEN_CONFLATION_PHRASES:
            assert phrase in all_runtime, (
                f"ontology phrase {phrase!r} not enforced by runtime checker"
            )

    def test_all_generic_filler_phrases_are_absolutely_forbidden(self):
        from checkers.evaluation.semantic_ontology import GENERIC_FILLER_PHRASES
        for phrase in GENERIC_FILLER_PHRASES:
            assert phrase in _FORBIDDEN_VOCAB, (
                f"ontology generic filler {phrase!r} not in _FORBIDDEN_VOCAB"
            )

    def test_central_influence_now_blocked_by_checker(self):
        # 'central influence' came from the ontology merge — it must fire when
        # the seed list does not introduce it.
        facts = {
            "opponent_can_recapture": False,
            "creates_immediate_threat": False,
            "center_control": False,
            "leaves_piece_isolated": False,
            "captures_count": 0,
            "net_gain": 0,
            "minimax_score": 1.0,
        }
        ws = _check_reasoning_truthfulness(
            "Maintains central influence over the board.", facts,
            seeds=["minimax_score=1.00 — highest-evaluated option"],
        )
        assert any("central influence" in w for w in ws), (
            f"checker did not block 'central influence' via ontology merge: {ws}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Phase-6 Fix 3 — adversity seeds gated on score_state
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversityGateOnScoreState:
    """Adversity context seeds must activate only in genuinely losing positions.

    Activation logic:
      - score_state in {CLEARLY_LOSING, SLIGHTLY_LOSING}: ON
      - score_state in {EQUAL, SLIGHTLY_WINNING, CLEARLY_WINNING}: OFF
        (even when the chosen move's per-move minimax_score is low,
        as in a forced-but-winning continuation).
      - score_state is None / missing: fall back to legacy raw-minimax gate.
    """

    def _move(self, mm: float, mat_adv: int = -2) -> dict:
        return {
            "type": "simple",
            "path": [[5, 0], [4, 1]],
            "captured": [],
            "facts": {
                "opponent_can_recapture": False,
                "leaves_piece_isolated": False,
                "creates_immediate_threat": False,
                "center_control": False,
                "results_in_king": False,
                "near_promotion": False,
                "captures_count": 0,
                "net_gain": 0,
                "minimax_score": mm,
                "material_advantage": mat_adv,
                "opponent_mobility_before": 10,
                "opponent_mobility_after": 10,
                "our_mobility_before": 4,
                "our_mobility_after": 4,
                "our_pieces_threatened_before": 1,
                "our_pieces_threatened_after": 0,
            },
        }

    def _has_adversity(self, seeds: list[str]) -> bool:
        joined = " ".join(seeds).lower()
        return any(marker in joined for marker in (
            "behind by",
            "material_advantage=-",
            "structural disadvantage",
            "reduces threatened pieces",
        ))

    def test_score_state_clearly_losing_activates_adversity(self):
        move = self._move(mm=-50.0)
        seeds = _build_grounded_reasoning_seeds(
            move, [move], score_state="CLEARLY_LOSING",
        )
        assert self._has_adversity(seeds), seeds

    def test_score_state_slightly_losing_activates_adversity(self):
        move = self._move(mm=-50.0)
        seeds = _build_grounded_reasoning_seeds(
            move, [move], score_state="SLIGHTLY_LOSING",
        )
        assert self._has_adversity(seeds), seeds

    def test_score_state_equal_suppresses_adversity_even_when_minimax_low(self):
        """Forced-but-winning line: per-move mm is low but the position is even.
        Adversity seeds must NOT fire — this is the bug we are fixing."""
        move = self._move(mm=-50.0, mat_adv=0)  # material is even
        seeds = _build_grounded_reasoning_seeds(
            move, [move], score_state="EQUAL",
        )
        assert not self._has_adversity(seeds), (
            f"adversity seeds wrongly fired with score_state=EQUAL: {seeds}"
        )

    def test_score_state_winning_suppresses_adversity_even_when_minimax_low(self):
        """Winning player choosing a forced move with low single-move mm must
        not be told the position is losing."""
        move = self._move(mm=-50.0, mat_adv=+3)  # we are materially ahead
        for ss in ("CLEARLY_WINNING", "SLIGHTLY_WINNING"):
            seeds = _build_grounded_reasoning_seeds(
                move, [move], score_state=ss,
            )
            assert not self._has_adversity(seeds), (
                f"adversity seeds wrongly fired with score_state={ss}: {seeds}"
            )

    def test_score_state_none_falls_back_to_minimax_gate(self):
        """Legacy behaviour preserved when no score_state is supplied:
        raw mm < -20 still activates adversity."""
        move_losing = self._move(mm=-50.0)
        seeds_losing = _build_grounded_reasoning_seeds(move_losing, [move_losing])
        assert self._has_adversity(seeds_losing)

        move_winning = self._move(mm=5.0)
        seeds_winning = _build_grounded_reasoning_seeds(move_winning, [move_winning])
        assert not self._has_adversity(seeds_winning)
