# checkers/tests/test_phase_g_quiet_routing.py
#
# Phase G, Step 1 regression tests.  Semantic-grounding overhaul targeted at
# the largest failure class identified by the human audit: the seed-reasoning
# prompt pushes the LLM to invent strategic narrative on quiet moves.
#
# Three minimal-safe changes are tested here:
#
#   (a) _classify_move_intent  — routes a chosen-move's facts dict to either
#        'tactical' or 'quiet' based on a pure-function predicate.
#
#   (b) _negative_grounding_seeds — emits explicit negative-fact seeds for the
#        four predicates the audit identified as most commonly fabricated as
#        positives: creates_immediate_threat, forced_opponent_jump_reply,
#        frozen_enemy_pieces, and forced_move_for_us.
#
#   (c) _build_seed_reasoning_prompt routing — the quiet-class branch produces
#        a shorter prompt that bans the strategic-vocabulary set from the
#        audit's hollow-prose category; the tactical-class branch is unchanged.
#
# These tests do NOT touch move selection, minimax, verifier, repair-loop, or
# diagnostics schema.

from __future__ import annotations

from checkers.agents.explainer_agent import (
    _build_seed_reasoning_prompt,
    _build_grounded_reasoning_seeds,
    _classify_move_intent,
    _negative_grounding_seeds,
)


# ── Shared fixtures ─────────────────────────────────────────────────────────

def _facts(**overrides) -> dict:
    """Default 'clean quiet' fact dict; overrides flip specific predicates."""
    base = {
        "captures_count": 0,
        "net_gain": 0,
        "creates_immediate_threat": False,
        "forced_opponent_jump_reply": False,
        "results_in_king": False,
        "opponent_can_recapture": False,
        "leaves_piece_isolated": False,
        "our_pieces_threatened_after": 0,
        "our_mobility_before": 8,
        "our_mobility_after": 8,
        "opponent_mobility_before": 8,
        "opponent_mobility_after": 8,
        "frozen_enemy_pieces": 0,
        "minimax_score": -2.0,
        "move_type": "simple",
    }
    base.update(overrides)
    return base


def _move(facts: dict, path=None) -> dict:
    return {
        "path": path or [[5, 4], [4, 3]],
        "type": facts.get("move_type", "simple"),
        "facts": facts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1.  _classify_move_intent — quiet vs tactical routing
# ═══════════════════════════════════════════════════════════════════════════

class TestQuietClassifier:
    """Deterministic predicate over the chosen-move facts dict."""

    def test_default_quiet_facts_classified_quiet(self):
        assert _classify_move_intent(_facts()) == "quiet"

    def test_captures_makes_tactical(self):
        assert _classify_move_intent(_facts(captures_count=1)) == "tactical"

    def test_immediate_threat_makes_tactical(self):
        assert _classify_move_intent(_facts(creates_immediate_threat=True)) == "tactical"

    def test_forced_opp_jump_makes_tactical(self):
        assert _classify_move_intent(_facts(forced_opponent_jump_reply=True)) == "tactical"

    def test_king_promotion_makes_tactical(self):
        assert _classify_move_intent(_facts(results_in_king=True)) == "tactical"

    def test_our_mobility_jump_makes_tactical(self):
        # Δmob = +2 on our side exceeds the small-change threshold of 1
        assert _classify_move_intent(
            _facts(our_mobility_before=8, our_mobility_after=10)
        ) == "tactical"

    def test_opponent_mobility_jump_makes_tactical(self):
        assert _classify_move_intent(
            _facts(opponent_mobility_before=8, opponent_mobility_after=6)
        ) == "tactical"

    def test_one_mobility_delta_still_quiet(self):
        # |Δ| = 1 on each side stays inside the quiet band
        assert _classify_move_intent(
            _facts(our_mobility_before=8, our_mobility_after=9,
                   opponent_mobility_before=8, opponent_mobility_after=7)
        ) == "quiet"

    def test_missing_fields_does_not_force_tactical(self):
        # Absence of optional fields must not trigger tactical
        assert _classify_move_intent({"captures_count": 0}) == "quiet"

    def test_none_facts_dict_is_quiet(self):
        # Defensive: a None facts dict should not crash and should default quiet.
        assert _classify_move_intent(None) == "quiet"

    def test_deterministic_across_repeated_calls(self):
        f = _facts(captures_count=1)
        assert _classify_move_intent(f) == _classify_move_intent(f) == "tactical"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  _negative_grounding_seeds — four target negatives
# ═══════════════════════════════════════════════════════════════════════════

class TestNegativeGroundingSeeds:
    """The four targeted negatives mirror the four most-commonly-fabricated
    positives from the human audit."""

    def test_emits_no_threat_negative_when_cit_false(self):
        out = _negative_grounding_seeds(_facts(creates_immediate_threat=False), 2, [])
        assert any("does not create an immediate threat" in s for s in out)

    def test_does_not_emit_threat_negative_when_cit_true(self):
        out = _negative_grounding_seeds(_facts(creates_immediate_threat=True), 2, [])
        assert not any("does not create an immediate threat" in s for s in out)

    def test_emits_opp_not_forced_when_forced_jump_false(self):
        out = _negative_grounding_seeds(
            _facts(forced_opponent_jump_reply=False), 2, []
        )
        assert any("opponent is not forced to respond with a jump" in s.lower() for s in out)

    def test_does_not_emit_opp_not_forced_when_forced_jump_true(self):
        out = _negative_grounding_seeds(
            _facts(forced_opponent_jump_reply=True), 2, []
        )
        assert not any("not forced to respond with a jump" in s.lower() for s in out)

    def test_emits_no_restriction_negative_when_frozen_zero(self):
        out = _negative_grounding_seeds(_facts(frozen_enemy_pieces=0), 2, [])
        assert any("does not restrict any opponent piece" in s.lower() for s in out)

    def test_does_not_emit_restriction_negative_when_frozen_nonzero(self):
        out = _negative_grounding_seeds(_facts(frozen_enemy_pieces=2), 2, [])
        assert not any("does not restrict" in s.lower() for s in out)

    def test_emits_multiple_legal_when_n_candidates_gt_one(self):
        out = _negative_grounding_seeds(_facts(), 5, [])
        assert any("multiple legal moves were available" in s.lower() for s in out)

    def test_does_not_emit_multiple_legal_when_only_one_candidate(self):
        out = _negative_grounding_seeds(_facts(), 1, [])
        assert not any("multiple legal moves" in s.lower() for s in out)

    def test_does_not_duplicate_when_seed_already_present(self):
        existing = ["The opponent is forced to respond with a jump (1 piece)."]
        out = _negative_grounding_seeds(
            _facts(forced_opponent_jump_reply=False), 2, existing
        )
        # The existing positive-seed text contains "forced to respond with a jump",
        # which suppresses the negative to avoid duplication.
        assert not any("not forced to respond with a jump" in s.lower() for s in out)

    def test_only_legal_move_existing_suppresses_multiple_legal_negative(self):
        existing = ["This is the only legal move available."]
        out = _negative_grounding_seeds(_facts(), 5, existing)
        assert not any("multiple legal moves" in s.lower() for s in out)

    def test_no_negatives_when_all_positive(self):
        # All predicates flipped to positive → no negatives emitted.
        # opponent_mobility_after < opponent_mobility_before so the
        # "mobility unchanged" negative (added in Phase G) does not fire.
        out = _negative_grounding_seeds(
            _facts(
                creates_immediate_threat=True,
                forced_opponent_jump_reply=True,
                frozen_enemy_pieces=2,
                opponent_mobility_after=6,  # reduced from default 8 → positive outcome
            ),
            n_candidates=1,
            existing_seeds=[],
        )
        assert out == []


# ═══════════════════════════════════════════════════════════════════════════
# 3.  _build_grounded_reasoning_seeds — negatives are injected
# ═══════════════════════════════════════════════════════════════════════════

class TestSeedBuilderIntegratesNegatives:
    """The seed builder's two return paths (single-candidate and
    multi-candidate) both inject the negative seeds before closing."""

    def test_multi_candidate_quiet_move_includes_negatives(self):
        chosen = _move(_facts())
        alts = [chosen, _move(_facts(), path=[[5, 2], [4, 1]])]
        seeds = _build_grounded_reasoning_seeds(chosen, alts)
        text = " | ".join(seeds).lower()
        assert "does not create an immediate threat" in text
        assert "opponent is not forced to respond with a jump" in text
        assert "does not restrict any opponent piece" in text
        assert "multiple legal moves were available" in text

    def test_single_candidate_forced_path_includes_negatives_except_multiple_legal(self):
        # Single candidate → "only legal move" path; "multiple legal" negative
        # should be suppressed but the other three should still fire.
        chosen = _move(_facts())
        seeds = _build_grounded_reasoning_seeds(chosen, [chosen])
        text = " | ".join(seeds).lower()
        assert "only legal move available" in text
        assert "multiple legal moves were available" not in text
        # The other three are still relevant
        assert "does not create an immediate threat" in text
        assert "opponent is not forced to respond with a jump" in text

    def test_tactical_capture_move_omits_negatives_correctly(self):
        # Captures+forced opp reply → these positives present, their negatives
        # must NOT appear (existing positive seed suppresses the negative).
        chosen = _move(_facts(
            captures_count=1, net_gain=1,
            forced_opponent_jump_reply=True,
            opponent_can_recapture=True,
        ))
        alts = [chosen, _move(_facts(), path=[[7, 0], [6, 1]])]
        seeds = _build_grounded_reasoning_seeds(chosen, alts)
        text = " | ".join(seeds).lower()
        assert "not forced to respond with a jump" not in text  # positive present


# ═══════════════════════════════════════════════════════════════════════════
# 4.  _build_seed_reasoning_prompt — quiet vs tactical routing
# ═══════════════════════════════════════════════════════════════════════════

# Quiet-mode forbidden vocabulary set (matches the audit's hollow-prose targets).
_QUIET_FORBIDDEN_WORDS = (
    "pressure", "forcing", "control", "influence",
    "initiative", "long-term", "structural",
    "positional pressure", "tactical pressure",
    "restricts", "restricting", "restriction",
    "narrows the gap",
)


class TestPromptRouting:
    """The prompt builder branches on move class.  Quiet branch is short and
    bans strategic vocabulary; tactical branch is unchanged."""

    def test_quiet_prompt_announces_move_class(self):
        chosen = _move(_facts())
        prompt = _build_seed_reasoning_prompt(chosen, ["S1.", "S2."])
        assert "Move class: quiet" in prompt

    def test_tactical_prompt_announces_move_class(self):
        chosen = _move(_facts(captures_count=2))
        prompt = _build_seed_reasoning_prompt(chosen, ["S1."])
        assert "Move class: tactical" in prompt

    def test_quiet_prompt_requests_short_paragraph(self):
        prompt = _build_seed_reasoning_prompt(_move(_facts()), [])
        assert "2 OR 3 sentences" in prompt or "2 or 3 sentences" in prompt.lower()

    def test_tactical_prompt_requests_3_to_5_sentences(self):
        prompt = _build_seed_reasoning_prompt(_move(_facts(captures_count=1)), [])
        assert "3-5 sentences" in prompt

    def test_quiet_prompt_bans_strategic_vocabulary(self):
        prompt = _build_seed_reasoning_prompt(_move(_facts()), []).lower()
        # The quiet prompt must explicitly list these words as forbidden.
        for word in _QUIET_FORBIDDEN_WORDS:
            assert word in prompt, f"quiet prompt missing forbidden word: {word!r}"

    def test_tactical_prompt_does_not_apply_strategic_ban(self):
        # The tactical prompt does NOT carry the quiet-mode positional vocabulary ban.
        # "tactical pressure" is correctly in the vague-filler ban even for tactical
        # prompts (it is generic filler, not a grounded tactical claim).
        prompt = _build_seed_reasoning_prompt(_move(_facts(captures_count=1)), [])
        assert "positional pressure" not in prompt.lower()

    def test_quiet_prompt_retains_forced_move_rule(self):
        prompt = _build_seed_reasoning_prompt(_move(_facts()), []).lower()
        assert "only legal move available" in prompt
        assert "do not assert that the move is forced" in prompt

    def test_quiet_prompt_retains_coordinate_hint(self):
        path = [[5, 4], [4, 3]]
        chosen = {"path": path, "type": "simple", "facts": _facts()}
        prompt = _build_seed_reasoning_prompt(chosen, [])
        assert "reference the move path" in prompt
        assert str(path) in prompt

    def test_quiet_prompt_mentions_negative_seed_usage(self):
        prompt = _build_seed_reasoning_prompt(_move(_facts()), []).lower()
        # The quiet prompt explicitly instructs the LLM to USE the negative
        # wording rather than asserting the opposite.
        assert "negative fact" in prompt or "negative" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Determinism — both helpers are pure functions
# ═══════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """The new helpers must be deterministic and side-effect-free."""

    def test_classify_is_pure(self):
        f = _facts(captures_count=1)
        v1 = _classify_move_intent(f)
        v2 = _classify_move_intent(f)
        v3 = _classify_move_intent(dict(f))
        assert v1 == v2 == v3 == "tactical"
        # Input dict unchanged.
        assert f["captures_count"] == 1

    def test_negative_seeds_pure(self):
        f = _facts()
        seeds = ["existing"]
        out1 = _negative_grounding_seeds(f, 2, seeds)
        out2 = _negative_grounding_seeds(f, 2, seeds)
        assert out1 == out2
        # Input not mutated.
        assert seeds == ["existing"]

    def test_prompt_is_deterministic(self):
        chosen = _move(_facts())
        p1 = _build_seed_reasoning_prompt(chosen, ["S1.", "S2."])
        p2 = _build_seed_reasoning_prompt(chosen, ["S1.", "S2."])
        assert p1 == p2

    def test_full_seed_builder_deterministic(self):
        chosen = _move(_facts())
        alts = [chosen, _move(_facts(), path=[[5, 2], [4, 1]])]
        s1 = _build_grounded_reasoning_seeds(chosen, alts)
        s2 = _build_grounded_reasoning_seeds(chosen, alts)
        assert s1 == s2


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Move-selection / diagnostics invariants — verify nothing leaks
# ═══════════════════════════════════════════════════════════════════════════

class TestInvariantsPreserved:
    """The new code must not modify chosen_move, must not introduce schema
    leaks into the prompt, and must not affect minimax outputs."""

    def test_chosen_move_dict_is_not_mutated_by_classifier(self):
        chosen = _move(_facts(captures_count=2))
        before = dict(chosen["facts"])
        _classify_move_intent(chosen.get("facts"))
        assert chosen["facts"] == before

    def test_chosen_move_dict_is_not_mutated_by_prompt_builder(self):
        chosen = _move(_facts())
        before_facts = dict(chosen["facts"])
        before_path = list(chosen["path"])
        _build_seed_reasoning_prompt(chosen, ["S1.", "S2."])
        assert chosen["facts"] == before_facts
        assert chosen["path"] == before_path

    def test_negative_seed_helper_does_not_mutate_inputs(self):
        f = _facts()
        seeds = ["S1.", "S2."]
        f_before = dict(f)
        s_before = list(seeds)
        _negative_grounding_seeds(f, 3, seeds)
        assert f == f_before
        assert seeds == s_before

    def test_prompt_does_not_leak_internal_diagnostics_field_names(self):
        prompt = _build_seed_reasoning_prompt(_move(_facts()), [])
        # Sanity check: no diagnostic field names should appear verbatim.
        for tok in ("explainer_diagnostics", "raw_llm_reasoning_pre_refinement",
                    "chosen_move_facts", "final_choice_source"):
            assert tok not in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Step 2 — Restriction-count grounding (positive frozen-pieces seed)
# ═══════════════════════════════════════════════════════════════════════════
#
# Audit T1 finding: when frozen_enemy_pieces > 0 the engine knows the count
# but the seed list previously did not surface it; the LLM invented its own
# integer ("restricts three opponent pieces from advancing").  Step 2 emits
# a grounded positive seed with the real count and correct singular/plural
# grammar.  The Step 1 negative ("does not restrict any opponent piece's
# forward movement") fires only when the count is zero, so positive and
# negative are mutually exclusive by value.

class TestStep2RestrictionGrounding:
    """Step 2: emit a positive frozen-pieces seed when frozen_enemy_pieces > 0."""

    @staticmethod
    def _seeds_for(fep_value: int):
        chosen = _move(_facts(frozen_enemy_pieces=fep_value))
        alts = [chosen, _move(_facts(), path=[[5, 2], [4, 1]])]
        return _build_grounded_reasoning_seeds(chosen, alts)

    def test_frozen_zero_emits_negative_only(self):
        seeds = self._seeds_for(0)
        text = " | ".join(seeds).lower()
        assert "does not restrict any opponent piece" in text
        assert "have restricted forward movement" not in text
        assert "has restricted forward movement" not in text

    def test_frozen_one_emits_singular_positive(self):
        seeds = self._seeds_for(1)
        text = " | ".join(seeds)
        assert "1 opponent piece has restricted forward movement after this move." in text
        # Singular grammar: 'piece has', not 'pieces have'
        assert "1 opponent pieces have" not in text
        # The negative must NOT be present
        text_l = text.lower()
        assert "does not restrict any opponent piece" not in text_l

    def test_frozen_three_emits_plural_positive(self):
        seeds = self._seeds_for(3)
        text = " | ".join(seeds)
        assert "3 opponent pieces have restricted forward movement after this move." in text
        # Plural grammar: 'pieces have', not 'piece has'
        assert "3 opponent piece has" not in text
        text_l = text.lower()
        assert "does not restrict any opponent piece" not in text_l

    def test_frozen_two_uses_plural(self):
        # Boundary: 2 must be plural ("pieces have").
        seeds = self._seeds_for(2)
        text = " | ".join(seeds)
        assert "2 opponent pieces have restricted forward movement after this move." in text

    def test_no_simultaneous_positive_and_negative(self):
        # Across the full plausible range, never both seeds at once.
        for fep in (0, 1, 2, 3, 5, 8):
            seeds = self._seeds_for(fep)
            text = " | ".join(seeds).lower()
            has_pos = "have restricted forward movement" in text or "has restricted forward movement" in text
            has_neg = "does not restrict any opponent piece" in text
            assert not (has_pos and has_neg), (
                f"fep={fep}: both positive and negative present — mutual exclusion broken"
            )

    def test_positive_seed_absent_when_field_missing(self):
        # Defensive: if frozen_enemy_pieces is not present in facts, do not
        # emit either the positive or the negative.
        f = _facts()
        f.pop("frozen_enemy_pieces", None)
        chosen = _move(f)
        alts = [chosen, _move(f, path=[[5, 2], [4, 1]])]
        seeds = _build_grounded_reasoning_seeds(chosen, alts)
        text = " | ".join(seeds).lower()
        assert "have restricted forward movement" not in text
        assert "has restricted forward movement" not in text
        # The Step 1 negative requires fep == 0 explicitly, so a missing
        # field should produce neither seed.
        assert "does not restrict any opponent piece" not in text

    def test_positive_seed_uses_engine_count_exactly(self):
        # The exact engine integer must be used; no rounding or substitution.
        for fep in (1, 4, 7, 11):
            seeds = self._seeds_for(fep)
            text = " | ".join(seeds)
            assert f"{fep} opponent" in text

    def test_no_strategic_interpretation_in_positive_seed(self):
        seeds = self._seeds_for(3)
        text = " | ".join(seeds).lower()
        # The seed text must NOT smuggle in strategic interpretation.
        for banned in ("pressure", "initiative", "control", "influence",
                       "dominance", "structural", "long-term"):
            # The banned token must not appear in the new positive seed text.
            # (It may appear elsewhere in the corpus via legacy seeds, so we
            #  search only the line(s) that match the new seed phrase.)
            matched_lines = [s for s in seeds
                             if "restricted forward movement" in s.lower()]
            for line in matched_lines:
                assert banned not in line.lower(), (
                    f"positive frozen-pieces seed leaks strategic word "
                    f"{banned!r}: {line!r}"
                )

    def test_no_schema_syntax_in_positive_seed(self):
        seeds = self._seeds_for(3)
        for line in seeds:
            if "restricted forward movement" in line.lower():
                assert "frozen_enemy_pieces" not in line
                assert "=" not in line
                assert "{" not in line

    def test_positive_seed_deterministic(self):
        a = self._seeds_for(3)
        b = self._seeds_for(3)
        assert a == b

    def test_positive_seed_in_single_candidate_path(self):
        # The single-legal-move branch also returns seeds; confirm the
        # positive frozen-pieces seed fires there too when applicable.
        chosen = _move(_facts(frozen_enemy_pieces=2))
        seeds = _build_grounded_reasoning_seeds(chosen, [chosen])
        text = " | ".join(seeds)
        assert "2 opponent pieces have restricted forward movement after this move." in text
        assert "only legal move available" in text.lower()

    def test_chosen_move_dict_not_mutated_by_positive_seed(self):
        f = _facts(frozen_enemy_pieces=4)
        chosen = _move(f)
        alts = [chosen, _move(_facts(), path=[[5, 2], [4, 1]])]
        before = dict(chosen["facts"])
        _build_grounded_reasoning_seeds(chosen, alts)
        assert chosen["facts"] == before
