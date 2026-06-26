# checkers/tests/test_phase_g_step3_mobility_direction.py
#
# Phase G, Step 3 regression tests.  Targeted T4 mobility-direction grounding.
#
# Two minimal-safe additions:
#
#   (a) _mobility_gap_seed — emits a grounded direction seed
#       ('narrowed by N' / 'widened by N' / 'remained unchanged') whenever the
#       four mobility fields are present.
#
#   (b) Three exact-phrase verifier checks (defence-in-depth) covering:
#         - "mobility remained unchanged" / "stays unchanged" / variants
#         - "gap narrowed" / "gap narrows" / "gap narrowing"
#         - "gap widened" / "gap widens" / "gap widening"
#
# These do NOT modify minimax, move selection, repair architecture, or any
# Step 1/2 grounding system.

from __future__ import annotations

from checkers.agents.explainer_agent import (
    _build_grounded_reasoning_seeds,
    _check_reasoning_truthfulness,
    _mobility_gap_seed,
)
from checkers.evaluation.unified_verifier import (
    _check_mobility_direction_phrases,
    contradiction_strings,
)


def _facts(**overrides) -> dict:
    base = {
        "captures_count": 0,
        "creates_immediate_threat": False,
        "forced_opponent_jump_reply": False,
        "results_in_king": False,
        "opponent_can_recapture": False,
        "leaves_piece_isolated": False,
        "our_pieces_threatened_after": 0,
        "frozen_enemy_pieces": 0,
        "our_mobility_before": 8,
        "our_mobility_after": 8,
        "opponent_mobility_before": 8,
        "opponent_mobility_after": 8,
        "minimax_score": 0.0,
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
# 1.  _mobility_gap_seed — direction summary
# ═══════════════════════════════════════════════════════════════════════════

class TestGapSeedDirection:
    def test_unchanged_both_sides_static(self):
        # gap_before = gap_after = 0
        s = _mobility_gap_seed(_facts())
        assert s == "The mobility gap remained unchanged."

    def test_unchanged_symmetric_decrease(self):
        # Sample 13/17 case: both sides decrease by 1; gap unchanged at 1.
        s = _mobility_gap_seed(_facts(
            our_mobility_before=5, our_mobility_after=4,
            opponent_mobility_before=6, opponent_mobility_after=5,
        ))
        assert s == "The mobility gap remained unchanged."

    def test_widened_by_one(self):
        # Sample 8 case: our 4->4, opp 10->11. |4-10|=6, |4-11|=7. Widened by 1.
        s = _mobility_gap_seed(_facts(
            our_mobility_before=4, our_mobility_after=4,
            opponent_mobility_before=10, opponent_mobility_after=11,
        ))
        assert s == "The mobility gap widened by 1."

    def test_narrowed_by_two(self):
        # our 4->5, opp 10->9. |gap| 6 -> 4. Narrowed by 2.
        s = _mobility_gap_seed(_facts(
            our_mobility_before=4, our_mobility_after=5,
            opponent_mobility_before=10, opponent_mobility_after=9,
        ))
        assert s == "The mobility gap narrowed by 2."

    def test_narrowed_when_we_overtake(self):
        # our 9->10, opp 10->10. |gap| 1 -> 0. Narrowed by 1.
        s = _mobility_gap_seed(_facts(
            our_mobility_before=9, our_mobility_after=10,
            opponent_mobility_before=10, opponent_mobility_after=10,
        ))
        assert s == "The mobility gap narrowed by 1."

    def test_none_when_field_missing(self):
        f = _facts()
        f.pop("our_mobility_before")
        assert _mobility_gap_seed(f) is None
        assert _mobility_gap_seed(None) is None
        assert _mobility_gap_seed({}) is None

    def test_deterministic(self):
        f = _facts(our_mobility_after=10, opponent_mobility_after=11)
        assert _mobility_gap_seed(f) == _mobility_gap_seed(f)

    def test_no_strategic_or_causal_language(self):
        for fep_after_pair in [(8, 8), (10, 11), (5, 9), (11, 10)]:
            s = _mobility_gap_seed(_facts(
                our_mobility_after=fep_after_pair[0],
                opponent_mobility_after=fep_after_pair[1],
            ))
            if s is None:
                continue
            lower = s.lower()
            for banned in ("pressure", "control", "initiative", "influence",
                           "dominance", "structural", "tactical", "advantage",
                           "long-term"):
                assert banned not in lower, f"{banned!r} appeared in {s!r}"


class TestSeedBuilderEmitsGapSeed:
    """The seed builder must include the gap-direction seed when computable."""

    def _seeds(self, **fact_kw):
        chosen = _move(_facts(**fact_kw))
        alts = [chosen, _move(_facts(), path=[[5, 2], [4, 1]])]
        return _build_grounded_reasoning_seeds(chosen, alts)

    def test_widened_seed_present(self):
        s = self._seeds(
            our_mobility_before=4, our_mobility_after=4,
            opponent_mobility_before=10, opponent_mobility_after=11,
        )
        assert "The mobility gap widened by 1." in s

    def test_narrowed_seed_present(self):
        s = self._seeds(
            our_mobility_before=4, our_mobility_after=5,
            opponent_mobility_before=10, opponent_mobility_after=9,
        )
        assert "The mobility gap narrowed by 2." in s

    def test_unchanged_seed_present(self):
        s = self._seeds(
            our_mobility_before=5, our_mobility_after=4,
            opponent_mobility_before=6, opponent_mobility_after=5,
        )
        assert "The mobility gap remained unchanged." in s

    def test_seed_emitted_in_single_candidate_path(self):
        chosen = _move(_facts(
            our_mobility_before=4, our_mobility_after=4,
            opponent_mobility_before=10, opponent_mobility_after=11,
        ))
        seeds = _build_grounded_reasoning_seeds(chosen, [chosen])
        assert "The mobility gap widened by 1." in seeds
        # And the single-legal branch still emits its closing seed.
        assert any("only legal move available" in s.lower() for s in seeds)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Verifier check — three exact-phrase patterns
# ═══════════════════════════════════════════════════════════════════════════

class TestVerifierMobilityDirectionPhrases:
    def test_unchanged_misclaim_when_individual_decreased(self):
        # Sample 13/17 pattern.  Text says unchanged but individual mobs moved.
        records = _check_mobility_direction_phrases(
            "mobility remained unchanged for both sides.",
            _facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=5,
            ),
        )
        assert len(records) == 1
        assert records[0].claim_type == "mobility_unchanged_misclaim"

    def test_unchanged_claim_when_truly_unchanged(self):
        records = _check_mobility_direction_phrases(
            "mobility remained unchanged for both sides.",
            _facts(),  # all mobilities equal 8
        )
        assert records == []

    def test_gap_narrowed_when_actually_widened(self):
        # Sample 8 pattern.
        records = _check_mobility_direction_phrases(
            "narrowing the mobility gap slightly through this advance "
            "while the gap narrowed under the engine's pressure.",
            _facts(
                our_mobility_before=4, our_mobility_after=4,
                opponent_mobility_before=10, opponent_mobility_after=11,
            ),
        )
        # The phrase "the gap narrowed" should match.
        assert any(r.claim_type == "gap_did_not_narrow" for r in records)

    def test_gap_narrowed_when_actually_narrowed(self):
        records = _check_mobility_direction_phrases(
            "the gap narrowed thanks to the mobility shift.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        assert records == []

    def test_gap_widened_when_actually_narrowed(self):
        records = _check_mobility_direction_phrases(
            "the gap widened after this play.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        assert any(r.claim_type == "gap_did_not_widen" for r in records)

    def test_gap_widened_when_actually_widened(self):
        records = _check_mobility_direction_phrases(
            "the gap widened a little after the advance.",
            _facts(
                our_mobility_before=4, our_mobility_after=4,
                opponent_mobility_before=10, opponent_mobility_after=11,
            ),
        )
        assert records == []

    def test_missing_text_or_facts_returns_empty(self):
        assert _check_mobility_direction_phrases("", _facts()) == []
        assert _check_mobility_direction_phrases("any text", None) == []
        f = _facts(); f.pop("our_mobility_before")
        assert _check_mobility_direction_phrases(
            "mobility remained unchanged", f
        ) == []

    def test_does_not_overfire_on_unrelated_text(self):
        # Generic prose containing none of the targeted phrases must not fire.
        records = _check_mobility_direction_phrases(
            "The piece advances to the centre and remains supported.",
            _facts(our_mobility_after=7),  # any value
        )
        assert records == []

    def test_legacy_narrowing_the_gap_check_still_independent(self):
        # On honest claims, the Step 3 check should not fire even on the
        # legacy "narrowing the gap" form — this confirms the surgical fix
        # only flags FACTUAL violations, not phrasing.
        records = _check_mobility_direction_phrases(
            "narrowing the gap was the goal here.",
            # Honest case: our 8->10, opp 8->8.  Gap before |8-8|=0,
            # gap after |10-8|=2.  Gap actually widened — but the phrase
            # is "narrowing the gap" (verb-first form, post-G1 surgical fix
            # now covers this).  This should fire because the claim is wrong.
            _facts(our_mobility_after=10, opponent_mobility_after=8),
        )
        # Surgical fix: verb-first "narrowing the gap" now triggers when
        # the gap did not actually narrow.  Direction: gap before=0, after=2
        # — gap widened, not narrowed.  Expect gap_did_not_narrow.
        assert any(r.claim_type == "gap_did_not_narrow" for r in records)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  G1 surgical fix — verb-first phrasings ("narrowing the gap" etc.)
# ═══════════════════════════════════════════════════════════════════════════
#
# The Step 3 regex originally matched only the noun-first verb form
# ("gap narrowed/narrows/narrowing").  Audit sample S8/S29 escaped because
# the LLM used the verb-first form ("narrowing the gap in mobility").  This
# tiny regex-coverage patch extends both regexes to match either word order
# without changing any other behaviour.

class TestSurgicalG1VerbFirstFix:
    """Verb-first phrasings ('narrowing the gap', 'widening the gap')
    must now trigger the same violation checks as the noun-first form."""

    def test_narrowing_the_gap_fires_when_widened(self):
        # Audit S8: our 6->6, opp 8->9. Gap |6-8|=2 -> |6-9|=3. Widened.
        # The LLM wrote "narrowing the gap in mobility".
        records = _check_mobility_direction_phrases(
            "slightly narrowing the gap in mobility without creating a threat.",
            _facts(
                our_mobility_before=6, our_mobility_after=6,
                opponent_mobility_before=8, opponent_mobility_after=9,
            ),
        )
        assert any(r.claim_type == "gap_did_not_narrow" for r in records)

    def test_widening_the_gap_fires_when_narrowed(self):
        # our 4->5, opp 10->9. Gap |4-10|=6 -> |5-9|=4. Narrowed.
        # LLM claims "widening the gap" — should fire.
        records = _check_mobility_direction_phrases(
            "this play results in widening the gap considerably.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        assert any(r.claim_type == "gap_did_not_widen" for r in records)

    def test_narrows_the_gap_present_tense_form_also_caught(self):
        records = _check_mobility_direction_phrases(
            "the move narrows the gap.",
            _facts(
                our_mobility_before=4, our_mobility_after=4,
                opponent_mobility_before=8, opponent_mobility_after=9,
            ),
        )
        assert any(r.claim_type == "gap_did_not_narrow" for r in records)

    def test_widens_the_mobility_gap_form_also_caught(self):
        records = _check_mobility_direction_phrases(
            "the advance widens the mobility gap further.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        assert any(r.claim_type == "gap_did_not_widen" for r in records)

    def test_honest_narrowing_the_gap_does_not_fire(self):
        # Honest narrowing: gap actually did narrow.
        records = _check_mobility_direction_phrases(
            "this exchange is narrowing the gap as expected.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        # Gap before |4-10|=6, after |5-9|=4. Narrowed by 2. Claim is true.
        assert not any(
            r.claim_type in ("gap_did_not_narrow", "gap_did_not_widen")
            for r in records
        )

    def test_honest_widening_the_gap_does_not_fire(self):
        records = _check_mobility_direction_phrases(
            "the engine accepts widening the gap as the lesser evil.",
            _facts(
                our_mobility_before=6, our_mobility_after=6,
                opponent_mobility_before=8, opponent_mobility_after=9,
            ),
        )
        # Gap before |6-8|=2, after |6-9|=3. Widened by 1. Claim is true.
        assert not any(
            r.claim_type in ("gap_did_not_narrow", "gap_did_not_widen")
            for r in records
        )

    def test_runtime_mirror_catches_narrowing_the_gap_when_widened(self):
        # E.1 parity: the ranker_agent runtime mirror catches the same
        # verb-first phrasing under the same engine facts.
        warnings = _check_reasoning_truthfulness(
            "Two of their pieces lose forward movement, slightly narrowing "
            "the gap in mobility without creating a threat.",
            _facts(
                our_mobility_before=6, our_mobility_after=6,
                opponent_mobility_before=8, opponent_mobility_after=9,
            ),
        )
        assert any("gap_did_not_narrow" in w for w in warnings)

    def test_runtime_mirror_catches_widening_the_gap_when_narrowed(self):
        warnings = _check_reasoning_truthfulness(
            "The advance is widening the gap as planned.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        assert any("gap_did_not_widen" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Symmetric surgical fix — verb-first unchanged-mobility forms
# ═══════════════════════════════════════════════════════════════════════════
#
# Audit residuals S13 and S18 used the verb-first form "does not alter mobility
# for either side" while one or both sides had actually changed.  The Step 3
# unchanged-mobility regex only matched the noun-first form ("mobility ...
# remained/stays/is unchanged").  This extension mirrors the G1 gap-direction
# fix by adding a verb-first alternation arm, while preserving the side-
# qualifier exclusion via negative lookahead (so single-side claims like
# "does not alter our mobility" still don't fire).

class TestSymmetricUnchangedVerbFirstFix:
    """Verb-first phrasings about unchanged mobility now trigger the same
    'mobility_unchanged_misclaim' check as the noun-first form."""

    def test_does_not_alter_mobility_fires_when_changed(self):
        # Audit S13: our 6->6 (unchanged), opp 7->6 (CHANGED).
        # LLM wrote "does not alter mobility for either side" — false.
        records = _check_mobility_direction_phrases(
            "the move does not alter mobility for either side.",
            _facts(
                our_mobility_before=6, our_mobility_after=6,
                opponent_mobility_before=7, opponent_mobility_after=6,
            ),
        )
        assert any(r.claim_type == "mobility_unchanged_misclaim" for r in records)

    def test_did_not_change_mobility_fires_when_changed(self):
        records = _check_mobility_direction_phrases(
            "this advance did not change mobility on either side.",
            _facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=5,
            ),
        )
        assert any(r.claim_type == "mobility_unchanged_misclaim" for r in records)

    def test_doesnt_affect_mobility_fires_when_changed(self):
        records = _check_mobility_direction_phrases(
            "the play doesn't affect mobility globally.",
            _facts(
                our_mobility_before=8, our_mobility_after=9,
                opponent_mobility_before=9, opponent_mobility_after=9,
            ),
        )
        assert any(r.claim_type == "mobility_unchanged_misclaim" for r in records)

    def test_does_not_modify_mobility_fires_when_changed(self):
        records = _check_mobility_direction_phrases(
            "the move does not modify mobility in any meaningful way.",
            _facts(
                our_mobility_before=10, our_mobility_after=9,
                opponent_mobility_before=5, opponent_mobility_after=4,
            ),
        )
        assert any(r.claim_type == "mobility_unchanged_misclaim" for r in records)

    def test_honest_unchanged_claim_does_not_fire(self):
        # Both sides actually unchanged → no contradiction even with
        # verb-first phrasing.
        records = _check_mobility_direction_phrases(
            "the move does not alter mobility for either side.",
            _facts(
                our_mobility_before=6, our_mobility_after=6,
                opponent_mobility_before=7, opponent_mobility_after=7,
            ),
        )
        assert not any(
            r.claim_type == "mobility_unchanged_misclaim" for r in records
        )

    def test_does_not_alter_our_mobility_does_not_fire_on_global_check(self):
        # Side-qualified claim ("our mobility") — must NOT trigger the
        # GLOBAL both-sides check, even if our mobility actually changed.
        # (One-sided claims are validated by per-side checks elsewhere.)
        records = _check_mobility_direction_phrases(
            "the move does not alter our mobility this turn.",
            _facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=6,
            ),
        )
        assert not any(
            r.claim_type == "mobility_unchanged_misclaim" for r in records
        )

    def test_does_not_alter_opponent_mobility_does_not_fire_on_global_check(self):
        records = _check_mobility_direction_phrases(
            "the move does not alter opponent mobility for this turn.",
            _facts(
                our_mobility_before=5, our_mobility_after=5,
                opponent_mobility_before=6, opponent_mobility_after=5,
            ),
        )
        assert not any(
            r.claim_type == "mobility_unchanged_misclaim" for r in records
        )

    def test_does_not_alter_the_overall_mobility_caught(self):
        # Optional "the" / "overall" qualifiers permitted by the regex.
        records = _check_mobility_direction_phrases(
            "this advance does not alter the overall mobility of the position.",
            _facts(
                our_mobility_before=4, our_mobility_after=4,
                opponent_mobility_before=10, opponent_mobility_after=11,
            ),
        )
        assert any(r.claim_type == "mobility_unchanged_misclaim" for r in records)

    def test_noun_first_form_still_caught_after_extension(self):
        # Regression guard: extending the regex must not break the original
        # noun-first form.
        records = _check_mobility_direction_phrases(
            "mobility remained unchanged after this move.",
            _facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=5,
            ),
        )
        assert any(r.claim_type == "mobility_unchanged_misclaim" for r in records)

    def test_runtime_mirror_catches_does_not_alter_mobility(self):
        # E.1 parity: ranker_agent runtime mirror catches verb-first form
        # under the same facts.
        warnings = _check_reasoning_truthfulness(
            "The move advances quietly and does not alter mobility for either side.",
            _facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=5,
            ),
        )
        assert any("mobility_unchanged_misclaim" in w for w in warnings)

    def test_runtime_mirror_silent_on_side_qualified_verb_first(self):
        # Parity guard: side-qualified verb-first must NOT fire the global
        # check in the runtime mirror.
        warnings = _check_reasoning_truthfulness(
            "The advance does not alter our mobility this turn.",
            _facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=6,
            ),
        )
        assert not any("mobility_unchanged_misclaim" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  E.1 parity — runtime mirror in ranker_agent matches the verifier
# ═══════════════════════════════════════════════════════════════════════════

class TestRuntimeMirrorParity:
    def test_runtime_catches_mobility_unchanged_misclaim(self):
        warnings = _check_reasoning_truthfulness(
            "The move advances quietly and mobility remained unchanged for both sides.",
            _facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=5,
            ),
        )
        assert any("mobility_unchanged_misclaim" in w for w in warnings)

    def test_runtime_catches_gap_narrowed_when_widened(self):
        warnings = _check_reasoning_truthfulness(
            "After the move, the gap narrowed under our pressure.",
            _facts(
                our_mobility_before=4, our_mobility_after=4,
                opponent_mobility_before=10, opponent_mobility_after=11,
            ),
        )
        assert any("gap_did_not_narrow" in w for w in warnings)

    def test_runtime_catches_gap_widened_when_narrowed(self):
        warnings = _check_reasoning_truthfulness(
            "The gap widened decisively after this exchange.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        assert any("gap_did_not_widen" in w for w in warnings)

    def test_runtime_silent_when_direction_correct(self):
        # Honest claims should not fire any of the three direction checks.
        warnings = _check_reasoning_truthfulness(
            "The gap narrowed by two as our mobility increased.",
            _facts(
                our_mobility_before=4, our_mobility_after=5,
                opponent_mobility_before=10, opponent_mobility_after=9,
            ),
        )
        for w in warnings:
            assert "gap_did_not_narrow" not in w
            assert "gap_did_not_widen" not in w
            assert "mobility_unchanged_misclaim" not in w


# ═══════════════════════════════════════════════════════════════════════════
# 4.  contradiction_strings — end-to-end string format
# ═══════════════════════════════════════════════════════════════════════════

class TestContradictionStringRendering:
    def test_unchanged_misclaim_string_format(self):
        strs = contradiction_strings(
            "mobility remained unchanged for both sides.",
            facts=_facts(
                our_mobility_before=5, our_mobility_after=4,
                opponent_mobility_before=6, opponent_mobility_after=5,
            ),
            reasoning_seeds=[],
        )
        joined = " | ".join(strs)
        assert "mobility_unchanged_misclaim" in joined

    def test_gap_did_not_narrow_string_format(self):
        strs = contradiction_strings(
            "the gap narrowed in the end.",
            facts=_facts(
                our_mobility_before=4, our_mobility_after=4,
                opponent_mobility_before=10, opponent_mobility_after=11,
            ),
            reasoning_seeds=[],
        )
        joined = " | ".join(strs)
        assert "gap_did_not_narrow" in joined
        # Includes the engine's actual values for diagnostics
        assert "|gap_before|=6" in joined
        assert "|gap_after|=7" in joined


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Determinism + non-mutation invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestInvariants:
    def test_seed_helper_does_not_mutate_facts(self):
        f = _facts(our_mobility_after=10, opponent_mobility_after=11)
        before = dict(f)
        _mobility_gap_seed(f)
        assert f == before

    def test_seed_builder_deterministic(self):
        chosen = _move(_facts(our_mobility_after=10, opponent_mobility_after=11))
        alts = [chosen, _move(_facts(), path=[[5, 2], [4, 1]])]
        s1 = _build_grounded_reasoning_seeds(chosen, alts)
        s2 = _build_grounded_reasoning_seeds(chosen, alts)
        assert s1 == s2

    def test_verifier_check_pure_function(self):
        f = _facts(
            our_mobility_before=5, our_mobility_after=4,
            opponent_mobility_before=6, opponent_mobility_after=5,
        )
        f_before = dict(f)
        text = "mobility remained unchanged for both sides."
        _check_mobility_direction_phrases(text, f)
        _check_mobility_direction_phrases(text, f)
        assert f == f_before
