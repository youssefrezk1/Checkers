# checkers/tests/test_comparative_verifier.py
#
# Step 3 of the Comparative Reasoning v2 roadmap: focused unit tests for
# the comparative verifier in `checkers/agents/comparative_reasoner.py`.
#
# Coverage matrix (locked by the roadmap):
#   - invalid index
#   - self-reference
#   - per-alt mismatch
#   - grouped-claim partial truth
#   - invalid tradeoff
#   - valid prose
#   - empty input
#   - chosen-move verifier isolation invariant

from __future__ import annotations

import inspect

from checkers.agents.comparative_reasoner import (
    ComparativeContradiction,
    sanitize_comparative_contradiction,
    verify_comparative_reasoning,
)


# ── Test fixtures ───────────────────────────────────────────────────────────


def _move(path, **facts):
    return {"path": path, "facts": dict(facts)}


CHOSEN_SAFE = _move(
    [(5, 4), (4, 3)],
    opponent_can_recapture=False,
    creates_immediate_threat=False,
    shot_sequence_available=False,
    captures_count=0,
    net_gain=0,
    leaves_piece_isolated=False,
    weakens_king_row=False,
    results_in_king=False,
    near_promotion=False,
    our_pieces_threatened_after=0,
    opponent_mobility_before=8,
    opponent_mobility_after=8,
)


def _alt_aggressive(path, opp_recap=True):
    return _move(
        path,
        creates_immediate_threat=True,
        shot_sequence_available=False,
        opponent_can_recapture=opp_recap,
        captures_count=0,
        net_gain=0,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        our_pieces_threatened_after=0,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
    )


def _alt_quiet(path):
    return _move(
        path,
        creates_immediate_threat=False,
        shot_sequence_available=False,
        opponent_can_recapture=False,
        captures_count=0,
        net_gain=0,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        our_pieces_threatened_after=0,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Empty input
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyInput:
    def test_empty_text_returns_empty_list(self):
        result = verify_comparative_reasoning("", [CHOSEN_SAFE], CHOSEN_SAFE)
        assert result == []

    def test_none_text_returns_empty_list(self):
        result = verify_comparative_reasoning(None, [CHOSEN_SAFE], CHOSEN_SAFE)
        assert result == []

    def test_empty_candidates_returns_empty_list(self):
        result = verify_comparative_reasoning(
            "Aggressive alternative [1] creates a threat.", [], CHOSEN_SAFE,
        )
        assert result == []

    def test_none_chosen_move_no_crash(self):
        # Should not raise even when chosen_move is None.
        result = verify_comparative_reasoning(
            "Aggressive alternative [1] creates a threat.",
            [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])],
            None,
        )
        # Returns whatever it could verify (no self-reference check, but
        # per-alt checks still apply).
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Index validity
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidIndex:
    def test_out_of_range_index_flagged(self):
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        # Only indices 0 and 1 exist; [5] is out of range.
        result = verify_comparative_reasoning(
            "Aggressive alternative [5] creates a threat.",
            candidates, CHOSEN_SAFE,
        )
        invalid = [c for c in result if c.type == "invalid_index"]
        assert len(invalid) == 1
        assert invalid[0].indices == (5,)

    def test_negative_index_flagged(self):
        # The regex only matches digits (positive integers), so a literal
        # "[-1]" will not match the index reference pattern at all. We
        # confirm that the verifier does not crash on prose that contains
        # bracketed numbers outside the alternative-reference grammar.
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Some prose with a bracketed number [99] not in a reference.",
            candidates, CHOSEN_SAFE,
        )
        # [99] is not preceded by "alternative"/"move", so it's ignored.
        assert all(c.type != "invalid_index" for c in result)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Self-reference
# ═══════════════════════════════════════════════════════════════════════════


class TestSelfReference:
    def test_self_reference_flagged(self):
        # Index 0 IS the chosen move (by path equality).
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Aggressive alternative [0] creates a threat.",
            candidates, CHOSEN_SAFE,
        )
        self_refs = [c for c in result if c.type == "self_reference"]
        assert len(self_refs) == 1
        assert self_refs[0].indices == (0,)

    def test_no_self_reference_when_chosen_path_differs(self):
        # Index 0 is NOT the chosen move because paths differ.
        alt = _alt_aggressive([(5, 2), (4, 3)])
        candidates = [alt, CHOSEN_SAFE]
        result = verify_comparative_reasoning(
            "Aggressive alternative [0] creates a threat.",
            candidates, CHOSEN_SAFE,
        )
        assert all(c.type != "self_reference" for c in result)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-alt mismatch
# ═══════════════════════════════════════════════════════════════════════════


class TestPerAltMismatch:
    def test_threat_claim_against_quiet_alt_flagged(self):
        # Alt at index 1 is QUIET (no threat). Claiming it creates a threat
        # must flag a per_alt_mismatch.
        candidates = [CHOSEN_SAFE, _alt_quiet([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Aggressive alternative [1] creates an immediate threat.",
            candidates, CHOSEN_SAFE,
        )
        mismatches = [c for c in result if c.type == "per_alt_mismatch"]
        assert len(mismatches) == 1
        assert mismatches[0].indices == (1,)
        assert mismatches[0].fact_key == "creates_immediate_threat"
        assert mismatches[0].expected is True
        assert mismatches[0].actual is False

    def test_correct_threat_claim_no_contradiction(self):
        # Alt at index 1 IS aggressive. The claim holds.
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Aggressive alternative [1] creates an immediate threat.",
            candidates, CHOSEN_SAFE,
        )
        assert all(c.type != "per_alt_mismatch" for c in result)

    def test_capture_claim_against_non_capturing_alt_flagged(self):
        candidates = [CHOSEN_SAFE, _alt_quiet([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Material alternative [1] captures two pieces.",
            candidates, CHOSEN_SAFE,
        )
        mismatches = [c for c in result if c.type == "per_alt_mismatch"]
        assert any(
            m.fact_key == "captures_count" and m.expected == "POSITIVE"
            for m in mismatches
        )

    def test_avoids_recapture_claim_against_unsafe_alt_flagged(self):
        # Alt allows recapture; prose says it avoids.
        unsafe = _alt_aggressive([(5, 2), (4, 3)], opp_recap=True)
        candidates = [CHOSEN_SAFE, unsafe]
        result = verify_comparative_reasoning(
            "Defensive alternative [1] avoids recapture.",
            candidates, CHOSEN_SAFE,
        )
        mismatches = [c for c in result if c.type == "per_alt_mismatch"]
        assert any(
            m.fact_key == "opponent_can_recapture" and m.expected is False
            for m in mismatches
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Grouped-claim partial truth
# ═══════════════════════════════════════════════════════════════════════════


class TestGroupedClaimPartial:
    def test_partial_truth_flagged(self):
        # [1] creates threat, [2] does NOT. Group claim is partial.
        candidates = [
            CHOSEN_SAFE,
            _alt_aggressive([(5, 2), (4, 3)]),
            _alt_quiet([(5, 0), (4, 1)]),
        ]
        result = verify_comparative_reasoning(
            "Aggressive alternatives [1] and [2] create immediate threats.",
            candidates, CHOSEN_SAFE,
        )
        partials = [c for c in result if c.type == "grouped_claim_partial"]
        assert len(partials) == 1
        assert partials[0].indices == (1, 2)
        # The "actual" field carries the failing subset.
        assert 2 in partials[0].actual
        assert 1 not in partials[0].actual

    def test_fully_true_group_no_contradiction(self):
        candidates = [
            CHOSEN_SAFE,
            _alt_aggressive([(5, 2), (4, 3)]),
            _alt_aggressive([(5, 0), (4, 1)]),
        ]
        result = verify_comparative_reasoning(
            "Aggressive alternatives [1] and [2] create immediate threats.",
            candidates, CHOSEN_SAFE,
        )
        assert all(c.type != "grouped_claim_partial" for c in result)

    def test_three_member_group_oxford_comma_format(self):
        # Verifier must parse "alternatives [1], [2], and [3]" correctly.
        candidates = [
            CHOSEN_SAFE,
            _alt_aggressive([(5, 2), (4, 3)]),
            _alt_quiet([(5, 0), (4, 1)]),
            _alt_aggressive([(5, 6), (4, 5)]),
        ]
        result = verify_comparative_reasoning(
            "Aggressive alternatives [1], [2], and [3] create immediate threats.",
            candidates, CHOSEN_SAFE,
        )
        partials = [c for c in result if c.type == "grouped_claim_partial"]
        assert len(partials) == 1
        # All three were named, but only [2] is the violator.
        assert set(partials[0].indices) == {1, 2, 3}
        assert partials[0].actual == (2,)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Tradeoff consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeoff:
    def test_valid_recapture_safety_tradeoff_no_contradiction(self):
        # Chosen is SAFE → claiming a safety tradeoff is consistent.
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Chosen move forfeits aggressive options in favour of "
            "recapture safety.",
            candidates, CHOSEN_SAFE,
        )
        assert all(c.type != "invalid_tradeoff" for c in result)

    def test_invalid_recapture_safety_tradeoff_flagged(self):
        # Chosen has recapture risk; claiming safety tradeoff is wrong.
        chosen_unsafe = _move(
            [(5, 4), (4, 3)],
            opponent_can_recapture=True,
            creates_immediate_threat=False,
            shot_sequence_available=False,
            captures_count=1,
            net_gain=1,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            our_pieces_threatened_after=1,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
        )
        candidates = [chosen_unsafe, _alt_aggressive([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Chosen move forfeits aggressive options in favour of "
            "recapture safety.",
            candidates, chosen_unsafe,
        )
        tradeoff_errs = [c for c in result if c.type == "invalid_tradeoff"]
        assert any(
            t.fact_key == "opponent_can_recapture"
            for t in tradeoff_errs
        )

    def test_invalid_forfeits_material_tradeoff_flagged(self):
        # Chosen captures something; claiming "forfeits material" is wrong.
        chosen_captures = _move(
            [(5, 4), (4, 3)],
            opponent_can_recapture=False,
            creates_immediate_threat=False,
            shot_sequence_available=False,
            captures_count=2,
            net_gain=2,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            our_pieces_threatened_after=0,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
        )
        candidates = [chosen_captures, _alt_quiet([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Chosen move forfeits material captures.",
            candidates, chosen_captures,
        )
        tradeoff_errs = [c for c in result if c.type == "invalid_tradeoff"]
        assert any(
            t.fact_key == "captures_count" for t in tradeoff_errs
        )

    def test_accepts_exposure_tradeoff_validated(self):
        # Chosen captures + has recapture risk → "accepts exposure" is valid.
        chosen_capturing_unsafe = _move(
            [(5, 4), (4, 3)],
            opponent_can_recapture=True,
            captures_count=1,
            net_gain=1,
            creates_immediate_threat=False,
            shot_sequence_available=False,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            our_pieces_threatened_after=1,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
        )
        candidates = [chosen_capturing_unsafe, _alt_quiet([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "Chosen move accepts exposure to win material, where defensive "
            "options would have stayed safe.",
            candidates, chosen_capturing_unsafe,
        )
        assert all(c.type != "invalid_tradeoff" for c in result)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Valid prose
# ═══════════════════════════════════════════════════════════════════════════


class TestValidProse:
    def test_clean_comparative_paragraph_no_contradictions(self):
        candidates = [
            CHOSEN_SAFE,
            _alt_aggressive([(5, 2), (4, 3)]),
            _alt_quiet([(5, 0), (4, 1)]),
        ]
        result = verify_comparative_reasoning(
            "Aggressive alternative [1] creates an immediate threat but "
            "allows recapture. Defensive alternative [2] avoids recapture. "
            "Chosen move forfeits aggressive options in favour of "
            "recapture safety.",
            candidates, CHOSEN_SAFE,
        )
        assert result == [], f"expected clean paragraph, got: {result}"

    def test_paragraph_with_no_alternative_references_clean(self):
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        result = verify_comparative_reasoning(
            "The chosen move proceeds quietly.",
            candidates, CHOSEN_SAFE,
        )
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# 8. Chosen-move verifier isolation invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestChosenMoveVerifierIsolation:
    """The comparative verifier must NEVER import or call the chosen-move
    verifier path. Locked roadmap invariant I2."""

    def test_module_source_does_not_reference_chosen_move_verifier(self):
        from checkers.agents import comparative_reasoner
        src = inspect.getsource(comparative_reasoner)
        # Forbidden references (these would couple us to chosen-move scope)
        for forbidden in (
            "from checkers.agents.ranker_agent",
            "import checkers.agents.ranker_agent",
            "from checkers.evaluation.unified_verifier",
            "import checkers.evaluation.unified_verifier",
            "from checkers.evaluation.claim_extractor",
            "import checkers.evaluation.claim_extractor",
            "from checkers.evaluation.claim_verifier",
            "import checkers.evaluation.claim_verifier",
            "_check_reasoning_truthfulness",
            "contradiction_strings(",
            "verify_all(",
            "extract_claims(",
            "verify_claims(",
        ):
            assert forbidden not in src, (
                f"isolation invariant violated: comparative module "
                f"references '{forbidden}'"
            )

    def test_verify_function_source_isolation(self):
        # The function body itself must not reference chosen-move verifiers.
        src = inspect.getsource(verify_comparative_reasoning)
        for forbidden in (
            "_check_reasoning_truthfulness",
            "verify_all",
            "contradiction_strings",
            "extract_claims",
            "ranker_agent",
            "unified_verifier",
        ):
            assert forbidden not in src

    def test_does_not_check_forbidden_vocabulary(self):
        # Even if the comparative paragraph contains a chosen-move-forbidden
        # phrase, the comparative verifier must NOT flag it (that's the
        # chosen-move verifier's job, on chosen-move prose).
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        prose = (
            "Aggressive alternative [1] creates a threat. "
            "Some content with the phrase 'activity score' appears here."
        )
        result = verify_comparative_reasoning(prose, candidates, CHOSEN_SAFE)
        # The 'activity score' forbidden phrase is not part of the
        # comparative scope; no contradiction for it.
        for c in result:
            assert "forbidden" not in c.type.lower()
            assert "vocab" not in c.type.lower()

    def test_does_not_check_schema_leak(self):
        # Raw "field=value" syntax in the prose is a chosen-move-verifier
        # concern. The comparative verifier ignores it.
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        prose = (
            "Aggressive alternative [1] creates a threat with "
            "opponent_can_recapture=true."
        )
        result = verify_comparative_reasoning(prose, candidates, CHOSEN_SAFE)
        for c in result:
            assert "schema" not in c.type.lower()
            assert "leak" not in c.type.lower()

    def test_does_not_check_chosen_move_numeric_fabrication(self):
        # A number that isn't in any seed but appears in chosen-move-style
        # numeric claims is NOT the comparative verifier's concern.
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        prose = (
            "Aggressive alternative [1] creates a threat. Chosen-move "
            "opponent mobility drops from 99 to 42."
        )
        result = verify_comparative_reasoning(prose, candidates, CHOSEN_SAFE)
        for c in result:
            assert "numeric" not in c.type.lower()
            assert "fabrication" not in c.type.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 9. Sanitizer
# ═══════════════════════════════════════════════════════════════════════════


class TestSanitizer:
    def test_invalid_index_sanitized(self):
        rec = ComparativeContradiction(type="invalid_index", indices=(5,))
        s = sanitize_comparative_contradiction(rec)
        assert "alternative [5]" in s
        assert "does not exist" in s

    def test_self_reference_sanitized(self):
        rec = ComparativeContradiction(type="self_reference", indices=(0,))
        s = sanitize_comparative_contradiction(rec)
        assert "alternative [0]" in s
        assert "chosen move" in s.lower()

    def test_per_alt_mismatch_sanitized_with_category(self):
        rec = ComparativeContradiction(
            type="per_alt_mismatch",
            indices=(3,),
            fact_key="creates_immediate_threat",
            expected=True,
            actual=False,
        )
        s = sanitize_comparative_contradiction(rec)
        assert "alternative [3]" in s
        assert "threat" in s.lower()  # category label

    def test_per_alt_mismatch_sanitizer_does_not_echo_raw_field_name(self):
        rec = ComparativeContradiction(
            type="per_alt_mismatch",
            indices=(3,),
            fact_key="opponent_can_recapture",
        )
        s = sanitize_comparative_contradiction(rec)
        # Field name with underscores must NOT appear verbatim.
        assert "opponent_can_recapture" not in s
        # Semantic category instead.
        assert "safety" in s.lower()

    def test_grouped_claim_sanitized(self):
        rec = ComparativeContradiction(
            type="grouped_claim_partial",
            indices=(1, 2, 3),
        )
        s = sanitize_comparative_contradiction(rec)
        assert "group" in s.lower()

    def test_invalid_tradeoff_sanitized(self):
        rec = ComparativeContradiction(
            type="invalid_tradeoff",
            fact_key="opponent_can_recapture",
        )
        s = sanitize_comparative_contradiction(rec)
        assert "tradeoff" in s.lower()
        assert "safety" in s.lower()  # category, not raw key

    def test_sanitizer_never_echoes_raw_numeric_values(self):
        rec = ComparativeContradiction(
            type="per_alt_mismatch",
            indices=(3,),
            fact_key="captures_count",
            expected="POSITIVE",
            actual=5,
        )
        s = sanitize_comparative_contradiction(rec)
        # Raw fact value 5 must not appear; only category "material".
        assert "5" not in s
        assert "material" in s.lower()

    def test_unknown_type_falls_back_to_generic_string(self):
        rec = ComparativeContradiction(type="some_unknown_type")
        s = sanitize_comparative_contradiction(rec)
        assert isinstance(s, str)
        assert len(s) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 10. Determinism (purity check)
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_inputs_same_results(self):
        candidates = [CHOSEN_SAFE, _alt_aggressive([(5, 2), (4, 3)])]
        prose = "Aggressive alternative [1] creates an immediate threat."
        r1 = verify_comparative_reasoning(prose, candidates, CHOSEN_SAFE)
        r2 = verify_comparative_reasoning(prose, candidates, CHOSEN_SAFE)
        assert r1 == r2

    def test_never_mutates_inputs(self):
        import copy
        candidates = [
            copy.deepcopy(CHOSEN_SAFE),
            _alt_aggressive([(5, 2), (4, 3)]),
        ]
        before = copy.deepcopy(candidates)
        before_chosen = copy.deepcopy(CHOSEN_SAFE)
        verify_comparative_reasoning(
            "Aggressive alternative [1] creates a threat.",
            candidates, CHOSEN_SAFE,
        )
        assert candidates == before
        assert CHOSEN_SAFE == before_chosen
