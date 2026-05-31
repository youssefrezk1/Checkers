# checkers/tests/test_comparative_chosen_move_claims.py
#
# Phase 1 fix — unit tests for chosen-move factual claim verification
# (Pass 3 of verify_comparative_reasoning).
#
# Coverage:
#   - false capture claim (BUG-1 from audit)
#   - implied capture via contrastive framing (BUG-2 from audit)
#   - false material-gain claim
#   - false immediate-threat claim
#   - false recapture-safety claim (chosen move labelled safe when it isn't)
#   - no false positives on correct prose
#   - correct prose about the chosen move passes clean
#   - sanitizer produces the expected hint string
#   - chosen_move_factual enters the refinement pipeline (monotonic gate)

from __future__ import annotations

from checkers.agents.comparative_reasoner import (
    ComparativeContradiction,
    RefinementCandidate,
    _evaluate_refinement_candidate,
    sanitize_comparative_contradiction,
    verify_comparative_reasoning,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _move(path, **facts):
    return {"path": path, "facts": dict(facts)}


# Chosen move: simple forward move, no capture, no threat, safe from recapture.
CHOSEN_QUIET = _move(
    [(5, 2), (4, 1)],
    captures_count=0,
    net_gain=0,
    results_in_king=False,
    creates_immediate_threat=False,
    opponent_can_recapture=False,
    leaves_piece_isolated=False,
    weakens_king_row=False,
    near_promotion=False,
    our_pieces_threatened_after=0,
    opponent_mobility_before=11,
    opponent_mobility_after=11,
)

# Chosen move: jump, captures 1, net_gain=1, opponent CAN recapture.
CHOSEN_CAPTURE = _move(
    [(4, 5), (2, 3)],
    captures_count=1,
    net_gain=1,
    results_in_king=False,
    creates_immediate_threat=False,
    opponent_can_recapture=True,
    leaves_piece_isolated=True,
    weakens_king_row=False,
    near_promotion=False,
    our_pieces_threatened_after=1,
    opponent_mobility_before=8,
    opponent_mobility_after=7,
)

# A generic quiet alternative (not the chosen move).
ALT_QUIET = _move(
    [(5, 0), (4, 1)],
    captures_count=0,
    net_gain=0,
    results_in_king=False,
    creates_immediate_threat=False,
    opponent_can_recapture=False,
    leaves_piece_isolated=False,
    weakens_king_row=False,
    near_promotion=False,
    our_pieces_threatened_after=0,
    opponent_mobility_before=11,
    opponent_mobility_after=11,
)

CANDIDATES_QUIET = [CHOSEN_QUIET, ALT_QUIET]
CANDIDATES_CAPTURE = [CHOSEN_CAPTURE, ALT_QUIET]


# ═════════════════════════════════════════════════════════════════════════════
# 1. BUG-1: explicit false capture claim about chosen move
# ═════════════════════════════════════════════════════════════════════════════

class TestExplicitFalseCaptureClaimBug1:
    def test_secures_a_capture_flagged_when_no_capture(self):
        text = (
            "While defensive options avoid recapture, "
            "the chosen move secures a capture."
        )
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result if c.type == "chosen_move_factual"]
        assert len(factual) >= 1
        assert any(c.fact_key == "captures_count" for c in factual)

    def test_chosen_move_secures_capture_has_positive_expected(self):
        text = "The chosen move secures a capture, gaining material."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "captures_count"]
        assert factual
        assert factual[0].expected == "POSITIVE"
        assert factual[0].actual == 0

    def test_no_false_positive_when_chosen_move_actually_captures(self):
        # chosen move has captures_count=1 — "secures a capture" is correct.
        text = "The chosen move secures a capture, winning material."
        result = verify_comparative_reasoning(text, CANDIDATES_CAPTURE, CHOSEN_CAPTURE)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "captures_count"]
        assert factual == []

    def test_clause_recorded_in_contradiction(self):
        text = "The chosen move secures a capture and advances."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result if c.type == "chosen_move_factual"]
        assert factual
        assert "chosen move" in factual[0].clause.lower()


# ═════════════════════════════════════════════════════════════════════════════
# 2. BUG-2: implied capture via contrastive framing
# ═════════════════════════════════════════════════════════════════════════════

class TestImpliedCaptureContrastiveFramingBug2:
    def test_sacrificed_chance_to_capture_flagged_when_no_capture(self):
        text = (
            "The defensive alternatives avoided risk but "
            "sacrificed the chance to capture."
        )
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result if c.type == "chosen_move_factual"]
        assert len(factual) >= 1
        assert any(c.fact_key == "captures_count" for c in factual)

    def test_implied_clause_uses_implied_label(self):
        text = "Alternatives sacrificed the chance to capture while the chosen move advanced."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "captures_count"]
        assert factual
        assert "implied" in factual[0].clause.lower()

    def test_no_false_positive_when_chosen_move_captures(self):
        # Framing is correct: the alternatives sacrificed the chance to capture
        # AND the chosen move actually did capture.
        text = "Alternatives sacrificed the chance to capture; the chosen move secures one."
        result = verify_comparative_reasoning(text, CANDIDATES_CAPTURE, CHOSEN_CAPTURE)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "captures_count"]
        assert factual == []


# ═════════════════════════════════════════════════════════════════════════════
# 3. False material-gain claim
# ═════════════════════════════════════════════════════════════════════════════

class TestFalseMaterialGainClaim:
    def test_gains_material_flagged_when_net_gain_zero(self):
        text = "The chosen move gains material and improves the position."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "net_gain"]
        assert len(factual) >= 1

    def test_wins_material_flagged_when_net_gain_zero(self):
        text = "The chosen move wins material over the alternatives."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "net_gain"]
        assert len(factual) >= 1

    def test_net_gain_phrase_flagged_when_zero(self):
        text = "The chosen move achieves a net gain and secures progress."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "net_gain"]
        assert len(factual) >= 1

    def test_no_false_positive_when_net_gain_positive(self):
        text = "The chosen move wins material over the alternatives."
        result = verify_comparative_reasoning(text, CANDIDATES_CAPTURE, CHOSEN_CAPTURE)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "net_gain"]
        assert factual == []


# ═════════════════════════════════════════════════════════════════════════════
# 4. False immediate-threat claim
# ═════════════════════════════════════════════════════════════════════════════

class TestFalseImmediateThreatClaim:
    def test_creates_threat_flagged_when_no_threat(self):
        text = "The chosen move creates an immediate threat for the opponent."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "creates_immediate_threat"]
        assert len(factual) >= 1

    def test_opens_immediate_threats_flagged(self):
        text = "The chosen move opens immediate threats while advancing."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "creates_immediate_threat"]
        assert len(factual) >= 1

    def test_no_false_positive_when_threat_is_true(self):
        chosen_threat = _move(
            [(5, 2), (4, 1)],
            captures_count=0, net_gain=0,
            results_in_king=False,
            creates_immediate_threat=True,
            opponent_can_recapture=False,
            leaves_piece_isolated=False,
            weakens_king_row=False, near_promotion=False,
            our_pieces_threatened_after=0,
            opponent_mobility_before=11, opponent_mobility_after=11,
        )
        text = "The chosen move creates an immediate threat."
        result = verify_comparative_reasoning(
            text, [chosen_threat, ALT_QUIET], chosen_threat,
        )
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "creates_immediate_threat"]
        assert factual == []


# ═════════════════════════════════════════════════════════════════════════════
# 5. False recapture-safety claim
# ═════════════════════════════════════════════════════════════════════════════

class TestFalseRecaptureSafetyClaim:
    def test_avoids_recapture_flagged_when_opponent_can_recapture(self):
        # chosen move has opponent_can_recapture=True but prose claims it's safe.
        text = "The chosen move avoids recapture and maintains safety."
        result = verify_comparative_reasoning(
            text, CANDIDATES_CAPTURE, CHOSEN_CAPTURE,
        )
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "opponent_can_recapture"]
        assert len(factual) >= 1
        assert factual[0].expected is False
        assert factual[0].actual is True

    def test_recapture_safety_phrase_flagged(self):
        text = "The chosen move prioritises recapture safety over initiative."
        result = verify_comparative_reasoning(
            text, CANDIDATES_CAPTURE, CHOSEN_CAPTURE,
        )
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "opponent_can_recapture"]
        assert len(factual) >= 1

    def test_no_false_positive_when_actually_safe(self):
        text = "The chosen move avoids recapture, staying safe."
        result = verify_comparative_reasoning(
            text, CANDIDATES_QUIET, CHOSEN_QUIET,
        )
        factual = [c for c in result
                   if c.type == "chosen_move_factual"
                   and c.fact_key == "opponent_can_recapture"]
        assert factual == []


# ═════════════════════════════════════════════════════════════════════════════
# 6. Clean prose passes — no false positives
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanProseNoFalsePositives:
    def test_correctly_describes_no_capture(self):
        text = (
            "The defensive alternatives [1] avoid recapture but do not capture. "
            "The chosen move advances without capturing, improving mobility."
        )
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result if c.type == "chosen_move_factual"]
        assert factual == []

    def test_correctly_describes_capture_chosen_move(self):
        text = (
            "Defensive alternatives [1] avoid exposure. "
            "The chosen move secures a capture, though it is left exposed."
        )
        result = verify_comparative_reasoning(
            text, CANDIDATES_CAPTURE, CHOSEN_CAPTURE,
        )
        factual = [c for c in result if c.type == "chosen_move_factual"]
        assert factual == []

    def test_no_chosen_move_reference_no_pass3_fires(self):
        # Sentence about alternatives only — Pass 3 must not fire.
        text = "Defensive alternatives [1] avoid recapture but do not advance."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        factual = [c for c in result if c.type == "chosen_move_factual"]
        assert factual == []

    def test_empty_text_returns_empty(self):
        result = verify_comparative_reasoning("", CANDIDATES_QUIET, CHOSEN_QUIET)
        assert result == []

    def test_none_chosen_move_no_pass3_crash(self):
        text = "The chosen move secures a capture and wins material."
        result = verify_comparative_reasoning(text, CANDIDATES_QUIET, None)
        # Pass 3 must not run when chosen_move is None; no crash allowed.
        assert isinstance(result, list)
        factual = [c for c in result if c.type == "chosen_move_factual"]
        assert factual == []


# ═════════════════════════════════════════════════════════════════════════════
# 7. Sanitizer output for chosen_move_factual
# ═════════════════════════════════════════════════════════════════════════════

class TestSanitizerChosenMoveFactual:
    def test_material_fact_key_gives_material_category(self):
        c = ComparativeContradiction(
            type="chosen_move_factual",
            fact_key="captures_count",
            expected="POSITIVE",
            actual=0,
        )
        hint = sanitize_comparative_contradiction(c)
        assert "material" in hint
        assert "chosen move" in hint

    def test_net_gain_fact_key_gives_material_category(self):
        c = ComparativeContradiction(
            type="chosen_move_factual",
            fact_key="net_gain",
            expected="POSITIVE",
            actual=0,
        )
        hint = sanitize_comparative_contradiction(c)
        assert "material" in hint
        assert "chosen move" in hint

    def test_safety_fact_key_gives_safety_category(self):
        c = ComparativeContradiction(
            type="chosen_move_factual",
            fact_key="opponent_can_recapture",
            expected=False,
            actual=True,
        )
        hint = sanitize_comparative_contradiction(c)
        assert "safety" in hint
        assert "chosen move" in hint

    def test_unknown_fact_key_falls_back_to_factual(self):
        c = ComparativeContradiction(
            type="chosen_move_factual",
            fact_key="unknown_key",
            expected=True,
            actual=False,
        )
        hint = sanitize_comparative_contradiction(c)
        assert "factual" in hint
        assert "chosen move" in hint


# ═════════════════════════════════════════════════════════════════════════════
# 8. chosen_move_factual enters the existing refinement pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestChosenMoveFactualEntersRefinementPipeline:
    def test_factual_contradiction_blocks_short_circuit(self):
        # If verify_comparative_reasoning returns >=1 contradiction, the
        # reject-sample loop in generate_comparative_reasoning does NOT
        # short-circuit (n_contras > 0).
        text = "The chosen move secures a capture."
        contras = verify_comparative_reasoning(text, CANDIDATES_QUIET, CHOSEN_QUIET)
        assert len(contras) > 0, "Expected at least one contradiction"

    def test_monotonic_gate_rejects_candidate_that_does_not_improve(self):
        # A candidate with the same or higher contradiction count is rejected.
        candidate = RefinementCandidate(raw="...", text="some text", n_contradictions=2)
        assert not _evaluate_refinement_candidate(candidate, baseline_count=1)
        assert not _evaluate_refinement_candidate(candidate, baseline_count=2)

    def test_monotonic_gate_accepts_candidate_that_strictly_improves(self):
        candidate = RefinementCandidate(raw="...", text="some text", n_contradictions=0)
        assert _evaluate_refinement_candidate(candidate, baseline_count=1)

    def test_sanitizer_produces_non_empty_hint_for_pipeline(self):
        # sanitize_comparative_contradiction must return a non-empty string so
        # build_comparative_refinement_user_prompt receives meaningful hints.
        c = ComparativeContradiction(
            type="chosen_move_factual",
            fact_key="captures_count",
            expected="POSITIVE",
            actual=0,
        )
        hint = sanitize_comparative_contradiction(c)
        assert isinstance(hint, str) and len(hint) > 0
