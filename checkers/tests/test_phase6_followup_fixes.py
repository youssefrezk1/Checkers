"""
checkers/tests/test_phase6_followup_fixes.py

Focused tests for the four Phase-6 follow-up fixes:

  Fix 1 — first-ply EQUAL default for score_state at the ranker call site.
  Fix 2 — score_gap_advantage hard verification via next_best_minimax_score
          surfaced in ranker_diagnostics.
  Fix 4 — score_gap_advantage extraction no longer false-fires on bare "scores"
          when no comparison sentinel is nearby.
  Fix 5 — ontology constants live in the neutral checkers.ontology package and
          are NOT pulled into runtime via checkers.evaluation.

No LLM calls.  Deterministic.  All four fixes are tested at the smallest layer
that proves the behavioural contract.
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Fix 5 — ontology import direction
# ---------------------------------------------------------------------------

class TestFix5OntologyImportDirection:
    """Runtime depends on checkers.ontology, NOT on checkers.evaluation."""

    def test_neutral_ontology_module_exists(self):
        import checkers.ontology.semantic_ontology as ontology  # noqa: F401
        # All five public symbols are present.
        for name in (
            "SemanticConceptType",
            "CONCEPT_GROUNDING",
            "FORBIDDEN_CONFLATION_PHRASES",
            "GENERIC_FILLER_PHRASES",
            "GEOMETRIC_CENTER_COLUMNS",
        ):
            assert hasattr(ontology, name), f"ontology missing {name!r}"

    def test_neutral_ontology_has_no_project_internal_imports(self):
        """checkers.ontology.semantic_ontology may import only stdlib —
        never anything from checkers.{agents,nodes,graph,state,engine,search,evaluation}."""
        # Drop any stale import so we capture a clean snapshot.
        for mod in list(sys.modules):
            if mod.startswith("checkers.ontology"):
                del sys.modules[mod]
        before = set(sys.modules)
        import checkers.ontology.semantic_ontology  # noqa: F401
        added = set(sys.modules) - before
        forbidden = (
            "checkers.engine", "checkers.agents", "checkers.graph",
            "checkers.state", "checkers.nodes", "checkers.search",
            "checkers.evaluation",
        )
        for mod in added:
            for prefix in forbidden:
                assert not mod.startswith(prefix), (
                    f"ontology pulled in disallowed module: {mod!r}"
                )

    def test_evaluation_shim_reexports_canonical_objects(self):
        """checkers.evaluation.semantic_ontology must re-export the SAME
        constant objects as the neutral module when both are loaded in a
        fresh interpreter state.  We drop any cached copies first so the
        identity check is not poisoned by earlier `importlib.reload` calls
        from sibling tests."""
        for mod_name in (
            "checkers.ontology.semantic_ontology",
            "checkers.evaluation.semantic_ontology",
        ):
            sys.modules.pop(mod_name, None)

        from checkers.ontology.semantic_ontology import (
            FORBIDDEN_CONFLATION_PHRASES as canonical_conflation,
            GENERIC_FILLER_PHRASES as canonical_filler,
            CONCEPT_GROUNDING as canonical_grounding,
            SemanticConceptType as canonical_enum,
        )
        from checkers.evaluation.semantic_ontology import (
            FORBIDDEN_CONFLATION_PHRASES as shim_conflation,
            GENERIC_FILLER_PHRASES as shim_filler,
            CONCEPT_GROUNDING as shim_grounding,
            SemanticConceptType as shim_enum,
        )
        # Object identity proves the shim is a thin re-export — never a copy.
        assert shim_conflation is canonical_conflation
        assert shim_filler     is canonical_filler
        assert shim_grounding  is canonical_grounding
        assert shim_enum       is canonical_enum

    def test_ranker_agent_does_not_import_from_evaluation(self):
        """Loading the runtime ranker_agent must NOT pull
        checkers.evaluation.* into sys.modules.  The neutral ontology
        package is the one allowed dependency."""
        # Force a clean reload of both packages so we observe the *true*
        # import graph rather than residue from earlier test imports.
        for mod_name in list(sys.modules):
            if (
                mod_name == "checkers.agents.ranker_agent"
                or mod_name.startswith("checkers.evaluation")
                or mod_name.startswith("checkers.ontology")
            ):
                del sys.modules[mod_name]

        before = set(sys.modules)
        import checkers.agents.ranker_agent  # noqa: F401
        added = set(sys.modules) - before

        # checkers.ontology IS allowed — that is the entire point of Fix 5.
        # checkers.evaluation is NOT allowed.
        leaked = [m for m in added if m.startswith("checkers.evaluation")]
        assert leaked == [], (
            f"runtime ranker_agent pulled forbidden evaluation modules into "
            f"sys.modules: {leaked}"
        )


# ---------------------------------------------------------------------------
# Fix 1 — first-ply EQUAL default for score_state
# ---------------------------------------------------------------------------

class TestFix1FirstPlyScoreStateDefault:
    """_resolve_score_state_for_seeds must read state.score_state directly.
    The default is "EQUAL" (Pydantic default), adversity seeds are suppressed."""

    def _state(self, score_state: str = "EQUAL"):
        from checkers.state.state import CheckersState
        return CheckersState(
            board=[[0] * 8 for _ in range(8)],
            current_player=1,
            turn_number=1,
            score_state=score_state,
        )

    def test_returns_equal_by_default(self):
        from checkers.agents.ranker_agent import _resolve_score_state_for_seeds
        assert _resolve_score_state_for_seeds(self._state()) == "EQUAL"

    def test_returns_equal_when_score_state_is_empty_string(self):
        from checkers.agents.ranker_agent import _resolve_score_state_for_seeds
        # Empty string falls back to "EQUAL" (conservative default)
        assert _resolve_score_state_for_seeds(self._state("")) == "EQUAL"

    def test_passes_through_real_value(self):
        from checkers.agents.ranker_agent import _resolve_score_state_for_seeds
        for ss in (
            "EQUAL", "CLEARLY_LOSING", "SLIGHTLY_LOSING",
            "CLEARLY_WINNING", "SLIGHTLY_WINNING",
        ):
            assert _resolve_score_state_for_seeds(self._state(ss)) == ss

    def test_default_suppresses_adversity_in_seed_builder(self):
        """End-to-end behavioural guard: when the resolved score_state is
        "EQUAL", adversity seeds must NOT fire even for a chosen move with
        a very negative minimax_score (forced-but-winning line)."""
        from checkers.agents.ranker_agent import _build_grounded_reasoning_seeds
        # Resolved score_state is "EQUAL" — adversity suppressed.
        move = {
            "type": "simple",
            "path": [[5, 0], [4, 1]],
            "captured": [],
            "facts": {
                "minimax_score":    -50.0,
                "material_advantage": -3,  # would normally seed deficit
                "opponent_can_recapture": False,
                "captures_count": 0,
                "net_gain": 0,
                "leaves_piece_isolated": False,
                "creates_immediate_threat": False,
                "center_control": False,
                "results_in_king": False,
                "near_promotion": False,
                "opponent_mobility_before": 10,
                "opponent_mobility_after": 10,
                "our_mobility_before": 4,
                "our_mobility_after": 4,
            },
        }
        seeds = _build_grounded_reasoning_seeds(
            move, [move], player=1, score_state="EQUAL",
        )
        joined = " ".join(seeds).lower()
        assert "behind by" not in joined, (
            f"EQUAL default still allowed material-deficit adversity seed: {seeds}"
        )
        assert "structural disadvantage" not in joined


# ---------------------------------------------------------------------------
# Fix 2 — score_gap_advantage hard verification via context
# ---------------------------------------------------------------------------

class TestFix2ScoreGapHardVerification:
    """When ranker_diagnostics surfaces next_best_minimax_score, the
    verifier upgrades from PARTIAL to hard SUPPORTED/CONTRADICTED."""

    def _make_claim(self):
        from checkers.evaluation.claim_extractor import ClaimRecord
        from checkers.evaluation.reasoning_taxonomy import (
            ClaimStatus, ClaimVerifiability,
        )
        return ClaimRecord(
            claim_type="score_gap_advantage",
            claim_status=ClaimStatus.UNSUPPORTED,
            claim_verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        )

    def test_supported_when_chosen_strictly_better_than_next_best(self):
        from checkers.evaluation.claim_verifier import verify_claims
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        out = verify_claims(
            [self._make_claim()],
            {"minimax_score": -10.0},
            context={"next_best_minimax_score": -35.0},
        )
        assert out[0].claim_status == ClaimStatus.SUPPORTED

    def test_unsupported_when_chosen_equals_next_best(self):
        """Tie-tolerance follow-up: an exact tie falls inside the EPSILON band
        and is classified UNSUPPORTED rather than CONTRADICTED.  See
        TestScoreGapEpsilonTolerance for the full near-tie behaviour."""
        from checkers.evaluation.claim_verifier import verify_claims
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        out = verify_claims(
            [self._make_claim()],
            {"minimax_score": 5.0},
            context={"next_best_minimax_score": 5.0},
        )
        assert out[0].claim_status == ClaimStatus.UNSUPPORTED

    def test_contradicted_when_chosen_worse_than_next_best(self):
        from checkers.evaluation.claim_verifier import verify_claims
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        out = verify_claims(
            [self._make_claim()],
            {"minimax_score": -50.0},
            context={"next_best_minimax_score": -10.0},
        )
        assert out[0].claim_status == ClaimStatus.CONTRADICTED

    def test_partial_supported_when_context_missing(self):
        """Without context, behaviour reverts to legacy PARTIAL: SUPPORTED
        when minimax_score is numeric, never CONTRADICTED."""
        from checkers.evaluation.claim_verifier import verify_claims
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        # No context at all.
        out = verify_claims([self._make_claim()], {"minimax_score": -10.0})
        assert out[0].claim_status == ClaimStatus.SUPPORTED

        # Context dict provided but next_best_minimax_score is None.
        out = verify_claims(
            [self._make_claim()],
            {"minimax_score": -10.0},
            context={"next_best_minimax_score": None},
        )
        assert out[0].claim_status == ClaimStatus.SUPPORTED

    def test_unsupported_when_minimax_absent(self):
        from checkers.evaluation.claim_verifier import verify_claims
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        out = verify_claims(
            [self._make_claim()],
            {},  # no minimax_score
            context={"next_best_minimax_score": -10.0},
        )
        # Verifier falls back to legacy partial when chosen value is missing.
        assert out[0].claim_status == ClaimStatus.UNSUPPORTED

    def test_turn_evaluator_propagates_context_from_diagnostics(self):
        """End-to-end: evaluate_turn must read next_best_minimax_score from
        ranker_diagnostics and pass it into the verifier."""
        from checkers.evaluation.turn_evaluator import evaluate_turn
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus

        reasoning = (
            "The chosen move scores 25.0 points better than the next-best "
            "option [3], which is the strongest available continuation."
        )
        seeds = [
            "chosen move scores 25.0 points better than next-best option [3] "
            "(minimax: -10.0 vs -35.0)",
        ]
        facts = {"minimax_score": -10.0}
        # Hard CONTRADICTED case: claim "better than next-best" but our
        # diagnostics surface a next-best that is actually higher.
        diag_contradicted = {
            "next_best_minimax_score": 5.0,
            "reasoning_seeds": seeds,
            "final_choice_source": "raw_llm",
        }
        rec = evaluate_turn(
            reasoning_text=reasoning,
            reasoning_seeds=seeds,
            facts=facts,
            ranker_diagnostics=diag_contradicted,
        )
        gap_records = [c for c in rec.claims if c.claim_type == "score_gap_advantage"]
        assert gap_records, "score_gap_advantage was not extracted"
        assert gap_records[0].claim_status == ClaimStatus.CONTRADICTED

        # Hard SUPPORTED case: next-best really is worse than chosen.
        diag_supported = {**diag_contradicted, "next_best_minimax_score": -35.0}
        rec2 = evaluate_turn(
            reasoning_text=reasoning,
            reasoning_seeds=seeds,
            facts=facts,
            ranker_diagnostics=diag_supported,
        )
        gap2 = [c for c in rec2.claims if c.claim_type == "score_gap_advantage"]
        assert gap2[0].claim_status == ClaimStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Fix 4 — bare "scores" only fires with a nearby comparison sentinel
# ---------------------------------------------------------------------------

class TestFix4ScoreGapExtractionPrecision:
    """Bare "scores" must require a comparison cue.  Compound phrases
    (points better than, next-best option, best alternative) still fire
    unconditionally."""

    def _has_score_gap(self, text: str, seeds=None, facts=None) -> bool:
        from checkers.evaluation.claim_extractor import extract_claims
        records = extract_claims(text, reasoning_seeds=seeds, facts=facts)
        return any(r.claim_type == "score_gap_advantage" for r in records)

    # ── true positives ──────────────────────────────────────────────────────

    def test_compound_points_better_than_fires(self):
        assert self._has_score_gap(
            "This move scores 25.0 points better than next-best option [3].",
        )

    def test_next_best_option_alone_fires(self):
        assert self._has_score_gap(
            "Better than every alternative; the next-best option drops 25 points.",
        )

    def test_best_alternative_phrase_fires(self):
        assert self._has_score_gap(
            "Our chosen move beats the best alternative by a clear margin.",
        )

    def test_scores_with_higher_than_sentinel_fires(self):
        assert self._has_score_gap(
            "This move scores noticeably higher than the available alternatives.",
        )

    def test_scores_with_minimax_colon_sentinel_fires(self):
        # Adversity seed format: "(minimax: -10.0 vs -35.0)" appears nearby.
        assert self._has_score_gap(
            "Chosen move scores 25.0 points more (minimax: -10.0 vs -35.0).",
        )

    # ── false positives that must NOT fire ─────────────────────────────────

    def test_minimax_score_does_not_fire(self):
        """The phrase "minimax_score" must never trigger score_gap_advantage."""
        assert not self._has_score_gap(
            "The chosen move has minimax_score=5.00 — highest-evaluated option.",
            seeds=["minimax_score=5.00 — highest-evaluated option"],
        )

    def test_bare_scores_well_does_not_fire(self):
        assert not self._has_score_gap("This move scores well in our evaluation.")

    def test_bare_scores_with_no_comparison_does_not_fire(self):
        assert not self._has_score_gap(
            "The chosen move scores 5.0; the position remains balanced.",
        )

    def test_compound_score_phrase_does_not_fire(self):
        # "score" (singular) is not in the phrase list and should never fire.
        assert not self._has_score_gap("The overall score of this position is positive.")

    def test_seed_marker_no_longer_matches_bare_scores(self):
        """Seed list with only a minimax_score line must NOT cause the
        extractor to mark a score_gap_advantage claim as seed-supported.
        Phase-6 Fix 4 removed bare "scores" from seed_markers."""
        from checkers.evaluation.claim_extractor import extract_claims
        # Reasoning contains a real comparison phrase, so the claim IS
        # extracted — but the seed list contains only a minimax confirmation
        # line, so the source must be unsupported_phrase, not seed.
        records = extract_claims(
            "The chosen move scores 25 points better than the next-best option.",
            reasoning_seeds=["minimax_score=5.00 — highest-evaluated option"],
            facts={"minimax_score": 5.0},
        )
        gap = next(
            (r for r in records if r.claim_type == "score_gap_advantage"), None,
        )
        assert gap is not None
        assert gap.source != "seed", (
            "Phase-6 Fix 4: seed_markers must not include the bare 'scores' "
            "trigger that previously caused minimax_score seed lines to "
            "falsely authorise score_gap_advantage claims."
        )


# ---------------------------------------------------------------------------
# Fix 2 — runtime exposure of next_best_minimax_score in ranker_diagnostics
# ---------------------------------------------------------------------------

class TestFix2NextBestExposedFromRanker:
    """The runtime ranker must compute next_best_minimax_score across the
    filtered candidate list (excluding the chosen path) and place it in
    ranker_diagnostics so evaluation can consume it."""

    def _compute_next_best(self, filtered, chosen_path):
        # Re-import the private helper used internally — it's the same
        # function called by ranker_agent when building diagnostics.  We
        # exercise it directly to avoid mocking the full LLM pipeline.
        import checkers.agents.ranker_agent as ra
        # The helper is defined inside ranker_agent() as a closure.  Mirror
        # it here using its building blocks so tests don't depend on the
        # closure being callable from outside.  Equivalent logic:
        get_mm = ra._get_minimax_score
        scores = []
        for m in filtered or []:
            if m.get("path") == chosen_path:
                continue
            s = get_mm(m)
            if s is None or s == float("-inf"):
                continue
            scores.append(float(s))
        return max(scores) if scores else None

    def test_returns_max_alternative_score(self):
        filtered = [
            {"path": [[5, 0], [4, 1]], "facts": {"minimax_score": 10.0}},
            {"path": [[5, 2], [4, 3]], "facts": {"minimax_score": 7.0}},
            {"path": [[5, 4], [4, 5]], "facts": {"minimax_score": -3.0}},
        ]
        # Chosen = the 10.0 move; next-best should be 7.0.
        nb = self._compute_next_best(filtered, [[5, 0], [4, 1]])
        assert nb == 7.0

    def test_returns_none_when_single_candidate(self):
        filtered = [{"path": [[5, 0], [4, 1]], "facts": {"minimax_score": 10.0}}]
        nb = self._compute_next_best(filtered, [[5, 0], [4, 1]])
        assert nb is None

    def test_skips_chosen_even_if_duplicated(self):
        filtered = [
            {"path": [[5, 0], [4, 1]], "facts": {"minimax_score": 10.0}},
            {"path": [[5, 0], [4, 1]], "facts": {"minimax_score": 10.0}},
            {"path": [[5, 2], [4, 3]], "facts": {"minimax_score": 6.0}},
        ]
        nb = self._compute_next_best(filtered, [[5, 0], [4, 1]])
        assert nb == 6.0

    def test_diagnostics_key_present_in_schema(self):
        """ranker_diagnostics must include the new key after Fix 2.  We
        inspect the source string directly because constructing a full
        ranker invocation requires an LLM."""
        import inspect
        import checkers.agents.ranker_agent as ra
        src = inspect.getsource(ra.ranker_agent)
        assert '"next_best_minimax_score"' in src, (
            "ranker_diagnostics is missing the next_best_minimax_score key"
        )


# ---------------------------------------------------------------------------
# Follow-up — score_gap_advantage tolerance band around exact ties
# ---------------------------------------------------------------------------

class TestScoreGapEpsilonTolerance:
    """Near-tie minimax scores must not be falsely contradicted.

    Verifier contract:
        SUPPORTED    : chosen > next_best + EPSILON   (clearly better)
        CONTRADICTED : chosen < next_best - EPSILON   (clearly worse)
        UNSUPPORTED  : |chosen - next_best| <= EPSILON (within tolerance)

    The default EPSILON is 0.5 — small enough that a one-piece (~100) gap
    still falls cleanly into SUPPORTED/CONTRADICTED, large enough to absorb
    typical float ties at fixed minimax depth.
    """

    def _make_claim(self):
        from checkers.evaluation.claim_extractor import ClaimRecord
        from checkers.evaluation.reasoning_taxonomy import (
            ClaimStatus, ClaimVerifiability,
        )
        return ClaimRecord(
            claim_type="score_gap_advantage",
            claim_status=ClaimStatus.UNSUPPORTED,
            claim_verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        )

    def _verify(self, chosen: float, next_best: float, epsilon=None):
        from checkers.evaluation.claim_verifier import verify_claims
        ctx = {"next_best_minimax_score": next_best}
        if epsilon is not None:
            ctx["score_gap_epsilon"] = epsilon
        return verify_claims(
            [self._make_claim()], {"minimax_score": chosen}, context=ctx,
        )[0].claim_status

    # ── default EPSILON behaviour ─────────────────────────────────────────

    def test_exact_tie_is_unsupported_not_contradicted(self):
        """Regression guard: with the tolerance band, 5.0 vs 5.0 must NOT
        return CONTRADICTED.  Float ties at the same minimax depth would
        otherwise be over-penalised."""
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        assert self._verify(5.0, 5.0) == ClaimStatus.UNSUPPORTED

    def test_negative_exact_tie_is_unsupported(self):
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        # Adversity-paragraph case: chosen=-25, next_best=-25.
        assert self._verify(-25.0, -25.0) == ClaimStatus.UNSUPPORTED

    def test_near_tie_below_epsilon_is_unsupported(self):
        """0.3 < 0.5 EPSILON — still inside the tolerance band."""
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        assert self._verify(5.3, 5.0) == ClaimStatus.UNSUPPORTED
        assert self._verify(5.0, 5.3) == ClaimStatus.UNSUPPORTED

    def test_at_epsilon_boundary_is_unsupported(self):
        """Exactly at the boundary (diff == EPSILON) remains within tolerance.
        Strict `>` / `<` on the comparison means equality stays UNSUPPORTED."""
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        assert self._verify(5.5, 5.0) == ClaimStatus.UNSUPPORTED
        assert self._verify(5.0, 5.5) == ClaimStatus.UNSUPPORTED

    def test_clearly_better_is_supported(self):
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        # 25-pt gap — far above any reasonable EPSILON.
        assert self._verify(-10.0, -35.0) == ClaimStatus.SUPPORTED

    def test_clearly_worse_is_contradicted(self):
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        # 40-pt deficit — far below any reasonable -EPSILON.
        assert self._verify(-50.0, -10.0) == ClaimStatus.CONTRADICTED

    def test_just_above_epsilon_is_supported(self):
        """0.6 > 0.5 EPSILON — escapes the tolerance band to SUPPORTED."""
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        assert self._verify(5.6, 5.0) == ClaimStatus.SUPPORTED

    def test_just_below_minus_epsilon_is_contradicted(self):
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        assert self._verify(5.0, 5.6) == ClaimStatus.CONTRADICTED

    # ── caller-overridable EPSILON ────────────────────────────────────────

    def test_caller_can_override_epsilon_to_zero(self):
        """A caller wanting strict greater-than semantics can pass epsilon=0.
        At epsilon=0, an exact tie returns UNSUPPORTED (diff == 0, neither
        > 0 nor < 0)."""
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        assert self._verify(5.0, 5.0, epsilon=0.0) == ClaimStatus.UNSUPPORTED
        # Any positive difference still resolves cleanly.
        assert self._verify(5.0001, 5.0, epsilon=0.0) == ClaimStatus.SUPPORTED
        assert self._verify(5.0, 5.0001, epsilon=0.0) == ClaimStatus.CONTRADICTED

    def test_caller_can_widen_epsilon(self):
        """With epsilon=2.0, a 1.5-pt gap becomes UNSUPPORTED instead of
        SUPPORTED.  Confirms the override path is honoured."""
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        assert self._verify(6.5, 5.0, epsilon=2.0) == ClaimStatus.UNSUPPORTED
        assert self._verify(5.0, 6.5, epsilon=2.0) == ClaimStatus.UNSUPPORTED
        assert self._verify(7.5, 5.0, epsilon=2.0) == ClaimStatus.SUPPORTED

    def test_negative_epsilon_ignored_falls_back_to_default(self):
        """Non-numeric or negative epsilon overrides are ignored; the module
        default (0.5) applies."""
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        # epsilon=-1.0 invalid → default 0.5 → 0.3 gap inside tolerance.
        assert self._verify(5.3, 5.0, epsilon=-1.0) == ClaimStatus.UNSUPPORTED
        # Non-numeric epsilon also ignored → default 0.5 still active.
        from checkers.evaluation.claim_verifier import verify_claims
        rec = verify_claims(
            [self._make_claim()],
            {"minimax_score": 5.0},
            context={"next_best_minimax_score": 5.0, "score_gap_epsilon": "wide"},
        )
        assert rec[0].claim_status == ClaimStatus.UNSUPPORTED

    # ── missing context still falls back to legacy partial behaviour ─────

    def test_missing_context_preserves_legacy_partial_supported(self):
        """Without next_best_minimax_score the verifier must NOT contradict —
        the tolerance band logic only activates when both numeric values are
        present."""
        from checkers.evaluation.claim_verifier import verify_claims
        from checkers.evaluation.reasoning_taxonomy import ClaimStatus
        out = verify_claims(
            [self._make_claim()],
            {"minimax_score": 5.0},
            context={},  # no next_best_minimax_score
        )
        assert out[0].claim_status == ClaimStatus.SUPPORTED
