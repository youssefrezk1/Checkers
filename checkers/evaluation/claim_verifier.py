# checkers/evaluation/claim_verifier.py
#
# Deterministic symbolic fact-checker for extracted claim records.
#
# PURPOSE
# -------
# Takes a list of ClaimRecord objects (as produced by claim_extractor.py) and
# a facts dict (as produced by compute_move_facts()) and updates each record's
# claim_status to one of:
#
#     SUPPORTED      — at least one symbolic fact unambiguously confirms the claim
#     CONTRADICTED   — at least one symbolic fact directly refutes the claim
#     UNSUPPORTED    — no symbolic fact confirms the claim (but none refutes it)
#     VAGUE          — claim is structurally unverifiable; fact cannot resolve it
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls.  All logic is pure boolean / numeric comparison.
# - Deterministic: same (claims, facts) → same output always.
# - Inputs are never mutated; new ClaimRecord objects are returned.
# - Only fact fields already produced by compute_move_facts() are used.
#   No new fields are invented.
# - Strategic claims (positional_pressure, strategic_initiative,
#   long_term_compensation) are structurally unverifiable and MUST never
#   become SUPPORTED regardless of facts.
#
# USAGE
# -----
# This module is imported only by evaluation scripts and tests.
# It must NEVER be imported by the runtime pipeline.

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Optional

from checkers.evaluation.reasoning_taxonomy import (
    ClaimStatus,
    HallucinationType,
)
from checkers.evaluation.claim_extractor import ClaimRecord


# ---------------------------------------------------------------------------
# Claim types that are structurally unverifiable
# ---------------------------------------------------------------------------
# These claims can never be SUPPORTED by symbolic move facts alone.
# They are strategic/subjective assertions; the best achievable status is VAGUE.

_ALWAYS_VAGUE: frozenset[str] = frozenset({
    "positional_pressure",
    "strategic_initiative",
    "long_term_compensation",
})


# ---------------------------------------------------------------------------
# Verification rules
# ---------------------------------------------------------------------------
# Each rule is a function with signature:
#   (facts: dict[str, Any]) -> ClaimStatus
#
# The function must be total: it always returns a ClaimStatus.
# Rules use only keys defined in the compute_move_facts() return dict.
# If a required fact key is absent, the rule returns UNSUPPORTED (conservative).

def _get(facts: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safe fact lookup. Returns default if key absent."""
    return facts.get(key, default)


def _verify_avoids_recapture(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : opponent_can_recapture is False
    CONTRADICTED: opponent_can_recapture is True
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "opponent_can_recapture")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is False:
        return ClaimStatus.SUPPORTED
    # val is True
    return ClaimStatus.CONTRADICTED


def _verify_can_be_recaptured(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : opponent_can_recapture is True
    CONTRADICTED: opponent_can_recapture is False
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "opponent_can_recapture")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_gains_material(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : net_gain > 0  (at least one opponent piece captured)
    CONTRADICTED: net_gain <= 0 when claim was made
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "net_gain")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if isinstance(val, (int, float)) and val > 0:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_promotes_to_king(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : results_in_king is True
    CONTRADICTED: results_in_king is False
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "results_in_king")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_near_promotion(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : near_promotion is True  AND results_in_king is not True
                 (piece is one step from promotion but has not yet promoted)
    CONTRADICTED: near_promotion is False and results_in_king is also False
                  (piece is neither near nor at promotion)
    UNSUPPORTED : near_promotion absent
    VAGUE       : near_promotion is False but results_in_king is True
                  (piece promoted — the near_promotion claim is vacuously wrong
                  but the broader trajectory claim is not easily refuted)

    NOTE: near_promotion is PARTIALLY_VERIFIABLE — we accept SUPPORTED
    even though it is not a hard tactical fact.
    """
    near = _get(facts, "near_promotion")
    king = _get(facts, "results_in_king")
    if near is None:
        return ClaimStatus.UNSUPPORTED
    if near is True and king is not True:
        return ClaimStatus.SUPPORTED
    if near is False and king is not True:
        return ClaimStatus.CONTRADICTED
    # near is False but king is True — promoted rather than near
    return ClaimStatus.UNSUPPORTED


def _verify_opponent_near_promotion(facts: dict[str, Any]) -> ClaimStatus:
    """
    Verifies claims about the OPPONENT'S piece being near promotion.
    Used when the LLM correctly cites the opponent_near_promotion board-context
    seed (e.g. "the opponent has a piece one step from promotion").

    SUPPORTED    : opponent_near_promotion is True
    CONTRADICTED : opponent_near_promotion is False
    UNSUPPORTED  : fact absent

    Distinct from _verify_near_promotion, which checks our own moving piece.
    """
    val = _get(facts, "opponent_near_promotion")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_mobility_decrease(facts: dict[str, Any]) -> ClaimStatus:
    """
    Primary path  — opponent mobility:
      SUPPORTED  : mobility_reduction > 0  (opponent has fewer moves after)
      CONTRADICTED: mobility_reduction < 0  (opponent gained mobility)
      val == 0    : opponent mobility unchanged — fall through to self-mobility check

    Secondary path — self mobility (covers "reduces our mobility" phrasings):
      SUPPORTED  : our_mobility_after < our_mobility_before
      CONTRADICTED: our_mobility_after > our_mobility_before

    UNSUPPORTED : all relevant facts absent or no net change in either path
    """
    val = _get(facts, "mobility_reduction")
    if val is not None and isinstance(val, (int, float)):
        if val > 0:
            return ClaimStatus.SUPPORTED
        if val < 0:
            return ClaimStatus.CONTRADICTED
        # val == 0: opponent mobility unchanged.  Do NOT return here — the claim
        # may be about our own mobility decreasing, so fall through to the
        # our_mobility_before/after check below.

    our_before = _get(facts, "our_mobility_before")
    our_after = _get(facts, "our_mobility_after")
    if our_before is not None and our_after is not None:
        if isinstance(our_before, (int, float)) and isinstance(our_after, (int, float)):
            if our_after < our_before:
                return ClaimStatus.SUPPORTED
            if our_after > our_before:
                return ClaimStatus.CONTRADICTED
    return ClaimStatus.UNSUPPORTED


def _verify_mobility_increase(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : our_mobility_after > our_mobility_before
    CONTRADICTED: our_mobility_after < our_mobility_before
    UNSUPPORTED : equal or facts absent
    """
    before = _get(facts, "our_mobility_before")
    after = _get(facts, "our_mobility_after")
    if before is None or after is None:
        return ClaimStatus.UNSUPPORTED
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if after > before:
            return ClaimStatus.SUPPORTED
        if after < before:
            return ClaimStatus.CONTRADICTED
    return ClaimStatus.UNSUPPORTED  # no change


def _verify_piece_isolated(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : leaves_piece_isolated is True
    CONTRADICTED: leaves_piece_isolated is False
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "leaves_piece_isolated")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_piece_connected(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : leaves_piece_isolated is False
    CONTRADICTED: leaves_piece_isolated is True
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "leaves_piece_isolated")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is False:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_weakens_king_row(facts: dict[str, Any]) -> ClaimStatus:
    """
    Verify a claim that moving from the back row structurally weakens
    the back-rank defence.

    The symbolic fact weakens_king_row (computed by move_facts._weakens_king_row)
    captures the exact condition:
        - piece started on the king row (row 7 for RED, row 0 for BLACK)
        - no capture or promotion compensates the departure
        - fewer than 3 of our own pieces remain on that row after the move

    SUPPORTED   : weakens_king_row is True
    CONTRADICTED: weakens_king_row is False  (back-row piece moved but structure intact)
    UNSUPPORTED : fact absent (non-back-row move — claim is unverifiable in context)
    """
    val = _get(facts, "weakens_king_row")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    # val is False — seed says structure is intact; claim of weakening is contradicted
    return ClaimStatus.CONTRADICTED


def _verify_creates_immediate_threat(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : creates_immediate_threat is True
    CONTRADICTED: creates_immediate_threat is False AND
                  shot_sequence_available is False
    UNSUPPORTED : facts absent
    """
    cit = _get(facts, "creates_immediate_threat")
    ssa = _get(facts, "shot_sequence_available")
    if cit is None and ssa is None:
        return ClaimStatus.UNSUPPORTED
    if cit is True or ssa is True:
        return ClaimStatus.SUPPORTED
    # Both present and both False
    if cit is False and ssa is False:
        return ClaimStatus.CONTRADICTED
    return ClaimStatus.UNSUPPORTED


def _verify_center_control(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED   : center_control is True  (engine evaluator confirms center control)
    CONTRADICTED: center_control is False (piece lands in center geometrically but
                  the evaluator did NOT award center control — claim is an overclaim)
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "center_control")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_shot_sequence_or_multi_jump(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : shot_sequence_available is True
    CONTRADICTED: shot_sequence_available is False
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "shot_sequence_available")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_blocks_landing_square(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : blocks_opponent_landing is True
    CONTRADICTED: blocks_opponent_landing is False
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "blocks_opponent_landing")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_forced_opponent_jump(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : forced_opponent_jump_reply is True
    CONTRADICTED: forced_opponent_jump_reply is False
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "forced_opponent_jump_reply")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if val is True:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# Phase 6 — adversity / losing-position claim verifiers
# ---------------------------------------------------------------------------
# These verifiers correspond to the seeds emitted by
# explainer_agent._build_adversity_context_seeds.  Each is purely symbolic — no
# strategic interpretation — and reads only fields already produced by
# compute_move_facts().

def _verify_material_deficit(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : material_advantage < 0   (we are behind in material)
    CONTRADICTED: material_advantage >= 0 (we are even or ahead)
    UNSUPPORTED : fact absent
    """
    val = _get(facts, "material_advantage")
    if val is None or not isinstance(val, (int, float)):
        return ClaimStatus.UNSUPPORTED
    if val < 0:
        return ClaimStatus.SUPPORTED
    return ClaimStatus.CONTRADICTED


def _verify_threat_reduction(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : our_pieces_threatened_before > 0 AND
                 our_pieces_threatened_after < our_pieces_threatened_before
    CONTRADICTED: our_pieces_threatened_after >= our_pieces_threatened_before
                  (claim asserts reduction but no reduction occurred)
    UNSUPPORTED : either field absent
    """
    before = _get(facts, "our_pieces_threatened_before")
    after  = _get(facts, "our_pieces_threatened_after")
    if before is None or after is None:
        return ClaimStatus.UNSUPPORTED
    if not (isinstance(before, (int, float)) and isinstance(after, (int, float))):
        return ClaimStatus.UNSUPPORTED
    if before > 0 and after < before:
        return ClaimStatus.SUPPORTED
    if after >= before:
        return ClaimStatus.CONTRADICTED
    # before == 0 and after < 0 is impossible — but be conservative
    return ClaimStatus.UNSUPPORTED


def _verify_score_gap_advantage(facts: dict[str, Any]) -> ClaimStatus:
    """
    LEGACY (context-free) verifier — preserved for backwards compatibility.

    score_gap_advantage is PARTIALLY_VERIFIABLE without cross-candidate data:
    a single move's facts dict cannot reproduce the gap-vs-best-alternative
    figure.  When no `next_best_minimax_score` context is supplied:

        SUPPORTED   : minimax_score present and numeric
        UNSUPPORTED : missing minimax_score
        Never returns CONTRADICTED.

    The hard-verifying path is exposed via
    `_verify_score_gap_advantage_with_context`, which the dispatch table
    prefers when `context["next_best_minimax_score"]` is available.
    """
    val = _get(facts, "minimax_score")
    if isinstance(val, (int, float)):
        return ClaimStatus.SUPPORTED
    return ClaimStatus.UNSUPPORTED


# Tolerance band around an exact-tie between chosen_score and next_best_score.
# Within ±SCORE_GAP_EPSILON the claim "scores better than next-best" is
# neither symbolically supported nor refuted — float ties at the same minimax
# depth are common, and CONTRADICTED would over-penalise legitimate near-ties.
#
# Default of 0.5 is a small fraction of a single piece value (≈ 100 in the
# engine's evaluator units), so "clearly better" / "clearly worse" verdicts
# remain unchanged.  Override via environment variable for experimentation.
import os as _os
_SCORE_GAP_EPSILON: float = float(_os.environ.get("SCORE_GAP_EPSILON", "0.5"))


def _verify_score_gap_advantage_with_context(
    facts: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
) -> ClaimStatus:
    """
    Phase-6 Fix 2 + tie-tolerance follow-up: hard verification when next-best
    candidate score is known, with a small tolerance band around exact ties.

    The runtime exposes `next_best_minimax_score` in ranker_diagnostics; the
    evaluator forwards it via context.  The active tolerance EPSILON is
    `context.get("score_gap_epsilon")` when numeric and non-negative,
    otherwise the module default `_SCORE_GAP_EPSILON`.

        SUPPORTED   : chosen > next_best + EPSILON   (clearly better)
        CONTRADICTED: chosen < next_best - EPSILON   (clearly worse)
        UNSUPPORTED : |chosen - next_best| <= EPSILON
                      (near-tie — the symbolic gap does not justify a
                      "better than next-best" claim, but the float values
                      are too close to safely refute it either)

    When context is missing or next_best_minimax_score is None / non-numeric,
    fall back to the legacy context-free verifier (never CONTRADICTED).
    """
    ctx_val: Any = None
    eps: float = _SCORE_GAP_EPSILON
    if isinstance(context, dict):
        ctx_val = context.get("next_best_minimax_score")
        ctx_eps = context.get("score_gap_epsilon")
        if isinstance(ctx_eps, (int, float)) and ctx_eps >= 0:
            eps = float(ctx_eps)

    chosen_val = _get(facts, "minimax_score")

    have_ctx    = isinstance(ctx_val, (int, float))
    have_chosen = isinstance(chosen_val, (int, float))

    if not (have_ctx and have_chosen):
        # Insufficient context — preserve conservative partial behaviour.
        return _verify_score_gap_advantage(facts)

    diff = float(chosen_val) - float(ctx_val)
    if diff > eps:
        return ClaimStatus.SUPPORTED
    if diff < -eps:
        return ClaimStatus.CONTRADICTED
    # Within the tolerance band — the claim "scores better than next-best" is
    # not symbolically supported, but the gap is too small to call a hard
    # contradiction.  Conservative outcome: UNSUPPORTED.
    return ClaimStatus.UNSUPPORTED


def _verify_mobility_asymmetry(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : opponent_mobility_before - our_mobility_before >= 3
                 (matches the adversity seed's symbolic threshold)
    CONTRADICTED: opponent_mobility_before - our_mobility_before < 0
                  (asymmetry is in our favour — claim of disadvantage refuted)
    UNSUPPORTED : facts absent OR gap is in [0, 3) (claim too strong but not refuted)
    """
    opp = _get(facts, "opponent_mobility_before")
    our = _get(facts, "our_mobility_before")
    if opp is None or our is None:
        return ClaimStatus.UNSUPPORTED
    if not (isinstance(opp, (int, float)) and isinstance(our, (int, float))):
        return ClaimStatus.UNSUPPORTED
    gap = opp - our
    if gap >= 3:
        return ClaimStatus.SUPPORTED
    if gap < 0:
        return ClaimStatus.CONTRADICTED
    return ClaimStatus.UNSUPPORTED


def _verify_minimax_confirmation(facts: dict[str, Any]) -> ClaimStatus:
    """
    SUPPORTED  : minimax_score key is present and is a number.
    UNSUPPORTED : key absent.

    NOTE: The verifier does not know whether this move was truly the highest-
    evaluated option — that comparison requires all candidates' scores.
    We can only verify that a minimax score was computed for this move.
    Absence of the key means the claim is unverifiable.
    """
    val = _get(facts, "minimax_score")
    if val is None:
        return ClaimStatus.UNSUPPORTED
    if isinstance(val, (int, float)):
        return ClaimStatus.SUPPORTED
    return ClaimStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# Always-VAGUE rule (structural)
# ---------------------------------------------------------------------------

def _verify_always_vague(facts: dict[str, Any]) -> ClaimStatus:
    return ClaimStatus.VAGUE


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Context-aware dispatch (Phase-6 Fix 2)
# ---------------------------------------------------------------------------
# Rules registered here receive (facts, context) instead of (facts).
# verify_claims() consults this map first; entries here take precedence over
# the same key in _VERIFICATION_RULES.  Verifiers that don't need context are
# kept in the regular dispatch table unchanged.

_CONTEXT_VERIFICATION_RULES: dict[str, Any] = {
    "score_gap_advantage": _verify_score_gap_advantage_with_context,
}


_VERIFICATION_RULES: dict[str, Any] = {
    "avoids_recapture":            _verify_avoids_recapture,
    "can_be_recaptured":           _verify_can_be_recaptured,
    "gains_material":              _verify_gains_material,
    "promotes_to_king":            _verify_promotes_to_king,
    "near_promotion":              _verify_near_promotion,
    "opponent_near_promotion":     _verify_opponent_near_promotion,
    "mobility_decrease":           _verify_mobility_decrease,
    "mobility_increase":           _verify_mobility_increase,
    "piece_isolated":              _verify_piece_isolated,
    "piece_connected":             _verify_piece_connected,
    "weakens_king_row":            _verify_weakens_king_row,
    "creates_immediate_threat":    _verify_creates_immediate_threat,
    "center_control":              _verify_center_control,
    "minimax_confirmation":        _verify_minimax_confirmation,
    # Phase 4.1 — previously unverified tactical claim types:
    "shot_sequence_or_multi_jump": _verify_shot_sequence_or_multi_jump,
    "blocks_landing_square":       _verify_blocks_landing_square,
    "forced_opponent_jump":        _verify_forced_opponent_jump,
    # Phase 6 — adversity / losing-position claim types:
    "material_deficit":            _verify_material_deficit,
    "threat_reduction":            _verify_threat_reduction,
    "score_gap_advantage":         _verify_score_gap_advantage,
    "mobility_asymmetry":          _verify_mobility_asymmetry,
    # Structurally unverifiable:
    "positional_pressure":         _verify_always_vague,
    "strategic_initiative":        _verify_always_vague,
    "long_term_compensation":      _verify_always_vague,
}


# ---------------------------------------------------------------------------
# Hallucination upgrade
# ---------------------------------------------------------------------------

def _upgrade_hallucination(
    record: ClaimRecord,
    new_status: ClaimStatus,
) -> Optional[HallucinationType]:
    """
    Return the appropriate HallucinationType given the verified status.

    CONTRADICTED → FACTUAL_CONTRADICTION  (always, overrides prior annotation)
    VAGUE        → OVERCLAIM              (strategic claim, only if not already set)
    SUPPORTED    → None                   (facts confirm the claim; clear prior annotation)
    Otherwise    → preserve existing annotation
    """
    if new_status == ClaimStatus.CONTRADICTED:
        return HallucinationType.FACTUAL_CONTRADICTION
    if new_status == ClaimStatus.SUPPORTED:
        # Facts now confirm the claim — clear any prior hallucination label
        return None
    if (
        new_status == ClaimStatus.VAGUE
        and record.hallucination_type is None
    ):
        # Unverifiable strategic claim — flag as overclaim
        return HallucinationType.OVERCLAIM
    return record.hallucination_type  # preserve existing annotation (UNSUPPORTED, etc.)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_claims(
    claims: list[ClaimRecord],
    facts: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
) -> list[ClaimRecord]:
    """
    Update ClaimRecord statuses using symbolic move facts.

    For each ClaimRecord, look up the deterministic verification rule for its
    claim_type and apply it against the provided facts dict.  Returns a new
    list of ClaimRecord objects with updated claim_status and hallucination_type
    fields.  The original list is never mutated.

    Parameters
    ----------
    claims : list[ClaimRecord]
        As produced by extract_claims().  May be empty.
    facts : dict[str, Any]
        As produced by compute_move_facts().  Keys not present are treated
        as absent (conservative — the verifier returns UNSUPPORTED, not error).
    context : dict or None, optional
        Optional cross-candidate context.  Currently consumed by
        score_gap_advantage to enable hard verification when
        next_best_minimax_score is available (Phase-6 Fix 2).  Unknown keys
        are ignored.  When None, context-aware verifiers fall back to their
        legacy partial behaviour.

    Returns
    -------
    list[ClaimRecord]
        New list of ClaimRecord instances.  Order is preserved.  Length equals
        len(claims).  Each record is an independent copy with updated status.

    Guarantees
    ----------
    - Deterministic: same inputs → same outputs always.
    - inputs are not mutated.
    - Claims with no rule in the dispatch table are returned unchanged.
    - Structurally unverifiable claims (positional_pressure etc.) receive
      VAGUE status and OVERCLAIM hallucination type — never SUPPORTED.
    """
    if not isinstance(facts, dict):
        # facts is None or wrong type — leave claims unchanged (conservative).
        # Note: an empty dict {} IS valid and will be processed; rules will
        # return UNSUPPORTED for absent keys, which is the correct behavior.
        return list(claims)

    ctx: Optional[dict[str, Any]] = context if isinstance(context, dict) else None

    result: list[ClaimRecord] = []

    for record in claims:
        ctx_rule = _CONTEXT_VERIFICATION_RULES.get(record.claim_type)
        if ctx_rule is not None:
            # Context-aware verifier — receives (facts, context).  Even when
            # `ctx` is None the verifier handles it (falls back to legacy).
            new_status: ClaimStatus = ctx_rule(facts, ctx)
        else:
            rule = _VERIFICATION_RULES.get(record.claim_type)
            if rule is None:
                # Unknown claim type — return unchanged (conservative)
                result.append(record)
                continue
            new_status = rule(facts)

        # Determine hallucination annotation
        new_hallucination = _upgrade_hallucination(record, new_status)

        # Build a new ClaimRecord with updated fields only
        # (dataclasses.replace produces a shallow copy)
        updated = replace(
            record,
            claim_status=new_status,
            hallucination_type=new_hallucination,
        )
        result.append(updated)

    return result
