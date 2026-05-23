# checkers/evaluation/metrics/pre_post_repair.py
#
# Factuality metric — measures how much the truthfulness-refinement loop
# inside ranker_agent reduces unsupported / contradicted claims by
# comparing the reasoning BEFORE and AFTER refinement.
#
# Source fields consumed from each evaluation_source/*.jsonl record:
#   - last_move_reasoning                                  (post-repair text)
#   - ranker_diagnostics["raw_llm_reasoning_pre_refinement"] (pre-repair text)
#   - ranker_diagnostics["reasoning_seeds"]
#   - chosen_move_facts
#
# Skip semantics:
#   - When raw_llm_reasoning_pre_refinement is None, no refinement happened
#     (or the pipeline pre-dates the snapshot field). The turn is reported
#     in `post_only_turns` and contributes only to post-repair stats.
#   - When pre == post text, the refinement loop produced an unchanged
#     output (typical case: no initial contradictions). Pre and post stats
#     are identical for such turns and they are counted in `unchanged_turns`.
#
# Determinism: identical inputs always produce identical outputs.
# No LLM calls. No external state. No imports from runtime pipeline.

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from checkers.evaluation.claim_extractor import extract_claims
from checkers.evaluation.claim_verifier import verify_claims
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


# ---------------------------------------------------------------------------
# Per-turn record
# ---------------------------------------------------------------------------

@dataclass
class _Counts:
    total:        int = 0
    supported:    int = 0
    contradicted: int = 0
    unsupported:  int = 0
    vague:        int = 0


@dataclass
class PrePostRepairTurn:
    """
    Pre/post repair outcome for a single turn.

    pre_*  values are None when no pre-refinement snapshot is available.
    post_* values are always populated (post = final reasoning).
    """
    turn_id: str
    refinement_attempted: bool
    pre_text_available: bool
    pre_total:        Optional[int] = None
    pre_supported:    Optional[int] = None
    pre_contradicted: Optional[int] = None
    pre_unsupported:  Optional[int] = None
    pre_vague:        Optional[int] = None
    post_total:        int = 0
    post_supported:    int = 0
    post_contradicted: int = 0
    post_unsupported:  int = 0
    post_vague:        int = 0
    contradiction_resolved: Optional[bool] = None   # True if pre>0 contra, post==0
    contradiction_introduced: Optional[bool] = None # True if pre==0 contra, post>0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrePostRepairSummary:
    """
    Corpus-level aggregate over a batch of turns.

    Rates are over turns that have both pre AND post counts (i.e.
    raw_llm_reasoning_pre_refinement was captured). post-only stats use
    every turn. `None` indicates not enough turns to compute the rate.
    """
    n_turns:           int = 0
    n_pre_available:   int = 0
    n_post_only:       int = 0
    n_unchanged:       int = 0   # pre_text == post_text exactly
    n_refinement_runs: int = 0   # reasoning_refinement_retry_count > 0

    # Claim-level rates (over claims found in pre/post reasoning across turns).
    # micro_* averages weight every claim equally; macro_* averages weight every
    # turn equally (a turn with 0 claims contributes 0 to the numerator).
    pre_repair_supported_rate_micro:        Optional[float] = None
    pre_repair_contradiction_rate_micro:    Optional[float] = None
    post_repair_supported_rate_micro:       Optional[float] = None
    post_repair_contradiction_rate_micro:   Optional[float] = None

    pre_repair_supported_rate_macro:        Optional[float] = None
    pre_repair_contradiction_rate_macro:    Optional[float] = None
    post_repair_supported_rate_macro:       Optional[float] = None
    post_repair_contradiction_rate_macro:   Optional[float] = None

    # Differences are POSITIVE when repair HELPS:
    #   contradiction_reduction = pre_contra_rate - post_contra_rate
    #   support_gain            = post_support_rate - pre_support_rate
    contradiction_reduction_micro: Optional[float] = None
    contradiction_reduction_macro: Optional[float] = None
    support_gain_micro:            Optional[float] = None
    support_gain_macro:            Optional[float] = None

    # Turn-level repair effectiveness over turns where pre had ≥1 contradiction.
    repair_effectiveness:   Optional[float] = None  # resolved / contradicted_turns
    n_contradicted_turns:   int = 0
    n_repair_resolved:      int = 0
    n_repair_introduced:    int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_statuses(
    reasoning: str,
    seeds: List[str],
    facts: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> _Counts:
    if not isinstance(reasoning, str) or not reasoning.strip():
        return _Counts()
    raw = extract_claims(reasoning, reasoning_seeds=seeds, facts=facts)
    verified = verify_claims(raw, facts or {}, context=context)
    c = _Counts(total=len(verified))
    for r in verified:
        if r.claim_status == ClaimStatus.SUPPORTED:
            c.supported += 1
        elif r.claim_status == ClaimStatus.CONTRADICTED:
            c.contradicted += 1
        elif r.claim_status == ClaimStatus.UNSUPPORTED:
            c.unsupported += 1
        elif r.claim_status == ClaimStatus.VAGUE:
            c.vague += 1
    return c


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Public API — per turn
# ---------------------------------------------------------------------------

def evaluate_pre_post_repair(
    record: Dict[str, Any],
    turn_id: Optional[str] = None,
) -> PrePostRepairTurn:
    """
    Compute pre/post repair claim counts for a single evaluation-source record.

    Parameters
    ----------
    record : dict
        One line from logs/evaluation_source/<game>.jsonl.
        Required keys (any missing → handled conservatively):
            last_move_reasoning, ranker_diagnostics, chosen_move_facts
    turn_id : str or None
        Override turn id. Defaults to record["turn_id"] or "unknown".
    """
    tid = (
        turn_id
        if isinstance(turn_id, str) and turn_id
        else (record.get("turn_id") if isinstance(record.get("turn_id"), str) else "unknown")
    )

    diag    = record.get("ranker_diagnostics") or {}
    facts   = record.get("chosen_move_facts") or {}
    seeds   = [s for s in (diag.get("reasoning_seeds") or []) if isinstance(s, str)]
    nb      = diag.get("next_best_minimax_score")
    ctx: Dict[str, Any] = {}
    if isinstance(nb, (int, float)):
        ctx["next_best_minimax_score"] = nb

    post_text = record.get("last_move_reasoning") or ""
    if not isinstance(post_text, str):
        post_text = ""

    pre_text  = diag.get("raw_llm_reasoning_pre_refinement")
    pre_avail = isinstance(pre_text, str) and bool(pre_text.strip())

    refinement_attempted = bool(diag.get("reasoning_contradiction_detected", False)) \
        or int(diag.get("reasoning_refinement_retry_count", 0) or 0) > 0

    post_counts = _count_statuses(post_text, seeds, facts, ctx or None)

    if pre_avail:
        pre_counts = _count_statuses(pre_text, seeds, facts, ctx or None)
        contradiction_resolved   = (pre_counts.contradicted > 0 and post_counts.contradicted == 0)
        contradiction_introduced = (pre_counts.contradicted == 0 and post_counts.contradicted > 0)
        return PrePostRepairTurn(
            turn_id=tid,
            refinement_attempted=refinement_attempted,
            pre_text_available=True,
            pre_total=pre_counts.total,
            pre_supported=pre_counts.supported,
            pre_contradicted=pre_counts.contradicted,
            pre_unsupported=pre_counts.unsupported,
            pre_vague=pre_counts.vague,
            post_total=post_counts.total,
            post_supported=post_counts.supported,
            post_contradicted=post_counts.contradicted,
            post_unsupported=post_counts.unsupported,
            post_vague=post_counts.vague,
            contradiction_resolved=contradiction_resolved,
            contradiction_introduced=contradiction_introduced,
        )

    return PrePostRepairTurn(
        turn_id=tid,
        refinement_attempted=refinement_attempted,
        pre_text_available=False,
        post_total=post_counts.total,
        post_supported=post_counts.supported,
        post_contradicted=post_counts.contradicted,
        post_unsupported=post_counts.unsupported,
        post_vague=post_counts.vague,
    )


# ---------------------------------------------------------------------------
# Public API — corpus
# ---------------------------------------------------------------------------

def aggregate_pre_post_repair(
    turns: List[PrePostRepairTurn],
) -> PrePostRepairSummary:
    """
    Aggregate a batch of per-turn PrePostRepairTurn records into corpus stats.

    All rates are returned as None when no turns contribute the data needed
    to compute them. Differences are POSITIVE when repair HELPS.
    """
    summary = PrePostRepairSummary(n_turns=len(turns))

    pre_turns  = [t for t in turns if t.pre_text_available and t.pre_total is not None]
    summary.n_pre_available = len(pre_turns)
    summary.n_post_only     = sum(1 for t in turns if not t.pre_text_available)
    summary.n_refinement_runs = sum(1 for t in turns if t.refinement_attempted)
    summary.n_unchanged       = sum(
        1 for t in turns
        if t.pre_text_available
        and t.pre_total       == t.post_total
        and t.pre_supported   == t.post_supported
        and t.pre_contradicted == t.post_contradicted
        and t.pre_unsupported == t.post_unsupported
        and t.pre_vague       == t.post_vague
    )

    # ── post-repair stats (every turn) ─────────────────────────────────────
    post_total_claims    = sum(t.post_total for t in turns)
    post_supported_claims    = sum(t.post_supported for t in turns)
    post_contradicted_claims = sum(t.post_contradicted for t in turns)

    summary.post_repair_supported_rate_micro     = _safe_div(post_supported_claims, post_total_claims)
    summary.post_repair_contradiction_rate_micro = _safe_div(post_contradicted_claims, post_total_claims)

    # Macro: average per-turn ratio over turns with ≥1 claim.
    post_turns_with_claims = [t for t in turns if t.post_total > 0]
    if post_turns_with_claims:
        summary.post_repair_supported_rate_macro = sum(
            t.post_supported / t.post_total for t in post_turns_with_claims
        ) / len(post_turns_with_claims)
        summary.post_repair_contradiction_rate_macro = sum(
            t.post_contradicted / t.post_total for t in post_turns_with_claims
        ) / len(post_turns_with_claims)

    if not pre_turns:
        return summary

    # ── pre-repair stats (only turns with snapshot) ────────────────────────
    pre_total_claims    = sum((t.pre_total or 0) for t in pre_turns)
    pre_supported_claims    = sum((t.pre_supported or 0) for t in pre_turns)
    pre_contradicted_claims = sum((t.pre_contradicted or 0) for t in pre_turns)

    summary.pre_repair_supported_rate_micro     = _safe_div(pre_supported_claims, pre_total_claims)
    summary.pre_repair_contradiction_rate_micro = _safe_div(pre_contradicted_claims, pre_total_claims)

    pre_turns_with_claims = [t for t in pre_turns if (t.pre_total or 0) > 0]
    if pre_turns_with_claims:
        summary.pre_repair_supported_rate_macro = sum(
            (t.pre_supported or 0) / (t.pre_total or 1) for t in pre_turns_with_claims
        ) / len(pre_turns_with_claims)
        summary.pre_repair_contradiction_rate_macro = sum(
            (t.pre_contradicted or 0) / (t.pre_total or 1) for t in pre_turns_with_claims
        ) / len(pre_turns_with_claims)

    # ── deltas (matched: only turns with both pre and post computed) ──────
    # Recompute post stats restricted to pre-available turns so deltas are
    # apples-to-apples.
    post_total_m         = sum(t.post_total for t in pre_turns)
    post_supported_m     = sum(t.post_supported for t in pre_turns)
    post_contradicted_m  = sum(t.post_contradicted for t in pre_turns)
    pre_total_m          = sum((t.pre_total or 0) for t in pre_turns)
    pre_supported_m      = sum((t.pre_supported or 0) for t in pre_turns)
    pre_contradicted_m   = sum((t.pre_contradicted or 0) for t in pre_turns)

    pre_contra_rate_m = _safe_div(pre_contradicted_m, pre_total_m)
    post_contra_rate_m = _safe_div(post_contradicted_m, post_total_m)
    pre_sup_rate_m    = _safe_div(pre_supported_m, pre_total_m)
    post_sup_rate_m   = _safe_div(post_supported_m, post_total_m)

    if pre_contra_rate_m is not None and post_contra_rate_m is not None:
        summary.contradiction_reduction_micro = pre_contra_rate_m - post_contra_rate_m
    if pre_sup_rate_m is not None and post_sup_rate_m is not None:
        summary.support_gain_micro = post_sup_rate_m - pre_sup_rate_m

    if pre_turns_with_claims:
        # Macro: average of per-turn (pre_rate - post_rate)
        per_turn_contra_red = [
            ((t.pre_contradicted or 0) / (t.pre_total or 1))
            - (t.post_contradicted / t.post_total if t.post_total > 0 else 0.0)
            for t in pre_turns_with_claims
        ]
        per_turn_sup_gain = [
            (t.post_supported / t.post_total if t.post_total > 0 else 0.0)
            - ((t.pre_supported or 0) / (t.pre_total or 1))
            for t in pre_turns_with_claims
        ]
        summary.contradiction_reduction_macro = sum(per_turn_contra_red) / len(per_turn_contra_red)
        summary.support_gain_macro = sum(per_turn_sup_gain) / len(per_turn_sup_gain)

    # ── repair effectiveness (turn-level) ──────────────────────────────────
    contradicted_turns = [t for t in pre_turns if (t.pre_contradicted or 0) > 0]
    summary.n_contradicted_turns = len(contradicted_turns)
    summary.n_repair_resolved    = sum(1 for t in contradicted_turns if t.contradiction_resolved)
    summary.n_repair_introduced  = sum(1 for t in pre_turns if t.contradiction_introduced)
    if contradicted_turns:
        summary.repair_effectiveness = summary.n_repair_resolved / len(contradicted_turns)

    return summary
