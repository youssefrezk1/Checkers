# checkers/evaluation/metrics/by_claim_type.py
#
# Factuality breakdown by claim_type.
#
# Identifies which strategic claim families are most error-prone and
# which the refinement loop repairs successfully.
#
# For every claim_type observed in the batch the aggregator reports:
#   total_claims
#   supported_rate / contradicted_rate / unsupported_rate / vague_rate
#   unverifiable_rate   (claims whose taxonomy category is structurally
#                        non-verifiable: NON_VERIFIABLE_VAGUE or
#                        FORBIDDEN_UNGROUNDED)
#   pre/post deltas     — computed only over turns where the pre-refinement
#                          reasoning snapshot is available
#       pre_total / post_total
#       pre_supported_rate / post_supported_rate
#       pre_contradicted_rate / post_contradicted_rate
#       contradiction_reduction
#       support_gain
#
# Deterministic.  No LLM calls.  No runtime pipeline imports.
#
# Backward-compatible with logs missing raw_llm_reasoning_pre_refinement:
# the per-type "pre_*" fields stay None and only "post_*" + corpus rates
# are populated for those turns.

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from checkers.evaluation.claim_extractor import ClaimRecord
from checkers.evaluation.claim_taxonomy import get_claim_spec, TaxonomyCategory
from checkers.evaluation.reasoning_taxonomy import ClaimStatus
from checkers.evaluation.metrics._record_helpers import pre_post_claims


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _Bucket:
    """Mutable per-status counts used during aggregation."""
    total:        int = 0
    supported:    int = 0
    contradicted: int = 0
    unsupported:  int = 0
    vague:        int = 0
    unverifiable: int = 0   # taxonomy-derived (category in NON_VERIFIABLE_VAGUE / FORBIDDEN_UNGROUNDED)


@dataclass
class ClaimTypeStats:
    """Aggregated statistics for one claim_type across the batch."""
    claim_type:          str
    category:            Optional[str]
    verifier_exists:     Optional[bool]
    total_claims:        int
    supported_rate:      Optional[float]
    contradicted_rate:   Optional[float]
    unsupported_rate:    Optional[float]
    vague_rate:          Optional[float]
    unverifiable_rate:   Optional[float]
    # Pre/post deltas — None when no pre snapshot was available for any
    # turn that produced this claim type.
    pre_total:                    Optional[int]   = None
    post_total:                   Optional[int]   = None
    pre_supported_rate:           Optional[float] = None
    post_supported_rate:          Optional[float] = None
    pre_contradicted_rate:        Optional[float] = None
    post_contradicted_rate:       Optional[float] = None
    contradiction_reduction:      Optional[float] = None   # pre - post (positive = repair helped)
    support_gain:                 Optional[float] = None   # post - pre (positive = repair helped)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimTypeSummary:
    """Corpus-level summary: by_claim_type plus a few headline aggregates."""
    n_turns:             int
    n_pre_available:     int
    total_claims:        int
    distinct_claim_types: int
    by_claim_type:       Dict[str, ClaimTypeStats] = field(default_factory=dict)
    # Top-K helpers ranked deterministically by frequency then claim_type name.
    most_contradicted_types: List[str] = field(default_factory=list)
    most_repaired_types:     List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["by_claim_type"] = {k: v.to_dict() if hasattr(v, "to_dict") else v
                              for k, v in self.by_claim_type.items()}
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NON_VERIFIABLE_CATEGORIES = frozenset({
    TaxonomyCategory.NON_VERIFIABLE_VAGUE.value,
    TaxonomyCategory.FORBIDDEN_UNGROUNDED.value,
})


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _bump(bucket: _Bucket, record: ClaimRecord, *, category_value: Optional[str]) -> None:
    bucket.total += 1
    s = record.claim_status
    if s == ClaimStatus.SUPPORTED:
        bucket.supported += 1
    elif s == ClaimStatus.CONTRADICTED:
        bucket.contradicted += 1
    elif s == ClaimStatus.UNSUPPORTED:
        bucket.unsupported += 1
    elif s == ClaimStatus.VAGUE:
        bucket.vague += 1
    if category_value in _NON_VERIFIABLE_CATEGORIES:
        bucket.unverifiable += 1


def _finalize(
    claim_type: str,
    overall: _Bucket,
    pre: _Bucket,
    post_for_matched: _Bucket,
    pre_available_any: bool,
) -> ClaimTypeStats:
    spec = get_claim_spec(claim_type)
    category_value = spec.category.value if spec is not None else None
    verifier_exists = spec.verifier_exists if spec is not None else None

    pre_total = pre.total if pre_available_any else None
    post_total = post_for_matched.total if pre_available_any else None

    pre_sup_rate    = _safe_div(pre.supported, pre.total) if pre_available_any else None
    pre_contra_rate = _safe_div(pre.contradicted, pre.total) if pre_available_any else None
    post_sup_rate   = _safe_div(post_for_matched.supported, post_for_matched.total) if pre_available_any else None
    post_contra_rate= _safe_div(post_for_matched.contradicted, post_for_matched.total) if pre_available_any else None

    contradiction_reduction: Optional[float] = None
    support_gain:            Optional[float] = None
    if pre_contra_rate is not None and post_contra_rate is not None:
        contradiction_reduction = pre_contra_rate - post_contra_rate
    if pre_sup_rate is not None and post_sup_rate is not None:
        support_gain = post_sup_rate - pre_sup_rate

    return ClaimTypeStats(
        claim_type=claim_type,
        category=category_value,
        verifier_exists=verifier_exists,
        total_claims=overall.total,
        supported_rate=_safe_div(overall.supported, overall.total),
        contradicted_rate=_safe_div(overall.contradicted, overall.total),
        unsupported_rate=_safe_div(overall.unsupported, overall.total),
        vague_rate=_safe_div(overall.vague, overall.total),
        unverifiable_rate=_safe_div(overall.unverifiable, overall.total),
        pre_total=pre_total,
        post_total=post_total,
        pre_supported_rate=pre_sup_rate,
        post_supported_rate=post_sup_rate,
        pre_contradicted_rate=pre_contra_rate,
        post_contradicted_rate=post_contra_rate,
        contradiction_reduction=contradiction_reduction,
        support_gain=support_gain,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_by_claim_type(
    records: Iterable[Mapping[str, Any]],
) -> ClaimTypeSummary:
    """
    Aggregate claims by claim_type across a batch of evaluation-source records.

    Parameters
    ----------
    records : iterable of evaluation-source dicts
        Each dict must contain at minimum last_move_reasoning; missing
        sub-fields are handled conservatively (treated as empty).

    Returns
    -------
    ClaimTypeSummary
        Always populated. Empty input → all rates None, by_claim_type {}.
    """
    overall: Dict[str, _Bucket] = {}
    pre_buckets: Dict[str, _Bucket] = {}
    post_buckets_for_matched: Dict[str, _Bucket] = {}
    matched_claim_types: set = set()

    n_turns         = 0
    n_pre_available = 0

    for rec in records:
        n_turns += 1
        pre_claims, post_claims = pre_post_claims(rec)

        # Overall bucket uses POST-REPAIR claims (final reasoning).
        for c in post_claims:
            spec = get_claim_spec(c.claim_type)
            cat = spec.category.value if spec is not None else None
            b = overall.setdefault(c.claim_type, _Bucket())
            _bump(b, c, category_value=cat)

        # Pre/post matched buckets: only when pre snapshot exists.
        if pre_claims is None:
            continue
        n_pre_available += 1
        for c in pre_claims:
            spec = get_claim_spec(c.claim_type)
            cat = spec.category.value if spec is not None else None
            b = pre_buckets.setdefault(c.claim_type, _Bucket())
            _bump(b, c, category_value=cat)
            matched_claim_types.add(c.claim_type)
        for c in post_claims:
            spec = get_claim_spec(c.claim_type)
            cat = spec.category.value if spec is not None else None
            b = post_buckets_for_matched.setdefault(c.claim_type, _Bucket())
            _bump(b, c, category_value=cat)
            matched_claim_types.add(c.claim_type)

    by_type: Dict[str, ClaimTypeStats] = {}
    all_types = set(overall.keys()) | matched_claim_types
    for ct in sorted(all_types):
        pre_available_any = ct in matched_claim_types and n_pre_available > 0
        by_type[ct] = _finalize(
            claim_type=ct,
            overall=overall.get(ct, _Bucket()),
            pre=pre_buckets.get(ct, _Bucket()),
            post_for_matched=post_buckets_for_matched.get(ct, _Bucket()),
            pre_available_any=pre_available_any,
        )

    # Top-K helpers (deterministic ordering: descending contradiction_rate /
    # reduction, then descending total_claims, then claim_type name).
    contradicted_sorted = sorted(
        [s for s in by_type.values()
         if s.contradicted_rate is not None and s.total_claims > 0],
        key=lambda s: (-(s.contradicted_rate or 0.0), -s.total_claims, s.claim_type),
    )
    repaired_sorted = sorted(
        [s for s in by_type.values()
         if s.contradiction_reduction is not None and (s.pre_total or 0) > 0],
        key=lambda s: (-(s.contradiction_reduction or 0.0), -(s.pre_total or 0), s.claim_type),
    )

    return ClaimTypeSummary(
        n_turns=n_turns,
        n_pre_available=n_pre_available,
        total_claims=sum(b.total for b in overall.values()),
        distinct_claim_types=len(by_type),
        by_claim_type=by_type,
        most_contradicted_types=[s.claim_type for s in contradicted_sorted[:10]],
        most_repaired_types=[s.claim_type for s in repaired_sorted[:10]],
    )
