# checkers/evaluation/metrics/by_source.py
#
# Grounding-source breakdown.
#
# Each ClaimRecord carries a `source` attribute set by the extractor:
#   "seed"               — phrase matched AND a supporting seed exists
#   "fact_phrase"        — phrase matched AND a supporting fact exists
#                          (no explicit seed)
#   "unsupported_phrase" — phrase matched but no seed or fact backs it
#   "unknown"            — fallback; should not normally occur
#
# Research question driving this module:
#   "Do symbolic grounding seeds reduce hallucination and contradiction?"
#
# For each source value the aggregator reports:
#   total
#   supported_rate
#   contradicted_rate
#   unsupported_rate
#   vague_rate
#   hallucination_rate           — share of claims with hallucination_type != None
#   hallucination_type_breakdown — count per HallucinationType for that source
#   pre/post deltas (optional)   — only when raw_llm_reasoning_pre_refinement
#                                  exists for the turn
#
# Also reports a top-level `seed_vs_unsupported` delta object that directly
# compares "seed" against "unsupported_phrase" — the headline answer to the
# research question.
#
# Deterministic.  No LLM calls.  No runtime pipeline imports.

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from checkers.evaluation.claim_extractor import ClaimRecord
from checkers.evaluation.reasoning_taxonomy import ClaimStatus
from checkers.evaluation.metrics._record_helpers import pre_post_claims


SOURCE_VALUES = ("seed", "fact_phrase", "unsupported_phrase", "unknown")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _Bucket:
    total:        int = 0
    supported:    int = 0
    contradicted: int = 0
    unsupported:  int = 0
    vague:        int = 0
    hallucinated: int = 0
    halluc_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class SourceStats:
    source:                 str
    total:                  int
    supported_rate:         Optional[float]
    contradicted_rate:      Optional[float]
    unsupported_rate:       Optional[float]
    vague_rate:             Optional[float]
    hallucination_rate:     Optional[float]
    hallucination_type_breakdown: Dict[str, int] = field(default_factory=dict)
    # Pre/post deltas (None when no pre-refinement snapshot in the batch).
    pre_total:              Optional[int]   = None
    post_total:             Optional[int]   = None
    pre_contradicted_rate:  Optional[float] = None
    post_contradicted_rate: Optional[float] = None
    pre_hallucination_rate: Optional[float] = None
    post_hallucination_rate:Optional[float] = None
    contradiction_reduction: Optional[float] = None
    hallucination_reduction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SeedVsUnsupportedDelta:
    """Headline comparison between seeded and unsupported-phrase claims."""
    n_seed:              int
    n_unsupported:       int
    contradicted_delta:  Optional[float]   # seed - unsupported (negative = seeds help)
    hallucination_delta: Optional[float]
    supported_delta:     Optional[float]   # seed - unsupported (positive = seeds help)
    vague_delta:         Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimSourceSummary:
    n_turns:          int
    n_pre_available:  int
    total_claims:     int
    by_source:        Dict[str, SourceStats] = field(default_factory=dict)
    seed_vs_unsupported: Optional[SeedVsUnsupportedDelta] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["by_source"] = {k: v.to_dict() if hasattr(v, "to_dict") else v
                          for k, v in self.by_source.items()}
        if self.seed_vs_unsupported is not None:
            d["seed_vs_unsupported"] = self.seed_vs_unsupported.to_dict()
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _source_key(c: ClaimRecord) -> str:
    return c.source if c.source in SOURCE_VALUES else "unknown"


def _bump(bucket: _Bucket, record: ClaimRecord) -> None:
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
    if record.hallucination_type is not None:
        bucket.hallucinated += 1
        # hallucination_type is a (str, Enum); store its string value.
        h_val = getattr(record.hallucination_type, "value", str(record.hallucination_type))
        bucket.halluc_breakdown[h_val] = bucket.halluc_breakdown.get(h_val, 0) + 1


def _finalize(
    source: str,
    overall: _Bucket,
    pre: _Bucket,
    post_matched: _Bucket,
    pre_available_any: bool,
) -> SourceStats:
    pre_total = pre.total if pre_available_any else None
    post_total = post_matched.total if pre_available_any else None

    pre_contra_rate    = _safe_div(pre.contradicted, pre.total) if pre_available_any else None
    post_contra_rate   = _safe_div(post_matched.contradicted, post_matched.total) if pre_available_any else None
    pre_halluc_rate    = _safe_div(pre.hallucinated, pre.total) if pre_available_any else None
    post_halluc_rate   = _safe_div(post_matched.hallucinated, post_matched.total) if pre_available_any else None

    contradiction_reduction = (
        pre_contra_rate - post_contra_rate
        if pre_contra_rate is not None and post_contra_rate is not None
        else None
    )
    hallucination_reduction = (
        pre_halluc_rate - post_halluc_rate
        if pre_halluc_rate is not None and post_halluc_rate is not None
        else None
    )

    return SourceStats(
        source=source,
        total=overall.total,
        supported_rate=_safe_div(overall.supported, overall.total),
        contradicted_rate=_safe_div(overall.contradicted, overall.total),
        unsupported_rate=_safe_div(overall.unsupported, overall.total),
        vague_rate=_safe_div(overall.vague, overall.total),
        hallucination_rate=_safe_div(overall.hallucinated, overall.total),
        hallucination_type_breakdown=dict(sorted(overall.halluc_breakdown.items())),
        pre_total=pre_total,
        post_total=post_total,
        pre_contradicted_rate=pre_contra_rate,
        post_contradicted_rate=post_contra_rate,
        pre_hallucination_rate=pre_halluc_rate,
        post_hallucination_rate=post_halluc_rate,
        contradiction_reduction=contradiction_reduction,
        hallucination_reduction=hallucination_reduction,
    )


def _build_seed_vs_unsupported(by_source: Dict[str, SourceStats]) -> Optional[SeedVsUnsupportedDelta]:
    seed = by_source.get("seed")
    unsup = by_source.get("unsupported_phrase")
    if seed is None or unsup is None or seed.total == 0 or unsup.total == 0:
        # Still emit when at least one side has data so the consumer can see it,
        # but only compute the deltas when both sides exist.
        return SeedVsUnsupportedDelta(
            n_seed=seed.total if seed else 0,
            n_unsupported=unsup.total if unsup else 0,
            contradicted_delta=None,
            hallucination_delta=None,
            supported_delta=None,
            vague_delta=None,
        )

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return a - b

    return SeedVsUnsupportedDelta(
        n_seed=seed.total,
        n_unsupported=unsup.total,
        contradicted_delta=_delta(seed.contradicted_rate, unsup.contradicted_rate),
        hallucination_delta=_delta(seed.hallucination_rate, unsup.hallucination_rate),
        supported_delta=_delta(seed.supported_rate, unsup.supported_rate),
        vague_delta=_delta(seed.vague_rate, unsup.vague_rate),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_by_source(
    records: Iterable[Mapping[str, Any]],
) -> ClaimSourceSummary:
    """
    Aggregate claims by their source attribution across a batch.

    "Source" is set by the extractor and reflects whether the claim was
    explicitly authorised by a seed, supported by a fact value, or matched
    a phrase pattern with no underlying support.

    Returns a ClaimSourceSummary with per-source stats and a headline
    seed-vs-unsupported delta computed from post-repair claims.
    """
    overall: Dict[str, _Bucket] = {k: _Bucket() for k in SOURCE_VALUES}
    pre_buckets: Dict[str, _Bucket] = {k: _Bucket() for k in SOURCE_VALUES}
    post_buckets_matched: Dict[str, _Bucket] = {k: _Bucket() for k in SOURCE_VALUES}

    n_turns = 0
    n_pre_available = 0

    for rec in records:
        n_turns += 1
        pre_claims, post_claims = pre_post_claims(rec)

        for c in post_claims:
            _bump(overall[_source_key(c)], c)

        if pre_claims is None:
            continue
        n_pre_available += 1
        for c in pre_claims:
            _bump(pre_buckets[_source_key(c)], c)
        for c in post_claims:
            _bump(post_buckets_matched[_source_key(c)], c)

    by_source: Dict[str, SourceStats] = {}
    for src in SOURCE_VALUES:
        by_source[src] = _finalize(
            source=src,
            overall=overall[src],
            pre=pre_buckets[src],
            post_matched=post_buckets_matched[src],
            pre_available_any=n_pre_available > 0,
        )

    return ClaimSourceSummary(
        n_turns=n_turns,
        n_pre_available=n_pre_available,
        total_claims=sum(b.total for b in overall.values()),
        by_source=by_source,
        seed_vs_unsupported=_build_seed_vs_unsupported(by_source),
    )
