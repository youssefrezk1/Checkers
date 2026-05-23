# checkers/evaluation/metrics/
#
# First evaluator layer for the proposal-authoritative pipeline.
#
# Modules are partitioned by metric family so they can be extended
# independently:
#
#   factuality  → pre_post_repair.py   (claim verification before vs. after
#                                       the truthfulness refinement loop)
#   grounding   → zero_claim.py        (sentence-level claim coverage and
#                                       filler detection)
#   diversity   → self_bleu.py         (corpus-level n-gram overlap; flags
#                                       template collapse / paraphrase reuse)
#
# Every module here is DETERMINISTIC, makes NO LLM calls, and reads only
# data already produced by the runtime pipeline. They never import from
# checkers.agents, checkers.nodes, checkers.engine, or checkers.graph.
"""Metrics package for reasoning-faithfulness evaluation."""

from checkers.evaluation.metrics.pre_post_repair import (
    PrePostRepairTurn,
    PrePostRepairSummary,
    evaluate_pre_post_repair,
    aggregate_pre_post_repair,
)
from checkers.evaluation.metrics.zero_claim import (
    SentenceCoverage,
    ZeroClaimTurn,
    ZeroClaimSummary,
    evaluate_zero_claim,
    aggregate_zero_claim,
)
from checkers.evaluation.metrics.self_bleu import (
    SelfBleuSummary,
    compute_self_bleu,
)
from checkers.evaluation.metrics.by_claim_type import (
    ClaimTypeStats,
    ClaimTypeSummary,
    aggregate_by_claim_type,
)
from checkers.evaluation.metrics.by_source import (
    SourceStats,
    SeedVsUnsupportedDelta,
    ClaimSourceSummary,
    aggregate_by_source,
)

__all__ = [
    "PrePostRepairTurn",
    "PrePostRepairSummary",
    "evaluate_pre_post_repair",
    "aggregate_pre_post_repair",
    "SentenceCoverage",
    "ZeroClaimTurn",
    "ZeroClaimSummary",
    "evaluate_zero_claim",
    "aggregate_zero_claim",
    "SelfBleuSummary",
    "compute_self_bleu",
    "ClaimTypeStats",
    "ClaimTypeSummary",
    "aggregate_by_claim_type",
    "SourceStats",
    "SeedVsUnsupportedDelta",
    "ClaimSourceSummary",
    "aggregate_by_source",
]
