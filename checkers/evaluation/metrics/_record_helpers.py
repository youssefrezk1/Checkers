# checkers/evaluation/metrics/_record_helpers.py
#
# Internal helpers shared by per-claim-type and per-source aggregators.
# Both modules need the same extract → verify pipeline applied to the
# pre-refinement reasoning (when present) and to the final reasoning.
#
# Keeping this in one place guarantees the two aggregators see identical
# claim lists for the same record — important because the per-source and
# per-claim-type breakdowns are routinely cross-tabulated downstream.
#
# Deterministic; no LLM calls; no runtime imports from agents/nodes/graph.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from checkers.evaluation.claim_extractor import ClaimRecord, extract_claims
from checkers.evaluation.claim_verifier import verify_claims
from checkers.evaluation.unified_verifier import verify_all


def extract_seeds(record: Dict[str, Any]) -> List[str]:
    diag = record.get("ranker_diagnostics") or {}
    seeds = diag.get("reasoning_seeds") or []
    return [s for s in seeds if isinstance(s, str)]


def extract_facts(record: Dict[str, Any]) -> Dict[str, Any]:
    f = record.get("chosen_move_facts")
    return dict(f) if isinstance(f, dict) else {}


def extract_verifier_context(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    diag = record.get("ranker_diagnostics") or {}
    nb = diag.get("next_best_minimax_score")
    if isinstance(nb, (int, float)):
        return {"next_best_minimax_score": nb}
    return None


def post_reasoning(record: Dict[str, Any]) -> str:
    val = record.get("last_move_reasoning")
    return val if isinstance(val, str) else ""


def pre_reasoning(record: Dict[str, Any]) -> Optional[str]:
    diag = record.get("ranker_diagnostics") or {}
    val = diag.get("raw_llm_reasoning_pre_refinement")
    if isinstance(val, str) and val.strip():
        return val
    return None


def claims_for_text(
    text: str,
    seeds: List[str],
    facts: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> List[ClaimRecord]:
    """
    Verify a reasoning string under the UNIFIED verifier (E.1).

    Wraps extract_claims + verify_claims + numeric (E.3) + schema-leak (E.4)
    under a single deterministic call so the metric layer always agrees
    with the runtime refinement loop on what counts as a contradiction.
    Returns [] for empty/non-string text.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return verify_all(text, reasoning_seeds=seeds, facts=facts or {}, context=context)


def pre_post_claims(
    record: Dict[str, Any],
) -> Tuple[Optional[List[ClaimRecord]], List[ClaimRecord]]:
    """
    Returns (pre_claims, post_claims).

    pre_claims is None when raw_llm_reasoning_pre_refinement is absent.
    post_claims is always a (possibly empty) list.
    """
    seeds   = extract_seeds(record)
    facts   = extract_facts(record)
    context = extract_verifier_context(record)

    pre_text = pre_reasoning(record)
    pre_claims: Optional[List[ClaimRecord]] = (
        claims_for_text(pre_text, seeds, facts, context)
        if pre_text is not None
        else None
    )
    post_claims = claims_for_text(post_reasoning(record), seeds, facts, context)
    return pre_claims, post_claims
