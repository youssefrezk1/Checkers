# checkers/evaluation/claim_recall_audit.py
#
# Phase 4.0: lightweight audit helper for claim extractor recall.
#
# PURPOSE
# -------
# Inspects a list of extracted ClaimRecord objects against the taxonomy
# registry to report which verifiable claim types were detected, which
# verifier-backed types are absent from the extraction, and which
# ambiguous or non-verifiable types appeared.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls. No I/O. Pure read-only inspection.
# - Does NOT alter claim statuses or any field of any ClaimRecord.
# - Does NOT filter or modify the extracted_claims list.
# - Deterministic: same inputs always produce the same report.
#
# USAGE
# -----
# This module is imported only by evaluation scripts and tests.
# It must NEVER be imported by the runtime pipeline.

from __future__ import annotations

from typing import Any

from checkers.evaluation.claim_taxonomy import (
    _CLAIM_REGISTRY,
    TaxonomyCategory,
    is_verifiable_claim_type,
    claim_type_has_verifier,
)

# Categories that are NOT cleanly verifiable by symbolic facts alone.
_NON_VERIFIABLE_CATEGORIES: frozenset[TaxonomyCategory] = frozenset({
    TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
    TaxonomyCategory.NON_VERIFIABLE_VAGUE,
    TaxonomyCategory.FORBIDDEN_UNGROUNDED,
    TaxonomyCategory.SCHEMA_LEAK,
})


def audit_claim_recall(
    extracted_claims: list[Any],
    seeds: list[str] | None = None,
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Inspect extracted claims and return a recall audit report.

    Parameters
    ----------
    extracted_claims : list[ClaimRecord]
        As returned by claim_extractor.extract_claims().  May be empty.
        This function never mutates the list or any record.
    seeds : list[str] or None
        Reasoning seeds (provided for context; not currently inspected).
    facts : dict or None
        Move facts dict (provided for context; not currently inspected).

    Returns
    -------
    dict with four keys:

    extracted_claim_types : list[str]
        Claim types present in extracted_claims, in appearance order.
        Unknown types (not in the registry) are included here.

    verifiable_claim_types_present : list[str]
        Subset of extracted_claim_types whose taxonomy category is VERIFIABLE.
        Order matches extracted_claim_types.

    missing_verifier_types : list[str]
        All registered claim types where verifier_exists=True that are absent
        from extracted_claims.  Ordered by registry insertion order.
        Useful for identifying recall gaps: these are types the extractor
        could have detected (a verifier exists) but did not.

    ambiguous_or_nonverifiable_types : list[str]
        Extracted claim types whose category is AMBIGUOUS_CONTEXT_REQUIRED,
        NON_VERIFIABLE_VAGUE, FORBIDDEN_UNGROUNDED, or SCHEMA_LEAK.
        Order matches extracted_claim_types.  Unknown types are excluded.

    Guarantees
    ----------
    - extracted_claims is never mutated.
    - No claim_status field is read or modified.
    - No claim_verifiability or other ClaimRecord field is modified.
    - The function never raises for empty inputs (returns empty lists).
    """
    extracted_types: list[str] = [c.claim_type for c in extracted_claims]

    verifiable_present: list[str] = [
        ct for ct in extracted_types if is_verifiable_claim_type(ct)
    ]

    extracted_set: set[str] = set(extracted_types)
    missing_verifier: list[str] = [
        ct
        for ct, spec in _CLAIM_REGISTRY.items()
        if spec.verifier_exists and ct not in extracted_set
    ]

    ambiguous_or_nonverifiable: list[str] = [
        ct for ct in extracted_types
        if _CLAIM_REGISTRY.get(ct) is not None
        and _CLAIM_REGISTRY[ct].category in _NON_VERIFIABLE_CATEGORIES
    ]

    return {
        "extracted_claim_types": extracted_types,
        "verifiable_claim_types_present": verifiable_present,
        "missing_verifier_types": missing_verifier,
        "ambiguous_or_nonverifiable_types": ambiguous_or_nonverifiable,
    }
