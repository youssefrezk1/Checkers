# checkers/evaluation/claim_taxonomy.py
#
# Phase 4.0: verifiable-claim taxonomy for extractor recall audit.
#
# PURPOSE
# -------
# Defines a strict taxonomy that classifies each existing claim type by its
# verifiability category, required fact fields, entity and direction
# requirements, polarity sensitivity, and whether a symbolic verifier exists.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls. No I/O. Pure taxonomy only.
# - Covers existing claim types only (as defined in claim_extractor._PHRASE_TABLE).
# - Does NOT alter claim extraction or verification behavior.
# - All outputs are JSON-serializable (enum values are lowercase strings).
#
# USAGE
# -----
# This module is imported only by evaluation scripts and tests.
# It must NEVER be imported by the runtime pipeline.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Taxonomy categories
# ---------------------------------------------------------------------------

class TaxonomyCategory(str, Enum):
    """
    Top-level verifiability category for a claim type.

    VERIFIABLE
        The claim maps to one or more deterministic fact fields and has a
        symbolic rule in claim_verifier._VERIFICATION_RULES that can return
        SUPPORTED or CONTRADICTED.  required_fact_fields is non-empty.

    AMBIGUOUS_CONTEXT_REQUIRED
        A fact field exists for this claim type but the claim wording introduces
        interpretation not fully captured by the field value alone.  A verifier
        may exist but cannot always establish a hard contradiction.

    NON_VERIFIABLE_VAGUE
        No symbolic fact field exists for this claim type.  The claim is
        qualitative or strategic and structurally cannot be confirmed or refuted.

    SCHEMA_LEAK
        Detection fires primarily on verbatim internal schema markers echoed by
        the LLM (e.g., raw "field=value" strings in reasoning text) rather than
        on natural-language claims.  No existing claim type currently falls here.

    FORBIDDEN_UNGROUNDED
        The claim uses vocabulary explicitly forbidden by the grounding system
        (e.g., "structural pressure") and is structurally unverifiable.
        Detection indicates a grounding violation, not a reasoning claim.
    """

    VERIFIABLE                 = "verifiable"
    AMBIGUOUS_CONTEXT_REQUIRED = "ambiguous_context_required"
    NON_VERIFIABLE_VAGUE       = "non_verifiable_vague"
    SCHEMA_LEAK                = "schema_leak"
    FORBIDDEN_UNGROUNDED       = "forbidden_ungrounded"


# ---------------------------------------------------------------------------
# ClaimSpec: per-claim-type specification record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimSpec:
    """
    Specification record for one claim type in the taxonomy registry.

    Fields
    ------
    claim_type : str
        Canonical claim identifier matching claim_extractor._PHRASE_TABLE entries.

    category : TaxonomyCategory
        Verifiability category of this claim type.

    required_fact_fields : tuple[str, ...]
        Fact dict keys required for symbolic verification.
        Empty tuple for NON_VERIFIABLE_VAGUE and FORBIDDEN_UNGROUNDED types.

    entity_requirement : str
        Whose piece or state the claim is about.
        One of: "our", "opponent", "none".

    direction_requirement : str
        For quantitative claims, the required direction of change.
        One of: "increase", "decrease", "none".

    polarity_sensitive : bool
        True when the claim can be negated in natural language to mean the
        opposite (e.g., "no material gain" negates gains_material).
        The extractor applies suppression or redirection logic for these types.

    verifier_exists : bool
        True when claim_verifier._VERIFICATION_RULES contains an entry for
        this claim type, including always-VAGUE rules for unverifiable types.
    """

    claim_type: str
    category: TaxonomyCategory
    required_fact_fields: tuple
    entity_requirement: str
    direction_requirement: str
    polarity_sensitive: bool
    verifier_exists: bool


# ---------------------------------------------------------------------------
# Claim registry
# ---------------------------------------------------------------------------
# Entries appear in the same order as claim_extractor._PHRASE_TABLE.
# Each entry corresponds exactly to one row in the phrase table.

_CLAIM_REGISTRY: dict[str, ClaimSpec] = {spec.claim_type: spec for spec in [

    # ── Safety / recapture ────────────────────────────────────────────────

    ClaimSpec(
        claim_type="avoids_recapture",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("opponent_can_recapture",),
        entity_requirement="opponent",
        direction_requirement="none",
        polarity_sensitive=True,   # "no recapture risk" negates the claim
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="can_be_recaptured",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("opponent_can_recapture",),
        entity_requirement="opponent",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),

    # ── Material ──────────────────────────────────────────────────────────

    ClaimSpec(
        claim_type="gains_material",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("net_gain", "captures_count"),
        entity_requirement="our",
        direction_requirement="increase",
        polarity_sensitive=True,   # "no material gain" suppressed by extractor
        verifier_exists=True,
    ),

    # ── Promotion ─────────────────────────────────────────────────────────

    ClaimSpec(
        claim_type="promotes_to_king",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("results_in_king",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="near_promotion",
        category=TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
        required_fact_fields=("near_promotion",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="opponent_near_promotion",
        category=TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
        required_fact_fields=("opponent_near_promotion",),
        entity_requirement="opponent",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),

    # ── Tactical threats ──────────────────────────────────────────────────

    ClaimSpec(
        claim_type="creates_immediate_threat",
        category=TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
        required_fact_fields=("creates_immediate_threat", "shot_sequence_available"),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="shot_sequence_or_multi_jump",
        category=TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
        required_fact_fields=("shot_sequence_available",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,   # Phase 4.1: _verify_shot_sequence_or_multi_jump added
    ),
    ClaimSpec(
        claim_type="blocks_landing_square",
        category=TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
        required_fact_fields=("blocks_opponent_landing",),
        entity_requirement="opponent",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,   # Phase 4.1: _verify_blocks_landing_square added
    ),
    ClaimSpec(
        claim_type="forced_opponent_jump",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("forced_opponent_jump_reply",),
        entity_requirement="opponent",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,   # Phase 4.1: _verify_forced_opponent_jump added
    ),

    # ── Structure / isolation ─────────────────────────────────────────────

    ClaimSpec(
        claim_type="piece_isolated",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("leaves_piece_isolated",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="piece_connected",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("leaves_piece_isolated",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),

    # ── Back-row ──────────────────────────────────────────────────────────

    ClaimSpec(
        claim_type="weakens_king_row",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("weakens_king_row",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),

    # ── Center control ────────────────────────────────────────────────────

    ClaimSpec(
        claim_type="center_control",
        category=TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
        required_fact_fields=("center_control",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),

    # ── Mobility ──────────────────────────────────────────────────────────

    ClaimSpec(
        # entity_requirement is "opponent" for the primary use case (reduces opponent
        # available replies via mobility_reduction).  The verifier also handles a
        # secondary "our" path: "reduces our mobility by N" compares our_mobility_after
        # vs our_mobility_before.  Both paths are covered by the same verifier rule;
        # splitting into two claim types is unnecessary at this stage.
        claim_type="mobility_decrease",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("mobility_reduction", "our_mobility_before", "our_mobility_after"),
        entity_requirement="opponent",  # primary; verifier also handles self-mobility "our" path
        direction_requirement="decrease",
        polarity_sensitive=True,   # direction-sensitive; extractor suppresses reversed phrasing
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="mobility_increase",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("our_mobility_before", "our_mobility_after"),
        entity_requirement="our",
        direction_requirement="increase",
        polarity_sensitive=True,   # "reduces our mobility by" redirected to mobility_decrease
        verifier_exists=True,
    ),

    # ── Adversity / losing-position context (Phase 6) ─────────────────────
    # Each spec mirrors one seed emitted by
    # ranker_agent._build_adversity_context_seeds.  All four are symbolically
    # grounded: a verifier exists in claim_verifier._VERIFICATION_RULES.

    ClaimSpec(
        claim_type="material_deficit",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("material_advantage",),
        entity_requirement="our",
        direction_requirement="decrease",
        polarity_sensitive=False,
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="threat_reduction",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=(
            "our_pieces_threatened_before",
            "our_pieces_threatened_after",
        ),
        entity_requirement="our",
        direction_requirement="decrease",
        polarity_sensitive=False,
        verifier_exists=True,
    ),
    ClaimSpec(
        # PARTIALLY verifiable — comparison vs other candidates is not in a
        # single move's facts dict.  Verifier accepts SUPPORTED only when
        # minimax_score is present; never returns CONTRADICTED.
        claim_type="score_gap_advantage",
        category=TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
        required_fact_fields=("minimax_score",),
        entity_requirement="our",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),
    ClaimSpec(
        claim_type="mobility_asymmetry",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=(
            "opponent_mobility_before",
            "our_mobility_before",
        ),
        entity_requirement="none",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),

    # ── Minimax confirmation ──────────────────────────────────────────────

    ClaimSpec(
        claim_type="minimax_confirmation",
        category=TaxonomyCategory.VERIFIABLE,
        required_fact_fields=("minimax_score",),
        entity_requirement="none",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,
    ),

    # ── Unverifiable strategic claims ─────────────────────────────────────
    # "positional_pressure" is FORBIDDEN_UNGROUNDED because its phrase list
    # includes "structural pressure", which is in the explicit forbidden
    # vocabulary enforced by _check_reasoning_truthfulness.

    ClaimSpec(
        claim_type="positional_pressure",
        category=TaxonomyCategory.FORBIDDEN_UNGROUNDED,
        required_fact_fields=(),
        entity_requirement="none",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,   # always-VAGUE rule exists in _VERIFICATION_RULES
    ),
    ClaimSpec(
        claim_type="strategic_initiative",
        category=TaxonomyCategory.NON_VERIFIABLE_VAGUE,
        required_fact_fields=(),
        entity_requirement="none",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,   # always-VAGUE rule exists in _VERIFICATION_RULES
    ),
    ClaimSpec(
        claim_type="long_term_compensation",
        category=TaxonomyCategory.NON_VERIFIABLE_VAGUE,
        required_fact_fields=(),
        entity_requirement="none",
        direction_requirement="none",
        polarity_sensitive=False,
        verifier_exists=True,   # always-VAGUE rule exists in _VERIFICATION_RULES
    ),
]}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_claim_spec(claim_type: str) -> Optional[ClaimSpec]:
    """Return the ClaimSpec for the given claim_type, or None if not registered."""
    return _CLAIM_REGISTRY.get(claim_type)


def is_verifiable_claim_type(claim_type: str) -> bool:
    """
    Return True if the claim type is categorized as VERIFIABLE.
    Returns False for unknown claim types and all other categories.
    """
    spec = _CLAIM_REGISTRY.get(claim_type)
    return spec is not None and spec.category == TaxonomyCategory.VERIFIABLE


def required_fields_for_claim(claim_type: str) -> tuple:
    """
    Return the tuple of required fact field names for the given claim type.
    Returns an empty tuple for unknown claim types and unverifiable types.
    """
    spec = _CLAIM_REGISTRY.get(claim_type)
    return spec.required_fact_fields if spec is not None else ()


def claim_type_has_verifier(claim_type: str) -> bool:
    """
    Return True when claim_verifier._VERIFICATION_RULES contains a rule
    for this claim type (including always-VAGUE rules).
    Returns False for unknown claim types.
    """
    spec = _CLAIM_REGISTRY.get(claim_type)
    return spec is not None and spec.verifier_exists
