# checkers/ontology/semantic_ontology.py
#
# Strict semantic ontology for reasoning faithfulness — NEUTRAL location.
#
# PURPOSE
# -------
# Defines and separates concepts that are commonly conflated in LLM-generated
# checkers reasoning, particularly the geometric vs. tactical center distinction.
# Runtime AND evaluation modules both depend on these constants; placing them
# here keeps the dependency direction strictly layered:
#
#     checkers.ontology  ← imported by checkers.agents (runtime)
#     checkers.ontology  ← imported by checkers.evaluation
#
# Neither side imports from the other.  This module imports nothing project-
# specific itself.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from checkers.{agents,nodes,graph,state,engine,search,evaluation}.
# - No side effects.  Pure constants only.
# - All string values are lowercase for consistent case-folding comparisons.

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


# ---------------------------------------------------------------------------
# 1. SemanticConceptType
# ---------------------------------------------------------------------------

class SemanticConceptType(str, Enum):
    """
    Canonical concept types used to classify reasoning claims.

    These are distinct; an LLM claim must be mapped to exactly one type.
    Conflating them is the primary source of semantic false positives.

    GEOMETRIC_CENTER
        A piece destination lies in columns {2, 3, 4, 5} of an 8x8 board.
        This is a pure geometric fact about the move path.
        It does NOT imply the piece exerts tactical control over the center.
        Grounding: destination column value (path[-1][1]).
        Allowed claim scope: "lands in center columns", "geometric center position".

    TACTICAL_CENTER_CONTROL
        The engine evaluator has computed center_control=True for this move.
        This means the piece actively exerts evaluator-defined tactical influence
        from its landing square (e.g., attacks or restricts key central squares).
        Grounding: facts["center_control"] == True.
        Allowed claim scope: "center_control=true", "controls the center",
                             "central control", "improves influence over central lanes".

    CENTRAL_PRESENCE
        A vague phrase conflating GEOMETRIC_CENTER and TACTICAL_CENTER_CONTROL.
        Examples: "central board presence", "central influence".
        ALWAYS DISALLOWED as a standalone claim — it is ontologically ambiguous.

    MOBILITY_ADVANTAGE
        A net change in the number of available moves (ours or opponent's).
        Grounding: facts["our_mobility_after"] vs facts["our_mobility_before"],
                   or opponent equivalents.

    STRUCTURAL_SAFETY
        Boolean structural properties of the landing square:
          - leaves_piece_isolated (connectivity)
          - weakens_king_row (back-rank integrity)

    GENERIC_FILLER
        Strategic-sounding phrases with no symbolic fact grounding.
        ALWAYS DISALLOWED regardless of position context.
    """

    GEOMETRIC_CENTER        = "geometric_center"
    TACTICAL_CENTER_CONTROL = "tactical_center_control"
    CENTRAL_PRESENCE        = "central_presence"
    MOBILITY_ADVANTAGE      = "mobility_advantage"
    STRUCTURAL_SAFETY       = "structural_safety"
    GENERIC_FILLER          = "generic_filler"


# ---------------------------------------------------------------------------
# 2. Grounding requirements per concept
# ---------------------------------------------------------------------------

CONCEPT_GROUNDING: dict[SemanticConceptType, dict[str, object]] = {
    SemanticConceptType.GEOMETRIC_CENTER: {
        # No fact key — derived from path geometry alone.
    },
    SemanticConceptType.TACTICAL_CENTER_CONTROL: {
        "center_control": True,
    },
    SemanticConceptType.CENTRAL_PRESENCE: {
        # FORBIDDEN — no grounding can justify this conflated phrase.
    },
    SemanticConceptType.MOBILITY_ADVANTAGE: {
        "our_mobility_before": None,
        "our_mobility_after":  None,
    },
    SemanticConceptType.STRUCTURAL_SAFETY: {
        "leaves_piece_isolated": None,
        "weakens_king_row": None,
    },
    SemanticConceptType.GENERIC_FILLER: {
        # FORBIDDEN — no symbolic fact can ground a generic filler phrase.
    },
}


# ---------------------------------------------------------------------------
# 3. Forbidden conflation phrases
# ---------------------------------------------------------------------------

FORBIDDEN_CONFLATION_PHRASES: FrozenSet[str] = frozenset({
    "central board presence",   # geometric ∩ tactical conflation
    "central influence",        # same conflation, vaguer
})


# ---------------------------------------------------------------------------
# 4. Generic filler phrases
# ---------------------------------------------------------------------------

GENERIC_FILLER_PHRASES: FrozenSet[str] = frozenset({
    "improves activity",        # vague — which fact supports this?
    "piece activity",           # sub-phrase of "improves piece activity"
    "more active position",     # directional without numeric grounding
    "maintains pressure",       # strategic claim — no fact field exists
    "positional adjustment",
})


# ---------------------------------------------------------------------------
# 5. Geometric center columns
# ---------------------------------------------------------------------------

GEOMETRIC_CENTER_COLUMNS: FrozenSet[int] = frozenset({2, 3, 4, 5})
