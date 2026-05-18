# checkers/evaluation/semantic_ontology.py
#
# DEPRECATED SHIM — re-exports from the neutral location.
#
# The canonical home of the semantic ontology constants is
# `checkers.ontology.semantic_ontology`.  This module exists only for
# backwards compatibility so existing evaluation-side imports continue to
# work without changes.
#
# Runtime code (checkers.agents.*, checkers.nodes.*, etc.) MUST import from
# `checkers.ontology.semantic_ontology` directly to preserve strict layering:
#
#     ✓ runtime → checkers.ontology
#     ✓ evaluation → checkers.ontology
#     ✗ runtime → checkers.evaluation   (forbidden direction)
#
# New code should import from the neutral path.  This shim re-exports the
# same symbols and adds nothing else.

from __future__ import annotations

from checkers.ontology.semantic_ontology import (  # noqa: F401
    SemanticConceptType,
    CONCEPT_GROUNDING,
    FORBIDDEN_CONFLATION_PHRASES,
    GENERIC_FILLER_PHRASES,
    GEOMETRIC_CENTER_COLUMNS,
)

__all__ = [
    "SemanticConceptType",
    "CONCEPT_GROUNDING",
    "FORBIDDEN_CONFLATION_PHRASES",
    "GENERIC_FILLER_PHRASES",
    "GEOMETRIC_CENTER_COLUMNS",
]
