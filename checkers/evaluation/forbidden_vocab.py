# checkers/evaluation/forbidden_vocab.py
#
# Shared registry of forbidden vocabulary, consulted by BOTH the runtime
# refinement checker (ranker_agent._check_reasoning_truthfulness) and the
# evaluator metric layer (unified_verifier).  Centralising the lists is
# what makes the E.1 invariant — runtime ↔ evaluator agreement on
# instruction-inconsistency contradictions — physically achievable: one
# source of truth, two consumers.
#
# Two tiers:
#
#   ABSOLUTE  : phrases that are NEVER allowed in well-grounded reasoning.
#               Their presence is always treated as an instruction violation
#               (INSTRUCTION_INCONSISTENCY hallucination type).
#
#   CONTEXT   : phrases that are forbidden UNLESS the seed list introduces
#               them verbatim.  The runtime checker and unified verifier
#               both consult the per-call seeds when applying these.
#
# The runtime ontology-merge code in ranker_agent._merge_ontology_phrases
# extends these lists in-place at module load time; downstream readers see
# the merged result automatically.
#
# NEVER imported by checkers.engine / checkers.nodes / checkers.graph.
# Plain Python lists; no side effects.

from __future__ import annotations

ABSOLUTE_FORBIDDEN_VOCAB: list[str] = [
    # Conversion / endgame jargon (invented)
    "conversion potential",
    "winning conversion",
    "trade conversion",
    "conversion score",
    "quiet_move_role",
    "winning_conversion_score",
    # King / escape concepts (not in any seed)
    "escape squares",
    "escape routes",
    "king escape",
    "king distance",
    "king_activity_score",
    # Diagonal invented concepts
    "diagonal pressure",
    "diagonal risks",
    "long diagonal",
    # Invented strategic framing
    "strategic goal",
    "positional adjustment",
    "real trap",
    # Internal metric names (LLM pattern-matches from pipeline logs)
    "counterplay_score",
    "creates_real_trap",         # Python field name — schema leak in reasoning text
    "restriction_score",         # Python field name — schema leak in reasoning text
    "coordination score",
    "activity score",
    "king activity score",
    "quiet move role",
    # Material accounting terms not in any seed
    "regulars_captured",
    # Vague positional framing — unsupported positional characterisations
    "structural restriction",
    "positional step",
    "neutral positional",
    # Generic filler — no symbolic fact can ground these phrases
    "improves activity",         # sub-phrase of "improves piece activity"
    "piece activity",            # broader "piece activity" variants
    "more active position",      # directional claim without numeric grounding
    "maintains pressure",        # strategic claim — no fact field exists
    # BUG-9 hollow strategic filler — evaluative labels with no measurable grounding
    "tangible positional advantage",
    "strong positional edge",
]

# Terms that are only forbidden when NOT present verbatim in the seed list.
# These are allowed when the seed explicitly introduces them.
CONTEXT_FORBIDDEN_VOCAB: list[str] = [
    "conversion",
    "traps",
    "no kings lost",
    "piece count unchanged",
    "no vulnerabilities",
    # Ontology conflation — geometric ∩ tactical center.  No seed emits this
    # phrase; any occurrence in LLM output is an unsupported semantic conflation.
    "central influence",
    "central board presence",
]
