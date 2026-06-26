# checkers/ontology/forbidden_vocab.py
#
# Neutral canonical registry of forbidden vocabulary.
# Consumed by BOTH the runtime agent (checkers.agents.explainer_agent) and the
# evaluator layer (checkers.evaluation.*) — placing it in checkers.ontology
# satisfies the Fix-5 isolation invariant: the runtime must not import from
# checkers.evaluation at module load time.
#
# checkers.evaluation.forbidden_vocab re-exports from here so all existing
# evaluation-layer imports continue to work unchanged.
#
# Two tiers:
#
#   ABSOLUTE  : phrases never allowed in well-grounded reasoning.
#   CONTEXT   : forbidden unless introduced verbatim in the seed list.

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
    "creates_real_trap",
    "restriction_score",
    "coordination score",
    "activity score",
    "king activity score",
    "quiet move role",
    # Material accounting terms not in any seed
    "regulars_captured",
    # Vague positional framing
    "structural restriction",
    "positional step",
    "neutral positional",
    # Generic filler
    "improves activity",
    "piece activity",
    "more active position",
    "maintains pressure",
    # BUG-9 hollow strategic filler
    "tangible positional advantage",
    "strong positional edge",
]

# Terms forbidden unless the seed list introduces them verbatim.
CONTEXT_FORBIDDEN_VOCAB: list[str] = [
    "conversion",
    "traps",
    "no kings lost",
    "piece count unchanged",
    "no vulnerabilities",
    "central influence",
    "central board presence",
]
