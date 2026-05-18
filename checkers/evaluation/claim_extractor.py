# checkers/evaluation/claim_extractor.py
#
# Deterministic claim extraction from ranker reasoning text.
#
# PURPOSE
# -------
# Converts a ranker reasoning paragraph into a list of structured ClaimRecord
# objects by matching known phrases against the text and cross-referencing
# with the reasoning seeds that authorised those claims.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls.  All extraction is regex/phrase-matching only.
# - Deterministic: same (reasoning_text, seeds, facts) → same output always.
# - Inputs are never mutated.
# - All outputs are JSON-serialisable via dataclasses.asdict().
#
# USAGE
# -----
# This module is imported only by evaluation scripts and tests.
# It must NEVER be imported by the runtime pipeline.

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from checkers.evaluation.reasoning_taxonomy import (
    ClaimStatus,
    ClaimVerifiability,
    HallucinationType,
    SeedRiskType,
)


# ---------------------------------------------------------------------------
# ClaimRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClaimRecord:
    """
    A single evaluatable claim extracted from a reasoning paragraph.

    Fields
    ------
    claim_type : str
        Canonical claim identifier from the phrase table (e.g. "avoids_recapture",
        "gains_material").

    claim_status : ClaimStatus
        Verification outcome.  At extraction time this is set to SUPPORTED
        (if a matching seed or consistent fact exists) or UNSUPPORTED (if no
        seed/fact backs the claim).  Deeper contradiction checking is deferred
        to a later evaluation step.

    claim_verifiability : ClaimVerifiability
        How objectively this claim can be checked against symbolic facts.

    seed_risk_type : SeedRiskType | None
        Risk classification of the seed that authorised this claim, or None
        if the claim was not seed-authorised.

    hallucination_type : HallucinationType | None
        If the claim is flagged as a potential hallucination at extraction time,
        this records the type.  None for non-hallucinated claims.

    matched_phrase : str | None
        The exact substring in the reasoning text that triggered this claim.

    matched_seed : str | None
        The exact seed string (from reasoning_seeds) that supports this claim,
        or None if no seed matched.

    source : str
        Provenance of the claim:
        - "seed"               — matched a phrase AND a supporting seed exists
        - "fact_phrase"         — matched a phrase AND a supporting fact exists
                                  (but no explicit seed)
        - "unsupported_phrase"  — matched a phrase but no seed or fact backs it
        - "unknown"            — fallback; should not normally occur
    """

    claim_type: str
    claim_status: ClaimStatus
    claim_verifiability: ClaimVerifiability
    seed_risk_type: Optional[SeedRiskType] = None
    hallucination_type: Optional[HallucinationType] = None
    matched_phrase: Optional[str] = None
    matched_seed: Optional[str] = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        d = asdict(self)
        # Enum values are already strings (str, Enum) but asdict returns
        # the raw value; ensure Optional enums become None not enum repr.
        for key in ("claim_status", "claim_verifiability",
                     "seed_risk_type", "hallucination_type"):
            v = d.get(key)
            if v is not None and hasattr(v, "value"):
                d[key] = v.value
        return d


# ---------------------------------------------------------------------------
# Phrase table
# ---------------------------------------------------------------------------
# Each entry maps a canonical claim_type to:
#   - phrases: list of lowercase substrings to search for in reasoning text
#   - verifiability: how objectively checkable
#   - seed_risk: default risk level when the claim originates from a seed
#   - seed_markers: list of substrings to look for in the seed list to confirm
#                   that the claim was authorised by a seed
#   - fact_field: the fact dict key that, if present, can support the claim
#                 (used when no seed matched but a fact value is consistent)
#   - fact_supports: a callable (value) -> bool that returns True when the
#                    fact value is consistent with the claim.  None means
#                    presence of the field alone is sufficient.

@dataclass
class _PhraseEntry:
    """Internal: one row of the phrase table."""
    claim_type: str
    phrases: list[str]
    verifiability: ClaimVerifiability
    seed_risk: SeedRiskType
    seed_markers: list[str]
    fact_field: Optional[str] = None
    fact_supports: Any = None  # Optional[Callable[[Any], bool]]


# The phrase table is intentionally conservative.  Phrases are chosen to
# minimise false positives: they are specific enough that a match almost
# certainly indicates the claim type, and short enough that paraphrases
# will still fire for the most common LLM wordings.
#
# The table is ordered by evaluation priority (safety first, then tactical,
# then structural, then strategic, then unverifiable).

_PHRASE_TABLE: list[_PhraseEntry] = [

    # ── Safety / recapture ────────────────────────────────────────────────
    _PhraseEntry(
        claim_type="avoids_recapture",
        phrases=[
            "avoids recapture",
            "no recapture",
            "cannot recapture",
            "without recapture risk",
            "no recapture risk",
            "safe from recapture",
            "opponent cannot recapture",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "opponent_can_recapture=false",
        ],
        fact_field="opponent_can_recapture",
        fact_supports=lambda v: v is False,
    ),
    _PhraseEntry(
        claim_type="can_be_recaptured",
        phrases=[
            "opponent can recapture",
            "can be recaptured",
            "recapture risk remains",
            "exposed to recapture",
            "vulnerable to recapture",
            "can recapture this piece",
            "opponent_can_recapture=true",
            "tactically exposed",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "opponent_can_recapture=true",
        ],
        fact_field="opponent_can_recapture",
        fact_supports=lambda v: v is True,
    ),

    # ── Material ──────────────────────────────────────────────────────────
    _PhraseEntry(
        claim_type="gains_material",
        phrases=[
            "captures a piece",
            "captures the piece",
            "captures an opponent",
            "captures opponent",
            "gaining a piece",
            "gains material",
            "material gain",
            "gains a piece",
            "wins material",
            "net gain",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "captures_count=",
            "net_gain=",
        ],
        fact_field="captures_count",
        fact_supports=lambda v: isinstance(v, (int, float)) and v > 0,
    ),

    # ── Promotion ─────────────────────────────────────────────────────────
    _PhraseEntry(
        claim_type="promotes_to_king",
        phrases=[
            "promotes to king",
            "promotes a piece",
            "crowns a piece",
            "becomes a king",
            "promoting to king",
            "promotion to king",
            "converts the piece into a king",
            "results_in_king=true",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "results_in_king=true",
        ],
        fact_field="results_in_king",
        fact_supports=lambda v: v is True,
    ),
    _PhraseEntry(
        claim_type="near_promotion",
        phrases=[
            "near promotion",
            "near_promotion=true",
            "promotion threat",
            "future promotion",
            "one step from promotion",
        ],
        verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=[
            "^near_promotion=true",   # anchored: must start with this, so
                                      # "opponent_near_promotion=true" never matches
            "future promotion threat",
        ],
        fact_field="near_promotion",
        fact_supports=lambda v: v is True,
    ),
    # Opponent-entity variant: emitted via entity-context redirect from near_promotion
    # processing when "opponent"/"enemy"/"opposing" precedes the matched phrase.
    # Also matches the literal seed-echo phrase "opponent_near_promotion=true".
    _PhraseEntry(
        claim_type="opponent_near_promotion",
        phrases=[
            "opponent_near_promotion=true",
        ],
        verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=["opponent_near_promotion=true"],
        fact_field="opponent_near_promotion",
        fact_supports=lambda v: v is True,
    ),

    # ── Tactical threats ──────────────────────────────────────────────────
    _PhraseEntry(
        claim_type="creates_immediate_threat",
        phrases=[
            "creates a threat",
            "creates immediate threat",
            "creates an immediate threat",
            "immediate tactical pressure",
            "applies pressure next turn",
            "creates pressure next",
            "threatens opponent",
            "creates tactical threat",
            "puts opponent on the defensive",
            "on the defensive next turn",
            "creates_immediate_threat=true",
        ],
        verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=[
            "creates_immediate_threat=true",
        ],
        fact_field="creates_immediate_threat",
        fact_supports=lambda v: v is True,
    ),
    _PhraseEntry(
        claim_type="shot_sequence_or_multi_jump",
        phrases=[
            "multi-jump sequence",
            "multi-jump",
            "shot sequence",
            "extend the attack",
            "shot_sequence_available=true",
        ],
        verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        seed_risk=SeedRiskType.OVERCLAIM_RISK,
        seed_markers=[
            "shot_sequence_available=true",
        ],
        fact_field="shot_sequence_available",
        fact_supports=lambda v: v is True,
    ),
    _PhraseEntry(
        claim_type="blocks_landing_square",
        phrases=[
            "blocks opponent landing",
            "blocks the opponent from landing",
            "denies the opponent a key landing",
            "denies a key landing",
            "blocks_opponent_landing=true",
            "landing square",
        ],
        verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "blocks_opponent_landing=true",
        ],
        fact_field="blocks_opponent_landing",
        fact_supports=lambda v: v is True,
    ),
    _PhraseEntry(
        claim_type="forced_opponent_jump",
        phrases=[
            "forced opponent jump",
            "forced_opponent_jump_reply=true",
            "constrained to a jump",
            "constrained to a single",         # "constrained to a single jump"
            "reply is constrained",            # "opponent reply is constrained"
            "opponent response is constrained",
            "forced capture reply",
            "limited to a jump",               # "limited to a single jump"
            "opponent is constrained",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "forced_opponent_jump_reply=true",
        ],
        fact_field="forced_opponent_jump_reply",
        fact_supports=lambda v: v is True,
    ),

    # ── Structure / isolation ─────────────────────────────────────────────
    _PhraseEntry(
        claim_type="piece_isolated",
        phrases=[
            "isolates the piece",
            "piece is isolated",
            "leaves_piece_isolated=true",
            "moved piece is isolated",
            "not supported by adjacent",
            "piece is not supported",
            "isolated without adjacent",        # "isolated without adjacent support"
            "isolated position",                # "lands in an isolated position"
            "no adjacent support",              # "no adjacent support"
            "without adjacent support",         # "without adjacent support"
            "landing in an isolated",           # "landing in an isolated position"
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "leaves_piece_isolated=true",
        ],
        fact_field="leaves_piece_isolated",
        fact_supports=lambda v: v is True,
    ),
    _PhraseEntry(
        claim_type="piece_connected",
        phrases=[
            "stays connected",
            "piece connected",
            "maintains connectivity",
            "does not isolate",
            "no isolation",
            "piece coordination",
            "preserves piece coordination",
            "keeping the moved piece connected",
            "leaves_piece_isolated=false",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=[
            "leaves_piece_isolated=false",
        ],
        fact_field="leaves_piece_isolated",
        fact_supports=lambda v: v is False,
    ),

    # ── Back-row / opening structure ──────────────────────────────────────
    _PhraseEntry(
        claim_type="weakens_king_row",
        phrases=[
            "weakens the back row",
            "weakens king row",
            "back-row defense is weakened",
            "back-row weakened",
            "weakens_king_row=true",
            "weakening the back row",
            # Fix 2C: new conditional seed phrases — only fire on positive assertion
            "weakens back-row defensive structure",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "weakens_king_row=true",
            "weakens back-row defensive structure",  # new seed from Fix 2C
        ],
        fact_field="weakens_king_row",
        fact_supports=lambda v: v is True,
    ),

    # ── Center control ────────────────────────────────────────────────────
    _PhraseEntry(
        claim_type="center_control",
        phrases=[
            "controls the center",
            "controls center",
            "central control",
            "occupies the center",
            "center control",
            "center_control=true",
            "central lanes",
            "central board presence",
            "influence over central",
        ],
        verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=[
            "center_control=true",
            "central lanes",
        ],
        fact_field="center_control",
        fact_supports=lambda v: v is True,
    ),

    # ── Mobility ──────────────────────────────────────────────────────────
    _PhraseEntry(
        claim_type="mobility_decrease",
        phrases=[
            "reduces mobility",
            "reducing mobility",
            "reduces opponent mobility",
            "reducing opponent mobility",
            "limits mobility",
            "limiting mobility",
            "restricts mobility",
            "restricts opponent",
            "fewer moves for",
            "cuts opponent moves",
            "restricting available replies",
            # Self-mobility decrease phrasings (our own moves shrink)
            "our mobility decreases",
            "reduces our mobility",
            "our mobility drops",
            "mobility decreases from",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=[
            "opponent_mobility_before=",
            "opponent_mobility_after=",
            "reduces opponent mobility",
            "decreases our mobility",
        ],
        fact_field="mobility_reduction",
        fact_supports=lambda v: isinstance(v, (int, float)) and v > 0,
    ),
    _PhraseEntry(
        claim_type="mobility_increase",
        phrases=[
            "increases our mobility",
            "improves our mobility",
            "more moves available",
            "increases mobility",
            "our mobility by",             # "increases our mobility by 1/2"
            "mobility rises to",           # "mobility rises to 8"
            "our mobility increases",      # "our mobility increases by"
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=[
            "our_mobility_before=",
            "our_mobility_after=",
            "increases our mobility",
        ],
        fact_field="our_mobility_after",
        # Fact support requires comparing our_mobility_after > our_mobility_before,
        # which needs two fields.  Handled specially in _check_fact_support().
        fact_supports=None,
    ),

    # ── Adversity / losing-position context (Phase 6) ─────────────────────
    # These claim types correspond to seeds emitted by
    # _build_adversity_context_seeds in ranker_agent.  Each one maps to a
    # symbolically grounded fact and can be verified deterministically.
    _PhraseEntry(
        claim_type="material_deficit",
        phrases=[
            "behind by",                  # "behind by 2 pieces"
            "material deficit",
            "material_advantage=-",       # raw seed echo
            "down a piece",
            "down two pieces",            # word-form
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "material_advantage=-",
            "behind by",
        ],
        fact_field="material_advantage",
        fact_supports=lambda v: isinstance(v, (int, float)) and v < 0,
    ),
    _PhraseEntry(
        claim_type="threat_reduction",
        phrases=[
            "reduces threatened pieces",
            "fewer threatened pieces",
            "improves immediate safety",
            "removes a threatened piece",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "reduces threatened pieces from",
            "our_pieces_threatened_before=",
            "our_pieces_threatened_after=",
        ],
        # Verification is multi-field; handled by claim_verifier rule, not
        # by extractor fact_supports.  Setting fact_field=None means the
        # extractor records source="unsupported_phrase" when no seed is
        # present — verifier still upgrades status via fact comparison.
        fact_field=None,
        fact_supports=None,
    ),
    _PhraseEntry(
        claim_type="score_gap_advantage",
        phrases=[
            # Compound phrases — unambiguous comparison claims.
            "points better than",
            "next-best option",
            "best alternative",
            # Short trigger — gated by a comparison-sentinel window check in
            # extract_claims (Phase-6 Fix 4) to avoid false matches like
            # "this move scores well" or "minimax score".
            "scores",
        ],
        # PARTIALLY_VERIFIABLE without cross-candidate context.  When the
        # ranker_diagnostics dict exposes next_best_minimax_score, the
        # context-aware verifier upgrades to hard SUPPORTED / CONTRADICTED.
        verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
        seed_risk=SeedRiskType.INTERPRETIVE,
        seed_markers=[
            # Specific markers only — bare "scores" was removed because every
            # seed list with a minimax_score line would otherwise match.
            "points better than",
            "next-best option",
        ],
        fact_field=None,
        fact_supports=None,
    ),
    _PhraseEntry(
        claim_type="mobility_asymmetry",
        phrases=[
            "structural disadvantage in available options",
            "mobility asymmetry",
            "fewer options than the opponent",
            "opponent has more options",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "structural disadvantage in available options",
            "opponent_mobility_before=",   # adversity seed echo
        ],
        fact_field=None,  # multi-field comparison handled by verifier
        fact_supports=None,
    ),

    # ── Minimax confirmation (always last seed) ───────────────────────────
    _PhraseEntry(
        claim_type="minimax_confirmation",
        phrases=[
            "minimax_score=",
            "minimax_score",
            "highest-evaluated option",
            "best available option",
            "least harmful available continuation",
            "least harmful continuation",
            "engine confirms",
            "engine evaluation",
        ],
        verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk=SeedRiskType.STRICT_FACT,
        seed_markers=[
            "minimax_score=",
        ],
        fact_field="minimax_score",
        fact_supports=lambda v: v is not None,
    ),

    # ── Unverifiable strategic claims ─────────────────────────────────────
    # These have no corresponding seed or fact field.  Their detection is
    # valuable precisely because they should not appear in well-grounded
    # reasoning.
    _PhraseEntry(
        claim_type="positional_pressure",
        phrases=[
            "positional pressure",
            "structural pressure",
            "strategic pressure",
            "pressure on the opponent",
            "applying pressure",
        ],
        verifiability=ClaimVerifiability.UNVERIFIABLE,
        seed_risk=SeedRiskType.MISLEADING,
        seed_markers=[],  # never seeded
        fact_field=None,
        fact_supports=None,
    ),
    _PhraseEntry(
        claim_type="strategic_initiative",
        phrases=[
            "strategic initiative",
            "seizes the initiative",
            "takes the initiative",
            "initiative",
        ],
        verifiability=ClaimVerifiability.UNVERIFIABLE,
        seed_risk=SeedRiskType.MISLEADING,
        seed_markers=[],
        fact_field=None,
        fact_supports=None,
    ),
    _PhraseEntry(
        claim_type="long_term_compensation",
        phrases=[
            "long-term compensation",
            "long term compensation",
            "compensates in the long",
            "future compensation",
        ],
        verifiability=ClaimVerifiability.UNVERIFIABLE,
        seed_risk=SeedRiskType.MISLEADING,
        seed_markers=[],
        fact_field=None,
        fact_supports=None,
    ),
]


# Build a quick-lookup index: claim_type → _PhraseEntry
_ENTRY_BY_TYPE: dict[str, _PhraseEntry] = {e.claim_type: e for e in _PHRASE_TABLE}


# ---------------------------------------------------------------------------
# Polarity detection helpers
# ---------------------------------------------------------------------------
# These helpers suppress or redirect claims whose matched phrase appears inside
# a negated or direction-reversed context.  They operate on the lowercased
# full reasoning text and use a 40-character look-behind window, matching the
# convention used in the runtime truthfulness checker.

# Phrases in gains_material that can meaningfully be negated.
# Action-verb phrases ("captures a piece") are excluded — they are rarely
# negated in natural language and require separate handling if needed.
_MATERIAL_NEGATABLE_PHRASES: frozenset[str] = frozenset({
    "material gain",
    "net gain",
    "gains material",
    "gains a piece",
    "gaining a piece",
    "wins material",
})

# Negation sentinels for material gain suppression.
# Mirror the sentinels used in _check_reasoning_truthfulness in ranker_agent.
_MATERIAL_NEGATION_SENTINELS: tuple[str, ...] = (
    "no ",
    "not ",
    "without ",
    "despite no",
    "lack of",
    "despite the lack",
    "no material",
)

# Opponent-entity context indicators for near_promotion claim disambiguation.
# When any of these appear in the look-behind window before a near_promotion
# phrase, the claim is about the opponent's piece — redirect to opponent_near_promotion.
_NEAR_PROMO_OPP_CONTEXT: tuple[str, ...] = (
    "opponent",
    "enemy",
    "opposing",
)

# Direction-negative sentinels for mobility_increase suppression.
# When one of these appears before an ambiguous "mobility by" phrase, the
# claim direction is negative (reduces, not increases).
_MOBILITY_DIRECTION_NEGATIVE: tuple[str, ...] = (
    "reduces",
    "reducing",      # present participle: "reducing our mobility by two"
    "decreases",
    "decreasing",    # present participle: "decreasing our mobility by one"
    "lower",
    "cuts",
    "drops",
    "falls",
)

# mobility_increase phrases that are directionally ambiguous — they do not
# encode direction themselves, so a surrounding context check is required.
_MOBILITY_INCREASE_AMBIGUOUS: frozenset[str] = frozenset({
    "our mobility by",   # "reduces our mobility by one" → false mobility_increase
})

# ---------------------------------------------------------------------------
# score_gap_advantage disambiguation (Phase-6 Fix 4)
# ---------------------------------------------------------------------------
# The bare phrase "scores" is short and matches valid non-comparison
# sentences (e.g. "this move scores well", "the minimax score is …").  To
# fire score_gap_advantage on bare "scores" we require at least one
# comparison sentinel within a small window of the match.  Compound phrases
# ("points better than", "next-best option", "best alternative") remain
# unambiguous and fire unconditionally.

_SCORE_GAP_AMBIGUOUS_PHRASES: frozenset[str] = frozenset({
    "scores",
})

_SCORE_GAP_COMPARISON_SENTINELS: tuple[str, ...] = (
    "points better",
    "points more",
    "points above",
    "better than",
    "higher than",
    "next-best",
    "next best",
    "best alternative",
    "vs ",
    "versus",
    "compared to",
    "alternative move",
    "alternative option",
    "minimax:",        # adversity seed format: "(minimax: A vs B)"
)


def _phrase_has_nearby_sentinel(
    text: str,
    phrase: str,
    sentinels: tuple[str, ...],
    window: int = 60,
) -> bool:
    """Return True iff any sentinel appears within `window` characters on
    EITHER side of the first occurrence of `phrase` in `text`.  All inputs
    are expected to be lowercase.  Returns False when `phrase` is not
    present or no sentinel is found in either window.

    Distinct from `_phrase_in_negated_context`, which only looks behind.
    Comparison sentinels for score_gap_advantage can appear either before
    ("better than the chosen move scores …") or after ("scores 25 points
    better").
    """
    idx = text.find(phrase)
    if idx == -1:
        return False
    before = text[max(0, idx - window): idx]
    after  = text[idx + len(phrase): idx + len(phrase) + window]
    return any((s in before) or (s in after) for s in sentinels)


def _phrase_in_negated_context(
    text: str,
    phrase: str,
    sentinels: tuple[str, ...],
    window: int = 40,
) -> bool:
    """
    Return True when the FIRST occurrence of `phrase` in `text` has at least
    one of `sentinels` in the `window` characters immediately before it.

    All inputs are expected to be lowercase.
    Returns False when `phrase` is not present or when no sentinel is found.
    """
    idx = text.find(phrase)
    if idx == -1:
        return False
    look_behind = text[max(0, idx - window): idx]
    return any(s in look_behind for s in sentinels)


# ---------------------------------------------------------------------------
# Seed matching helpers
# ---------------------------------------------------------------------------

def _find_matching_seed(
    seed_markers: list[str],
    seeds: list[str],
) -> Optional[str]:
    """
    Return the first seed string that contains any of the seed_markers,
    or None if no seed matches.  Case-insensitive.

    Markers prefixed with "^" are start-of-string anchors: the seed must
    START WITH the marker text (after stripping "^") rather than contain it
    as a substring.  Use "^near_promotion=true" to avoid a false match inside
    "opponent_near_promotion=true".
    """
    if not seeds or not seed_markers:
        return None
    for seed in seeds:
        seed_lower = seed.lower()
        for marker in seed_markers:
            if marker.startswith("^"):
                if seed_lower.startswith(marker[1:].lower()):
                    return seed
            elif marker.lower() in seed_lower:
                return seed
    return None


def _check_fact_support(
    entry: _PhraseEntry,
    facts: dict[str, Any],
) -> bool:
    """
    Return True if the fact dict contains a value consistent with this claim.
    """
    if entry.fact_field is None:
        return False
    value = facts.get(entry.fact_field)
    if value is None:
        return False
    if entry.fact_supports is not None:
        return bool(entry.fact_supports(value))
    # If no predicate is defined, presence alone is sufficient.
    return True


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_claims(
    reasoning_text: str,
    reasoning_seeds: Optional[list[str]] = None,
    facts: Optional[dict[str, Any]] = None,
) -> list[ClaimRecord]:
    """
    Extract structured claims from a ranker reasoning paragraph.

    Parameters
    ----------
    reasoning_text : str
        The final reasoning string from ``last_move_reasoning``.
    reasoning_seeds : list[str] or None
        The seed list from ``ranker_diagnostics["reasoning_seeds"]``.
        Used to determine whether a detected claim was authorised.
    facts : dict or None
        The engine-computed facts dict for the chosen move.
        Used as a secondary support check when no seed matched.

    Returns
    -------
    list[ClaimRecord]
        One record per detected claim, ordered by appearance in the
        phrase table (safety → tactical → structural → strategic).
        Deterministic: same inputs always produce the same output.

    Notes
    -----
    - This function never mutates its inputs.
    - This function never calls an LLM or any external service.
    - Claims are detected by case-insensitive substring matching.
    - A claim appears at most once in the output even if its phrase
      matches multiple times in the text.
    """
    if not reasoning_text:
        return []

    seeds: list[str] = list(reasoning_seeds) if reasoning_seeds else []
    fact_dict: dict[str, Any] = dict(facts) if facts else {}
    text_lower = reasoning_text.lower()

    records: list[ClaimRecord] = []

    for entry in _PHRASE_TABLE:
        # ── Phase 1: find the first matching phrase in the text ────────
        matched_phrase: Optional[str] = None
        for phrase in entry.phrases:
            if phrase.lower() in text_lower:
                matched_phrase = phrase
                break

        if matched_phrase is None:
            continue  # claim type not present in this reasoning text

        # ── Phase 1.5: dedup — skip if this claim_type was already added ─────
        # Prevents a double record when a redirect (e.g. near_promotion →
        # opponent_near_promotion) fires before the direct phrase entry.
        if any(r.claim_type == entry.claim_type for r in records):
            continue

        # ── Phase 1a: comparison-context suppression for score_gap_advantage ─
        # The bare phrase "scores" matches valid non-comparison sentences
        # ("this move scores well", "the minimax score is …").  We only fire
        # score_gap_advantage on bare "scores" when a comparison sentinel is
        # nearby (Phase-6 Fix 4).  Compound phrases ("points better than",
        # "next-best option", "best alternative") remain unambiguous and skip
        # this gate.
        if entry.claim_type == "score_gap_advantage":
            if matched_phrase.lower() in _SCORE_GAP_AMBIGUOUS_PHRASES:
                if not _phrase_has_nearby_sentinel(
                    text_lower,
                    matched_phrase.lower(),
                    _SCORE_GAP_COMPARISON_SENTINELS,
                ):
                    continue  # bare "scores" with no nearby comparison cue

        # ── Phase 1b: polarity / direction suppression ────────────────
        #
        # gains_material — suppress when the matched phrase is inside a
        # negated context ("no material gain", "despite no net gain", …).
        if entry.claim_type == "gains_material":
            if (
                matched_phrase.lower() in _MATERIAL_NEGATABLE_PHRASES
                and _phrase_in_negated_context(
                    text_lower,
                    matched_phrase.lower(),
                    _MATERIAL_NEGATION_SENTINELS,
                )
            ):
                continue  # negated — do not emit a gains_material record

        # near_promotion — entity-context check.
        # If "opponent", "enemy", or "opposing" appears in the 40-char window
        # before the matched phrase, the claim is about the opponent's piece.
        # Redirect to opponent_near_promotion; do NOT emit a near_promotion record.
        if entry.claim_type == "near_promotion":
            if _phrase_in_negated_context(
                text_lower,
                matched_phrase.lower(),
                _NEAR_PROMO_OPP_CONTEXT,
                window=40,
            ):
                redirect = _ENTRY_BY_TYPE.get("opponent_near_promotion")
                if redirect is not None:
                    r_seed = _find_matching_seed(redirect.seed_markers, seeds)
                    r_fact = _check_fact_support(redirect, fact_dict)
                    if r_seed is not None:
                        r_src, r_status = "seed", ClaimStatus.SUPPORTED
                    elif r_fact:
                        r_src, r_status = "fact_phrase", ClaimStatus.SUPPORTED
                    else:
                        r_src, r_status = "unsupported_phrase", ClaimStatus.UNSUPPORTED
                    records.append(ClaimRecord(
                        claim_type="opponent_near_promotion",
                        claim_status=r_status,
                        claim_verifiability=redirect.verifiability,
                        seed_risk_type=redirect.seed_risk if r_seed else None,
                        hallucination_type=None,
                        matched_phrase=matched_phrase,
                        matched_seed=r_seed,
                        source=r_src,
                    ))
                continue  # do not emit a near_promotion record

        # mobility_increase — when an ambiguous phrase like "our mobility by"
        # is preceded by a direction-negative word ("reduces our mobility by"),
        # suppress mobility_increase and redirect to mobility_decrease instead
        # (only if mobility_decrease was not already extracted).
        if entry.claim_type == "mobility_increase":
            if (
                matched_phrase.lower() in _MOBILITY_INCREASE_AMBIGUOUS
                and _phrase_in_negated_context(
                    text_lower,
                    matched_phrase.lower(),
                    _MOBILITY_DIRECTION_NEGATIVE,
                )
            ):
                already_types = {r.claim_type for r in records}
                if "mobility_decrease" not in already_types:
                    redirect = _ENTRY_BY_TYPE.get("mobility_decrease")
                    if redirect is not None:
                        r_seed = _find_matching_seed(redirect.seed_markers, seeds)
                        r_fact = _check_fact_support(redirect, fact_dict)
                        if r_seed is not None:
                            r_src, r_status = "seed", ClaimStatus.SUPPORTED
                        elif r_fact:
                            r_src, r_status = "fact_phrase", ClaimStatus.SUPPORTED
                        else:
                            r_src, r_status = "unsupported_phrase", ClaimStatus.UNSUPPORTED
                        records.append(ClaimRecord(
                            claim_type="mobility_decrease",
                            claim_status=r_status,
                            claim_verifiability=redirect.verifiability,
                            seed_risk_type=redirect.seed_risk if r_seed else None,
                            hallucination_type=None,
                            matched_phrase=matched_phrase,
                            matched_seed=r_seed,
                            source=r_src,
                        ))
                continue  # suppress the mobility_increase record

        # ── Phase 2: check seed support ───────────────────────────────
        matched_seed = _find_matching_seed(entry.seed_markers, seeds)

        # ── Phase 3: check fact support (fallback) ────────────────────
        fact_supported = _check_fact_support(entry, fact_dict)

        # ── Phase 4: determine status and source ─────────────────────
        if matched_seed is not None:
            source = "seed"
            status = ClaimStatus.SUPPORTED
        elif fact_supported:
            source = "fact_phrase"
            status = ClaimStatus.SUPPORTED
        else:
            source = "unsupported_phrase"
            status = ClaimStatus.UNSUPPORTED

        # ── Phase 5: determine seed risk type ─────────────────────────
        # If the claim was seed-authorised, use the entry's default risk.
        # If the claim was not seeded, seed_risk_type is None (no seed to
        # classify).
        seed_risk = entry.seed_risk if matched_seed is not None else None

        # ── Phase 6: flag potential hallucinations ────────────────────
        # At extraction time we only flag the most obvious case:
        # an unverifiable claim with no seed backing is a fabricated claim.
        hallucination: Optional[HallucinationType] = None
        if (
            entry.verifiability == ClaimVerifiability.UNVERIFIABLE
            and status == ClaimStatus.UNSUPPORTED
        ):
            hallucination = HallucinationType.FABRICATED_CLAIM

        # ── Phase 7: build record ─────────────────────────────────────
        records.append(ClaimRecord(
            claim_type=entry.claim_type,
            claim_status=status,
            claim_verifiability=entry.verifiability,
            seed_risk_type=seed_risk,
            hallucination_type=hallucination,
            matched_phrase=matched_phrase,
            matched_seed=matched_seed,
            source=source,
        ))

    return records
