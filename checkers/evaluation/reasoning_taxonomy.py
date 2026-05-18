# checkers/evaluation/reasoning_taxonomy.py
#
# Taxonomy definitions for reasoning faithfulness evaluation.
#
# PURPOSE
# -------
# This module defines the official controlled vocabulary used throughout
# the evaluation framework to label, classify, and compare LLM-generated
# reasoning against the symbolic facts produced by compute_move_facts().
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No side effects. No I/O. Pure taxonomy only.
# - All enum values are lowercase strings to allow JSON serialisation and
#   stable cross-tool comparisons without case-folding.
# - All categories are mutually exclusive within their enum; a single
#   observation may receive at most one label from each enum.
#
# USAGE
# -----
# These enums are imported only by evaluation scripts, test harnesses, and
# audit notebooks. They must NEVER be imported by the runtime pipeline.

from enum import Enum


# ---------------------------------------------------------------------------
# 1. ClaimVerifiability
# ---------------------------------------------------------------------------

class ClaimVerifiability(str, Enum):
    """
    Classifies how objectively a reasoning claim can be verified against
    the symbolic facts produced by compute_move_facts().

    A claim is the smallest coherent assertion in the LLM reasoning paragraph
    (e.g. "avoids recapture", "gains material", "improves piece coordination").

    Levels
    ------
    FULLY_VERIFIABLE
        The claim maps exactly to one or more deterministic boolean/integer
        fields in the facts dict.  A contradiction can be detected mechanically
        by comparing the claim wording to the field value.
        Examples: opponent_can_recapture, captures_count, results_in_king,
                  leaves_piece_isolated, mobility_reduction direction.

    PARTIALLY_VERIFIABLE
        The claim references a fact that exists but whose wording introduces
        an interpretation not fully captured by the field value alone.  A
        contradiction may exist but cannot always be detected automatically.
        Examples: "creates a promotion threat" (near_promotion=True does not
        verify the path is unblocked), "puts opponent on the defensive"
        (creates_immediate_threat=True only verifies one jump exists).

    UNVERIFIABLE
        The claim invokes concepts for which no symbolic fact is computed.
        It may be plausible but cannot be confirmed or refuted from the
        available fact dict.
        Examples: strategic initiative, long-term compensation, positional
        activity (when not tied to a specific numeric field).
    """

    FULLY_VERIFIABLE   = "fully_verifiable"
    PARTIALLY_VERIFIABLE = "partially_verifiable"
    UNVERIFIABLE       = "unverifiable"


# ---------------------------------------------------------------------------
# 2. ClaimStatus
# ---------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    """
    Records the verification outcome for a single claim extracted from
    LLM-generated reasoning.

    This is the result of comparing the claim wording to the symbolic facts.
    It is assigned per-claim during post-hoc evaluation and stored in the
    evaluation trace.

    Statuses
    --------
    SUPPORTED
        The claim wording is consistent with the corresponding fact value.
        No contradiction detected.

    CONTRADICTED
        The claim wording directly contradicts the corresponding fact value.
        Example: reasoning says "avoids recapture" but
                 opponent_can_recapture=True.

    UNSUPPORTED
        The claim references a concept for which the fact dict contains no
        relevant field or the field value does not support the claim, but
        a hard contradiction cannot be established.
        Example: "improves long-term king coordination" when no coordination
                 fact is computed.

    VAGUE
        The claim uses language too imprecise to evaluate.  It neither
        contradicts nor supports a specific fact.
        Example: "this is a good positional step".

    NOT_CHECKED
        The claim was not subjected to a verification check in this run,
        typically because the relevant fact field was absent from the
        fact dict or the claim fell outside the scope of the current
        checker.
    """

    SUPPORTED    = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED  = "unsupported"
    VAGUE        = "vague"
    NOT_CHECKED  = "not_checked"


# ---------------------------------------------------------------------------
# 3. HallucinationType
# ---------------------------------------------------------------------------

class HallucinationType(str, Enum):
    """
    Classifies the specific kind of hallucination detected in LLM reasoning.

    A hallucination is any statement in the reasoning paragraph that is
    not grounded in the symbolic facts, the reasoning seeds, or the rules
    of the game as provided in the system prompt.

    Types
    -----
    FACTUAL_CONTRADICTION
        The reasoning states a claim that directly contradicts a deterministic
        fact value.
        Example: "the move gains a piece" when captures_count=0.

    CONTEXT_INCONSISTENCY
        The reasoning is internally consistent but inconsistent with the
        game context (phase, score state, or strategic priorities) provided
        in the prompt.
        Example: claiming "we are ahead in material" when material_advantage<0.

    LOGICAL_INCONSISTENCY
        The reasoning contains two claims that are mutually contradictory
        within the paragraph itself, independent of fact values.
        Example: "the move avoids all threats" followed by "one piece remains
                 under attack".

    INSTRUCTION_INCONSISTENCY
        The reasoning violates an explicit instruction in the system or seed
        prompt (e.g. uses a forbidden vocabulary term, cites an internal
        metric name, or uses a label explicitly prohibited by the prompt).

    FABRICATED_CLAIM
        The reasoning introduces a concept, fact, or figure that was not
        provided in the seeds, the facts dict, or the game prompt, and
        that cannot be derived from them.
        Example: a specific move count or king distance value that was never
                 computed.

    OVERCLAIM
        The reasoning makes a claim that is directionally consistent with a
        fact but asserts more strength, certainty, or scope than the fact
        supports.
        Example: "a multi-jump sequence is available" when only a single jump
                 exists (shot_sequence_available=True but two_for_one_potential=False).

    WRONG_MOVE_REFERENCE
        The reasoning attributes a property to a move index or piece that
        does not match the chosen move or the comparison move cited in the
        seed.
        Example: describing the properties of move [2] as belonging to the
                 chosen move [0].
    """

    FACTUAL_CONTRADICTION   = "factual_contradiction"
    CONTEXT_INCONSISTENCY   = "context_inconsistency"
    LOGICAL_INCONSISTENCY   = "logical_inconsistency"
    INSTRUCTION_INCONSISTENCY = "instruction_inconsistency"
    FABRICATED_CLAIM        = "fabricated_claim"
    OVERCLAIM               = "overclaim"
    WRONG_MOVE_REFERENCE    = "wrong_move_reference"


# ---------------------------------------------------------------------------
# 4. SeedRiskType
# ---------------------------------------------------------------------------

class SeedRiskType(str, Enum):
    """
    Classifies the grounding risk level of each individual reasoning seed
    emitted by _build_grounded_reasoning_seeds().

    Seeds are the structured fact-derived statements fed to the LLM as the
    authorised basis for its reasoning paragraph.  Even within the seed list,
    different seeds carry different grounding quality.

    Risk Types
    ----------
    STRICT_FACT
        The seed wording is a direct, literal transcription of a single
        deterministic fact field value with no added interpretation.
        The claim in the paragraph can be fully verified against the field.
        Examples: "opponent_can_recapture=false",
                  "captures_count=2, net_gain=2".

    INTERPRETIVE
        The seed introduces wording that goes slightly beyond the raw field
        value by adding directional or contextual language.  The core claim
        is verifiable but the added framing is interpretive.
        Examples: "reduces opponent mobility by N, restricting available
                  replies" (delta is exact; "restricting" is interpretive),
                  "preserves piece coordination" (adjacency is exact;
                  "coordination" is interpretive).

    OVERCLAIM_RISK
        The seed wording makes a claim that is stronger or broader than the
        underlying field can support.  An LLM that faithfully expands this
        seed will produce reasoning that overclaims.
        Example: "a multi-jump sequence is available to extend the attack"
                 when shot_sequence_available is a single-jump boolean.

    REDUNDANT
        The seed duplicates information already present in another seed in
        the same list.  Redundancy inflates a single fact's apparent weight
        in the reasoning paragraph.
        Example: creates_immediate_threat and shot_sequence_available are
                 computed by identical code but seeded as separate statements.

    MISLEADING
        The seed is technically derived from a fact but its phrasing could
        lead the LLM to draw an incorrect inference even without adding
        unsupported claims.
        Example: "develops a piece forward — improves piece activity" for any
                 simple forward move, regardless of whether the piece actually
                 improves its strategic role.
    """

    STRICT_FACT    = "strict_fact"
    INTERPRETIVE   = "interpretive"
    OVERCLAIM_RISK = "overclaim_risk"
    REDUNDANT      = "redundant"
    MISLEADING     = "misleading"


# ---------------------------------------------------------------------------
# 5. TrajectoryEventType
# ---------------------------------------------------------------------------

class TrajectoryEventType(str, Enum):
    """
    Labels discrete events in the per-turn pipeline execution trajectory.

    A trajectory is the ordered sequence of events that occurred between
    receiving a board position and producing the final chosen move.
    Recording these events allows evaluation of pipeline stability, retry
    rates, override rates, and reasoning fallback patterns.

    Events
    ------
    RAW_LLM_SUCCESS
        The LLM returned a valid JSON response on the first attempt with a
        parseable chosen_index and no contradictions in the reasoning.

    PARSE_FAILURE
        The LLM response could not be parsed as valid JSON or did not contain
        a usable chosen_index field.

    API_FAILURE
        The API call to the LLM backend raised an exception or returned a
        non-200 status code.

    RETRY_USED
        A retry of the LLM call was triggered (due to parse failure,
        contradiction, or API failure).

    RETRY_REPAIRED
        A retry succeeded: the retried response was parseable and either
        contained no contradictions or had fewer contradictions than the
        original.

    RETRY_FAILED
        A retry was attempted but still produced an unusable or contradicted
        response.

    OVERRIDE_USED
        The minimax guardrail overrode the LLM's chosen index and substituted
        the best minimax move instead.

    SEED_FALLBACK_USED
        The pipeline fell back to seeded reasoning (generated by
        _generate_seeded_reasoning) rather than accepting the LLM's
        free-form reasoning paragraph.

    PYTHON_RESCUE_USED
        A deterministic Python fallback selected the final move because the
        LLM pipeline failed to produce a usable result within the retry budget.

    FINAL_MOVE_LEGAL
        The move ultimately played is confirmed legal for the given board
        position and player.

    FINAL_MOVE_ILLEGAL
        The move ultimately played is illegal.  This event should never occur
        in a correctly functioning pipeline and is included for completeness
        of the trajectory log schema.
    """

    RAW_LLM_SUCCESS     = "raw_llm_success"
    PARSE_FAILURE       = "parse_failure"
    API_FAILURE         = "api_failure"
    RETRY_USED          = "retry_used"
    RETRY_REPAIRED      = "retry_repaired"
    RETRY_FAILED        = "retry_failed"
    OVERRIDE_USED       = "override_used"
    SEED_FALLBACK_USED  = "seed_fallback_used"
    PYTHON_RESCUE_USED  = "python_rescue_used"
    FINAL_MOVE_LEGAL    = "final_move_legal"
    FINAL_MOVE_ILLEGAL  = "final_move_illegal"
