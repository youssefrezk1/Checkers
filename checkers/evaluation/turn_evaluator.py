# checkers/evaluation/turn_evaluator.py
#
# Per-turn evaluation record.
#
# PURPOSE
# -------
# Evaluates ONE completed explainer turn by:
#   1. Extracting claims from the reasoning text (claim_extractor).
#   2. Verifying each claim against symbolic move facts (claim_verifier).
#   3. Classifying the reasoning_path and trajectory_events from the
#      explainer_diagnostics dict.
#   4. Returning a structured TurnEvaluationRecord with aggregate counts
#      and flags.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls.
# - No JSONL logging, batch replay, or aggregate reporting.
# - Deterministic: same inputs → same output always.
# - Inputs are never mutated.
#
# USAGE
# -----
# This module is imported only by evaluation scripts and tests.
# It must NEVER be imported by the runtime pipeline.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from checkers.evaluation.claim_extractor import ClaimRecord, extract_claims
from checkers.evaluation.claim_verifier import verify_claims
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


# ---------------------------------------------------------------------------
# Reasoning path labels
# ---------------------------------------------------------------------------

REASONING_PATH_SEEDED_LLM        = "seeded_llm"
REASONING_PATH_REFINEMENT_REPAIRED = "refinement_repaired"
REASONING_PATH_SEED_FALLBACK      = "seed_fallback"
REASONING_PATH_HARDCODED_FALLBACK = "hardcoded_fallback"
REASONING_PATH_UNKNOWN            = "unknown"

# ---------------------------------------------------------------------------
# Trajectory event labels
# ---------------------------------------------------------------------------

TRAJ_API_FAILURE                      = "api_failure"
TRAJ_RETRY_USED                       = "retry_used"
TRAJ_RETRY_REPAIRED                   = "retry_repaired"
TRAJ_RETRY_FAILED                     = "retry_failed"
TRAJ_OVERRIDE_USED                    = "override_used"
TRAJ_SEED_FALLBACK                    = "seed_fallback_used"
TRAJ_PYTHON_RESCUE                    = "python_rescue_used"
# Pre-repair contradiction events (derived from reasoning_initial_contradictions)
TRAJ_INTERNAL_CONTRADICTION           = "internal_contradiction_detected"
TRAJ_CONTRADICTION_REPAIRED           = "contradiction_repaired"
TRAJ_CONTRADICTION_SEED_FALLBACK      = "contradiction_fell_back_to_seed_summary"


# ---------------------------------------------------------------------------
# TurnEvaluationRecord
# ---------------------------------------------------------------------------

@dataclass
class TurnEvaluationRecord:
    """
    Structured evaluation result for one ranker turn.

    Fields
    ------
    turn_id : str
        Caller-supplied identifier (e.g. "game_3_turn_12" or a UUID).

    claims : list[ClaimRecord]
        All claims extracted and verified from the reasoning paragraph.
        Order matches the phrase table order in claim_extractor.

    total_claims : int
        len(claims)

    supported_count : int
        Claims with ClaimStatus.SUPPORTED.

    contradicted_count : int
        Claims with ClaimStatus.CONTRADICTED.

    unsupported_count : int
        Claims with ClaimStatus.UNSUPPORTED.

    vague_count : int
        Claims with ClaimStatus.VAGUE.

    has_contradiction : bool
        True if contradicted_count > 0.

    has_unsupported : bool
        True if unsupported_count > 0.

    has_vague : bool
        True if vague_count > 0.

    reasoning_path : str
        Classification of which pipeline code path produced the reasoning.
        One of: seeded_llm, refinement_repaired, seed_fallback,
                hardcoded_fallback, unknown.

    trajectory_events : list[str]
        Ordered pipeline events inferred from ranker_diagnostics.
        Subset of: api_failure, retry_used, retry_repaired, retry_failed,
                   override_used, seed_fallback_used, python_rescue_used.
    """

    turn_id: str
    claims: List[ClaimRecord]
    total_claims: int
    supported_count: int
    contradicted_count: int
    unsupported_count: int
    vague_count: int
    has_contradiction: bool
    has_unsupported: bool
    has_vague: bool
    reasoning_path: str
    trajectory_events: List[str]
    # ── Phase 2 provenance fields (optional; default-safe for old records) ────
    final_choice_source: str = ""
    override_branch_name: Optional[str] = None
    best_score_tie_count: int = 0
    minimax_best_path: Optional[List] = None
    minimax_best_score: Optional[float] = None
    retry_all_paths: List = field(default_factory=list)
    raw_llm_reasoning_pre_refinement: Optional[str] = None
    # ── Phase 2.2 tie-break fields (evaluation/logging metadata only) ────────
    tied_candidate_paths: List = field(default_factory=list)
    # ── Phase 2.1 faithful-reasoning note (evaluation/logging metadata only) ─
    provenance_note: str = ""
    # ── Phase 2.3a retry diversity fields (evaluation/logging metadata only) ─
    retry_rejection_reasons: List[str] = field(default_factory=list)
    retry_duplicate_count: int = 0


# ---------------------------------------------------------------------------
# Reasoning path classifier
# ---------------------------------------------------------------------------

def _classify_reasoning_path(diag: dict[str, Any]) -> str:
    """
    Infer which pipeline code path produced the reasoning text.

    Priority order (first match wins):
    1. reasoning_is_seed_fallback=True        → seed_fallback
    2. api_call_failure_count > 0             → hardcoded_fallback
    3. reasoning_refinement_retry_count > 0
       AND NOT reasoning_has_unresolved_contradiction → refinement_repaired
    4. default                                → seeded_llm

    Returns "unknown" only when diagnostics is not a valid dict.
    """
    if not isinstance(diag, dict):
        return REASONING_PATH_UNKNOWN

    if diag.get("reasoning_is_seed_fallback", False):
        return REASONING_PATH_SEED_FALLBACK

    if (diag.get("api_call_failure_count", 0) or 0) > 0:
        return REASONING_PATH_HARDCODED_FALLBACK

    retry_count = diag.get("reasoning_refinement_retry_count", 0) or 0
    unresolved  = diag.get("reasoning_has_unresolved_contradiction", False)
    if retry_count > 0 and not unresolved:
        return REASONING_PATH_REFINEMENT_REPAIRED

    return REASONING_PATH_SEEDED_LLM


# ---------------------------------------------------------------------------
# Trajectory event builder
# ---------------------------------------------------------------------------

def _build_trajectory_events(diag: dict[str, Any]) -> List[str]:
    """
    Build an ordered list of trajectory event labels from ranker_diagnostics.

    Events are appended in the order they would have occurred during execution.
    Each event is only added when the corresponding diagnostic flag is set.
    """
    if not isinstance(diag, dict):
        return []

    events: List[str] = []

    # API failure — first thing that can go wrong
    if (diag.get("api_call_failure_count", 0) or 0) > 0:
        events.append(TRAJ_API_FAILURE)

    # Retry loop
    retry_attempts = diag.get("override_retry_attempts", 0) or 0
    if retry_attempts > 0:
        events.append(TRAJ_RETRY_USED)
        if diag.get("override_retry_resolved", False):
            events.append(TRAJ_RETRY_REPAIRED)
        else:
            events.append(TRAJ_RETRY_FAILED)

    # Override branch
    if diag.get("override_branch_name") is not None:
        events.append(TRAJ_OVERRIDE_USED)

    # Python rescue (deterministic fallback move selection)
    if diag.get("override_fallback_applied", False):
        events.append(TRAJ_PYTHON_RESCUE)

    # Seed fallback (reasoning only — move still selected, reasoning is seeds joined)
    if diag.get("reasoning_is_seed_fallback", False):
        events.append(TRAJ_SEED_FALLBACK)

    # Pre-repair contradiction events (new fields from ranker_diagnostics)
    if diag.get("reasoning_contradiction_detected", False):
        events.append(TRAJ_INTERNAL_CONTRADICTION)
        if diag.get("reasoning_contradiction_repaired", False):
            events.append(TRAJ_CONTRADICTION_REPAIRED)
        elif diag.get("reasoning_is_seed_fallback", False):
            events.append(TRAJ_CONTRADICTION_SEED_FALLBACK)

    return events


# ---------------------------------------------------------------------------
# Provenance note builder
# ---------------------------------------------------------------------------

def _build_provenance_note(diag: dict[str, Any]) -> str:
    """
    Build a plain-text audit note from ranker_diagnostics provenance fields.

    Returns "" for clean raw_llm and single_candidate turns (no note needed).
    Never raises; uses safe .get() defaults throughout.

    Rules (each appends an independent clause; combined with a single space):
    1. python_fallback  — LLM choice overridden by python safety net.
    2. retry_llm        — initial LLM path rejected; only if paths differ.
    3. best_score_tie_count > 1 — N moves tied at same minimax score.
    4. retry degeneracy — retry_all_paths contains duplicate entries.
    """
    if not isinstance(diag, dict):
        return ""

    parts: List[str] = []
    source     = diag.get("final_choice_source", "")
    raw_path   = diag.get("raw_llm_choice_path")
    final_path = diag.get("final_chosen_path")

    def _norm(p: Any) -> list:
        """Normalise a path to a list-of-lists for comparison."""
        try:
            return [list(sq) for sq in (p or [])]
        except (TypeError, ValueError):
            return []

    # Rule 1: python safety net overrode the LLM's choice
    if source == "python_fallback":
        raw_str   = str(raw_path)   if raw_path   is not None else "unknown"
        final_str = str(final_path) if final_path is not None else "unknown"
        parts.append(
            f"[DECISION] python safety override applied: "
            f"LLM chose {raw_str}, final move is {final_str}."
        )

    # Rule 2: retry corrected the LLM but chose a different path
    elif source == "retry_llm":
        if raw_path is not None and final_path is not None:
            if _norm(raw_path) != _norm(final_path):
                parts.append(
                    f"[DECISION] LLM initial choice {str(raw_path)} rejected; "
                    f"retry selected {str(final_path)}."
                )

    # Rule 3: tie — multiple moves had equal best minimax score
    tie_count = (diag.get("best_score_tie_count") or 0)
    if isinstance(tie_count, int) and tie_count > 1:
        score = diag.get("minimax_best_score")
        score_str = (
            f" at score {score:.2f}" if isinstance(score, (int, float)) else ""
        )
        final_str = str(final_path) if final_path is not None else "unknown"

        # Determine why this particular tied move was selected
        best_path  = diag.get("minimax_best_path")
        tie_reason = diag.get("tie_break_reason")

        if tie_reason == "promotion":
            reason_str = "promotion preferred within tie window"
        elif source == "python_fallback":
            reason_str = "python fallback selected a best-score tied move"
        else:
            try:
                final_is_best = (
                    _norm(final_path) == _norm(best_path)
                    if final_path is not None and best_path is not None
                    else None
                )
            except Exception:
                final_is_best = None

            if final_is_best is True:
                reason_str = "final move agreed with minimax-best among tied moves"
            elif final_is_best is False:
                # Emphasise equal score — this is not a suboptimal choice
                reason_str = (
                    f"LLM chose a different tied move with equal minimax score "
                    f"(minimax-best was {str(best_path)})"
                )
            else:
                reason_str = "selection reason unknown"

        # Include peer paths only when the tie set is small enough to be readable
        tied_paths = list(diag.get("tied_candidate_paths") or [])
        peer_str = (
            f" Tied paths: {tied_paths}."
            if tied_paths and len(tied_paths) <= 5
            else ""
        )

        parts.append(
            f"[TIE] {tie_count} moves tied{score_str}; "
            f"selected {final_str} — {reason_str}.{peer_str}"
        )

    # Rule 4: retry degeneracy — same path attempted more than once
    retry_paths = list(diag.get("retry_all_paths") or [])
    if len(retry_paths) > 1:
        seen: set = set()
        dupes: List[Any] = []
        for p in retry_paths:
            key = str(p)
            if key in seen and key not in {str(d) for d in dupes}:
                dupes.append(p)
            seen.add(key)
        if dupes:
            dupe_str = ", ".join(str(d) for d in dupes)
            parts.append(
                f"[RETRY_DEGENERATE] retry loop repeated path(s): {dupe_str}."
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Retry duplicate counter (Phase 2.3a)
# ---------------------------------------------------------------------------

def _count_retry_duplicates(retry_paths: List) -> int:
    """
    Count how many entries in retry_paths are exact repeats of an earlier entry.

    Uses str() normalisation consistent with _build_provenance_note Rule 4.
    Returns 0 for empty or single-entry lists.
    """
    if len(retry_paths) <= 1:
        return 0
    seen: set = set()
    dupes = 0
    for p in retry_paths:
        key = str(p)
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    return dupes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_turn(
    reasoning_text: str,
    reasoning_seeds: Optional[List[str]] = None,
    facts: Optional[dict[str, Any]] = None,
    ranker_diagnostics: Optional[dict[str, Any]] = None,
    turn_id: Optional[str] = None,
) -> TurnEvaluationRecord:
    """
    Evaluate one completed ranker turn.

    Steps:
    1. extract_claims(reasoning_text, reasoning_seeds, facts)
    2. verify_claims(claims, facts)
    3. Classify reasoning_path from ranker_diagnostics.
    4. Build trajectory_events from ranker_diagnostics.
    5. Aggregate counts and flags.

    Parameters
    ----------
    reasoning_text : str
        The final reasoning paragraph from last_move_reasoning.
    reasoning_seeds : list[str] or None
        Seeds from ranker_diagnostics["reasoning_seeds"]. Used by
        extract_claims() to determine seed support.
    facts : dict or None
        Engine-computed facts from compute_move_facts(). Used by
        verify_claims() to check factual accuracy.
    ranker_diagnostics : dict or None
        Full ranker_diagnostics dict. Used for path/event classification.
    turn_id : str or None
        Caller-supplied identifier. Defaults to "unknown" if not provided.

    Returns
    -------
    TurnEvaluationRecord
        Fully populated record. Never raises; on bad/missing input returns
        a minimal record with empty claims.
    """
    tid: str = turn_id if isinstance(turn_id, str) and turn_id else "unknown"
    diag: dict[str, Any] = ranker_diagnostics if isinstance(ranker_diagnostics, dict) else {}
    seeds: List[str] = list(reasoning_seeds) if reasoning_seeds else []
    fact_dict: dict[str, Any] = dict(facts) if isinstance(facts, dict) else {}

    # ── Step 1: extract ────────────────────────────────────────────────────
    raw_claims = extract_claims(reasoning_text, reasoning_seeds=seeds, facts=fact_dict)

    # ── Step 2: verify ─────────────────────────────────────────────────────
    # Phase-6 Fix 2: build a verifier context dict from ranker_diagnostics so
    # score_gap_advantage can do hard verification when next_best_minimax_score
    # is available.  Unknown keys in `context` are ignored by the verifier.
    verifier_context: dict[str, Any] = {}
    nb = diag.get("next_best_minimax_score")
    if isinstance(nb, (int, float)):
        verifier_context["next_best_minimax_score"] = nb
    claims = verify_claims(raw_claims, fact_dict, context=verifier_context or None)

    # ── Step 3: counts and flags ───────────────────────────────────────────
    total        = len(claims)
    supported    = sum(1 for c in claims if c.claim_status == ClaimStatus.SUPPORTED)
    contradicted = sum(1 for c in claims if c.claim_status == ClaimStatus.CONTRADICTED)
    unsupported  = sum(1 for c in claims if c.claim_status == ClaimStatus.UNSUPPORTED)
    vague        = sum(1 for c in claims if c.claim_status == ClaimStatus.VAGUE)

    # ── Step 4: path and events ────────────────────────────────────────────
    # Pass the RAW ranker_diagnostics to the path classifier so that non-dict
    # inputs correctly produce REASONING_PATH_UNKNOWN (the classifier handles
    # type-checking internally).  The normalised `diag` (defaulting to {}) is
    # used only for trajectory events and seed extraction where {} is correct.
    reasoning_path     = _classify_reasoning_path(ranker_diagnostics)
    trajectory_events  = _build_trajectory_events(diag)

    return TurnEvaluationRecord(
        turn_id=tid,
        claims=claims,
        total_claims=total,
        supported_count=supported,
        contradicted_count=contradicted,
        unsupported_count=unsupported,
        vague_count=vague,
        has_contradiction=contradicted > 0,
        has_unsupported=unsupported > 0,
        has_vague=vague > 0,
        reasoning_path=reasoning_path,
        trajectory_events=trajectory_events,
        # Phase 2 provenance — safe defaults when fields absent from older diagnostics
        final_choice_source=diag.get("final_choice_source", ""),
        override_branch_name=diag.get("override_branch_name"),
        best_score_tie_count=diag.get("best_score_tie_count", 0) or 0,
        minimax_best_path=diag.get("minimax_best_path"),
        minimax_best_score=diag.get("minimax_best_score"),
        retry_all_paths=list(diag.get("retry_all_paths") or []),
        raw_llm_reasoning_pre_refinement=diag.get("raw_llm_reasoning_pre_refinement"),
        # Phase 2.2 tie-break field
        tied_candidate_paths=list(diag.get("tied_candidate_paths") or []),
        # Phase 2.1 — evaluation/logging metadata only; reasoning text is unchanged
        provenance_note=_build_provenance_note(diag),
        # Phase 2.3a retry diversity — safe defaults for old records
        retry_rejection_reasons=list(diag.get("retry_rejection_reasons") or []),
        retry_duplicate_count=_count_retry_duplicates(list(diag.get("retry_all_paths") or [])),
    )
