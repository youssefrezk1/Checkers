from __future__ import annotations
from typing import Any, Optional, Union
from pydantic import BaseModel, Field
from checkers.engine.board import BLACK

class CheckersState(BaseModel):

    # ── Board ────────────────────────────────────────────
    board: list[list[int]] = Field(
        default_factory=lambda: [[0]*8 for _ in range(8)]
    )
    current_player: int = Field(default=BLACK)
    turn_number: int = Field(default=0)

    # ── Proposal Agent output ────────────────────────────
    # Raw JSON string from LLM or cleaned list after format_checker.
    proposed_moves: Union[list[dict[str, Any]], str] = Field(default_factory=list)

    legal_moves: list[dict[str, Any]] = Field(default_factory=list)

    # ── Chosen move (sole authority: deterministic_proposal_node) ─────────────
    # chosen_move is WRITTEN ONLY by deterministic_proposal_node. ranker_agent
    # treats it as read-only and never reassigns it. Cleared by state_manager.
    chosen_move: Optional[dict[str, Any]] = Field(default=None)
    # Natural-language explanation of the chosen move, produced by ranker_agent.
    # Reasoning-only output — has no effect on which move was selected.
    last_move_reasoning: Optional[str] = Field(default=None)

    # ── Deterministic proposal output ─────────────────────
    # Written by deterministic_proposal_node; cleared by state_manager each turn.
    # chosen_move_score: minimax_score of the deterministically selected move.
    chosen_move_score: Optional[float] = Field(default=None)
    # unchosen_moves: all legal moves EXCEPT the chosen one, preserving full facts.
    # Passed to ranker_agent for comparative explanation only — never used to
    # re-rank, re-score, or revisit the proposal's selection.
    unchosen_moves: list[dict[str, Any]] = Field(default_factory=list)

    # ── Candidate-menu evaluation snapshot ───────────────────────────────────
    # Snapshot of the full candidate list (paths + minimax_score) made visible
    # to ranker_agent when it generates its explanation. Used purely for
    # evaluation/replay so reviewers can see the same menu the ranker saw.
    # NOT a decision channel — the proposal has already selected by the time
    # this snapshot is taken.
    ranker_filtered_menu: Optional[list[dict[str, Any]]] = Field(default=None)

    # ── Legacy ranker retry / thesis metrics ──────────────────────────────────
    # Retained for log-schema compatibility with the pre-simplified pipeline.
    # In the proposal-authoritative pipeline the ranker performs no decision
    # work and never retries or falls back, so these counters stay at 0.
    ranker_retry_count: int = Field(default=0)
    ranker_failure_count: int = Field(default=0)
    ranker_fallback_count: int = Field(default=0)
    ranker_retry_budget: int = Field(default=3)

    # ── Retry control ────────────────────────────────────
    retry_count: int = Field(default=0)
    retry_budget: int = Field(default=3)

    # ── Orchestrator tracking ────────────────────────────
    pipeline: str = Field(default="normal")
    last_completed_node: Optional[str] = Field(default=None)

    # ── Game termination ─────────────────────────────────
    game_over: bool = Field(default=False)
    winner: Optional[int] = Field(default=None)
    draw: bool = Field(default=False)

    # ── Position history for draw detection ──────────────
    position_history: list[int] = Field(default_factory=list)

    # ── Validator feedback ────────────────────────────────
    # Written by validator when ALL proposed moves are illegal.
    # Read by proposal agent on retry to understand what went wrong.
    # Cleared by state_manager at the start of each new turn.
    feedback: Optional[str] = Field(default=None) 
    
    # ── Format checker tracking ───────────────────────────
    # Counts how many times format_checker had to auto-repair
    # or fully reject LLM output. Used for thesis evaluation.
    format_error_count: int = Field(default=0)


    # ── Proposal quality tracking ─────────────────────────
    # True when fewer than 3 structurally valid moves were passed to the
    # validator (allowed only if the engine lists fewer than 3 legal moves).
    insufficient_proposals: bool = Field(default=False)


    # ── Inter turn memory ─────────────────────────────────
    strategic_context: Optional[dict[str, Any]] = Field(default=None)

    # ── Move history ──────────────────────────────────────
    move_history: list[dict[str, Any]] = Field(default_factory=list)

    # ── Logging (set by logger_node on first ply, not reset) ─
    game_log_id: Optional[str] = Field(default=None)

    # ── Phase 8: Symbolic-first decision ─────────────────────────────────────
    # Written by symbolic_decision node; cleared by state_manager each turn.
    #
    # symbolic_scored_moves: full legal move list sorted best-first.
    # Each entry: {"move": dict, "minimax_score": float, "rank": int}
    # Proposal reads this as its pre-sorted candidate pool.
    symbolic_scored_moves: list[dict[str, Any]] = Field(default_factory=list)
    symbolic_best_move: Optional[dict[str, Any]] = Field(default=None)
    symbolic_best_score: float = Field(default=0.0)
    symbolic_second_best_score: Optional[float] = Field(default=None)
    symbolic_gap: float = Field(default=0.0)
    # Legacy compat fields — bypass is no longer used; always None/False.
    symbolic_bypass: bool = Field(default=False)
    symbolic_bypass_reason: Optional[str] = Field(default=None)
    # Thesis instrumentation — set by symbolic_decision and ranker_agent.
    llm_invoked: bool = Field(default=False)
    llm_agreed_with_symbolic_best: Optional[bool] = Field(default=None)
    proposal_diagnostics: Optional[dict[str, Any]] = Field(default=None)
    # Structured diagnostics emitted by ranker_agent when it generates the
    # explanation for the proposal-chosen move. Tracks reasoning provenance
    # (seeds, contradictions, refinement attempts, seed-fallback usage) plus
    # neutral placeholders for legacy decision-path keys (override/retry
    # counters always 0, final_choice_source="proposal_authoritative") so
    # evaluation schemas remain stable. Persists until the next ranker_agent
    # call (state_manager does NOT clear this field between turns).
    ranker_diagnostics: Optional[dict[str, Any]] = Field(default=None)
    # Full compute_move_facts() output for the proposal-chosen move.
    # Captured by ranker_agent from chosen_move["facts"] for evaluation export;
    # cleared by state_manager each turn. Read-only mirror of proposal output —
    # zero gameplay impact.
    chosen_move_facts: Optional[dict[str, Any]] = Field(default=None)