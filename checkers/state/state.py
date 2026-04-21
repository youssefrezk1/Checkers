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

    # ── Ranker output ────────────────────────────────────
    chosen_move: Optional[dict[str, Any]] = Field(default=None)
    # Explanation for the applied move (ranker or symbolic fallback); cleared each ply.
    last_move_reasoning: Optional[str] = Field(default=None)

    # ── Ranker decision-time snapshot (for evaluation) ───────────────────────
    # Captures the exact filtered candidate list the ranker saw (order + minimax_score),
    # so evaluation can measure filtered gaps against the same menu.
    ranker_filtered_menu: Optional[list[dict[str, Any]]] = Field(default=None)

    # ── Ranker retry / thesis metrics ────────────────────
    # Per-ply attempts that returned no valid choice; reset in state_manager.
    ranker_retry_count: int = Field(default=0)
    # Ranker attempts that failed (LLM/parse/invalid index); cumulative session total.
    ranker_failure_count: int = Field(default=0)
    # Times symbolic ranker_fallback ran after retries exhausted; cumulative.
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