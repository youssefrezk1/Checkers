# checkers/agents/scorer_agent.py
#
# Standalone scoring helper that combines:
#   - symbolic_decision's search_root_all_scores pass (all legal moves)
#   - minimax_scorer's selective-D8 / promotion-race D10 upgrade
#   - move_facts enrichment for every legal move
#
# Does NOT touch graph.py, symbolic_decision.py, or minimax_scorer.py.
# Those nodes continue to operate unchanged. This module is a pure
# preparation step that can replace them in a future graph refactor.
#
# Public API:
#   score_all_legal_moves(board, player, position_history=None)
#       -> (enriched_moves, best_score, second_best_score, gap)
#
#   Each enriched move dict is:
#       {"type": str, "path": list, "captured": list, "facts": dict}
#   where facts includes all compute_move_facts fields plus:
#       facts["minimax_score"] : float  (after D8 if triggered)
#       facts["symbolic_rank"] : int    (1 = best-first)
#
# Configuration (all inherited from minimax_scorer / minimax env):
#   MINIMAX_ENABLED           (default true)  — false → neutral scores 0.0
#   MINIMAX_DEPTH             (default 3)     — shared search depth
#   SELECTIVE_D8_ENABLED      (default false) — D8 upgrade trigger
#   SELECTIVE_D8_PIECE_THRESHOLD, SELECTIVE_D8_GAP_THRESHOLD, SELECTIVE_D8_DEPTH
#   PROMOTION_RACE_VERIFY_*   — post-D8 D10 verification

from __future__ import annotations

import logging
from typing import Any

from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.evaluation import LOSS_SCORE
from checkers.engine.zobrist import compute_hash
from checkers.engine.move_facts import compute_move_facts
from checkers.search.minimax_core import (
    search_root_all_scores,
    clear_transposition_table,
)
from checkers.engine.minimax import MINIMAX_DEPTH
from checkers.nodes.minimax_scorer import (
    _apply_selective_d8,
    MINIMAX_ENABLED,
    SELECTIVE_D8_ENABLED,
)

# Same constant as symbolic_decision — keep in sync if that file changes it.
REPETITION_PENALTY = 60

# Depth used for the initial full-legal-move scoring pass.
# Matches MINIMAX_DEPTH so all pipeline components score at the same depth.
SCORER_DEPTH = MINIMAX_DEPTH

logger = logging.getLogger(__name__)


def score_all_legal_moves(
    board: list[list[int]],
    player: int,
    position_history: list[int] | None = None,
) -> tuple[list[dict[str, Any]], float, float | None, float]:
    """
    Score and annotate every legal move for *player* on *board*.

    Pipeline:
      1. get_all_legal_moves — ground truth legal move list
      2. search_root_all_scores at SCORER_DEPTH with repetition penalty
      3. compute_move_facts for each move
      4. _apply_selective_d8 upgrade (if SELECTIVE_D8_ENABLED)
      5. re-sort by final minimax_score and re-assign symbolic_rank

    When MINIMAX_ENABLED=False (ablation), skips steps 2 and 4 and
    returns 0.0 scores / rank=0 so the ranker prompt still works.

    Returns:
        enriched_moves  — list of move dicts sorted best-first, each:
                              {"type", "path", "captured", "facts"}
                          facts contains all compute_move_facts fields plus:
                              facts["minimax_score"]: float
                              facts["symbolic_rank"]: int (1=best)
        best_score      — float (minimax score of the best move)
        second_best_score — float | None
        gap             — best_score - second_best_score  (inf when only 1 move)
    """
    legal = get_all_legal_moves(board, player)

    if not legal:
        return [], float(LOSS_SCORE), None, 0.0

    # ── Ablation mode: skip search, still compute facts ───────────────────────
    if not MINIMAX_ENABLED:
        enriched: list[dict[str, Any]] = []
        for move in legal:
            facts = compute_move_facts(board, move, player)
            facts["minimax_score"] = 0.0
            facts["symbolic_rank"] = 0
            enriched.append({
                "type": move["type"],
                "path": move["path"],
                "captured": move.get("captured", []),
                "facts": facts,
            })
        return enriched, 0.0, None, 0.0

    # ── Step 1: full-legal-move search pass ───────────────────────────────────
    clear_transposition_table()
    _, _, raw_scored, _ = search_root_all_scores(
        board=board,
        current_player=player,
        depth=SCORER_DEPTH,
        legal_moves=legal,
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )

    # ── Step 2: repetition penalty (mirrors symbolic_decision logic) ──────────
    history_set: dict[int, int] = {}
    if position_history:
        for h in position_history:
            history_set[h] = history_set.get(h, 0) + 1

    scored: list[tuple[dict[str, Any], float]] = []
    for move, score in raw_scored:
        if history_set:
            child = apply_move(board, move)
            child_hash = compute_hash(child)
            repeat_count = history_set.get(child_hash, 0)
            if repeat_count > 0:
                score -= REPETITION_PENALTY * repeat_count
                logger.debug(
                    "[scorer_agent] repetition_penalty path=%s count=%d penalty=%.0f",
                    move.get("path"), repeat_count, REPETITION_PENALTY * repeat_count,
                )
        scored.append((move, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # ── Step 3: compute move facts and build initial enriched list ────────────
    enriched = []
    for rank, (move, score) in enumerate(scored, start=1):
        facts = compute_move_facts(board, move, player)
        facts["minimax_score"] = round(score, 2)
        facts["symbolic_rank"] = rank
        enriched.append({
            "type": move["type"],
            "path": move["path"],
            "captured": move.get("captured", []),
            "facts": facts,
        })

    # ── Step 4: selective D8 / promotion-race D10 upgrade ────────────────────
    # _apply_selective_d8 reads facts["minimax_score"] and promotion fields
    # already populated in step 3. It returns deepcopied candidates with
    # updated facts["minimax_score"] and facts["symbolic_rank"] when triggered.
    if SELECTIVE_D8_ENABLED:
        enriched = _apply_selective_d8(board, player, enriched)
        # Re-sort and re-number ranks so list stays consistent after D8 reordering.
        enriched.sort(key=lambda x: x["facts"].get("minimax_score", 0.0), reverse=True)
        for rank, entry in enumerate(enriched, start=1):
            entry["facts"]["symbolic_rank"] = rank

    # ── Summary stats from final scores ──────────────────────────────────────
    best_score = enriched[0]["facts"]["minimax_score"]
    second_best_score: float | None = (
        enriched[1]["facts"]["minimax_score"] if len(enriched) > 1 else None
    )
    if second_best_score is not None:
        gap: float = best_score - second_best_score
    else:
        gap = float("inf")

    logger.info(
        "[scorer_agent] depth=%d best=%.2f gap=%.2f n_legal=%d d8=%s",
        SCORER_DEPTH, best_score, gap, len(legal), SELECTIVE_D8_ENABLED,
    )

    return enriched, best_score, second_best_score, gap
