# nodes/minimax_scorer.py
#
# LangGraph node: scores each candidate move with shallow minimax search.
# Runs AFTER validator (which enriches legal_moves with facts),
# and BEFORE ranker_agent (which makes the final decision).
#
# Behavior:
#   - reads state.legal_moves (already enriched by validator)
#   - for each move, calls score_move_with_minimax
#   - attaches facts["minimax_score"] to each move
#   - does NOT overwrite any existing facts
#   - returns updated legal_moves list
#
# Configuration:
#   MINIMAX_ENABLED=true   (default) — enable this node
#   MINIMAX_ENABLED=false  — skip scoring, useful for ablation study
#   MINIMAX_DEPTH=2        (default) — search depth

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from checkers.state.state import CheckersState
from checkers.engine.minimax import score_move_with_minimax, MINIMAX_DEPTH

# ── Configuration ─────────────────────────────────────────────────────────────
_enabled_env = os.environ.get("MINIMAX_ENABLED", "true").lower()
MINIMAX_ENABLED = _enabled_env in ("1", "true", "yes", "on")


def minimax_scorer(state: CheckersState) -> dict:
    """
    Scores each candidate move in state.legal_moves using shallow minimax.
    Attaches facts["minimax_score"] to each move dict.

    If MINIMAX_ENABLED=false, passes through unchanged (ablation mode).
    """
    legal = state.legal_moves

    if not legal:
        return {"last_completed_node": "minimax_scorer"}

    if not MINIMAX_ENABLED:
        # Ablation mode — attach a neutral score so ranker prompt still works
        updated = []
        for move in legal:
            m = deepcopy(move)
            m.setdefault("facts", {})
            m["facts"]["minimax_score"] = 0.0
            updated.append(m)
        return {
            "legal_moves": updated,
            "last_completed_node": "minimax_scorer",
        }

    board  = state.board
    player = state.current_player
    updated: list[dict[str, Any]] = []

    for move in legal:
        m = deepcopy(move)
        m.setdefault("facts", {})
        try:
            score = score_move_with_minimax(board, move, player, depth=MINIMAX_DEPTH)
        except Exception as e:
            # Never crash the pipeline — fall back to neutral score
            print(f"[minimax_scorer] scoring failed for move {move.get('path')}: {e}")
            score = 0.0
        m["facts"]["minimax_score"] = round(score, 2)
        updated.append(m)

    return {
        "legal_moves": updated,
        "last_completed_node": "minimax_scorer",
    }