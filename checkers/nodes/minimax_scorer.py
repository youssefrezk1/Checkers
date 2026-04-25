# nodes/minimax_scorer.py
#
# LangGraph node: attaches minimax scores to each shortlisted candidate move.
# Runs AFTER validator (which enriches legal_moves with facts),
# and BEFORE ranker_agent (which makes the final decision).
#
# Phase 8 behavior:
#   - Checks state.symbolic_scored_moves for pre-computed depth-3 scores
#     (from symbolic_decision node which ran earlier this turn)
#   - If a candidate move's path matches an entry in symbolic_scored_moves,
#     reuses that score and rank — avoids redundant depth-3 search
#   - If no pre-computed score is found (e.g. during tests/ablation), falls
#     back to running score_move_with_minimax at PIPELINE_SCORER_DEPTH
#   - Also attaches facts["symbolic_rank"] so ranker sees global ranking position
#
# Configuration:
#   MINIMAX_ENABLED=true   (default) — enable this node
#   MINIMAX_ENABLED=false  — skip scoring (ablation mode)
#   PIPELINE_SCORER_DEPTH=3 (default) — fallback depth if pre-computed not found

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from checkers.state.state import CheckersState
from checkers.engine.minimax import score_move_with_minimax, MINIMAX_DEPTH

# ── Configuration ─────────────────────────────────────────────────────────────
_enabled_env = os.environ.get("MINIMAX_ENABLED", "true").lower()
MINIMAX_ENABLED = _enabled_env in ("1", "true", "yes", "on")

# Fallback depth when symbolic_scored_moves cache is unavailable.
# Defaults to MINIMAX_DEPTH so all scoring paths use the same depth.
PIPELINE_SCORER_DEPTH = int(os.environ.get("PIPELINE_SCORER_DEPTH", str(MINIMAX_DEPTH)))


def _build_score_lookup(symbolic_scored_moves: list[dict]) -> dict[tuple, tuple[float, int]]:
    """
    Build a path → (minimax_score, rank) lookup from symbolic_scored_moves.
    Path key is a tuple of tuples: ((r1,c1), (r2,c2), ...).
    """
    lookup: dict[tuple, tuple[float, int]] = {}
    for entry in symbolic_scored_moves:
        raw_path = entry.get("move", {}).get("path", [])
        if raw_path:
            key = tuple(tuple(sq) for sq in raw_path)
            lookup[key] = (entry["minimax_score"], entry.get("rank", 0))
    return lookup


def minimax_scorer(state: CheckersState) -> dict:
    """
    Attaches minimax_score and symbolic_rank to each candidate in state.legal_moves.

    Primary path (Phase 8): reuses pre-computed depth-3 scores from
    state.symbolic_scored_moves (populated by symbolic_decision this turn).

    Fallback path: runs score_move_with_minimax at PIPELINE_SCORER_DEPTH
    if the move is not found in the pre-computed cache.

    If MINIMAX_ENABLED=false: attaches neutral score 0.0 (ablation mode).
    """
    legal = state.legal_moves

    if not legal:
        return {"last_completed_node": "minimax_scorer"}
    

    if not MINIMAX_ENABLED:
        # Ablation mode — attach neutral score so ranker prompt still works
        updated = []
        for move in legal:
            m = deepcopy(move)
            m.setdefault("facts", {})
            m["facts"]["minimax_score"] = 0.0
            m["facts"]["symbolic_rank"] = 0
            updated.append(m)
        return {
            "legal_moves": updated,
            "last_completed_node": "minimax_scorer",
        }

    board  = state.board
    player = state.current_player
    updated: list[dict[str, Any]] = []

    # Build lookup from symbolic_decision pre-computed scores (if available)
    score_lookup = _build_score_lookup(state.symbolic_scored_moves)
    cache_hits = 0
    cache_misses = 0

    for move in legal:
        m = deepcopy(move)
        m.setdefault("facts", {})

        path_key = tuple(tuple(sq) for sq in move.get("path", []))
        cached = score_lookup.get(path_key)

        if cached is not None:
            # Reuse depth-3 score computed by symbolic_decision — no re-search
            score, rank = cached
            m["facts"]["minimax_score"] = score
            m["facts"]["symbolic_rank"] = rank
            cache_hits += 1
        else:
            # Fallback: run the search (e.g. tests, ablation, edge cases)
            try:
                score = score_move_with_minimax(board, move, player, depth=PIPELINE_SCORER_DEPTH)
            except Exception as e:
                print(f"[minimax_scorer] scoring failed for move {move.get('path')}: {e}")
                score = 0.0
            m["facts"]["minimax_score"] = round(score, 2)
            m["facts"]["symbolic_rank"] = 0   # unknown rank on fallback
            cache_misses += 1

        updated.append(m)

    if cache_misses > 0:
        print(f"[minimax_scorer] cache_hits={cache_hits} cache_misses={cache_misses} "
              f"(fallback to depth={PIPELINE_SCORER_DEPTH} search)")

    return {
        "legal_moves": updated,
        "last_completed_node": "minimax_scorer",
    }