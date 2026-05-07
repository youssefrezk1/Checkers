# nodes/minimax_scorer.py
#
# LangGraph node: attaches minimax scores to each shortlisted candidate move.
# Runs AFTER validator (which enriches legal_moves with facts),
# and BEFORE ranker_agent (which makes the final decision).
#
# Primary path (symbolic_scored_moves cache hit):
#   Reuses pre-computed scores from symbolic_decision (populated earlier this
#   turn) via a path-keyed lookup.  No re-search is performed.
#
# Fallback path (cache miss or incomplete cache):
#   Runs search_root_all_scores ONCE over all candidate moves at
#   MINIMAX_DEPTH (the same depth used everywhere else).  Scores and ranks
#   from that joint search replace the stale zeros.  score_move_with_minimax
#   is intentionally NOT used here — it produces non-comparable per-move
#   scores (no shared TT, no sibling context).
#
# Selective-D8 policy (SELECTIVE_D8_ENABLED=true):
#   After scoring at MINIMAX_DEPTH, if the position is in endgame
#   (pieces <= threshold) AND the top-gap is nonzero but small
#   (<= gap threshold), re-scores the SAME candidate list at
#   SELECTIVE_D8_DEPTH and replaces minimax_score / symbolic_rank.
#   Exact ties (top-gap == 0) are skipped by default to avoid wasted search.
#
# Configuration:
#   MINIMAX_ENABLED=true                  (default) — enable this node
#   MINIMAX_ENABLED=false                 — skip scoring (ablation mode)
#   PIPELINE_SCORER_DEPTH=<MINIMAX_DEPTH> fallback depth (default = MINIMAX_DEPTH)
#   SELECTIVE_D8_ENABLED=false            (default) — off; set true to enable
#   SELECTIVE_D8_PIECE_THRESHOLD=14       max total pieces to consider D8
#   SELECTIVE_D8_GAP_THRESHOLD=30         max top-gap to trigger D8
#   SELECTIVE_D8_DEPTH=8                  depth used when triggered
#   SELECTIVE_D8_INCLUDE_EXACT_TIES=false skip positions where top-gap == 0

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from checkers.state.state import CheckersState
from checkers.engine.minimax import MINIMAX_DEPTH
from checkers.search.minimax_core import (
    search_root_all_scores,
    clear_transposition_table,
)
from checkers.search.selective_d8 import (
    _apply_selective_d8,
    SELECTIVE_D8_ENABLED,
)

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

    Primary path: reuses pre-computed scores from state.symbolic_scored_moves
    (populated by symbolic_decision this turn at MINIMAX_DEPTH).

    Fallback path: when the cache is missing or incomplete, runs
    search_root_all_scores ONCE over the full candidate list at
    PIPELINE_SCORER_DEPTH.  score_move_with_minimax is NOT used — it
    produces non-comparable per-move scores.

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
    cache_hits   = 0
    cache_misses = 0

    # First pass: fill from cache, track misses
    pre_scored: list[dict[str, Any]] = []
    missed_moves: list[dict[str, Any]] = []
    for move in legal:
        m = deepcopy(move)
        m.setdefault("facts", {})
        path_key = tuple(tuple(sq) for sq in move.get("path", []))
        cached = score_lookup.get(path_key)
        if cached is not None:
            score, rank = cached
            m["facts"]["minimax_score"] = score
            m["facts"]["symbolic_rank"] = rank
            cache_hits += 1
        else:
            # Mark for joint fallback — do not score individually
            m["facts"]["minimax_score"] = 0.0
            m["facts"]["symbolic_rank"] = 0
            cache_misses += 1
            missed_moves.append(move)
        pre_scored.append(m)

    if cache_misses > 0:
        # Fallback: ONE joint search_root_all_scores call over the full candidate
        # list.  Never call score_move_with_minimax — it gives non-comparable scores.
        print(
            f"[minimax_scorer] cache_hits={cache_hits} cache_misses={cache_misses} "
            f"— running joint fallback at depth={PIPELINE_SCORER_DEPTH}"
        )
        try:
            clear_transposition_table()
            _, _, fallback_scored, _ = search_root_all_scores(
                board=board,
                current_player=player,
                depth=PIPELINE_SCORER_DEPTH,
                legal_moves=legal,   # full candidate list as root moves
                use_tt=True,
                use_tactical_extension=True,
                use_phase7a=True,
            )
            # Build lookup from the joint result
            fallback_lookup: dict[tuple, tuple[float, int]] = {
                tuple(tuple(sq) for sq in mv["path"]): (float(sc), rank + 1)
                for rank, (mv, sc) in enumerate(fallback_scored)
            }
            # Overwrite all candidates (cache hits keep their score; cache misses
            # get the joint-search score; both ranks are now globally consistent)
            updated: list[dict[str, Any]] = []
            for m in pre_scored:
                pk = tuple(tuple(sq) for sq in m.get("path", []))
                if pk in fallback_lookup:
                    fb_score, fb_rank = fallback_lookup[pk]
                    m["facts"]["minimax_score"] = round(fb_score, 2)
                    m["facts"]["symbolic_rank"] = fb_rank
                updated.append(m)
        except Exception as exc:
            print(f"[minimax_scorer] joint fallback failed: {exc} — keeping cache scores")
            updated = pre_scored
    else:
        updated = pre_scored

    # ── Selective-D8 upgrade ──────────────────────────────────────────────────
    if SELECTIVE_D8_ENABLED:
        updated = _apply_selective_d8(board, player, updated)

    return {
        "legal_moves": updated,
        "last_completed_node": "minimax_scorer",
    }


