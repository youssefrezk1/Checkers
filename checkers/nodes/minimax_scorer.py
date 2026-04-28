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
import time
from copy import deepcopy
from typing import Any

from checkers.state.state import CheckersState
from checkers.engine.minimax import score_move_with_minimax, MINIMAX_DEPTH
from checkers.search.minimax_core import (
    search_root_all_scores,
    get_d6_top_gap,
    clear_transposition_table,
)

# ── Configuration ─────────────────────────────────────────────────────────────
_enabled_env = os.environ.get("MINIMAX_ENABLED", "true").lower()
MINIMAX_ENABLED = _enabled_env in ("1", "true", "yes", "on")

# Fallback depth when symbolic_scored_moves cache is unavailable.
# Defaults to MINIMAX_DEPTH so all scoring paths use the same depth.
PIPELINE_SCORER_DEPTH = int(os.environ.get("PIPELINE_SCORER_DEPTH", str(MINIMAX_DEPTH)))

# Selective-D8 policy
_d8_enabled_env = os.environ.get("SELECTIVE_D8_ENABLED", "false").lower()
SELECTIVE_D8_ENABLED         = _d8_enabled_env in ("1", "true", "yes", "on")
SELECTIVE_D8_PIECE_THRESHOLD = int(os.environ.get("SELECTIVE_D8_PIECE_THRESHOLD", "14"))
SELECTIVE_D8_GAP_THRESHOLD   = float(os.environ.get("SELECTIVE_D8_GAP_THRESHOLD", "30"))
SELECTIVE_D8_DEPTH           = int(os.environ.get("SELECTIVE_D8_DEPTH", "8"))
_ties_env = os.environ.get("SELECTIVE_D8_INCLUDE_EXACT_TIES", "false").lower()
SELECTIVE_D8_INCLUDE_EXACT_TIES = _ties_env in ("1", "true", "yes", "on")

# Promotion-race stability verification (post-D8, candidate-set only)
_promo_verify_env = os.environ.get("PROMOTION_RACE_VERIFY_ENABLED", "true").lower()
PROMOTION_RACE_VERIFY_ENABLED = _promo_verify_env in ("1", "true", "yes", "on")
PROMOTION_RACE_VERIFY_DEPTH = int(os.environ.get("PROMOTION_RACE_VERIFY_DEPTH", "10"))
PROMOTION_RACE_VERIFY_MARGIN = float(os.environ.get("PROMOTION_RACE_VERIFY_MARGIN", "15.0"))
PROMOTION_RACE_PIECE_THRESHOLD = int(os.environ.get("PROMOTION_RACE_PIECE_THRESHOLD", "14"))


_PROMOTION_CRITICAL_ROLES = {"PROMOTION_PUSH", "CONVERSION", "BLOCK_PROMOTION"}


def _path_key(path: list[Any]) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _is_terminal_like_score(score: float | None) -> bool:
    return score is None or abs(float(score)) >= 9000.0


def _is_promotion_conversion_critical(cand: dict[str, Any]) -> bool:
    facts = cand.get("facts", {}) or {}
    if facts.get("near_promotion", False):
        return True
    if facts.get("results_in_king", False):
        return True
    if facts.get("opponent_near_promotion", False):
        return True
    return facts.get("quiet_move_role") in _PROMOTION_CRITICAL_ROLES


def _build_scored_lookup(
    scored: list[tuple[dict[str, Any], float]]
) -> dict[tuple, tuple[float, int]]:
    lookup: dict[tuple, tuple[float, int]] = {}
    for rank, (mv, sc) in enumerate(scored):
        lookup[_path_key(mv["path"])] = (float(sc), rank + 1)
    return lookup


def _apply_scored_lookup_to_candidates(
    candidates: list[dict[str, Any]],
    scored_lookup: dict[tuple, tuple[float, int]],
) -> list[dict[str, Any]]:
    upgraded: list[dict[str, Any]] = []
    for cand in candidates:
        c = deepcopy(cand)
        pk = _path_key(c.get("path", []))
        if pk in scored_lookup:
            score, rank = scored_lookup[pk]
            c.setdefault("facts", {})
            c["facts"]["minimax_score"] = round(score, 2)
            c["facts"]["symbolic_rank"] = rank
        upgraded.append(c)
    return upgraded


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


def _apply_selective_d8(
    board: list[list[Any]],
    player: int,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Selective-D8 upgrade.

    Re-scores `candidates` with search_root_all_scores at SELECTIVE_D8_DEPTH
    when ALL of the following hold:
      1. total pieces on board <= SELECTIVE_D8_PIECE_THRESHOLD   (endgame)
      2. D6 top-gap is strictly positive (not an exact tie) OR
         SELECTIVE_D8_INCLUDE_EXACT_TIES is true
      3. D6 top-gap <= SELECTIVE_D8_GAP_THRESHOLD                (D6 uncertain)

    On trigger: replaces minimax_score and symbolic_rank for every candidate
    with the D8 scores.  The candidate list (paths) is unchanged — only scores.
    Returns the original list unmodified if the trigger conditions are not met.
    """
    # ── piece count ───────────────────────────────────────────────────────────
    total_pieces = sum(
        1 for r in range(8) for c in range(8) if board[r][c] != 0
    )
    if total_pieces > SELECTIVE_D8_PIECE_THRESHOLD:
        print(
            f"[SELECTIVE_D8] skipped: pieces={total_pieces} > threshold={SELECTIVE_D8_PIECE_THRESHOLD}"
        )
        return candidates

    # ── D6 top-gap from current minimax_scores on candidates ─────────────────
    # Reconstruct a scored list compatible with get_d6_top_gap:
    # list of (move_dict, score) sorted descending.
    scored_proxy = sorted(
        [(c, c.get("facts", {}).get("minimax_score", 0.0)) for c in candidates],
        key=lambda x: -x[1],
    )
    d6_top_gap = get_d6_top_gap([(None, sc) for _, sc in scored_proxy])

    # Exact-tie guard
    include_exact_ties = os.environ.get("SELECTIVE_D8_INCLUDE_EXACT_TIES", "false").lower() in ("1", "true", "yes", "on")
    if d6_top_gap == 0.0 and not include_exact_ties:
        print(
            f"[SELECTIVE_D8] skipped_exact_tie: pieces={total_pieces} d6_top_gap=0.0"
        )
        return candidates

    if d6_top_gap > SELECTIVE_D8_GAP_THRESHOLD:
        print(
            f"[SELECTIVE_D8] skipped: pieces={total_pieces} d6_top_gap={d6_top_gap:.1f}"
            f" > threshold={SELECTIVE_D8_GAP_THRESHOLD}"
        )
        return candidates

    # ── Trigger D8 on the candidate list, not all legal moves ────────────────
    # Build a move-dict list from candidates so search_root_all_scores treats
    # exactly those paths as the root moves (respects DEBUG_ALL_LEGAL / proposal
    # filtering that already happened upstream).
    candidate_move_list = [
        {"type": c.get("type", "simple"), "path": c["path"],
         "captured": c.get("captured", [])}
        for c in candidates
    ]

    clear_transposition_table()
    t0 = time.perf_counter()
    d8_best_move, d8_best_score, d8_scored, d8_stats = search_root_all_scores(
        board=board,
        current_player=player,
        depth=SELECTIVE_D8_DEPTH,
        legal_moves=candidate_move_list,   # scored over candidates, not all legal
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    nodes = getattr(d8_stats, "nodes", None)

    print(
        f"[SELECTIVE_D8] triggered: pieces={total_pieces} d6_top_gap={d6_top_gap:.1f}"
        f" depth={SELECTIVE_D8_DEPTH} elapsed_ms={elapsed_ms} nodes={nodes}"
        f" d8_best={d8_best_move['path'] if d8_best_move else None}"
        f" d8_best_score={round(d8_best_score, 2) if d8_best_score is not None else None}"
    )

    d8_lookup = _build_scored_lookup(d8_scored)
    upgraded = _apply_scored_lookup_to_candidates(candidates, d8_lookup)

    # ── Promotion-race stability verification (optional) ──────────────────────
    if (
        PROMOTION_RACE_VERIFY_ENABLED
        and total_pieces <= PROMOTION_RACE_PIECE_THRESHOLD
        and len(scored_proxy) >= 2
        and d8_best_move is not None
    ):
        d6_best_cand, d6_best_score = scored_proxy[0]
        d6_best_key = _path_key(d6_best_cand["path"])
        d8_best_key = _path_key(d8_best_move["path"])
        d8_score_of_d6_best = d8_lookup.get(d6_best_key, (None, 0))[0]
        d8_downgrade_gap = (
            float(d8_best_score - d8_score_of_d6_best)
            if d8_score_of_d6_best is not None and d8_best_score is not None
            else float("-inf")
        )
        verify_triggered = (
            d6_best_key != d8_best_key
            and _is_promotion_conversion_critical(d6_best_cand)
            and d8_downgrade_gap >= PROMOTION_RACE_VERIFY_MARGIN
            and not _is_terminal_like_score(float(d6_best_score))
            and not _is_terminal_like_score(float(d8_best_score))
            and not _is_terminal_like_score(d8_score_of_d6_best)
        )
        if verify_triggered:
            clear_transposition_table()
            _, d10_best_score, d10_scored, _ = search_root_all_scores(
                board=board,
                current_player=player,
                depth=PROMOTION_RACE_VERIFY_DEPTH,
                legal_moves=candidate_move_list,
                use_tt=True,
                use_tactical_extension=True,
                use_phase7a=True,
            )
            d10_lookup = _build_scored_lookup(d10_scored)
            d10_best_move = d10_scored[0][0] if d10_scored else None
            d10_best_key = _path_key(d10_best_move["path"]) if d10_best_move is not None else None
            d10_score_of_d8_best = d10_lookup.get(d8_best_key, (None, 0))[0]
            d10_prefer_gap = (
                float(d10_best_score - d10_score_of_d8_best)
                if d10_score_of_d8_best is not None and d10_best_score is not None
                else float("-inf")
            )
            use_d10 = (
                d10_best_key is not None
                and d10_best_key != d8_best_key
                and d10_best_key == d6_best_key
                and d10_prefer_gap >= PROMOTION_RACE_VERIFY_MARGIN
                and not _is_terminal_like_score(float(d10_best_score))
                and not _is_terminal_like_score(d10_score_of_d8_best)
            )
            decision = "use_d10" if use_d10 else "use_d8"
            print(
                "[PROMOTION_RACE_VERIFY] triggered: "
                f"d6_best={d6_best_cand['path']} d8_best={d8_best_move['path']} "
                f"d10_best={d10_best_move['path'] if d10_best_move else None} "
                f"decision={decision} gap={d8_downgrade_gap:.1f}"
            )
            if use_d10:
                upgraded = _apply_scored_lookup_to_candidates(candidates, d10_lookup)

    return upgraded