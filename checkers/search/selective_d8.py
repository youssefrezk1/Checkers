"""
search/selective_d8.py

Selective-D8 / Promotion-Race-D10 upgrade logic, extracted from
nodes/minimax_scorer.py so the simplified pipeline (scorer_agent)
can use it without importing a legacy node file.

Public API:
    _apply_selective_d8(board, player, candidates) -> list[dict]

Configuration (ENV-overridable at process start):
    SELECTIVE_D8_ENABLED             false   — enable D8 upgrade
    SELECTIVE_D8_PIECE_THRESHOLD     14      — max pieces to trigger
    SELECTIVE_D8_GAP_THRESHOLD       30      — max D6 gap to trigger
    SELECTIVE_D8_DEPTH               8       — search depth when triggered
    SELECTIVE_D8_INCLUDE_EXACT_TIES  false   — re-read inline; see note below
    PROMOTION_RACE_VERIFY_ENABLED    true    — post-D8 D10 verification
    PROMOTION_RACE_VERIFY_DEPTH      10
    PROMOTION_RACE_VERIFY_MARGIN     15.0
    PROMOTION_RACE_PIECE_THRESHOLD   14

No imports from checkers.state, checkers.nodes, or checkers.agents.
"""
from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any

from checkers.search.minimax_core import (
    search_root_all_scores,
    get_d6_top_gap,
    clear_transposition_table,
)

# ── Selective-D8 configuration ────────────────────────────────────────────────
_d8_enabled_env = os.environ.get("SELECTIVE_D8_ENABLED", "false").lower()
SELECTIVE_D8_ENABLED         = _d8_enabled_env in ("1", "true", "yes", "on")
SELECTIVE_D8_PIECE_THRESHOLD = int(os.environ.get("SELECTIVE_D8_PIECE_THRESHOLD", "14"))
SELECTIVE_D8_GAP_THRESHOLD   = float(os.environ.get("SELECTIVE_D8_GAP_THRESHOLD", "30"))
SELECTIVE_D8_DEPTH           = int(os.environ.get("SELECTIVE_D8_DEPTH", "8"))
# SELECTIVE_D8_INCLUDE_EXACT_TIES is intentionally NOT a module-level constant.
# Tests mutate os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] after import;
# _apply_selective_d8 re-reads os.environ inline so those mutations take effect.

# ── Promotion-race verification configuration ─────────────────────────────────
_promo_verify_env = os.environ.get("PROMOTION_RACE_VERIFY_ENABLED", "true").lower()
PROMOTION_RACE_VERIFY_ENABLED  = _promo_verify_env in ("1", "true", "yes", "on")
PROMOTION_RACE_VERIFY_DEPTH    = int(os.environ.get("PROMOTION_RACE_VERIFY_DEPTH", "10"))
PROMOTION_RACE_VERIFY_MARGIN   = float(os.environ.get("PROMOTION_RACE_VERIFY_MARGIN", "15.0"))
PROMOTION_RACE_PIECE_THRESHOLD = int(os.environ.get("PROMOTION_RACE_PIECE_THRESHOLD", "14"))

_PROMOTION_CRITICAL_ROLES = {"PROMOTION_PUSH", "CONVERSION", "BLOCK_PROMOTION"}


# ── Private helpers ───────────────────────────────────────────────────────────

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
    scored: list[tuple[dict[str, Any], float]],
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


# ── Public API ────────────────────────────────────────────────────────────────

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
    scored_proxy = sorted(
        [(c, c.get("facts", {}).get("minimax_score", 0.0)) for c in candidates],
        key=lambda x: -x[1],
    )
    d6_top_gap = get_d6_top_gap([(None, sc) for _, sc in scored_proxy])

    # Exact-tie guard — re-reads os.environ so test-time mutations take effect.
    include_exact_ties = os.environ.get(
        "SELECTIVE_D8_INCLUDE_EXACT_TIES", "false"
    ).lower() in ("1", "true", "yes", "on")
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

    # ── Trigger D8 on the candidate list ─────────────────────────────────────
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
        legal_moves=candidate_move_list,
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
