#!/usr/bin/env python3
"""
checkers/baseline_eval/generate_hard_scenarios.py

Generate a large benchmark of HARD single-ply checkers scenarios for the
baseline scenario suite.

Pipeline (no LLM calls, fully reproducible from a seed)
-------------------------------------------------------
  1. Random self-play rollouts from the initial position.
     Each turn picks a random legal move; we snapshot every position whose
     side-to-move is RED and which still has multiple legal moves.
  2. Each candidate position is scored with score_all_legal_moves so we know
     the minimax_score, ranks, score_gap, and per-move facts.
  3. The position is classified into one or more tactical categories using
     deterministic feature rules over the scored move list.
  4. Difficulty filters are applied:
        legal_moves_count >= --min-legal
        score_gap         >= --min-gap        (best - second_best)
        category is non-empty
  5. Positions are deduplicated by a stable hash of the board+side_to_move.
  6. The first --target-count surviving positions (after shuffle for
     diversity, deterministic given the seed) are written out as JSON.

Output JSON schema (each entry)
-------------------------------
  scenario_id        str   "gen_<seed>_<index>"
  category           str   primary tactical category
  tactical_tags      list  all matching category tags
  board              list  8x8 board snapshot
  side_to_move       int   RED == 1
  legal_moves_count  int   number of legal moves for side_to_move
  best_move_index    int   index in the engine legal move list of the top move
  best_score         float minimax_score of best move
  second_best_score  float minimax_score of second-best move
  score_gap          float best_score - second_best_score
  best_move_path     list  path of the best move (for cross-check)
  generation_source  str   "self_play_rollout"
  description        str   short human-readable label

Reproducibility
---------------
Given identical --seed, --max-rollouts, --max-plies, --target-count, and the
same engine code, this script produces an identical JSON file.

Usage
-----
  venv/bin/python3 -m checkers.baseline_eval.generate_hard_scenarios \\
      --target-count 100 --max-rollouts 2000 \\
      --out logs/hard_scenarios.json --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

# Default to ablation-free scoring during generation. We don't want the
# generator's own filtering to depend on flaky symbolic add-ons; the suite
# can re-score with whatever settings it likes at evaluation time.
os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from checkers.engine.board import (
    BLACK, EMPTY, RED, RED_KING,
    create_initial_board,
)
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.agents.scorer_agent import score_all_legal_moves


# ── Categories ───────────────────────────────────────────────────────────────

CATEGORIES: tuple[str, ...] = (
    "mandatory_capture",
    "multi_jump",
    "promotion",
    "near_promotion",
    "king_move",
    "recapture_risk",
    "mobility_tradeoff",
    "safe_vs_unsafe_capture",
    "losing_position",
    "quiet_tie",
    "tactical_trap",
    "forced_reply",
    "back_row_weakness",
    "endgame",
)


def _classify(
    scored: list[dict[str, Any]],
    legal: list[dict[str, Any]],
    board: list[list[int]],
    score_gap: float,
    best_score: float,
) -> list[str]:
    """Return zero or more category tags for a scored position."""
    tags: list[str] = []
    has_jump        = any(m.get("type") == "jump" for m in legal)
    has_multi_jump  = any(len(m.get("captured") or []) > 1 for m in legal)
    n_pieces        = sum(1 for row in board for v in row if v != EMPTY)
    n_red           = sum(1 for row in board for v in row if v in (RED, RED_KING))
    has_red_king    = any(v == RED_KING for row in board for v in row)

    if has_jump:
        tags.append("mandatory_capture")
    if has_multi_jump:
        tags.append("multi_jump")

    # promotion / near-promotion
    if any((m.get("facts") or {}).get("results_in_king") for m in scored):
        tags.append("promotion")
    if any((m.get("facts") or {}).get("near_promotion") for m in scored):
        tags.append("near_promotion")

    if has_red_king:
        tags.append("king_move")

    # Recapture-risk differences: at least one move with opponent_can_recapture
    # AND at least one without — model has to discriminate.
    recap = [bool((m.get("facts") or {}).get("opponent_can_recapture")) for m in scored]
    if recap and any(recap) and not all(recap):
        tags.append("recapture_risk")

    # Safe vs unsafe capture: at least two captures with different recapture
    # status.
    captures = [m for m in scored if (m.get("type") == "jump")]
    if len(captures) >= 2:
        cap_recap = [bool((m.get("facts") or {}).get("opponent_can_recapture"))
                     for m in captures]
        if any(cap_recap) and not all(cap_recap):
            tags.append("safe_vs_unsafe_capture")

    # Mobility tradeoff: large spread in opponent_mobility_after across moves
    mobilities = [(m.get("facts") or {}).get("opponent_mobility_after")
                  for m in scored]
    mobilities_num = [v for v in mobilities if isinstance(v, (int, float))]
    if len(mobilities_num) >= 2 and (max(mobilities_num) - min(mobilities_num) >= 3):
        tags.append("mobility_tradeoff")

    # Losing position: RED is clearly behind from best move's POV.
    if best_score is not None and best_score < -20:
        tags.append("losing_position")

    # Quiet tie: many simple moves, gap is tiny.
    if not has_jump and len(scored) >= 4 and 0 <= score_gap <= 2:
        tags.append("quiet_tie")

    # Forced reply: at least one move with forced_opponent_jump_reply
    if any((m.get("facts") or {}).get("forced_opponent_jump_reply") for m in scored):
        tags.append("forced_reply")

    # Tactical trap: jumps available, but the best jump loses material vs
    # a different jump (rank-1 jump opp_can_recapture differs from another).
    if has_jump and len(captures) >= 2:
        sorted_caps = sorted(captures,
                             key=lambda m: (m.get("facts") or {}).get("minimax_score", 0),
                             reverse=True)
        if sorted_caps[0] is not sorted_caps[-1]:
            top_recap = bool((sorted_caps[0].get("facts") or {}).get("opponent_can_recapture"))
            bot_recap = bool((sorted_caps[-1].get("facts") or {}).get("opponent_can_recapture"))
            if top_recap != bot_recap:
                tags.append("tactical_trap")

    # Back-row weakness: RED has fewer than 2 pieces in row 7
    red_back_row = sum(1 for c in range(8) if board[7][c] in (RED, RED_KING))
    if red_back_row <= 1 and n_red >= 3:
        tags.append("back_row_weakness")

    # Endgame: few total pieces
    if n_pieces <= 6:
        tags.append("endgame")

    return tags


def _dedup_key(board: list[list[int]], side_to_move: int) -> str:
    flat = ",".join(str(v) for row in board for v in row)
    return hashlib.sha1(f"{side_to_move}|{flat}".encode()).hexdigest()


def _score_position_for_red(
    board: list[list[int]],
) -> tuple[list[dict[str, Any]], float, Optional[float], float]:
    """Wrapper that always scores from RED's perspective."""
    return score_all_legal_moves(board, RED, position_history=None)


def _self_play_collect(
    target_unique: int,
    max_rollouts: int,
    max_plies: int,
    rng: random.Random,
) -> list[tuple[list[list[int]], int]]:
    """
    Run random self-play rollouts and return a deduplicated pool of
    (board_snapshot, side_to_move=RED) tuples.
    """
    seen: set[str] = set()
    pool: list[tuple[list[list[int]], int]] = []

    for _ in range(max_rollouts):
        board = create_initial_board()
        player = RED
        for _ply in range(max_plies):
            legal = get_all_legal_moves(board, player)
            if not legal:
                break
            # Snapshot RED-to-move positions only.
            if player == RED and len(legal) >= 2:
                key = _dedup_key(board, player)
                if key not in seen:
                    seen.add(key)
                    pool.append(([row[:] for row in board], player))
                    if len(pool) >= target_unique:
                        return pool
            move = rng.choice(legal)
            board = apply_move(board, move)
            player = BLACK if player == RED else RED
        if len(pool) >= target_unique:
            break
    return pool


# ── Main ─────────────────────────────────────────────────────────────────────

def _build_scenario_entry(
    board: list[list[int]],
    side_to_move: int,
    seed: int,
    idx: int,
    min_gap: float,
    min_legal: int,
) -> Optional[dict[str, Any]]:
    legal = get_all_legal_moves(board, side_to_move)
    if len(legal) < min_legal:
        return None
    scored, best_score, second_best, gap = _score_position_for_red(board)
    if gap == float("inf"):
        # only one legal move at root — does not qualify as hard
        return None
    if gap < min_gap:
        return None
    tags = _classify(scored, legal, board, gap, best_score)
    if not tags:
        return None
    # Find best_move_index in the engine legal move list
    best_path = scored[0]["path"]
    best_index: Optional[int] = None
    for i, m in enumerate(legal):
        if list(m.get("path")) == list(best_path):
            best_index = i
            break
    return {
        "scenario_id":        f"gen_{seed}_{idx:04d}",
        "category":           tags[0],
        "tactical_tags":      list(tags),
        "board":              [[int(v) for v in row] for row in board],
        "side_to_move":       int(side_to_move),
        "legal_moves_count":  int(len(legal)),
        "best_move_index":    (int(best_index) if best_index is not None else None),
        "best_score":         float(best_score),
        "second_best_score":  (float(second_best) if second_best is not None else None),
        "score_gap":          float(gap) if gap != float("inf") else None,
        "best_move_path":     [[int(sq[0]), int(sq[1])] for sq in best_path],
        "generation_source":  "self_play_rollout",
        "description":        ", ".join(tags),
    }


def generate(
    target_count: int,
    max_rollouts: int,
    max_plies: int,
    seed: int,
    min_gap: float,
    min_legal: int,
) -> list[dict[str, Any]]:
    """Programmatic entry point. Returns a list of scenario entries."""
    rng = random.Random(seed)
    # Collect a generous superset so that filters can drop ~half without
    # starving the target count.
    superset_target = max(target_count * 4, 50)
    pool = _self_play_collect(
        target_unique=superset_target,
        max_rollouts=max_rollouts,
        max_plies=max_plies,
        rng=rng,
    )

    # Deterministic shuffle for category diversity.
    rng.shuffle(pool)

    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, (board, side) in enumerate(pool):
        entry = _build_scenario_entry(
            board=board,
            side_to_move=side,
            seed=seed,
            idx=idx,
            min_gap=min_gap,
            min_legal=min_legal,
        )
        if entry is None:
            continue
        if entry["scenario_id"] in seen_ids:
            continue
        seen_ids.add(entry["scenario_id"])
        entries.append(entry)
        if len(entries) >= target_count:
            break
    return entries


def save_scenarios(entries: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Generated scenarios file must be a JSON list: {p}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hard single-ply scenarios for the baseline suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target-count", type=int, default=100,
                        help="Number of scenarios to write (default: 100).")
    parser.add_argument("--max-rollouts", type=int, default=2000,
                        help="Maximum self-play rollouts before giving up.")
    parser.add_argument("--max-plies", type=int, default=80,
                        help="Maximum plies per rollout.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (controls reproducibility).")
    parser.add_argument("--min-gap", type=float, default=3.0,
                        help="Minimum score_gap for a scenario to qualify.")
    parser.add_argument("--min-legal", type=int, default=4,
                        help="Minimum legal-moves count for a scenario.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSON file path.")
    args = parser.parse_args()

    entries = generate(
        target_count=args.target_count,
        max_rollouts=args.max_rollouts,
        max_plies=args.max_plies,
        seed=args.seed,
        min_gap=args.min_gap,
        min_legal=args.min_legal,
    )
    save_scenarios(entries, args.out)

    # Category distribution summary
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    print(f"[generate] wrote {len(entries)} scenarios -> {args.out}")
    print(f"[generate] seed={args.seed} min_gap={args.min_gap} "
          f"min_legal={args.min_legal}")
    print("[generate] category distribution (primary tag):")
    for cat in sorted(counts):
        print(f"  {cat:<28} {counts[cat]}")


if __name__ == "__main__":
    main()
