"""
extract_known_failure_positions.py

Extracts board-before-RED-turn positions from the known-failure trace and
scores each position at depth 6 using the current engine. Writes a
diagnostic benchmark JSON to logs/known_failure_positions_20260425_144451.json.

Usage:
    venv/bin/python3 extract_known_failure_positions.py

Expected runtime: 2–5 min depending on machine (11 positions × D6 search).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.evaluation import _caged_king_count, _is_king_caged
from checkers.search.minimax_core import search_root_all_scores, clear_transposition_table

TRACE_PATH = "logs/game_20260425_144451_493544.jsonl"
OUT_PATH = "logs/known_failure_positions_20260425_144451.json"
TARGET_RED_TURNS = [35, 37, 41, 43, 45, 49, 51, 55, 59, 61, 63]
DEPTH = 6

PIECE_SYM = {0: ".", RED: "r", BLACK: "b", RED_KING: "R", BLACK_KING: "B"}


def _make_start() -> list[list[int]]:
    b = [[0] * 8 for _ in range(8)]
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = BLACK
    return b


def _board_to_list(b: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in b]


def _board_repr(b: list[list[int]]) -> str:
    return "\n".join(
        f"  {r}: " + " ".join(PIECE_SYM.get(b[r][c], "?") for c in range(8))
        for r in range(8)
    )


def _total_pieces(b: list[list[int]]) -> int:
    return sum(1 for r in range(8) for c in range(8) if b[r][c] != 0)


def _tag_position(board: list[list[int]], player: int, chosen_move: dict) -> list[str]:
    tags: list[str] = []
    path = chosen_move.get("path", [])
    # Promotion
    if chosen_move.get("promotion") or (
        player == RED and path and path[-1][0] == 0
    ) or (
        player == BLACK and path and path[-1][0] == 7
    ):
        tags.append("promotion_race")
    # Caged king at root
    if _caged_king_count(board, player) > 0:
        tags.append("caged_king")
    # Delayed cage: move leads to caged king after one ply
    if not _caged_king_count(board, player):
        after = apply_move(board, chosen_move)
        if _caged_king_count(after, player) > 0:
            tags.append("delayed_cage")
    # King activation: king is moved
    king = RED_KING if player == RED else BLACK_KING
    if path and board[path[0][0]][path[0][1]] == king:
        tags.append("king_activation")
    # Endgame
    if _total_pieces(board) <= 12:
        tags.append("endgame_conversion")
    # Tactical (jump)
    if chosen_move.get("type") == "jump" or chosen_move.get("move_type") == "jump":
        tags.append("tactical")
    return tags


def _path_key(path) -> str:
    return "→".join(f"({r},{c})" for r, c in path)


def main():
    # ── Load trace ────────────────────────────────────────────────────────────
    with open(TRACE_PATH) as f:
        records = [json.loads(line) for line in f]
    rec_by_turn: dict[int, dict] = {r["turn"]: r for r in records}

    # ── Replay board ──────────────────────────────────────────────────────────
    board = _make_start()
    boards: dict[int, list[list[int]]] = {0: _board_to_list(board)}
    for rec in records:
        t = rec["turn"]
        move = {
            "type": rec["move_type"],
            "path": rec["path"],
            "captured": rec.get("captured", []),
        }
        board = apply_move(board, move)
        boards[t] = _board_to_list(board)

    # ── Process each target turn ───────────────────────────────────────────────
    positions = []
    print(f"{'Turn':>4}  {'Old chosen':>20}  {'D6 best':>20}  {'Gap':>7}  Tags")
    print("-" * 75)

    for turn in TARGET_RED_TURNS:
        board_before = boards[turn - 1]  # board before this RED turn
        player = RED

        rec = rec_by_turn.get(turn, {})
        old_path = rec.get("path", [])
        old_move_type = rec.get("move_type", "simple")
        old_chosen = {
            "type": old_move_type,
            "path": old_path,
            "captured": rec.get("captured", []),
            "promotion": rec.get("promotion", False),
        }

        # Parse old winning_assessment score
        wa = rec.get("winning_assessment", "")
        try:
            old_score_str = wa.replace("score=", "").strip()
            old_logged_score = float(old_score_str) if old_score_str else None
        except ValueError:
            old_logged_score = None

        legal = get_all_legal_moves(board_before, player)

        # ── D6 search ──────────────────────────────────────────────────────
        t0 = time.time()
        clear_transposition_table()
        _, _, scored, _ = search_root_all_scores(
            board=board_before,
            current_player=player,
            depth=DEPTH,
            legal_moves=legal,
            use_tt=True,
            use_tactical_extension=True,
            use_phase7a=True,
        )
        elapsed = time.time() - t0
        scored.sort(key=lambda x: x[1], reverse=True)

        best_move, best_score = scored[0] if scored else (None, None)

        # Score of old chosen move
        old_chosen_d6 = next(
            (sc for mv, sc in scored if mv["path"] == [list(sq) for sq in old_path]
             or mv["path"] == [sq for sq in old_path]),
            None,
        )
        # Fallback: path comparison normalising tuples/lists
        if old_chosen_d6 is None:
            op_key = tuple(tuple(sq) for sq in old_path)
            old_chosen_d6 = next(
                (sc for mv, sc in scored
                 if tuple(tuple(sq) for sq in mv["path"]) == op_key),
                None,
            )

        gap = (
            round(best_score - old_chosen_d6, 1)
            if best_score is not None and old_chosen_d6 is not None
            else None
        )

        tags = _tag_position(board_before, player, old_chosen)

        # Score table
        score_table = [
            {
                "path": mv["path"],
                "type": mv["type"],
                "score": round(sc, 2),
                "is_old_chosen": tuple(tuple(sq) for sq in mv["path"])
                == tuple(tuple(sq) for sq in old_path),
                "is_d6_best": i == 0,
                "caged_after": _caged_king_count(apply_move(board_before, mv), player),
            }
            for i, (mv, sc) in enumerate(scored)
        ]

        entry = {
            "source_trace": TRACE_PATH,
            "turn": turn,
            "board": _board_to_list(board_before),
            "side_to_move": "RED",
            "total_pieces": _total_pieces(board_before),
            "legal_move_count": len(legal),
            "old_chosen_move": old_chosen,
            "old_logged_score": old_logged_score,
            "old_chosen_d6_score": round(old_chosen_d6, 2) if old_chosen_d6 is not None else None,
            "d6_best_move": best_move["path"] if best_move else None,
            "d6_best_score": round(best_score, 2) if best_score is not None else None,
            "gap_d6_best_vs_old_chosen": gap,
            "d6_score_table": score_table,
            "tags": tags,
            "search_elapsed_s": round(elapsed, 2),
        }
        positions.append(entry)

        old_str = _path_key(old_path) if old_path else "?"
        best_str = _path_key(best_move["path"]) if best_move else "?"
        gap_str = f"{gap:+.1f}" if gap is not None else "  n/a"
        tag_str = ",".join(tags) if tags else "-"
        print(f"{turn:>4}  {old_str:>20}  {best_str:>20}  {gap_str:>7}  {tag_str}")

    # ── Write output ──────────────────────────────────────────────────────────
    out = {
        "meta": {
            "source_trace": TRACE_PATH,
            "depth": DEPTH,
            "n_positions": len(positions),
            "target_turns": TARGET_RED_TURNS,
            "description": (
                "Known-failure endgame positions from game_20260425_144451_493544. "
                "All positions are board-before-RED-turn snapshots extracted by "
                "replaying the full trace. D6 scores use current engine "
                "(caged_king evaluator term active)."
            ),
        },
        "positions": positions,
    }

    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(positions)} positions → {OUT_PATH}")


if __name__ == "__main__":
    main()
