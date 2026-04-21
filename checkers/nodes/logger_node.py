# nodes/logger_node.py
#
# Runs after win_condition each ply. Observes state, prints a turn summary,
# appends one JSONL line, and writes a final JSON summary when game_over.
# Does not set game_over, winner, draw, or board — only game_log_id + last_completed_node.

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from checkers.state.state import CheckersState
from checkers.engine.board import RED, BLACK, print_board
from checkers.engine.move_facts import count_pieces

LOG_DIR = os.environ.get("CHECKERS_LOG_DIR", "logs")
PRINT_TO_TERMINAL = os.environ.get("CHECKERS_LOGGER_PRINT", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _player_label(p: int) -> str:
    return "RED" if p == RED else "BLACK"


def _winner_line(state: CheckersState) -> str:
    if state.draw:
        return "Draw"
    if state.winner == RED:
        return "RED"
    if state.winner == BLACK:
        return "BLACK"
    return "N/A"


def _strategic_excerpt(ctx: dict[str, Any]) -> tuple[list[Any], Any, Any]:
    priorities = ctx.get("strategic_priorities")
    if priorities is None:
        priorities = []
    material = ctx.get("material_advantage")
    winning_score = ctx.get("winning_score")
    winning = f"score={winning_score}" if winning_score is not None else None
    return priorities, material, winning


def _piece_summary_line(label: str, counts: dict[str, int]) -> str:
    return (
        f"  {label}: {counts['total']} pieces "
        f"({counts['regular']} regular, {counts['kings']} kings)"
    )


def _format_move_line(move: dict[str, Any]) -> str:
    path = move.get("path") or []
    if len(path) >= 2:
        start, end = path[0], path[-1]
        seg = f"from [{start[0]},{start[1]}] to [{end[0]},{end[1]}]"
    else:
        seg = str(path)
    cap = move.get("captured") or []
    return f"{move.get('type', '?')} {seg}  captured {cap}"


def _compute_final_metrics(
    state: CheckersState,
) -> dict[str, Any]:
    mh = state.move_history
    total_captures_red = sum(
        len(r.get("move", {}).get("captured", []))
        for r in mh
        if r.get("player") == RED
    )
    total_captures_black = sum(
        len(r.get("move", {}).get("captured", []))
        for r in mh
        if r.get("player") == BLACK
    )
    total_promotions = sum(1 for r in mh if r.get("promotion", False))
    tn = state.turn_number
    fb = state.ranker_fallback_count
    rate = (fb / tn) if tn > 0 else 0.0
    return {
        "format_error_count": state.format_error_count,
        "ranker_failure_count": state.ranker_failure_count,
        "ranker_fallback_count": fb,
        "ranker_fallback_rate": rate,
        "total_captures_red": total_captures_red,
        "total_captures_black": total_captures_black,
        "total_promotions": total_promotions,
    }


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        print(f"[logger_node] warning: JSONL append failed ({path}): {e}", file=sys.stderr)


def _write_summary(path: str, payload: dict[str, Any]) -> None:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    except OSError as e:
        print(f"[logger_node] warning: summary write failed ({path}): {e}", file=sys.stderr)


def logger_node(state: CheckersState) -> dict:
    game_log_id: Optional[str] = state.game_log_id
    if game_log_id is None:
        game_log_id = "game_" + datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )

    last_entry = state.move_history[-1] if state.move_history else None
    move = (last_entry or {}).get("move") or {}
    player_who_moved = (last_entry or {}).get("player")
    reasoning = (last_entry or {}).get("last_move_reasoning")
    promotion = bool((last_entry or {}).get("promotion", False))

    ctx = state.strategic_context or {}
    priorities, material_adv, winning_asm = _strategic_excerpt(ctx)
    archive_summary = ctx.get("archive_summary")

    # ── Terminal ─────────────────────────────────────────────
    if PRINT_TO_TERMINAL:
        bar = "═" * 35
        print()
        print(bar)
        if last_entry is not None and player_who_moved is not None:
            print(f"Turn {state.turn_number} | {_player_label(int(player_who_moved))} just moved")
        else:
            print(f"Turn {state.turn_number} | (no move_history entry)")
        print(bar)
        if move:
            print(f"Move: {_format_move_line(move)}")
            print(f"Promotion: {'Yes' if promotion else 'No'}")
        if reasoning:
            print(f"Reasoning: {reasoning}")
        print("\nBoard:")
        print_board(state.board)
        rc = count_pieces(state.board, RED)
        bc = count_pieces(state.board, BLACK)
        print("Piece counts:")
        print(_piece_summary_line("RED", rc))
        print(_piece_summary_line("BLACK", bc))
        print("\nStrategic priorities:")
        if priorities:
            for i, p in enumerate(priorities, 1):
                print(f"  {i}. {p}")
        else:
            print("  (none)")
        print("\nMetrics:")
        print(
            f"  format_errors={state.format_error_count}  "
            f"ranker_failures={state.ranker_failure_count}  "
            f"fallbacks={state.ranker_fallback_count}"
        )
        if state.game_over:
            print()
            print("═" * 35)
            print("GAME OVER")
            print(f"Winner: {_winner_line(state)}")
            print(f"Total turns: {state.turn_number}")
            print("═" * 35)
        print()

    # ── JSONL (one line per ply after a move exists) ──────────
    if last_entry is not None:
        jsonl_record = {
            "turn": state.turn_number,
            "player_who_moved": player_who_moved,
            "move_type": move.get("type"),
            "path": move.get("path"),
            "captured": move.get("captured", []),
            "promotion": promotion,
            "reasoning": reasoning,
            "game_over": state.game_over,
            "winner": state.winner,
            "draw": state.draw,
            "metrics": {
                "format_error_count": state.format_error_count,
                "ranker_failure_count": state.ranker_failure_count,
                "ranker_fallback_count": state.ranker_fallback_count,
            },
            "strategic_priorities": priorities,
            "material_advantage": material_adv,
            "winning_assessment": winning_asm,
        }
        # game_log_id is already "game_YYYYMMDD_HHMMSS"; avoid "game_game_..." in path
        _append_jsonl(os.path.join(LOG_DIR, f"{game_log_id}.jsonl"), jsonl_record)

    # ── Final summary ─────────────────────────────────────────
    if state.game_over:
        summary = {
            "game_log_id": game_log_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_turns": state.turn_number,
            "winner": state.winner,
            "draw": state.draw,
            "final_board": state.board,
            "move_history": state.move_history,
            "final_metrics": _compute_final_metrics(state),
            "archive_summary": archive_summary,
        }
        _write_summary(
            os.path.join(LOG_DIR, f"summary_{game_log_id}.json"),
            summary,
        )

    return {
        "game_log_id": game_log_id,
        "last_completed_node": "logger_node",
    }
