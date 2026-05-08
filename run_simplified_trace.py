#!/usr/bin/env python3
# run_simplified_trace.py — simplified pipeline runner (AI-vs-human)
# RED: scorer_node → deterministic_proposal_node → ranker_agent → update_agent
# BLACK: human terminal input → update_agent (AI pipeline bypassed entirely)
"""
Game trace using the simplified pipeline: RED = AI, BLACK = human.

Per-turn flow:
  RED turn  — graph runs scorer → proposal → ranker → update_agent then
              stops (interrupt_after update_agent).  The graph does NOT loop.
  BLACK turn — legal moves printed with indices; human enters a move index;
              update_agent is called directly (scorer/proposal/ranker skipped).

Key differences from run_full_trace.py:
  - USE_SIMPLIFIED_PIPELINE is forced to "true" before the graph is imported.
  - state_manager, win_condition, logger_node and inter_turn_memory are called
    inside update_agent (not as separate graph nodes).
  - Completion is detected by last_completed_node == "update_agent".

Usage:
  python run_simplified_trace.py [--max-turns N] [--quiet]
"""

from __future__ import annotations

# ── Force simplified pipeline BEFORE importing the graph ──────────────────────
import os
os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
# Suppress logger_node stdout; logs still go to logs/ via update_agent Phase C.
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import argparse
import sys
import textwrap
import uuid
from typing import Any, Optional

from checkers.graph.graph import checkers_graph
from checkers.state.state import CheckersState
from checkers.agents.update_agent import update_agent as _update_agent_fn
from checkers.engine.board import RED, BLACK, create_initial_board, print_board
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves

BAR  = "═" * 56
RULE = "─" * 56

_ANSI_RED   = "\033[91m"
_ANSI_RESET = "\033[0m"


# ── Display helpers ────────────────────────────────────────────────────────────

def _red_if(val: Any, cond: bool) -> str:
    return f"{_ANSI_RED}{val}{_ANSI_RESET}" if cond else str(val)


def _player_label(p: int) -> str:
    return "RED" if p == RED else "BLACK"


def _fmt_engine_move(i: int, m: dict[str, Any]) -> str:
    return (
        f"[{i}] type={m.get('type')} path={m.get('path')} "
        f"captured={m.get('captured', [])}"
    )


def _fmt_scored_move(i: int, m: dict[str, Any]) -> str:
    facts = m.get("facts") or {}
    score = facts.get("minimax_score", "n/a")
    rank  = facts.get("symbolic_rank", "?")
    cap   = m.get("captured") or []
    cap_s = f"  captures {cap}" if cap else ""
    return (
        f"  [{i}] rank={rank} score={score:>8}  "
        f"{m.get('type')} {m.get('path')}{cap_s}"
    )


def _fmt_applied_move(player: int, move: dict[str, Any], promotion: bool) -> str:
    path = move.get("path") or []
    cap  = move.get("captured") or []
    if len(path) >= 2:
        a, b = path[0], path[-1]
        seg = (
            f"{move.get('type', '?')} from [{a[0]},{a[1]}] to [{b[0]},{b[1]}]"
            f" captured {cap}"
        )
    else:
        seg = str(move)
    return f"{_player_label(player)} played: {seg}\nPromotion: {'Yes' if promotion else 'No'}"


def _strategic_block(ctx: Optional[dict[str, Any]]) -> None:
    print("── STRATEGIC CONTEXT ──")
    if not ctx:
        print("  (none)")
        return
    print(f"Game phase:  {ctx.get('game_phase', 'unknown')}")
    print(f"Score state: {ctx.get('score_state', 'unknown')}")
    print(f"Winning score: {ctx.get('winning_score', 'n/a')}")
    priorities = ctx.get("strategic_priorities") or []
    print("Strategic priorities:")
    if priorities:
        for i, p in enumerate(priorities, 1):
            print(f"  {i}. {p}")
    else:
        print("  (none)")
    patterns = ctx.get("active_patterns") or []
    print(f"Active patterns: {', '.join(str(x) for x in patterns) or 'none'}")
    print(f"Trends: material={ctx.get('material_trend', '?')}  center={ctx.get('center_trend', '?')}")


def _print_final_summary(state: dict[str, Any], quiet: bool) -> None:
    mh = state.get("move_history") or []
    def _winner_text() -> str:
        if state.get("draw"):  return "Draw"
        w = state.get("winner")
        if w == RED:   return "RED"
        if w == BLACK: return "BLACK"
        return "N/A"
    if quiet:
        print(BAR)
        print(f"Winner: {_winner_text()}  turns={state.get('turn_number')}")
        print(BAR)
        return
    total_cap_r = sum(
        len(r.get("move", {}).get("captured", []))
        for r in mh if r.get("player") == RED
    )
    total_cap_b = sum(
        len(r.get("move", {}).get("captured", []))
        for r in mh if r.get("player") == BLACK
    )
    total_prom  = sum(1 for r in mh if r.get("promotion", False))
    gid = state.get("game_log_id") or "(no game_log_id)"
    print(BAR)
    print("GAME OVER")
    print(BAR)
    print(f"Winner:            {_winner_text()}")
    print(f"Total turns:       {state.get('turn_number')}")
    print(f"Captures RED:      {total_cap_r}")
    print(f"Captures BLACK:    {total_cap_b}")
    print(f"Promotions:        {total_prom}")
    print("\nFinal board:")
    print_board(state["board"])
    print("\nLogs saved to:")
    print(f"  logs/{gid}.jsonl")
    print(f"  logs/summary_{gid}.json")
    print(BAR)


# ── Graph streaming ────────────────────────────────────────────────────────────

def _stream_one_ply(
    acc: dict[str, Any],
    quiet: bool,
    show_scorer: bool = True,
    recursion_limit: int = 50,
) -> tuple[dict[str, Any], bool]:
    """
    Run exactly one RED ply through the simplified graph.
    scorer_node → deterministic_proposal_node → ranker_agent → update_agent,
    then the stream is interrupted (interrupt_after=["update_agent"]) so the
    graph does NOT loop back to scorer_node for BLACK.

    Returns (updated_acc, success) where success means update_agent completed.
    """
    saw_update_agent = False
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": recursion_limit,
    }

    try:
        for chunk in checkers_graph.stream(
            acc,
            stream_mode="updates",
            interrupt_after=["update_agent"],
            config=cfg,
        ):
            for node_name, delta in chunk.items():
                if node_name in ("__interrupt__", "__end__"):
                    continue
                if not isinstance(delta, dict):
                    continue
                acc.update(delta)

                if quiet:
                    if node_name == "update_agent":
                        saw_update_agent = True
                    continue

                # ── Verbose node-by-node output ────────────────────────────

                if node_name == "scorer_node" and show_scorer:
                    lm = acc.get("legal_moves") or []
                    print(f"── SCORER NODE ({len(lm)} moves scored) ──")
                    for i, m in enumerate(lm):
                        print(_fmt_scored_move(i, m))
                    best = acc.get("symbolic_best_score")
                    gap  = acc.get("symbolic_gap")
                    print(f"best_score={best}  gap={gap}")
                    print()

                elif node_name == "deterministic_proposal_node":
                    # deterministic_proposal_node writes the shortlist to
                    # legal_moves (not proposed_moves) so ranker_agent sees it.
                    pm = acc.get("legal_moves") or []
                    print(f"── DETERMINISTIC PROPOSAL ({len(pm)} candidates) ──")
                    for i, m in enumerate(pm):
                        facts = m.get("facts") or {}
                        score = facts.get("minimax_score", "n/a")
                        rank  = facts.get("symbolic_rank", "?")
                        cap   = m.get("captured") or []
                        cap_s = f"  captures {cap}" if cap else ""
                        print(
                            f"  [{i}] rank={rank} score={score:>8}  "
                            f"{m.get('type')} {m.get('path')}{cap_s}"
                        )
                    print()


                elif node_name == "ranker_agent":
                    cm = acc.get("chosen_move")
                    lm = acc.get("legal_moves") or []
                    print("── RANKER AGENT ──")
                    if cm:
                        idx = -1
                        chosen_facts: dict = {}
                        best_score = float("-inf")
                        best_idx   = -1
                        for i, m in enumerate(lm):
                            f = m.get("facts") or {}
                            s = f.get("minimax_score", float("-inf"))
                            try:
                                s = float(s)
                            except (TypeError, ValueError):
                                s = float("-inf")
                            if s > best_score:
                                best_score = s
                                best_idx   = i
                            if m.get("type") == cm.get("type") and m.get("path") == cm.get("path"):
                                idx = i
                                chosen_facts = f
                        cap   = cm.get("captured") or []
                        cap_s = f" captures {cap}" if cap else ""
                        print(f"Chose index: {idx}")
                        print(f"Move: {cm.get('type')} {cm.get('path')}{cap_s}")
                        chosen_score = chosen_facts.get("minimax_score", float("-inf"))
                        try:
                            chosen_score = float(chosen_score)
                        except (TypeError, ValueError):
                            chosen_score = float("-inf")
                        gap_from_best = round(best_score - chosen_score, 2)
                        print(
                            f"Chosen facts: minimax_score={chosen_facts.get('minimax_score', 'n/a')}  "
                            f"net_gain={chosen_facts.get('net_gain', 'n/a')}  "
                            f"opp_recapture={chosen_facts.get('opponent_can_recapture', 'n/a')}  "
                            f"counterplay={chosen_facts.get('counterplay_score', 'n/a')}  "
                            f"king_activity={chosen_facts.get('king_activity_score', 'n/a')}"
                        )
                        _reasoning = (acc.get("last_move_reasoning") or "").strip()
                        print("Reasoning:")
                        if _reasoning:
                            # Print indented, word-wrapped at 100 chars.
                            # textwrap.fill never cuts mid-word.
                            print(textwrap.fill(
                                _reasoning,
                                width=100,
                                initial_indent="  ",
                                subsequent_indent="  ",
                                break_long_words=False,
                                break_on_hyphens=False,
                            ))
                        else:
                            print("  (none)")

                        print(f"── RANKER QUALITY ──")
                        print(f"best_legal_idx={best_idx}  best_minimax={best_score}")
                        print(f"chosen_idx={idx}  chosen_minimax={chosen_score}")
                        print(f"minimax_gap_from_best={_red_if(gap_from_best, gap_from_best > 50)}")
                    else:
                        print("Chose index: (none — ranker failure)")
                    print()

                elif node_name == "update_agent":
                    saw_update_agent = True
                    mh = acc.get("move_history") or []
                    if mh:
                        last = mh[-1]
                        pm   = last.get("player", RED)
                        mov  = last.get("move")
                        prom = bool(last.get("promotion"))
                    else:
                        pm, mov, prom = RED, None, False

                    print("── MOVE APPLIED (update_agent) ──")
                    if mov:
                        print(_fmt_applied_move(int(pm), mov, prom))
                    print("\n── BOARD AFTER MOVE ──")
                    print_board(acc["board"])
                    rc = count_pieces(acc["board"], RED)
                    bc = count_pieces(acc["board"], BLACK)
                    print("\n── PIECE COUNTS ──")
                    print(f"RED:   {rc['total']} ({rc['regular']} regular, {rc['kings']} kings)")
                    print(f"BLACK: {bc['total']} ({bc['regular']} regular, {bc['kings']} kings)")
                    print()
                    _strategic_block(acc.get("strategic_context"))
                    print()
                    print(RULE)
                    print()

    except Exception as e:
        print(f"[run_simplified_trace] graph stream error: {e}", file=sys.stderr)

    return acc, saw_update_agent


# ── RED ply (AI via simplified graph) ─────────────────────────────────────────

def _run_red_ply(acc: dict[str, Any], quiet: bool) -> dict[str, Any]:
    turn_no = acc.get("turn_number", 0)
    display_turn = turn_no + 1
    legal = get_all_legal_moves(acc["board"], RED)

    if not quiet:
        print(BAR)
        print(f"TURN {display_turn} | RED to move  (AI — simplified pipeline)")
        print(BAR)
        print("── BOARD BEFORE MOVE ──")
        print_board(acc["board"])
        print(f"\n── ENGINE LEGAL MOVES ({len(legal)} available) ──")
        for i, m in enumerate(legal):
            print(_fmt_engine_move(i, m))
        print()

    # scorer_node is the graph entry point; last_completed_node is only
    # for observability — reset to None so the first turn starts cleanly.
    acc["last_completed_node"] = None
    acc, ok = _stream_one_ply(acc, quiet, show_scorer=not quiet)

    if not ok and not quiet:
        print(
            "[run_simplified_trace] warning: graph did not complete update_agent.",
            file=sys.stderr,
        )

    return acc


# ── BLACK ply (human input, applied via update_agent) ─────────────────────────

def _run_black_ply(acc: dict[str, Any], quiet: bool) -> dict[str, Any]:
    turn_no = acc.get("turn_number", 0)
    display_turn = turn_no + 1
    board  = acc["board"]
    legal  = get_all_legal_moves(board, BLACK)

    if not legal:
        print("[run_simplified_trace] BLACK has no legal moves.", file=sys.stderr)
        return acc

    print(BAR)
    print(f"TURN {display_turn} | BLACK to move  (YOU)")
    print(BAR)
    print("── BOARD BEFORE MOVE ──")
    print_board(board)
    print()
    print(f"── YOUR AVAILABLE MOVES ({len(legal)} available) ──")
    for i, m in enumerate(legal):
        print(_fmt_engine_move(i, m))

    while True:
        try:
            raw = input(f"\nEnter move index [0-{len(legal) - 1}]: ").strip()
            k = int(raw)
            if 0 <= k < len(legal):
                break
            print(f"  Invalid — enter a number between 0 and {len(legal) - 1}.")
        except (ValueError, EOFError):
            print("  Invalid input, please enter a number.")

    move = legal[k]

    if not quiet:
        path = move.get("path") or []
        if len(path) >= 2:
            a, b = path[0], path[-1]
            print(f"\nApplied: {move.get('type')} from [{a[0]},{a[1]}] to [{b[0]},{b[1]}]")
        else:
            print(f"\nApplied: {move}")
        print(RULE)
        print()

    # Apply the human move directly via update_agent (Phase A-D).
    # Calling the graph would restart from scorer_node and overwrite chosen_move
    # with the LLM's choice.  Calling update_agent directly preserves it.
    acc["chosen_move"]         = move
    acc["last_move_reasoning"] = "BLACK human move"

    _valid = set(CheckersState.model_fields.keys())
    _state = CheckersState(**{k: v for k, v in acc.items() if k in _valid})
    _ua_result = _update_agent_fn(_state)
    acc.update(_ua_result)
    ok = _ua_result.get("last_completed_node") == "update_agent"
    if not ok:
        print(
            "[run_simplified_trace] warning: BLACK update_agent did not complete.",
            file=sys.stderr,
        )

    # ── Smoke test: human move was applied, not overwritten by AI pipeline ──
    # update_agent calls state_manager which records the applied move in
    # move_history[-1]["move"].  Verify it matches what the human chose.
    # Paths are normalized to list-of-lists so tuple/list representation
    # differences (introduced by Pydantic validation) do not cause false fails.
    def _norm_path(p: Any) -> list:
        return [list(sq) for sq in (p or [])]

    mh = acc.get("move_history") or []
    if mh:
        applied = mh[-1].get("move") or {}
        if (
            applied.get("type") != move.get("type")
            or _norm_path(applied.get("path")) != _norm_path(move.get("path"))
        ):
            print(
                "[SMOKE TEST FAIL] BLACK applied move does not match human's choice!\n"
                f"  chosen : {move.get('type')} {move.get('path')}\n"
                f"  applied: {applied.get('type')} {applied.get('path')}",
                file=sys.stderr,
            )
        elif not quiet:
            print(
                f"[SMOKE TEST OK] BLACK move applied correctly: "
                f"{move.get('type')} {move.get('path')}"
            )

    return acc


# ── Main game loop ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplified pipeline trace: RED=AI graph, BLACK=human."
    )
    parser.add_argument(
        "--max-turns", type=int, default=200,
        help="Safety cap on plies (half-moves).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print the final summary.",
    )
    args = parser.parse_args()

    if not args.quiet:
        print(BAR)
        print(
            f"SIMPLIFIED PIPELINE | MINIMAX_ENABLED={os.environ.get('MINIMAX_ENABLED', 'unset')} "
            f"| MINIMAX_DEPTH={os.environ.get('MINIMAX_DEPTH', 'unset')} "
            f"| USE_SIMPLIFIED_PIPELINE={os.environ.get('USE_SIMPLIFIED_PIPELINE')}"
        )
        print(BAR)

    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()

    while True:
        if acc.get("game_over"):
            _print_final_summary(acc, quiet=args.quiet)
            return

        if (acc.get("turn_number") or 0) >= args.max_turns:
            print("GAME INCOMPLETE: max turns reached", file=sys.stderr)
            if not args.quiet:
                print(BAR)
                print("GAME INCOMPLETE: max turns reached")
                print(BAR)
                print("\nFinal board:")
                print_board(acc["board"])
                gid = acc.get("game_log_id") or "?"
                print(f"\nLogs (if any): logs/{gid}.jsonl")
            return

        if acc["current_player"] == RED:
            acc = _run_red_ply(acc, args.quiet)
        else:
            acc = _run_black_ply(acc, args.quiet)


if __name__ == "__main__":
    main()
