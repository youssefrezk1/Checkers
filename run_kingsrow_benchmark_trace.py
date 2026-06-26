#!/usr/bin/env python3
# run_kingsrow_benchmark_trace.py — simplified pipeline runner formatted for benchmarking
"""
Game trace using the simplified pipeline: RED = AI, BLACK = human.
Focuses terminal presentation on move-selection quality and diagnostics,
suppressing reasoning-heavy node logs while preserving exact pipeline execution.
"""

from __future__ import annotations

import os
# Force simplified pipeline configuration
os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
# Suppress logger_node stdout; logs still save via updater_agent logs/ integration
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import argparse
import sys
import uuid
import json
import time
import copy
from typing import Any, Optional

from checkers.graph.graph import checkers_graph
from checkers.state.state import CheckersState
from checkers.agents.updater_agent import updater_agent as _update_agent_fn
from checkers.engine.board import RED, BLACK, create_initial_board, print_board
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves, apply_move


def _fmt_score(score: Any) -> str:
    try:
        val = float(score)
        if val == 0.0:
            return "0.00"
        return f"{val:+.2f}"
    except (ValueError, TypeError):
        return str(score)


def _stream_one_ply(
    acc: dict[str, Any],
    quiet: bool,
    board_before: list[list[int]],
    display_turn: int,
    kr_engine: Any = None,
    kr_time: float = 1.0,
    kr_depth: int = 6,
    recursion_limit: int = 50,
) -> tuple[dict[str, Any], bool]:
    """
    Run exactly one RED ply through the simplified graph, formatting terminal
    outputs specifically for benchmark analysis when the turn completes.
    """
    saw_update_agent = False
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": recursion_limit,
    }

    # Captured data to persist diagnostics before updater_agent clears them
    captured_legal_moves = []
    captured_chosen_move = {}
    captured_chosen_score = 0.0

    try:
        for chunk in checkers_graph.stream(
            acc,
            stream_mode="updates",
            interrupt_after=["updater_agent"],
            config=cfg,
        ):
            for node_name, delta in chunk.items():
                if node_name in ("__interrupt__", "__end__"):
                    continue
                if not isinstance(delta, dict):
                    continue
                
                # Cache diagnostics from proposal node before state transitions clear it
                if node_name == "proposer_agent":
                    captured_legal_moves = delta.get("legal_moves") or []
                    captured_chosen_move = delta.get("chosen_move") or {}
                    captured_chosen_score = delta.get("chosen_move_score") or 0.0
                
                acc.update(delta)

                if node_name == "updater_agent":
                    saw_update_agent = True
                    
                    if quiet:
                        continue

                    # Extract diagnostics from cached values
                    legal_moves = captured_legal_moves
                    chosen_move = captured_chosen_move
                    chosen_path = chosen_move.get("path")
                    
                    # Compute ranks and gaps
                    chosen_idx = -1
                    for idx, m in enumerate(legal_moves):
                        if m.get("path") == chosen_path:
                            chosen_idx = idx
                            break

                    best_move_match = "Yes" if chosen_idx == 0 else "No"
                    inside_top_3 = "Yes" if (0 <= chosen_idx < 3) else "No"
                    chosen_rank = f"#{chosen_idx + 1}" if chosen_idx >= 0 else "N/A"
                    
                    if legal_moves and chosen_idx >= 0:
                        best_score = legal_moves[0]["facts"].get("minimax_score", 0.0)
                        chosen_score = chosen_move.get("facts", {}).get("minimax_score", 0.0)
                        gap = abs(best_score - chosen_score)
                        gap_str = f"{gap:.2f}"
                    else:
                        gap_str = "0.00"

                    # 1. Output turn headers
                    print("=" * 50)
                    print(f"TURN {display_turn} - RED")
                    print("=" * 50)

                    # 2. Output Board state before the move
                    print("\nBOARD BEFORE RED MOVE:")
                    print_board(board_before)

                    # 3. Output Best Move Analysis table
                    print("\nBEST MOVE ANALYSIS")
                    print("--------------------------------------------------")
                    print(f"{'Rank':<6}{'Move Path':<24}{'Score':<10}")
                    for idx, m in enumerate(legal_moves):
                        rank_str = f"#{idx + 1}"
                        path_str = str(m.get("path", []))
                        score_val = m["facts"].get("minimax_score", 0.0)
                        score_str = _fmt_score(score_val)
                        suffix = "   <- BEST" if idx == 0 else ""
                        if m.get("path") == chosen_path:
                            suffix += " (Chosen)"
                        print(f"{rank_str:<6}{path_str:<24}{score_str:<10}{suffix}")
                    
                    # 4. Output Chosen Move details
                    print("\nChosen Move:")
                    print(f"  path:  {chosen_path}")
                    print(f"  type:  {chosen_move.get('type')}")
                    print(f"  score: {_fmt_score(captured_chosen_score)}")

                    # 5. Output Result diagnostics
                    print("\nRESULT:")
                    print(f"  best_move_match:      {best_move_match}")
                    print(f"  inside_top_3:         {inside_top_3}")
                    print(f"  chosen_rank:          {chosen_rank}")
                    print(f"  score_gap_from_best:  {gap_str}")

                    # 6. KingsRow Comparison
                    kr_match = "N/A"
                    kr_top3_match = "N/A"
                    kr_score_gap = "N/A"
                    kr_rank = "N/A"
                    kr_result = None
                    
                    if kr_engine and legal_moves:
                        print(f"\nQuerying KingsRow (evaluating {len(legal_moves)} legal moves at target depth {kr_depth}, {kr_time}s budget each)...")
                        kr_evals = []
                        for lm in legal_moves:
                            test_board = [row[:] for row in board_before]
                            # Use lm directly as it is a move dict containing 'type', 'path', 'captured'
                            apply_move(test_board, lm)
                            # Evaluate the board from BLACK's perspective (since RED just moved)
                            kr_after = kr_engine.get_best_move(test_board, BLACK, kr_time, kr_depth)
                            # The score for RED is the negative of BLACK's score
                            score_for_red = -kr_after["score"]
                            kr_evals.append({
                                "move": lm,
                                "kr_score": score_for_red,
                                "kr_depth": kr_after["depth"]
                            })
                            
                        # Sort descending: highest score is best for RED
                        kr_evals.sort(key=lambda x: x["kr_score"], reverse=True)
                        kr_best_score = kr_evals[0]["kr_score"]
                        
                        our_kr_score = 0.0
                        kr_match_rank = -1
                        for i, e in enumerate(kr_evals):
                            if e["move"]["path"] == chosen_path:
                                kr_match_rank = i + 1
                                our_kr_score = e["kr_score"]
                                break
                                
                        if kr_match_rank > 0:
                            kr_match = "Yes" if kr_match_rank == 1 else "No"
                            kr_top3_match = "Yes" if kr_match_rank <= 3 else "No"
                            kr_rank = f"#{kr_match_rank}"
                            kr_gap = abs(kr_best_score - our_kr_score)
                            kr_score_gap = f"{kr_gap:.2f}"
                        else:
                            kr_gap = 0.0
                            
                        # Log to JSONL
                        log_record = {
                            "turn": display_turn,
                            "our_move": {"path": chosen_path, "type": chosen_move.get("type"), "score": captured_chosen_score},
                            "our_rank": chosen_idx + 1 if chosen_idx >= 0 else None,
                            "kr_best_move": kr_evals[0],
                            "kr_top_moves": kr_evals[:3],
                            "kr_match": kr_match == "Yes",
                            "kr_top3_match": kr_top3_match == "Yes",
                            "kr_rank": kr_match_rank,
                            "kr_score_gap": kr_gap,
                            "kr_time": kr_time,
                            "kingsrow_depth": kr_depth
                        }
                        gid = acc.get("game_log_id") or "unknown"
                        import os
                        os.makedirs("logs", exist_ok=True)
                        with open(f"logs/benchmark_{gid}.jsonl", "a") as f:
                            f.write(json.dumps(log_record) + "\n")
                            
                        print("\nKINGSROW COMPARISON")
                        print("-" * 50)
                        print(f"  KingsRow depth used: {kr_depth}")
                        for i, e in enumerate(kr_evals[:3]):
                            print(f"  KR #{i+1}: path {e['move']['path']} (Score: {e['kr_score']}, Depth: {e['kr_depth']})")
                        print()
                        print(f"  Our move match:      {kr_match}")
                        print(f"  Our move Top-3:      {kr_top3_match}")
                        print(f"  Our move KR rank:    {kr_rank}")
                        print(f"  Score gap (KR diff): {kr_score_gap}")
                    else:
                        print("\nFuture KingsRow fields (placeholder for later integration):")
                        print("  kingsrow_best_move:   null")
                        print("  kingsrow_match:       null")
                        print("  kingsrow_top3_match:  null")
                    print()
                    
                    # 7. Output board and stats after move execution
                    print("BOARD AFTER RED MOVE:")
                    print_board(acc["board"])
                    rc = count_pieces(acc["board"], RED)
                    bc = count_pieces(acc["board"], BLACK)
                    print("Laying piece counts...")
                    print(f"RED:   {rc['total']} ({rc['regular']} regular, {rc['kings']} kings)")
                    print(f"BLACK: {bc['total']} ({bc['regular']} regular, {bc['kings']} kings)")
                    print()
                    print("-" * 50)
                    print()

    except Exception as e:
        print(f"[run_kingsrow_benchmark_trace] graph stream error: {e}", file=sys.stderr)

    return acc, saw_update_agent


def _run_red_ply(acc: dict[str, Any], quiet: bool, kr_engine: Any = None, kr_time: float = 1.0, kr_depth: int = 6) -> dict[str, Any]:
    turn_no = acc.get("turn_number", 0)
    display_turn = turn_no + 1
    
    # Capture the board representation before the move is applied
    board_before = [row[:] for row in acc["board"]]

    acc["last_completed_node"] = None
    acc, ok = _stream_one_ply(acc, quiet, board_before, display_turn, kr_engine, kr_time, kr_depth)

    if not ok and not quiet:
        print(
            "[run_kingsrow_benchmark_trace] warning: graph did not complete updater_agent.",
            file=sys.stderr,
        )

    return acc


def _run_black_ply(acc: dict[str, Any], quiet: bool) -> dict[str, Any]:
    turn_no = acc.get("turn_number", 0)
    display_turn = turn_no + 1
    board  = acc["board"]
    legal  = get_all_legal_moves(board, BLACK)

    if not legal:
        print("[run_kingsrow_benchmark_trace] BLACK has no legal moves.", file=sys.stderr)
        return acc

    print("=" * 50)
    print(f"TURN {display_turn} - BLACK (YOU)")
    print("=" * 50)
    print("\nBOARD BEFORE MOVE:")
    print_board(board)
    print()
    print(f"--- YOUR AVAILABLE MOVES ({len(legal)} available) ---")
    
    for i, m in enumerate(legal):
        path = m.get("path") or []
        if len(path) >= 2:
            a, b = path[0], path[-1]
            print(f"[{i}] type={m.get('type')} from [{a[0]},{a[1]}] to [{b[0]},{b[1]}]")
        else:
            print(f"[{i}] {m}")

    while True:
        try:
            raw = input(f"\nEnter move index [0-{len(legal) - 1}]: ").strip()
            k = int(raw)
            if 0 <= k < len(legal):
                break
            print(f"  Invalid - enter a number between 0 and {len(legal) - 1}.")
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
        print()

    # Apply the human move directly via updater_agent
    acc["chosen_move"]         = move
    acc["last_move_reasoning"] = "BLACK human move"

    _valid = set(CheckersState.model_fields.keys())
    _state = CheckersState(**{k: v for k, v in acc.items() if k in _valid})
    _ua_result = _update_agent_fn(_state)
    acc.update(_ua_result)
    ok = _ua_result.get("last_completed_node") == "updater_agent"
    if not ok:
        print(
            "[run_kingsrow_benchmark_trace] warning: BLACK updater_agent did not complete.",
            file=sys.stderr,
        )

    # Smoke test validation
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
                f"{move.get('type')} {move.get('path')}\n"
            )

    return acc


def _print_final_summary(state: dict[str, Any], quiet: bool) -> None:
    mh = state.get("move_history") or []
    def _winner_text() -> str:
        if state.get("draw"):  return "Draw"
        w = state.get("winner")
        if w == RED:   return "RED"
        if w == BLACK: return "BLACK"
        return "N/A"
    
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
    
    print("=" * 50)
    print("GAME OVER SUMMARY")
    print("=" * 50)
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
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark trace: RED=AI, BLACK=human. Focuses on move evaluation quality."
    )
    parser.add_argument(
        "--max-turns", type=int, default=200,
        help="Safety cap on plies (half-moves).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print the final summary.",
    )
    parser.add_argument(
        "--kr-disable", action="store_true",
        help="Skip KingsRow engine evaluation.",
    )
    parser.add_argument(
        "--kr-time", type=float, default=1.0,
        help="Time budget in seconds for KingsRow per search (default 1.0).",
    )
    parser.add_argument(
        "--kingsrow-depth", type=int, default=6,
        help="Fixed depth limit for KingsRow to match our engine's search horizon (default 6).",
    )
    args = parser.parse_args()

    kr_engine = None
    if not args.kr_disable:
        dll_path = os.environ.get("KINGSROW_DLL_PATH", r"C:\Program Files (x86)\CheckerBoard\engines\Kingsrow64.dll")
        if os.path.exists(dll_path):
            from checkers.engine.kingsrow_interface import KingsRowEngine
            kr_engine = KingsRowEngine(dll_path)
            if not args.quiet:
                print(f"Initialized KingsRow engine from: {dll_path}")
        else:
            print(f"Warning: KingsRow DLL not found at {dll_path}. Running without KingsRow.", file=sys.stderr)

    if not args.quiet:
        print("=" * 50)
        print(f"STARTING KINGSROW BENCHMARK TRACE")
        print(f"MINIMAX_ENABLED:          {os.environ.get('MINIMAX_ENABLED', 'unset')}")
        print(f"MINIMAX_DEPTH:            {os.environ.get('MINIMAX_DEPTH', 'unset')}")
        print("=" * 50)
        print()

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
                print("\nGAME INCOMPLETE: max turns reached")
                print("\nFinal board:")
                print_board(acc["board"])
                gid = acc.get("game_log_id") or "?"
                print(f"\nLogs (if any): logs/{gid}.jsonl")
            return

        if acc["current_player"] == RED:
            acc = _run_red_ply(acc, args.quiet, kr_engine, args.kr_time, args.kingsrow_depth)
        else:
            acc = _run_black_ply(acc, args.quiet)


if __name__ == "__main__":
    main()
