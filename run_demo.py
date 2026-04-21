#!/usr/bin/env python3
"""
Pipeline demo: inter_turn_memory → proposal_agent → format_checker → validator

  python run_demo.py

Stops after validator (no ranker / state_manager). Requires Ollama for proposal step.
"""

from __future__ import annotations

import json
from pprint import pprint

from checkers.agents.proposal_agent import proposal_agent
from checkers.engine.board import RED, create_initial_board, print_board
from checkers.engine.rules import get_all_legal_moves
from checkers.nodes.format_checker import format_checker
from checkers.nodes.inter_turn_memory import inter_turn_memory
from checkers.nodes.validator import _moves_match, validator
from checkers.state.state import CheckersState


def merge(state: CheckersState, patch: dict) -> CheckersState:
    return CheckersState(**{**state.model_dump(), **patch})


def _print_json(label: str, obj: object) -> None:
    print(f"\n{label}")
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _trace_proposals_vs_legal(state: CheckersState) -> None:
    """Print pass/fail per format-cleaned proposal vs engine legal moves."""
    proposed = state.proposed_moves
    legal = state.legal_moves
    if not isinstance(proposed, list):
        print("(no list of proposed_moves to trace)")
        return
    if not isinstance(legal, list):
        print("(no legal_moves list)")
        return

    print("\n--- Per-proposal vs engine (after validator) ---")
    print(
        "Each row: format-cleaned proposal → OK if path matches some "
        "entry in legal_moves (enriched)."
    )
    for i, pr in enumerate(proposed):
        ok = any(_moves_match(pr, lm) for lm in legal)
        status = "LEGAL (matched)" if ok else "illegal (no path match)"
        print(f"  [{i}] {status}  type={pr.get('type')} path={pr.get('path')}")


def main() -> None:
    print("=" * 72)
    print("PIPELINE: memory → proposal → format_checker → validator")
    print("=" * 72)
    print("""
  [1] inter_turn_memory   → strategic_context
  [2] proposal_agent      → proposed_moves (raw JSON: {"selected_indices":[...]})
  [3] format_checker      → proposed_moves (list of dicts)
  [4] validator           → legal_moves (enriched, deduped) or feedback
""")

    print("=" * 72)
    print("[0] START")
    print("=" * 72)
    state = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    )
    print(f"current_player: RED ({RED}), turn_number={state.turn_number}")
    print("\nBoard (r/R = RED, b/B = BLACK):")
    print_board(state.board)

    n_engine = len(get_all_legal_moves(state.board, state.current_player))
    print(f"\nEngine: {n_engine} legal moves for current_player (ground truth count).")

    print("\n" + "=" * 72)
    print("[1] inter_turn_memory")
    print("=" * 72)
    patch1 = inter_turn_memory(state)
    print("patch keys:", list(patch1.keys()))
    ctx = patch1.get("strategic_context") or {}
    print(f"  game_phase: {ctx.get('game_phase')}")
    print(f"  strategic_priorities (first 5): {ctx.get('strategic_priorities', [])[:5]}")
    state = merge(state, patch1)

    print("\n" + "=" * 72)
    print("[2] proposal_agent (Ollama)")
    print("=" * 72)
    patch2 = proposal_agent(state)
    raw = patch2.get("proposed_moves")
    if isinstance(raw, str):
        print(f"proposed_moves: str, len={len(raw)}")
        print("--- raw (first 600 chars) ---")
        print(raw[:600] + ("..." if len(raw) > 600 else ""))
    else:
        print(f"proposed_moves: {raw!r}")
    state = merge(state, patch2)

    print("\n" + "=" * 72)
    print("[3] format_checker")
    print("=" * 72)
    patch3 = format_checker(state)
    print("patch keys:", list(patch3.keys()))
    print(f"  format_error_count: {patch3.get('format_error_count')}")
    print(f"  insufficient_proposals: {patch3.get('insufficient_proposals')}")
    if patch3.get("feedback"):
        print("  feedback:", patch3.get("feedback")[:400])
    pm = patch3.get("proposed_moves")
    if isinstance(pm, list):
        print(f"  cleaned proposals: {len(pm)}")
    state = merge(state, patch3)

    print("\n" + "=" * 72)
    print("[4] validator (symbolic rules)")
    print("=" * 72)
    patch4 = validator(state)
    print("patch keys:", list(patch4.keys()))
    print(f"  last_completed_node: {patch4.get('last_completed_node')!r}")

    lm = patch4.get("legal_moves")
    if patch4.get("feedback"):
        print("\n--- validator feedback (all proposals illegal) ---")
        print(patch4.get("feedback"))
    else:
        print(f"  legal_moves count (enriched + deduped): {len(lm) if isinstance(lm, list) else lm}")

    state = merge(state, patch4)

    # Trace: each format-cleaned proposal vs surviving legal set
    _trace_proposals_vs_legal(state)

    if isinstance(lm, list) and len(lm) > 0:
        print("\n--- Sample validated legal move (enriched, facts truncated) ---")
        sample = lm[0]
        short = {
            "type": sample.get("type"),
            "path": sample.get("path"),
            "captured": sample.get("captured"),
            "facts_keys": list((sample.get("facts") or {}).keys())[:12],
        }
        pprint(short)
        if len(lm) > 1:
            print(f"... and {len(lm) - 1} more enriched move(s).")

    print("\n" + "=" * 72)
    print("FINAL snapshot")
    print("=" * 72)
    print(f"last_completed_node: {state.last_completed_node}")
    print(f"len(proposed_moves) after format_checker: {len(state.proposed_moves) if isinstance(state.proposed_moves, list) else 'n/a'}")
    print(f"len(legal_moves) after validator: {len(state.legal_moves) if isinstance(state.legal_moves, list) else 'n/a'}")
    print(f"format_error_count: {state.format_error_count}")
    print(f"retry_count: {state.retry_count}")
    print("\nDone (stopped after validator).")


if __name__ == "__main__":
    main()
