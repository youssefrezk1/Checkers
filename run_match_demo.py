#!/usr/bin/env python3
"""
Multi-ply match demo: one *match turn* = RED (LLM pipeline) then BLACK (symbolic).

  python run_match_demo.py
  python run_match_demo.py --turns 3
  python run_match_demo.py --turns 5 --seed 42

Match turn 1 = first RED ply + first BLACK ply (5 turns ⇒ 5 RED + 5 BLACK).
Engine `turn_number` still increments once per *ply* inside `state_manager`.

BLACK picks a *random* legal move (varied positions) so the validator sees many
boards. Use `--seed` for reproducibility.

RED ranker: `ranker_agent` (default `phi4-mini`, env `OLLAMA_RANKER_MODEL`) chooses an
index over `validator.legal_moves` + one-sentence reasoning. Use `--stub-ranker` to
force index 0 (no ranker LLM). Strategic context for the ranker is on by default;
set env `RANKER_INCLUDE_STRATEGIC_CONTEXT=false` to A/B test facts-only.

RED ply retries: if format_checker or validator rejects everything, the demo loops
back to `proposal_agent` with `state.feedback` until `retry_count >= retry_budget`
(default 3). Retries also append a short numbered list of engine-legal moves.
If the ranker LLM fails or returns an invalid index, the demo retries the ranker
until `ranker_retry_count >= ranker_retry_budget`, then applies `ranker_fallback`
(legal_moves[0]) and increments `ranker_fallback_count` for evaluation metrics.

Proposal: LLM outputs JSON `{"selected_indices":[...]}` into the engine’s numbered legal moves.
Requires Ollama for RED (proposal + ranker unless `--stub-ranker`).
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from checkers.agents.proposal_agent import proposal_agent
from checkers.agents.ranker_agent import (
    OLLAMA_RANKER_MODEL,
    RANKER_INCLUDE_STRATEGIC_CONTEXT,
    ranker_agent,
)
from checkers.engine.board import RED, BLACK, create_initial_board, print_board
from checkers.engine.rules import get_all_legal_moves
from checkers.nodes.format_checker import format_checker
from checkers.nodes.inter_turn_memory import inter_turn_memory
from checkers.nodes.state_manager import state_manager
from checkers.nodes.ranker_fallback import ranker_fallback
from checkers.nodes.validator import validator
from checkers.nodes.win_condition import win_condition
from checkers.state.state import CheckersState


def merge(state: CheckersState, patch: dict) -> CheckersState:
    return CheckersState(**{**state.model_dump(), **patch})


def _side_name(p: int) -> str:
    return "RED" if p == RED else "BLACK"


def _winner_display(w: int | None) -> str:
    if w is None:
        return "None"
    return _side_name(w)


def _truncate(text: str, max_len: int = 600) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _move_json(m: dict[str, Any]) -> str:
    return json.dumps(
        {"type": m.get("type"), "path": m.get("path"), "captured": m.get("captured")},
        sort_keys=True,
    )


def _proposals_preview(pm: list | str) -> str:
    if isinstance(pm, str):
        return f"str ({len(pm)} chars): {_truncate(pm, 400)}"
    if not pm:
        return "[] (empty)"
    lines = [f"  [{i}] {_move_json(x)}" for i, x in enumerate(pm[:8])]
    extra = f"\n  ... and {len(pm) - 8} more" if len(pm) > 8 else ""
    return f"{len(pm)} dict(s)\n" + "\n".join(lines) + extra


def _verbose_red_header(
    label: str,
    state: CheckersState,
    *,
    engine_legal_n: int | None = None,
) -> None:
    print(f"\n  {'─' * 68}")
    print(f"  {label}")
    print(f"  {'─' * 68}")
    print(
        f"  current_player={_side_name(state.current_player)}  "
        f"turn_number={state.turn_number}  "
        f"retry_count={state.retry_count}  "
        f"format_error_count={state.format_error_count}  "
        f"ranker_failure_count={state.ranker_failure_count}  "
        f"ranker_fallback_count={state.ranker_fallback_count}"
    )
    if engine_legal_n is not None:
        print(f"  engine legal moves (ground truth): {engine_legal_n}")


def _trace_red_ply_attempt(attempt: int, state: CheckersState, *, stub_ranker: bool) -> None:
    """High-level banner so logs are easy to scan."""
    budget = state.retry_budget
    rc = state.retry_count
    tail = " → stub[0]" if stub_ranker else " → ranker_agent"
    print(f"\n  {'═' * 68}")
    print(
        f"  RED PLY  │  ATTEMPT {attempt}  │  retry_count={rc} / budget={budget}  "
        f"│  proposal → format_checker → validator{tail}"
    )
    print(f"  {'═' * 68}")


def _engine_legal_moves_hint(board: list[list[int]], player: int, *, max_lines: int = 22) -> str:
    """Short, copy-friendly list for the proposer on retry (not persisted in state)."""
    moves = get_all_legal_moves(board, player)
    if not moves:
        return "Engine reports zero legal moves for this player (position may be terminal)."
    lines = [
        f"Engine legal moves ({len(moves)} total). Reply with "
        f'{{"selected_indices": [i,j,...]}} using only indices 0..{len(moves) - 1} (see user prompt for full list):',
    ]
    for i, m in enumerate(moves[:max_lines]):
        lines.append(
            f"  [{i}] type={m['type']!r} path={m['path']!r} captured={m.get('captured', [])!r}"
        )
    if len(moves) > max_lines:
        lines.append(f"  ... and {len(moves) - max_lines} more.")
    return "\n".join(lines)


def _state_for_proposal(state: CheckersState) -> CheckersState:
    """
    When there is feedback from a prior failed attempt, append engine legal moves
    so the model can copy a real path. Does not mutate persisted feedback alone:
    we build a one-off merged view for the LLM call only.
    """
    fb = state.feedback
    if not fb or not str(fb).strip():
        return state
    hint = _engine_legal_moves_hint(state.board, state.current_player)
    return merge(
        state,
        {
            "feedback": str(fb).rstrip()
            + "\n\n---\n"
            + hint,
        },
    )


def _legal_index_for_chosen(legal_moves: list[dict[str, Any]], chosen: dict[str, Any]) -> int:
    for i, m in enumerate(legal_moves):
        if m.get("path") == chosen.get("path") and m.get("type") == chosen.get("type"):
            return i
    return -1


def run_llm_side(
    state: CheckersState,
    *,
    verbose: bool,
    stub_ranker: bool = False,
) -> tuple[CheckersState, bool]:
    """
    Once per ply: inter_turn_memory, then loop
    proposal → format_checker → validator until some proposal is legal or
    retry_count >= retry_budget. Then ranker_agent (or stub) + state_manager + win_condition.

    Returns (new_state, ok).
    """
    n_engine = len(get_all_legal_moves(state.board, state.current_player))
    if verbose:
        _verbose_red_header("RED PLY SETUP: inter_turn_memory (before)", state, engine_legal_n=n_engine)

    patch = inter_turn_memory(state)
    state = merge(state, patch)

    if verbose:
        _verbose_red_header("RED PLY SETUP: after inter_turn_memory", state, engine_legal_n=n_engine)
        sc = state.strategic_context
        if sc:
            keys = list(sc.keys()) if isinstance(sc, dict) else type(sc).__name__
            print(f"  strategic_context keys: {keys}")
            if isinstance(sc, dict) and "summary" in sc:
                print(f"  summary preview: {_truncate(str(sc['summary']), 300)}")
        else:
            print("  strategic_context: None")
        if state.move_history:
            last = state.move_history[-1]
            print(
                f"  last move_history entry turn={last.get('turn')}: "
                f"{_move_json(last.get('move', {}))}"
            )

    attempt = 0
    while True:
        if state.retry_count >= state.retry_budget:
            if verbose:
                print(
                    f"\n  STOP: retry_count ({state.retry_count}) >= retry_budget "
                    f"({state.retry_budget}) — no more proposal attempts for this ply."
                )
            else:
                print(
                    f"  [RED] gave up: retry_count {state.retry_count} >= budget "
                    f"{state.retry_budget}."
                )
            return state, False

        attempt += 1
        if verbose:
            _trace_red_ply_attempt(attempt, state, stub_ranker=stub_ranker)
        elif attempt > 1:
            print(
                f"  [RED] retry attempt {attempt} (retry_count={state.retry_count}, "
                f"budget={state.retry_budget}) — calling proposer again with feedback."
            )

        if verbose and state.feedback:
            print("  Feedback passed to proposer (validator/format_checker):")
            for line in str(state.feedback).split("\n"):
                print(f"    │ {line}")
            print("  (Proposer also receives appended engine legal-move list in the user prompt.)")

        proposal_in = _state_for_proposal(state)
        if verbose:
            _verbose_red_header(
                "STEP: proposal_agent (calling LLM)",
                state,
                engine_legal_n=n_engine,
            )

        patch = proposal_agent(proposal_in)
        state = merge(state, patch)

        if verbose:
            _verbose_red_header("STEP: after proposal_agent", state, engine_legal_n=n_engine)
            pm = state.proposed_moves
            print(f"  proposed_moves: {_proposals_preview(pm)}")

        if verbose:
            _verbose_red_header("STEP: format_checker", state, engine_legal_n=n_engine)

        patch = format_checker(state)
        state = merge(state, patch)

        if verbose:
            _verbose_red_header("STEP: after format_checker", state, engine_legal_n=n_engine)
            pm = state.proposed_moves
            print(f"  proposed_moves: {_proposals_preview(pm)}")
            print(
                f"  format_error_count={state.format_error_count}  "
                f"insufficient_proposals={state.insufficient_proposals}"
            )

        if isinstance(state.proposed_moves, list) and len(state.proposed_moves) == 0:
            if verbose:
                fb = (state.feedback or "").upper()
                if "INSUFFICIENT_PROPOSAL_COUNT" in fb:
                    print(
                        "  ROUTE: format_checker — need ≥3 distinct valid indices while engine has ≥3 "
                        "legal moves → retry (see feedback)."
                    )
                else:
                    print(
                        "  ROUTE: format_checker returned zero proposals → retry "
                        "(feedback in state for next proposal)."
                    )
            if state.retry_count >= state.retry_budget:
                return state, False
            continue

        if verbose:
            _verbose_red_header("STEP: validator", state, engine_legal_n=n_engine)

        patch = validator(state)
        state = merge(state, patch)

        if verbose:
            _verbose_red_header("STEP: after validator", state, engine_legal_n=n_engine)
            print(f"  validator legal_moves (accepted proposals, deduped): {len(state.legal_moves)}")
            for i, m in enumerate(state.legal_moves[:12]):
                facts = m.get("facts") or {}
                print(
                    f"    [{i}] {_move_json(m)}  "
                    f"net_gain={facts.get('net_gain', 'n/a')} "
                    f"captures={facts.get('captures_count', 'n/a')}"
                )
            if len(state.legal_moves) > 12:
                print(f"    ... and {len(state.legal_moves) - 12} more")
            if state.feedback:
                print("  outcome: no proposal matched engine legality — will retry if budget left.")
                print("  validator feedback:")
                for line in str(state.feedback).split("\n"):
                    print(f"    │ {line}")
            else:
                print(
                    "  outcome: at least one legal proposal — continuing to "
                    + ("stub ranker [0]." if stub_ranker else "ranker_agent.")
                )

        if state.legal_moves:
            break

        if state.retry_count >= state.retry_budget:
            if verbose:
                print(
                    f"\n  STOP: retry_count ({state.retry_count}) >= retry_budget after validator."
                )
            return state, False

        if verbose:
            print("  ROUTE: looping back to proposal_agent with feedback above.\n")
        continue

    if stub_ranker:
        chosen = state.legal_moves[0]
        reasoning = "Stub ranker: always index 0."
        if verbose:
            print(f"\n  {'─' * 68}")
            print(f"  APPLY MOVE (stub ranker): legal_moves[0]: {_move_json(chosen)}")
            print(f"  {'─' * 68}")
        state = merge(
            state,
            {
                "chosen_move": chosen,
                "last_move_reasoning": reasoning,
                "ranker_retry_count": 0,
            },
        )
    else:
        while True:
            if verbose:
                _verbose_red_header("STEP: ranker_agent (LLM)", state, engine_legal_n=n_engine)
                print(
                    f"  ranker_retry_count={state.ranker_retry_count} / "
                    f"budget={state.ranker_retry_budget}"
                )
            patch = ranker_agent(state)
            state = merge(state, patch)
            if state.chosen_move is not None:
                if verbose:
                    ch = state.chosen_move
                    ri = _legal_index_for_chosen(state.legal_moves, ch)
                    print(f"  ranker chose index {ri}: {_move_json(ch)}")
                    print(f"  last_move_reasoning: {_truncate(state.last_move_reasoning or '', 500)}")
                    print(f"\n  {'─' * 68}")
                    print(f"  APPLY MOVE: {_move_json(ch)}")
                    print(f"  {'─' * 68}")
                break
            if state.ranker_retry_count < state.ranker_retry_budget:
                if verbose:
                    print(
                        "  ROUTE: ranker failed (LLM/parse/invalid index) — retry ranker "
                        f"(failure total session: {state.ranker_failure_count})."
                    )
                continue
            if verbose:
                print(
                    "  ROUTE: ranker retries exhausted — ranker_fallback (legal_moves[0])."
                )
            patch = ranker_fallback(state)
            state = merge(state, patch)
            if state.chosen_move is None:
                if verbose:
                    print("  STOP: ranker_fallback could not pick a move (no legal_moves).")
                return state, False
            if verbose:
                ch = state.chosen_move
                print(f"  fallback move: {_move_json(ch)}")
                print(f"  last_move_reasoning: {_truncate(state.last_move_reasoning or '', 500)}")
                print(f"\n  {'─' * 68}")
                print(f"  APPLY MOVE: {_move_json(ch)}")
                print(f"  {'─' * 68}")
            break
    patch = state_manager(state)
    state = merge(state, patch)
    patch = win_condition(state)
    state = merge(state, patch)
    return state, True


def run_symbolic_side(
    state: CheckersState,
    rng: random.Random,
    *,
    verbose: bool,
) -> tuple[CheckersState, int]:
    """
    inter_turn_memory → random legal move → state_manager → win_condition.
    Returns (new_state, chosen_index).
    """
    if verbose:
        n0 = len(get_all_legal_moves(state.board, state.current_player))
        print(f"\n  {'─' * 68}")
        print("  BLACK: inter_turn_memory")
        print(f"  {'─' * 68}")
        print(
            f"  current_player={_side_name(state.current_player)}  "
            f"turn_number={state.turn_number}  engine legal: {n0}"
        )

    patch = inter_turn_memory(state)
    state = merge(state, patch)

    moves = get_all_legal_moves(state.board, state.current_player)
    if not moves:
        return state, -1

    idx = rng.randrange(len(moves))
    chosen = moves[idx]

    if verbose:
        print(f"\n  BLACK: random choice index {idx} / {len(moves)} legal moves")
        cap = 24
        for i, m in enumerate(moves[:cap]):
            mark = " <-- chosen" if i == idx else ""
            print(f"    [{i}] {_move_json(m)}{mark}")
        if len(moves) > cap:
            print(f"    ... ({len(moves) - cap} more lines omitted)")
        print(f"  applying: {_move_json(chosen)}")

    state = merge(state, {"chosen_move": chosen})
    patch = state_manager(state)
    state = merge(state, patch)
    patch = win_condition(state)
    state = merge(state, patch)
    return state, idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match demo: each --turns unit is RED then BLACK "
            "(5 turns = 5 red plies + 5 black plies)."
        )
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=5,
        metavar="N",
        help="Maximum match turns (RED+BLACK each). Default 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for BLACK's random legal move (omit for nondeterministic).",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Less console output (no per-node validator trace).",
    )
    parser.add_argument(
        "--stub-ranker",
        action="store_true",
        help="RED: skip ranker LLM and always play validator legal_moves[0].",
    )
    args = parser.parse_args()
    max_turns = max(1, args.turns)
    verbose = not args.brief
    rng = random.Random(args.seed)

    llm_side = RED
    opponent_side = BLACK

    state = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    )

    print("=" * 72)
    print("MATCH DEMO")
    print("  • One *match turn* = RED (LLM+validator) then BLACK (random legal move).")
    print(f"  • Max match turns: {max_turns} ⇒ up to {max_turns} RED + {max_turns} BLACK plies.")
    print("  • state.turn_number increments once per applied move (state_manager).")
    print("  • RED: on format/validator failure, re-calls proposer with feedback until retry_budget.")
    print(f"  • RED retry_budget (CheckersState default): {CheckersState().retry_budget}")
    if args.stub_ranker:
        print("  • RED ranker: stub (--stub-ranker, always index 0).")
    else:
        print(f"  • RED ranker: Ollama `{OLLAMA_RANKER_MODEL}` (env OLLAMA_RANKER_MODEL).")
        print(
            "  • Ranker strategic_context: "
            + ("on" if RANKER_INCLUDE_STRATEGIC_CONTEXT else "off (RANKER_INCLUDE_STRATEGIC_CONTEXT)")
        )
    print(f"  • BLACK random seed: {args.seed!r}" if args.seed is not None else "  • BLACK: random (no --seed)")
    print("=" * 72)

    for match_turn in range(1, max_turns + 1):
        if state.game_over:
            print(f"\n--- Game already over before match turn {match_turn} ---")
            break

        print(f"\n{'#' * 72}")
        print(f"# MATCH TURN {match_turn} / {max_turns}  (RED ply, then BLACK ply)")
        print(f"{'#' * 72}")

        # ── RED ───────────────────────────────────────────────────
        if state.current_player != llm_side:
            print("WARNING: expected RED to move; state mismatch.")
            break

        legal_before = get_all_legal_moves(state.board, state.current_player)
        if not legal_before:
            print("RED has no legal moves — BLACK wins.")
            state = merge(
                state,
                {
                    "game_over": True,
                    "winner": BLACK,
                    "draw": False,
                },
            )
            break

        if verbose:
            print(f"\n>>> RED — start of match turn {match_turn} (engine ply, before LLM)")
        else:
            print(f"\n--- RED (LLM)  turn_number={state.turn_number} ---")
        print_board(state.board)

        state, ok = run_llm_side(state, verbose=verbose, stub_ranker=args.stub_ranker)
        print(
            f"\n  RESULT RED: turn_number={state.turn_number}, "
            f"game_over={state.game_over}, winner={_winner_display(state.winner)}"
        )
        if not ok:
            print(
                "  RED pipeline stopped: no legal proposal after retries "
                f"(retry_count={state.retry_count}, budget={state.retry_budget})."
            )
            still_legal = get_all_legal_moves(state.board, state.current_player)
            if still_legal:
                print(
                    "  Engine still reports legal move(s) the proposals never matched; "
                    f"ground truth example: {_move_json(still_legal[0])}"
                    + (f" ({len(still_legal)} total)" if len(still_legal) > 1 else "")
                )
            break
        if state.game_over:
            print(f"\nGame over: winner={_winner_display(state.winner)} draw={state.draw}")
            break

        # ── BLACK ─────────────────────────────────────────────────
        if state.current_player != opponent_side:
            print("WARNING: expected BLACK to move after RED.")
            break

        legal_b = get_all_legal_moves(state.board, state.current_player)
        if not legal_b:
            print("BLACK has no legal moves — RED wins.")
            state = merge(
                state,
                {
                    "game_over": True,
                    "winner": RED,
                    "draw": False,
                },
            )
            break

        if verbose:
            print(f"\n>>> BLACK — same match turn {match_turn} (random legal move)")
        else:
            print(f"\n--- BLACK (random)  turn_number={state.turn_number} ---")
        print_board(state.board)

        state, black_idx = run_symbolic_side(state, rng, verbose=verbose)
        if black_idx < 0:
            print("BLACK had no legal moves (unexpected after check).")
            break
        print(
            f"\n  RESULT BLACK: picked index {black_idx}, turn_number={state.turn_number}, "
            f"game_over={state.game_over}, winner={_winner_display(state.winner)}"
        )
        if state.game_over:
            print(f"\nGame over: winner={_winner_display(state.winner)} draw={state.draw}")
            break

        print(f"\n  --- End match turn {match_turn} (RED + BLACK plies done) ---")

    print("\n" + "=" * 72)
    print("FINAL")
    print("=" * 72)
    print(f"turn_number (plies applied): {state.turn_number}")
    print(f"game_over: {state.game_over}  winner: {_winner_display(state.winner)}  draw: {state.draw}")
    print(f"move_history length: {len(state.move_history)}")
    if state.move_history:
        print("move_history (compact):")
        for rec in state.move_history:
            pl = _side_name(rec.get("player", -1))
            print(
                f"  ply_turn={rec.get('turn')} {pl} promo={rec.get('promotion')} "
                f"{_move_json(rec.get('move', {}))}"
            )
    print("\nFinal board:")
    print_board(state.board)
    print("\nDone.")


if __name__ == "__main__":
    main()
