# checkers/evaluation/run_manual_eval_pipeline.py
#
# Manual-monitor game runner.
#
# PURPOSE
# -------
# Run a checkers game where:
#   - RED  = full AI pipeline (unchanged)
#   - BLACK = you, entering a legal-move index each turn
#
# After every RED turn it prints:
#   chosen move · reasoning · seeds · grouped facts ·
#   ranker_diagnostics summary · live evaluate_turn() claim table
#
# At game end it saves JSONL + aggregate JSON and prints a summary.
#
# ROOT CAUSE NOTE
# ---------------
# update_agent calls state_manager which CLEARS chosen_move,
# last_move_reasoning, chosen_move_facts, ranker_diagnostics back to None
# before returning to LangGraph.  We therefore capture these fields from
# the *ranker_agent* stream chunk, before update_agent fires.
#
# USAGE
# -----
#   python -m checkers.evaluation.run_manual_eval_pipeline
#   python -m checkers.evaluation.run_manual_eval_pipeline --max-turns 60 --verbose
#   python -m checkers.evaluation.run_manual_eval_pipeline --output-dir logs/manual_eval/session1

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── ANSI colour helpers (degrade gracefully) ──────────────────────────────────
_TTY = sys.stdout.isatty()
def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def _grn(t: str) -> str: return _ansi("32;1", t)
def _red(t: str) -> str: return _ansi("31;1", t)
def _ylw(t: str) -> str: return _ansi("33;1", t)
def _cyn(t: str) -> str: return _ansi("36;1", t)
def _bld(t: str) -> str: return _ansi("1",    t)
def _dim(t: str) -> str: return _ansi("2",    t)
def _mag(t: str) -> str: return _ansi("35;1", t)


# ── Fact groups for clean display ─────────────────────────────────────────────
_FACT_GROUPS: Dict[str, List[str]] = {
    "Material":  ["captures_count", "net_gain", "kings_captured", "results_in_king", "near_promotion"],
    "Safety":    ["opponent_can_recapture", "our_pieces_threatened_before",
                  "our_pieces_threatened_after", "recapturable_piece_is_king"],
    "Mobility":  ["our_mobility_before", "our_mobility_after",
                  "opponent_mobility_before", "opponent_mobility_after", "mobility_reduction"],
    "Tactical":  ["creates_immediate_threat", "blocks_opponent_landing",
                  "forced_opponent_jump_reply", "shot_sequence_available",
                  "center_control", "threat_after"],
    "Structure": ["leaves_piece_isolated", "weakens_king_row", "moves_from_king_row"],
    "Minimax":   ["minimax_score", "score_gap"],
}
_GROUPED_KEYS = {k for g in _FACT_GROUPS.values() for k in g}


# Keys that are interesting even when False/0 — always show these
_ALWAYS_SHOW = {
    "opponent_can_recapture", "creates_immediate_threat", "leaves_piece_isolated",
    "results_in_king", "near_promotion", "weakens_king_row", "center_control",
    "forced_opponent_jump_reply", "shot_sequence_available",
}


def _print_facts(facts: Dict[str, Any], verbose: bool = False) -> None:
    if not facts:
        print(_red("    [WARNING] chosen_move_facts is empty — facts not available."))
        return
    for group, keys in _FACT_GROUPS.items():
        items = []
        for k in keys:
            if k not in facts or facts[k] is None:
                continue
            v = facts[k]
            # In compact mode skip boring zero/False values for non-key facts
            if not verbose and k not in _ALWAYS_SHOW:
                if v is False or v == 0:
                    continue
            items.append((k, v))
        if not items:
            continue
        print(f"  {_cyn(group)}:")
        for k, v in items:
            if v is True:
                disp = _grn(str(v))
            elif v is False:
                disp = _ylw(str(v))
            elif isinstance(v, (int, float)) and v > 0:
                disp = _grn(str(v))
            elif isinstance(v, (int, float)) and v < 0:
                disp = _red(str(v))
            else:
                disp = str(v)
            print(f"    {k:<36s} {disp}")
    # 'Other' (internal engine fields) only shown with --verbose
    if verbose:
        extras = [(k, v) for k, v in facts.items()
                  if k not in _GROUPED_KEYS and v is not None and k != "path"]
        if extras:
            print(f"  {_cyn('Other')} {_dim('(verbose)')}:")
            for k, v in extras:
                print(f"    {k:<36s} {v}")


def _print_eval_record(record: Any, verbose: bool = False) -> None:
    """Print TurnEvaluationRecord claim table."""
    STATUS_COLOR = {
        "supported":    _grn,
        "unsupported":  _ylw,
        "contradicted": _red,
        "vague":        _dim,
    }
    path_label = record.reasoning_path or "unknown"
    print(
        f"\n  {_bld('── Claim Evaluation ──')}  "
        f"path={_cyn(path_label)}  "
        f"total={record.total_claims}  "
        f"{_grn('supp=' + str(record.supported_count))}  "
        f"{_ylw('unsupp=' + str(record.unsupported_count))}  "
        f"{_red('contra=' + str(record.contradicted_count))}  "
        f"{_dim('vague=' + str(record.vague_count))}"
    )
    if record.total_claims == 0:
        print(_ylw("    (no claim phrases detected in reasoning text)"))
    for c in record.claims:
        st = c.claim_status if isinstance(c.claim_status, str) else c.claim_status.value
        col = STATUS_COLOR.get(st, _dim)
        seed_tag = _dim(" ← seed") if c.matched_seed else ""
        phrase = f'"{c.matched_phrase}"' if c.matched_phrase else "(no phrase)"
        print(f"    [{col(st[:5].upper())}]  {c.claim_type:<32s}  {phrase}{seed_tag}")
        if verbose and c.hallucination_type:
            ht = c.hallucination_type if isinstance(c.hallucination_type, str) else c.hallucination_type.value
            print(f"             hallucination={_red(ht)}")
    if record.trajectory_events:
        print(f"  {_bld('Trajectory:')} {record.trajectory_events}")
    if record.provenance_note:
        print(f"  {_bld('Provenance:')} {_ylw(record.provenance_note)}")


def _print_diag_summary(rd: Dict[str, Any], verbose: bool) -> None:
    if not rd:
        print(_ylw("    [WARNING] ranker_diagnostics is empty."))
        return
    retry    = rd.get("reasoning_refinement_retry_count", 0) or 0
    contra   = rd.get("reasoning_contradiction_detected", False)
    repaired = rd.get("reasoning_contradiction_repaired", False)
    fallbk   = rd.get("reasoning_is_seed_fallback", False)
    print(
        f"\n  {_bld('── Diagnostics ──')}  "
        f"retry={retry}  "
        f"contradiction={_red('YES') if contra else _grn('no')}  "
        f"repaired={_grn('YES') if repaired else 'no'}  "
        f"seed_fallback={_ylw('YES') if fallbk else 'no'}"
    )
    ics = rd.get("reasoning_initial_contradictions") or []
    for ic in ics:
        print(_red(f"    ⚠ initial: {ic}"))
    if verbose:
        skip = {"reasoning_seeds", "reasoning_initial_contradictions",
                "reasoning_final_contradictions"}
        for k, v in rd.items():
            if k not in skip:
                print(f"    {k}: {v}")


# ── Runtime import (deferred so dotenv loads first) ───────────────────────────
def _import_runtime():
    os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
    os.environ.setdefault("CHECKERS_LOGGER_PRINT",   "false")
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
    from checkers.graph.graph              import checkers_graph
    from checkers.state.state              import CheckersState
    from checkers.agents.update_agent      import update_agent as _ua
    from checkers.engine.board             import RED, BLACK, create_initial_board, print_board
    from checkers.engine.rules             import get_all_legal_moves
    from checkers.engine.move_facts        import compute_move_facts
    return (checkers_graph, CheckersState, _ua,
            RED, BLACK, create_initial_board, print_board,
            get_all_legal_moves, compute_move_facts)


# ── Evaluator ─────────────────────────────────────────────────────────────────
from checkers.evaluation.turn_evaluator import evaluate_turn


# ── Main game loop ────────────────────────────────────────────────────────────
def run_manual_game(
    max_turns:  int  = 200,
    verbose:    bool = False,
    output_dir: str  = "logs/manual_eval",
) -> Dict[str, Any]:
    (
        checkers_graph, CheckersState, _ua,
        RED, BLACK,
        create_initial_board, print_board,
        get_all_legal_moves, compute_move_facts,
    ) = _import_runtime()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    eval_records: List[Any]       = []
    move_log:     List[Dict]      = []

    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()

    print(_bld("\n╔═══════════════════════════════════════════════════════════╗"))
    print(_bld("║      Manual-Monitor Checkers Evaluation Runner            ║"))
    print(_bld("╚═══════════════════════════════════════════════════════════╝"))
    print(f"  Session: {session_id}   max_turns={max_turns}   verbose={verbose}")
    print(f"  RED = AI pipeline    BLACK = you (enter move index, or 'q' to quit)")
    print()

    while not acc.get("game_over") and (acc.get("turn_number") or 0) < max_turns:
        player  = acc["current_player"]
        turn_no = acc.get("turn_number") or 0
        board   = acc["board"]

        # ── board display ─────────────────────────────────────────────────────
        player_label = _grn("RED (AI)") if player == RED else _ylw("BLACK (you)")
        print(_bld(f"\n{'─'*62}"))
        print(_bld(f"  Turn {turn_no + 1}  —  {player_label}"))
        print(_bld(f"{'─'*62}"))
        print_board(board)
        print()

        if player == RED:
            # ── RED: run AI pipeline, capturing ranker_agent delta ────────────
            print(_cyn("  [AI] Running pipeline…"))
            acc["last_completed_node"] = None
            cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 60}

            # These are cleared by state_manager inside update_agent.
            # We capture them from the ranker_agent stream chunk.
            _captured: Dict[str, Any] = {}

            try:
                for chunk in checkers_graph.stream(
                    acc, stream_mode="updates",
                    interrupt_after=["update_agent"], config=cfg,
                ):
                    for node_name, delta in chunk.items():
                        if node_name in ("__interrupt__", "__end__"):
                            continue
                        if not isinstance(delta, dict):
                            continue
                        # ── Capture ranker_agent fields BEFORE update_agent clears them
                        if node_name == "ranker_agent":
                            for field in ("chosen_move", "last_move_reasoning",
                                          "chosen_move_facts", "ranker_diagnostics"):
                                if field in delta and delta[field] is not None:
                                    _captured[field] = delta[field]
                        acc.update(delta)
            except Exception as exc:
                print(_red(f"  [AI] Pipeline error: {exc}"))
                import traceback
                traceback.print_exc()
                break

            # ── Use captured fields (ranker_agent values) ─────────────────────
            chosen_move = _captured.get("chosen_move") or []
            reasoning   = _captured.get("last_move_reasoning") or ""
            rd          = _captured.get("ranker_diagnostics") or {}
            facts       = _captured.get("chosen_move_facts") or {}
            seeds       = rd.get("reasoning_seeds") or []
            game_log_id = acc.get("game_log_id") or session_id
            turn_id     = f"{game_log_id}_t{acc.get('turn_number', turn_no + 1)}"

            # Warn on missing data
            move_path = chosen_move.get("path", []) if isinstance(chosen_move, dict) else (
                chosen_move if isinstance(chosen_move, list) else [])
            if not move_path:
                print(_red("  [WARNING] chosen_move path is empty — AI may have had no legal moves."))
            if not reasoning:
                print(_red("  [WARNING] last_move_reasoning is empty — reasoning not generated."))
            if not facts:
                print(_red("  [WARNING] chosen_move_facts is empty — facts not available."))

            # ── Chosen move ───────────────────────────────────────────────────
            print(f"\n  {_bld('Chosen move:')}  {_grn(str(move_path))}")

            # ── Reasoning ─────────────────────────────────────────────────────
            print(f"\n  {_bld('── Reasoning ──')}")
            if reasoning:
                # Split on ". " to show sentence-per-line; avoid over-splitting
                sentences = [s.strip() for s in reasoning.replace("  ", " ").split(". ") if s.strip()]
                for sent in sentences:
                    if not sent.endswith("."):
                        sent += "."
                    print(f"    {sent}")
            else:
                print(_ylw("    (empty)"))

            # ── Seeds ─────────────────────────────────────────────────────────
            print(f"\n  {_bld('── Seeds ──')}")
            if seeds:
                for s in seeds:
                    print(f"    • {_dim(s)}")
            else:
                print(_ylw("    (none)"))

            # ── Facts ─────────────────────────────────────────────────────────
            print(f"\n  {_bld('── Move facts ──')}")
            _print_facts(facts, verbose=verbose)

            # ── Diagnostics ───────────────────────────────────────────────────
            _print_diag_summary(rd, verbose)

            # ── evaluate_turn() ───────────────────────────────────────────────
            record = evaluate_turn(
                reasoning_text=reasoning,
                reasoning_seeds=seeds,
                facts=facts,
                ranker_diagnostics=rd,
                turn_id=turn_id,
            )
            _print_eval_record(record, verbose)
            eval_records.append(record)

            def _norm_path(p: Any) -> Optional[list]:
                """Normalise a move path to list-of-lists for comparison and storage."""
                if p is None:
                    return None
                try:
                    return [list(sq) for sq in p]
                except (TypeError, ValueError):
                    return None

            # raw_llm_choice_path: the initial LLM choice BEFORE any override
            # correction, read directly from ranker_diagnostics.  This is the
            # authoritative source — the same key that _build_provenance_note uses.
            _raw_llm_path_norm  = _norm_path(rd.get("raw_llm_choice_path"))
            _final_path_norm    = _norm_path(move_path)
            move_log.append({
                "turn":                    turn_no + 1,
                "player":                  "RED",
                "move":                    move_path,
                "turn_id":                 turn_id,
                "reasoning_path":          record.reasoning_path,
                "total_claims":            record.total_claims,
                "supported":               record.supported_count,
                "unsupported":             record.unsupported_count,
                "contradicted":            record.contradicted_count,
                "vague":                   record.vague_count,
                "trajectory_events":       record.trajectory_events,
                "reasoning":               reasoning,
                "seeds":                   seeds,
                "facts":                   {k: v for k, v in (facts or {}).items()},
                # ── Phase 2 provenance ──────────────────────────────────────────
                "final_choice_source":     record.final_choice_source or None,
                "raw_llm_choice_path":     _raw_llm_path_norm,
                "final_path_matches_raw_llm": (
                    (_final_path_norm == _raw_llm_path_norm)
                    if _raw_llm_path_norm is not None and _final_path_norm is not None
                    else None
                ),
                "best_score_tie_count":    record.best_score_tie_count,
                "minimax_best_path":       record.minimax_best_path,
                "tied_candidate_paths":    record.tied_candidate_paths or None,
                "provenance_note":         record.provenance_note or None,
                # ── Phase 2.3a retry diversity ──────────────────────────────
                "retry_all_paths":         record.retry_all_paths or None,
                "retry_rejection_reasons": record.retry_rejection_reasons or None,
                "retry_duplicate_count":   record.retry_duplicate_count or None,
            })

        else:
            # ── BLACK: human input ────────────────────────────────────────────
            legal = get_all_legal_moves(board, BLACK)
            if not legal:
                print(_ylw("  No legal moves for BLACK — game over."))
                break

            print(f"  {_bld('Your legal moves')} (BLACK):")
            for i, m in enumerate(legal):
                mpath = m.get("path", m)
                mtype = m.get("type", "?")
                print(f"    [{_bld(str(i))}] {mpath}  ({mtype})")

            choice: Optional[int] = None
            while choice is None:
                try:
                    raw = input(_bld("  Enter index (or 'q' to quit): ")).strip()
                except EOFError:
                    raw = "q"
                if raw.lower() in ("q", "quit", "exit"):
                    print("  Quitting.")
                    acc["game_over"] = True
                    break
                try:
                    idx = int(raw)
                    if 0 <= idx < len(legal):
                        choice = idx
                    else:
                        print(_ylw(f"  ⚠  Valid range: 0–{len(legal) - 1}"))
                except ValueError:
                    print(_ylw("  ⚠  Enter a number."))

            if acc.get("game_over"):
                break

            chosen = legal[choice]  # type: ignore[index]
            acc["chosen_move"]         = chosen
            acc["last_move_reasoning"] = "BLACK manual move"
            valid  = set(CheckersState.model_fields.keys())
            state  = CheckersState(**{k: v for k, v in acc.items() if k in valid})
            result = _ua(state)
            acc.update(result)
            chosen_path = chosen.get("path", chosen) if isinstance(chosen, dict) else chosen
            print(f"  {_cyn('BLACK played:')} {chosen_path}")
            move_log.append({"turn": turn_no + 1, "player": "BLACK", "move": chosen_path})

    # ── Game end ──────────────────────────────────────────────────────────────
    winner  = acc.get("winner")
    is_draw = acc.get("draw", False)
    gid     = acc.get("game_log_id") or session_id

    print(_bld(f"\n{'═'*62}"))
    print(_bld("  GAME OVER"))
    if is_draw:
        print("  Result: Draw")
    elif winner == 1:
        print(_grn("  Result: RED wins"))
    elif winner == 2:
        print(_red("  Result: BLACK wins"))
    else:
        print(_ylw("  Result: Turn limit / quit"))
    print(_bld(f"{'═'*62}"))

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_turns  = len(eval_records)
    total_claims = sum(r.total_claims    for r in eval_records)
    supp         = sum(r.supported_count    for r in eval_records)
    unsupp       = sum(r.unsupported_count  for r in eval_records)
    contra       = sum(r.contradicted_count for r in eval_records)
    vague_       = sum(r.vague_count        for r in eval_records)
    def pct(n: int) -> str:
        return f"{100 * n / max(total_claims, 1):.1f}%"

    path_counts: Counter  = Counter(r.reasoning_path for r in eval_records)
    event_counts: Counter = Counter()
    for r in eval_records:
        for ev in (r.trajectory_events or []):
            event_counts[ev] += 1

    bad: List = [
        (r, c) for r in eval_records for c in r.claims
        if (c.claim_status if isinstance(c.claim_status, str)
            else c.claim_status.value) in ("unsupported", "contradicted")
    ]

    print(f"\n  {_bld('RED-turn aggregate:')}")
    print(f"    Total RED turns       : {_bld(str(total_turns))}")
    print(f"    Total claims          : {_bld(str(total_claims))}")
    print(f"    Supported             : {_grn(str(supp))}  ({pct(supp)})")
    print(f"    Unsupported           : {_ylw(str(unsupp))}  ({pct(unsupp)})")
    print(f"    Contradicted          : {_red(str(contra))}  ({pct(contra)})")
    print(f"    Vague                 : {_dim(str(vague_))}  ({pct(vague_)})")
    print(f"    Reasoning paths       : {dict(path_counts)}")
    print(f"    Trajectory events     : {dict(event_counts)}")

    if bad:
        print(f"\n  {_bld(_ylw('Unsupported / contradicted claims:'))}")
        for rec, c in bad:
            st  = c.claim_status if isinstance(c.claim_status, str) else c.claim_status.value
            col = _red if st == "contradicted" else _ylw
            print(f"    [{col(st.upper())}] turn={rec.turn_id}  "
                  f"type={c.claim_type}  phrase=\"{c.matched_phrase}\"")

    # ── Save logs ─────────────────────────────────────────────────────────────
    aggregate = {
        "session_id":              session_id,
        "game_log_id":             gid,
        "timestamp_utc":           datetime.now(timezone.utc).isoformat(),
        "total_red_turns":         total_turns,
        "total_claims":            total_claims,
        "supported_pct":           round(100 * supp   / max(total_claims, 1), 1),
        "unsupported_pct":         round(100 * unsupp / max(total_claims, 1), 1),
        "contradicted_pct":        round(100 * contra / max(total_claims, 1), 1),
        "vague_pct":               round(100 * vague_ / max(total_claims, 1), 1),
        "reasoning_path_counts":   dict(path_counts),
        "trajectory_event_counts": dict(event_counts),
        "winner":                  winner,
        "is_draw":                 is_draw,
        "turn_count":              acc.get("turn_number", 0),
    }

    log_path = out_dir / f"manual_game_{session_id}.jsonl"
    with open(log_path, "w", encoding="utf-8") as fh:
        for entry in move_log:
            fh.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    agg_path = out_dir / f"manual_aggregate_{session_id}.json"
    agg_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  {_bld('Logs saved:')}")
    print(f"    Move log  → {log_path}")
    print(f"    Aggregate → {agg_path}")
    print()
    return aggregate


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Manual-monitor checkers game.\n"
            "RED = full AI pipeline.  BLACK = you (enter move index).\n"
            "Every RED turn prints move, reasoning, seeds, facts, "
            "diagnostics, and live claim evaluation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m checkers.evaluation.run_manual_eval_pipeline\n"
            "  python -m checkers.evaluation.run_manual_eval_pipeline --max-turns 60 --verbose\n"
            "  python -m checkers.evaluation.run_manual_eval_pipeline "
            "--output-dir logs/manual_eval/session1\n"
        ),
    )
    p.add_argument("--max-turns",  type=int, default=200,
                   help="Max total half-turns before forced end (default: 200).")
    p.add_argument("--verbose",    action="store_true",
                   help="Print full ranker_diagnostics and hallucination annotations.")
    p.add_argument("--output-dir", type=str, default="logs/manual_eval",
                   help="Dir for move-log JSONL and aggregate JSON (default: logs/manual_eval).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_manual_game(
        max_turns=args.max_turns,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )
