#!/usr/bin/env python3
"""
run_simplified_trace_reasoning.py — Deep reasoning inspection tool.

Human-readable interactive inspection mode for validating:
  1. chosen reasoning quality     5. refinement behavior
  2. comparative reasoning         6. seed usage
  3. factual grounding             7. hallucination detection
  4. verifier behavior             8. contradiction repair flow

NOT an evaluator. Pure inspection / debugging layer on top of the
simplified pipeline. Gameplay behavior is UNCHANGED.

Architecture:
  Reuses all logic from run_simplified_trace.py. Adds a rich display
  layer that renders structured diagnostics from ranker_diagnostics.

Usage:
  python run_simplified_trace_reasoning.py [flags]

Press q / quit / Ctrl-C at the move prompt to exit with logs saved.
"""

from __future__ import annotations

# ── Force simplified pipeline BEFORE importing the graph ─────────────────────
import os
os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import argparse
import contextlib
import difflib
import io
import json
import re
import signal
import sys
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from checkers.graph.graph import checkers_graph
from checkers.state.state import CheckersState
from checkers.agents.update_agent import update_agent as _update_agent_fn
from checkers.engine.board import RED, BLACK, create_initial_board, print_board
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves

# Optional: rich claim extraction for --show-claims
try:
    from checkers.evaluation.unified_verifier import verify_all as _verify_all
    _VERIFY_AVAILABLE = True
except ImportError:
    _VERIFY_AVAILABLE = False

# Optional: prompt reconstruction for --save-prompts
try:
    from checkers.agents.ranker_agent import _build_seed_reasoning_prompt
    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False


# ── ANSI color palette ────────────────────────────────────────────────────────
RST  = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"

GRN  = "\033[92m"   # verified clean / SUPPORTED
ERR  = "\033[91m"   # contradictions / CONTRADICTED / errors
YLW  = "\033[93m"   # warnings / vague / UNSUPPORTED
CYN  = "\033[96m"   # seeds / facts / metadata
MAG  = "\033[95m"   # comparative reasoning
BLU  = "\033[94m"   # refinement activity
WHT  = "\033[97m"   # neutral text
GRY  = "\033[90m"   # dim / secondary


def _c(text: Any, *codes: str) -> str:
    return "".join(codes) + str(text) + RST


def _badge(label: str, ok: bool) -> str:
    color = GRN + BOLD if ok else ERR + BOLD
    return _c(f" {label} ", color)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _wrap(text: str, width: int = 90, indent: int = 4) -> str:
    pad = " " * indent
    return textwrap.fill(
        text, width=width,
        initial_indent=pad,
        subsequent_indent=pad,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _player_label(p: int) -> str:
    return "RED" if p == RED else "BLACK"


BAR  = "═" * 72
RULE = "─" * 72


# ── CLI argument parser ───────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deep reasoning inspection tool for the simplified pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Press q / quit / Ctrl-C at the move prompt to save logs and exit.",
    )

    cmp = p.add_mutually_exclusive_group()
    cmp.add_argument(
        "--comparative-on", dest="comparative",
        action="store_const", const=True,
        help="Force comparative stage ON for this run.",
    )
    cmp.add_argument(
        "--comparative-off", dest="comparative",
        action="store_const", const=False,
        help="Force comparative stage OFF for this run.",
    )
    p.set_defaults(comparative=None)

    p.add_argument("--max-turns",    type=int, default=200,
                   help="Safety cap on half-moves (default 200).")
    p.add_argument("--save-prompts", action="store_true",
                   help="Reconstruct and save LLM prompts to per-turn files.")
    p.add_argument("--show-claims",  action="store_true",
                   help="Run verify_all for full claim extraction diagnostics.")
    p.add_argument("--show-seeds",   action="store_true", default=True,
                   help="Show reasoning and comparative seeds (default ON).")
    p.add_argument("--no-show-seeds", dest="show_seeds", action="store_false")
    p.add_argument("--show-verifier", action="store_true", default=True,
                   help="Show verifier outputs (default ON).")
    p.add_argument("--no-show-verifier", dest="show_verifier", action="store_false")
    p.add_argument("--show-refinement", action="store_true", default=True,
                   help="Show refinement activity (default ON).")
    p.add_argument("--no-show-refinement", dest="show_refinement", action="store_false")
    p.add_argument("--compact", action="store_true",
                   help="Compact output — fewer detail lines.")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose output — extra diagnostic lines.")
    p.add_argument("--quiet",   action="store_true",
                   help="Print only the final summary.")
    return p


# ── Trace logger ──────────────────────────────────────────────────────────────

class TraceLogger:
    """Accumulates trace data and flushes to disk incrementally."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "per_turn").mkdir(exist_ok=True)
        self._text_buf: list[str] = []
        self._jsonl_buf: list[dict] = []
        self._term_path  = run_dir / "terminal_trace.txt"
        self._jsonl_path = run_dir / "structured_trace.jsonl"

    # ── line-level terminal log ───────────────────────────────────────────────

    def tee(self, line: str) -> None:
        """Print to stdout AND buffer the ANSI-stripped version."""
        print(line)
        self._text_buf.append(_strip_ansi(line))

    def buf(self, line: str) -> None:
        """Buffer only (no print) — for lines already printed elsewhere."""
        self._text_buf.append(_strip_ansi(line))

    # ── structured record ─────────────────────────────────────────────────────

    def record(self, data: dict) -> None:
        self._jsonl_buf.append(data)

    # ── per-turn artifact dump ────────────────────────────────────────────────

    def save_turn(self, turn_no: int, artifacts: dict[str, Any]) -> None:
        turn_dir = self.run_dir / "per_turn" / f"turn_{turn_no:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        for name, obj in artifacts.items():
            fpath = turn_dir / name
            try:
                if isinstance(obj, str):
                    fpath.write_text(obj, encoding="utf-8")
                elif obj is not None:
                    fpath.write_text(
                        json.dumps(obj, indent=2, default=str),
                        encoding="utf-8",
                    )
            except Exception as exc:
                sys.stderr.write(f"[TraceLogger] cannot write {fpath}: {exc}\n")

    # ── flush buffers to disk ─────────────────────────────────────────────────

    def flush(self) -> None:
        if self._text_buf:
            try:
                with self._term_path.open("a", encoding="utf-8") as fh:
                    for line in self._text_buf:
                        fh.write(line + "\n")
            except Exception as exc:
                sys.stderr.write(f"[TraceLogger] text flush error: {exc}\n")
            self._text_buf.clear()

        if self._jsonl_buf:
            try:
                with self._jsonl_path.open("a", encoding="utf-8") as fh:
                    for rec in self._jsonl_buf:
                        fh.write(json.dumps(rec, default=str) + "\n")
            except Exception as exc:
                sys.stderr.write(f"[TraceLogger] jsonl flush error: {exc}\n")
            self._jsonl_buf.clear()

    def write_summary(self, summary: dict) -> None:
        try:
            (self.run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            sys.stderr.write(f"[TraceLogger] summary write error: {exc}\n")


# ── Helper: capture print_board output ───────────────────────────────────────

def _capture_board(board: list) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_board(board)
    return buf.getvalue()


# ── Helper: fact coloring ─────────────────────────────────────────────────────

def _fact_color(key: str, value: Any) -> str:
    if key in ("opponent_can_recapture", "moved_piece_is_threatened",
               "leaves_piece_isolated", "weakens_king_row"):
        if value is True:  return ERR
        if value is False: return GRN
    if key in ("creates_immediate_threat", "shot_sequence_available",
               "blocks_opponent_landing", "results_in_king", "near_promotion",
               "center_control"):
        if value is True:  return GRN
        if value is False: return GRY
    if key == "our_pieces_threatened_after":
        try:
            n = int(value)
            if n == 0: return GRN
            if n == 1: return YLW
            return ERR
        except (TypeError, ValueError):
            pass
    if key in ("net_gain", "minimax_score"):
        try:
            n = float(value)
            if n > 0:   return GRN
            if n < -10: return ERR
        except (TypeError, ValueError):
            pass
    return WHT


# ── Helper: annotate board lines with move origin/destination markers ─────────

def _annotate_board_lines(board_text: str, path: list) -> list[str]:
    """
    Return board text as a list of lines with the move origin highlighted in
    yellow and the destination highlighted in green.  Leaves every other line
    unchanged so the log file (ANSI-stripped) looks identical to today.

    Board format assumed: each data row starts with a single-digit row number
    followed by a space, e.g. `5 r . r . r . r .`.  Column c occupies
    character index 2 + c*2 on that line.
    """
    lines = board_text.splitlines()
    if not path or len(path) < 2:
        return lines
    try:
        origin_r, origin_c = int(path[0][0]), int(path[0][1])
        dest_r,   dest_c   = int(path[-1][0]), int(path[-1][1])
    except (IndexError, TypeError, ValueError):
        return lines

    result = []
    for line in lines:
        if line and line[0].isdigit() and len(line) > 1 and line[1] == " ":
            row_n = int(line[0])
            chars = list(line)
            for r, c, color in ((origin_r, origin_c, YLW), (dest_r, dest_c, GRN)):
                if row_n == r:
                    pos = 2 + c * 2
                    if 0 <= pos < len(chars):
                        chars[pos] = f"{color}{chars[pos]}{RST}"
            line = "".join(chars)
        result.append(line)
    return result


# ── Helper: compact candidate lookup table rows ───────────────────────────────

def _candidate_table_rows(candidates: list) -> list[str]:
    """
    Build display rows for the candidate move lookup table shown in the
    comparative section.  Each row maps an index to the key facts the
    comparative prose references via [N] bracket notation.

    Returns plain strings (no ANSI) so callers can colour them.
    """
    rows: list[str] = []
    for i, m in enumerate(candidates):
        path = m.get("path") or []
        facts = m.get("facts") or {}
        mtype = (m.get("type") or "?")[:6]

        # Compact path notation: (r,c)→(r,c) or longer for multi-jump
        if path:
            try:
                path_str = "→".join(f"({r},{c})" for r, c in path)
            except (TypeError, ValueError):
                path_str = str(path)[:22]
        else:
            path_str = "?"

        recap = "Y" if facts.get("opponent_can_recapture") else "n"
        iso   = "Y" if facts.get("leaves_piece_isolated")  else "n"

        mob_b = facts.get("opponent_mobility_before")
        mob_a = facts.get("opponent_mobility_after")
        if mob_b is not None and mob_a is not None:
            delta = int(mob_a) - int(mob_b)
            delta_str = f"{delta:+d}"
        else:
            delta_str = "?"

        rows.append((i, path_str, mtype, recap, iso, delta_str))
    return rows


# ── Helper: refinement diff lines ────────────────────────────────────────────

def _refinement_diff_lines(raw_pre: str, final: str) -> list[str]:
    """
    Return a minimal unified-diff of raw_pre → final reasoning.
    Returns [] when the texts are identical or either is empty.
    Omits the --- / +++ / @@ header lines — only changed content is shown.
    """
    if not raw_pre or not final or raw_pre == final:
        return []
    diff = difflib.unified_diff(
        raw_pre.splitlines(),
        final.splitlines(),
        lineterm="",
        n=1,
    )
    out: list[str] = []
    for line in diff:
        if line.startswith(("---", "+++", "@@")):
            continue  # skip file-header and hunk markers
        out.append(line)
    return out


# ── Section A: board + selected move ─────────────────────────────────────────

def _section_board(board: list, move: dict, turn_no: int, player: int,
                   log: TraceLogger) -> None:
    path  = move.get("path") or []
    cap   = move.get("captured") or []
    mtype = move.get("type", "?")

    log.tee("")
    log.tee(_c(BAR, CYN))
    log.tee(_c(
        f"  TURN {turn_no}  │  {_player_label(player)} (AI — simplified pipeline)",
        BOLD + WHT,
    ))
    log.tee(_c(BAR, CYN))
    log.tee("")
    log.tee(_c("  ┌─ A. BOARD + SELECTED MOVE", BOLD + WHT))
    log.tee(_c(RULE, GRY))

    board_text = _capture_board(board)
    for line in _annotate_board_lines(board_text, path):
        log.tee(line)

    if len(path) >= 2:
        a, b = path[0], path[-1]
        cap_part = f"  captures {_c(cap, ERR)}" if cap else ""
        log.tee(
            f"  Move: {_c(mtype.upper(), BOLD + WHT)}"
            f"  {_c(a, CYN)} → {_c(b, CYN)}{cap_part}"
        )
    else:
        log.tee(f"  Move: {move}")

    log.tee(f"  Full path: {_c(path, CYN)}")
    if cap:
        log.tee(f"  Captured:  {_c(cap, ERR)}")
    log.tee("")


# ── Section B: chosen move facts ─────────────────────────────────────────────

_FACT_GROUPS: list[tuple[str, list[str]]] = [
    ("Minimax",   ["minimax_score", "symbolic_rank"]),
    ("Material",  ["captures_count", "net_gain"]),
    ("Safety",    ["opponent_can_recapture", "moved_piece_is_threatened",
                   "our_pieces_threatened_before", "our_pieces_threatened_after"]),
    ("Structure", ["leaves_piece_isolated", "weakens_king_row"]),
    ("Promotion", ["results_in_king", "near_promotion"]),
    ("Tactical",  ["creates_immediate_threat", "shot_sequence_available",
                   "blocks_opponent_landing", "forced_opponent_jump_reply",
                   "max_opponent_jump_captures"]),
    ("Mobility",  ["opponent_mobility_before", "opponent_mobility_after",
                   "our_mobility_before", "our_mobility_after"]),
    ("Activity",  ["king_activity_score", "counterplay_score",
                   "winning_conversion_score", "center_control",
                   "restriction_score", "mobility_reduction"]),
]


def _section_facts(facts: dict, compact: bool, log: TraceLogger) -> None:
    log.tee(_c("  ┌─ B. CHOSEN MOVE FACTS", BOLD + WHT))
    log.tee(_c(RULE, GRY))

    if compact:
        keys = ["minimax_score", "captures_count", "net_gain",
                "opponent_can_recapture", "our_pieces_threatened_after",
                "leaves_piece_isolated", "creates_immediate_threat"]
        for k in keys:
            if k not in facts:
                continue
            v = facts[k]
            log.tee(f"    {_c(k, CYN):<55} {_c(v, _fact_color(k, v))}")
    else:
        for group, keys in _FACT_GROUPS:
            items = [(k, facts[k]) for k in keys if k in facts]
            if not items:
                continue
            log.tee(_c(f"  {group}:", CYN + BOLD))
            for k, v in items:
                log.tee(f"    {_c(k, CYN):<55} {_c(v, _fact_color(k, v))}")

    log.tee("")


# ── Section C: seeds ──────────────────────────────────────────────────────────

def _section_seeds(r_seeds: list, c_seeds: list, log: TraceLogger) -> None:
    log.tee(_c("  ┌─ C. SEEDS", BOLD + WHT))
    log.tee(_c(RULE, GRY))

    if r_seeds:
        log.tee(_c(f"  Reasoning seeds ({len(r_seeds)}):", CYN + BOLD))
        for i, s in enumerate(r_seeds, 1):
            log.tee(f"    {_c(i, GRY)}. {_c(s, CYN)}")
    else:
        log.tee(_c("  Reasoning seeds: (none)", GRY))

    log.tee("")

    if c_seeds:
        log.tee(_c(f"  Comparative seeds ({len(c_seeds)}):", MAG + BOLD))
        for i, s in enumerate(c_seeds, 1):
            log.tee(f"    {_c(i, GRY)}. {_c(s, MAG)}")
    else:
        log.tee(_c("  Comparative seeds: (none)", GRY))

    log.tee("")


# ── Section D: raw reasoning pipeline ────────────────────────────────────────

def _section_reasoning_pipeline(
    diag: dict,
    show_verifier: bool,
    show_refinement: bool,
    log: TraceLogger,
) -> None:
    log.tee(_c("  ┌─ D. RAW REASONING PIPELINE", BOLD + WHT))
    log.tee(_c(RULE, GRY))

    raw_pre            = diag.get("raw_llm_reasoning_pre_refinement") or ""
    chosen_reasoning   = diag.get("chosen_reasoning") or ""
    init_contras       = diag.get("reasoning_initial_contradictions") or []
    final_contras      = diag.get("reasoning_final_contradictions") or []
    retry_count        = diag.get("reasoning_refinement_retry_count", 0)
    has_unresolved     = diag.get("reasoning_has_unresolved_contradiction", False)
    was_repaired       = diag.get("reasoning_contradiction_repaired", False)

    # D1 — pre-refinement reasoning
    log.tee(_c("  Pre-refinement (raw LLM output):", BLU + BOLD))
    if raw_pre:
        log.tee(_wrap(raw_pre))
    else:
        log.tee(_c("    (none)", GRY))
    log.tee("")

    # D2 — initial contradictions
    if show_verifier:
        n = len(init_contras)
        badge = _badge("CLEAN", n == 0)
        log.tee(f"  Contradictions after first verification: {badge} {n} found")
        for i, c in enumerate(init_contras, 1):
            log.tee(f"    {_c(i, GRY)}. {_c(c, ERR)}")
        if init_contras:
            log.tee("")

    # D3 — refinement
    if show_refinement:
        if retry_count > 0:
            log.tee(_c(f"  Refinement attempts: {retry_count}", BLU + BOLD))
            if was_repaired:
                log.tee(_c("  → All contradictions RESOLVED", GRN + BOLD))
            elif has_unresolved:
                log.tee(_c("  → Contradictions UNRESOLVED — keeping unrefined LLM text", YLW))

            diff_lines = _refinement_diff_lines(raw_pre, chosen_reasoning)
            if diff_lines:
                log.tee(_c("  Refinement diff (raw → final):", BLU))
                for dline in diff_lines:
                    if dline.startswith("+"):
                        log.tee(_c(f"    {dline}", GRN))
                    elif dline.startswith("-"):
                        log.tee(_c(f"    {dline}", ERR))
                    else:
                        log.tee(_c(f"    {dline}", GRY))

            log.tee("")
        elif init_contras:
            log.tee(_c("  Refinement: not triggered", GRY))
            log.tee("")

    # D4 — final reasoning
    log.tee(_c("  Final chosen reasoning:", BOLD + WHT))
    if chosen_reasoning:
        log.tee(_wrap(chosen_reasoning))
    else:
        log.tee(_c("    (none)", GRY))
    log.tee("")

    # D5 — final verifier badge
    if show_verifier:
        n = len(final_contras)
        badge = _badge("CLEAN", n == 0)
        log.tee(f"  Final verifier: {badge} {n} contradictions remaining")
        for i, c in enumerate(final_contras, 1):
            log.tee(f"    {_c(i, GRY)}. {_c(c, ERR)}")
        log.tee("")


# ── Section E: comparative reasoning pipeline ────────────────────────────────

def _section_comparative(diag: dict, all_candidates: list, log: TraceLogger) -> None:
    log.tee(_c("  ┌─ E. COMPARATIVE REASONING PIPELINE", BOLD + MAG))
    log.tee(_c(RULE, GRY))

    was_skipped  = diag.get("comparative_was_skipped", True)
    skip_reason  = diag.get("comparative_skip_reason") or "unknown"

    _SKIP_LABELS = {
        "single_legal_move":      "only 1 legal move — no alternatives exist",
        "insufficient_candidates": "only 2 legal moves — need ≥3 for comparison",
        "no_informative_groups":  "no alternatives cluster into informative groups",
        "no_seeds":               "no seeds generated from groups",
        "api_failure":            "API call failed across all attempts",
        "all_samples_rejected":   "all generated samples failed JSON parsing",
        "stage_disabled":         "comparative stage disabled (env flag)",
        "unknown":                "reason not recorded",
    }

    if was_skipped:
        label = _SKIP_LABELS.get(skip_reason, skip_reason)
        log.tee(_c(f"  SKIPPED — {skip_reason}: {label}", YLW))
        log.tee("")
        return

    # Candidate lookup table — resolves [N] bracket references in the prose
    if all_candidates:
        log.tee(_c("  Candidate move index  ([N] = index used in comparative prose):", MAG + BOLD))
        hdr = f"    {'[i]':5}  {'path':24}  {'type':7}  {'recap':6}  {'iso':5}  {'Δopp-mob':8}"
        log.tee(_c(hdr, GRY + BOLD))
        log.tee(_c("    " + "─" * 60, GRY))
        for row in _candidate_table_rows(all_candidates):
            i, path_str, mtype, recap, iso, delta_str = row
            recap_col = ERR if recap == "Y" else GRN
            iso_col   = ERR if iso   == "Y" else GRN
            log.tee(
                f"    {_c(f'[{i}]', BOLD + MAG):<16}"
                f"  {_c(path_str, CYN):<30}"
                f"  {mtype:<7}"
                f"  {_c(recap, recap_col):<16}"
                f"  {_c(iso, iso_col):<15}"
                f"  {_c(delta_str, WHT)}"
            )
        log.tee("")

    # Groups
    groups = diag.get("comparative_groups") or {}
    non_empty = {t: ms for t, ms in groups.items() if ms}
    if non_empty:
        log.tee(_c(f"  Informative groups ({len(non_empty)} themes):", MAG + BOLD))
        for theme, members in non_empty.items():
            idxs = [
                str(m[0]) if isinstance(m, (list, tuple)) else str(m)
                for m in members
            ]
            log.tee(f"    {_c(theme, BOLD + MAG)}: [{', '.join(idxs)}]")
        log.tee("")

    # Seeds
    c_seeds = diag.get("comparative_seeds") or []
    if c_seeds:
        log.tee(_c(f"  Comparative seeds ({len(c_seeds)}):", MAG))
        for i, s in enumerate(c_seeds, 1):
            log.tee(f"    {_c(i, GRY)}. {_c(s, MAG)}")
        log.tee("")

    # Generation stats
    n_samples     = diag.get("comparative_generation_samples_used", 0)
    sample_counts = diag.get("comparative_sample_contradiction_counts") or []
    short_circ    = diag.get("comparative_generation_short_circuited", False)
    log.tee(
        f"  Generation: samples={_c(n_samples, CYN)}"
        f"  contradictions/sample={_c(sample_counts, CYN)}"
        f"  short-circuited={_c(short_circ, YLW if short_circ else GRN)}"
    )
    log.tee("")

    # Refinement stats
    init_c  = diag.get("comparative_initial_contradictions", 0)
    final_c = diag.get("comparative_final_contradictions", 0)
    ref_try = diag.get("comparative_refinement_attempts", 0)
    badge   = _badge("CLEAN", final_c == 0)
    log.tee(
        f"  Comparative contradictions: "
        f"initial={_c(init_c, ERR if init_c > 0 else GRN)}  "
        f"final={badge}({final_c})  "
        f"refinement_attempts={_c(ref_try, BLU)}"
    )
    log.tee("")

    # Final text
    comp_text = diag.get("comparative_paragraph_text") or ""
    if comp_text:
        log.tee(_c("  Final comparative paragraph:", MAG + BOLD))
        log.tee(_wrap(comp_text))
        log.tee("")


# ── Section F: claim extraction diagnostics ──────────────────────────────────

def _section_claims(
    reasoning: str,
    facts: dict,
    seeds: list,
    log: TraceLogger,
) -> list[dict]:
    """Run verify_all and display full claim records. Returns claim dicts."""
    log.tee(_c("  ┌─ F. CLAIM EXTRACTION DIAGNOSTICS", BOLD + WHT))
    log.tee(_c(RULE, GRY))

    if not _VERIFY_AVAILABLE:
        log.tee(_c("  verify_all not available (ImportError)", YLW))
        log.tee("")
        return []
    if not reasoning or not facts:
        log.tee(_c("  No reasoning or facts available.", GRY))
        log.tee("")
        return []

    try:
        records = _verify_all(
            reasoning,
            reasoning_seeds=seeds,
            facts=facts,
        )
    except Exception as exc:
        log.tee(_c(f"  [claim extraction error: {exc}]", YLW))
        log.tee("")
        return []

    def _status(r: Any) -> str:
        s = getattr(r, "claim_status", None)
        return str(s.name if hasattr(s, "name") else s or "UNKNOWN").upper()

    def _hall(r: Any) -> Optional[str]:
        h = getattr(r, "hallucination_type", None)
        if h is None:
            return None
        return str(h.name if hasattr(h, "name") else h)

    supported    = [r for r in records if _status(r) == "SUPPORTED"]
    contradicted = [r for r in records if _status(r) == "CONTRADICTED"]
    vague        = [r for r in records if _status(r) in ("UNSUPPORTED", "NOT_VERIFIABLE")]

    log.tee(
        f"  Claims: total={_c(len(records), WHT)}  "
        f"{_c(f'supported={len(supported)}', GRN)}  "
        f"{_c(f'contradicted={len(contradicted)}', ERR)}  "
        f"{_c(f'vague/unverifiable={len(vague)}', YLW)}"
    )
    log.tee("")

    _HALL_COLOR = {
        "FABRICATED_CLAIM":          ERR,
        "INSTRUCTION_INCONSISTENCY": ERR,
        "VAGUE_CLAIM":               YLW,
        "SCHEMA_LEAK":               ERR,
    }

    def _show_group(title: str, group: list, color: str) -> None:
        if not group:
            return
        log.tee(_c(f"  {title} ({len(group)}):", color + BOLD))
        for r in group:
            text = getattr(r, "matched_phrase", str(r))
            hall = _hall(r)
            hall_str = f"  {_c(f'[{hall}]', _HALL_COLOR.get(hall or '', GRY))}" if hall else ""
            log.tee(f"    • {_c(str(text)[:120], color)}{hall_str}")

    _show_group("SUPPORTED", supported, GRN)
    _show_group("CONTRADICTED", contradicted, ERR)
    _show_group("Vague / non-verifiable", vague, YLW)
    log.tee("")

    claim_dicts: list[dict] = []
    _FIELD_MAP = (
        ("claim_text",       "matched_phrase"),
        ("claim_type",       "claim_type"),
        ("status",           "claim_status"),
        ("hallucination_type", "hallucination_type"),
        ("verifiable_type",  "claim_verifiability"),
    )
    for r in records:
        d: dict[str, Any] = {}
        for key, attr in _FIELD_MAP:
            v = getattr(r, attr, None)
            d[key] = str(v.name if hasattr(v, "name") else v) if v is not None else None
        claim_dicts.append(d)
    return claim_dicts


# ── Full per-turn render ──────────────────────────────────────────────────────

def _render_turn(acc: dict, turn_no: int, args: argparse.Namespace,
                 log: TraceLogger) -> None:
    chosen  = acc.get("chosen_move") or {}
    facts   = chosen.get("facts") or {}
    board   = acc.get("board") or []
    player  = acc.get("current_player", 1)
    diag    = acc.get("ranker_diagnostics") or {}

    r_seeds = diag.get("reasoning_seeds") or []
    c_seeds = diag.get("comparative_seeds") or []

    # A — board + move
    _section_board(board, chosen, turn_no, player, log)

    # B — chosen facts
    if not args.compact or args.verbose:
        _section_facts(facts, args.compact, log)

    # C — seeds
    if args.show_seeds:
        _section_seeds(r_seeds, c_seeds, log)

    # D — reasoning pipeline
    _section_reasoning_pipeline(
        diag, args.show_verifier, args.show_refinement, log,
    )

    # E — comparative pipeline (always render if not skipped, or verbose)
    comp_skipped = diag.get("comparative_was_skipped", True)
    if not comp_skipped or args.verbose:
        all_candidates = acc.get("legal_moves") or []
        _section_comparative(diag, all_candidates, log)

    # F — claim extraction
    claim_dicts: list[dict] = []
    if args.show_claims:
        chosen_only = diag.get("chosen_reasoning") or acc.get("last_move_reasoning") or ""
        claim_dicts = _section_claims(chosen_only, facts, r_seeds, log)

    # ── Per-turn file artifacts ───────────────────────────────────────────────
    comp_artifact = {
        k: diag.get(k) for k in (
            "comparative_paragraph_text", "comparative_seeds",
            "comparative_groups", "comparative_generation_samples_used",
            "comparative_sample_contradiction_counts",
            "comparative_initial_contradictions",
            "comparative_final_contradictions",
            "comparative_refinement_attempts",
            "comparative_was_skipped", "comparative_skip_reason",
        )
    }

    artifacts: dict[str, Any] = {
        "board_before.json":           board,
        "chosen_move.json":            chosen,
        "chosen_facts.json":           facts,
        "seeds.json":                  r_seeds,
        "comparative_seeds.json":      c_seeds,
        "raw_reasoning.txt":           diag.get("raw_llm_reasoning_pre_refinement") or "",
        "final_reasoning.txt":         diag.get("chosen_reasoning") or "",
        "contradictions_initial.json": diag.get("reasoning_initial_contradictions") or [],
        "contradictions_final.json":   diag.get("reasoning_final_contradictions") or [],
        "comparative.json":            comp_artifact,
        "verifier_claims.json":        claim_dicts,
    }

    if args.save_prompts and _PROMPTS_AVAILABLE and r_seeds:
        try:
            prompt = _build_seed_reasoning_prompt(chosen, r_seeds)
            artifacts["prompt_seed_reasoning.txt"] = prompt
        except Exception:
            pass

    log.save_turn(turn_no, artifacts)

    # ── JSONL structured record ───────────────────────────────────────────────
    log.record({
        "turn":         turn_no,
        "player":       _player_label(player),
        "chosen_move":  {"type": chosen.get("type"), "path": chosen.get("path"),
                         "captured": chosen.get("captured")},
        "facts_summary": {
            "minimax_score":              facts.get("minimax_score"),
            "captures_count":             facts.get("captures_count"),
            "net_gain":                   facts.get("net_gain"),
            "opponent_can_recapture":     facts.get("opponent_can_recapture"),
            "our_pieces_threatened_after":facts.get("our_pieces_threatened_after"),
        },
        "n_reasoning_seeds":              len(r_seeds),
        "n_comparative_seeds":            len(c_seeds),
        "initial_contradictions":         len(diag.get("reasoning_initial_contradictions") or []),
        "final_contradictions":           len(diag.get("reasoning_final_contradictions") or []),
        "refinement_attempts":            diag.get("reasoning_refinement_retry_count", 0),
        "was_repaired":                   diag.get("reasoning_contradiction_repaired", False),
        "comparative_skipped":            comp_skipped,
        "comparative_final_contradictions": diag.get("comparative_final_contradictions", 0),
    })
    log.flush()


# ── Post-update board display ─────────────────────────────────────────────────

def _render_post_update(acc: dict, log: TraceLogger) -> None:
    board = acc.get("board") or []
    log.tee(_c(RULE, GRY))
    log.tee(_c("  BOARD AFTER MOVE", BOLD + WHT))
    board_text = _capture_board(board)
    for line in board_text.splitlines():
        log.tee(line)
    rc = count_pieces(board, RED)
    bc = count_pieces(board, BLACK)
    log.tee(
        f"  RED: {rc['total']} ({rc['regular']} reg, {rc['kings']} kings)"
        f"   BLACK: {bc['total']} ({bc['regular']} reg, {bc['kings']} kings)"
    )
    log.tee("")


# ── Graph streaming — one RED ply ─────────────────────────────────────────────

def _stream_red_ply(
    acc: dict,
    args: argparse.Namespace,
    log: TraceLogger,
    recursion_limit: int = 50,
) -> tuple[dict, bool]:
    """
    Stream one RED ply through the simplified graph with deep inspection.
    Modeled directly after run_simplified_trace._stream_one_ply().
    """
    saw_update = False
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": recursion_limit,
    }
    turn_no = (acc.get("turn_number") or 0) + 1

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

                if args.quiet:
                    if node_name == "update_agent":
                        saw_update = True
                    continue

                if node_name == "scorer_node" and args.verbose:
                    lm = acc.get("legal_moves") or []
                    log.tee(_c(
                        f"\n  [scorer_node] {len(lm)} moves scored"
                        f"  best={acc.get('symbolic_best_score')}"
                        f"  gap={acc.get('symbolic_gap')}",
                        DIM,
                    ))

                elif node_name == "deterministic_proposal_node" and not args.compact:
                    cm = acc.get("chosen_move") or {}
                    pd = acc.get("proposal_diagnostics") or {}
                    _gap = pd.get("gap")
                    _n   = pd.get("n_legal", "?")
                    if isinstance(_gap, (int, float)):
                        _gap_str = f"gap={_gap:+.2f}"
                    elif _n == 1 or _n == "1":
                        _gap_str = "gap=N/A (forced — only 1 move)"
                    else:
                        _gap_str = f"gap=N/A (n_legal={_n})"
                    log.tee(
                        _c("  [proposal] ", DIM)
                        + _c(f"CHOSEN: {cm.get('type')} {cm.get('path')}", BOLD + WHT)
                        + _c(
                            f"  score={acc.get('chosen_move_score')}"
                            f"  {_gap_str}"
                            f"  n_legal={_n}",
                            DIM,
                        )
                    )

                elif node_name == "ranker_agent":
                    _render_turn(acc, turn_no, args, log)

                elif node_name == "update_agent":
                    saw_update = True
                    if not args.compact:
                        _render_post_update(acc, log)

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        sys.stderr.write(f"[run_simplified_trace_reasoning] graph error: {exc}\n")

    return acc, saw_update


# ── RED ply driver ────────────────────────────────────────────────────────────

def _run_red_ply(
    acc: dict, args: argparse.Namespace, log: TraceLogger,
) -> dict:
    acc["last_completed_node"] = None
    acc, ok = _stream_red_ply(acc, args, log)
    if not ok:
        sys.stderr.write(
            "[run_simplified_trace_reasoning] warning: graph did not complete update_agent.\n"
        )
    return acc


# ── BLACK ply (human input — identical logic to run_simplified_trace.py) ─────

def _run_black_ply(
    acc: dict, args: argparse.Namespace, log: TraceLogger,
) -> dict:
    board  = acc["board"]
    legal  = get_all_legal_moves(board, BLACK)
    turn_no = (acc.get("turn_number") or 0) + 1

    if not legal:
        sys.stderr.write("[run_simplified_trace_reasoning] BLACK has no legal moves.\n")
        return acc

    if not args.quiet:
        log.tee("")
        log.tee(_c(BAR, GRY))
        log.tee(_c(f"  TURN {turn_no}  │  BLACK to move  (YOU)", BOLD + WHT))
        log.tee(_c(BAR, GRY))
        board_text = _capture_board(board)
        for line in board_text.splitlines():
            log.tee(line)
        log.tee(_c(f"  Available moves ({len(legal)}):", WHT))
        for i, m in enumerate(legal):
            cap = m.get("captured") or []
            cap_s = f"  captures {cap}" if cap else ""
            log.tee(f"  [{i}] {m.get('type')} {m.get('path')}{cap_s}")

    while True:
        try:
            raw = input(
                f"\n  Enter move index [0-{len(legal)-1}]  (q = quit, save logs): "
            ).strip().lower()
        except EOFError:
            raw = "q"
        if raw in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        try:
            k = int(raw)
            if 0 <= k < len(legal):
                break
            print(f"  Out of range — enter 0 to {len(legal)-1}.")
        except ValueError:
            print("  Not a number — try again.")

    move = legal[k]
    if not args.quiet:
        path = move.get("path") or []
        if len(path) >= 2:
            a, b = path[0], path[-1]
            log.tee(_c(f"\n  Applied: {move.get('type')} from {a} to {b}", GRY))

    acc["chosen_move"]         = move
    acc["last_move_reasoning"] = "BLACK human move"

    _valid  = set(CheckersState.model_fields.keys())
    _state  = CheckersState(**{k: v for k, v in acc.items() if k in _valid})
    _result = _update_agent_fn(_state)
    acc.update(_result)

    # Smoke-test — verify move was applied correctly (same check as run_simplified_trace.py)
    def _np(p: Any) -> list:
        return [list(sq) for sq in (p or [])]

    mh = acc.get("move_history") or []
    if mh:
        applied = mh[-1].get("move") or {}
        if (
            applied.get("type") != move.get("type")
            or _np(applied.get("path")) != _np(move.get("path"))
        ):
            sys.stderr.write(
                "[SMOKE TEST FAIL] BLACK applied move does not match chosen!\n"
                f"  chosen : {move.get('type')} {move.get('path')}\n"
                f"  applied: {applied.get('type')} {applied.get('path')}\n"
            )

    return acc


# ── Final summary ─────────────────────────────────────────────────────────────

def _print_final_summary(state: dict, log: TraceLogger) -> None:
    w = state.get("winner")
    draw = state.get("draw", False)
    winner_text = "Draw" if draw else ("RED" if w == RED else ("BLACK" if w == BLACK else "N/A"))
    gid = state.get("game_log_id") or "?"

    log.tee("")
    log.tee(_c(BAR, CYN + BOLD))
    log.tee(_c("  GAME OVER", BOLD + WHT))
    log.tee(_c(BAR, CYN + BOLD))
    log.tee(f"  Winner:      {_c(winner_text, GRN + BOLD)}")
    log.tee(f"  Total turns: {state.get('turn_number')}")
    log.tee(f"  Game log:    logs/{gid}.jsonl")
    log.tee(f"  Trace dir:   {log.run_dir}")
    log.tee("")


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _graceful_exit(
    acc: dict,
    log: TraceLogger,
    run_dir: Path,
    reason: str = "interrupted",
) -> None:
    print(_c("\n  Saving trace data...", YLW))
    log.flush()
    log.write_summary({
        "status":           reason,
        "turns_completed":  acc.get("turn_number", 0),
        "run_dir":          str(run_dir),
        "game_over":        acc.get("game_over", False),
        "winner":           acc.get("winner"),
    })
    print(_c(f"  Trace saved → {run_dir}", GRN))


# ── Global refs for signal handler ───────────────────────────────────────────
_G_LOG: Optional[TraceLogger] = None
_G_ACC: Optional[dict] = None


def _signal_handler(*_: Any) -> None:
    if _G_LOG is not None and _G_ACC is not None:
        _graceful_exit(_G_ACC, _G_LOG, _G_LOG.run_dir, reason="signal")
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _G_LOG, _G_ACC

    args = _build_parser().parse_args()

    # Apply comparative env override (must happen before checkers_graph import)
    if args.comparative is True:
        os.environ["RANKER_COMPARATIVE_STAGE_ENABLED"] = "1"
    elif args.comparative is False:
        os.environ["RANKER_COMPARATIVE_STAGE_ENABLED"] = "0"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path("logs") / "manual_reasoning_trace" / timestamp
    log       = TraceLogger(run_dir)
    _G_LOG    = log

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if not args.quiet:
        comp_state = os.environ.get("RANKER_COMPARATIVE_STAGE_ENABLED", "1")
        comp_label = "OFF" if comp_state.lower() in ("0", "false", "no", "off") else "ON"
        log.tee("")
        log.tee(_c(BAR, CYN + BOLD))
        log.tee(_c("  REASONING INSPECTION MODE  │  simplified pipeline", BOLD + WHT))
        log.tee(_c(f"  Run directory: {run_dir}", DIM))
        log.tee(_c(
            f"  MINIMAX={os.environ.get('MINIMAX_ENABLED','?')}"
            f"  DEPTH={os.environ.get('MINIMAX_DEPTH','?')}"
            f"  COMPARATIVE={comp_label}"
            f"  SEEDS={'OFF' if os.environ.get('RANKER_SEEDS_DISABLED','').lower() in ('1','true','yes','on') else 'ON'}",
            DIM,
        ))
        log.tee(_c(BAR, CYN + BOLD))
        log.tee("")

    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()
    _G_ACC = acc

    try:
        while True:
            _G_ACC = acc

            if acc.get("game_over"):
                _print_final_summary(acc, log)
                break

            if (acc.get("turn_number") or 0) >= args.max_turns:
                sys.stderr.write("[run_simplified_trace_reasoning] max turns reached.\n")
                if not args.quiet:
                    log.tee(_c("  GAME INCOMPLETE: max turns reached.", YLW))
                break

            if acc["current_player"] == RED:
                acc = _run_red_ply(acc, args, log)
            else:
                acc = _run_black_ply(acc, args, log)

    except KeyboardInterrupt:
        _graceful_exit(acc, log, run_dir, reason="keyboard_interrupt")
        return

    log.flush()
    log.write_summary({
        "status":       "complete",
        "turns_completed": acc.get("turn_number", 0),
        "run_dir":      str(run_dir),
        "game_over":    acc.get("game_over", False),
        "winner":       acc.get("winner"),
        "winner_label": (
            _player_label(acc["winner"])
            if acc.get("winner") is not None
            else "draw"
        ),
    })


if __name__ == "__main__":
    main()
