#!/usr/bin/env python3
"""
run_presentation_trace.py — Presentation-oriented trace runner.

Identical system behaviour to run_simplified_trace.py:
  RED = AI (scorer → proposer → explainer → updater pipeline)
  BLACK = human terminal input

Output is designed for thesis demonstrations and slide screenshots:
  ● Colored board with clear piece symbols
  ● Titled panels for each pipeline stage
  ● Color-coded verifier findings (red = contradiction, green = clean/repaired)
  ● No JSON dumps, no internal node names, no debug traces

Automatic export — every completed AI turn is saved to:
  presentation_traces/<game_id>/turn_NNN.html   (browser-ready, colored)
  presentation_traces/<game_id>/turn_NNN.txt    (plain text, no ANSI)

Usage:
  python run_presentation_trace.py [--max-turns N]
"""

from __future__ import annotations

# ── Environment — must be set before importing the graph ──────────────────────
import os
os.environ["USE_SIMPLIFIED_PIPELINE"] = "true"
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import argparse
import datetime
import difflib
import html as _html
import pathlib
import re
import sys
import textwrap
import uuid
from typing import Any, Optional

from checkers.graph.graph import checkers_graph
from checkers.state.state import CheckersState
from checkers.agents.updater_agent import updater_agent as _update_agent_fn
from checkers.engine.board import (
    RED, BLACK, EMPTY, RED_KING, BLACK_KING,
    create_initial_board,
)
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves


# ── ANSI palette ──────────────────────────────────────────────────────────────

class _C:
    RST  = "\033[0m"
    BOLD = "\033[1m"
    DIM  = "\033[2m"
    RED  = "\033[91m"
    GRN  = "\033[92m"
    YLW  = "\033[93m"
    BLU  = "\033[94m"
    MAG  = "\033[95m"
    CYN  = "\033[96m"
    WHT  = "\033[97m"


# ── Layout constants ──────────────────────────────────────────────────────────

_W = 74          # total panel width including border characters


def _hrule(char: str = "═", width: int = _W) -> str:
    return char * width


# ── Panel builder ─────────────────────────────────────────────────────────────

def _box(title: str, lines: list[str], title_color: str = _C.WHT, width: int = _W) -> str:
    """
    ╭── TITLE ──────────────────────────────────────────────────╮
    │  line 1                                                   │
    ╰───────────────────────────────────────────────────────────╯
    """
    inner = width - 2
    if title:
        t = f"  {title}  "
        dash_total = inner - len(title) - 4
        dl = dash_total // 2
        dr = dash_total - dl
        top = f"╭{'─' * dl}{title_color}{_C.BOLD}{t}{_C.RST}{'─' * dr}╮"
    else:
        top = f"╭{'─' * inner}╮"

    parts = [top]
    for line in lines:
        pad = max(0, inner - 2 - _visible_len(line))
        parts.append(f"│  {line}{' ' * pad}│")
    parts.append(f"╰{'─' * inner}╯")
    return "\n".join(parts)


def _visible_len(s: str) -> int:
    """String length ignoring ANSI escape sequences."""
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


# ── Board renderer (ANSI terminal) ────────────────────────────────────────────

_PIECE_GLYPH: dict[int, str] = {
    EMPTY:      "   ",
    RED:        f" {_C.BOLD}{_C.RED}r{_C.RST} ",
    RED_KING:   f" {_C.BOLD}{_C.RED}R{_C.RST} ",
    BLACK:      f" {_C.BOLD}{_C.CYN}b{_C.RST} ",
    BLACK_KING: f" {_C.BOLD}{_C.CYN}B{_C.RST} ",
}
_DARK_EMPTY = f" {_C.DIM}·{_C.RST} "


def _render_board(board: list[list[int]], highlight: Optional[list] = None) -> list[str]:
    """Board as ANSI-colored text lines for terminal display."""
    hl_set: set[tuple[int, int]] = set()
    if highlight:
        for sq in highlight:
            if isinstance(sq, (list, tuple)) and len(sq) >= 2:
                hl_set.add((int(sq[0]), int(sq[1])))

    col_header = "      " + "   ".join(str(c) for c in range(8))
    sep_top    = "    ┌" + "───┬" * 7 + "───┐"
    sep_mid    = "    ├" + "───┼" * 7 + "───┤"
    sep_bot    = "    └" + "───┴" * 7 + "───┘"

    lines = [col_header, sep_top]
    for r in range(8):
        cells = []
        for c in range(8):
            piece = board[r][c]
            if (r, c) in hl_set:
                glyph_raw = _PIECE_GLYPH.get(piece, "???")
                cells.append(f"{_C.YLW}[{_C.RST}{glyph_raw}{_C.YLW}]{_C.RST}")
            elif (r + c) % 2 == 1:
                cells.append(_PIECE_GLYPH.get(piece, _DARK_EMPTY) if piece != EMPTY else _DARK_EMPTY)
            else:
                cells.append("   ")
        lines.append(f" {r:2d} │" + "│".join(cells) + "│")
        if r < 7:
            lines.append(sep_mid)
    lines.append(sep_bot)
    lines.append("")
    lines.append(
        f"     {_C.BOLD}{_C.RED}r{_C.RST}=red   "
        f"{_C.BOLD}{_C.RED}R{_C.RST}=red king   "
        f"{_C.BOLD}{_C.CYN}b{_C.RST}=black   "
        f"{_C.BOLD}{_C.CYN}B{_C.RST}=black king"
    )
    return lines


# ── Board renderer (plain text — no ANSI, for .txt files) ────────────────────

_PIECE_PLAIN: dict[int, str] = {
    EMPTY:      " · ",
    RED:        " r ",
    RED_KING:   " R ",
    BLACK:      " b ",
    BLACK_KING: " B ",
}
_LIGHT_PLAIN = "   "


def _board_to_plain(board: list[list[int]], highlight: Optional[list] = None) -> str:
    """Board as plain ASCII text (no ANSI), for .txt output."""
    hl_set: set[tuple[int, int]] = set()
    if highlight:
        for sq in highlight:
            if isinstance(sq, (list, tuple)) and len(sq) >= 2:
                hl_set.add((int(sq[0]), int(sq[1])))

    lines = ["      0   1   2   3   4   5   6   7",
             "   ┌───┬───┬───┬───┬───┬───┬───┬───┐"]
    for r in range(8):
        cells = []
        for c in range(8):
            piece = board[r][c]
            if (r + c) % 2 == 1:
                glyph = _PIECE_PLAIN.get(piece, " ? ")
                cells.append(f"[{glyph[1]}]" if (r, c) in hl_set else glyph)
            else:
                cells.append(_LIGHT_PLAIN)
        lines.append(f" {r:2d} │" + "│".join(cells) + "│")
        if r < 7:
            lines.append("   ├───┼───┼───┼───┼───┼───┼───┤")
    lines.append("   └───┴───┴───┴───┴───┴───┴───┘")
    lines.append("   r=red  R=red king  b=black  B=black king  [x]=move path")
    return "\n".join(lines)


# ── Board renderer (HTML, for .html files) ────────────────────────────────────

_PIECE_HTML: dict[int, str] = {
    EMPTY:      '<span class="dim">·</span>',
    RED:        '<span class="r">r</span>',
    RED_KING:   '<span class="r">R</span>',
    BLACK:      '<span class="b">b</span>',
    BLACK_KING: '<span class="b">B</span>',
}


def _board_to_html(board: list[list[int]], highlight: Optional[list] = None) -> str:
    """Board as an HTML <pre> block with colored piece spans."""
    hl_set: set[tuple[int, int]] = set()
    if highlight:
        for sq in highlight:
            if isinstance(sq, (list, tuple)) and len(sq) >= 2:
                hl_set.add((int(sq[0]), int(sq[1])))

    # Note: box-drawing characters are safe HTML; no escaping needed.
    lines = ["      0   1   2   3   4   5   6   7",
             "   ┌───┬───┬───┬───┬───┬───┬───┬───┐"]
    for r in range(8):
        cells_html = []
        for c in range(8):
            piece = board[r][c]
            if (r + c) % 2 == 1:
                glyph = _PIECE_HTML.get(piece, '<span class="dim">·</span>')
                if (r, c) in hl_set:
                    cells_html.append(f'<span class="hl"> {glyph} </span>')
                else:
                    cells_html.append(f" {glyph} ")
            else:
                cells_html.append("   ")
        lines.append(f" {r:2d} │" + "│".join(cells_html) + "│")
        if r < 7:
            lines.append("   ├───┼───┼───┼───┼───┼───┼───┤")
    lines.append("   └───┴───┴───┴───┴───┴───┴───┘")
    legend = (
        '   <span class="r">r</span>=red  '
        '<span class="r">R</span>=red king  '
        '<span class="b">b</span>=black  '
        '<span class="b">B</span>=black king  '
        '<span class="hl">[ ]</span>=move path'
    )
    lines.append(legend)
    return '<pre class="board">' + "\n".join(lines) + "</pre>"


# ── Move formatting ───────────────────────────────────────────────────────────

def _fmt_coord(sq: Any) -> str:
    try:
        return f"({int(sq[0])}, {int(sq[1])})"
    except (TypeError, IndexError, ValueError):
        return str(sq)


def _fmt_move_human(move: dict[str, Any]) -> str:
    path  = move.get("path") or []
    mtype = move.get("type", "simple")
    cap   = move.get("captured") or []
    if len(path) >= 2:
        kind = "JUMP" if mtype == "jump" else "MOVE"
        if len(path) > 2:
            path_str = "  →  ".join(_fmt_coord(sq) for sq in path)
        else:
            path_str = f"{_fmt_coord(path[0])}  →  {_fmt_coord(path[-1])}"
        base = f"{kind}  {path_str}"
    else:
        base = str(move)
    if cap:
        base += "   ·   captures " + ", ".join(_fmt_coord(c) for c in cap)
    return base


def _fmt_score_state_ansi(ss: str) -> str:
    labels = {
        "CLEARLY_WINNING":  f"{_C.GRN}Clearly winning{_C.RST}",
        "SLIGHTLY_WINNING": f"{_C.GRN}Slightly winning{_C.RST}",
        "EQUAL":            f"{_C.YLW}Equal position{_C.RST}",
        "SLIGHTLY_LOSING":  f"{_C.RED}Slightly losing{_C.RST}",
        "CLEARLY_LOSING":   f"{_C.RED}Clearly losing{_C.RST}",
    }
    return labels.get(ss, ss)


def _fmt_score_state_plain(ss: str) -> str:
    labels = {
        "CLEARLY_WINNING":  "Clearly winning",
        "SLIGHTLY_WINNING": "Slightly winning",
        "EQUAL":            "Equal position",
        "SLIGHTLY_LOSING":  "Slightly losing",
        "CLEARLY_LOSING":   "Clearly losing",
    }
    return labels.get(ss, ss)


def _score_state_html_class(ss: str) -> str:
    if "WINNING" in ss:
        return "pos-w"
    if "LOSING" in ss:
        return "pos-l"
    return "pos-e"


def _clean_contradiction(raw: str) -> str:
    """Strip internal prefixes; return human-readable contradiction text."""
    s = raw
    for prefix in ("REASONING_CONTRADICTION: ", "COMPARATIVE_CONTRADICTION: "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s[:1].upper() + s[1:] if s else s


_HUMANIZE_PATTERNS: list[tuple[str, str]] = [
    # Ordered most-specific first so overlapping tokens don't shadow each other.
    ("forced_opponent_jump_reply=false",
     "Unsupported forced-reply claim — the opponent was not required to respond with a jump"),
    ("negative_fact_omission",
     "Negative fact omitted — the explanation did not mention a computed decrease in mobility"),
    ("opponent_can_recapture=true",
     "Recapture risk understated — the moved piece can actually be recaptured by the opponent"),
    ("opponent_can_recapture=false",
     "False recapture claim — the opponent cannot actually recapture the moved piece"),
    ("near_promotion=false",
     "Promotion proximity overstated — the piece is not near a promotion square after this move"),
    ("captures_count=0",
     "False capture claim — this move does not capture any piece"),
    ("results_in_king=false",
     "False promotion claim — this move does not result in a king"),
    ("forced_move_for_us=false",
     "False forced-move claim — other legal moves were available at this position"),
    ("creates_immediate_threat=false",
     "False threat claim — this move does not create an immediate tactical threat"),
    ("any_piece_isolated=true",
     "Isolation risk understated — a piece is left isolated after this move, contradicting the claimed safety"),
    ("gap_did_not_narrow",
     "Incorrect mobility claim — the mobility gap did not narrow after this move"),
    ("gap_did_not_widen",
     "Incorrect mobility claim — the mobility gap did not widen after this move"),
    ("mobility_unchanged_misclaim",
     "Incorrect mobility claim — mobility did change for one or both players after this move"),
    ("inversion detected",
     "Inverted claim — the explanation states the opposite of what the engine computed"),
    ("fabricated comparison value",
     "Unsupported comparison — the explanation references a value not computed by the engine"),
    ("fabricated_claim",
     "Claim contradicted by board facts — the explanation contains a statement not supported by computed facts"),
    ("factual_contradiction",
     "Claim contradicted by board facts — a statement in the explanation conflicts with computed facts"),
    ("forbidden term",
     "Unsupported term used — this phrase is not grounded in the engine's computed facts"),
    ("not found in seeds",
     "Unsupported term used — this phrase is not grounded in the engine's computed facts"),
    ("used but not in seeds",
     "Unsupported term used — this phrase is not grounded in the engine's computed facts"),
    ("unsupported numeric statement",
     "Unsupported numeric claim — this number is not present in the engine's computed facts"),
    ("unsupported numeric assertion",
     "Unsupported numeric claim — this number is not present in the engine's computed facts"),
    ("unsupported absence claim",
     "Unsupported absence claim — the engine's facts do not support this negative assertion"),
    ("captures_count=",
     "Incorrect capture count — the explanation claims a different number of captures than actually occurred"),
    ("overclaims mobility",
     "Mobility overclaimed — the explanation exaggerates the mobility advantage"),
    ("claims mobility reduction",
     "Unsupported mobility claim — the engine does not confirm a reduction in opponent mobility"),
    ("our-mobility increase",
     "Unsupported mobility claim — the engine does not show an increase in our mobility"),
    ("narrowing the gap",
     "Incorrect gap claim — the explanation claims the mobility gap narrows, but board facts show otherwise"),
    ("claims mobility unchanged",
     "Incorrect mobility claim — mobility changed for at least one player after this move"),
    ("claims 'gap narrowed'",
     "Incorrect mobility claim — the mobility gap did not narrow after this move"),
    ("claims 'gap widened'",
     "Incorrect mobility claim — the mobility gap did not widen after this move"),
    ("deliberate-choice framing",
     "Unsupported deliberate-choice claim — the move was selected deterministically by the engine"),
    ("geometrically impossible",
     "Geometrically impossible claim — the described board configuration does not match computed facts"),
    ("numeric mismatch",
     "Numeric mismatch — an explanation value conflicts with the engine's computed value"),
    ("schema-leak",
     "Internal term in explanation — a technical field name appeared in the reasoning text"),
    ("contradicted by facts",
     "Claim contradicted by board facts — the explanation contains a statement not supported by computed facts"),
    ("specific safe-reply count",
     "Unsupported reply count — the explanation references a specific number of opponent replies not computed by the engine"),
    ("single opponent jump",
     "Incorrect opponent jump claim — the opponent has more than one available jump"),
    ("used but this is the only",
     "Inappropriate forced-move framing — this phrase implies a constraint that does not apply here"),
    ("center of the board",
     "Unsupported center-control claim — the engine's facts do not support this board-center claim"),
    ("without numeric",
     "Unsupported comparative claim — a comparison was made without numerical support from the engine"),
    ("our-mobility decrease seeded but",
     "Negative fact omitted — the explanation did not mention the computed decrease in our mobility"),
]


def _humanize_contradiction(raw: str) -> str:
    """
    Translate an internal verifier contradiction string to plain English.
    Strips the prefix, then maps known technical patterns to display-ready
    sentences.  Falls back to _clean_contradiction for unrecognised strings.
    """
    s = raw
    for prefix in ("REASONING_CONTRADICTION: ", "COMPARATIVE_CONTRADICTION: "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    for token, human in _HUMANIZE_PATTERNS:
        if token in s:
            return human

    # Fallback: prefix already stripped, just capitalize
    return s[:1].upper() + s[1:] if s else s


def _strip_alt_indices(text: str) -> str:
    """Remove LLM-generated alternative index refs like [1], [2], and [3]."""
    text = re.sub(r'\b(?:and|or)\s+\[\d+\]', '', text)  # "and [N]" → ""
    text = re.sub(r'\[\d+\](?:,\s*)?', '', text)         # "[N]," or "[N]" → ""
    text = re.sub(r'  +', ' ', text)                      # collapse double spaces
    return text.strip()


# ── Turn record — collected from stream snapshots ─────────────────────────────

def _collect_turn_record(
    board_before: list[list[int]],
    capture: dict[str, Any],
    acc: dict[str, Any],
    turn_no: int,
) -> dict[str, Any]:
    """
    Build a self-contained turn record from:
      board_before — snapshot taken at the start of _run_red_ply
      capture      — fields snapshotted mid-stream (before state_manager clears them)
      acc          — final state after updater_agent completes
      turn_no      — display turn number (1-indexed)
    """
    mh       = acc.get("move_history") or []
    last_mh  = mh[-1] if mh else {}
    move     = last_mh.get("move") or {}

    diag     = capture.get("explainer_diagnostics") or {}
    chosen_r = (diag.get("chosen_reasoning") or capture.get("last_move_reasoning") or "").strip()
    comp_r   = _strip_alt_indices((diag.get("comparative_paragraph_text") or "").strip())

    rc = count_pieces(acc["board"], RED)
    bc = count_pieces(acc["board"], BLACK)

    # Move path from move_history (reliable even after state_manager clears chosen_move)
    move_path = move.get("path") or []

    return {
        "turn_number":       turn_no,
        "board_before":      [row[:] for row in board_before],
        "board_after":       [row[:] for row in acc["board"]],
        "move_path":         move_path,
        "move": {
            "type":      move.get("type", "?"),
            "path":      move_path,
            "captured":  move.get("captured") or [],
        },
        "chosen_move_score": capture.get("chosen_move_score"),
        "score_state":       capture.get("score_state", acc.get("score_state", "EQUAL")),
        "n_legal":           (capture.get("proposer_diagnostics") or {}).get("n_legal", 0),
        "unchosen_moves": [
            {
                "index":         i + 1,
                "type":          m.get("type", "?"),
                "path":          m.get("path") or [],
                "captured":      m.get("captured") or [],
                "minimax_score": (m.get("facts") or {}).get("minimax_score"),
                "facts":         m.get("facts") or {},
            }
            for i, m in enumerate(capture.get("unchosen_moves") or [])
        ],
        "explanation":   chosen_r,
        "comparative":   comp_r,
        "verifier": {
            "contradiction_detected":  diag.get("reasoning_contradiction_detected", False),
            "contradiction_repaired":  diag.get("reasoning_contradiction_repaired", False),
            "n_retries":               diag.get("reasoning_refinement_retry_count", 0),
            "initial_contradictions": [
                _humanize_contradiction(c)
                for c in (diag.get("reasoning_initial_contradictions") or [])
            ],
            "final_contradictions": [
                _humanize_contradiction(c)
                for c in (diag.get("reasoning_final_contradictions") or [])
            ],
            "explanation_before_repair": (
                diag.get("raw_llm_reasoning_pre_refinement") or ""
            ).strip(),
            "explanation_after_repair": chosen_r,
        },
        "promotion":        bool(last_mh.get("promotion")),
        "piece_counts": {
            "red":   rc,
            "black": bc,
        },
    }


# ── Plain text export ─────────────────────────────────────────────────────────

_TXT_W = 70


def _txt_section(title: str, lines: list[str]) -> str:
    bar = "─" * _TXT_W
    body = "\n".join(f"  {l}" for l in lines)
    return f"\n{title}\n{bar}\n{body}\n"


def _turn_to_txt(record: dict[str, Any], game_id: str) -> str:
    tn    = record["turn_number"]
    ss    = record["score_state"]
    score = record["chosen_move_score"]
    move  = record["move"]
    ver   = record["verifier"]

    parts: list[str] = []
    parts.append("=" * _TXT_W)
    parts.append(f"  TURN {tn}  |  RED (AI)  |  {game_id}")
    parts.append("=" * _TXT_W)

    # Board before
    parts.append(_txt_section(
        "BOARD BEFORE MOVE",
        _board_to_plain(record["board_before"]).splitlines(),
    ))

    # Selected move
    move_lines = [_fmt_move_human(move)]
    if score is not None:
        move_lines.append(f"Engine score:  {score:+.2f}")
    move_lines.append(f"Material:      {_fmt_score_state_plain(ss)}")
    n_alt = record["n_legal"] - 1
    if n_alt > 0:
        move_lines.append(f"Alternatives:  {n_alt} other legal move(s) considered")
    if record["unchosen_moves"]:
        move_lines.append("Alternatives (not chosen):")
        for m in record["unchosen_moves"]:
            idx = m.get("index", "?")
            s2 = f"  [{m['minimax_score']:+.2f}]" if m.get("minimax_score") is not None else ""
            move_lines.append(f"  [{idx}] {_fmt_move_human(m)}{s2}")
    parts.append(_txt_section("SELECTED MOVE", move_lines))

    # Explanation
    expl = record["explanation"] or "(none)"
    wrapped = textwrap.wrap(expl, width=_TXT_W - 4)
    if record["comparative"]:
        wrapped += ["", "Why not the alternatives:"]
        wrapped += textwrap.wrap(record["comparative"], width=_TXT_W - 4)
    parts.append(_txt_section("AI EXPLANATION", wrapped))

    # Verifier
    vlines: list[str] = []
    if not ver["contradiction_detected"]:
        vlines.append("[OK] No contradictions detected — explanation accepted as-is.")
    else:
        vlines.append(f"[CONTRADICTION] {len(ver['initial_contradictions'])} issue(s) detected:")
        for i, c in enumerate(ver["initial_contradictions"], 1):
            for j, wl in enumerate(textwrap.wrap(c, width=_TXT_W - 8)):
                vlines.append(f"  {'%d.' % i if j == 0 else '   '} {wl}")
        vlines.append("")
        if ver["contradiction_repaired"]:
            vlines.append(f"[REPAIRED] Resolved in {ver['n_retries']} refinement attempt(s).")
            before = ver["explanation_before_repair"]
            after  = ver["explanation_after_repair"]
            if before and after and before != after:
                vlines.append("")
                vlines.append("BEFORE repair:")
                vlines += [f"  {l}" for l in textwrap.wrap(before, width=_TXT_W - 6)]
                vlines.append("")
                vlines.append("AFTER repair:")
                vlines += [f"  {l}" for l in textwrap.wrap(after, width=_TXT_W - 6)]
        else:
            vlines.append(
                f"[UNRESOLVED] Contradictions remain after {ver['n_retries']} attempt(s)."
            )
            for c in ver["final_contradictions"][:3]:
                vlines.append(f"  · {c}")
    parts.append(_txt_section("VERIFIER FINDINGS", vlines))

    # Board after
    parts.append(_txt_section(
        "BOARD AFTER MOVE",
        _board_to_plain(record["board_after"], highlight=record["move_path"]).splitlines(),
    ))

    # Piece counts
    rc = record["piece_counts"]["red"]
    bc = record["piece_counts"]["black"]
    promo = "  ★ Promotion to king this turn!" if record["promotion"] else ""
    pc_lines = [
        f"RED:   {rc['total']:2d} pieces  ({rc['regular']} regular, {rc['kings']} kings)",
        f"BLACK: {bc['total']:2d} pieces  ({bc['regular']} regular, {bc['kings']} kings)",
    ]
    if promo:
        pc_lines.append(promo)
    parts.append(_txt_section("PIECE COUNT", pc_lines))

    parts.append("=" * _TXT_W)
    return "\n".join(parts)


# ── HTML export ───────────────────────────────────────────────────────────────

_HTML_CSS = """\
:root {
  --bg:#1e1e2e; --surface:#24273a; --border:#45475a;
  --text:#cdd6f4; --dim:#6c7086; --subtext:#a6adc8;
  --red:#f38ba8; --green:#a6e3a1; --yellow:#f9e2af;
  --cyan:#89dceb; --mauve:#cba6f7; --blue:#89b4fa;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:"Courier New",Courier,monospace;
     font-size:14px;padding:2em 2.5em;line-height:1.5;max-width:900px;}
h1{color:var(--mauve);font-size:1.5em;font-weight:bold;
   margin-bottom:.25em;letter-spacing:.03em;}
.subtitle{color:var(--dim);font-size:.85em;margin-bottom:1.8em;}
.panel{border:1px solid var(--border);border-radius:6px;
       padding:1em 1.2em;margin-bottom:1.2em;}
.pt{font-weight:bold;letter-spacing:.1em;font-size:.8em;
    text-transform:uppercase;margin-bottom:.6em;color:var(--subtext);}
.board{font-family:monospace;white-space:pre;line-height:1.8;
       font-size:13px;}
.r{color:var(--red);font-weight:bold;}
.b{color:var(--cyan);font-weight:bold;}
.dim{color:var(--dim);}
.hl{background:rgba(249,226,175,.18);color:var(--yellow);font-weight:bold;}
.move-desc{color:var(--yellow);font-weight:bold;font-size:1.05em;
           margin-bottom:.5em;}
.kv{display:flex;gap:1em;margin:.2em 0;}
.kv .k{color:var(--dim);min-width:140px;}
.kv .v{color:var(--text);}
.score-v{color:var(--yellow);}
.pos-w{color:var(--green);}
.pos-e{color:var(--yellow);}
.pos-l{color:var(--red);}
.alt-list{margin-top:.6em;}
.alt{color:var(--dim);margin:.15em 0;}
.expl{line-height:1.75;color:var(--text);}
.comp{color:var(--dim);margin-top:.8em;font-style:italic;}
.comp-label{color:var(--dim);font-size:.85em;margin-bottom:.3em;}
.ok-badge{color:var(--green);font-weight:bold;}
.warn-badge{color:var(--red);font-weight:bold;}
.ok-repair{color:var(--green);}
.fail-repair{color:var(--red);}
.contradiction-list{margin:.4em 0 .4em 1em;}
.contradiction-list li{color:var(--red);margin:.2em 0;}
.before-label{color:var(--dim);font-size:.85em;margin:.8em 0 .2em;}
.after-label{color:var(--green);font-size:.85em;margin:.8em 0 .2em;}
.before-text{color:var(--dim);font-style:italic;line-height:1.7;}
.after-text{color:var(--green);line-height:1.7;}
.promo{color:var(--yellow);font-weight:bold;margin-top:.4em;}
.pc-row{margin:.15em 0;}
.red-label{color:var(--red);font-weight:bold;}
.black-label{color:var(--cyan);font-weight:bold;}
"""


def _turn_to_html(record: dict[str, Any], game_id: str) -> str:
    tn    = record["turn_number"]
    ss    = record["score_state"]
    score = record["chosen_move_score"]
    move  = record["move"]
    ver   = record["verifier"]
    rc    = record["piece_counts"]["red"]
    bc    = record["piece_counts"]["black"]
    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def e(s: Any) -> str:
        """HTML-escape a value."""
        return _html.escape(str(s))

    # ── Board before ──────────────────────────────────────────────────────
    board_before_html = _board_to_html(record["board_before"])
    board_after_html  = _board_to_html(record["board_after"], highlight=record["move_path"])

    # ── Move section ──────────────────────────────────────────────────────
    move_desc = e(_fmt_move_human(move))
    score_str = f'<span class="score-v">{score:+.2f}</span>' if score is not None else "n/a"
    ss_class  = _score_state_html_class(ss)
    n_alt     = record["n_legal"] - 1

    alt_html = ""
    if record["unchosen_moves"]:
        items = []
        for m in record["unchosen_moves"]:
            idx = m.get("index", "?")
            ms = f'  <span class="score-v">[{m["minimax_score"]:+.2f}]</span>' \
                 if m.get("minimax_score") is not None else ""
            items.append(
                f'<div class="alt"><span class="dim">[{e(str(idx))}]</span>'
                f' {e(_fmt_move_human(m))}{ms}</div>'
            )
        alt_html = (
            '<div class="alt-list">'
            '<div class="dim" style="margin-bottom:.3em;font-size:.85em;">'
            'Alternatives (not chosen):</div>'
            + "".join(items) + "</div>"
        )

    move_section = f"""\
<div class="panel">
  <div class="pt">Selected Move</div>
  <div class="move-desc">{move_desc}</div>
  <div class="kv"><span class="k">Engine score</span><span class="v">{score_str}</span></div>
  <div class="kv"><span class="k">Material balance</span>
       <span class="v {ss_class}">{e(_fmt_score_state_plain(ss))}</span></div>
  {"<div class='kv'><span class='k'>Alternatives</span>"
   f"<span class='v'>{n_alt} other legal move(s) considered</span></div>"
   if n_alt > 0 else ""}
  {alt_html}
</div>"""

    # ── Explanation section ───────────────────────────────────────────────
    expl_text = e(record["explanation"]) if record["explanation"] else "<em>(none)</em>"
    comp_html = ""
    if record["comparative"]:
        comp_html = (
            '<div class="comp">'
            '<div class="comp-label">Why not the alternatives:</div>'
            + e(record["comparative"])
            + "</div>"
        )
    expl_section = f"""\
<div class="panel">
  <div class="pt">AI Explanation</div>
  <div class="expl">{expl_text}</div>
  {comp_html}
</div>"""

    # ── Verifier section ─────────────────────────────────────────────────
    if not ver["contradiction_detected"]:
        verifier_body = '<span class="ok-badge">✓</span>  No contradictions detected — explanation accepted as-is.'
    else:
        n_c  = len(ver["initial_contradictions"])
        items = "".join(f"<li>{e(c)}</li>" for c in ver["initial_contradictions"])
        contr_block = (
            f'<span class="warn-badge">⚠</span>  {n_c} contradiction(s) detected:'
            f'<ul class="contradiction-list">{items}</ul>'
        )
        if ver["contradiction_repaired"]:
            repair_status = (
                f'<div class="ok-repair">✓  Repaired successfully '
                f'({ver["n_retries"]} refinement attempt(s)).</div>'
            )
            before = ver["explanation_before_repair"]
            after  = ver["explanation_after_repair"]
            before_after = ""
            if before and after and before != after:
                before_after = (
                    '<div class="before-label">BEFORE repair:</div>'
                    f'<div class="before-text">{e(before)}</div>'
                    '<div class="after-label">AFTER repair:</div>'
                    f'<div class="after-text">{e(after)}</div>'
                )
        else:
            remaining = "".join(
                f"<li>{e(c)}</li>" for c in ver["final_contradictions"][:3]
            )
            repair_status = (
                f'<div class="fail-repair">✗  Contradictions could not be fully resolved '
                f'after {ver["n_retries"]} attempt(s).</div>'
                f'<ul class="contradiction-list">{remaining}</ul>'
            )
            before_after = ""
        verifier_body = contr_block + repair_status + before_after

    verifier_section = f"""\
<div class="panel">
  <div class="pt">Verifier Findings</div>
  {verifier_body}
</div>"""

    # ── Piece count section ───────────────────────────────────────────────
    promo_html = '<div class="promo">★  Promoted to king this turn!</div>' \
                 if record["promotion"] else ""
    pc_section = f"""\
<div class="panel">
  <div class="pt">Piece Count after Move</div>
  <div class="pc-row">
    <span class="red-label">RED  </span>
    {rc['total']} pieces &nbsp; ({rc['regular']} regular, {rc['kings']} kings)
  </div>
  <div class="pc-row">
    <span class="black-label">BLACK</span>
    {bc['total']} pieces &nbsp; ({bc['regular']} regular, {bc['kings']} kings)
  </div>
  {promo_html}
</div>"""

    # ── Full document ─────────────────────────────────────────────────────
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Turn {tn} — Checkers AI</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<h1>Turn {tn} — Red (AI)</h1>
<div class="subtitle">Checkers AI · Simplified Pipeline · {ts} · {e(game_id)}</div>

<div class="panel">
  <div class="pt">Board before move</div>
  {board_before_html}
</div>

{move_section}

{expl_section}

{verifier_section}

<div class="panel">
  <div class="pt">Board after move</div>
  {board_after_html}
</div>

{pc_section}

</body>
</html>
"""


# ── Save orchestrator ─────────────────────────────────────────────────────────

def _save_turn(
    record: dict[str, Any],
    out_dir: pathlib.Path,
    game_id: str,
) -> None:
    """Write turn_NNN.html and turn_NNN.txt for one completed AI turn."""
    tn   = record["turn_number"]
    stem = f"turn_{tn:03d}"

    html_path = out_dir / f"{stem}.html"
    txt_path  = out_dir / f"{stem}.txt"

    html_path.write_text(_turn_to_html(record, game_id), encoding="utf-8")
    txt_path.write_text(_turn_to_txt(record, game_id),  encoding="utf-8")

    n_alt = len(record["unchosen_moves"])
    print(
        f"  {_C.DIM}↳  saved  {_C.RST}"
        f"{_C.GRN}{html_path}{_C.RST}"
        f"{_C.DIM}  &  {_C.RST}"
        f"{_C.GRN}{txt_path}{_C.RST}"
    )
    print(f"  {_C.DIM}Selected move exported{_C.RST}")
    print(f"  {_C.DIM}{n_alt} alternative(s) exported{_C.RST}")


# ── Presentation panels (terminal display) ────────────────────────────────────

def _show_board_panel(board: list[list[int]], title: str,
                      highlight: Optional[list] = None) -> None:
    lines = _render_board(board, highlight=highlight)
    print(_box(title, lines, title_color=_C.CYN))
    print()


def _show_move_panel(acc: dict[str, Any]) -> None:
    cm       = acc.get("chosen_move") or {}
    score    = acc.get("chosen_move_score")
    diag     = acc.get("proposer_diagnostics") or {}
    ss       = acc.get("score_state", "EQUAL")
    unchosen = acc.get("unchosen_moves") or []

    lines: list[str] = [
        f"{_C.BOLD}{_C.YLW}{_fmt_move_human(cm)}{_C.RST}",
        "",
    ]
    if score is not None:
        lines.append(f"Engine score      {_C.YLW}{score:+.2f}{_C.RST}")
    lines.append(f"Material balance  {_fmt_score_state_ansi(ss)}")
    n_legal = diag.get("n_legal", len(unchosen) + (1 if cm else 0))
    if n_legal:
        lines.append(f"Alternatives      {n_legal - 1} other legal move(s) considered")
    if unchosen:
        lines.append("")
        lines.append(f"{_C.DIM}Alternatives (not chosen):{_C.RST}")
        for i, m in enumerate(unchosen):
            alt_score = (m.get("facts") or {}).get("minimax_score")
            score_str = f"  [{alt_score:+.2f}]" if alt_score is not None else ""
            lines.append(f"  {_C.DIM}[{i+1}] {_fmt_move_human(m)}{score_str}{_C.RST}")

    print(_box("SELECTED MOVE", lines, title_color=_C.YLW))
    print()


def _show_explanation_panel(acc: dict[str, Any]) -> None:
    diag         = acc.get("explainer_diagnostics") or {}
    chosen_r     = (diag.get("chosen_reasoning") or acc.get("last_move_reasoning") or "").strip()
    comparative  = (diag.get("comparative_paragraph_text") or "").strip()

    lines: list[str] = []
    if chosen_r:
        for wl in textwrap.wrap(chosen_r, width=_W - 6):
            lines.append(wl)
    else:
        lines.append(f"{_C.DIM}(no explanation generated){_C.RST}")

    if comparative:
        lines.append("")
        lines.append(f"{_C.DIM}Why not the alternatives:{_C.RST}")
        for wl in textwrap.wrap(_strip_alt_indices(comparative), width=_W - 6):
            lines.append(f"  {_C.DIM}{wl}{_C.RST}")

    print(_box("AI EXPLANATION", lines, title_color=_C.MAG))
    print()


def _show_verifier_panel(acc: dict[str, Any]) -> None:
    diag      = acc.get("explainer_diagnostics") or {}
    detected  = diag.get("reasoning_contradiction_detected", False)
    repaired  = diag.get("reasoning_contradiction_repaired", False)
    n_retries = diag.get("reasoning_refinement_retry_count", 0)
    initial   = diag.get("reasoning_initial_contradictions") or []
    final     = diag.get("reasoning_final_contradictions") or []
    pre_text  = (diag.get("raw_llm_reasoning_pre_refinement") or "").strip()
    post_text = (diag.get("chosen_reasoning") or acc.get("last_move_reasoning") or "").strip()

    lines: list[str] = []
    if not detected:
        lines.append(f"{_C.GRN}✓  No contradictions detected — explanation accepted as-is.{_C.RST}")
    else:
        lines.append(f"{_C.RED}⚠  {len(initial)} contradiction(s) detected:{_C.RST}")
        for c in initial:
            for i, wl in enumerate(textwrap.wrap(_humanize_contradiction(c), width=_W - 10)):
                lines.append(f"{_C.RED}{'   •  ' if i == 0 else '      '}{wl}{_C.RST}")
        lines.append("")
        if repaired:
            lines.append(f"{_C.GRN}✓  Repaired successfully ({n_retries} refinement attempt(s)).{_C.RST}")
            if pre_text and post_text and pre_text != post_text:
                lines.append("")
                lines.append(f"{_C.DIM}BEFORE repair:{_C.RST}")
                for wl in textwrap.wrap(pre_text, width=_W - 10):
                    lines.append(f"   {_C.DIM}{wl}{_C.RST}")
                lines.append("")
                lines.append(f"{_C.GRN}AFTER repair:{_C.RST}")
                for wl in textwrap.wrap(post_text, width=_W - 10):
                    lines.append(f"   {_C.GRN}{wl}{_C.RST}")
        else:
            lines.append(f"{_C.RED}✗  Could not fully resolve after {n_retries} attempt(s).{_C.RST}")
            if final:
                lines.append(f"{_C.RED}   Remaining ({len(final)}):{_C.RST}")
                for c in final[:3]:
                    lines.append(f"   {_C.RED}• {_humanize_contradiction(c)}{_C.RST}")

    print(_box("VERIFIER FINDINGS", lines, title_color=_C.YLW))
    print()


def _show_piece_counts(board: list[list[int]]) -> None:
    rc = count_pieces(board, RED)
    bc = count_pieces(board, BLACK)
    lines = [
        f"{_C.BOLD}{_C.RED}RED  {_C.RST}  {rc['total']:2d} pieces  "
        f"({rc['regular']} regular,  {rc['kings']} kings)",
        f"{_C.BOLD}{_C.CYN}BLACK{_C.RST}  {bc['total']:2d} pieces  "
        f"({bc['regular']} regular,  {bc['kings']} kings)",
    ]
    print(_box("PIECE COUNT", lines, title_color=_C.DIM))
    print()


# ── Repair analysis helpers ───────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split a reasoning paragraph into individual sentences."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _extract_bad_phrases(contradictions: list[str]) -> list[str]:
    """Extract the concrete bad phrases from a list of contradiction strings."""
    phrases: list[str] = []
    for c in contradictions:
        quoted = re.findall(r"'([^']+)'", c)
        phrases.extend(p.lower() for p in quoted)
        c_low = c.lower()
        if "claims mobility reduction" in c_low:
            phrases.extend(["reduces mobility", "reducing mobility", "limits mobility",
                            "restricts mobility", "cuts opponent moves"])
        if "claims avoids recapture" in c_low or "claims avoid recapture" in c_low:
            phrases.extend(["avoids recapture", "no recapture", "safe from recapture"])
        if "claims no isolation" in c_low:
            phrases.extend(["does not isolate", "no isolation", "maintains connectivity",
                            "stays connected"])
        if "claims creates_immediate_threat" in c_low:
            phrases.extend(["creates a threat", "creates immediate threat",
                            "applies pressure next turn"])
        if "claims capture but" in c_low:
            phrases.extend(["captures a piece", "captures the piece", "gaining a piece"])
        if "claims material gain" in c_low:
            phrases.extend(["gains material", "material gain"])
        if "claims promotion" in c_low:
            phrases.extend(["promotes to king", "becomes a king", "crowns a piece"])
        if "claims center_control" in c_low:
            phrases.extend(["controls the center", "central control",
                            "central board presence"])
        if "forced opponent reply" in c_low or "forced_opponent_jump_reply" in c_low:
            phrases.extend(["forces the opponent", "forcing the opponent",
                            "opponent must respond", "no choice but to respond"])
    return list(dict.fromkeys(phrases))  # deduplicate, preserve order


def _diff_sentences(before: str, after: str) -> tuple[list[str], list[str]]:
    """
    Sentence-level diff between two reasoning paragraphs.
    Returns (removed_sentences, added_sentences).
    """
    b_sents = _split_sentences(before)
    a_sents = _split_sentences(after)
    matcher = difflib.SequenceMatcher(None, b_sents, a_sents, autojunk=False)
    removed: list[str] = []
    added:   list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete'):
            removed.extend(b_sents[i1:i2])
        if tag in ('replace', 'insert'):
            added.extend(a_sents[j1:j2])
    return removed, added


# ── Repair display panels ─────────────────────────────────────────────────────

def _final_output_block(lines: list[str], text: str, color: str = _C.WHT) -> None:
    """Append the FINAL ACCEPTED OUTPUT section to an in-progress lines list."""
    lines.append("═" * (_W - 6))
    lines.append(f"{_C.BOLD}{_C.CYN}FINAL ACCEPTED OUTPUT{_C.RST}")
    lines.append(_C.DIM + "─" * (_W - 6) + _C.RST)
    if text:
        for wl in textwrap.wrap(text, width=_W - 6):
            lines.append(f"  {color}{wl}{_C.RST}")
    else:
        lines.append(f"  {_C.DIM}(not generated this turn){_C.RST}")


def _show_move_repair_panel(diag: dict) -> None:
    """
    Repair audit panel for Move Selection Reasoning.
    Always shows FINAL ACCEPTED OUTPUT at the bottom regardless of contradictions.
    """
    detected  = diag.get("reasoning_contradiction_detected", False)
    repaired  = diag.get("reasoning_contradiction_repaired", False)
    initial   = diag.get("reasoning_initial_contradictions") or []
    n_retries = diag.get("reasoning_refinement_retry_count", 0)
    pre_text  = (diag.get("raw_llm_reasoning_pre_refinement") or "").strip()
    post_text = (diag.get("chosen_reasoning") or "").strip()

    lines: list[str] = []

    if not detected:
        lines.append(f"{_C.GRN}[NO CONTRADICTIONS DETECTED]{_C.RST}")
        lines.append("")
    else:
        bad_phrases = _extract_bad_phrases(initial)

        # ── ORIGINAL TEXT ─────────────────────────────────────────────────────
        lines.append(f"{_C.BOLD}{_C.WHT}ORIGINAL TEXT{_C.RST}")
        lines.append(_C.DIM + "─" * (_W - 6) + _C.RST)
        if pre_text:
            for s in _split_sentences(pre_text):
                is_bad = any(p in s.lower() for p in bad_phrases)
                if is_bad:
                    lines.append(f"  {_C.RED}[CONTRADICTION]{_C.RST}")
                    for wl in textwrap.wrap(f'"{s}"', width=_W - 8):
                        lines.append(f"  {_C.RED}{wl}{_C.RST}")
                else:
                    for wl in textwrap.wrap(s, width=_W - 6):
                        lines.append(f"  {_C.DIM}{wl}{_C.RST}")
        else:
            lines.append(f"  {_C.DIM}(pre-repair text not available){_C.RST}")
        lines.append("")

        # ── VERIFIER FINDINGS ─────────────────────────────────────────────────
        lines.append(f"{_C.BOLD}{_C.WHT}VERIFIER FINDINGS{_C.RST}")
        lines.append(_C.DIM + "─" * (_W - 6) + _C.RST)
        for c in initial:
            human    = _humanize_contradiction(c)
            raw_fact = _clean_contradiction(c)
            lines.append(f"  {_C.RED}• {human}{_C.RST}")
            if raw_fact != human and len(raw_fact) <= 110:
                lines.append(f"    {_C.DIM}[supporting fact] {raw_fact}{_C.RST}")
        lines.append("")

        # ── REPAIRED TEXT ─────────────────────────────────────────────────────
        lines.append(f"{_C.BOLD}{_C.WHT}REPAIRED TEXT{_C.RST}")
        lines.append(_C.DIM + "─" * (_W - 6) + _C.RST)
        if repaired and post_text:
            if pre_text:
                before_set = {s.lower().strip() for s in _split_sentences(pre_text)}
                for s in _split_sentences(post_text):
                    is_new = s.lower().strip() not in before_set
                    if is_new:
                        lines.append(f"  {_C.GRN}[REPAIRED]{_C.RST}")
                        for wl in textwrap.wrap(f'"{s}"', width=_W - 8):
                            lines.append(f"  {_C.GRN}{wl}{_C.RST}")
                    else:
                        for wl in textwrap.wrap(s, width=_W - 6):
                            lines.append(f"  {_C.DIM}{wl}{_C.RST}")
            else:
                for wl in textwrap.wrap(post_text, width=_W - 6):
                    lines.append(f"  {_C.GRN}{wl}{_C.RST}")
            lines.append("")
            lines.append(f"  {_C.GRN}✓  Repaired in {n_retries} attempt(s){_C.RST}")
        elif not repaired:
            final_contras = diag.get("reasoning_final_contradictions") or []
            remain_str = (
                f"  ({len(final_contras)} contradiction(s) remain)" if final_contras else ""
            )
            if post_text:
                for wl in textwrap.wrap(post_text, width=_W - 6):
                    lines.append(f"  {_C.DIM}{wl}{_C.RST}")
            lines.append("")
            lines.append(
                f"  {_C.RED}✗  Could not resolve after {n_retries} attempt(s)"
                f"{remain_str}{_C.RST}"
            )
        else:
            lines.append(f"  {_C.DIM}(no repaired text available){_C.RST}")

        # ── DIFF VIEW ─────────────────────────────────────────────────────────
        if pre_text and post_text and pre_text != post_text:
            lines.append("")
            lines.append(f"{_C.BOLD}{_C.WHT}DIFF VIEW{_C.RST}")
            lines.append(_C.DIM + "─" * (_W - 6) + _C.RST)
            removed, added = _diff_sentences(pre_text, post_text)
            if removed:
                lines.append(f"  {_C.RED}REMOVED:{_C.RST}")
                for s in removed:
                    for i, wl in enumerate(textwrap.wrap(s, width=_W - 12)):
                        lines.append(f"    {_C.RED}{'·' if i == 0 else ' '} {wl}{_C.RST}")
            if added:
                lines.append(f"  {_C.GRN}ADDED:{_C.RST}")
                for s in added:
                    for i, wl in enumerate(textwrap.wrap(s, width=_W - 12)):
                        lines.append(f"    {_C.GRN}{'·' if i == 0 else ' '} {wl}{_C.RST}")
        lines.append("")

    # ── FINAL ACCEPTED OUTPUT (always shown) ──────────────────────────────────
    _final_output_block(lines, post_text)

    title_color = _C.GRN if (not detected or repaired) else _C.RED
    print(_box("A. MOVE SELECTION REASONING", lines, title_color=title_color))
    print()


def _show_comparative_repair_panel(diag: dict) -> None:
    """
    Repair audit panel for Comparative Reasoning.
    Always shows FINAL ACCEPTED OUTPUT at the bottom regardless of contradictions.
    Pre-repair text is not retained in diagnostics; only counts are available.
    """
    was_skipped   = diag.get("comparative_was_skipped", True)
    skip_reason   = diag.get("comparative_skip_reason")
    final_text    = (diag.get("comparative_paragraph_text") or "").strip()
    initial_n     = diag.get("comparative_initial_contradictions", 0)
    final_n       = diag.get("comparative_final_contradictions", 0)
    refine_n      = diag.get("comparative_refinement_attempts", 0)
    short_circuit = diag.get("comparative_generation_short_circuited", False)
    provider      = diag.get("comparative_provider", "")

    lines: list[str] = []

    # ── Skipped ───────────────────────────────────────────────────────────────
    if was_skipped:
        reason_map = {
            "no_informative_groups":    "No informative comparison groups found",
            "no_seeds":                 "No comparison seeds could be built",
            "api_failure":              "API calls failed during generation",
            "all_samples_rejected":     "All generated samples were rejected",
            "single_legal_move":        "Only one legal move — nothing to compare",
            "binary_comparative_failed": "Binary comparison failed",
        }
        reason_str = reason_map.get(skip_reason or "", skip_reason or "unknown reason")
        lines.append(f"{_C.DIM}⊘  Comparative reasoning skipped: {reason_str}{_C.RST}")
        lines.append("")
        _final_output_block(lines, final_text)
        print(_box("B. COMPARATIVE REASONING", lines, title_color=_C.DIM))
        print()
        return

    # ── Deterministic binary (no LLM contradiction tracking) ─────────────────
    if provider == "deterministic_binary":
        lines.append(
            f"{_C.GRN}[DETERMINISTIC BINARY — no contradiction tracking]{_C.RST}"
        )
        lines.append("")
        _final_output_block(lines, final_text)
        print(_box("B. COMPARATIVE REASONING", lines, title_color=_C.GRN))
        print()
        return

    # ── Clean: no contradictions ──────────────────────────────────────────────
    if short_circuit or initial_n == 0:
        lines.append(f"{_C.GRN}[NO CONTRADICTIONS DETECTED]{_C.RST}")
        lines.append("")
        _final_output_block(lines, final_text)
        print(_box("B. COMPARATIVE REASONING", lines, title_color=_C.GRN))
        print()
        return

    # ── Contradictions detected ───────────────────────────────────────────────
    lines.append(
        f"{_C.RED}⚠  {initial_n} contradiction(s) found in best generated sample{_C.RST}"
    )
    lines.append(f"   Refinement attempts: {refine_n}")
    lines.append("")
    lines.append(
        f"{_C.DIM}Note: pre-repair comparative text is not retained in diagnostics.{_C.RST}"
    )
    lines.append(
        f"{_C.DIM}      Only the final verified paragraph and counts are available.{_C.RST}"
    )
    lines.append("")

    if final_n == 0 and final_text:
        lines.append(f"{_C.GRN}✓  Resolved after {refine_n} attempt(s){_C.RST}")
        lines.append("")
        lines.append(f"{_C.GRN}REPAIRED TEXT:{_C.RST}")
        lines.append(_C.DIM + "─" * (_W - 6) + _C.RST)
        for wl in textwrap.wrap(final_text, width=_W - 6):
            lines.append(f"  {_C.GRN}{wl}{_C.RST}")
        title_color = _C.GRN
    else:
        lines.append(
            f"{_C.RED}✗  Could not resolve after {refine_n} attempt(s)  "
            f"({final_n} remaining){_C.RST}"
        )
        lines.append(f"{_C.RED}   Comparative paragraph was suppressed.{_C.RST}")
        title_color = _C.RED

    lines.append("")
    _final_output_block(lines, final_text)
    print(_box("B. COMPARATIVE REASONING", lines, title_color=title_color))
    print()


# ── Game-over summary ─────────────────────────────────────────────────────────

def _show_game_over(state: dict[str, Any]) -> None:
    print()
    print(_C.BOLD + _hrule() + _C.RST)
    print(_C.BOLD + "  GAME OVER".center(_W) + _C.RST)
    print(_C.BOLD + _hrule() + _C.RST)
    print()

    if state.get("draw"):
        result = f"{_C.YLW}Draw{_C.RST}"
    elif state.get("winner") == RED:
        result = f"{_C.BOLD}{_C.RED}Red wins{_C.RST}"
    elif state.get("winner") == BLACK:
        result = f"{_C.BOLD}{_C.CYN}Black wins{_C.RST}"
    else:
        result = "No winner recorded"

    mh    = state.get("move_history") or []
    cap_r = sum(len((r.get("move") or {}).get("captured") or []) for r in mh if r.get("player") == RED)
    cap_b = sum(len((r.get("move") or {}).get("captured") or []) for r in mh if r.get("player") == BLACK)
    prom  = sum(1 for r in mh if r.get("promotion"))
    lines = [
        f"Result         {result}",
        f"Total turns    {state.get('turn_number', '?')}",
        f"Red captures   {cap_r}",
        f"Black captures {cap_b}",
        f"Promotions     {prom}",
    ]
    print(_box("FINAL SUMMARY", lines))
    print()
    _show_board_panel(state["board"], "FINAL BOARD POSITION")
    gid = state.get("game_log_id") or "(none)"
    print(f"  {_C.DIM}Pipeline logs: logs/{gid}.jsonl{_C.RST}")
    print()


# ── Graph streaming — one RED ply ─────────────────────────────────────────────

def _stream_one_ply_presentation(
    acc: dict[str, Any],
    recursion_limit: int = 50,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """
    Run one RED ply: scorer → proposer → explainer → updater.

    Returns (acc, saw_updater, capture) where capture holds fields that
    state_manager clears before the stream returns — necessary for export.

    stdout is redirected to /dev/null for the duration of each graph node's
    execution so that pipeline-internal print statements (e.g. [SELECTIVE_D8],
    [RANKER], [EXPLAINER_SEED_REASONING]) never appear between our styled
    display panels.  stdout is restored to its real target before every
    _show_* call and suppressed again immediately after.
    """
    saw_updater = False
    capture: dict[str, Any] = {}
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": recursion_limit,
    }

    _real_stdout = sys.stdout
    _devnull = open(os.devnull, "w")
    sys.stdout = _devnull  # suppress before first node runs

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
                acc.update(delta)

                if node_name == "proposer_agent":
                    # Snapshot fields that state_manager will clear.
                    capture["chosen_move"]          = dict(acc.get("chosen_move") or {})
                    capture["chosen_move_score"]    = acc.get("chosen_move_score")
                    capture["unchosen_moves"]       = list(acc.get("unchosen_moves") or [])
                    capture["score_state"]          = acc.get("score_state", "EQUAL")
                    capture["proposer_diagnostics"] = dict(acc.get("proposer_diagnostics") or {})
                    sys.stdout = _real_stdout
                    _show_move_panel(acc)
                    sys.stdout = _devnull

                elif node_name == "explainer_agent":
                    # Snapshot explanation data before state_manager clears it.
                    capture["explainer_diagnostics"] = dict(acc.get("explainer_diagnostics") or {})
                    capture["last_move_reasoning"]   = acc.get("last_move_reasoning", "")
                    sys.stdout = _real_stdout
                    _show_explanation_panel(acc)
                    _show_verifier_panel(acc)
                    _diag = acc.get("explainer_diagnostics") or {}
                    _show_move_repair_panel(_diag)
                    _show_comparative_repair_panel(_diag)
                    sys.stdout = _devnull

                elif node_name == "updater_agent":
                    saw_updater = True
                    # chosen_move is now None (cleared by state_manager).
                    # Retrieve the path from move_history instead.
                    mh   = acc.get("move_history") or []
                    path = (mh[-1].get("move") or {}).get("path") if mh else None
                    sys.stdout = _real_stdout
                    _show_board_panel(acc["board"], "BOARD AFTER MOVE", highlight=path)
                    _show_piece_counts(acc["board"])
                    sys.stdout = _devnull

    except Exception as e:
        sys.stdout = _real_stdout
        print(f"\n[run_presentation_trace] graph error: {e}", file=sys.stderr)
    finally:
        sys.stdout = _real_stdout
        _devnull.close()

    return acc, saw_updater, capture


# ── RED ply — AI ──────────────────────────────────────────────────────────────

def _run_red_ply(
    acc: dict[str, Any],
    out_dir: Optional[pathlib.Path] = None,
    game_id: str = "",
) -> dict[str, Any]:
    board_before = [row[:] for row in acc["board"]]
    turn_no      = (acc.get("turn_number") or 0) + 1

    print()
    print(_C.BOLD + _hrule() + _C.RST)
    print(_C.BOLD + _C.RED + f"  TURN {turn_no}   |   RED — AI".center(_W) + _C.RST)
    print(_C.BOLD + _hrule() + _C.RST)
    print()

    _show_board_panel(acc["board"], "BOARD POSITION")

    acc["last_completed_node"] = None
    acc, ok, capture = _stream_one_ply_presentation(acc)

    if not ok:
        print(
            "[run_presentation_trace] warning: graph did not complete updater_agent.",
            file=sys.stderr,
        )

    if ok and out_dir is not None:
        record = _collect_turn_record(board_before, capture, acc, turn_no)
        _save_turn(record, out_dir, game_id)

    return acc


# ── BLACK ply — human ─────────────────────────────────────────────────────────

def _run_black_ply(acc: dict[str, Any]) -> dict[str, Any]:
    turn_no = (acc.get("turn_number") or 0) + 1
    board   = acc["board"]
    legal   = get_all_legal_moves(board, BLACK)

    if not legal:
        print("[run_presentation_trace] BLACK has no legal moves.", file=sys.stderr)
        return acc

    print()
    print(_C.BOLD + _hrule() + _C.RST)
    print(_C.BOLD + _C.CYN + f"  TURN {turn_no}   |   BLACK — YOUR MOVE".center(_W) + _C.RST)
    print(_C.BOLD + _hrule() + _C.RST)
    print()

    _show_board_panel(board, "BOARD POSITION")

    move_lines: list[str] = []
    for i, m in enumerate(legal):
        move_lines.append(f"{_C.BOLD}{_C.CYN}[{i}]{_C.RST}  {_fmt_move_human(m)}")
    print(_box("YOUR AVAILABLE MOVES", move_lines, title_color=_C.CYN))
    print()

    while True:
        try:
            raw = input(f"  Enter move index [0–{len(legal) - 1}]: ").strip()
            k   = int(raw)
            if 0 <= k < len(legal):
                break
            print(f"  Please enter a number between 0 and {len(legal) - 1}.")
        except (ValueError, EOFError):
            print("  Invalid input — enter a number.")

    move = legal[k]
    print()

    acc["chosen_move"]         = move
    acc["last_move_reasoning"] = "BLACK human move"

    _valid  = set(CheckersState.model_fields.keys())
    _state  = CheckersState(**{k: v for k, v in acc.items() if k in _valid})
    _result = _update_agent_fn(_state)
    acc.update(_result)

    ok = _result.get("last_completed_node") == "updater_agent"
    if not ok:
        print(
            "[run_presentation_trace] warning: BLACK updater_agent did not complete.",
            file=sys.stderr,
        )

    def _norm(p: Any) -> list:
        return [list(sq) for sq in (p or [])]

    mh = acc.get("move_history") or []
    if mh:
        applied = mh[-1].get("move") or {}
        if (
            applied.get("type") != move.get("type")
            or _norm(applied.get("path")) != _norm(move.get("path"))
        ):
            print("[SMOKE TEST FAIL] Applied move does not match chosen move!", file=sys.stderr)

    conf_lines = [f"{_C.BOLD}{_C.CYN}{_fmt_move_human(move)}{_C.RST}"]
    if mh and mh[-1].get("promotion"):
        conf_lines.append(f"{_C.YLW}  ★  Promoted to king!{_C.RST}")
    print(_box("MOVE APPLIED", conf_lines, title_color=_C.CYN))
    print()

    _show_board_panel(acc["board"], "BOARD AFTER MOVE")
    _show_piece_counts(acc["board"])

    return acc


# ── Main game loop ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Presentation trace: RED = AI pipeline, BLACK = human."
    )
    parser.add_argument("--max-turns", type=int, default=200,
                        help="Safety cap on plies (half-moves).")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable automatic turn export to disk.")
    args = parser.parse_args()

    game_id = datetime.datetime.now().strftime("game_%Y%m%d_%H%M%S")
    out_dir: Optional[pathlib.Path] = None

    if not args.no_save:
        out_dir = pathlib.Path("presentation_traces") / game_id
        out_dir.mkdir(parents=True, exist_ok=True)

    # Opening screen
    print()
    print(_C.BOLD + _hrule() + _C.RST)
    print(_C.BOLD + "  CHECKERS AI — SIMPLIFIED PIPELINE DEMO".center(_W) + _C.RST)
    print(_C.BOLD + _hrule() + _C.RST)
    intro_lines = [
        f"   {_C.BOLD}{_C.RED}RED{_C.RST}   =  AI  (scorer → proposer → explainer → verifier)",
        f"   {_C.BOLD}{_C.CYN}BLACK{_C.RST} =  You  (enter a move index when prompted)",
        "",
        f"   Board coordinates: (row, col),  rows and cols numbered 0–7.",
        f"   {_C.BOLD}{_C.RED}r{_C.RST} = red regular   {_C.BOLD}{_C.RED}R{_C.RST} = red king",
        f"   {_C.BOLD}{_C.CYN}b{_C.RST} = black regular  {_C.BOLD}{_C.CYN}B{_C.RST} = black king",
    ]
    if out_dir:
        intro_lines += [
            "",
            f"   {_C.DIM}Traces saved to:{_C.RST}  {_C.GRN}{out_dir}/{_C.RST}",
            f"   {_C.DIM}Each AI turn → turn_NNN.html  &  turn_NNN.txt{_C.RST}",
        ]
    print(_box("HOW TO PLAY", intro_lines))
    print()
    input("  Press Enter to start the game…")
    print()

    acc = CheckersState(
        board=create_initial_board(),
        current_player=RED,
        turn_number=0,
    ).model_dump()

    while True:
        if acc.get("game_over"):
            _show_game_over(acc)
            if out_dir:
                print(f"  {_C.DIM}All traces in:{_C.RST}  {_C.GRN}{out_dir}/{_C.RST}")
            return

        if (acc.get("turn_number") or 0) >= args.max_turns:
            print()
            print(f"{_C.RED}Game incomplete: maximum turns reached.{_C.RST}")
            _show_board_panel(acc["board"], "FINAL BOARD")
            if out_dir:
                print(f"  {_C.DIM}Traces saved in:{_C.RST}  {_C.GRN}{out_dir}/{_C.RST}")
            return

        if acc["current_player"] == RED:
            acc = _run_red_ply(acc, out_dir=out_dir, game_id=game_id)
        else:
            acc = _run_black_ply(acc)


if __name__ == "__main__":
    main()
