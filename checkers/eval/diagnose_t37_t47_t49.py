#!/usr/bin/env python3
"""
Diagnostic: evaluate_board_breakdown() for T37, T47, T49.

Reconstructs exact board states from the trace, applies each candidate move,
and prints a side-by-side term breakdown table.

Boards are taken verbatim from trace output. Moves are the exact legal moves
reported by the engine at each turn. Evaluation is called at depth-0 (leaf)
from RED's perspective — same convention as minimax leaf nodes.

Usage:
    python3 diagnose_t37_t47_t49.py
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkers.engine.board import EMPTY, RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import apply_move
from checkers.engine.evaluation import evaluate_board_breakdown

# ── Board helpers ─────────────────────────────────────────────────────────────

def empty_board():
    return [[EMPTY] * 8 for _ in range(8)]


def print_board(board):
    syms = {EMPTY: ".", RED: "r", BLACK: "b", RED_KING: "R", BLACK_KING: "B"}
    print("  0 1 2 3 4 5 6 7")
    for r, row in enumerate(board):
        print(r, " ".join(syms[c] for c in row))
    print()


def breakdown_after_move(board, move_path, move_captured, current_player_after, root_player):
    """Apply move, return (board_after, breakdown_dict)."""
    m = {"type": "simple", "path": move_path, "captured": move_captured}
    board_after = apply_move(board, m)
    bd = evaluate_board_breakdown(board_after, current_player_after, root_player)
    return board_after, bd


def print_comparison(title: str, breakdowns: list[tuple[str, dict]]):
    """Print a term-by-term table for a set of candidate moves."""
    terms = [
        "material", "mobility", "center", "promotion_threat",
        "promotion_proximity", "back_row_guard", "isolation",
        "connectivity_support", "frozen_restriction",
        "king_centralization", "king_mobility", "king_chase_pressure",
        "vulnerability", "structure", "endgame",
        "simplification_when_ahead", "confinement_bonus", "column_centrality",
        "total",
    ]
    col_w = 20
    labels = [label for label, _ in breakdowns]
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    # Header
    header = f"{'TERM':<28}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for t in terms:
        vals = [bd.get(t, 0.0) for _, bd in breakdowns]
        row = f"{'  '+t:<28}" + "".join(f"{v:>{col_w}.1f}" for v in vals)
        # Highlight rows with large variance
        spread = max(vals) - min(vals) if vals else 0
        flag = "  ◄" if spread >= 20 else ""
        print(row + flag)
    print()


# ── T37 Board ─────────────────────────────────────────────────────────────────
# RED to move. Board from trace (before RED's move at T37).
# After BLACK promoted at (7,6) = BLACK_KING.
#
#   0 1 2 3 4 5 6 7
# 0 . b . b . b . .
# 1 . . . . . . r .
# 2 . r . . . . . b
# 3 . . . . . . . .
# 4 . r . . . . . b
# 5 r . r . r . . .
# 6 . . . . . r . .
# 7 . . . . r . B .

def build_t37():
    b = empty_board()
    # BLACK men
    b[0][1] = BLACK; b[0][3] = BLACK; b[0][5] = BLACK
    b[2][7] = BLACK
    b[4][7] = BLACK
    # BLACK king
    b[7][6] = BLACK_KING
    # RED men
    b[1][6] = RED
    b[2][1] = RED
    b[4][1] = RED
    b[5][0] = RED; b[5][2] = RED; b[5][4] = RED
    b[6][5] = RED
    b[7][4] = RED
    return b


T37_MOVES = [
    # (label, path, captured)
    ("(1,6)→(0,7) PROMO",  [(1,6),(0,7)], []),   # index 0 — PROMOTION
    ("(6,5)→(5,4) CHOSEN", [(6,5),(5,4)], []),   # index 6 — chosen
    ("(2,1)→(1,0) push",   [(2,1),(1,0)], []),
    ("(5,2)→(4,3) center", [(5,2),(4,3)], []),
    ("(7,4)→(6,3) back",   [(7,4),(6,3)], []),
]

# ── T47 Board ─────────────────────────────────────────────────────────────────
# RED to move. Board from trace (before RED's move at T47).
#
#   0 1 2 3 4 5 6 7
# 0 . b . b . b . R   ← RED_KING at (0,7)
# 1 . . . . . . . .
# 2 . r . . . . . .
# 3 . . r . . . b .
# 4 . . . r . . . .
# 5 r . r . . . . .
# 6 . . . . . B . .   ← BLACK_KING at (6,5)
# 7 . . . . . . . .

def build_t47():
    b = empty_board()
    # BLACK men
    b[0][1] = BLACK; b[0][3] = BLACK; b[0][5] = BLACK
    b[3][6] = BLACK
    # BLACK king
    b[6][5] = BLACK_KING
    # RED king
    b[0][7] = RED_KING
    # RED men
    b[2][1] = RED
    b[3][2] = RED
    b[4][3] = RED
    b[5][0] = RED; b[5][2] = RED
    return b


T47_MOVES = [
    # (label, path, captured)
    ("(3,2)→(2,3) CHOSEN+105", [(3,2),(2,3)], []),   # chosen — scores +105
    ("(2,1)→(1,0) push -87",   [(2,1),(1,0)], []),   # 1 step from promo
    ("(4,3)→(3,4) quiet -65",  [(4,3),(3,4)], []),
    ("(5,2)→(4,1) quiet -3",   [(5,2),(4,1)], []),
    ("(5,0)→(4,1) quiet -84",  [(5,0),(4,1)], []),
    ("(2,1)→(1,2) boom-655",   [(2,1),(1,2)], []),   # extreme outlier
]

# ── T49 Board ─────────────────────────────────────────────────────────────────
# RED to move. Board from trace (before RED's move at T49).
#
#   0 1 2 3 4 5 6 7
# 0 . b . b . b . R   ← RED_KING at (0,7)
# 1 . . . . . . . .
# 2 . r . r . . . .
# 3 . . . . . . b .
# 4 . . . r . . . .
# 5 r . r . B . . .   ← BLACK_KING at (5,4)
# 6 . . . . . . . .
# 7 . . . . . . . .

def build_t49():
    b = empty_board()
    # BLACK men
    b[0][1] = BLACK; b[0][3] = BLACK; b[0][5] = BLACK
    b[3][6] = BLACK
    # BLACK king
    b[5][4] = BLACK_KING
    # RED king
    b[0][7] = RED_KING
    # RED men
    b[2][1] = RED; b[2][3] = RED
    b[4][3] = RED
    b[5][0] = RED; b[5][2] = RED
    return b


T49_MOVES = [
    # (label, path, captured)
    ("(4,3)→(3,2) CHOSEN+108", [(4,3),(3,2)], []),   # chosen — scores +108
    ("(4,3)→(3,4) quiet -219", [(4,3),(3,4)], []),
    ("(2,1)→(1,0) push -418",  [(2,1),(1,0)], []),   # near promo, extreme
    ("(2,3)→(1,4) push -394",  [(2,3),(1,4)], []),   # near promo, extreme
    ("(5,2)→(4,1) quiet -418", [(5,2),(4,1)], []),
    ("(2,1)→(1,2) boom-474",   [(2,1),(1,2)], []),   # worst
]


# ── Main ─────────────────────────────────────────────────────────────────────

def run_turn(title, board_fn, moves, turn_player=RED, root_player=RED):
    board = board_fn()
    print(f"\n{'#'*80}")
    print(f"  {title} — Board before RED's move")
    print(f"{'#'*80}")
    print_board(board)

    opponent = BLACK if root_player == RED else RED
    # After RED moves it's BLACK's turn → current_player_after = BLACK
    current_player_after = opponent

    breakdowns = []
    for label, path, captured in moves:
        _, bd = breakdown_after_move(board, path, captured, current_player_after, root_player)
        breakdowns.append((label, bd))

    print_comparison(title, breakdowns)

    # Also print individual breakdown for moves with extreme or suspicious scores
    suspicious = [
        (label, bd) for label, bd in breakdowns
        if abs(bd.get("total", 0)) >= 100
        or ("push" in label or "PROMO" in label or "CHOSEN" in label)
    ]
    for label, bd in suspicious:
        print(f"  DETAIL [{label}]:")
        for k, v in bd.items():
            if abs(v) > 0.01:
                print(f"    {k:35s} {v:+.1f}")
        print()


if __name__ == "__main__":
    run_turn("T37: Missed Promotion vs Chosen Shuffle", build_t37, T37_MOVES)
    run_turn("T47: Backward Move +105 vs Near-Promotion -87", build_t47, T47_MOVES)
    run_turn("T49: Backward Move +108 vs Near-Promotion -418", build_t49, T49_MOVES)
