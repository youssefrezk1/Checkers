#!/usr/bin/env python3
"""
T37 Principal Variation Diagnostic.

Reconstructs the T37 board (RED to move), then for two candidate moves:
  A) promotion:        (1,6) → (0,7)
  B) defensive shuffle: (6,5) → (5,4)

Extracts the depth-6 principal variation by greedily following the best
child at each ply — equivalent to the PV the engine actually searches.

Also verifies that the promotion move survives _apply_safety_filter after
the promotion-exemption fix.

Usage:
    python3 diagnose_t37_pv.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkers.engine.board import EMPTY, RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.engine.evaluation import evaluate_board_breakdown, evaluate_board
from checkers.search.minimax_core import negamax, SearchStats, clear_transposition_table
from checkers.agents.ranker_agent import _apply_safety_filter

DEPTH = 6

syms = {EMPTY: ".", RED: "r", BLACK: "b", RED_KING: "R", BLACK_KING: "B"}


def show_board(b, indent=4):
    pad = " " * indent
    print(pad + "  0 1 2 3 4 5 6 7")
    for ri, row in enumerate(b):
        print(pad + f"{ri} " + " ".join(syms[c] for c in row))


def opp(player):
    return BLACK if player == RED else RED


# ── T37 board (exact from trace, before RED's move) ─────────────────────────
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
    b = [[EMPTY] * 8 for _ in range(8)]
    b[0][1] = BLACK; b[0][3] = BLACK; b[0][5] = BLACK
    b[2][7] = BLACK
    b[4][7] = BLACK
    b[7][6] = BLACK_KING
    b[1][6] = RED
    b[2][1] = RED
    b[4][1] = RED
    b[5][0] = RED; b[5][2] = RED; b[5][4] = RED
    b[6][5] = RED
    b[7][4] = RED
    return b


def best_child(board, player, root_player, depth):
    """Return (best_move, score) from current board at given depth."""
    legal = get_all_legal_moves(board, player)
    if not legal:
        return None, None
    best_move = None
    best_score = float("-inf") if player == root_player else float("inf")
    for m in legal:
        child = apply_move(board, m)
        clear_transposition_table()
        score = negamax(
            board=child,
            depth=max(0, depth - 1),
            current_player=opp(player),
            root_player=root_player,
            alpha=float("-inf"),
            beta=float("inf"),
            stats=SearchStats(),
            use_tt=False,
        )
        if player == root_player:
            if score > best_score:
                best_score = score
                best_move = m
        else:
            if score < best_score:
                best_score = score
                best_move = m
    return best_move, best_score


def extract_pv(start_board, first_move, root_player, total_depth):
    """
    Walk the principal variation greedily for total_depth plies.
    Returns list of (player_label, move_path, board_after, score_at_depth).
    """
    pv = []
    board = apply_move(start_board, first_move)
    player = opp(root_player)      # first reply is opponent's
    remaining = total_depth - 1    # one ply used by first_move

    pv.append((
        "RED" if root_player == RED else "BLACK",
        first_move["path"],
        board,
        None,   # score computed at root, filled later
    ))

    for ply in range(remaining):
        legal = get_all_legal_moves(board, player)
        if not legal:
            break
        # pick best child at depth=(remaining - ply) from root_player perspective
        m, _ = best_child(board, player, root_player, max(1, remaining - ply))
        if m is None:
            break
        board = apply_move(board, m)
        label = "RED" if player == RED else "BLACK"
        pv.append((label, m["path"], board, None))
        player = opp(player)

    return pv, board   # last board = leaf


def print_pv(pv, label):
    print(f"\n  PV for {label}:")
    for i, (pl, path, board, _) in enumerate(pv):
        print(f"    ply {i+1:2d} | {pl:5s} plays {path}")
    print()


def print_breakdown(bd, indent=6):
    pad = " " * indent
    important = [
        "material", "mobility", "center", "promotion_threat",
        "promotion_proximity", "back_row_guard", "isolation",
        "king_centralization", "king_mobility", "king_chase_pressure",
        "vulnerability", "confinement_bonus", "simplification_when_ahead",
        "structure", "endgame", "total",
    ]
    for t in important:
        v = bd.get(t, 0.0)
        if abs(v) > 0.0:
            print(f"{pad}{t:35s} {v:+.1f}")


def make_move_dict(path, mtype="simple", captured=None):
    return {"type": mtype, "path": path, "captured": captured or []}


# ── Safety filter check ──────────────────────────────────────────────────────

def check_safety_filter(board):
    """
    Build a minimal legal-move list with facts for the two key moves and
    run it through _apply_safety_filter. Reports whether promotion survived.
    """
    legal_moves_with_facts = [
        {
            "type": "simple", "path": [(1, 6), (0, 7)], "captured": [],
            "facts": {
                "minimax_score": -25.0,
                "results_in_king": True,
                "opponent_can_recapture": True,   # corner king is reachable
                "our_pieces_threatened_after": 1,
                "net_gain": 0,
                "counterplay_score": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
            },
        },
        {
            "type": "simple", "path": [(6, 5), (5, 4)], "captured": [],
            "facts": {
                "minimax_score": 20.0,
                "results_in_king": False,
                "opponent_can_recapture": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "counterplay_score": 1,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
            },
        },
        {
            "type": "simple", "path": [(2, 1), (1, 0)], "captured": [],
            "facts": {
                "minimax_score": -25.0,
                "results_in_king": False,
                "opponent_can_recapture": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "counterplay_score": 1,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
            },
        },
    ]
    filtered, idx_map = _apply_safety_filter(
        legal_moves_with_facts,
        strategic_priorities=["PROMOTE", "TRADE_WHEN_AHEAD"],
        score_state="EQUAL",
    )
    filtered_paths = [m["path"] for m in filtered]
    promo_survived = [(1, 6), (0, 7)] in filtered_paths
    print("\n" + "=" * 70)
    print("SAFETY FILTER CHECK")
    print("=" * 70)
    print(f"  Input moves:    {[m['path'] for m in legal_moves_with_facts]}")
    print(f"  Filtered moves: {filtered_paths}")
    print(f"  filtered_menu_size: {len(filtered)}")
    print(f"  Promotion (1,6)→(0,7) survived filter: {promo_survived}")
    if not promo_survived:
        print("  *** BUG: Promotion was STILL removed by safety filter! ***")
    return promo_survived


# ── Depth-6 score for each candidate ────────────────────────────────────────

def score_candidate(board, move_path, player):
    m = make_move_dict(move_path)
    child = apply_move(board, m)
    clear_transposition_table()
    score = negamax(
        board=child,
        depth=DEPTH - 1,
        current_player=opp(player),
        root_player=player,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
    )
    return score


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    board = build_t37()

    print("=" * 70)
    print("T37 BOARD (before RED's move)")
    print("=" * 70)
    show_board(board)

    # Count pieces
    red_men = sum(1 for r in board for c in r if c == RED)
    red_kings = sum(1 for r in board for c in r if c == RED_KING)
    blk_men = sum(1 for r in board for c in r if c == BLACK)
    blk_kings = sum(1 for r in board for c in r if c == BLACK_KING)
    print(f"  RED: {red_men} men, {red_kings} kings | BLACK: {blk_men} men, {blk_kings} kings")

    # ── Safety filter
    check_safety_filter(board)

    # ── Depth-6 scores
    print("\n" + "=" * 70)
    print(f"DEPTH-{DEPTH} MINIMAX SCORES (root=RED, use_tt=False for isolation)")
    print("=" * 70)

    candidates = [
        ("(1,6)→(0,7) PROMOTION",       [(1, 6), (0, 7)]),
        ("(6,5)→(5,4) chosen shuffle",   [(6, 5), (5, 4)]),
        ("(2,1)→(1,0) push row-1",       [(2, 1), (1, 0)]),
        ("(5,2)→(4,3) center",           [(5, 2), (4, 3)]),
        ("(7,4)→(6,3) backward",         [(7, 4), (6, 3)]),
    ]
    scores = {}
    for label, path in candidates:
        sc = score_candidate(board, path, RED)
        scores[label] = sc
        print(f"  {label:40s} {sc:+.1f}")

    # ── PV extraction ─────────────────────────────────────────────────────────
    for label, path in candidates[:2]:   # only promo and shuffle — the two key ones
        m = make_move_dict(path)
        print("\n" + "=" * 70)
        print(f"PRINCIPAL VARIATION: {label}  (depth={DEPTH})")
        print("=" * 70)

        pv, leaf_board = extract_pv(board, m, RED, DEPTH)
        print_pv(pv, label)

        # Print each board in PV
        for i, (pl, path_ply, b_after, _) in enumerate(pv):
            print(f"  Board after ply {i+1} ({pl} plays {path_ply}):")
            show_board(b_after, indent=4)

        # Leaf evaluation
        # At leaf it's BLACK's turn to move (even ply depth=6 from RED's move)
        # Determine whose turn it is at the leaf
        leaf_player = RED if len(pv) % 2 == 0 else BLACK
        bd = evaluate_board_breakdown(leaf_board, leaf_player, RED)

        print(f"  Leaf board (ply {len(pv)}, {('RED' if leaf_player==RED else 'BLACK')} to move at leaf):")
        show_board(leaf_board, indent=4)
        print(f"  evaluate_board_breakdown (root=RED perspective):")
        print_breakdown(bd)

        # Also check whether the leaf position has immediate threats
        blk_jumps = [mv for mv in get_all_legal_moves(leaf_board, BLACK) if mv["type"] == "jump"]
        red_jumps = [mv for mv in get_all_legal_moves(leaf_board, RED) if mv["type"] == "jump"]
        print(f"  Leaf tension: BLACK has {len(blk_jumps)} jumps, RED has {len(red_jumps)} jumps")
        if blk_jumps:
            for jm in blk_jumps:
                print(f"    BLACK jump: {jm['path']} captures {jm['captured']}")

    # ── Summary comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    promo_score = scores["(1,6)→(0,7) PROMOTION"]
    shuffle_score = scores["(6,5)→(5,4) chosen shuffle"]
    print(f"  Promotion  (1,6)→(0,7): {promo_score:+.1f}")
    print(f"  Shuffle    (6,5)→(5,4): {shuffle_score:+.1f}")
    print(f"  Gap (promo - shuffle):  {promo_score - shuffle_score:+.1f}")
    if promo_score > shuffle_score:
        print("  → Promotion is BETTER by minimax at depth 6.")
        print("  → If ranker/override chose the shuffle, that is a pipeline issue.")
    else:
        print(f"  → Shuffle outscores promotion by {shuffle_score - promo_score:.1f} pts.")
        print("  → If the gap is large and the leaf shows a corner-trap, classification: legitimate tactic.")
        print("  → If the leaf shows good material but bad king/corner terms, classification: evaluation bug.")


if __name__ == "__main__":
    main()
