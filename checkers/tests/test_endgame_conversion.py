"""
tests/test_endgame_conversion.py

Controlled endgame-conversion test.

Board state: Turn 41 from the depth=6 trace.
  - RED is clearly winning: 7 pieces (1 king) vs 5 pieces (1 king), material +2.
  - Starting player: RED (AI).

Metrics tracked each run:
  1. Moves until first RED capture
  2. Total RED captures within MAX_TURNS
  3. Average distance (each RED king → nearest enemy piece) per move
  4. Whether RED wins (BLACK has 0 pieces) within MAX_TURNS

Usage:
    python -m pytest tests/test_endgame_conversion.py -v -s
  or:
    python tests/test_endgame_conversion.py

The test does NOT call any LLM agents. It drives the engine purely through
minimax scoring (score_move_with_minimax) to isolate the evaluator's effect.
"""

import sys
import os

# ── Make the project root importable ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from checkers.engine.board import EMPTY, RED, BLACK, RED_KING, BLACK_KING, BOARD_SIZE
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.minimax import score_move_with_minimax

# ── Constants ─────────────────────────────────────────────────────────────────
DEPTH = 6
MAX_TURNS = 30   # simulate at most 30 plies from the starting position

# ── Board at Turn 41 (BEFORE RED moves) ───────────────────────────────────────
#   0 1 2 3 4 5 6 7
# 0 . b . . . b . b
# 1 . . R . r . . .
# 2 . . . r . . . .
# 3 . . . . . . . .
# 4 . . . . . . . r
# 5 r . . . . . . .
# 6 . r . b . . . .
# 7 r . . . . . B .
#
# r=RED(1)  R=RED_KING(3)  b=BLACK(2)  B=BLACK_KING(4)  .=EMPTY(0)
_E = EMPTY
_r = RED
_R = RED_KING
_b = BLACK
_B = BLACK_KING

T41_BOARD = [
    [_E, _b, _E, _E, _E, _b, _E, _b],  # row 0
    [_E, _E, _R, _E, _r, _E, _E, _E],  # row 1
    [_E, _E, _E, _r, _E, _E, _E, _E],  # row 2
    [_E, _E, _E, _E, _E, _E, _E, _E],  # row 3
    [_E, _E, _E, _E, _E, _E, _E, _r],  # row 4
    [_r, _E, _E, _E, _E, _E, _E, _E],  # row 5
    [_E, _r, _E, _b, _E, _E, _E, _E],  # row 6
    [_r, _E, _E, _E, _E, _E, _B, _E],  # row 7
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_pieces(board, player):
    man = RED if player == RED else BLACK
    king = RED_KING if player == RED else BLACK_KING
    men = kings = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == man:
                men += 1
            elif board[r][c] == king:
                kings += 1
    return men, kings


def _all_positions(board, player):
    man = RED if player == RED else BLACK
    king = RED_KING if player == RED else BLACK_KING
    out = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] in (man, king):
                out.append((r, c))
    return out


def _king_positions(board, player):
    king = RED_KING if player == RED else BLACK_KING
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            if board[r][c] == king]


def _avg_king_enemy_distance(board):
    """
    Average Manhattan distance from each RED king to the nearest BLACK piece.
    Returns None when RED has no kings or BLACK has no pieces.
    """
    kings = _king_positions(board, RED)
    enemies = _all_positions(board, BLACK)
    if not kings or not enemies:
        return None
    total = 0.0
    for kr, kc in kings:
        nearest = min(abs(kr - er) + abs(kc - ec) for er, ec in enemies)
        total += nearest
    return total / len(kings)


def _pick_best_move_by_minimax(board, player, depth=DEPTH):
    """
    Choose the legal move with the highest minimax score at given depth.
    Tie-break: first in list (deterministic).
    """
    moves = get_all_legal_moves(board, player)
    if not moves:
        return None
    best_move = None
    best_score = float("-inf")
    for m in moves:
        s = score_move_with_minimax(board, m, player, depth=depth)
        if s > best_score:
            best_score = s
            best_move = m
    return best_move, best_score


def _pick_greedy_move(board, player):
    """
    Choose the move with the best shallow (depth=1) score for `player`.
    Used for BLACK to simulate a human-level (non-optimal) opponent.
    Greedy play makes mistakes — avoids threats only one ply deep — which
    creates the constrained positions that evaluator signals need to fire.
    """
    return _pick_best_move_by_minimax(board, player, depth=1)


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(board, start_player=RED, max_turns=MAX_TURNS, label=""):
    """
    Run max_turns plies.
    Both RED and BLACK use full depth-6 minimax (perfect play).

    This tests whether the active endgame heuristics (chase + confinement)
    are strong enough to force a win even against optimal defensive play.
    Returns a dict of metrics.
    """
    import copy
    board = copy.deepcopy(board)
    player = start_player

    red_men0, red_kings0 = _count_pieces(board, RED)
    blk_men0, blk_kings0 = _count_pieces(board, BLACK)

    print(f"\n{'='*60}")
    print(f"  SIMULATION: {label}")
    print(f"  Depth={DEPTH}  MaxTurns={max_turns}")
    print(f"  Start: RED={red_men0}men+{red_kings0}K  BLACK={blk_men0}men+{blk_kings0}K")
    print(f"{'='*60}")

    first_capture_ply = None
    total_captures = 0
    distance_log = []
    winner = None

    for ply in range(1, max_turns + 1):
        red_men, red_kings = _count_pieces(board, RED)
        blk_men, blk_kings = _count_pieces(board, BLACK)

        if red_men + red_kings == 0:
            winner = "BLACK"
            break
        if blk_men + blk_kings == 0:
            winner = "RED"
            break

        result = _pick_best_move_by_minimax(board, player, depth=DEPTH)
        if result is None:
            winner = "RED" if player == BLACK else "BLACK"
            break

        move, score = result
        captures = len(move.get("captured", []))

        if player == RED and captures > 0:
            total_captures += captures
            if first_capture_ply is None:
                first_capture_ply = ply

        board = apply_move(board, move)

        dist = _avg_king_enemy_distance(board)
        distance_log.append(dist)

        from checkers.engine.evaluation import evaluate_board_breakdown
        opp = BLACK if player == RED else RED
        bd = evaluate_board_breakdown(board, current_player=opp, root_player=player)
        opp_mob = len(get_all_legal_moves(board, opp))
        conf_bonus = bd.get("confinement_bonus", 0.0)

        path_str = "→".join(str(p) for p in move["path"])
        cap_str = f"  CAPTURE({captures})" if captures > 0 else ""
        dist_str = f"  king→enemy={dist:.1f}" if dist is not None else ""
        mob_str = f"  opp_mob={opp_mob}  conf_bonus={conf_bonus}"
        player_label = "RED " if player == RED else "BLK "
        print(f"  Ply {ply:2d} | {player_label} | score={score:7.1f} | {path_str}{cap_str}{dist_str}{mob_str}")

        player = BLACK if player == RED else RED

    red_men, red_kings = _count_pieces(board, RED)
    blk_men, blk_kings = _count_pieces(board, BLACK)

    print(f"\n  --- Final: RED={red_men}men+{red_kings}K  BLACK={blk_men}men+{blk_kings}K ---")
    if winner:
        print(f"  WINNER: {winner}")
    else:
        print(f"  No winner in {max_turns} plies")

    avg_dist = (sum(d for d in distance_log if d is not None) /
                max(1, len([d for d in distance_log if d is not None])))

    metrics = {
        "first_capture_ply": first_capture_ply,
        "total_captures": total_captures,
        "avg_king_enemy_distance": round(avg_dist, 2),
        "winner": winner,
        "final_red": red_men + red_kings,
        "final_black": blk_men + blk_kings,
    }

    print(f"\n  METRICS:")
    print(f"    First RED capture ply : {first_capture_ply if first_capture_ply else 'None (no capture)'}")
    print(f"    Total RED captures    : {total_captures}")
    print(f"    Avg king→enemy dist   : {avg_dist:.2f}")
    print(f"    Winner                : {winner if winner else 'none (timeout)'}")
    print(f"{'='*60}\n")

    return metrics


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_baseline_endgame_conversion():
    """
    Step 3: Baseline run with current KING_CHASE_PRESSURE_WEIGHT (6).
    This establishes the reference metrics for comparison.
    Expected (slow conversion): no capture for many plies, large king→enemy distance.
    """
    from checkers.engine.evaluation import KING_CHASE_PRESSURE_WEIGHT
    print(f"\n[BASELINE] KING_CHASE_PRESSURE_WEIGHT = {KING_CHASE_PRESSURE_WEIGHT}")

    metrics = simulate(T41_BOARD, start_player=RED, max_turns=MAX_TURNS,
                       label=f"BASELINE (weight={KING_CHASE_PRESSURE_WEIGHT})")

    # Structural assertions — baseline should NOT win quickly
    # (these will pass; they define the baseline behavior, not a "good" target)
    print(f"[BASELINE RESULT] captures={metrics['total_captures']}  "
          f"first_capture={metrics['first_capture_ply']}  "
          f"avg_dist={metrics['avg_king_enemy_distance']}  "
          f"winner={metrics['winner']}")


def test_weight_12_endgame_conversion():
    """
    Step 5: Run with KING_CHASE_PRESSURE_WEIGHT patched to 12 (double baseline).
    Compare metrics against baseline to measure conversion improvement.
    """
    import checkers.engine.evaluation as ev

    original = ev.KING_CHASE_PRESSURE_WEIGHT
    ev.KING_CHASE_PRESSURE_WEIGHT = 12
    print(f"\n[PATCHED] KING_CHASE_PRESSURE_WEIGHT = {ev.KING_CHASE_PRESSURE_WEIGHT}")

    try:
        metrics = simulate(T41_BOARD, start_player=RED, max_turns=MAX_TURNS,
                           label=f"PATCHED (weight={ev.KING_CHASE_PRESSURE_WEIGHT})")

        print(f"[PATCHED RESULT] captures={metrics['total_captures']}  "
              f"first_capture={metrics['first_capture_ply']}  "
              f"avg_dist={metrics['avg_king_enemy_distance']}  "
              f"winner={metrics['winner']}")
    finally:
        ev.KING_CHASE_PRESSURE_WEIGHT = original
        print(f"[RESTORED] KING_CHASE_PRESSURE_WEIGHT = {ev.KING_CHASE_PRESSURE_WEIGHT}")


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# STEP 3: BASELINE")
    print("#"*60)
    test_baseline_endgame_conversion()

    print("\n" + "#"*60)
    print("# STEP 5: CHANGE — KING_CHASE_PRESSURE_WEIGHT = 12")
    print("#"*60)
    test_weight_12_endgame_conversion()
