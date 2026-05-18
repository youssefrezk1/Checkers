# checkers/data/pdn_importer/scenario_generator.py
"""
Scenario generator — second stage of the legality-stress pipeline.

Input : list of raw position dicts produced by pdn_parser.parse_pdn_file()
Output: filtered, classified JSONL records exported to
        checkers/data/legality_stress/scenarios.jsonl

Each exported scenario:
    scenario_id         : "src_G{game_idx}_P{ply_idx}"
    source_file         : PDN filename
    game_index          : int
    ply_index           : int
    board               : 8×8 list[list[int]]  (0-4 piece codes)
    side_to_move        : "BLACK" | "RED"
    hidden_legal_moves  : list of engine move dicts (NOT shown to LLMs)
    category            : one of 9 scenario categories (see below)
    expected_rule       : natural-language rule description
    difficulty          : "easy" | "medium" | "hard"

Category detection logic
------------------------
Priority order (first match wins):

  1. mandatory_capture      — captures exist → ALL legal moves are captures
  2. multi_jump_required    — ≥1 legal move has ≥2 captures (multi-hop)
  3. king_vs_man_confusion  — mixed king+man positions where the mover has
                              both king and man pieces
  4. promotion_state_update — any legal move lands a man on the back rank
  5. wrong_direction_trap   — man pieces that have NO forward simple moves
                              (trapped or would move backward if they could)
  6. occupied_destination_trap — simple move candidates where ALL destination
                                  squares are occupied (≥1 piece fully hemmed)
  7. wrong_player_piece_trap — opponent pieces far outnumber own pieces
                               (opponent ≥ 2× own)
  8. state_update_after_capture — exactly one capture available
  9. crowded_board           — total pieces ≥ 16 (dense board)
  fallback: mandatory_capture (at least we know captures are forced)

Difficulty heuristic
--------------------
  hard   : multi-jump, or captures mandatory with ≥ 3 legal moves, or
            promotion possible AND mandatory capture
  medium : any capture, or promotion possible, or piece imbalance ≥ 2
  easy   : everything else
"""

import os
import json
import copy
import logging
from typing import Optional

from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING, BOARD_SIZE
)
from checkers.engine.rules import get_all_legal_moves
from checkers.data.pdn_importer.fen_utils import str_to_side

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "mandatory_capture",
    "multi_jump_required",
    "king_vs_man_confusion",
    "promotion_state_update",
    "wrong_direction_trap",
    "occupied_destination_trap",
    "wrong_player_piece_trap",
    "state_update_after_capture",
    "crowded_board",
]

EXPECTED_RULES = {
    "mandatory_capture":         "If any capture is available, the player MUST capture; non-capture moves are illegal.",
    "multi_jump_required":       "When a multi-jump sequence is available, the player must complete the full sequence without stopping early.",
    "king_vs_man_confusion":     "Kings move in all four diagonal directions; men are restricted to their forward direction only.",
    "promotion_state_update":    "When a man reaches the opponent's back rank it is immediately promoted to king; the turn ends at that square.",
    "wrong_direction_trap":      "Men cannot move backward; only kings have four-directional mobility.",
    "occupied_destination_trap": "A piece cannot move to a square that is already occupied by any piece.",
    "wrong_player_piece_trap":   "A player may only move their own pieces; opponent pieces cannot be moved.",
    "state_update_after_capture":"After a capture the captured piece must be immediately removed from the board before evaluating further moves.",
    "crowded_board":             "On a crowded board mandatory-capture and direction rules interact in non-obvious ways; all pieces must be tracked carefully.",
}


# ---------------------------------------------------------------------------
# Board analysis helpers
# ---------------------------------------------------------------------------

def _count_pieces(board, side: int) -> tuple:
    """Return (men_count, king_count) for the given side."""
    if side == RED:
        man, king = RED, RED_KING
    else:
        man, king = BLACK, BLACK_KING

    men = sum(
        1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        if board[r][c] == man
    )
    kings = sum(
        1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        if board[r][c] == king
    )
    return men, kings


def _total_pieces(board) -> int:
    return sum(
        1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        if board[r][c] != EMPTY
    )


def _has_promotion_move(legal_moves, side: int) -> bool:
    """True if any legal simple move or jump-landing promotes a man."""
    back_rank = 0 if side == RED else BOARD_SIZE - 1
    for mv in legal_moves:
        last_r, last_c = mv["path"][-1]
        if last_r == back_rank:
            return True
    return False


def _man_forward_blocked(board, side: int) -> bool:
    """
    True if at least one man of `side` has NO forward simple moves at all
    (blocked in its forward direction by own or opponent pieces).
    """
    if side == RED:
        man_piece = RED
        forward_dirs = [(-1, -1), (-1, 1)]
    else:
        man_piece = BLACK
        forward_dirs = [(1, -1), (1, 1)]

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != man_piece:
                continue
            blocked = True
            for dr, dc in forward_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == EMPTY:
                        blocked = False
                        break
            if blocked:
                return True
    return False


def _has_occupied_destination(board, side: int) -> bool:
    """
    True if at least one piece of `side` has ALL of its diagonal neighbours
    occupied (cannot move forward to any empty square, ignoring captures).
    This is a softer signal of the 'occupied_destination_trap' scenario.
    """
    if side == RED:
        own_pieces = (RED, RED_KING)
    else:
        own_pieces = (BLACK, BLACK_KING)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] not in own_pieces:
                continue
            all_blocked = True
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr][nc] == EMPTY:
                            all_blocked = False
            if all_blocked:
                return True
    return False


# ---------------------------------------------------------------------------
# Category + difficulty classification
# ---------------------------------------------------------------------------

def _classify(board, side: int, legal_moves: list) -> tuple:
    """
    Returns (category: str, difficulty: str).
    """
    has_captures = any(mv["type"] == "jump" for mv in legal_moves)
    all_captures = has_captures and all(mv["type"] == "jump" for mv in legal_moves)
    has_multi    = any(len(mv["captured"]) >= 2 for mv in legal_moves)
    n_legal      = len(legal_moves)

    own_men, own_kings = _count_pieces(board, side)
    opp_side = BLACK if side == RED else RED
    opp_men, opp_kings = _count_pieces(board, opp_side)
    own_total = own_men + own_kings
    opp_total = opp_men + opp_kings

    has_promo   = _has_promotion_move(legal_moves, side)
    total_pcs   = _total_pieces(board)

    # --- category ---
    if has_multi:
        category = "multi_jump_required"
    elif all_captures and n_legal >= 1:
        category = "mandatory_capture"
    elif own_kings > 0 and own_men > 0:
        category = "king_vs_man_confusion"
    elif has_promo:
        category = "promotion_state_update"
    elif _man_forward_blocked(board, side):
        category = "wrong_direction_trap"
    elif _has_occupied_destination(board, side):
        category = "occupied_destination_trap"
    elif opp_total >= 2 * own_total and own_total > 0:
        category = "wrong_player_piece_trap"
    elif has_captures and n_legal == 1:
        category = "state_update_after_capture"
    elif total_pcs >= 16:
        category = "crowded_board"
    else:
        # Fallback — if there are any captures at all, label as mandatory_capture
        category = "mandatory_capture" if has_captures else "crowded_board"

    # --- difficulty ---
    if has_multi or (all_captures and n_legal >= 3) or (has_promo and has_captures):
        difficulty = "hard"
    elif has_captures or has_promo or abs(own_total - opp_total) >= 2:
        difficulty = "medium"
    else:
        difficulty = "easy"

    return category, difficulty


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_scenarios(raw_positions: list) -> list:
    """
    Convert raw position dicts → validated, classified scenario dicts.
    Duplicates (same board + side) are deduplicated.
    """
    seen = set()
    scenarios = []

    for pos in raw_positions:
        board = pos["board"]
        side_str = pos["side_to_move"]
        side = str_to_side(side_str)

        # De-duplicate on (board_fingerprint, side)
        fingerprint = (
            tuple(cell for row in board for cell in row),
            side_str
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        legal_moves = get_all_legal_moves(board, side)
        if not legal_moves:
            continue

        category, difficulty = _classify(board, side, legal_moves)

        src = os.path.splitext(pos["source_file"])[0]
        scenario_id = f"{src}_G{pos['game_index']}_P{pos['ply_index']}"

        scenarios.append({
            "scenario_id":       scenario_id,
            "source_file":       pos["source_file"],
            "game_index":        pos["game_index"],
            "ply_index":         pos["ply_index"],
            "board":             board,
            "side_to_move":      side_str,
            "hidden_legal_moves": legal_moves,
            "category":          category,
            "expected_rule":     EXPECTED_RULES[category],
            "difficulty":        difficulty,
        })

    return scenarios


def export_jsonl(scenarios: list, output_path: str) -> None:
    """Write scenarios to a JSONL file, one JSON object per line."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sc in scenarios:
            f.write(json.dumps(sc) + '\n')
    log.info("Exported %d scenarios to %s", len(scenarios), output_path)


def category_counts(scenarios: list) -> dict:
    """Return {category: count} dict."""
    counts = {cat: 0 for cat in CATEGORIES}
    for sc in scenarios:
        counts[sc["category"]] = counts.get(sc["category"], 0) + 1
    return counts
