"""
Simplified symbolic evaluator for checkers minimax.

Design goals:
- deterministic and interpretable
- stable score ordering for ranker guardrails
- robust core terms only (no stacked fragile special cases)
"""

from __future__ import annotations

from checkers.engine.board import (
    BLACK,
    BLACK_KING,
    BOARD_SIZE,
    RED,
    RED_KING,
    is_own_piece,
)
from checkers.engine.rules import get_all_legal_moves

# ── Terminal rewards ──────────────────────────────────────────────────────────
WIN_SCORE = 10_000
LOSS_SCORE = -10_000

# ── Core weights ──────────────────────────────────────────────────────────────
MAN_VALUE = 100
KING_VALUE = 175
MOBILITY_WEIGHT_OPENING = 3
MOBILITY_WEIGHT_ENDGAME = 5
CENTER_WEIGHT = 6
PROMOTION_THREAT_WEIGHT = 14
ADVANCEMENT_WEIGHT = 3
VULNERABLE_MAN_PENALTY = 12
VULNERABLE_KING_PENALTY = 20
STRUCTURE_WEIGHT = 4
ENDGAME_MODEST_WEIGHT = 16

_CENTER_SQUARES = frozenset((r, c) for r in (3, 4) for c in (2, 3, 4, 5))


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


def _count_material(board: list[list[int]], player: int) -> tuple[int, int]:
    man = RED if player == RED else BLACK
    king = RED_KING if player == RED else BLACK_KING
    men = 0
    kings = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if p == man:
                men += 1
            elif p == king:
                kings += 1
    return men, kings


def _total_pieces(board: list[list[int]]) -> int:
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != 0:
                n += 1
    return n


def _center_control(board: list[list[int]], player: int) -> int:
    count = 0
    for r, c in _CENTER_SQUARES:
        if is_own_piece(board[r][c], player):
            count += 1
    return count


def _promotion_threats(board: list[list[int]], player: int) -> int:
    threats = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if player == RED and p == RED and r == 1:
                threats += 1
            elif player == BLACK and p == BLACK and r == 6:
                threats += 1
    return threats


def _advancement(board: list[list[int]], player: int) -> int:
    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if player == RED and p == RED:
                score += (7 - r)
            elif player == BLACK and p == BLACK:
                score += r
    return score


def _threatened_squares(board: list[list[int]], attacker: int) -> set[tuple[int, int]]:
    threatened: set[tuple[int, int]] = set()
    for m in get_all_legal_moves(board, attacker):
        if m.get("type") == "jump":
            for cap in m.get("captured", []):
                threatened.add((cap[0], cap[1]))
    return threatened


def _vulnerability_penalty(
    board: list[list[int]],
    player: int,
    threatened_by_opp: set[tuple[int, int]],
) -> int:
    man = RED if player == RED else BLACK
    king = RED_KING if player == RED else BLACK_KING
    penalty = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) not in threatened_by_opp:
                continue
            p = board[r][c]
            if p == man:
                penalty += VULNERABLE_MAN_PENALTY
            elif p == king:
                penalty += VULNERABLE_KING_PENALTY
    return penalty


def _structure_score(board: list[list[int]], player: int) -> int:
    # Reward local diagonal connectivity; light, stable signal.
    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_own_piece(board[r][c], player):
                continue
            neighbors = 0
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and is_own_piece(board[rr][cc], player):
                    neighbors += 1
            if neighbors >= 2:
                score += 2
            elif neighbors == 1:
                score += 1
    return score


def _modest_endgame_adjustment(
    our_mobility: int,
    opp_mobility: int,
    our_kings: int,
    opp_kings: int,
    total_pieces: int,
) -> int:
    # Single modest endgame term: king activity + mobility edge.
    if total_pieces > 10:
        return 0
    king_edge = (our_kings - opp_kings) * 2
    mobility_edge = our_mobility - opp_mobility
    return (king_edge + mobility_edge) * ENDGAME_MODEST_WEIGHT


def evaluate_board(
    board: list[list[int]],
    current_player: int,
    root_player: int,
) -> float:
    opp = _opponent(root_player)

    our_moves = get_all_legal_moves(board, root_player)
    opp_moves = get_all_legal_moves(board, opp)

    # Terminal from side-to-move perspective.
    if current_player == root_player and not our_moves:
        return float(LOSS_SCORE)
    if current_player == opp and not opp_moves:
        return float(WIN_SCORE)

    our_men, our_kings = _count_material(board, root_player)
    opp_men, opp_kings = _count_material(board, opp)
    total_pieces = _total_pieces(board)

    material = (our_men * MAN_VALUE + our_kings * KING_VALUE) - (
        opp_men * MAN_VALUE + opp_kings * KING_VALUE
    )

    our_mob = len(our_moves)
    opp_mob = len(opp_moves)
    mob_w = MOBILITY_WEIGHT_ENDGAME if total_pieces <= 10 else MOBILITY_WEIGHT_OPENING
    mobility = (our_mob - opp_mob) * mob_w

    center = (_center_control(board, root_player) - _center_control(board, opp)) * CENTER_WEIGHT
    promo = (_promotion_threats(board, root_player) - _promotion_threats(board, opp)) * PROMOTION_THREAT_WEIGHT
    advance = (_advancement(board, root_player) - _advancement(board, opp)) * ADVANCEMENT_WEIGHT

    our_threatened = _threatened_squares(board, opp)
    opp_threatened = _threatened_squares(board, root_player)
    vulnerability = _vulnerability_penalty(board, opp, opp_threatened) - _vulnerability_penalty(
        board, root_player, our_threatened
    )

    structure = (_structure_score(board, root_player) - _structure_score(board, opp)) * STRUCTURE_WEIGHT

    endgame = _modest_endgame_adjustment(
        our_mobility=our_mob,
        opp_mobility=opp_mob,
        our_kings=our_kings,
        opp_kings=opp_kings,
        total_pieces=total_pieces,
    )

    return float(material + mobility + center + promo + advance + vulnerability + structure + endgame)