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
VULNERABLE_MAN_PENALTY = 12
VULNERABLE_KING_PENALTY = 20
STRUCTURE_WEIGHT = 4
ENDGAME_MODEST_WEIGHT = 16
BACK_ROW_GUARD_WEIGHT = 8
PROMOTION_PROXIMITY_WEIGHT = 2
SIMPLIFICATION_AHEAD_WEIGHT = 1
ISOLATION_PENALTY_WEIGHT = 6
CONNECTIVITY_SUPPORT_WEIGHT = 3
FROZEN_RESTRICTION_WEIGHT = 4
ENDGAME_FEATURE_PIECE_THRESHOLD = 10
KING_CENTRALIZATION_WEIGHT = 4
KING_MOBILITY_WEIGHT = 3
KING_CHASE_PRESSURE_WEIGHT = 2

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


def _promotion_proximity(board: list[list[int]], player: int) -> int:
    """
    Promotion proximity for men only.
    Higher means our men have progressed farther toward crowning (not distance remaining).
    """
    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if player == RED and p == RED:
                score += (7 - r)
            elif player == BLACK and p == BLACK:
                score += r
    return score


def _back_row_guard(board: list[list[int]], player: int) -> int:
    """
    Count uncrowned men preserved on the home back row.
    This is a light defensive-structure signal.
    """
    if player == RED:
        home_row = 7
        man = RED
    else:
        home_row = 0
        man = BLACK
    return sum(1 for c in range(BOARD_SIZE) if board[home_row][c] == man)


def _isolation_count(board: list[list[int]], player: int) -> int:
    """
    Count isolated men (no adjacent friendly piece in 8-neighborhood).
    Kept simple and structural; no tactical lookahead.
    """
    man = RED if player == RED else BLACK
    count = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != man:
                continue
            has_friend = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and is_own_piece(board[rr][cc], player):
                        has_friend = True
                        break
                if has_friend:
                    break
            if not has_friend:
                count += 1
    return count


def _support_connectivity(board: list[list[int]], player: int) -> int:
    """
    Count men with direct backward-diagonal friendly support.
    Distinct from _structure_score: this tracks immediate support anchors only.
    """
    man = RED if player == RED else BLACK
    back_row_step = 1 if player == RED else -1
    support = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != man:
                continue
            supported = False
            for dc in (-1, 1):
                rr, cc = r + back_row_step, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and is_own_piece(board[rr][cc], player):
                    supported = True
                    break
            if supported:
                support += 1
    return support


def _frozen_piece_count(board: list[list[int]], player: int) -> int:
    """
    Count pieces with no legal move in the current position.
    Uses legal move starts as the movable-piece set; robust and deterministic.
    """
    movable_starts = {tuple(m["path"][0]) for m in get_all_legal_moves(board, player)}
    frozen = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if is_own_piece(board[r][c], player) and (r, c) not in movable_starts:
                frozen += 1
    return frozen


def _king_positions(board: list[list[int]], player: int) -> list[tuple[int, int]]:
    king = RED_KING if player == RED else BLACK_KING
    out: list[tuple[int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == king:
                out.append((r, c))
    return out


def _all_piece_positions(board: list[list[int]], player: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if is_own_piece(board[r][c], player):
                out.append((r, c))
    return out


def _king_centralization(board: list[list[int]], player: int) -> int:
    """
    Reward kings that are closer to the center 4 squares.
    Uses Manhattan distance-to-nearest-center converted to a closeness score.
    """
    centers = ((3, 3), (3, 4), (4, 3), (4, 4))
    score = 0
    for kr, kc in _king_positions(board, player):
        nearest = min(abs(kr - cr) + abs(kc - cc) for cr, cc in centers)
        score += max(0, 8 - nearest)
    return score


def _king_mobility_count(board: list[list[int]], player: int) -> int:
    """
    Count legal moves whose moving piece is a king.
    Separate from global mobility to expose king-specific endgame activity.
    """
    king = RED_KING if player == RED else BLACK_KING
    count = 0
    for move in get_all_legal_moves(board, player):
        sr, sc = move["path"][0]
        if board[sr][sc] == king:
            count += 1
    return count


def _king_chase_pressure(board: list[list[int]], player: int) -> int:
    """
    Reward kings being near opponent pieces (simple Manhattan-distance pressure).
    """
    opp = _opponent(player)
    opponents = _all_piece_positions(board, opp)
    if not opponents:
        return 0
    score = 0
    for kr, kc in _king_positions(board, player):
        nearest = min(abs(kr - or_) + abs(kc - oc) for or_, oc in opponents)
        score += max(0, 12 - nearest)
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


def _simplification_bonus_when_ahead(material_edge: int, total_pieces: int) -> int:
    """
    When materially ahead, prefer simpler positions slightly.
    Conservative by design: small weight and capped advantage scale.
    """
    if material_edge <= 0:
        return 0
    advantage_units = min(4, material_edge // MAN_VALUE)
    if advantage_units <= 0:
        return 0
    pieces_removed = max(0, 24 - total_pieces)
    return advantage_units * pieces_removed * SIMPLIFICATION_AHEAD_WEIGHT


def evaluate_board_breakdown(
    board: list[list[int]],
    current_player: int,
    root_player: int,
    use_phase7a: bool = True,
) -> dict[str, float]:
    """
    Returns a per-term breakdown for explainability/debugging.
    Score convention matches evaluate_board (root-player perspective).
    """
    opp = _opponent(root_player)

    our_moves = get_all_legal_moves(board, root_player)
    opp_moves = get_all_legal_moves(board, opp)

    if current_player == root_player and not our_moves:
        return {"terminal": float(LOSS_SCORE), "total": float(LOSS_SCORE)}
    if current_player == opp and not opp_moves:
        return {"terminal": float(WIN_SCORE), "total": float(WIN_SCORE)}

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
    proximity = (_promotion_proximity(board, root_player) - _promotion_proximity(board, opp)) * PROMOTION_PROXIMITY_WEIGHT
    back_row_guard = (_back_row_guard(board, root_player) - _back_row_guard(board, opp)) * BACK_ROW_GUARD_WEIGHT
    isolation = (_isolation_count(board, opp) - _isolation_count(board, root_player)) * ISOLATION_PENALTY_WEIGHT
    connectivity_support = (
        _support_connectivity(board, root_player) - _support_connectivity(board, opp)
    ) * CONNECTIVITY_SUPPORT_WEIGHT
    frozen_restriction = (
        _frozen_piece_count(board, opp) - _frozen_piece_count(board, root_player)
    ) * FROZEN_RESTRICTION_WEIGHT
    king_centralization = 0.0
    king_mobility = 0.0
    king_chase_pressure = 0.0
    if use_phase7a and total_pieces <= ENDGAME_FEATURE_PIECE_THRESHOLD:
        king_centralization = float(
            (_king_centralization(board, root_player) - _king_centralization(board, opp))
            * KING_CENTRALIZATION_WEIGHT
        )
        king_mobility = float(
            (_king_mobility_count(board, root_player) - _king_mobility_count(board, opp))
            * KING_MOBILITY_WEIGHT
        )
        king_chase_pressure = float(
            (_king_chase_pressure(board, root_player) - _king_chase_pressure(board, opp))
            * KING_CHASE_PRESSURE_WEIGHT
        )

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
    simplification = _simplification_bonus_when_ahead(material, total_pieces)

    total = float(
        material
        + mobility
        + center
        + promo
        + proximity
        + back_row_guard
        + isolation
        + connectivity_support
        + frozen_restriction
        + king_centralization
        + king_mobility
        + king_chase_pressure
        + vulnerability
        + structure
        + endgame
        + simplification
    )
    return {
        "material": float(material),
        "mobility": float(mobility),
        "center": float(center),
        "promotion_threat": float(promo),
        "promotion_proximity": float(proximity),
        "back_row_guard": float(back_row_guard),
        "isolation": float(isolation),
        "connectivity_support": float(connectivity_support),
        "frozen_restriction": float(frozen_restriction),
        "king_centralization": float(king_centralization),
        "king_mobility": float(king_mobility),
        "king_chase_pressure": float(king_chase_pressure),
        "vulnerability": float(vulnerability),
        "structure": float(structure),
        "endgame": float(endgame),
        "simplification_when_ahead": float(simplification),
        "total": total,
    }


def evaluate_board(
    board: list[list[int]],
    current_player: int,
    root_player: int,
    use_phase7a: bool = True,
) -> float:
    breakdown = evaluate_board_breakdown(board, current_player, root_player, use_phase7a=use_phase7a)
    return float(breakdown["total"])