# nodes/inter_turn_memory.py
#
# Purely symbolic inter-turn memory node.
# Computes board facts, trends, patterns, and strategic priorities
# for the current player before the proposal agent runs.
#
# Priority logic rewritten (v2):
#   - MAX_PRIORITIES = 6 hard cap (phi4-mini cannot reliably apply more)
#   - RESOLVE_TACTICS removed (ranker Step 2 handles captures unconditionally)
#   - DEFEND_PIECES always in Tier 1 when our_vulnerable_pieces > 0
#   - Tiers: Safety → Promotion → Material → Phase → Positional → Pattern

from __future__ import annotations

from typing import Any, Optional

from checkers.engine.board import (
    BLACK,
    BLACK_KING,
    BOARD_SIZE,
    RED,
    RED_KING,
    in_bounds,
    is_own_piece,
)
from checkers.engine.rules import get_all_legal_moves
from checkers.state.state import CheckersState

MAX_PRIORITIES = 6


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


def _center_distance(row: int, col: int) -> int:
    centers = ((3, 3), (3, 4), (4, 3), (4, 4))
    return min(abs(row - r) + abs(col - c) for r, c in centers)


def _count_piece_types(board: list[list[int]], player: int) -> tuple[int, int]:
    if player == RED:
        man, king = RED, RED_KING
    else:
        man, king = BLACK, BLACK_KING
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


def _count_center_pieces(board: list[list[int]], player: int) -> int:
    count = 0
    for r in (3, 4):
        for c in (2, 3, 4, 5):
            if is_own_piece(board[r][c], player):
                count += 1
    return count


def _count_promotion_threats(board: list[list[int]], player: int) -> int:
    count = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if player == RED and p == RED and r in (1, 2):
                count += 1
            elif player == BLACK and p == BLACK and r in (5, 6):
                count += 1
    return count


def _capturable_squares(moves: list[dict[str, Any]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for m in moves:
        if m.get("type") == "jump":
            for sq in m.get("captured", []):
                if isinstance(sq, (list, tuple)) and len(sq) == 2:
                    out.add((sq[0], sq[1]))
    return out


def _count_vulnerable(
    board: list[list[int]], player: int, opp_moves: list[dict[str, Any]]
) -> int:
    capturable = _capturable_squares(opp_moves)
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if is_own_piece(board[r][c], player) and (r, c) in capturable:
                n += 1
    return n


def _count_protected(board: list[list[int]], player: int) -> int:
    behind_dr = 1 if player == RED else -1
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_own_piece(board[r][c], player):
                continue
            protected = False
            for dc in (-1, 1):
                rr = r + behind_dr
                cc = c + dc
                if in_bounds(rr, cc) and is_own_piece(board[rr][cc], player):
                    protected = True
                    break
            if protected:
                n += 1
    return n


def _count_back_row(board: list[list[int]], player: int) -> int:
    row = 7 if player == RED else 0
    n = 0
    for c in range(BOARD_SIZE):
        if is_own_piece(board[row][c], player):
            n += 1
    return n


def _count_isolated(board: list[list[int]], player: int) -> int:
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_own_piece(board[r][c], player):
                continue
            has_diag_friend = False
            for dr in (-1, 1):
                for dc in (-1, 1):
                    rr = r + dr
                    cc = c + dc
                    if in_bounds(rr, cc) and is_own_piece(board[rr][cc], player):
                        has_diag_friend = True
                        break
                if has_diag_friend:
                    break
            if not has_diag_friend:
                n += 1
    return n


def _king_centrality(board: list[list[int]], player: int) -> int:
    king_piece = RED_KING if player == RED else BLACK_KING
    total = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == king_piece:
                total += _center_distance(r, c)
    return total


def _opp_flank_counts(board: list[list[int]], player: int) -> tuple[int, int]:
    opp = _opponent(player)
    left = 0
    right = 0
    for r in range(BOARD_SIZE):
        for c in (0, 1):
            if is_own_piece(board[r][c], opp):
                left += 1
        for c in (6, 7):
            if is_own_piece(board[r][c], opp):
                right += 1
    return left, right


def _game_phase(total_pieces: int, our_kings: int, opp_kings: int) -> str:
    total_kings = our_kings + opp_kings

    # If any kings exist → not opening anymore
    if total_kings > 0:
        if total_pieces > 10:
            return "MIDGAME"
        return "ENDGAME"

    # No kings yet → early structure matters
    if total_pieces > 20:
        return "OPENING"
    elif total_pieces > 12:
        return "MIDGAME"
    else:
        return "ENDGAME"


def _append_once(out: list[str], item: str) -> None:
    if item not in out:
        out.append(item)


def inter_turn_memory(state: CheckersState) -> dict:
    board = state.board
    player = state.current_player
    opp = _opponent(player)

    # ── Symbolic board facts ─────────────────────────────────────────────────
    our_men, our_kings = _count_piece_types(board, player)
    opp_men, opp_kings = _count_piece_types(board, opp)
    our_moves = get_all_legal_moves(board, player)
    opp_moves = get_all_legal_moves(board, opp)

    material_advantage = (our_men + 2 * our_kings) - (opp_men + 2 * opp_kings)
    king_advantage = our_kings - opp_kings
    total_pieces = our_men + our_kings + opp_men + opp_kings
    our_mobility = len(our_moves)
    opp_mobility = len(opp_moves)
    mobility_advantage = our_mobility - opp_mobility
    our_center_pieces = _count_center_pieces(board, player)
    opp_center_pieces = _count_center_pieces(board, opp)
    center_control_advantage = our_center_pieces - opp_center_pieces
    our_promotion_threats = _count_promotion_threats(board, player)
    opp_promotion_threats = _count_promotion_threats(board, opp)
    our_vulnerable_pieces = _count_vulnerable(board, player, opp_moves)
    opp_vulnerable_pieces = _count_vulnerable(board, opp, our_moves)
    _our_protected_pieces = _count_protected(board, player)
    our_back_row_count = _count_back_row(board, player)
    _opp_back_row_count = _count_back_row(board, opp)
    _our_king_centrality = _king_centrality(board, player)
    _our_isolated_pieces = _count_isolated(board, player)
    left_flank_opp, right_flank_opp = _opp_flank_counts(board, player)

    position_is_stable = (
        not any(m.get("type") == "jump" for m in our_moves)
        and not any(m.get("type") == "jump" for m in opp_moves)
        and our_promotion_threats == 0
        and opp_promotion_threats == 0
    )
    game_phase = _game_phase(total_pieces, our_kings, opp_kings)

    # ── Sliding window ───────────────────────────────────────────────────────
    prev = state.strategic_context or {}
    turn_history: list[dict[str, Any]] = list(prev.get("turn_history", []))
    archive_summary: list[dict[str, Any]] = list(prev.get("archive_summary", []))
    turn_snapshot = {
        "material_advantage": material_advantage,
        "mobility_advantage": mobility_advantage,
        "center_control_advantage": center_control_advantage,
        "our_mobility": our_mobility,
        "opp_promotion_threats": opp_promotion_threats,
        "opp_left_flank_count": left_flank_opp,
        "opp_right_flank_count": right_flank_opp,
        "game_phase": game_phase,
        "position_is_stable": position_is_stable,
    }
    turn_history.append(turn_snapshot)
    if len(turn_history) > 5:
        oldest = turn_history.pop(0)
        archive_summary.append(
            {
                "material_advantage": oldest.get("material_advantage", 0),
                "game_phase": oldest.get("game_phase", "MIDGAME"),
            }
        )

    # ── Trends ───────────────────────────────────────────────────────────────
    material_trend: Optional[int] = None
    mobility_trend: Optional[int] = None
    center_trend: Optional[int] = None
    if len(turn_history) >= 4:
        now = turn_history[-1]
        past = turn_history[-4]
        material_trend = now["material_advantage"] - past["material_advantage"]
        mobility_trend = now["mobility_advantage"] - past["mobility_advantage"]
        center_trend = now["center_control_advantage"] - past["center_control_advantage"]

    # ── Pattern detection ────────────────────────────────────────────────────
    active_patterns: list[str] = []
    if len(turn_history) >= 3:
        t0, t1, t2 = turn_history[-3], turn_history[-2], turn_history[-1]
        if all(t["opp_left_flank_count"] >= 2 for t in (t0, t1, t2)):
            active_patterns.append("OPPONENT_LEFT_FLANK_PUSH")
        if all(t["opp_right_flank_count"] >= 2 for t in (t0, t1, t2)):
            active_patterns.append("OPPONENT_RIGHT_FLANK_PUSH")
        if t0["material_advantage"] > t1["material_advantage"] > t2["material_advantage"]:
            active_patterns.append("MATERIAL_BLEEDING")
        if all(t["center_control_advantage"] < 0 for t in (t0, t1, t2)):
            active_patterns.append("WE_ARE_LOSING_CENTER")
        if all(t["our_mobility"] < 4 for t in (t0, t1, t2)):
            active_patterns.append("MOBILITY_TRAP")
        if all(t["opp_promotion_threats"] >= 2 for t in (t0, t1, t2)):
            active_patterns.append("OPPONENT_PROMOTION_PRESSURE")
# ── Winning score ────────────────────────────────────────────────────────
    # Clamp mobility to avoid wild score swings from single-piece moves
    clamped_mobility = max(-3, min(3, mobility_advantage))
    winning_score = (
        (material_advantage * 3)
        + (king_advantage * 2)
        + clamped_mobility
        + center_control_advantage
    )

    # ── Stagnation detection ────────────────────────────────────────────────
    stagnation_detected = False

    if len(turn_history) >= 4:
        recent4 = turn_history[-4:]

        same_material = all(
            t["material_advantage"] == recent4[0]["material_advantage"]
            for t in recent4
        )

        same_center = all(
            t["center_control_advantage"] == recent4[0]["center_control_advantage"]
            for t in recent4
        )

        similar_mobility = (
            max(t["our_mobility"] for t in recent4) -
            min(t["our_mobility"] for t in recent4)
        ) <= 1

        if (
            same_material
            and same_center
            and similar_mobility
            and winning_score >= 0
            and game_phase != "OPENING"
        ):
            stagnation_detected = True
            active_patterns.append("STAGNATION_LOOP_RISK")

    # ── Score-state classification ───────────────────────────────────────────
    if winning_score >= 12:
        score_state = "CLEARLY_WINNING"
    elif winning_score >= 4:
        score_state = "SLIGHTLY_WINNING"
    elif winning_score <= -12:
        score_state = "CLEARLY_LOSING"
    elif winning_score <= -4:
        score_state = "SLIGHTLY_LOSING"
    else:
        score_state = "EQUAL"

    # ── Priority construction (v2) ───────────────────────────────────────────
    # Hard cap: MAX_PRIORITIES = 6.
    # phi4-mini on 8GB RAM cannot reliably apply more than 6 ordered constraints.
    # Tiers are evaluated in order; once the cap is reached, lower tiers are skipped.
    # RESOLVE_TACTICS removed: the ranker's Step 2 handles captures unconditionally
    # and does not need a priority to trigger it.

    priorities: list[str] = []

    def _add(p: str) -> bool:
        """Add p to priorities if cap not reached and not already present."""
        if len(priorities) >= MAX_PRIORITIES:
            return False
        _append_once(priorities, p)
        return True

    # TIER 1 — Immediate safety
    # Must come first so the ranker's strategic context reinforces Step 1 safety check.
    if our_vulnerable_pieces > 0:
        _add("DEFEND_PIECES")
    if "MATERIAL_BLEEDING" in active_patterns:
        _add("DEFEND")
    if material_trend is not None and material_trend < 0 and material_advantage >= 0:
        _add("DEFEND")

    # TIER 2 — Promotion threats (time-critical, 1-2 move horizon)
    if opp_promotion_threats > 0:
        _add("BLOCK_PROMOTION")
    if our_promotion_threats > 0:
        _add("PROMOTE")

    # TIER 3 — Score-state / material exploitation
    if score_state == "CLEARLY_WINNING":
        if game_phase != "OPENING":
            _add("CONVERT_ADVANTAGE")
            _add("TRADE_WHEN_AHEAD")
        _add("PLAY_SAFE")

    elif score_state == "SLIGHTLY_WINNING":
        _add("HOLD_ADVANTAGE")
        if game_phase != "OPENING" and material_advantage > 0:
            _add("TRADE_WHEN_AHEAD")

    elif score_state == "CLEARLY_LOSING":
        _add("SEEK_COUNTERPLAY")
        _add("COMPLICATE")
        _add("AVOID_TRADES")

    elif score_state == "SLIGHTLY_LOSING":
        _add("CREATE_THREATS")
        _add("AVOID_TRADES")

    else:
        if game_phase != "OPENING" and material_advantage > 0:
            _add("TRADE_WHEN_AHEAD")
        elif material_advantage < 0:
            _add("AVOID_TRADES")

    # TIER 4 — Phase-specific structural goals
    if game_phase == "OPENING":
        _add("CONTROL_CENTER")
        _add("DEVELOP_PIECES")
        if our_back_row_count >= 2:
            _add("MAINTAIN_BACK_ROW")

    elif game_phase == "MIDGAME":
        if center_control_advantage < 0:
            _add("CONTROL_CENTER")
        _add("INCREASE_MOBILITY")

        if our_kings > 0:
            _add("ACTIVATE_KINGS")

        if score_state in ("CLEARLY_WINNING", "SLIGHTLY_WINNING"):
            _add("REDUCE_OPP_MOBILITY")
        elif score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING"):
            _add("CREATE_THREATS")
            if our_kings > 0:
                _add("ACTIVATE_KINGS")

    elif game_phase == "ENDGAME":
        _add("ACTIVATE_KINGS")
        _add("PROMOTE")
        if stagnation_detected:
            _add("REDUCE_OPP_MOBILITY")
            _add("CREATE_THREATS")
        if score_state in ("CLEARLY_WINNING", "SLIGHTLY_WINNING"):
            _add("CONVERT_ADVANTAGE")
            _add("REDUCE_OPP_MOBILITY")
        elif score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING"):
            _add("SEEK_COUNTERPLAY")

    # TIER 5 — Positional refinements
    if opp_vulnerable_pieces > 0:
        _add("ATTACK_WEAK_PIECES")
    if mobility_advantage > 2:
        _add("REDUCE_OPP_MOBILITY")
    if _our_isolated_pieces > 2:
        _add("CONSOLIDATE_PIECES")
    if score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING"):
        _add("CREATE_THREATS")

    # TIER 6 — Pattern response
    if len(priorities) < MAX_PRIORITIES:
        if "STAGNATION_LOOP_RISK" in active_patterns:
            if score_state in ("CLEARLY_WINNING", "SLIGHTLY_WINNING"):
                _add("CONVERT_ADVANTAGE")
                _add("REDUCE_OPP_MOBILITY")
                _add("TRADE_WHEN_AHEAD")
                if game_phase == "ENDGAME":
                    _add("ACTIVATE_KINGS")
            else:
                _add("CONTROL_CENTER")
                _add("INCREASE_MOBILITY")
                _add("ATTACK_WEAK_PIECES")
        elif "MOBILITY_TRAP" in active_patterns:
            _add("INCREASE_MOBILITY")
        elif "WE_ARE_LOSING_CENTER" in active_patterns:
            _add("CONTROL_CENTER")
        elif "OPPONENT_LEFT_FLANK_PUSH" in active_patterns:
            _add("DEFEND_LEFT_FLANK")
        elif "OPPONENT_RIGHT_FLANK_PUSH" in active_patterns:
            _add("DEFEND_RIGHT_FLANK")
        elif "OPPONENT_PROMOTION_PRESSURE" in active_patterns:
            _add("BLOCK_PROMOTION")

    # Score-state modifiers
    if score_state == "CLEARLY_WINNING":
        _add("PLAY_SAFE")
    elif score_state == "CLEARLY_LOSING":
        _add("COMPLICATE")
        _add("SEEK_COUNTERPLAY")

    # ── strategic_context output ─────────────────────────────────────────────
    strategic_context = {
        "stagnation_detected": stagnation_detected,
        "score_state": score_state,
        "material_advantage": material_advantage,
        "king_advantage": king_advantage,
        "mobility_advantage": mobility_advantage,
        "center_control_advantage": center_control_advantage,
        "our_promotion_threats": our_promotion_threats,
        "opp_promotion_threats": opp_promotion_threats,
        "our_vulnerable_pieces": our_vulnerable_pieces,
        "opp_vulnerable_pieces": opp_vulnerable_pieces,
        "our_back_row_count": our_back_row_count,
        "position_is_stable": position_is_stable,
        "game_phase": game_phase,
        "material_trend": material_trend,
        "mobility_trend": mobility_trend,
        "center_trend": center_trend,
        "active_patterns": active_patterns,
        "strategic_priorities": priorities,
        "winning_score": winning_score,
        "turn_number": state.turn_number,
        "turn_history": turn_history,
        "archive_summary": archive_summary,
        "our_protected_pieces": _our_protected_pieces,
        "our_king_centrality": _our_king_centrality,
        "our_isolated_pieces": _our_isolated_pieces,
        "opp_back_row_count": _opp_back_row_count,
    }
    return {
        "strategic_context": strategic_context,
        "last_completed_node": "inter_turn_memory",
    }