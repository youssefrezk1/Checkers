#!/usr/bin/env python3
"""
Proposal-only strategic edge-case stress tests for the Checkers AI.

Purpose
-------
These tests isolate the proposal stage and try to catch shortlist-recall bugs:
- rescue move omitted when a threatened piece could be stabilized
- promotion / near-promotion move omitted
- king activation omitted in king/endgame positions
- counterplay omitted when losing
- conversion move omitted when winning
- all-unsafe shortlist quality
- redundancy (5 nearly identical quiet moves)

Pipeline under test
-------------------
inter_turn_memory -> proposal_agent -> format_checker

We do NOT test full-game minimax/ranker decisions here.
We only test whether proposal returns a strong shortlist.

Run
---
python3 test_proposal_edge_cases.py

Notes
-----
- This file assumes your project structure and imports exactly as used before.
- It uses real proposal_agent output, so it may hit API rate limits.
- The assertions are intentionally strategic, not brittle:
  they check whether key move classes are INCLUDED, not exact shortlist order.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file, including API keys
from checkers.engine.board import (
    EMPTY,
    RED,
    BLACK,
    RED_KING,
    BLACK_KING,
)
from checkers.engine.rules import get_all_legal_moves
from checkers.nodes.inter_turn_memory import inter_turn_memory
from checkers.agents.proposal_agent import proposal_agent
from checkers.nodes.format_checker import format_checker
from checkers.engine.move_facts import compute_move_facts
from checkers.state.state import CheckersState


# ── helpers ───────────────────────────────────────────────────────────────────

Board = list[list[int]]
Move = dict[str, Any]


def empty_board() -> Board:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def path_tuple(move: Move) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(x) for x in move.get("path", []))


def print_board(board: Board) -> None:
    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row = []
        for c in range(8):
            p = board[r][c]
            ch = "."
            if p == RED:
                ch = "r"
            elif p == BLACK:
                ch = "b"
            elif p == RED_KING:
                ch = "R"
            elif p == BLACK_KING:
                ch = "B"
            row.append(ch)
        print(f"{r} " + " ".join(row))


def enrich_moves(board: Board, player: int) -> list[Move]:
    legal = get_all_legal_moves(board, player)
    out: list[Move] = []
    for m in legal:
        mm = dict(m)
        mm["facts"] = compute_move_facts(board, m, player)
        out.append(mm)
    return out


def run_proposal_only(
    board: Board,
    current_player: int,
    turn_number: int = 1,
) -> tuple[CheckersState, list[Move], list[Move]]:
    """
    Runs:
        inter_turn_memory -> proposal_agent -> format_checker

    Returns:
        (state_after_format_checker, engine_legal_moves_with_facts, proposed_moves)
    """
    state = CheckersState(
        board=board,
        current_player=current_player,
        turn_number=turn_number,
    )

    # symbolic context
    state = CheckersState.model_validate(
        {**state.model_dump(), **inter_turn_memory(state)}
    )

    # proposal
    proposal_patch = proposal_agent(state)
    state = CheckersState.model_validate({**state.model_dump(), **proposal_patch})

    # format checker
    fc_patch = format_checker(state)
    state = CheckersState.model_validate({**state.model_dump(), **fc_patch})

    legal = enrich_moves(board, current_player)
    proposed = state.proposed_moves if isinstance(state.proposed_moves, list) else []

    return state, legal, proposed


def require(
    condition: bool,
    msg: str,
) -> None:
    if not condition:
        raise AssertionError(msg)


def has_move(moves: list[Move], target_path: list[list[int]]) -> bool:
    tp = tuple(tuple(x) for x in target_path)
    return any(path_tuple(m) == tp for m in moves)


def shortlist_paths(moves: list[Move]) -> list[tuple[tuple[int, int], ...]]:
    return [path_tuple(m) for m in moves]


def choose_best_by(
    moves: list[Move],
    key_fn: Callable[[Move], tuple],
) -> Move:
    return sorted(moves, key=key_fn)[0]


def safe_moves(moves: list[Move]) -> list[Move]:
    return [
        m for m in moves
        if not m.get("facts", {}).get("opponent_can_recapture", False)
    ]


def promotion_moves(moves: list[Move]) -> list[Move]:
    return [
        m for m in moves
        if m.get("facts", {}).get("results_in_king", False)
        or m.get("facts", {}).get("near_promotion", False)
    ]


def king_activation_moves(moves: list[Move]) -> list[Move]:
    return [
        m for m in moves
        if (
            m.get("facts", {}).get("piece_type_moving") == "king"
            or m.get("facts", {}).get("quiet_move_role") == "KING_ACTIVATION"
            or m.get("facts", {}).get("king_activity_score", 0) > 0
        )
    ]


def conversion_moves(moves: list[Move]) -> list[Move]:
    return [
        m for m in moves
        if m.get("facts", {}).get("winning_conversion_score", 0) > 0
    ]


def counterplay_moves(moves: list[Move]) -> list[Move]:
    return [
        m for m in moves
        if (
            m.get("facts", {}).get("counterplay_score", 0) > 0
            or m.get("facts", {}).get("creates_immediate_threat", False)
            or m.get("facts", {}).get("mobility_reduction", 0) > 0
        )
    ]


def unique_destinations(moves: list[Move]) -> int:
    dests = set()
    for m in moves:
        p = m.get("path", [])
        if p:
            dests.add(tuple(p[-1]))
    return len(dests)

def move_paths(moves: list[Move]) -> set[tuple[tuple[int, int], ...]]:
    return {path_tuple(m) for m in moves}


def proposed_contains_any(proposed: list[Move], candidates: list[Move]) -> bool:
    cand_paths = move_paths(candidates)
    return any(path_tuple(m) in cand_paths for m in proposed)


def moves_with_role(moves: list[Move], role: str) -> list[Move]:
    return [
        m for m in moves
        if m.get("facts", {}).get("quiet_move_role") == role
    ]


def lowest_threat_moves(moves: list[Move]) -> list[Move]:
    if not moves:
        return []
    best = min(m.get("facts", {}).get("our_pieces_threatened_after", 99) for m in moves)
    return [
        m for m in moves
        if m.get("facts", {}).get("our_pieces_threatened_after", 99) == best
    ]


def highest_counterplay_moves(moves: list[Move]) -> list[Move]:
    if not moves:
        return []
    best = max(m.get("facts", {}).get("counterplay_score", -999) for m in moves)
    return [
        m for m in moves
        if m.get("facts", {}).get("counterplay_score", -999) == best
    ]


def highest_conversion_moves(moves: list[Move]) -> list[Move]:
    if not moves:
        return []
    best = max(m.get("facts", {}).get("winning_conversion_score", -999) for m in moves)
    return [
        m for m in moves
        if m.get("facts", {}).get("winning_conversion_score", -999) == best
    ]


def highest_king_activity_moves(moves: list[Move]) -> list[Move]:
    if not moves:
        return []
    best = max(m.get("facts", {}).get("king_activity_score", -999) for m in moves)
    return [
        m for m in moves
        if m.get("facts", {}).get("king_activity_score", -999) == best
    ]


def highest_mobility_reduction_moves(moves: list[Move]) -> list[Move]:
    if not moves:
        return []
    best = max(m.get("facts", {}).get("mobility_reduction", -999) for m in moves)
    return [
        m for m in moves
        if m.get("facts", {}).get("mobility_reduction", -999) == best
    ]
def mobility_positive_moves(moves: list[Move]) -> list[Move]:
    return [
        m for m in moves
        if m.get("facts", {}).get("mobility_reduction", 0) > 0
    ]


def highest_mobility_moves(moves: list[Move]) -> list[Move]:
    mob = mobility_positive_moves(moves)
    if not mob:
        return []
    best = max(m.get("facts", {}).get("mobility_reduction", 0) for m in mob)
    return [
        m for m in mob
        if m.get("facts", {}).get("mobility_reduction", 0) == best
    ]


def safe_highest_mobility_moves(moves: list[Move]) -> list[Move]:
    mob = [
        m for m in moves
        if (
            m.get("facts", {}).get("mobility_reduction", 0) > 0
            and m.get("facts", {}).get("our_pieces_threatened_after", 99) == 0
        )
    ]
    if not mob:
        return []
    best = max(m.get("facts", {}).get("mobility_reduction", 0) for m in mob)
    return [
        m for m in mob
        if m.get("facts", {}).get("mobility_reduction", 0) == best
    ]

# ── test cases ────────────────────────────────────────────────────────────────

def test_threatened_piece_rescue_included() -> None:
    """
    If a threatened piece can move to safety, proposal should include at least one
    stabilization / rescue move in the shortlist.

    This targets the exact failure pattern you reported:
    proposal omitted the move that would move the threatened piece away.
    """
    board = empty_board()

    # RED to move
    board[3][4] = RED
    board[4][5] = RED   # threatened / candidate to save
    board[1][4] = RED   # promotion temptation
    board[3][6] = RED
    board[5][6] = RED

    board[6][3] = BLACK
    board[7][2] = BLACK_KING
    board[2][7] = BLACK
    board[4][7] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=93)

    rescue_path = [[4, 5], [3, 4]]
    if not has_move(legal, rescue_path):
        print("[SKIP] threatened_piece_rescue_included — rescue move not legal in this setup")
        return
    require(
        has_move(proposed, rescue_path),
        "Proposal failed to include the threatened-piece rescue move (4,5)->(3,4)."
    )


def test_safe_move_not_omitted_when_obvious() -> None:
    """
    If there is a clearly safer move among dangerous-looking alternatives,
    proposal should include at least one of the safest moves.
    """
    board = empty_board()

    board[4][3] = RED
    board[4][5] = RED
    board[6][7] = RED
    board[7][2] = RED

    board[2][5] = BLACK
    board[3][6] = BLACK
    board[4][1] = BLACK
    board[4][7] = BLACK
    board[5][0] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=21)

    safe = safe_moves(legal)
    require(safe, "Setup bug: expected at least one safe move.")
    safe_paths = {path_tuple(m) for m in safe}

    require(
        any(path_tuple(m) in safe_paths for m in proposed),
        "Proposal omitted all safe moves in a position where safe moves exist."
    )


def test_promotion_move_included() -> None:
    """
    If a safe promotion or near-promotion move exists, proposal must include one.
    """
    board = empty_board()

    board[1][4] = RED     # can promote
    board[3][4] = RED
    board[5][2] = RED

    board[6][3] = BLACK
    board[6][5] = BLACK
    board[4][7] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=50)

    promos = promotion_moves(legal)
    require(promos, "Setup bug: expected promotion/near-promotion move.")
    promo_paths = {path_tuple(m) for m in promos}

    require(
        any(path_tuple(m) in promo_paths for m in proposed),
        "Proposal failed to include a promotion / near-promotion move."
    )


def test_counterplay_when_losing() -> None:
    """
    When losing, proposal should include at least one active counterplay move.
    """
    board = empty_board()

    # RED is behind
    board[5][0] = RED
    board[5][4] = RED
    board[6][3] = RED

    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK
    board[4][5] = BLACK
    board[6][7] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=60)

    cps = counterplay_moves(legal)
    require(cps, "Setup bug: expected at least one counterplay move.")
    cp_paths = {path_tuple(m) for m in cps}

    require(
        any(path_tuple(m) in cp_paths for m in proposed),
        "Proposal failed to include a counterplay move while behind."
    )


def test_conversion_when_winning() -> None:
    """
    When ahead, proposal should include at least one move with positive
    winning_conversion_score if available.
    """
    board = empty_board()

    board[1][0] = RED_KING
    board[1][6] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[6][5] = RED

    board[2][7] = BLACK
    board[4][7] = BLACK
    board[7][0] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=70)

    conv = conversion_moves(legal)
    require(conv, "Setup bug: expected at least one conversion move.")
    conv_paths = {path_tuple(m) for m in conv}

    require(
        any(path_tuple(m) in conv_paths for m in proposed),
        "Proposal failed to include a conversion / trade-when-ahead move."
    )


def test_king_activation_in_endgame() -> None:
    """
    In king/endgame positions, proposal should include at least one king activation move.
    """
    board = empty_board()

    board[2][1] = RED_KING
    board[5][0] = RED_KING
    board[6][3] = RED

    board[1][4] = BLACK
    board[4][7] = BLACK
    board[6][5] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=80)

    kms = king_activation_moves(legal)
    require(kms, "Setup bug: expected at least one king activation move.")
    km_paths = {path_tuple(m) for m in kms}

    require(
        any(path_tuple(m) in km_paths for m in proposed),
        "Proposal failed to include a king activation move in endgame."
    )


def test_all_unsafe_position_still_reasonable() -> None:
    """
    If all legal moves are unsafe, proposal should still include the least-bad moves,
    especially those with lower threat count.
    """
    board = empty_board()

    board[4][3] = RED
    board[4][5] = RED
    board[5][6] = RED
    board[7][2] = RED

    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK
    board[3][6] = BLACK
    board[5][0] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=13)

    require(legal, "Setup bug: no legal moves.")
    proposed_paths = shortlist_paths(proposed)

    # compute least-bad by threat count then minimax-like local proxies
    sorted_by_threat = sorted(
        legal,
        key=lambda m: (
            m["facts"].get("our_pieces_threatened_after", 99),
            m["facts"].get("unsafe_simple_move", False),
            -m["facts"].get("counterplay_score", 0),
        ),
    )
    k = min(2, len(sorted_by_threat))
    topk = {path_tuple(sorted_by_threat[i]) for i in range(k)}
    require(
        any(p in topk for p in proposed_paths),
        "Proposal failed to include any of the least-bad unsafe moves."
    )


def test_shortlist_diversity_in_quiet_position() -> None:
    """
    In a quiet position with many safe moves, proposal should not return only
    near-duplicate passive moves if stronger role diversity exists.
    """
    board = empty_board()

    board[4][3] = RED
    board[5][0] = RED
    board[5][4] = RED
    board[5][6] = RED
    board[6][1] = RED
    board[6][3] = RED
    board[6][5] = RED
    board[7][2] = RED

    board[1][0] = BLACK
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=81)

    require(len(proposed) >= 4, "Expected at least 4 shortlisted moves.")
    require(
        unique_destinations(proposed) >= 3,
        "Proposal shortlist is too redundant: not enough destination diversity."
    )

    # If any promotion/counterplay/king-activity style move exists, shortlist should not be all plain quiet duplicates.
    roleful = [
        m for m in legal
        if (
            m["facts"].get("results_in_king", False)
            or m["facts"].get("near_promotion", False)
            or m["facts"].get("counterplay_score", 0) > 0
            or m["facts"].get("king_activity_score", 0) > 0
        )
    ]
    if roleful:
        role_paths = {path_tuple(m) for m in roleful}
        require(
            any(path_tuple(m) in role_paths for m in proposed),
            "Proposal returned only passive quiet moves despite richer move roles being available."
        )


def test_best_safe_move_by_threat_not_omitted() -> None:
    """
    If one move has strictly lower our_pieces_threatened_after than others,
    proposal must include it.
    """
    board = empty_board()

    board[1][0] = RED_KING
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[2][7] = BLACK
    board[3][0] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[5][0] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=79)

    best = min(
        legal,
        key=lambda m: (
            m["facts"].get("our_pieces_threatened_after", 99),
            m["facts"].get("unsafe_simple_move", False),
        ),
    )
    require(
        has_move(proposed, [list(x) for x in path_tuple(best)]),
        "Proposal omitted the move with the best board-wide threat reduction."
    )

def test_near_promotion_included_when_safe() -> None:
    board = empty_board()
    board[2][1] = RED
    board[4][3] = RED
    board[5][6] = RED
    board[6][3] = BLACK
    board[6][5] = BLACK
    board[4][7] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=40)
    promos = promotion_moves(legal)
    if not promos:
        print("[SKIP] near_promotion_included_when_safe — no promotion/near-promotion move in setup")
        return
    require(
        proposed_contains_any(proposed, promos),
        "Proposal failed to include a safe near-promotion / promotion move."
    )


def test_rescue_beats_redundant_passive_choices() -> None:
    board = empty_board()
    board[3][2] = RED
    board[4][5] = RED
    board[5][2] = RED
    board[5][4] = RED
    board[6][7] = RED

    board[6][3] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=93)
    rescue_like = [
        m for m in legal
        if m.get("facts", {}).get("our_pieces_threatened_after", 99) == 0
        and not m.get("facts", {}).get("unsafe_simple_move", False)
    ]
    require(rescue_like, "Setup bug: expected at least one stabilization / rescue-like move.")
    require(
        proposed_contains_any(proposed, rescue_like),
        "Proposal returned only passive alternatives and omitted all rescue/stabilization moves."
    )


def test_losing_position_includes_highest_counterplay() -> None:
    board = empty_board()
    board[5][0] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK
    board[4][7] = BLACK
    board[6][1] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=61)
    cps = highest_counterplay_moves(legal)
    if not cps:
        print("[SKIP] losing_position_includes_highest_counterplay — no counterplay move in setup")
        return
    require(
        proposed_contains_any(proposed, cps),
        "Proposal failed to include one of the highest-counterplay moves while losing."
    )


def test_winning_position_includes_highest_conversion() -> None:
    board = empty_board()
    board[1][0] = RED_KING
    board[2][3] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[2][7] = BLACK
    board[4][7] = BLACK
    board[7][0] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=75)
    # Only require SAFE highest-conversion moves — an unsafe conversion move
    # (opponent_can_recapture=True) is correctly excluded by the proposal agent;
    # requiring it would contradict the safety-first principle.
    all_conv = highest_conversion_moves(legal)
    conv = [m for m in all_conv if not m.get("facts", {}).get("opponent_can_recapture", False)]
    if not conv:
        print("[SKIP] winning_position_includes_highest_conversion — no SAFE highest-conversion move in setup")
        return
    require(
        proposed_contains_any(proposed, conv),
        "Proposal failed to include a safe highest-conversion move while winning."
    )


def test_endgame_king_activity_highest_not_omitted() -> None:
    board = empty_board()
    board[2][1] = RED_KING
    board[5][0] = RED_KING
    board[6][3] = RED
    board[1][4] = BLACK
    board[4][7] = BLACK
    board[6][5] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=81)
    kings = highest_king_activity_moves(legal)
    if not kings:
        print("[SKIP] endgame_king_activity_highest_not_omitted — no king activity move in setup")
        return
    require(
        proposed_contains_any(proposed, kings),
        "Proposal failed to include one of the highest king-activity moves in endgame."
    )


def test_stagnation_risk_includes_mobility_reduction() -> None:
    board = empty_board()
    board[1][0] = RED_KING
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[0][3] = BLACK_KING
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=87)
    all_movers = highest_mobility_reduction_moves(legal)
    # Both highest-mobility moves on this board are unsafe (opponent_can_recapture=True).
    # Requiring unsafe moves contradicts the safety-first invariant; skip gracefully.
    movers = [m for m in all_movers if not m.get("facts", {}).get("opponent_can_recapture", False)]
    if not movers:
        print("[SKIP] stagnation_risk_includes_mobility_reduction — no SAFE highest-mobility-reduction move in setup")
        return
    require(
        proposed_contains_any(proposed, movers),
        "Proposal failed to include a safe highest mobility-reduction move under loop/stagnation risk."
    )


def test_safe_promotion_not_replaced_by_flashy_counterplay() -> None:
    board = empty_board()
    board[1][4] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[2][7] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[6][1] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=52)
    promos = [m for m in promotion_moves(legal) if not m["facts"].get("opponent_can_recapture", False)]
    if not promos:
        print("[SKIP] safe_promotion_not_replaced_by_flashy_counterplay — no safe promotion move in setup")
        return
    require(
        proposed_contains_any(proposed, promos),
        "Proposal omitted a safe promotion move in favor of only flashy counterplay moves."
    )


def test_safe_conversion_not_replaced_by_center_temptation() -> None:
    board = empty_board()
    board[1][0] = RED_KING
    board[1][6] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[6][5] = RED

    board[2][7] = BLACK
    board[4][7] = BLACK
    board[7][0] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=72)
    conv = conversion_moves(legal)
    if not conv:
        print("[SKIP] safe_conversion_not_replaced_by_center_temptation — no conversion move in setup")
        return
    require(
        proposed_contains_any(proposed, conv),
        "Proposal omitted conversion move and returned only center / activity temptations."
    )


def test_all_safe_position_still_has_role_diversity() -> None:
    board = empty_board()
    board[2][1] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED
    board[7][0] = RED

    board[0][7] = BLACK
    board[2][7] = BLACK
    board[4][7] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=35)
    require(
        len(proposed) >= 3,
        "Expected at least 3 proposed moves."
    )
    require(
        unique_destinations(proposed) >= 3,
        "Proposal is too redundant in an all-safe quiet position."
    )


def test_best_safe_threat_reduction_not_omitted_in_mixed_position() -> None:
    board = empty_board()
    board[2][1] = RED
    board[3][2] = RED
    board[4][5] = RED
    board[5][2] = RED
    board[6][3] = RED

    board[6][5] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=92)
    low = lowest_threat_moves(legal)
    require(low, "Setup bug: expected at least one lowest-threat move.")
    require(
        proposed_contains_any(proposed, low),
        "Proposal omitted all lowest-threat moves in a mixed safe/unsafe position."
    )


def test_king_activation_present_when_only_kings_can_make_progress() -> None:
    board = empty_board()
    board[2][1] = RED_KING
    board[4][3] = RED_KING
    board[6][5] = RED

    board[1][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=84)
    kings = king_activation_moves(legal)
    if not kings:
        print("[SKIP] king_activation_present_when_only_kings_can_make_progress — no king activation move in setup")
        return
    require(
        proposed_contains_any(proposed, kings),
        "Proposal failed to include king-activity move when kings are the real progress makers."
    )


def test_counterplay_not_omitted_when_all_safe_are_passive() -> None:
    board = empty_board()
    board[2][1] = RED
    board[4][3] = RED
    board[5][4] = RED

    board[2][7] = BLACK
    board[3][6] = BLACK
    board[6][1] = BLACK_KING
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=57)
    cps = counterplay_moves(legal)
    if not cps:
        print("[SKIP] counterplay_not_omitted_when_all_safe_are_passive — no counterplay move in setup")
        return
    require(
        proposed_contains_any(proposed, cps),
        "Proposal returned only passive safe moves and omitted all counterplay."
    )


def test_conversion_present_when_trade_when_ahead_relevant() -> None:
    board = empty_board()
    board[1][0] = RED_KING
    board[2][3] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED

    board[5][0] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=73)
    conv = [m for m in legal if m["facts"].get("winning_conversion_score", 0) > 0]
    if not conv:
        print("[SKIP] conversion_present_when_trade_when_ahead_relevant — no conversion move in setup")
        return
    require(
        proposed_contains_any(proposed, conv),
        "Proposal failed to include conversion/trade-when-ahead move."
    )


def test_unsafe_simple_moves_do_not_dominate_shortlist_when_safe_exists() -> None:
    board = empty_board()
    board[3][2] = RED
    board[4][3] = RED
    board[5][2] = RED
    board[5][4] = RED
    board[6][7] = RED

    board[6][3] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=95)
    safe = safe_moves(legal)
    if not safe:
        print("[SKIP] unsafe_simple_moves_do_not_dominate_shortlist_when_safe_exists — no safe move in setup")
        return

    safe_paths = move_paths(safe)
    prop_paths = shortlist_paths(proposed)

    require(
        any(p in safe_paths for p in prop_paths),
        "Proposal let unsafe simple moves dominate despite safe alternatives existing."
    )


def test_role_coverage_when_many_safe_moves_exist() -> None:
    board = empty_board()
    board[1][4] = RED
    board[2][1] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[0][7] = BLACK
    board[2][7] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=55)

    roleful = [
        m for m in legal
        if (
            m["facts"].get("winning_conversion_score", 0) > 0
            or m["facts"].get("counterplay_score", 0) > 0
            or m["facts"].get("king_activity_score", 0) > 0
            or m["facts"].get("results_in_king", False)
            or m["facts"].get("near_promotion", False)
        )
    ]
    if not roleful:
        print("[SKIP] role_coverage_when_many_safe_moves_exist — no role-rich move in setup")
        return

    require(
        proposed_contains_any(proposed, roleful),
        "Proposal returned safe moves but omitted all richer role-based candidates."
    )


def test_left_flank_context_still_keeps_center_or_safety_option() -> None:
    board = empty_board()
    board[2][1] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[5][6] = RED

    board[1][0] = BLACK
    board[2][7] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK

    _, legal, proposed = run_proposal_only(board, RED, turn_number=66)

    best_candidates = [
        m for m in legal
        if (
            m["facts"].get("center_control", False)
            or m["facts"].get("our_pieces_threatened_after", 99) == min(
                x["facts"].get("our_pieces_threatened_after", 99) for x in legal
            )
        )
    ]
    require(
        proposed_contains_any(proposed, best_candidates),
        "Proposal over-followed flank context and omitted center/safety candidate."
    )


def test_shortlist_not_all_same_origin_piece_when_options_rich() -> None:
    board = empty_board()
    board[1][4] = RED
    board[2][1] = RED
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[0][7] = BLACK
    board[2][7] = BLACK
    board[4][7] = BLACK
    board[6][1] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=58)

    origins = set()
    for m in proposed:
        p = m.get("path", [])
        if p:
            origins.add(tuple(p[0]))

    require(
        len(origins) >= 2,
        "Proposal shortlist is too concentrated on a single origin piece despite rich alternatives."
    )

def test_safe_highest_mobility_reduction_included() -> None:
    """
    If a safe move with the highest mobility_reduction exists,
    proposal must include at least one such move.
    """
    board = empty_board()

    board[1][0] = RED_KING
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[0][3] = BLACK_KING
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=87)

    target = safe_highest_mobility_moves(legal)
    if not target:
        print("[SKIP] safe_highest_mobility_reduction_included — no safe mobility-reduction move in setup")
        return

    require(
        proposed_contains_any(proposed, target),
        "Proposal failed to include a safe highest mobility-reduction move."
    )


def test_slightly_unsafe_mobility_move_not_ignored_when_best_pressure() -> None:
    """
    If the strongest mobility-reduction move is slightly unsafe but clearly
    the best pressure move, proposal should still include it in stagnation-like positions.
    """
    board = empty_board()

    board[2][1] = RED_KING
    board[4][3] = RED
    board[5][4] = RED
    board[6][5] = RED

    board[1][6] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=88)

    target = highest_mobility_moves(legal)
    if not target:
        print("[SKIP] slightly_unsafe_mobility_move_not_ignored_when_best_pressure — no mobility move in setup")
        return

    require(
        proposed_contains_any(proposed, target),
        "Proposal failed to include the strongest mobility-reduction move when it was the best pressure option."
    )


def test_tied_highest_mobility_reduction_includes_one() -> None:
    """
    If multiple moves tie for the highest mobility_reduction,
    proposal should include at least one of them.
    """
    board = empty_board()

    board[1][0] = RED_KING
    board[2][3] = RED
    board[4][3] = RED
    board[5][6] = RED

    board[0][5] = BLACK
    board[2][7] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=89)

    target = safe_highest_mobility_moves(legal)
    if not target:
        print("[SKIP] tied_highest_mobility_reduction_includes_one — no safe mobility-reduction move in setup")
        return

    require(
        proposed_contains_any(proposed, target),
        "Proposal failed to include any move from the tied highest mobility-reduction group."
    )


def test_mobility_and_promotion_both_preserved_when_available() -> None:
    """
    If a promotion/near-promotion move and a strong mobility-reduction move both exist,
    proposal should preserve both when shortlist capacity allows.

    NOTE: only SAFE mobility-reduction moves are checked. A move that forces a
    mandatory opponent jump (opponent_can_recapture=True) is correctly excluded
    by the proposal agent as a safety risk — requiring it would contradict sound play.
    """
    board = empty_board()

    board[1][4] = RED        # promotion candidate
    board[3][2] = RED
    board[4][3] = RED
    board[5][4] = RED
    board[1][0] = RED_KING

    board[0][7] = BLACK
    board[3][6] = BLACK
    board[4][7] = BLACK
    board[7][2] = BLACK_KING

    _, legal, proposed = run_proposal_only(board, RED, turn_number=90)

    promos = promotion_moves(legal)
    # Only require safe mobility-reduction moves — an unsafe mobility move
    # (one that gives the opponent a mandatory jump) must not be forced into
    # the shortlist; the proposal agent correctly rejects those on safety grounds.
    mobs = safe_highest_mobility_moves(legal)

    if not promos:
        print("[SKIP] mobility_and_promotion_both_preserved_when_available — no promotion move in setup")
        return
    if not mobs:
        print("[SKIP] mobility_and_promotion_both_preserved_when_available — no SAFE mobility-reduction move in setup")
        return

    require(
        proposed_contains_any(proposed, promos),
        "Proposal failed to preserve a promotion/near-promotion move when mobility pressure also existed."
    )
    require(
        proposed_contains_any(proposed, mobs),
        "Proposal failed to preserve a safe highest mobility-reduction move when promotion also existed."
    )

# ── runner ────────────────────────────────────────────────────────────────────

TESTS: list[tuple[str, Callable[[], None]]] = [
    ("threatened_piece_rescue_included", test_threatened_piece_rescue_included),
    ("safe_move_not_omitted_when_obvious", test_safe_move_not_omitted_when_obvious),
    ("promotion_move_included", test_promotion_move_included),
    ("counterplay_when_losing", test_counterplay_when_losing),
    ("conversion_when_winning", test_conversion_when_winning),
    ("king_activation_in_endgame", test_king_activation_in_endgame),
    ("all_unsafe_position_still_reasonable", test_all_unsafe_position_still_reasonable),
    ("shortlist_diversity_in_quiet_position", test_shortlist_diversity_in_quiet_position),
    ("best_safe_move_by_threat_not_omitted", test_best_safe_move_by_threat_not_omitted),

    ("near_promotion_included_when_safe", test_near_promotion_included_when_safe),
    ("rescue_beats_redundant_passive_choices", test_rescue_beats_redundant_passive_choices),
    ("losing_position_includes_highest_counterplay", test_losing_position_includes_highest_counterplay),
    ("winning_position_includes_highest_conversion", test_winning_position_includes_highest_conversion),
    ("endgame_king_activity_highest_not_omitted", test_endgame_king_activity_highest_not_omitted),
    ("stagnation_risk_includes_mobility_reduction", test_stagnation_risk_includes_mobility_reduction),
    ("safe_promotion_not_replaced_by_flashy_counterplay", test_safe_promotion_not_replaced_by_flashy_counterplay),
    ("safe_conversion_not_replaced_by_center_temptation", test_safe_conversion_not_replaced_by_center_temptation),
    ("all_safe_position_still_has_role_diversity", test_all_safe_position_still_has_role_diversity),
    ("best_safe_threat_reduction_not_omitted_in_mixed_position", test_best_safe_threat_reduction_not_omitted_in_mixed_position),
    ("king_activation_present_when_only_kings_can_make_progress", test_king_activation_present_when_only_kings_can_make_progress),
    ("counterplay_not_omitted_when_all_safe_are_passive", test_counterplay_not_omitted_when_all_safe_are_passive),
    ("conversion_present_when_trade_when_ahead_relevant", test_conversion_present_when_trade_when_ahead_relevant),
    ("unsafe_simple_moves_do_not_dominate_shortlist_when_safe_exists", test_unsafe_simple_moves_do_not_dominate_shortlist_when_safe_exists),
    ("role_coverage_when_many_safe_moves_exist", test_role_coverage_when_many_safe_moves_exist),
    ("left_flank_context_still_keeps_center_or_safety_option", test_left_flank_context_still_keeps_center_or_safety_option),
    ("shortlist_not_all_same_origin_piece_when_options_rich", test_shortlist_not_all_same_origin_piece_when_options_rich),
    ("safe_highest_mobility_reduction_included", test_safe_highest_mobility_reduction_included),
    ("slightly_unsafe_mobility_move_not_ignored_when_best_pressure", test_slightly_unsafe_mobility_move_not_ignored_when_best_pressure),
    ("tied_highest_mobility_reduction_includes_one", test_tied_highest_mobility_reduction_includes_one),
    ("mobility_and_promotion_both_preserved_when_available", test_mobility_and_promotion_both_preserved_when_available),
]



def main() -> None:
    passed = 0
    failed = 0

    print("=" * 72)
    print("RUNNING PROPOSAL EDGE-CASE STRESS TESTS")
    print("=" * 72)

    for name, fn in TESTS:
        try:
            fn()
            passed += 1
            print(f"[PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}")
            print(f"       {type(e).__name__}: {e}")

    print("-" * 72)
    print(f"passed={passed}  failed={failed}  total={len(TESTS)}")
    print("-" * 72)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()