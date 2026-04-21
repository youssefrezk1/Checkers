#!/usr/bin/env python3
"""
Regression test for the Turn 33 ranker blunder.

Goal:
- Proposal already contains all 3 legal moves
- Minimax scores clearly prefer move 0 over move 2
- Ranker must NOT choose move 2 because of mobility_reduction heuristics

Expected:
- chosen move path should be [(4, 3), (3, 2)]  OR [(4, 3), (3, 4)]
- it must NOT choose [(5, 0), (4, 1)]
"""

from __future__ import annotations

from checkers.state.state import CheckersState
from checkers.agents.ranker_agent import ranker_agent
from checkers.engine.board import EMPTY, RED, BLACK, BLACK_KING


def empty_board():
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def test_turn33_ranker_should_not_choose_worst_minimax():
    board = empty_board()

    # Position reconstructed from your trace at Turn 33
    board[0][7] = BLACK
    board[1][0] = BLACK
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK
    board[2][7] = BLACK
    board[3][6] = BLACK
    board[7][2] = BLACK_KING
    board[7][4] = BLACK_KING

    board[4][3] = RED
    board[4][7] = RED
    board[5][0] = RED
    board[6][1] = RED
    board[7][0] = RED

    legal_moves = [
        {
            "type": "simple",
            "path": [(4, 3), (3, 2)],
            "captured": [],
            "facts": {
                "minimax_score": -58.3,
                "counterplay_score": 2,
                "king_activity_score": 0,
                "net_gain": 0,
                "opponent_can_recapture": True,
                "unsafe_simple_move": True,
                "our_pieces_threatened_after": 1,
                "mobility_reduction": 9,
                "creates_immediate_threat": True,
            },
        },
        {
            "type": "simple",
            "path": [(4, 3), (3, 4)],
            "captured": [],
            "facts": {
                "minimax_score": -59.3,
                "counterplay_score": 2,
                "king_activity_score": 0,
                "net_gain": 0,
                "opponent_can_recapture": True,
                "unsafe_simple_move": True,
                "our_pieces_threatened_after": 1,
                "mobility_reduction": 9,
                "creates_immediate_threat": True,
            },
        },
        {
            "type": "simple",
            "path": [(5, 0), (4, 1)],
            "captured": [],
            "facts": {
                "minimax_score": -79.8,
                "counterplay_score": -1,
                "king_activity_score": 0,
                "net_gain": 0,
                "opponent_can_recapture": True,
                "unsafe_simple_move": True,
                "our_pieces_threatened_after": 3,
                "mobility_reduction": 10,
                "creates_immediate_threat": False,
            },
        },
    ]

    state = CheckersState(
        board=board,
        current_player=RED,
        turn_number=33,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "MIDGAME",
            "winning_score": -25,
            "score_state": "CLEARLY_LOSING",
            "strategic_priorities": [
                "DEFEND",
                "SEEK_COUNTERPLAY",
                "COMPLICATE",
                "AVOID_TRADES",
                "INCREASE_MOBILITY",
                "CREATE_THREATS",
            ],
            "active_patterns": [
                "OPPONENT_LEFT_FLANK_PUSH",
                "OPPONENT_RIGHT_FLANK_PUSH",
                "MATERIAL_BLEEDING",
            ],
        },
    )

    patch = ranker_agent(state)
    chosen = patch.get("chosen_move")
    assert chosen is not None, "ranker_agent returned no chosen_move"

    chosen_path = chosen.get("path")
    assert chosen_path in ([(4, 3), (3, 2)], [(4, 3), (3, 4)]), (
        f"Ranker chose the wrong move: {chosen_path}. "
        "It should not choose [(5, 0), (4, 1)] in this Turn 33 regression."
    )

    assert chosen_path != [(5, 0), (4, 1)], (
        "Ranker repeated the Turn 33 blunder by choosing the worst minimax move."
    )


if __name__ == "__main__":
    test_turn33_ranker_should_not_choose_worst_minimax()
    print("PASS: Turn 33 regression fixed.")