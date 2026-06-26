#!/usr/bin/env python3
"""
Regression test for the Turn 33 ranker blunder.

Original bug:
- Old ranker_agent (combined select + explain) could choose [(5, 0), (4, 1)]
  (minimax score −79.8) over [(4, 3), (3, 2)] (−79.8 → −58.3) because
  mobility_reduction heuristics overrode the minimax ranking.

After the proposer/explainer split:
- Move selection is now deterministic: proposer_agent picks the highest
  minimax score and sets chosen_move + chosen_move_score on state.
- explainer_agent only generates a grounded reasoning paragraph for the
  pre-selected move; it never re-selects, overrides, or mutates chosen_move.

What this test now verifies:
- explainer_agent passes chosen_move through unchanged (no mutation).
- It must NOT return [(5, 0), (4, 1)] when the proposer pre-selected
  [(4, 3), (3, 2)] — the pass-through guarantee makes this trivially true,
  but the assertion is kept to document the original regression intent.
- explainer_diagnostics is populated with reasoning_seeds and core keys.
"""

from __future__ import annotations

import checkers.agents.explainer_agent as ranker_module
from checkers.agents.explainer_agent import explainer_agent
from checkers.engine.board import EMPTY, RED, BLACK, BLACK_KING
from checkers.state.state import CheckersState


def empty_board():
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def test_turn33_ranker_should_not_choose_worst_minimax(monkeypatch):
    board = empty_board()

    # Position reconstructed from the trace at Turn 33.
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

    # proposer_agent selects the minimax-best move before explainer_agent runs.
    chosen  = legal_moves[0]   # score -58.3 — best of the three
    unchosen = legal_moves[1:]

    # Bypass LLM calls: no API key in CI / unit-test environment.
    # Same pattern used by test_explainer_agent_step6.py.
    monkeypatch.setattr(
        ranker_module, "_generate_seeded_reasoning",
        lambda *a, **kw: ("Moves toward center to create threats.", ["seed1"]),
    )
    monkeypatch.setattr(
        ranker_module, "_check_reasoning_truthfulness",
        lambda *a, **kw: [],
    )
    # Disable the comparative stage so generate_comparative_reasoning is never called.
    monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")

    state = CheckersState(
        board=board,
        current_player=RED,
        turn_number=33,
        legal_moves=legal_moves,
        chosen_move=chosen,
        chosen_move_score=-58.3,
        unchosen_moves=unchosen,
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

    patch = explainer_agent(state)

    # explainer_agent is pass-through: chosen_move must be returned unchanged.
    result_path = patch["chosen_move"]["path"]
    assert result_path == [(4, 3), (3, 2)], (
        f"explainer_agent mutated chosen_move: got {result_path}. "
        "The worst-minimax path [(5, 0), (4, 1)] must never be emitted."
    )
    assert result_path != [(5, 0), (4, 1)], (
        "Turn 33 regression: explainer_agent returned the worst minimax path."
    )

    # Reasoning is produced (may be empty string if LLM unavailable, but the
    # key must exist and hold a str).
    assert "last_move_reasoning" in patch
    assert isinstance(patch["last_move_reasoning"], str)

    # Core diagnostics fields must be present.
    assert patch.get("last_completed_node") == "explainer_agent"
    diag = patch.get("explainer_diagnostics") or {}
    assert isinstance(diag, dict), "explainer_diagnostics must be a dict"
    assert isinstance(diag.get("reasoning_seeds"), list), (
        "reasoning_seeds must be a list in explainer_diagnostics"
    )
    assert "final_choice_source" in diag, (
        "final_choice_source missing from explainer_diagnostics"
    )


if __name__ == "__main__":
    test_turn33_ranker_should_not_choose_worst_minimax()
    print("PASS: Turn 33 regression verified (pass-through + diagnostics).")
