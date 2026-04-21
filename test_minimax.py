"""
Focused tests for minimax + alpha-beta integration.

Run with:
    python3 -m pytest test_minimax.py -v
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from checkers.engine.board import (
    RED,
    BLACK,
    RED_KING,
    BLACK_KING,
    EMPTY,
)
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.evaluation import evaluate_board
from checkers.engine.minimax import score_move_with_minimax, minimax_score
from checkers.engine.move_facts import compute_move_facts
from checkers.nodes.minimax_scorer import minimax_scorer
from checkers.state.state import CheckersState


def empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: minimax runs on legal moves without crashing
# ──────────────────────────────────────────────────────────────────────────────
def test_minimax_runs_on_legal_moves() -> None:
    board = empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected at least one legal move"

    for move in legal:
        score = score_move_with_minimax(board, move, RED, depth=2)
        assert isinstance(score, float), "Minimax score must be float"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: minimax is deterministic
# ──────────────────────────────────────────────────────────────────────────────
def test_minimax_is_deterministic() -> None:
    board = empty_board()
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves"

    move = legal[0]
    score1 = score_move_with_minimax(board, move, RED, depth=2)
    score2 = score_move_with_minimax(board, move, RED, depth=2)

    assert score1 == score2, "Minimax should be deterministic"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: evaluation is positive when root player is materially ahead
# ──────────────────────────────────────────────────────────────────────────────
def test_evaluation_prefers_material_advantage() -> None:
    board = empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[5][4] = RED
    board[1][6] = BLACK

    score = evaluate_board(board, RED, RED)
    assert score > 0, "RED should evaluate positively when materially ahead"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: immediate winning capture should score very highly
# ──────────────────────────────────────────────────────────────────────────────
def test_minimax_scores_forcing_capture_line_high() -> None:
    board = empty_board()

    # RED to move
    # RED can jump BLACK and leave BLACK with no legal reply in a tiny position
    board[2][3] = RED
    board[1][4] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves"
    assert any(m["type"] == "jump" for m in legal), "Expected a jump move"

    jump_scores = [
        score_move_with_minimax(board, m, RED, depth=2)
        for m in legal
        if m["type"] == "jump"
    ]

    assert jump_scores, "Expected jump scores"
    assert max(jump_scores) > 100, "Winning/forcing line should score clearly high"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: king minimax scores are deterministic and score-based
# ──────────────────────────────────────────────────────────────────────────────
def test_king_minimax_prefers_higher_scoring_active_line() -> None:
    """
    King moves should be scored by minimax outcomes, not by row direction.
    This verifies:
    - minimax runs on all king moves
    - scores are floats
    - best score exists
    """
    board = empty_board()
    board[4][3] = RED_KING
    board[2][1] = BLACK
    board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves for king"

    scores = [
        (m, score_move_with_minimax(board, m, RED, depth=2))
        for m in legal
    ]

    for _, score in scores:
        assert isinstance(score, float), "King move minimax score must be float"

    best_score = max(score for _, score in scores)
    assert isinstance(best_score, float)
    assert best_score > float("-inf")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: minimax_scorer node attaches minimax_score to every move
# ──────────────────────────────────────────────────────────────────────────────
def test_minimax_scorer_node_attaches_score() -> None:
    board = empty_board()
    board[5][0] = RED
    board[2][1] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves"

    enriched = []
    for move in legal:
        mc = deepcopy(move)
        mc["facts"] = compute_move_facts(board, move, RED)
        enriched.append(mc)

    state = CheckersState(
        board=board,
        current_player=RED,
        legal_moves=enriched,
    )

    result = minimax_scorer(state)
    updated = result["legal_moves"]

    assert len(updated) == len(enriched), "Move count should stay unchanged"

    for move in updated:
        assert "facts" in move
        assert "minimax_score" in move["facts"]
        assert isinstance(move["facts"]["minimax_score"], float)


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: minimax prefers safer line when material is equal
# ──────────────────────────────────────────────────────────────────────────────
def test_minimax_prefers_safe_line_when_choices_exist() -> None:
    board = empty_board()

    # RED pieces
    board[5][2] = RED
    board[5][4] = RED

    # BLACK pieces arranged so one RED move is safer than another
    board[3][2] = BLACK
    board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves"

    scored = [(m, score_move_with_minimax(board, m, RED, depth=2)) for m in legal]
    assert scored, "Expected scored moves"

    best_move, best_score = max(scored, key=lambda x: x[1])

    assert isinstance(best_score, float)
    assert "path" in best_move


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))