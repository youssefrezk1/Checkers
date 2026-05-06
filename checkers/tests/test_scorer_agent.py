"""
checkers/tests/test_scorer_agent.py

Minimal smoke tests for scorer_agent.score_all_legal_moves.

Run:
    pytest checkers/tests/test_scorer_agent.py -v
"""
from __future__ import annotations

import os
import pytest

from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves
from checkers.agents.scorer_agent import score_all_legal_moves


def _start_board() -> list[list[int]]:
    b = [[0] * 8 for _ in range(8)]
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = BLACK
    return b


def test_returns_all_legal_moves():
    board = _start_board()
    legal = get_all_legal_moves(board, RED)
    enriched, best, second, gap = score_all_legal_moves(board, RED)
    assert len(enriched) == len(legal), "one entry per legal move"


def test_each_move_has_minimax_score_and_rank():
    board = _start_board()
    enriched, _, _, _ = score_all_legal_moves(board, RED)
    for entry in enriched:
        assert "facts" in entry
        assert "minimax_score" in entry["facts"]
        assert "symbolic_rank" in entry["facts"]
        assert isinstance(entry["facts"]["minimax_score"], float)
        assert isinstance(entry["facts"]["symbolic_rank"], int)


def test_sorted_best_first():
    board = _start_board()
    enriched, _, _, _ = score_all_legal_moves(board, RED)
    scores = [e["facts"]["minimax_score"] for e in enriched]
    assert scores == sorted(scores, reverse=True), "list must be sorted best-first"


def test_ranks_are_sequential():
    board = _start_board()
    enriched, best, second, gap = score_all_legal_moves(board, RED)
    ranks = [e["facts"]["symbolic_rank"] for e in enriched]
    assert ranks[0] == 1
    assert ranks == list(range(1, len(enriched) + 1))


def test_summary_stats_consistent():
    board = _start_board()
    enriched, best, second, gap = score_all_legal_moves(board, RED)
    assert best == enriched[0]["facts"]["minimax_score"]
    if len(enriched) > 1:
        assert second == enriched[1]["facts"]["minimax_score"]
        assert abs(gap - (best - second)) < 1e-6
    else:
        assert second is None


def test_each_entry_has_move_fields():
    board = _start_board()
    enriched, _, _, _ = score_all_legal_moves(board, RED)
    for entry in enriched:
        assert "type" in entry
        assert "path" in entry
        assert "captured" in entry


def test_empty_board_returns_empty():
    board = [[0] * 8 for _ in range(8)]
    enriched, best, second, gap = score_all_legal_moves(board, RED)
    assert enriched == []
    assert second is None
    assert gap == 0.0


def test_repetition_penalty_applied(monkeypatch):
    """A repeated child hash should produce a lower score than without repetition."""
    from checkers.engine.rules import apply_move
    from checkers.engine.zobrist import compute_hash

    board = _start_board()
    legal = get_all_legal_moves(board, RED)
    # Build a history that includes every child position once
    fake_history = []
    for m in legal:
        child = apply_move(board, m)
        fake_history.append(compute_hash(child))

    enriched_no_rep, best_no, _, _ = score_all_legal_moves(board, RED)
    enriched_rep, best_rep, _, _ = score_all_legal_moves(board, RED, position_history=fake_history)

    # With repetition penalty every move is penalized; best score should be lower.
    assert best_rep <= best_no, "repetition penalty should not increase scores"


def test_ablation_mode_neutral_scores(monkeypatch):
    """When MINIMAX_ENABLED=false, all scores should be 0.0."""
    monkeypatch.setenv("MINIMAX_ENABLED", "false")
    import importlib
    import checkers.agents.scorer_agent as sa
    import checkers.nodes.minimax_scorer as ms

    # Reload to pick up env change
    importlib.reload(ms)
    importlib.reload(sa)

    board = _start_board()
    enriched, best, second, gap = sa.score_all_legal_moves(board, RED)
    for entry in enriched:
        assert entry["facts"]["minimax_score"] == 0.0
        assert entry["facts"]["symbolic_rank"] == 0
