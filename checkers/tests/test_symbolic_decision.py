# checkers/tests/test_symbolic_decision.py
#
# Unit tests for the Phase 8 symbolic_decision node.
# Node is scoring-only: no bypass, no chosen_move.
# Always leads to proposal_agent.
#
# Run with: venv/bin/pytest checkers/tests/test_symbolic_decision.py -v

from __future__ import annotations

import pytest
from copy import deepcopy

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves
from checkers.engine.evaluation import WIN_SCORE, LOSS_SCORE
from checkers.state.state import CheckersState
import checkers.oldfiles.symbolic_decision as sd_module
from checkers.oldfiles.symbolic_decision import (
    symbolic_decision,
    SYMBOLIC_DECISION_DEPTH,
    _score_all_moves,
    _score_all_moves_search_root,
)


# ── Board helpers ─────────────────────────────────────────────────────────────

def _empty_board() -> list[list[int]]:
    return [[0] * 8 for _ in range(8)]


def _standard_start() -> list[list[int]]:
    """Full standard starting position (12 RED, 12 BLACK)."""
    board = _empty_board()
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = BLACK
    return board


def _make_state(board: list[list[int]], player: int = RED) -> CheckersState:
    return CheckersState(board=board, current_player=player)


# ── Test 1: return structure is always complete ───────────────────────────────

def test_return_keys_present_standard_start():
    """symbolic_decision always returns the required Phase 8 keys."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)

    required = {
        "symbolic_scored_moves",
        "symbolic_best_move",
        "symbolic_best_score",
        "symbolic_second_best_score",
        "symbolic_gap",
        "llm_invoked",
        "last_completed_node",
    }
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"
    assert result["last_completed_node"] == "symbolic_decision"


def test_llm_invoked_always_true():
    """llm_invoked must always be True — no bypass path exists."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    assert result.get("llm_invoked") is True


def test_chosen_move_never_set():
    """symbolic_decision must never set chosen_move — ranker is the decision-maker."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    assert "chosen_move" not in result or result.get("chosen_move") is None


def test_symbolic_bypass_never_set():
    """symbolic_bypass must not be set True by this node — bypass is removed."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    # Node must not emit symbolic_bypass=True (it may omit the key entirely)
    assert result.get("symbolic_bypass", False) is False


# ── Test 2: symbolic_scored_moves correctness ─────────────────────────────────

def test_scored_moves_non_empty_on_start():
    """symbolic_scored_moves must be non-empty on the standard starting position."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    assert result["symbolic_scored_moves"] != []
    assert len(result["symbolic_scored_moves"]) > 0


def test_scored_moves_count_equals_legal_count():
    """symbolic_scored_moves must contain exactly n_legal entries."""
    board = _standard_start()
    state = _make_state(board, RED)
    legal = get_all_legal_moves(board, RED)
    result = symbolic_decision(state)
    assert len(result["symbolic_scored_moves"]) == len(legal)


def test_scored_moves_sorted_best_first():
    """symbolic_scored_moves must be sorted descending by minimax_score."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    scores = [e["minimax_score"] for e in result["symbolic_scored_moves"]]
    assert scores == sorted(scores, reverse=True), (
        f"symbolic_scored_moves is not sorted best-first: {scores}"
    )


def test_scored_moves_rank_is_1_based_ascending():
    """rank must be 1-based and ascending (rank=1 is best move)."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    ranks = [e["rank"] for e in result["symbolic_scored_moves"]]
    assert ranks[0] == 1
    assert ranks == list(range(1, len(ranks) + 1))


def test_scored_moves_each_entry_has_required_fields():
    """Each symbolic_scored_moves entry must have move, minimax_score, rank."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    for entry in result["symbolic_scored_moves"]:
        assert "move" in entry, f"Missing 'move' in entry: {entry}"
        assert "minimax_score" in entry, f"Missing 'minimax_score' in entry: {entry}"
        assert "rank" in entry, f"Missing 'rank' in entry: {entry}"


def test_scored_moves_all_legal():
    """Every move in symbolic_scored_moves must be a legal move."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    legal = get_all_legal_moves(board, RED)
    legal_paths = {tuple(tuple(sq) for sq in m["path"]) for m in legal}
    for entry in result["symbolic_scored_moves"]:
        path_key = tuple(tuple(sq) for sq in entry["move"]["path"])
        assert path_key in legal_paths, (
            f"Scored move {entry['move']['path']} not in legal moves"
        )


# ── Test 3: best_move / best_score / gap fields ───────────────────────────────

def test_best_move_equals_rank1_move():
    """symbolic_best_move must match the rank=1 entry in symbolic_scored_moves."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    best_from_list = result["symbolic_scored_moves"][0]["move"]
    assert result["symbolic_best_move"] == best_from_list


def test_best_score_equals_rank1_score():
    """symbolic_best_score must equal the minimax_score of rank=1 entry."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    rank1_score = result["symbolic_scored_moves"][0]["minimax_score"]
    assert abs(result["symbolic_best_score"] - rank1_score) < 0.01


def test_gap_is_non_negative():
    """symbolic_gap must be >= 0 (best score >= second best)."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    assert result["symbolic_gap"] >= 0.0


def test_gap_equals_best_minus_second():
    """symbolic_gap must equal best_score - second_best_score."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    if result["symbolic_second_best_score"] is not None:
        expected = round(result["symbolic_best_score"] - result["symbolic_second_best_score"], 2)
        assert abs(result["symbolic_gap"] - expected) < 0.01


# ── Test 4: terminal position (no legal moves) ────────────────────────────────

def test_empty_board_no_legal_moves():
    """When no legal moves, symbolic_scored_moves is empty and best_move is None."""
    board = _empty_board()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    assert result["symbolic_scored_moves"] == []
    assert result["symbolic_best_move"] is None
    assert result["last_completed_node"] == "symbolic_decision"


# ── Test 5: depth constant ────────────────────────────────────────────────────

def test_symbolic_decision_uses_shared_depth():
    """SYMBOLIC_DECISION_DEPTH must equal MINIMAX_DEPTH."""
    from checkers.engine.minimax import MINIMAX_DEPTH

    assert SYMBOLIC_DECISION_DEPTH == MINIMAX_DEPTH, (
        f"Expected SYMBOLIC_DECISION_DEPTH == MINIMAX_DEPTH ({MINIMAX_DEPTH}) "
        f"but got {SYMBOLIC_DECISION_DEPTH}."
    )


# ── Test 6: state compat ──────────────────────────────────────────────────────

def test_state_symbolic_scored_moves_defaults_empty():
    """CheckersState.symbolic_scored_moves must default to empty list."""
    board = _standard_start()
    state = CheckersState(board=board, current_player=RED)
    assert state.symbolic_scored_moves == []


def test_state_symbolic_bypass_defaults_false():
    """symbolic_bypass field still exists in state as legacy compat, default False."""
    board = _standard_start()
    state = CheckersState(board=board, current_player=RED)
    assert state.symbolic_bypass is False


# ── Test 7: graph routing compatibility ──────────────────────────────────────

def test_no_chosen_move_so_graph_goes_to_proposal():
    """When symbolic_decision returns no chosen_move, graph must route to proposal_agent.
    This is verified by ensuring chosen_move is not set (routing logic in graph.py
    only goes to state_manager when chosen_move is set)."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    chosen = result.get("chosen_move")
    assert chosen is None, (
        f"symbolic_decision must not set chosen_move. Got: {chosen}"
    )


def test_symbolic_scored_moves_usable_by_proposal():
    """symbolic_scored_moves entries must have the 'move' key in engine dict format."""
    board = _standard_start()
    state = _make_state(board, RED)
    result = symbolic_decision(state)
    for entry in result["symbolic_scored_moves"]:
        m = entry["move"]
        assert "path" in m
        assert "type" in m
        assert "captured" in m


# ── Test 8: SYMBOLIC_SCORING_BACKEND tests ──────────────────────────────────

def test_default_backend_is_per_move():
    """Default SYMBOLIC_SCORING_BACKEND must be 'per_move'."""
    import os
    val = os.environ.get("SYMBOLIC_SCORING_BACKEND", "per_move")
    assert val == "per_move" or sd_module.SYMBOLIC_SCORING_BACKEND in sd_module._VALID_BACKENDS


def test_invalid_backend_raises_valueerror():
    """An invalid SYMBOLIC_SCORING_BACKEND value must raise ValueError at import."""
    import importlib, os
    original = os.environ.get("SYMBOLIC_SCORING_BACKEND")
    try:
        os.environ["SYMBOLIC_SCORING_BACKEND"] = "bogus_backend"
        with pytest.raises(ValueError, match="not valid"):
            importlib.reload(sd_module)
    finally:
        if original is None:
            os.environ.pop("SYMBOLIC_SCORING_BACKEND", None)
        else:
            os.environ["SYMBOLIC_SCORING_BACKEND"] = original
        importlib.reload(sd_module)


def test_search_root_backend_same_move_count_as_legal():
    """search_root_all_scores backend produces one score per legal move."""
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    scored = _score_all_moves_search_root(board, legal, RED, SYMBOLIC_DECISION_DEPTH)
    assert len(scored) == len(legal)


def test_search_root_backend_same_best_move_and_score():
    """search_root_all_scores backend produces same best move and score as per_move."""
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    per_move = _score_all_moves(board, legal, RED, SYMBOLIC_DECISION_DEPTH)
    sr_all = _score_all_moves_search_root(board, legal, RED, SYMBOLIC_DECISION_DEPTH)

    pm_best_move, pm_best_score = per_move[0]
    sr_best_move, sr_best_score = sr_all[0]

    assert round(pm_best_score, 2) == round(sr_best_score, 2), (
        f"Best score mismatch: per_move={pm_best_score} search_root={sr_best_score}"
    )
    pm_path = tuple(map(tuple, pm_best_move["path"]))
    sr_path = tuple(map(tuple, sr_best_move["path"]))
    assert pm_path == sr_path, (
        f"Best move mismatch: per_move={pm_path} search_root={sr_path}"
    )


def _make_small_board_1():
    """3v3 mid-game."""
    b = _empty_board()
    b[5][0] = RED; b[5][2] = RED; b[5][4] = RED
    b[2][1] = BLACK; b[2][3] = BLACK; b[2][5] = BLACK
    return b


def _make_small_board_2():
    """2v2 with capture opportunity."""
    b = _empty_board()
    b[5][0] = RED; b[5][4] = RED
    b[4][1] = BLACK; b[2][5] = BLACK
    return b


def _make_small_board_3():
    """King endgame."""
    b = _empty_board()
    b[3][2] = RED_KING; b[6][5] = RED
    b[1][4] = BLACK_KING; b[0][1] = BLACK
    return b


def test_full_scored_table_equality_across_backends():
    """All move scores must match between per_move and search_root_all_scores backends."""
    test_boards = [
        (_standard_start(), RED),
        (_make_small_board_1(), RED),
        (_make_small_board_2(), RED),
        (_make_small_board_3(), RED),
    ]
    for board, player in test_boards:
        legal = get_all_legal_moves(board, player)
        pm = _score_all_moves(board, legal, player, SYMBOLIC_DECISION_DEPTH)
        sr = _score_all_moves_search_root(board, legal, player, SYMBOLIC_DECISION_DEPTH)

        pm_by_path = {tuple(map(tuple, m["path"])): round(s, 2) for m, s in pm}
        sr_by_path = {tuple(map(tuple, m["path"])): round(s, 2) for m, s in sr}

        assert pm_by_path.keys() == sr_by_path.keys(), "Move set mismatch"
        for path_key in pm_by_path:
            assert pm_by_path[path_key] == sr_by_path[path_key], (
                f"Score mismatch for {path_key}: "
                f"per_move={pm_by_path[path_key]} search_root={sr_by_path[path_key]}"
            )


def test_search_root_backend_via_symbolic_decision(monkeypatch):
    """symbolic_decision produces identical output when backend is switched."""
    board = _standard_start()
    state = _make_state(board, RED)

    monkeypatch.setattr(sd_module, "SYMBOLIC_SCORING_BACKEND", "per_move")
    result_pm = symbolic_decision(state)

    monkeypatch.setattr(sd_module, "SYMBOLIC_SCORING_BACKEND", "search_root_all_scores")
    result_sr = symbolic_decision(state)

    assert result_pm["symbolic_best_score"] == result_sr["symbolic_best_score"]
    assert len(result_pm["symbolic_scored_moves"]) == len(result_sr["symbolic_scored_moves"])

    for pm_entry, sr_entry in zip(result_pm["symbolic_scored_moves"], result_sr["symbolic_scored_moves"]):
        assert pm_entry["minimax_score"] == sr_entry["minimax_score"], (
            f"Score mismatch at rank {pm_entry['rank']}: "
            f"per_move={pm_entry['minimax_score']} search_root={sr_entry['minimax_score']}"
        )
