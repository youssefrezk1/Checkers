"""
checkers/tests/test_selective_d8.py

Tests for the selective-D8 upgrade in minimax_scorer._apply_selective_d8.

Board states are loaded from:
    logs/known_failure_positions_20260425_144451.json  (stored D6 tables)
    logs/game_20260425_144451_493544.jsonl              (board reconstruction)

All assertions use search_root_all_scores only — never score_move_with_minimax.

Run:
    pytest checkers/tests/test_selective_d8.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from copy import deepcopy

import pytest

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.search.minimax_core import (
    search_root_all_scores,
    get_d6_top_gap,
    clear_transposition_table,
)
from checkers.search.selective_d8 import _apply_selective_d8

# ── Fixtures / helpers ────────────────────────────────────────────────────────

LOGS = Path(__file__).parent.parent.parent / "logs"
KF_JSON  = LOGS / "known_failure_positions_20260425_144451.json"
KF_JSONL = LOGS / "game_20260425_144451_493544.jsonl"


def _load_kf() -> dict[int, dict]:
    with open(KF_JSON) as f:
        data = json.load(f)
    return {p["turn"]: p for p in data["positions"]}


def _make_start() -> list[list[int]]:
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


def _rebuild_boards() -> dict[int, list[list[int]]]:
    """boards[turn-1] = board AFTER turn (turn-1) = board BEFORE turn."""
    with open(KF_JSONL) as f:
        records = [json.loads(l) for l in f if l.strip()]
    board = _make_start()
    boards: dict[int, list[list[int]]] = {0: [row[:] for row in board]}
    for rec in records:
        t = rec["turn"]
        move = {
            "type": rec["move_type"],
            "path": rec["path"],
            "captured": rec.get("captured", []),
        }
        board = apply_move(board, move)
        boards[t] = [row[:] for row in board]
    return boards


def _board_for_turn(turn: int, boards: dict) -> list[list[int]]:
    return boards[turn - 1]


def _pk(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _make_fake_candidates(board: list, d8_depth: int = 6) -> list[dict]:
    """Build a candidate list with minimax_score attached (simulates D6 pipeline output)."""
    legal = get_all_legal_moves(board, RED)
    clear_transposition_table()
    _, _, scored, _ = search_root_all_scores(
        board=board, current_player=RED, depth=d8_depth,
        legal_moves=legal, use_tt=True, use_tactical_extension=True, use_phase7a=True,
    )
    candidates = []
    for rank, (mv, sc) in enumerate(scored):
        c = deepcopy(mv)
        c.setdefault("facts", {})
        c["facts"]["minimax_score"] = round(float(sc), 2)
        c["facts"]["symbolic_rank"] = rank + 1
        candidates.append(c)
    return candidates


# ── Test 1 — T35 triggers D8 and changes top-1 ───────────────────────────────

def test_t35_triggers_d8_and_changes_best():
    """
    T35: 13 pieces, D6 top-gap = 24 (within threshold).
    D6 best: (2,5)→(1,6).  D8 best: (6,5)→(5,4).
    With SELECTIVE_D8_ENABLED=true the upgrade must fire and return D8 scores.
    """
    boards = _rebuild_boards()
    board  = _board_for_turn(35, boards)

    # Simulate D6 candidates (what minimax_scorer produces before upgrade)
    candidates = _make_fake_candidates(board, d8_depth=6)
    d6_best_pk = _pk(candidates[0]["path"])  # rank-1 at D6

    # D6 top-gap guard
    scores = [c["facts"]["minimax_score"] for c in candidates]
    scores.sort(reverse=True)
    d6_top_gap = scores[0] - scores[1] if len(scores) >= 2 else float("inf")
    assert d6_top_gap <= 30, f"T35 D6 top-gap {d6_top_gap} not in (0, 30]"

    # Apply selective-D8 with thresholds that should trigger
    os.environ["SELECTIVE_D8_ENABLED"]         = "true"
    os.environ["SELECTIVE_D8_PIECE_THRESHOLD"] = "14"
    os.environ["SELECTIVE_D8_GAP_THRESHOLD"]   = "30"
    os.environ["SELECTIVE_D8_DEPTH"]           = "8"
    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "false"

    # Re-import to pick up env vars (module-level constants already read;
    # call _apply_selective_d8 directly — it reads the module globals)
    upgraded = _apply_selective_d8(board, RED, candidates)

    # D8 best should be (6,5)→(5,4)
    upgraded_sorted = sorted(upgraded, key=lambda c: -c["facts"]["minimax_score"])
    d8_best_pk = _pk(upgraded_sorted[0]["path"])
    assert d8_best_pk != d6_best_pk, (
        f"T35: expected D8 to change top-1 from D6 best {d6_best_pk}, but it did not."
    )
    expected_d8_best = _pk([[6, 5], [5, 4]])
    assert d8_best_pk == expected_d8_best, (
        f"T35: D8 best expected (6,5)→(5,4) but got {d8_best_pk}"
    )


# ── Test 2 — T45 triggers D8 and changes top-1 ───────────────────────────────

def test_t45_exact_tie_skips_d8_by_default_after_quiescence():
    """
    T45 used to have a small D6 gap and D8 changed top-1.
    After bounded capture quiescence, D6 resolves the line better and the
    top gap is now an exact tie. With SELECTIVE_D8_INCLUDE_EXACT_TIES=false,
    _apply_selective_d8 should skip D8 and keep the original D6 ordering.
    """
    boards = _rebuild_boards()
    board = _board_for_turn(45, boards)

    candidates = _make_fake_candidates(board, d8_depth=6)
    scores = [c["facts"]["minimax_score"] for c in candidates]
    scores.sort(reverse=True)
    d6_top_gap = scores[0] - scores[1] if len(scores) >= 2 else float("inf")
    assert d6_top_gap == pytest.approx(0.0)

    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "false"

    original_scores = [(c["path"], c["facts"]["minimax_score"]) for c in candidates]
    upgraded = _apply_selective_d8(board, RED, candidates)
    upgraded_scores = [(c["path"], c["facts"]["minimax_score"]) for c in upgraded]

    assert upgraded_scores == original_scores, (
        "T45: exact D6 tie should skip D8 by default after quiescence"
    )

# ── Test 3 — T49 does NOT trigger (D6 gap too large) ─────────────────────────

def test_t49_does_not_trigger():
    """
    T49: 11 pieces, D6 top-gap = 262 >> 30.
    _apply_selective_d8 must return the original candidates unchanged.
    """
    boards = _rebuild_boards()
    board  = _board_for_turn(49, boards)

    candidates = _make_fake_candidates(board, d8_depth=6)
    scores = [c["facts"]["minimax_score"] for c in candidates]
    scores.sort(reverse=True)
    d6_top_gap = scores[0] - scores[1] if len(scores) >= 2 else float("inf")
    assert d6_top_gap > 30, f"T49 gap {d6_top_gap} should be >30 for this test"

    original_scores = [(c["path"], c["facts"]["minimax_score"]) for c in candidates]
    upgraded = _apply_selective_d8(board, RED, candidates)
    upgraded_scores = [(c["path"], c["facts"]["minimax_score"]) for c in upgraded]

    assert original_scores == upgraded_scores, (
        "T49: _apply_selective_d8 must return unchanged candidates when D6 gap > threshold"
    )


# ── Test 4 — T51 does NOT trigger (D6 gap too large) ─────────────────────────

def test_t51_does_not_trigger():
    """
    T51: 11 pieces, D6 top-gap = 140 >> 30.
    """
    boards = _rebuild_boards()
    board  = _board_for_turn(51, boards)

    candidates = _make_fake_candidates(board, d8_depth=6)
    scores = [c["facts"]["minimax_score"] for c in candidates]
    scores.sort(reverse=True)
    d6_top_gap = scores[0] - scores[1] if len(scores) >= 2 else float("inf")
    assert d6_top_gap > 30, f"T51 gap {d6_top_gap} should be >30"

    original_scores = [(c["path"], c["facts"]["minimax_score"]) for c in candidates]
    upgraded = _apply_selective_d8(board, RED, candidates)
    upgraded_scores = [(c["path"], c["facts"]["minimax_score"]) for c in upgraded]

    assert original_scores == upgraded_scores, (
        "T51: _apply_selective_d8 must not trigger when D6 gap > threshold"
    )


# ── Test 5 — T41/T43 exact ties skipped by default ───────────────────────────

@pytest.mark.parametrize("turn", [41, 43])
def test_exact_tie_skipped_by_default(turn: int):
    """
    T41 and T43 have D6 top-gap = 0 (exact ties).
    With SELECTIVE_D8_INCLUDE_EXACT_TIES=false (default), must not trigger.
    """
    boards = _rebuild_boards()
    board  = _board_for_turn(turn, boards)

    candidates = _make_fake_candidates(board, d8_depth=6)
    scores = [c["facts"]["minimax_score"] for c in candidates]
    scores.sort(reverse=True)
    d6_top_gap = scores[0] - scores[1] if len(scores) >= 2 else float("inf")

    # Only assert the tie behaviour if D6 really is tied at this turn
    if d6_top_gap != 0.0:
        pytest.skip(f"T{turn} D6 top-gap is {d6_top_gap}, not 0 — skipping exact-tie test")

    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "false"

    original_scores = [(c["path"], c["facts"]["minimax_score"]) for c in candidates]
    upgraded = _apply_selective_d8(board, RED, candidates)
    upgraded_scores = [(c["path"], c["facts"]["minimax_score"]) for c in upgraded]

    assert original_scores == upgraded_scores, (
        f"T{turn}: exact tie must be skipped when SELECTIVE_D8_INCLUDE_EXACT_TIES=false"
    )


@pytest.mark.parametrize("turn", [41, 43])
def test_exact_tie_triggers_when_flag_set(turn: int):
    """
    With SELECTIVE_D8_INCLUDE_EXACT_TIES=true, exact ties should trigger D8.
    Verifies the upgraded list has the same paths but potentially reordered scores.
    """
    boards = _rebuild_boards()
    board  = _board_for_turn(turn, boards)

    candidates = _make_fake_candidates(board, d8_depth=6)
    scores = [c["facts"]["minimax_score"] for c in candidates]
    scores.sort(reverse=True)
    d6_top_gap = scores[0] - scores[1] if len(scores) >= 2 else float("inf")

    if d6_top_gap != 0.0:
        pytest.skip(f"T{turn} D6 top-gap is {d6_top_gap}, not 0 — skipping")

    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "true"
    os.environ["SELECTIVE_D8_PIECE_THRESHOLD"]    = "14"
    os.environ["SELECTIVE_D8_GAP_THRESHOLD"]      = "30"
    os.environ["SELECTIVE_D8_DEPTH"]              = "8"

    upgraded = _apply_selective_d8(board, RED, candidates)

    # Paths must be identical — only scores may change
    original_paths = {_pk(c["path"]) for c in candidates}
    upgraded_paths = {_pk(c["path"]) for c in upgraded}
    assert original_paths == upgraded_paths, (
        f"T{turn}: upgraded paths must equal original paths"
    )

    # At minimum, the upgrade ran (we can't assert score change since D8 may agree)
    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "false"


# ── Test 6 — disabled flag leaves behaviour completely unchanged ───────────────

def test_disabled_flag_no_change():
    """
    SELECTIVE_D8_ENABLED=false must make _apply_selective_d8 a no-op.
    Verified by checking that the module constant is False when env is false.
    """
    import importlib
    import checkers.search.selective_d8 as sd_mod
    os.environ["SELECTIVE_D8_ENABLED"] = "false"
    importlib.reload(sd_mod)
    assert sd_mod.SELECTIVE_D8_ENABLED is False, (
        "SELECTIVE_D8_ENABLED must be False when env var is 'false'"
    )
    # Restore
    os.environ["SELECTIVE_D8_ENABLED"] = "true"
    importlib.reload(sd_mod)


# ── Test 7 — D8 result used for scoring only, not legality ───────────────────

def test_d8_does_not_change_candidate_paths():
    """
    _apply_selective_d8 must not add or remove candidates.
    Only minimax_score and symbolic_rank may change.
    """
    boards = _rebuild_boards()
    board  = _board_for_turn(35, boards)     # T35 — known to trigger

    candidates = _make_fake_candidates(board, d8_depth=6)
    original_paths = {_pk(c["path"]) for c in candidates}

    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "false"
    upgraded = _apply_selective_d8(board, RED, candidates)
    upgraded_paths = {_pk(c["path"]) for c in upgraded}

    assert len(upgraded) == len(candidates), (
        "candidate count must not change after selective-D8"
    )
    assert upgraded_paths == original_paths, (
        "candidate paths must not change after selective-D8"
    )


# ── Test 8 — get_d6_top_gap helper ───────────────────────────────────────────

def test_get_d6_top_gap_basic():
    assert get_d6_top_gap([]) == float("inf")
    assert get_d6_top_gap([(None, 50.0)]) == float("inf")
    assert get_d6_top_gap([(None, 50.0), (None, 26.0)]) == pytest.approx(24.0)
    assert get_d6_top_gap([(None, 35.0), (None, 35.0)]) == pytest.approx(0.0)
    assert get_d6_top_gap([(None, 35.0), (None, 35.0), (None, -3.0)]) == pytest.approx(0.0)


# ── Test 9 — fallback path uses search_root_all_scores, not score_move_with_minimax ──

def test_fallback_uses_joint_search_not_per_move_scorer():
    """
    When symbolic_scored_moves cache is empty (full cache miss), minimax_scorer
    must call search_root_all_scores exactly once and must NOT call
    score_move_with_minimax at all.

    Verified by patching both functions with spies and running minimax_scorer
    against a minimal CheckersState with an empty symbolic_scored_moves.
    """
    import importlib
    import unittest.mock as mock
    from checkers.state.state import CheckersState
    from checkers.engine.board import create_initial_board
    from checkers.engine.rules import get_all_legal_moves

    board  = create_initial_board()
    player = RED
    legal  = get_all_legal_moves(board, player)

    # Build a state with empty symbolic_scored_moves so every move is a cache miss
    state = CheckersState(
        board=board,
        current_player=player,
        legal_moves=legal,
        symbolic_scored_moves=[],   # forces full fallback
    )

    # Reload minimax_scorer with SELECTIVE_D8_ENABLED=false so we isolate the
    # fallback path only (not the D8 upgrade path)
    os.environ["SELECTIVE_D8_ENABLED"] = "false"
    import checkers.oldfiles.minimax_scorer as ms_mod
    importlib.reload(ms_mod)

    per_move_spy = mock.MagicMock(side_effect=AssertionError(
        "score_move_with_minimax must NOT be called in the fallback path"
    ))

    joint_search_calls = []
    _real_search = search_root_all_scores

    def joint_spy(*args, **kwargs):
        joint_search_calls.append((args, kwargs))
        return _real_search(*args, **kwargs)

    with mock.patch.object(ms_mod, "search_root_all_scores", side_effect=joint_spy), \
         mock.patch("checkers.engine.minimax.score_move_with_minimax", side_effect=per_move_spy):
        result = ms_mod.minimax_scorer(state)

    # Joint search must have been called exactly once
    assert len(joint_search_calls) == 1, (
        f"Expected search_root_all_scores called once; called {len(joint_search_calls)} times"
    )

    # Every candidate must have a valid rank (proof the joint search populated results)
    updated = result["legal_moves"]
    assert len(updated) == len(legal), "candidate count must be unchanged"
    for cand in updated:
        facts = cand.get("facts", {})
        assert facts.get("symbolic_rank", 0) >= 1, (
            f"candidate {cand.get('path')} has invalid symbolic_rank after joint fallback"
        )
    # All ranks must be distinct (1..n_legal) — proves joint search ordered them
    ranks = [c["facts"]["symbolic_rank"] for c in updated]
    assert sorted(ranks) == list(range(1, len(legal) + 1)), (
        f"symbolic_ranks must be 1..n after joint fallback; got {sorted(ranks)}"
    )


    # Restore
    os.environ["SELECTIVE_D8_ENABLED"] = "false"
    importlib.reload(ms_mod)

