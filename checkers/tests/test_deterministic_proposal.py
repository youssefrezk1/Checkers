"""
checkers/tests/test_deterministic_proposal.py

Tests for select_proposal_candidates in deterministic_proposal.py.

Run:
    pytest checkers/tests/test_deterministic_proposal.py -v
"""
from __future__ import annotations

import pytest

from checkers.engine.board import RED, BLACK
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.deterministic_proposal import select_proposal_candidates


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _start_board() -> list[list[int]]:
    """Standard opening board (12 RED + 12 BLACK pieces)."""
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


def _board_with_one_legal_move() -> list[list[int]]:
    """A board where RED has exactly one legal move (and BLACK can't capture it)."""
    b = [[0] * 8 for _ in range(8)]
    # Single RED piece at (5, 0) — can only go to (4, 1)
    b[5][0] = RED
    # BLACK piece far away at (0, 7)
    b[0][7] = BLACK
    return b


def _board_with_capture() -> list[list[int]]:
    """
    Board with a forced capture sequence for RED.

    RED at (4, 3) can jump BLACK at (3, 4) -> landing at (2, 5).
    RED also has a quiet move at (5, 2).
    """
    b = [[0] * 8 for _ in range(8)]
    b[4][3] = RED
    b[5][2] = RED
    b[3][4] = BLACK
    b[0][7] = BLACK   # lone BLACK so game is not over
    return b


def _scored(board, player=RED):
    """Helper: return enriched scored moves from scorer_agent."""
    enriched, _, _, _ = score_all_legal_moves(board, player)
    return enriched


# ── Count correctness ──────────────────────────────────────────────────────────

def test_returns_five_from_start():
    board = _start_board()
    scored = _scored(board)
    assert len(scored) >= 5, "opening position needs >= 5 legal moves for this test"
    result = select_proposal_candidates(scored)
    assert len(result) == 5


def test_returns_all_when_fewer_than_five():
    board = _board_with_one_legal_move()
    scored = _scored(board)
    assert len(scored) == 1
    result = select_proposal_candidates(scored)
    assert len(result) == 1


def test_returns_min_k_n():
    board = _start_board()
    scored = _scored(board)
    n = len(scored)
    for k in range(1, n + 2):
        result = select_proposal_candidates(scored, k=k)
        assert len(result) == min(k, n), f"expected {min(k, n)} for k={k}, n={n}"


def test_empty_scored_moves_returns_empty():
    result = select_proposal_candidates([])
    assert result == []


# ── Source integrity — all returned moves come from scored_moves ──────────────

def _path_key(m: dict) -> tuple:
    return tuple(tuple(sq) for sq in m["path"])


def test_all_results_from_scored_moves():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    scored_keys = {_path_key(m) for m in scored}
    for r in result:
        assert _path_key(r) in scored_keys, "result contains a move not in scored_moves"


def test_no_duplicates_in_result():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    keys = [_path_key(m) for m in result]
    assert len(keys) == len(set(keys)), "result contains duplicate moves"


def test_result_elements_are_same_objects():
    """Returned dicts must be the exact objects from scored_moves (no copies)."""
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    scored_by_path = {_path_key(m): m for m in scored}
    for r in result:
        assert r is scored_by_path[_path_key(r)], "result element is a copy, not the original"


# ── Minimax top-3 inclusion ───────────────────────────────────────────────────

def test_top3_minimax_included_when_enough_moves():
    """When n >= 5, the three highest-minimax-score moves must appear in result."""
    board = _start_board()
    scored = _scored(board)
    assert len(scored) >= 5, "need >= 5 moves to test mm-pin"

    top3_keys = {_path_key(scored[i]) for i in range(3)}
    result = select_proposal_candidates(scored)
    result_keys = {_path_key(r) for r in result}
    assert top3_keys <= result_keys, (
        f"top-3 minimax moves not all included.\n"
        f"top3_keys={top3_keys}\nresult_keys={result_keys}"
    )


def test_top3_are_highest_minimax_in_result():
    """Verify top-3 by minimax_score are in the result, regardless of symbolic sort order."""
    board = _start_board()
    scored = _scored(board)
    assert len(scored) >= 5

    sorted_by_mm = sorted(scored, key=lambda m: m["facts"]["minimax_score"], reverse=True)
    top3_keys = {_path_key(m) for m in sorted_by_mm[:3]}

    result = select_proposal_candidates(scored)
    result_keys = {_path_key(r) for r in result}
    assert top3_keys <= result_keys


# ── Capture inclusion ─────────────────────────────────────────────────────────

def test_capture_included_when_available():
    """A capture move must appear in the shortlist when legal captures exist."""
    board = _board_with_capture()
    scored = _scored(board)
    jump_in_scored = any(m["type"] == "jump" for m in scored)
    if not jump_in_scored:
        pytest.skip("no jump moves in this position")

    result = select_proposal_candidates(scored)
    assert any(m["type"] == "jump" for m in result), "capture move missing from shortlist"


# ── Determinism ───────────────────────────────────────────────────────────────

def test_deterministic_output():
    """Same inputs must always produce the same output."""
    board = _start_board()
    scored = _scored(board)
    result_a = select_proposal_candidates(scored)
    result_b = select_proposal_candidates(scored)
    assert [_path_key(m) for m in result_a] == [_path_key(m) for m in result_b]


def test_deterministic_with_strategic_context():
    board = _start_board()
    scored = _scored(board)
    ctx = {
        "score_state": "SLIGHTLY_WINNING",
        "game_phase": "MIDGAME",
        "strategic_priorities": ["CONVERT_ADVANTAGE"],
    }
    result_a = select_proposal_candidates(scored, strategic_context=ctx)
    result_b = select_proposal_candidates(scored, strategic_context=ctx)
    assert [_path_key(m) for m in result_a] == [_path_key(m) for m in result_b]


# ── Fewer-than-5 edge cases ───────────────────────────────────────────────────

def test_single_move_board():
    board = _board_with_one_legal_move()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    assert len(result) == 1
    assert _path_key(result[0]) in {_path_key(m) for m in scored}


def test_k_zero_returns_empty():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored, k=0)
    assert result == []


def test_k_one_returns_one_move():
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored, k=1)
    assert len(result) == 1
    assert _path_key(result[0]) in {_path_key(m) for m in scored}


def test_facts_preserved_in_result():
    """Returned dicts retain minimax_score and symbolic_rank from scorer_agent."""
    board = _start_board()
    scored = _scored(board)
    result = select_proposal_candidates(scored)
    for r in result:
        assert "facts" in r
        assert "minimax_score" in r["facts"]
        assert "symbolic_rank" in r["facts"]


# ── Turn 9 regression ─────────────────────────────────────────────────────────

def _fake_move(path_tag: int, minimax_score: float, threatened_after: int = 0) -> dict:
    """Create a synthetic scored move for unit testing."""
    return {
        "type": "simple",
        "path": [[path_tag, 0], [path_tag - 1, 1]],
        "captured": [],
        "facts": {
            "minimax_score": minimax_score,
            "symbolic_rank": int(100 - minimax_score),
            "captures_count": 0,
            "results_in_king": False,
            "near_promotion": False,
            "our_pieces_threatened_after": threatened_after,
            "unsafe_simple_move": threatened_after > 0,
            "opponent_can_recapture": False,
            "leaves_piece_isolated": False,
            "center_control": False,
            "mobility_reduction": 0,
            "winning_conversion_score": 0,
            "counterplay_score": 0,
            "quiet_move_role": "QUIET_DEFAULT",
        },
    }


def test_turn9_rank1_rank2_rank3_always_included():
    """
    Regression: Turn 9 produced [rank5, rank8, rank4, rank9, rank3] because the old
    _mm_pin used pop+insert in a loop — pinning rank 2 inserted it before rank 1,
    pushing rank 1 back outside the window and ejecting it from the shortlist.

    Reproduces the failure condition: symbolic sort puts ranks 1 and 2 LAST
    (high our_pieces_threatened_after makes them appear unsafe) while ranks 3-9
    appear safe and fill the first 5 slots of the symbolic window.

    The fix guarantees rank 1 is always in the output, and ranks 2 and 3 are
    included when target >= 2 / >= 3.
    """
    # 9 moves sorted best-first by minimax_score (as scorer_agent produces).
    # Ranks 1 and 2 have threatened_after > 0 so the symbolic sort pushes them
    # to the END of the list, replicating the Turn 9 condition.
    scored_moves = [
        _fake_move(9, minimax_score=99.0, threatened_after=3),  # rank 1 (best, symbolically unsafe)
        _fake_move(8, minimax_score=98.0, threatened_after=2),  # rank 2 (symbolically unsafe)
        _fake_move(7, minimax_score=97.0, threatened_after=0),  # rank 3
        _fake_move(6, minimax_score=96.0, threatened_after=0),  # rank 4
        _fake_move(5, minimax_score=95.0, threatened_after=0),  # rank 5
        _fake_move(4, minimax_score=94.0, threatened_after=0),  # rank 6
        _fake_move(3, minimax_score=93.0, threatened_after=0),  # rank 7
        _fake_move(2, minimax_score=92.0, threatened_after=0),  # rank 8
        _fake_move(1, minimax_score=91.0, threatened_after=0),  # rank 9
    ]

    result = select_proposal_candidates(scored_moves)
    assert len(result) == 5, f"expected 5 candidates, got {len(result)}"

    result_paths = {_path_key(m) for m in result}
    rank1_pk = _path_key(scored_moves[0])
    rank2_pk = _path_key(scored_moves[1])
    rank3_pk = _path_key(scored_moves[2])

    assert rank1_pk in result_paths, (
        "rank 1 (best minimax) must always be in output; "
        f"got symbolic_ranks: {[m['facts']['symbolic_rank'] for m in result]}"
    )
    assert rank2_pk in result_paths, (
        "rank 2 must be in output when target=5; "
        f"got symbolic_ranks: {[m['facts']['symbolic_rank'] for m in result]}"
    )
    assert rank3_pk in result_paths, (
        "rank 3 must be in output when target=5; "
        f"got symbolic_ranks: {[m['facts']['symbolic_rank'] for m in result]}"
    )


def test_top1_always_included_regardless_of_symbolic_sort():
    """Rank 1 must be in output even for k=1 or k=2."""
    scored_moves = [
        _fake_move(5, minimax_score=99.0, threatened_after=5),  # rank 1, very unsafe symbolically
        _fake_move(4, minimax_score=98.0, threatened_after=0),
        _fake_move(3, minimax_score=97.0, threatened_after=0),
        _fake_move(2, minimax_score=96.0, threatened_after=0),
        _fake_move(1, minimax_score=95.0, threatened_after=0),
    ]
    rank1_pk = _path_key(scored_moves[0])

    for target_k in (1, 2, 3, 4, 5):
        result = select_proposal_candidates(scored_moves, k=target_k)
        result_paths = {_path_key(m) for m in result}
        assert rank1_pk in result_paths, (
            f"rank 1 missing from output with k={target_k}: "
            f"got {[m['facts']['symbolic_rank'] for m in result]}"
        )


# ── FIX 1: diversity pool constraint and minimax-heavy mode ───────────────────

def _fake_move_scored(row: int, minimax_score: float, **fact_overrides) -> dict:
    """Synthetic move with explicit minimax_score; all other facts default to safe."""
    facts = {
        "minimax_score": minimax_score,
        "symbolic_rank": 0,
        "captures_count": 0,
        "results_in_king": False,
        "near_promotion": False,
        "our_pieces_threatened_after": 0,
        "unsafe_simple_move": False,
        "opponent_can_recapture": False,
        "leaves_piece_isolated": False,
        "center_control": False,
        "mobility_reduction": 0,
        "winning_conversion_score": 0,
        "counterplay_score": 0,
        "quiet_move_role": "QUIET_DEFAULT",
    }
    facts.update(fact_overrides)
    return {
        "type": "simple",
        "path": [[row, 0], [row - 1, 1]],
        "captured": [],
        "facts": facts,
    }


def test_diversity_fill_never_exceeds_rank6():
    """
    T33/T35-style regression: symbolic sort must not inject rank-7..10 moves
    into the shortlist when the position is not minimax-heavy.

    Setup: 10 moves. Ranks 1-3 have threatened_after=0 (safe, good minimax).
    Ranks 4-6 also safe. Ranks 7-10 also safe but low minimax score.
    The score gap between rank-1 and rank-2 is < _TOP_GAP_HEAVY (19 pts, threshold=20)
    so minimax-heavy mode does NOT fire.

    Expectation: all 5 slots come from ranks 1-6 only; no rank-7..10 move appears.
    """
    scored_moves = [
        _fake_move_scored(10, 50.0),   # rank 1
        _fake_move_scored(9,  31.0),   # rank 2  — gap=19.0, well below _TOP_GAP_HEAVY=200
        _fake_move_scored(8,  30.0),   # rank 3
        _fake_move_scored(7,  29.0),   # rank 4
        _fake_move_scored(6,  28.0),   # rank 5
        _fake_move_scored(5,  27.0),   # rank 6  ← pool boundary
        _fake_move_scored(4,  10.0),   # rank 7  — must NOT appear
        _fake_move_scored(3,   9.0),   # rank 8  — must NOT appear
        _fake_move_scored(2,   8.0),   # rank 9  — must NOT appear
        _fake_move_scored(1,   7.0),   # rank 10 — must NOT appear
    ]
    pool_paths = {_path_key(m) for m in scored_moves[:6]}
    excluded_paths = {_path_key(m) for m in scored_moves[6:]}

    result = select_proposal_candidates(
        scored_moves,
        strategic_context={"game_phase": "MIDGAME", "score_state": "EQUAL"},
    )
    assert len(result) == 5
    for m in result:
        pk = _path_key(m)
        assert pk in pool_paths, (
            f"rank-7+ move appeared in shortlist: path={m['path']}, "
            f"score={m['facts']['minimax_score']}"
        )
        assert pk not in excluded_paths


def test_minimax_heavy_fires_on_endgame():
    """ENDGAME game_phase triggers minimax-heavy mode — result is pure top-k by score."""
    scored_moves = [
        _fake_move_scored(10, 40.0),   # rank 1
        _fake_move_scored(9,  39.0),   # rank 2
        _fake_move_scored(8,  38.0),   # rank 3
        _fake_move_scored(7,  10.0),   # rank 4
        _fake_move_scored(6,   9.0),   # rank 5
        _fake_move_scored(5,   8.0),   # rank 6
        _fake_move_scored(4,   7.0),   # rank 7
    ]
    result = select_proposal_candidates(
        scored_moves,
        strategic_context={"game_phase": "ENDGAME", "score_state": "EQUAL"},
    )
    assert len(result) == 5
    # Result must be exactly the top-5 by minimax score (ranks 1-5)
    expected_paths = {_path_key(m) for m in scored_moves[:5]}
    result_paths   = {_path_key(m) for m in result}
    assert result_paths == expected_paths, (
        f"endgame minimax-heavy should yield top-5; got scores "
        f"{[m['facts']['minimax_score'] for m in result]}"
    )


def test_minimax_heavy_fires_on_saturation():
    """abs(best_score) >= 9_000.0 (near WIN_SCORE=10_000) triggers minimax-heavy mode."""
    scored_moves = [
        _fake_move_scored(10, 9500.0),  # rank 1 — near win sentinel
        _fake_move_scored(9,  9480.0),  # rank 2
        _fake_move_scored(8,  9460.0),  # rank 3
        _fake_move_scored(7,   100.0),  # rank 4
        _fake_move_scored(6,    90.0),  # rank 5
        _fake_move_scored(5,    80.0),  # rank 6
    ]
    result = select_proposal_candidates(
        scored_moves,
        strategic_context={"game_phase": "MIDGAME", "score_state": "CLEARLY_WINNING"},
    )
    assert len(result) == 5
    expected_paths = {_path_key(m) for m in scored_moves[:5]}
    result_paths   = {_path_key(m) for m in result}
    assert result_paths == expected_paths


def test_minimax_heavy_fires_on_large_top_gap():
    """top_gap >= 200.0 (>=2 full pieces) triggers minimax-heavy mode."""
    scored_moves = [
        _fake_move_scored(10, 300.0),  # rank 1
        _fake_move_scored(9,   90.0),  # rank 2  — gap=210 >= _TOP_GAP_HEAVY
        _fake_move_scored(8,   80.0),  # rank 3
        _fake_move_scored(7,   10.0),  # rank 4
        _fake_move_scored(6,    9.0),  # rank 5
        _fake_move_scored(5,    8.0),  # rank 6
    ]
    result = select_proposal_candidates(
        scored_moves,
        strategic_context={"game_phase": "MIDGAME", "score_state": "EQUAL"},
    )
    assert len(result) == 5
    expected_paths = {_path_key(m) for m in scored_moves[:5]}
    result_paths   = {_path_key(m) for m in result}
    assert result_paths == expected_paths, (
        f"large top-gap (>=200) should yield minimax top-5; got scores "
        f"{[m['facts']['minimax_score'] for m in result]}"
    )


def test_rank4_rank5_not_displaced_by_rank9_when_normal_gap():
    """
    Top-4/top-5 exclusion regression (T35 variant): symbolic sort must not eject
    rank-4 or rank-5 in favor of rank-9 when the position is NOT minimax-heavy.

    Gap between rank-1 and rank-2 = 1.0 (well below _TOP_GAP_HEAVY=20).
    Ranks 4 and 5 have slightly higher symbolic priority (center_control=True)
    than ranks 7-10, but all are within the pool.
    After fix: rank-7..10 paths are excluded from fill; rank-4 and rank-5 appear.
    """
    scored_moves = [
        _fake_move_scored(10, 20.0),   # rank 1
        _fake_move_scored(9,  19.0),   # rank 2
        _fake_move_scored(8,  18.0),   # rank 3
        _fake_move_scored(7,  17.0),   # rank 4
        _fake_move_scored(6,  16.0),   # rank 5
        _fake_move_scored(5,  15.0),   # rank 6  ← pool boundary
        _fake_move_scored(4,   1.0, center_control=True),  # rank 7 (boosted symb rank)
        _fake_move_scored(3,   0.5, center_control=True),  # rank 8
        _fake_move_scored(2,   0.4),   # rank 9
        _fake_move_scored(1,   0.3),   # rank 10
    ]
    rank4_pk = _path_key(scored_moves[3])
    rank5_pk = _path_key(scored_moves[4])
    excluded = {_path_key(m) for m in scored_moves[6:]}

    result = select_proposal_candidates(
        scored_moves,
        strategic_context={"game_phase": "MIDGAME", "score_state": "EQUAL"},
    )
    assert len(result) == 5
    result_paths = {_path_key(m) for m in result}

    assert rank4_pk in result_paths, "rank-4 should appear; rank-7+ must not displace it"
    assert rank5_pk in result_paths, "rank-5 should appear; rank-7+ must not displace it"
    for pk in excluded:
        assert pk not in result_paths, f"rank-7+ move should not appear: {pk}"
