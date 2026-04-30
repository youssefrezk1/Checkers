# checkers/tests/test_format_checker_rank.py
#
# Tests for MINIMAX_RANK validation in format_checker._finalize_cleaned_moves.
#
# Validation rule:
#   When engine_legal_n >= 5, every path listed in
#   state.proposal_diagnostics["required_minimax_rank_paths"] must appear
#   in the cleaned proposal.  If any is missing, format_checker returns a
#   retry with MISSING_REQUIRED_MINIMAX_RANKS feedback.
#
# Design invariants verified here:
#   - Rejection never injects the missing move.
#   - Feedback contains no raw indices or coordinate paths.
#   - N < 5 completely bypasses rank validation.
#   - All existing proposal_agent tests still pass (separate file).
#
# Run with: venv/bin/pytest checkers/tests/test_format_checker_rank.py -v

from __future__ import annotations

import json

from checkers.engine.board import RED
from checkers.engine.rules import get_all_legal_moves
from checkers.nodes.format_checker import format_checker
from checkers.state.state import CheckersState


# ── Board helpers ─────────────────────────────────────────────────────────────

def _standard_start() -> list[list[int]]:
    board = [[0] * 8 for _ in range(8)]
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = 2  # BLACK
    return board


def _four_move_board() -> list[list[int]]:
    """Board with exactly 4 RED legal moves (two pieces, two moves each)."""
    board = [[0] * 8 for _ in range(8)]
    board[6][1] = RED   # moves to (5,0) and (5,2)
    board[6][5] = RED   # moves to (5,4) and (5,6)
    return board


def _make_scored_moves(legal: list[dict]) -> list[dict]:
    """Wrap legal moves in symbolic_scored_moves format with descending scores."""
    return [
        {"move": m, "minimax_score": float(len(legal) - i), "rank": i}
        for i, m in enumerate(legal)
    ]


def _make_state(
    board: list[list[int]],
    legal: list[dict],
    selected_indices: list[int],
    rank1_idx: int | None = None,
    rank2_idx: int | None = None,
    rank3_idx: int | None = None,
) -> CheckersState:
    """
    Build a CheckersState where:
      - symbolic_scored_moves = all legal moves (ranked by score descending)
      - proposed_moves = JSON string for selected_indices (into scored basis)
      - proposal_diagnostics.required_minimax_rank_paths = paths for given ranks
    """
    req: dict[str, list] = {}
    if rank1_idx is not None:
        req["1"] = [list(sq) for sq in legal[rank1_idx]["path"]]
    if rank2_idx is not None:
        req["2"] = [list(sq) for sq in legal[rank2_idx]["path"]]
    if rank3_idx is not None:
        req["3"] = [list(sq) for sq in legal[rank3_idx]["path"]]

    return CheckersState(
        board=board,
        current_player=RED,
        symbolic_scored_moves=_make_scored_moves(legal),
        proposed_moves=json.dumps({"selected_indices": selected_indices}),
        proposal_diagnostics={
            "required_minimax_rank_paths": req if req else None,
            "n_moves": len(legal),
        },
    )


# ── Test 1: N >= 5 and all required rank paths present → passes ───────────────

def test_all_ranks_present_passes():
    """
    When all MINIMAX_RANK_1/2/3 paths are in the proposal, format_checker
    must return a non-empty proposed_moves list and no feedback.
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 5, f"need >=5 moves; got {len(legal)}"

    # RANK_1 at index 0, RANK_2 at 1, RANK_3 at 2; propose 0-4 (all three included)
    state = _make_state(board, legal, [0, 1, 2, 3, 4], rank1_idx=0, rank2_idx=1, rank3_idx=2)
    result = format_checker(state)

    assert len(result["proposed_moves"]) == 5, (
        f"All ranks present — must pass; got proposed_moves={result['proposed_moves']}"
    )
    assert result.get("feedback") is None, (
        f"No feedback expected on pass; got: {result.get('feedback')}"
    )


# ── Test 2: N >= 5 and RANK_1 missing → rejects and retries ──────────────────

def test_rank1_missing_rejects():
    """
    Proposing indices [1,2,3,4,5] when RANK_1 is at index 0 must trigger
    a MISSING_REQUIRED_MINIMAX_RANKS rejection with retry_count incremented.
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 6

    # RANK_1=index 0 is excluded from the proposal [1,2,3,4,5]
    state = _make_state(board, legal, [1, 2, 3, 4, 5], rank1_idx=0, rank2_idx=1, rank3_idx=2)
    result = format_checker(state)

    assert result["proposed_moves"] == [], (
        "Missing RANK_1 must return empty proposed_moves"
    )
    assert result["retry_count"] == state.retry_count + 1, (
        "retry_count must be incremented on rank rejection"
    )
    assert "MISSING_REQUIRED_MINIMAX_RANKS" in result["feedback"], (
        f"Expected MISSING_REQUIRED_MINIMAX_RANKS in feedback; got: {result['feedback']}"
    )


# ── Test 3: N >= 5 and RANK_2 missing → rejects ───────────────────────────────

def test_rank2_missing_rejects():
    """
    Proposing [0,2,3,4,5] when RANK_1=0, RANK_2=1, RANK_3=2 must reject
    (RANK_2 at index 1 is absent).
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 6

    state = _make_state(board, legal, [0, 2, 3, 4, 5], rank1_idx=0, rank2_idx=1, rank3_idx=2)
    result = format_checker(state)

    assert result["proposed_moves"] == [], "Missing RANK_2 must reject"
    assert "MISSING_REQUIRED_MINIMAX_RANKS" in result["feedback"]


def test_rank3_missing_rejects():
    """
    Proposing [0,1,3,4,5] when RANK_3=2 is absent must also reject.
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 6

    state = _make_state(board, legal, [0, 1, 3, 4, 5], rank1_idx=0, rank2_idx=1, rank3_idx=2)
    result = format_checker(state)

    assert result["proposed_moves"] == [], "Missing RANK_3 must reject"
    assert "MISSING_REQUIRED_MINIMAX_RANKS" in result["feedback"]


# ── Test 4: Feedback contains no raw indices or coordinate paths ──────────────

def test_feedback_contains_no_exact_indices_or_paths():
    """
    The rejection feedback must be the fixed human-readable message only.
    It must not leak the missing index number or any coordinate path.
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 6

    rank1_path = legal[0]["path"]
    state = _make_state(board, legal, [1, 2, 3, 4, 5], rank1_idx=0, rank2_idx=1, rank3_idx=2)
    result = format_checker(state)

    feedback = result["feedback"]
    assert feedback is not None

    # Must be exactly the specified feedback string
    expected = (
        "MISSING_REQUIRED_MINIMAX_RANKS:\n"
        "Your shortlist missed one or more required MINIMAX_RANK_1/2/3 moves.\n"
        "Keep any MINIMAX_RANK moves already selected.\n"
        "Add the missing MINIMAX_RANK moves before adding diversity choices.\n"
        "Return exactly 5 distinct valid indices."
    )
    assert feedback == expected, (
        f"Feedback must match exactly.\nExpected:\n{expected}\nGot:\n{feedback}"
    )

    # No raw coordinate path from any rank in the feedback
    for sq in rank1_path:
        assert str(sq) not in feedback, (
            f"Coordinate {sq} from RANK_1 path must not appear in feedback"
        )


# ── Test 5: N < 5 → rank validation skipped, normal count rule applies ────────

def test_n_less_than_5_skips_rank_validation():
    """
    When N < 5, rank validation is not applied even if required_minimax_rank_paths
    is set.  A phantom path (not in the actual legal list) is placed in
    required_minimax_rank_paths; since N=4 the guard skips, and the proposal
    passes on count alone.
    """
    board = _four_move_board()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) == 4, f"expected 4 legal moves; got {len(legal)}"

    # Phantom path that does NOT exist among the 4 legal moves
    phantom_path = [[0, 1], [1, 0]]

    state = CheckersState(
        board=board,
        current_player=RED,
        symbolic_scored_moves=_make_scored_moves(legal),
        proposed_moves=json.dumps({"selected_indices": [0, 1, 2, 3]}),
        proposal_diagnostics={
            "required_minimax_rank_paths": {"1": phantom_path},
        },
    )
    result = format_checker(state)

    # N=4 → rank guard fires only for N>=5, so phantom path is irrelevant
    assert len(result["proposed_moves"]) == 4, (
        "N<5 must skip rank validation; proposal should pass on count"
    )
    assert result.get("feedback") is None


# ── Test 6: Rejection returns empty list — no injection ───────────────────────

def test_rejection_returns_empty_no_injection():
    """
    On MISSING_REQUIRED_MINIMAX_RANKS rejection, proposed_moves must be []
    (the missing rank move must not be silently added to the output).
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 6

    rank1_path = legal[0]["path"]
    state = _make_state(board, legal, [1, 2, 3, 4, 5], rank1_idx=0, rank2_idx=1, rank3_idx=2)
    result = format_checker(state)

    assert result["proposed_moves"] == [], (
        "Rejected result must have proposed_moves == []"
    )
    # The missing RANK_1 move must not have been injected
    assert not any(
        m.get("path") == rank1_path
        for m in (result["proposed_moves"] or [])
    ), "RANK_1 path must not be injected into the rejection result"


# ── Test 7: proposal_diagnostics=None → rank validation silently skipped ──────

def test_no_diagnostics_skips_rank_validation():
    """
    When proposal_diagnostics is None (fallback / parse-error path), rank
    validation must be skipped entirely.  A valid 5-index proposal must pass.
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 5

    state = CheckersState(
        board=board,
        current_player=RED,
        symbolic_scored_moves=_make_scored_moves(legal),
        proposed_moves=json.dumps({"selected_indices": [0, 1, 2, 3, 4]}),
        proposal_diagnostics=None,
    )
    result = format_checker(state)

    assert len(result["proposed_moves"]) == 5, (
        "No diagnostics → rank validation skipped; proposal must pass on count"
    )
